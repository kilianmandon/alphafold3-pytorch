import math
import torch
from torch import nn
from feature_extraction.ref_struct_features import RefStructFeatures
from sparse_utils import BlockSparseTensor
import tensortrace as ttr
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask

from common import AdaptiveLayerNorm, AdaptiveZeroInit, AttentionPairBias, ConditionedTransitionBlock
import utils

def hotfix_mangle_layout(ref_space_uid, ref_struct: RefStructFeatures):
    ref_space_uid = ref_struct.to_token_layout(ref_space_uid)
    ref_space_uid[..., :, :] = ref_space_uid[..., :, :1]
    ref_space_uid = torch.flatten(ref_space_uid, start_dim=-2)
    return ref_space_uid


class AtomAttentionEncoder(nn.Module):
    # Implements Algorithm 5 from the paper
    def __init__(self, c_s, c_z, c_atom=128, c_atom_pair=16, c_token=384, use_trunk=False):
        super().__init__()
        self.embed_ref_pos = nn.Linear(3, c_atom, bias=False)
        self.embed_ref_mask = nn.Linear(1, c_atom, bias=False)
        self.embed_ref_element = nn.Linear(c_atom, c_atom, bias=False)
        self.embed_ref_charge = nn.Linear(1, c_atom, bias=False)
        self.embed_ref_atom_name = nn.Linear(256, c_atom, bias=False)

        self.single_to_pair_row = nn.Linear(c_atom, c_atom_pair, bias=False)
        self.single_to_pair_col = nn.Linear(c_atom, c_atom_pair, bias=False)
        self.embed_pair_offsets = nn.Linear(3, c_atom_pair, bias=False)
        self.embed_pair_distances = nn.Linear(1, c_atom_pair, bias=False)
        self.embed_pair_mask = nn.Linear(1, c_atom_pair, bias=False)

        self.pair_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(c_atom_pair, c_atom_pair, bias=False),
            nn.ReLU(),
            nn.Linear(c_atom_pair, c_atom_pair, bias=False),
            nn.ReLU(),
            nn.Linear(c_atom_pair, c_atom_pair, bias=False)
        )

        self.atom_transformer = DiffusionTransformer(c_a=128, c_z=16, N_head=4, c_s=128, N_block=3, split_ada_qk=True)
        self.project_atom_features = nn.Linear(c_atom, c_token, bias=False)

        self.use_trunk = use_trunk
        if use_trunk:
            self.trunk_layer_norm_s = nn.LayerNorm(c_s, bias=False)
            self.trunk_linear_s = nn.Linear(c_s, c_atom, bias=False)
            self.trunk_layer_norm_z = nn.LayerNorm(c_z, bias=False)
            self.trunk_linear_z = nn.Linear(c_z, c_atom_pair, bias=False)
            self.trunk_linear_r = nn.Linear(3, c_atom, bias=False)


    def forward(self, ref_struct: RefStructFeatures, r=None, s_trunk=None, z=None):
        ref_space_uid = ref_struct.ref_space_uid
        ref_pos = ref_struct.positions
        block_mask = ref_struct.block_mask
        batch_shape = ref_space_uid.shape[:-2]

        single_cond = self.per_atom_cond(ref_struct)

        single_act = single_cond.clone()

        wrong_ref_space_uid = hotfix_mangle_layout(ref_space_uid, ref_struct)
        ref_space_left = BlockSparseTensor.from_broadcast(ref_space_uid[..., :, None], block_mask, batch_shape)
        ref_space_right = BlockSparseTensor.from_broadcast(wrong_ref_space_uid[..., None, :], block_mask, batch_shape)
        ref_pos_left = BlockSparseTensor.from_broadcast(ref_pos[..., :, None, :], block_mask, batch_shape)
        ref_pos_right = BlockSparseTensor.from_broadcast(ref_pos[..., None, :, :], block_mask, batch_shape)

        offsets_valid = (ref_space_left == ref_space_right).to(dtype=torch.float32)

        offsets = ref_pos_left - ref_pos_right


        pair_act = self.embed_pair_offsets(offsets) * offsets_valid

        sq_dists = torch.sum(offsets**2, dim=-1, keepdim=True)

        pair_act += self.embed_pair_distances(1/(1+sq_dists)) * offsets_valid

        if self.use_trunk:
            s_trunk = ref_struct.to_atom_layout(s_trunk, has_atom_dimension=False)

            batch_idx, p_idx, l_idx = BlockSparseTensor.block_mask_to_index(block_mask)
            token_indices = utils.unify_batch_dimension(ref_struct.token_index, batch_shape)
            z = utils.unify_batch_dimension(z, batch_shape)
            i_idx = token_indices[batch_idx, p_idx]
            j_idx = token_indices[batch_idx, l_idx]
            z = BlockSparseTensor(z[batch_idx, i_idx, j_idx], block_mask)

            single_cond += self.trunk_linear_s(self.trunk_layer_norm_s(s_trunk))
            pair_act += self.trunk_linear_z(self.trunk_layer_norm_z(z))

            # Note: The paper uses the old, non-trunk-updated value
            # for queries_single_cond here
            single_act = single_cond + self.trunk_linear_r(r)


        row_act = self.single_to_pair_row(torch.relu(single_cond))
        row_act = BlockSparseTensor.from_broadcast(row_act[..., :, None, :], block_mask, batch_shape)
        col_act = self.single_to_pair_col(torch.relu(single_cond))
        col_act = BlockSparseTensor.from_broadcast(col_act[..., None, :, :], block_mask, batch_shape)

        pair_act += row_act + col_act
        pair_act += self.embed_pair_mask(offsets_valid)

        pair_act += self.pair_mlp(pair_act)

        single_act = self.atom_transformer(
            single_act,
            single_cond,
            pair_act,
            block_mask
        )


        token_act = ref_struct.to_token_layout(single_act)
        token_act = torch.relu(self.project_atom_features(token_act))

        token_act = utils.masked_mean(token_act, ref_struct.token_layout_ref_mask[..., None], axis=-2)

        skip = (single_act, single_cond, pair_act)

        return token_act, skip

    def per_atom_cond(self, ref_struct: RefStructFeatures):


        mask = ref_struct.mask[..., None].to(torch.float32)
        element = ref_struct.element.long()
        charge = ref_struct.charge[..., None].to(torch.float32)
        name_chars = ref_struct.atom_name_chars.long()

        elements_1h = nn.functional.one_hot(element, 128).to(torch.float32)
        atom_names_1h = nn.functional.one_hot(name_chars, 64).to(torch.float32)
        atom_names_1h = atom_names_1h.reshape(atom_names_1h.shape[:-2] + (-1,))

        act = self.embed_ref_pos(ref_struct.positions)
        act += self.embed_ref_mask(mask)
        act += self.embed_ref_element(elements_1h)
        act += self.embed_ref_charge(torch.arcsinh(charge))

        act += self.embed_ref_atom_name(atom_names_1h)
        act *= mask

        return act


class AtomAttentionDecoder(nn.Module):
    # Implements Algorithm 6 from the paper
    def __init__(self, c_a, c_q):
        super().__init__()
        self.linear_a = nn.Linear(c_a, c_q, bias=False)
        self.atom_transformer = DiffusionTransformer(c_a=128, c_z=16, N_head=4, c_s=128, N_block=3, split_ada_qk=True)
        self.layer_norm_q = nn.LayerNorm(c_q, bias=False)
        self.linear_out = nn.Linear(c_q, 3, bias=False)

    def forward(self, a, q_skip, c_skip, p_skip, ref_struct: RefStructFeatures):
        a = self.linear_a(a)
        a_q = ref_struct.to_atom_layout(a, has_atom_dimension = False)
        q = a_q + q_skip
        q = self.atom_transformer(q, c_skip, p_skip, ref_struct.block_mask)
        r = self.linear_out(self.layer_norm_q(q))
        return r

class DiffusionTransformer(nn.Module):
    def __init__(self, c_a, c_z, N_head, c_s, N_block, split_ada_qk=False):
        super().__init__()
        self.att_pair_bias = nn.ModuleList([AttentionPairBias(c_a, c_z, N_head, c_s, adaptive=True, biased_layer_norm_z=False, split_ada_qk=split_ada_qk) for _ in range(N_block)])
        self.cond_trans = nn.ModuleList([ConditionedTransitionBlock(c_a, c_s) for _ in range(N_block)])
        self.N_block = N_block


    def forward(self, a, s, z, block_mask: BlockMask):
        for att_pair_block, cond_trans_block in zip(self.att_pair_bias, self.cond_trans):
            a += att_pair_block(a, z, block_mask, s=s)
            a += cond_trans_block(a, s)

        return a