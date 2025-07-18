import math
import torch
from torch import nn

from common import AdaptiveLayerNorm, AdaptiveZeroInit, ConditionedTransitionBlock
import utils



class AtomAttentionPairBias(nn.Module):
    def __init__(self, N_head, c_in=128, c_z=16):
        super().__init__()
        # According to the paper, these should be shared
        self.layer_norm_q = AdaptiveLayerNorm(c_in, c_in)
        self.layer_norm_k = AdaptiveLayerNorm(c_in, c_in)
        c = c_in//N_head

        self.layer_norm_z = nn.LayerNorm(c_z, bias=False)
        self.linear_q = nn.Linear(c_in, c*N_head)
        self.linear_k = nn.Linear(c_in, c*N_head, bias=False)
        self.linear_v = nn.Linear(c_in, c*N_head, bias=False)
        self.linear_b = nn.Linear(c_z, N_head, bias=False)
        self.linear_g = nn.Linear(c_in, c*N_head, bias=False)

        self.ada_zero_init = AdaptiveZeroInit(c_in, c_in, c_in)
        self.c = c
        self.N_head = N_head

    def forward(self, a_q, a_k, z, s_q, s_k):
        N_head = self.N_head
        c = self.c

        a_q = self.layer_norm_q(a_q, s_q)
        a_k = self.layer_norm_k(a_k, s_k)

        q = self.linear_q(a_q).unflatten(-1, (N_head, c))
        k = self.linear_k(a_k).unflatten(-1, (N_head, c))
        v = self.linear_v(a_k).unflatten(-1, (N_head, c))

        b = self.linear_b(self.layer_norm_z(z))
        b = torch.einsum('...qkn->...nqk', b)

        g = torch.sigmoid(self.linear_g(a_q))

        # Values here get really large, might be numerically unstable? Calculated in fp32 in AF3
        # Even changing the layout to qkn instead of nqk significantly increases deviation from AF3
        q = q / math.sqrt(c)
        att = torch.einsum('...qnc,...knc->...nqk', q, k) + b

        att = torch.softmax(att, dim=-1)

        o = torch.einsum('...nqk,...knc->...qnc', att, v)
        o = g * o.flatten(start_dim=-2)

        # According to the paper, there should be an additional linear layer before ada_zero_init
        o = self.ada_zero_init(o, s_q)

        return o



class AtomTransformer(nn.Module):
    def __init__(self, c_in=16):
        super().__init__()
        self.N_block = 3
        self.N_head = 4
        self.attn_blocks = nn.ModuleList(
            [AtomAttentionPairBias(self.N_head) for _ in range(self.N_block)])
        self.transition_blocks = nn.ModuleList(
            [ConditionedTransitionBlock(c_a=128, c_s=128, n=2) for _ in range(self.N_block)])

    def forward(self, a_q, atom_layout, s_q, pair_cond):
        s_k = atom_layout.queries_to_keys(s_q, n_feat_dims=1)

        for attn_block, transition_block in zip(self.attn_blocks, self.transition_blocks):
            a_k = atom_layout.queries_to_keys(a_q, 1)
            a_q += attn_block(a_q, a_k, pair_cond, s_q, s_k)
            a_q += transition_block(a_q, s_q)

        return a_q


class AtomAttentionEncoder(nn.Module):
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

        self.atom_transformer = AtomTransformer()
        self.project_atom_features = nn.Linear(c_atom, c_token, bias=False)

        self.use_trunk = use_trunk
        if use_trunk:
            self.trunk_layer_norm_s = nn.LayerNorm(c_s, bias=False)
            self.trunk_linear_s = nn.Linear(c_s, c_atom, bias=False)
            self.trunk_layer_norm_z = nn.LayerNorm(c_z, bias=False)
            self.trunk_linear_z = nn.Linear(c_z, c_atom_pair, bias=False)
            self.trunk_linear_r = nn.Linear(3, c_atom, bias=False)


    def forward(self, ref_struct, r=None, s_trunk=None, z=None):
        atom_layout = ref_struct['atom_layout']
        per_atom = self.per_atom_cond(ref_struct)
        queries_single_cond = atom_layout.tokens_to_queries(per_atom, 1)
        queries_act = queries_single_cond.clone()
        queries_mask = atom_layout.tokens_to_queries.target_mask

        keys_mask = atom_layout.tokens_to_keys.target_mask

        queries_ref_space_uid = atom_layout.tokens_to_queries(
            ref_struct['ref_space_uid'], 0)

        # Note: Correct indexing would look like this
        # keys_ref_space_uid = atom_layout.tokens_to_keys(ref_struct['ref_space_uid'], 0)
        # But we follow Deepminds implementation:
        keys_ref_space_uid = atom_layout.queries_to_keys(
            ref_struct['ref_space_uid'], 0)

        queries_ref_pos = atom_layout.tokens_to_queries(
            ref_struct['ref_pos'], 1)
        keys_ref_pos = atom_layout.tokens_to_keys(ref_struct['ref_pos'], 1)

        offsets_valid = (queries_ref_space_uid.unsqueeze(-1) ==
                         keys_ref_space_uid.unsqueeze(-2)).float()
        offsets_valid = offsets_valid.unsqueeze(-1)
        offsets = queries_ref_pos.unsqueeze(-2) - keys_ref_pos.unsqueeze(-3)

        pair_act = self.embed_pair_offsets(offsets) * offsets_valid
        sq_dists = offsets.square().sum(dim=-1, keepdim=True)

        pair_act += self.embed_pair_distances(1/(1+sq_dists)) * offsets_valid

        if self.use_trunk:
            s_trunk = atom_layout.pure_tokens_to_queries(s_trunk, 1)
            z = atom_layout.pure_pair_to_qk(z, 1)

            queries_single_cond += self.trunk_linear_s(
                self.trunk_layer_norm_s(s_trunk))
            pair_act += self.trunk_linear_z(self.trunk_layer_norm_z(z))

            # Note: The paper uses the old, non-trunk-updated value
            # for queries_single_cond here
            queries_act = queries_single_cond + self.trunk_linear_r(r)

        keys_single_cond = atom_layout.queries_to_keys(queries_single_cond, 1)

        row_act = self.single_to_pair_row(torch.relu(queries_single_cond))
        col_act = self.single_to_pair_col(torch.relu(keys_single_cond))
        pair_act += row_act[:, :, None, :] + col_act[:, None, :, :]

        pair_act += self.embed_pair_mask(offsets_valid)

        pair_act += self.pair_mlp(pair_act)

        queries_act = self.atom_transformer(
            queries_act,
            atom_layout,
            queries_single_cond,
            pair_act
        )

        queries_act *= queries_mask[..., None]

        token_act = atom_layout.queries_to_tokens(queries_act, 1)
        token_act = torch.relu(self.project_atom_features(token_act))

        # atom_count_per_res = torch.sum(ref_struct['ref_mask'], dim=-1)
        # atom_count_per_res = atom_count_per_res[..., None]

        # token_act = token_act.sum(
            # dim=-2) / torch.clip(atom_count_per_res, min=1e-10)
        token_act = utils.masked_mean(token_act, ref_struct['ref_mask'][..., None], dim=-2)

        skip = (queries_act, queries_single_cond, pair_act)

        return token_act, skip

    def per_atom_cond(self, flat_ref_struct):
        keys = ['ref_pos', 'ref_mask', 'ref_element', 'ref_charge', 'ref_atom_name_chars']

        def diff(t1, t2):
            if t1.shape != t2.shape:
                print(f"Shape mismatch: {t1.shape} | {t2.shape}")
                return
            return torch.nonzero((t1-t2).abs() > 1e-3).numpy()

        mask = flat_ref_struct['ref_mask'][..., None].to(torch.float32)
        element = flat_ref_struct['ref_element']
        charge = flat_ref_struct['ref_charge'][..., None].to(torch.float32)
        name_chars = flat_ref_struct['ref_atom_name_chars']
        elements_1h = nn.functional.one_hot(element, 128).to(torch.float32)
        atom_names_1h = nn.functional.one_hot(name_chars, 64).to(torch.float32)
        atom_names_1h = atom_names_1h.reshape(atom_names_1h.shape[:-2] + (-1,))

        act = self.embed_ref_pos(flat_ref_struct['ref_pos'])
        act += self.embed_ref_mask(mask)
        act += self.embed_ref_element(elements_1h)
        act += self.embed_ref_charge(torch.arcsinh(charge))

        act += self.embed_ref_atom_name(atom_names_1h)
        act *= mask

        return act


class AtomAttentionDecoder(nn.Module):
    def __init__(self, c_a, c_q):
        super().__init__()
        self.linear_a = nn.Linear(c_a, c_q, bias=False)
        self.atom_transformer = AtomTransformer()
        self.layer_norm_q = nn.LayerNorm(c_q, bias=False)
        self.linear_out = nn.Linear(c_q, 3, bias=False)

    def forward(self, a, q_skip, c_skip, p_skip, ref_struct):
        atom_layout = ref_struct['atom_layout']
        a = self.linear_a(a)
        a_q = atom_layout.pure_tokens_to_queries(a, n_feat_dims=1)
        q = a_q + q_skip
        q = self.atom_transformer(q, atom_layout, c_skip, p_skip)
        r = self.linear_out(self.layer_norm_q(q))
        return r
