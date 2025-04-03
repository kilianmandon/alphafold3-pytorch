import torch
from torch import nn

from common import AdaptiveLayerNorm, AdaptiveZeroInit
import utils



class CrossAttentionBlock(nn.Module):
    def __init__(self, n_heads, c_in=128, key_dim=128, value_dim=128):
        super().__init__()
        self.n_heads = n_heads
        self.layer_norm_q = AdaptiveLayerNorm(c_in, c_in)
        self.layer_norm_k = AdaptiveLayerNorm(c_in, c_in)
        self.key_dim_per_head = key_dim//n_heads
        self.value_dim_per_head = value_dim//n_heads
        self.q_projection = nn.Linear(c_in, key_dim)
        self.k_projection = nn.Linear(c_in, key_dim, bias=False)
        self.v_projection = nn.Linear(c_in, value_dim, bias=False)
        self.gating_query = nn.Linear(c_in, value_dim, bias=False)

        self.ada_zero_init = AdaptiveZeroInit(c_in, c_in, c_in)

    def forward(self, x_q, x_k, mask_q, mask_k, pair_logits, single_cond_q, single_cond_k):
        batch_dim = x_q.shape[:-2]
        n_queries = x_q.shape[-2]
        n_keys = x_k.shape[-2]
        device = x_q.device

        x_q = self.layer_norm_q(x_q, single_cond_q)
        x_k = self.layer_norm_k(x_k, single_cond_k)
        q = self.q_projection(x_q)
        q = q.unflatten(-1, (self.n_heads, self.key_dim_per_head))
        k = self.k_projection(x_k)
        k = k.unflatten(-1, (self.n_heads, self.key_dim_per_head))
        # I think it should be like this, but AlphaFold's masks computation is weird.
        # but no keys are masked anyway
        # bias = (mask_q[..., None] & mask_k[..., None, :]).to(torch.float32) * -1e9
        # bias = bias.unsqueeze(-3)
        bias = torch.zeros(batch_dim + (1, n_queries, n_keys), device=device)
        scale = self.key_dim_per_head ** -0.5
        # Values here get really large, might be numerically unstable? Calculated in fp32 in AF3
        logits = torch.einsum('...qnc,...knc->...nqk', q * scale, k) + bias
        if pair_logits is not None:
            logits += pair_logits
        attn_weights = torch.softmax(logits, dim=-1)
        v = self.v_projection(x_k)
        v = v.unflatten(-1, (self.n_heads, self.value_dim_per_head))

        out = torch.einsum('...nqk,...knc->...qnc', attn_weights, v)
        out = out.flatten(start_dim=-2)
        gate_logits = self.gating_query(x_q)

        out *= torch.sigmoid(gate_logits)
        out = self.ada_zero_init(out, single_cond_q)
        return out


class TransitionBlock(nn.Module):
    def __init__(self, n_intermediate_factor, c_in=128):
        super().__init__()
        self.ada_layer_norm = AdaptiveLayerNorm(c_in, c_in)
        c_inter = n_intermediate_factor * c_in
        self.glu1 = nn.Linear(c_in, c_inter, bias=False)
        self.silu = nn.SiLU()
        self.glu2 = nn.Linear(c_in, c_inter, bias=False)
        self.ada_zero_init = AdaptiveZeroInit(c_inter, c_in, c_in)

    def forward(self, x, single_cond):
        x = self.ada_layer_norm(x, single_cond)
        x = self.silu(self.glu1(x)) * self.glu2(x)
        x = self.ada_zero_init(x, single_cond)
        return x


class AtomTransformer(nn.Module):
    def __init__(self, c_in=16):
        super().__init__()
        self.n_blocks = 3
        self.n_heads = 4
        self.pair_input_layer_norm = nn.LayerNorm(c_in, bias=False)
        self.pair_logits_projection = nn.Linear(
            c_in, self.n_blocks * self.n_heads, bias=False)
        self.attn_blocks = nn.ModuleList(
            [CrossAttentionBlock(self.n_heads) for _ in range(self.n_blocks)])
        self.transition_blocks = nn.ModuleList(
            [TransitionBlock(n_intermediate_factor=2) for _ in range(self.n_blocks)])

    def forward(self, queries_act, atom_layout, queries_single_cond, pair_cond):
        keys_single_cond = atom_layout.queries_to_keys(queries_single_cond, n_feat_dims=1)
        queries_mask = atom_layout.tokens_to_queries.target_mask
        keys_mask = atom_layout.tokens_to_keys.target_mask

        pair_act = self.pair_input_layer_norm(pair_cond)
        pair_logits = self.pair_logits_projection(pair_act)
        # (num_subsets, num_queries, num_keys, num_blocks, num_heads)
        pair_logits = pair_logits.unflatten(-1, (self.n_blocks, self.n_heads))
        # (num_blocks, num_subsets, num_heads, num_queries, num_keys)
        # pair_logits = pair_logits.permute((3, 0, 4, 1, 2))
        pair_logits = torch.einsum('...ijklo->...liojk', pair_logits)

        for i, (attn_block, transition_block) in enumerate(zip(self.attn_blocks, self.transition_blocks)):
            # keys_act = queries_act.reshape(-1, 128)[key_inds, :]
            keys_act = atom_layout.queries_to_keys(queries_act, 1)
            queries_act += attn_block(queries_act, keys_act, queries_mask,
                                      keys_mask, pair_logits[i], queries_single_cond, keys_single_cond)
            queries_act += transition_block(queries_act, queries_single_cond)

        return queries_act


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

        self.cross_att_transformer = AtomTransformer()
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
        # real_atom_count = torch.count_nonzero(ref_struct['mask'])
        # padded_atom_count = flat_ref_struct['mask'].shape[0]
        # key_inds = self.calculate_key_inds(real_atom_count, padded_atom_count)
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
            ref_struct['positions'], 1)
        keys_ref_pos = atom_layout.tokens_to_keys(ref_struct['positions'], 1)

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

        queries_act = self.cross_att_transformer(
            queries_act,
            atom_layout,
            queries_single_cond,
            pair_act
        )

        queries_act *= queries_mask[..., None]

        token_act = atom_layout.queries_to_tokens(queries_act, 1)
        token_act = torch.relu(self.project_atom_features(token_act))

        # atom_count_per_res = torch.sum(ref_struct['mask'], dim=-1)
        # atom_count_per_res = atom_count_per_res[..., None]

        # token_act = token_act.sum(
            # dim=-2) / torch.clip(atom_count_per_res, min=1e-10)
        token_act = utils.masked_mean(token_act, ref_struct['mask'][..., None], dim=-2)

        skip = (queries_act, queries_single_cond, pair_act)

        return token_act, skip

    def per_atom_cond(self, flat_ref_struct):
        keys = ['positions', 'mask', 'element', 'charge', 'atom_name_chars']

        def diff(t1, t2):
            if t1.shape != t2.shape:
                print(f"Shape mismatch: {t1.shape} | {t2.shape}")
                return
            return torch.nonzero((t1-t2).abs() > 1e-3).numpy()

        mask = flat_ref_struct['mask'][..., None].to(torch.float32)
        element = flat_ref_struct['element']
        charge = flat_ref_struct['charge'][..., None].to(torch.float32)
        name_chars = flat_ref_struct['atom_name_chars']
        elements_1h = nn.functional.one_hot(element, 128).to(torch.float32)
        atom_names_1h = nn.functional.one_hot(name_chars, 64).to(torch.float32)
        atom_names_1h = atom_names_1h.reshape(atom_names_1h.shape[:-2] + (-1,))

        act = self.embed_ref_pos(flat_ref_struct['positions'])
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
