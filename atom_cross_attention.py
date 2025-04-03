import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
import tqdm

import utils


class InputEmbedder(nn.Module):
    def __init__(self, c_z=128, c_s=384, tf_dim=447, rel_feat_dim=139):
        super().__init__()
        self.left_single = nn.Linear(tf_dim, c_z, bias=False)
        self.right_single = nn.Linear(tf_dim, c_z, bias=False)
        self.position_activations = nn.Linear(rel_feat_dim, c_z, bias=False)
        self.bond_embedding = nn.Linear(1, c_z, bias=False)
        self.single_embedding = nn.Linear(tf_dim, c_s, bias=False)
        self.atom_cross_att = AtomAttentionEncoder(c_s, c_z)

    def relative_encoding(self, batch, rmax=32, smax=2):
        token_index = batch['token_index']
        residue_index = batch['residue_index']
        asym_id = batch['asym_id']
        entity_id = batch['entity_id']
        sym_id = batch['sym_id']

        left_token_index, right_token_index = token_index[...,
                                                          None], token_index[..., None, :]
        left_residue_index, right_residue_index = residue_index[...,
                                                                None], residue_index[..., None, :]
        left_asym_id, right_asym_id = asym_id[..., None], asym_id[..., None, :]
        left_entity_id, right_entity_id = entity_id[...,
                                                    None], entity_id[..., None, :]
        left_sym_id, right_sym_id = sym_id[..., None], sym_id[..., None, :]

        same_chain = left_asym_id == right_asym_id
        same_residue = left_residue_index == right_residue_index
        same_entity = left_entity_id == right_entity_id

        residue_dist = torch.clip(
            left_residue_index-right_residue_index+rmax, 0, 2*rmax)
        residue_dist[~same_chain] = 2*rmax+1

        token_dist = torch.clip(
            left_token_index-right_token_index + rmax, 0, 2*rmax)
        token_dist[~(same_chain & same_residue)] = 2*rmax+1

        chain_dist = torch.clip(left_sym_id-right_sym_id+smax, 0, 2*smax)
        chain_dist[~same_chain] = 2*smax+1

        a_rel_pos = F.one_hot(residue_dist, 2*rmax+2)
        a_rel_token = F.one_hot(token_dist, 2*rmax+2)
        a_rel_chain = F.one_hot(chain_dist, 2*smax+2)

        p = torch.cat(
            (a_rel_pos, a_rel_token, same_entity[..., None], a_rel_chain), dim=-1)
        rel_feat = p.float()
        rel_enc = self.position_activations(rel_feat)

        return rel_enc, rel_feat

    def forward(self, batch):
        # Implements Line 1 to Line 5 from Algorithm 1
        target_feat = batch['target_feat']
        token_act, _ = self.atom_cross_att(batch['ref_struct'])
        s_input = torch.cat((target_feat, token_act), dim=-1)

        s_init = self.single_embedding(s_input)
        a = self.left_single(s_input)
        b = self.right_single(s_input)
        z_init = a[..., None, :] + b[..., None, :, :]

        rel_enc, rel_feat = self.relative_encoding(batch)
        z_init += rel_enc
        z_init += self.bond_embedding(batch['contact_matrix'])

        return s_input, s_init, z_init, rel_feat


class Evoformer(nn.Module):
    # AF: N_cycle=11
    def __init__(self, N_cycle=11, c_s=384, c_z=128, msa_feat_dim=34, tf_dim=447):
        super().__init__()
        self.input_embedder = InputEmbedder(c_s=c_s, c_z=c_z)

        self.layer_norm_prev_z = nn.LayerNorm(c_z)
        self.prev_z_embedding = nn.Linear(c_z, c_z, bias=False)
        self.layer_norm_prev_s = nn.LayerNorm(c_s)
        self.prev_s_embedding = nn.Linear(c_s, c_s, bias=False)

        self.template_embedder = TemplateEmbedder(c_z)
        self.msa_module = MSAModule(msa_feat_dim=msa_feat_dim, tf_dim=tf_dim)
        self.pairformer = PairFormer()
        self.N_cycle = N_cycle
        self.c_s = c_s
        self.c_z = c_z

    def forward(self, batch):
        c_s = self.c_s
        c_z = self.c_z

        batch_shape = batch['target_feat'].shape[:-2]
        N_token = batch['target_feat'].shape[-2]
        single_mask = batch['single_mask']
        device = batch['target_feat'].device

        s_input, s_init, z_init, rel_enc = self.input_embedder(batch)

        prev_s = torch.zeros(batch_shape+(N_token, c_s), device=device)
        prev_z = torch.zeros(batch_shape+(N_token, N_token, c_z), device=device)

        for i in tqdm.tqdm(range(self.N_cycle)):
            sub_batch = copy.copy(batch)
            sub_batch['msa_feat'] = sub_batch['msa_feat'][..., i]
            sub_batch['msa_mask'] = sub_batch['msa_mask'][..., i]

            z = z_init + self.prev_z_embedding(self.layer_norm_prev_z(prev_z))
            z += self.template_embedder(batch, z)
            # Note: += in the paper for the next line, not +
            z = self.msa_module(sub_batch, s_input, z)
            s = s_init + self.prev_s_embedding(self.layer_norm_prev_s(prev_s))
            s, z = self.pairformer(s, z, single_mask)
            prev_s, prev_z = s, z

        return s_input, s, z, rel_enc


class TemplateEmbedder(nn.Module):
    def __init__(self, c_z, c=64, N_blocks=2):
        super().__init__()
        self.linear_a = nn.Linear(106, c, bias=False)
        self.linear_z = nn.Linear(c_z, c, bias=False)
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.layer_norm_v = nn.LayerNorm(c)
        self.linear_out = nn.Linear(c, c_z, bias=False)
        self.pair_stack = nn.ModuleList(
            [PairStack(c, N_intermediate=2) for _ in range(N_blocks)])
        self.c = c

    def forward(self, batch, z):
        # TODO: Implement actual templates
        batch_shape = batch['target_feat'].shape[:-2]
        N_token = batch['target_feat'].shape[-2]
        N_templates = 4
        single_mask = batch['single_mask']
        device = batch['target_feat'].device

        dummy_a = torch.zeros(batch_shape+(N_token, N_token, N_templates, 106), device=device)
        dummy_aatype = torch.zeros(batch_shape+(N_token,), device=device).long()
        dummy_aatype = F.one_hot(dummy_aatype, 31)
        dummy_a[..., 40:71] = dummy_aatype[..., None, :, None, :]
        dummy_a[..., 71:102] = dummy_aatype[..., :, None, None, :]
        u = torch.zeros(batch_shape+(N_token, N_token, self.c), device=device)
        for i in range(N_templates):
            v = self.linear_z(self.layer_norm_z(z)) + \
                self.linear_a(dummy_a[..., i, :])
            for block in self.pair_stack:
                v = block(v, single_mask)
            u += self.layer_norm_v(v)

        u = u / N_templates
        u = self.linear_out(torch.relu(u))

        return u


class OuterProductMean(nn.Module):
    def __init__(self, c_m=64, c=32, c_z=128):
        super().__init__()
        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_a = nn.Linear(c_m, c, bias=False)
        self.linear_b = nn.Linear(c_m, c, bias=False)
        self.linear_out = nn.Linear(c**2, c_z)

    def forward(self, msa_feat, msa_mask):
        # msa_feat: Shape (*, N_seq, N_tokens, c_m)
        # mask: Shape (*, N_seq, N_tokens)
        N_seq, N_tokens = msa_feat.shape[-3:-1]
        m = self.layer_norm(msa_feat)
        m_a = self.linear_a(m) * msa_mask[..., None]
        m_b = self.linear_b(m) * msa_mask[..., None]

        ab = torch.einsum('...sic,...sjd->...ijcd', m_a, m_b)
        ab = ab.flatten(start_dim=-2)
        z = self.linear_out(ab)

        norm = torch.einsum('...si,...sj->...ij', msa_mask, msa_mask)
        z = z / (norm[..., None] + 1e-3)
        return z


class MSAPairWeightedAveraging(nn.Module):
    def __init__(self, c=32, N_head=8, c_m=64, c_z=128):
        super().__init__()
        self.layer_norm_m = nn.LayerNorm(c_m)
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear_v = nn.Linear(c_m, c*N_head, bias=False)
        self.linear_b = nn.Linear(c_z, N_head, bias=False)
        self.linear_g = nn.Linear(c_m, c*N_head, bias=False)
        self.linear_out = nn.Linear(c*N_head, c_m, bias=False)
        self.N_head = N_head
        self.c = c

    def forward(self, m, z, single_mask):
        # m has shape (*, N_seq, N_token, c_m)
        # z has shape (*, N_token, N_token, c_z)
        m = self.layer_norm_m(m)
        v = self.linear_v(m).unflatten(-1, (self.N_head, self.c))
        b = self.linear_b(self.layer_norm_z(z))
        g = torch.sigmoid(self.linear_g(m))

        b += -1e9 * (1-single_mask[..., None, :, None])

        w = torch.softmax(b, dim=-2)
        o = torch.einsum('...ijh,...sjhc->...sihc', w, v)
        o = g * o.flatten(start_dim=-2)
        m = self.linear_out(o)
        return m


class Transition(nn.Module):
    def __init__(self, c, n=4):
        super().__init__()
        self.layer_norm = nn.LayerNorm(c)
        self.linear_a = nn.Linear(c, n*c, bias=False)
        self.linear_b = nn.Linear(c, n*c, bias=False)
        self.linear_out = nn.Linear(n*c, c, bias=False)

    def forward(self, x):
        x = self.layer_norm(x)
        a = self.linear_a(x)
        b = self.linear_b(x)
        x = self.linear_out(F.silu(a) * b)
        return x


class TriangleMultiplication(nn.Module):
    def __init__(self, c_z, c, outgoing=True):
        super().__init__()
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear_a1 = nn.Linear(c_z, c, bias=False)
        self.linear_a2 = nn.Linear(c_z, c, bias=False)
        self.linear_b1 = nn.Linear(c_z, c, bias=False)
        self.linear_b2 = nn.Linear(c_z, c, bias=False)
        self.linear_g = nn.Linear(c_z, c_z, bias=False)
        self.layer_norm_out = nn.LayerNorm(c)
        self.linear_out = nn.Linear(c, c_z, bias=False)
        self.outgoing = outgoing

    def forward(self, z, single_mask):
        pair_mask = single_mask[..., :, None] * single_mask[..., None, :]
        z = self.layer_norm_z(z)
        a = torch.sigmoid(self.linear_a1(z)) * self.linear_a2(z)
        b = torch.sigmoid(self.linear_b1(z)) * self.linear_b2(z)
        g = torch.sigmoid(self.linear_g(z))

        a = a * pair_mask[..., None]
        b = b * pair_mask[..., None]

        if self.outgoing:
            o = torch.einsum('...ikc,...jkc->...ijc', a, b)
        else:
            # Note: Correct Indexing would look like this
            # o = torch.einsum('...kic,...kjc->...ijc', a, b)
            o = torch.einsum('...kjc,...kic->...ijc', a, b)
        z = g * self.linear_out(self.layer_norm_out(o))

        return z


class TriangleAttention(nn.Module):
    def __init__(self, c_z, c, N_head=4, starting_node=True):
        super().__init__()
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear_q = nn.Linear(c_z, N_head*c, bias=False)
        self.linear_k = nn.Linear(c_z, N_head*c, bias=False)
        self.linear_v = nn.Linear(c_z, N_head*c, bias=False)
        self.linear_b = nn.Linear(c_z, N_head, bias=False)
        self.linear_g = nn.Linear(c_z, N_head*c, bias=False)
        self.linear_out = nn.Linear(c*N_head, c_z, bias=False)
        self.N_head = N_head
        self.c = c
        self.starting_node = starting_node

    def forward(self, z, single_mask):
        N_head = self.N_head
        c = self.c

        z = self.layer_norm_z(z)
        q = self.linear_q(z).unflatten(-1, (N_head, c))
        k = self.linear_k(z).unflatten(-1, (N_head, c))
        v = self.linear_v(z).unflatten(-1, (N_head, c))
        g = self.linear_g(z).unflatten(-1, (N_head, c))

        if not self.starting_node and False:
            q = q.transpose(-3, -4)
            k = k.transpose(-3, -4)
            v = v.transpose(-3, -4)
            g = g.transpose(-3, -4)

        bias = self.linear_b(z)

        if self.starting_node:
            bias = bias[..., None, :, :, :]
            a = 1/math.sqrt(c) * \
                torch.einsum('...ijhc,...ikhc->ijkh', q, k) + bias
            # a = 1/math.sqrt(c) * torch.einsum('...jihc,...kihc->ijkh', q, k) + bias
            # a = 1/math.sqrt(c) * torch.einsum('...ijhc,...kjhc->jikh', q, k) + bias
        else:
            # I'm pretty sure this would be the correct variant for indexing
            # bias = bias[..., None, :, :].transpose(-2, -4)
            bias = bias[..., None, :, :, :].transpose(-3, -4)
            a = 1/math.sqrt(c) * \
                torch.einsum('...ijhc,...kjhc->ijkh', q, k) + bias

        a += -1e9 * (1-single_mask[..., None, None, :, None])
        a = torch.softmax(a, dim=-2)

        if self.starting_node:
            o = torch.einsum('...ijkh,...ikhc->...ijhc', a, v)
        else:
            o = torch.einsum('...ijkh,...kjhc->ijhc', a, v)

        o = torch.sigmoid(g) * o
        o = o.flatten(-2)
        z = self.linear_out(o)

        return z


class SharedDropout(nn.Module):
    def __init__(self, p, shared_dim):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.shared_dim = shared_dim

    def forward(self, x):
        mask_shape = list(x.shape)
        mask_shape[self.shared_dim] = 1
        mask = torch.ones(mask_shape, device=x.device)
        mask = self.dropout(mask)
        return x*mask


class DropoutRowwise(SharedDropout):
    def __init__(self, p):
        super().__init__(p, shared_dim=-2)


class DropoutColumnwise(SharedDropout):
    def __init__(self, p):
        super().__init__(p, shared_dim=-3)


class PairStack(nn.Module):
    def __init__(self, c_z, N_head=4, N_intermediate=4):
        super().__init__()
        c_att = c_z//N_head
        self.dropout_rowwise = DropoutRowwise(0.25)
        self.dropout_columnwise = DropoutColumnwise(0.25)
        self.triangle_mult_outgoing = TriangleMultiplication(
            c_z=c_z, c=c_z, outgoing=True)
        self.triangle_mult_incoming = TriangleMultiplication(
            c_z=c_z, c=c_z, outgoing=False)
        self.triangle_att_starting = TriangleAttention(
            c_z=c_z, c=c_att, N_head=N_head, starting_node=True)
        self.triangle_att_ending = TriangleAttention(
            c_z=c_z, c=c_att, N_head=N_head, starting_node=False)
        self.transition = Transition(c_z, n=N_intermediate)

    def forward(self, z, single_mask):
        z += self.dropout_rowwise(self.triangle_mult_outgoing(z, single_mask))
        z += self.dropout_rowwise(self.triangle_mult_incoming(z, single_mask))
        z += self.dropout_rowwise(self.triangle_att_starting(z, single_mask))
        z += self.dropout_columnwise(self.triangle_att_ending(z, single_mask))
        z += self.transition(z)
        return z


class MSAModuleBlock(nn.Module):
    def __init__(self, c_m, c_z):
        super().__init__()
        self.dropout_rowwise = DropoutRowwise(0.15)
        self.opm = OuterProductMean()
        self.msa_pair_weighted = MSAPairWeightedAveraging(c=8)
        self.transition = Transition(c_m)
        self.core = PairStack(c_z)

    def forward(self, m, z, msa_mask, single_mask):
        z += self.opm(m, msa_mask)
        m += self.dropout_rowwise(self.msa_pair_weighted(m, z, single_mask))
        m += self.transition(m)

        z = self.core(z, single_mask)
        return m, z


class MSAModule(nn.Module):
    def __init__(self, N_block=4, c_m=64, c_z=128, msa_feat_dim=None, tf_dim=None):
        super().__init__()
        self.linear_m = nn.Linear(msa_feat_dim, c_m, bias=False)
        self.linear_s = nn.Linear(tf_dim, c_m, bias=False)
        self.blocks = nn.ModuleList(
            [MSAModuleBlock(c_m, c_z) for _ in range(N_block)])

    def forward(self, batch, s, z):
        msa_feat = batch['msa_feat']
        msa_mask = batch['msa_mask']
        single_mask = batch['single_mask']

        m = self.linear_m(msa_feat)
        m += self.linear_s(s)

        for block in self.blocks:
            m, z = block(m, z, msa_mask, single_mask)
        return z


class AttentionPairBias(nn.Module):
    def __init__(self, c_a, c_z, N_head, c_s=None, adaptive=False, biased_layer_norm_z=True):
        super().__init__()
        c = c_a//N_head
        if adaptive:
            self.layer_norm_a = AdaptiveLayerNorm(c_a, c_s)
            # Should be initialized with bias=-2
            self.linear_out_adaptive = nn.Linear(c_s, c_a)
        else:
            self.layer_norm_a = nn.LayerNorm(c_a)

        self.layer_norm_z = nn.LayerNorm(c_z, bias=biased_layer_norm_z)
        self.linear_q = nn.Linear(c_a, c*N_head)
        self.linear_k = nn.Linear(c_a, c*N_head, bias=False)
        self.linear_v = nn.Linear(c_a, c*N_head, bias=False)
        self.linear_b = nn.Linear(c_z, N_head, bias=False)
        self.linear_g = nn.Linear(c_a, c*N_head, bias=False)
        self.linear_out = nn.Linear(c*N_head, c_a, bias=False)

        self.N_head = N_head
        self.c = c
        self.adaptive = adaptive

    def forward(self, a, z, mask, s=None):
        N_head = self.N_head
        c = self.c

        if s is not None:
            a = self.layer_norm_a(a, s)
        else:
            a = self.layer_norm_a(a)

        q = self.linear_q(a).unflatten(-1, (N_head, c))
        k = self.linear_k(a).unflatten(-1, (N_head, c))
        v = self.linear_v(a).unflatten(-1, (N_head, c))

        b = self.linear_b(self.layer_norm_z(z))

        g = torch.sigmoid(self.linear_g(a))


        q = q / math.sqrt(c)
        a = torch.einsum('...ihc,...jhc->...ijh', q, k) + b
        a += -1e9 * (1-mask[..., None, :, None])
        a = torch.softmax(a, dim=-2)

        o = torch.einsum('...ijh,...jhc->...ihc', a, v)
        o = g * o.flatten(-2)
        o = self.linear_out(o)

        if self.adaptive:
            o = torch.sigmoid(self.linear_out_adaptive(s)) * o

        return o


class PairFormerBlock(nn.Module):
    def __init__(self, c_z=128, c_s=384):
        super().__init__()
        self.core = PairStack(c_z)
        self.att_pair_bias = AttentionPairBias(c_s, c_z, N_head=16)
        self.single_transition = Transition(c_s)

    def forward(self, s, z, single_mask):
        z = self.core(z, single_mask)
        s += self.att_pair_bias(s, z, single_mask)
        s += self.single_transition(s)
        return s, z


class PairFormer(nn.Module):
    def __init__(self, N_block=48):
        super().__init__()
        self.blocks = nn.ModuleList([PairFormerBlock()
                                    for _ in range(N_block)])

    def forward(self, s, z, single_mask):
        for block in tqdm.tqdm(self.blocks):
            s, z = block(s, z, single_mask)
        return s, z


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, c_a, c_s):
        super().__init__()
        self.layer_norm = nn.LayerNorm(c_a, elementwise_affine=False)
        self.single_cond_layer_norm = nn.LayerNorm(c_s, bias=False)
        self.single_cond_scale = nn.Linear(c_s, c_a)
        self.single_cond_bias = nn.Linear(c_s, c_a, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, single_cond):
        x = self.layer_norm(x)
        single_cond = self.single_cond_layer_norm(single_cond)
        single_scale = self.single_cond_scale(single_cond)
        single_bias = self.single_cond_bias(single_cond)
        x = self.sigmoid(single_scale) * x + single_bias
        return x


class AdaptiveZeroInit(nn.Module):
    def __init__(self, c_in, c_in_cond, c_out):
        super().__init__()
        self.linear_transition = nn.Linear(c_in, c_out, bias=False)
        # For training: Initialization to weight 0, bias -2
        self.linear_cond = nn.Linear(c_in_cond, c_out)

    def forward(self, x, single_cond):
        out = self.linear_transition(x)
        gate = self.linear_cond(single_cond)
        out = torch.sigmoid(gate) * out
        return out


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


def diff(t1, t2, return_err=False, atol=1e-4, rtol=1e-5):
    if t1.shape != t2.shape:
        print(f"Shape mismatch: {t1.shape} | {t2.shape}")
        return

    adiff = (t1-t2).abs()
    rdiff = adiff / (0.5 * t1.abs() + 0.5 * t2.abs())
    diff_inds = torch.nonzero((adiff > atol) & (rdiff > rtol)).numpy()

    if return_err:
        return diff_inds, adiff, rdiff
    else:
        return diff_inds


def compare(v, name, return_err=False, inds=None, atol=1e-4, rtol=1e-5):
    w = torch.load(
        f'kilian/feature_extraction/test_outputs/{name}.pt', weights_only=False)
    if inds is not None:
        w = w[inds]
        v = v[inds]
    return diff(v, w, return_err, atol=atol, rtol=rtol)
