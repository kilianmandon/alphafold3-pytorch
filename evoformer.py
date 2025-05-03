import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
import tqdm
from torch.nn.attention.flex_attention import flex_attention

from common import AttentionPairBias, Transition
from input_embedder import InputEmbedder


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

        if torch.cuda.is_available():
            self.flex_attention = torch.compile(flex_attention)
        else:
            self.flex_attention = flex_attention

    def forward(self, z, single_mask):
        N_head = self.N_head
        c = self.c
        N_token = z.shape[-3]
        batch_shape = z.shape[:-3]

        z = self.layer_norm_z(z)
        q = self.linear_q(z).unflatten(-1, (N_head, c))
        k = self.linear_k(z).unflatten(-1, (N_head, c))
        v = self.linear_v(z).unflatten(-1, (N_head, c))
        g = self.linear_g(z).unflatten(-1, (N_head, c))

        bias = self.linear_b(z)

        if self.starting_node:
            bias = bias[..., None, :, :, :]
            bias += -1e9 * (1-single_mask[..., None, None, :, None])
            q = torch.einsum('...ijhc->...ihjc', q)
            k = torch.einsum('...ikhc->...ihkc', k)
            v = torch.einsum('...ikhc->...ihkc', v)
            bias = torch.einsum('...ijkh->...ihjk', bias)
            
        else:
            # I'm pretty sure this would be the correct variant for indexing
            # bias = bias[..., None, :, :].transpose(-2, -4)
            bias = bias[..., None, :, :, :].transpose(-3, -4)
            bias += -1e9 * (1-single_mask[..., None, None, :, None])
            # Layout conversion
            q = torch.einsum('...ijhc->...jhic', q)
            k = torch.einsum('...kjhc->...jhkc', k)
            v = torch.einsum('...kjhc->...jhkc', v)
            bias = torch.einsum('...ijkh->...jhik', bias)

        q = torch.flatten(q, end_dim=-4)
        k = torch.flatten(k, end_dim=-4)
        bias = torch.flatten(bias, end_dim=-4)

        def bias_score_mod(score, b, h, q_idx, kv_idx):
            # Broadcasting of bias index over missing token dimension
            b = b // N_token
            return score + bias[b, h, q_idx, kv_idx]

        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()

        o = self.flex_attention(q, k, v, score_mod=bias_score_mod)

        o = o.reshape(batch_shape + (N_token, N_head, N_token, c))

        if self.starting_node:
            o = torch.einsum('...ihjc->...ijhc', o)
        else:
            o = torch.einsum('...jhic->...ijhc', o)

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