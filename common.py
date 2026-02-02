import math
from torch.nn.attention.flex_attention import BlockMask, flex_attention
import utils
import torch
from torch import nn
import torch.nn.functional as F
import tensortrace as ttr

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


class AttentionPairBias(nn.Module):
    def __init__(self, c_a, c_z, N_head, c_s=None, adaptive=False, biased_layer_norm_z=True, split_ada_qk=False, no_compilation=False):
        super().__init__()
        c = c_a//N_head
        if adaptive:
            if split_ada_qk:
                self.layer_norm_q = AdaptiveLayerNorm(c_a, c_s)
                self.layer_norm_k = AdaptiveLayerNorm(c_a, c_s)
            else:
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

        if torch.cuda.is_available() and not no_compilation:
            self.flex_attention = torch.compile(flex_attention)
        else:
            self.flex_attention = flex_attention

        self.N_head = N_head
        self.c = c
        self.adaptive = adaptive
        self.split_ada_qk = split_ada_qk

    def forward(self, a, z, block_mask: BlockMask, s=None):
        batch_shape = a.shape[:-2]
        N_head = self.N_head
        N_token = a.shape[-2]
        c = self.c

        if s is not None:
            if self.split_ada_qk:
                a_q = self.layer_norm_q(a, s)
                a_k = self.layer_norm_k(a, s)
            else:   
                a_q = self.layer_norm_a(a, s)
                a_k = a_q
        else:
            a_q = self.layer_norm_a(a)
            a_k = a_q

        q = self.linear_q(a_q).unflatten(-1, (N_head, c))
        k = self.linear_k(a_k).unflatten(-1, (N_head, c))
        v = self.linear_v(a_k).unflatten(-1, (N_head, c))
        g = self.linear_g(a_q).unflatten(-1, (N_head, c))

        bias = self.linear_b(self.layer_norm_z(z))

        q = torch.einsum('...ihc->...hic', q)
        k = torch.einsum('...jhc->...hjc', k)
        v = torch.einsum('...jhc->...hjc', v)

        q = utils.unify_batch_dimension(q, batch_shape)
        k = utils.unify_batch_dimension(k, batch_shape)
        v = utils.unify_batch_dimension(v, batch_shape)
        bias = utils.unify_batch_dimension(bias, batch_shape)

        def bias_score_mod(score, b, h, q_idx, kv_idx):
            return score + bias[b, q_idx, kv_idx, h]

        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
        o = self.flex_attention(q, k, v, score_mod=bias_score_mod, block_mask=block_mask)

        o = o.reshape(batch_shape + (N_head, N_token, c))
        o = torch.einsum('...hjc->...jhc', o)

        o = torch.sigmoid(g) * o
        o = o.flatten(-2)

        o = self.linear_out(o)

        if self.adaptive:
            o = torch.sigmoid(self.linear_out_adaptive(s)) * o

        return o

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


class ConditionedTransitionBlock(nn.Module):
    def __init__(self, c_a, c_s, n=2):
        super().__init__()
        self.adaptive_layernorm = AdaptiveLayerNorm(c_a, c_s)
        self.linear_a1 = nn.Linear(c_a, n*c_a, bias=False)
        self.linear_a2 = nn.Linear(c_a, n*c_a, bias=False)
        # Note: This should be initialized with bias -2
        self.ada_zero_init = AdaptiveZeroInit(n*c_a, c_s, c_a)
    
    def forward(self, a, s):
        a = self.adaptive_layernorm(a, s)
        b = F.silu(self.linear_a1(a)) * self.linear_a2(a)
        a = self.ada_zero_init(b, s)
        return a
