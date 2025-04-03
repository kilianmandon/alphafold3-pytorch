import math
import torch
from torch import nn
import torch.nn.functional as F

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

