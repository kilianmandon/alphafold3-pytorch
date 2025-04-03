from torch import nn
from torch.nn import functional as F
import torch
import tqdm

from atom_cross_attention import AdaptiveLayerNorm, AdaptiveZeroInit, AtomAttentionDecoder, AtomAttentionEncoder, AttentionPairBias, Transition
import utils





class DiffusionModule(nn.Module):
    def __init__(self, c_atom=128, c_a=768, c_s=384, c_z=128, sigma_data=16):
        super().__init__()
        self.diffusion_conditioning = DiffusionConditioning(sigma_data)
        self.atom_att_enc = AtomAttentionEncoder(c_s, c_z, use_trunk=True, c_token=c_a)
        self.diffusion_transformer = DiffusionTransformer(c_a, c_z, N_head=16, c_s=c_s, N_block=24)
        self.atom_att_dec = AtomAttentionDecoder(c_a, c_atom)

        self.layer_norm_s = nn.LayerNorm(c_s, bias=False)
        self.linear_s = nn.Linear(c_s, c_a, bias=False)
        self.layer_norm_a = nn.LayerNorm(c_a, bias=False)

        self.sigma_data = sigma_data

    def forward(self, x_noisy, t_hat, s_inputs, s_trunk, z_trunk, rel_enc, ref_struct, mask):
        # x_noisy has shape (**batch_shape, N_blocks, 32, 3)
        # t_hat has shape (**batch_shape, )
        atom_layout = ref_struct['atom_layout']

        # compare_all({
        #     'x_noisy': atom_layout.queries_to_tokens(x_noisy, n_feat_dims=1),
        #     's_inputs': s_inputs,
        #     's_trunk': s_trunk,
        #     'z_trunk': z_trunk,
        #     'rel_enc': rel_enc,
        # }, 'debug_diff_start_it2', 'Diff Start Iteration 2', atom_layout)

        s, z = self.diffusion_conditioning(t_hat, s_inputs, s_trunk, z_trunk, rel_enc)
        r=x_noisy / torch.sqrt(t_hat**2+self.sigma_data**2)[..., None, None, None]


        a, (q_skip, c_skip, p_skip) = self.atom_att_enc(ref_struct, r=r, s_trunk=s_trunk, z=z)


        a += self.linear_s(self.layer_norm_s(s))
        a = self.diffusion_transformer(a, s, z, mask, atom_layout)

        a = self.layer_norm_a(a)

        r_update = self.atom_att_dec.forward(a, q_skip, c_skip, p_skip, ref_struct)

        d_skip = self.sigma_data**2 / (self.sigma_data**2+t_hat**2)
        d_scale = self.sigma_data * t_hat / torch.sqrt(self.sigma_data**2 + t_hat**2)
        d_skip = d_skip[..., None, None, None]
        d_scale = d_scale[..., None, None, None]
        
        x_out = d_skip * x_noisy + d_scale * r_update

        return x_out



class DiffusionConditioning(nn.Module):
    def __init__(self, sigma_data, c_z=128, c_s=384, c_fourier=256, rel_feat_dim=139, tf_dim=447):
        super().__init__()
        self.sigma_data = sigma_data

        self.linear_z = nn.Linear(rel_feat_dim + c_z, c_z, bias=False)
        self.layer_norm_z = nn.LayerNorm(rel_feat_dim + c_z, bias=False)
        self.z_transition = nn.ModuleList([Transition(c_z, n=2) for _ in range(2)])

        self.layer_norm_s = nn.LayerNorm(tf_dim + c_s, bias=False)
        self.linear_s = nn.Linear(tf_dim + c_s, c_s, bias=False)
        self.s_transition = nn.ModuleList([Transition(c_s, n=2) for _ in range(2)])

        self.layer_norm_fourier = nn.LayerNorm(c_fourier, bias=False)
        self.linear_fourier = nn.Linear(c_fourier, c_s, bias=False)

        self.fourier_w = nn.Parameter(torch.randn((c_fourier,)), requires_grad=False)
        self.fourier_b = nn.Parameter(torch.randn((c_fourier,)), requires_grad=False)

    def fourier_embedding(self, t_hat):
        # t_hat has shape (**batch_shape,)
        # out should have shape (**batch_shape, 1, c_fourier)
        t_hat = t_hat[..., None, None]
        x = t_hat * self.fourier_w + self.fourier_b
        return torch.cos(2 * torch.pi * x)

    def forward(self, t_hat, s_inputs, s_trunk, z_trunk, rel_feat):
        z = torch.cat((z_trunk, rel_feat), dim=-1)
        z = self.linear_z(self.layer_norm_z(z))
        for block in self.z_transition:
            z += block(z)

        s = torch.cat((s_trunk, s_inputs), dim=-1)
        s = self.linear_s(self.layer_norm_s(s))
        c_noise = 1/4 * torch.log(t_hat/self.sigma_data)
        n = self.fourier_embedding(c_noise)
        s += self.linear_fourier(self.layer_norm_fourier(n))

        for block in self.s_transition:
            s += block(s)
        
        return s, z

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


class DiffusionTransformer(nn.Module):
    def __init__(self, c_a, c_z, N_head, c_s, N_block):
        super().__init__()
        self.att_pair_bias = nn.ModuleList([AttentionPairBias(c_a, c_z, N_head, c_s, adaptive=True, biased_layer_norm_z=False) for _ in range(N_block)])
        self.cond_trans = nn.ModuleList([ConditionedTransitionBlock(c_a, c_s) for _ in range(N_block)])
        self.N_block = N_block


    def forward(self, a, s, z, mask, atom_layout):
        for att_pair_block, cond_trans_block in zip(self.att_pair_bias, self.cond_trans):
            # b = att_pair_block(a, z, mask, s=s)
            # a = b + cond_trans_block(a, s)
            a += att_pair_block(a, z, mask, s=s)
            a += cond_trans_block(a, s)

        return a

class DiffusionSampler(nn.Module):
    def __init__(self, noise_steps, 
                 gamma_0=0.8, gamma_min=1.0, noise_scale=1.003, step_scale=1.5):
        super().__init__()
        self.noise_steps = noise_steps
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale

        self.center_random_aug = CenterRandomAugmentation()

    def noise_schedule(self, t, sigma_data=16, smin=0.0004, smax=160.0, p=7):
        return sigma_data * (smax ** (1/p) + t * (smin**(1/p) - smax**(1/p))) ** p

    def forward(self, diffusion_module, s_inputs, s_trunk, z_trunk, rel_enc, ref_struct, mask, noise_data=None):
        # q2k_mask has shape (**batch_shape, N_block, 32,)
        t2q_mask = ref_struct['atom_layout'].tokens_to_queries.target_mask
        batch_shape = t2q_mask.shape[:-2]
        N_block = t2q_mask.shape[-2]
        device = s_inputs.device
        
        noise_levels = self.noise_schedule(torch.linspace(0, 1, self.noise_steps+1, device=device))
        x_shape = batch_shape + (N_block, 32, 3)

        if noise_data is not None:
            x = noise_data['init_pos']
        else:
            x = noise_levels[0] * torch.randn(x_shape, device=device)
        # Note: For Debugging
        # x = torch.load('kilian/pure_test_outputs/diff_initial_positions.pt', weights_only=False)[0].to(device=device)
        # x = ref_struct['atom_layout'].tokens_to_queries(x, n_feat_dims=1)

        for i, (c_prev, c) in tqdm.tqdm(enumerate(zip(noise_levels[:-1], noise_levels[1:])), total=self.noise_steps):

            if noise_data is not None:
                rand_rot = noise_data['aug_rot'][i]
                rand_trans = noise_data['aug_trans'][i]
            else:
                rand_rot = rand_trans = None

            x = self.center_random_aug(x, t2q_mask, ref_struct, rand_rot=rand_rot, rand_trans=rand_trans)

            # compare_all({
            #     'x': ref_struct['atom_layout'].queries_to_tokens(x, n_feat_dims=1),
            # }, 'debug_diff_post_aug', 'Diff Post Aug', ref_struct['atom_layout'])

            gamma = self.gamma_0 if c > self.gamma_min else 0
            t_hat = c_prev * (gamma + 1)

            if noise_data is not None:
                noise = noise_data['noise'][i]
            else:
                noise = self.noise_scale * torch.sqrt(t_hat**2 - c_prev**2) * torch.randn(x_shape, device=device)
            # Note: For Debugging
            # noise = torch.load(f'kilian/pure_test_outputs/diff_noise_{i}.pt', weights_only=False).to(device=device)
            # noise = ref_struct['atom_layout'].tokens_to_queries(noise, n_feat_dims=1)

            x_noisy = x+noise
            x_denoised = diffusion_module.forward(x_noisy, t_hat, s_inputs, s_trunk, z_trunk, rel_enc, ref_struct, mask)

            delta = (x_noisy-x_denoised)/t_hat
            dt = c - t_hat
            x = x_noisy + self.step_scale * dt * delta
            # compare_all({'x': ref_struct['atom_layout'].queries_to_tokens(x, n_feat_dims=1), 
            #              'delta': ref_struct['atom_layout'].queries_to_tokens(delta, n_feat_dims=1),
            #              'x_denoised': ref_struct['atom_layout'].queries_to_tokens(x_denoised, n_feat_dims=1),
            #              }, 
            #              'debug_diff_sampler', 'Diffusion Sampler', ref_struct['atom_layout'])
            pass

        return x

class CenterRandomAugmentation(nn.Module):
    def __init__(self, s_trans=1):
        super().__init__()
        self.s_trans=s_trans

    def forward(self, x, mask, ref_struct, rand_rot=None, rand_trans=None):
        # x has shape (**batch_dims, N_blocks, 32, 3)
        device = x.device
        atom_layout = ref_struct['atom_layout']
        batch_shape = x.shape[:-3]

        # compare_all({'x': atom_layout.queries_to_tokens(x, n_feat_dims=1)}, 'debug_diff_aug_start', 'Diff Aug Start', atom_layout) 
        x = x - utils.masked_mean(x, mask[..., None], dim=(-2,-3), keepdim=True)
        # compare_all({'x': atom_layout.queries_to_tokens(x, n_feat_dims=1)}, 'debug_diff_aug_centered', 'Diff Aug Centered', atom_layout) 

        if rand_rot is None:
            rand_quats = torch.randn(batch_shape+(1,1,4), device=device)
            rand_quats /= torch.linalg.norm(rand_quats, dim=-1, keepdim=True)
            t = self.s_trans * torch.randn(batch_shape+(1,1,3), device=device)

            x = utils.quat_vector_mul(rand_quats, x) + t
        else:
            x = torch.einsum('ji,...j->...i', rand_rot, x) + rand_trans
        # compare_all({'x': atom_layout.queries_to_tokens(x, n_feat_dims=1)}, 'debug_diff_aug_out', 'Diff Aug Complete', atom_layout) 

        return x


