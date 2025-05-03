import pickle
from torch import nn
import torch
from evoformer import Evoformer
from diffusion import DiffusionModule, DiffusionSampler
import utils

class Model(nn.Module):
    def __init__(self, N_cycle=11, noise_steps=30):
        super().__init__()
        self.evoformer = Evoformer(N_cycle=N_cycle)
        self.diffusion_module = DiffusionModule()
        self.diffusion_sampler = DiffusionSampler(noise_steps=noise_steps)

    def forward(self, batch):
        s_input, s_trunk, z_trunk, rel_enc = self.evoformer(batch)

        x_flat = self.diffusion_sampler(self.diffusion_module,
                               s_input, s_trunk, z_trunk, rel_enc, 
                               batch['ref_struct'], batch['single_mask'])

        atom_layout = batch['ref_struct']['atom_layout']
        token_mask = batch['ref_struct']['mask']
        x_out = atom_layout.queries_to_tokens(x_flat, n_feat_dims=1)

        return x_out, token_mask

