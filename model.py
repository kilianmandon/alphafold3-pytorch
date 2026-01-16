import pickle
from torch import nn
import torch
from evoformer import Evoformer
from diffusion import DiffusionModule, DiffusionSampler
import utils
import tensortrace as ttr

class Model(nn.Module):
    def __init__(self, N_cycle=11, noise_steps=30):
        super().__init__()
        self.evoformer = Evoformer(N_cycle=N_cycle)
        self.diffusion_module = DiffusionModule()
        self.diffusion_sampler = DiffusionSampler(noise_steps=noise_steps)

    def forward(self, batch):
        ref_struct = batch['ref_struct']
        single_mask = batch['token_features']['single_mask']

        with ttr.Chapter('evoformer'):
            s_input, s_trunk, z_trunk, rel_enc = self.evoformer(batch)

        with ttr.Chapter('diffusion'):
            x_flat = self.diffusion_sampler(self.diffusion_module,
                                s_input, s_trunk, z_trunk, rel_enc, 
                                ref_struct, single_mask)

        atom_layout = ref_struct['atom_layout']
        token_mask = ref_struct['ref_mask']
        x_out = atom_layout.queries_to_tokens(x_flat, n_feat_dims=1)

        return x_out, token_mask

