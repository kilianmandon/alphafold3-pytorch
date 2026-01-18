import pickle
from torch import nn
import torch
from evoformer import Evoformer
from diffusion import DiffusionModule, DiffusionSampler
from feature_extraction.feature_extraction import Batch
import utils
import tensortrace as ttr

class Model(nn.Module):
    def __init__(self, N_cycle=11, noise_steps=30):
        super().__init__()
        self.evoformer = Evoformer(N_cycle=N_cycle)
        self.diffusion_module = DiffusionModule()
        self.diffusion_sampler = DiffusionSampler(noise_steps=noise_steps)

    def forward(self, batch: Batch):
        ref_struct = batch.ref_struct
        single_mask = batch.token_features.mask

        s_input, s_trunk, z_trunk, rel_enc = self.evoformer(batch)

        x_flat = self.diffusion_sampler(self.diffusion_module,
                            s_input, s_trunk, z_trunk, rel_enc, 
                            ref_struct, single_mask)


        return x_flat

