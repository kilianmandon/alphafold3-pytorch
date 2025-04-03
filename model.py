import pickle
from torch import nn
import torch
from evoformer import Evoformer
from diffusion import DiffusionModule, DiffusionSampler

class Model(nn.Module):
    def __init__(self, N_cycle=11, noise_steps=30):
        super().__init__()
        self.evoformer = Evoformer(N_cycle=N_cycle)
        self.diffusion_module = DiffusionModule()
        self.diffusion_sampler = DiffusionSampler(noise_steps=noise_steps)

    def forward(self, batch):
        s_input, s_trunk, z_trunk, rel_enc = self.evoformer(batch)

        data = {'s_input': s_input, 's_trunk': s_trunk, 'z_trunk': z_trunk, 'rel_enc': rel_enc, 'batch': batch}
        data = move_to_device(data, 'cpu')
        with open('pytorch_variables.pkl', 'wb') as file:
            pickle.dump(data, file)
        # device = batch['target_feat'].device
        # with open('pytorch_variables.pkl', 'rb') as file:
        #     data = pickle.load(file)
        # data = move_to_device(data, device)

        # s_input = data['s_input']
        # s_trunk = data['s_trunk']
        # z_trunk = data['z_trunk']
        # rel_enc = data['rel_enc']
        # batch = data['batch']
        

        x_flat = self.diffusion_sampler(self.diffusion_module,
                               s_input, s_trunk, z_trunk, rel_enc, 
                               batch['ref_struct'], batch['single_mask'])

        atom_layout = batch['ref_struct']['atom_layout']
        token_mask = batch['ref_struct']['mask']
        x_out = atom_layout.queries_to_tokens(x_flat, n_feat_dims=1)

        return x_out, token_mask


def move_to_device(obj, device):
    """Recursively move all tensors in a nested structure to a specified device."""
    if isinstance(obj, torch.Tensor):  
        return obj.to(device)  # Move tensor to device
    elif isinstance(obj, dict):  
        return {key: move_to_device(value, device) for key, value in obj.items()}  
    elif isinstance(obj, list):  
        return [move_to_device(item, device) for item in obj]  
    elif isinstance(obj, tuple):  
        return tuple(move_to_device(item, device) for item in obj)  
    elif hasattr(obj, "__dict__"):  
        # If the object has a __dict__, it's a custom class -> move its attributes
        for attr in vars(obj):  
            setattr(obj, attr, move_to_device(getattr(obj, attr), device))
        return obj  
    else:
        return obj  # Return unchanged if not a tensor, dict, list, or tuple
