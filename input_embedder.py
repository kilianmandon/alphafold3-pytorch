import torch
import tensortrace as ttr
from torch import nn
import torch.nn.functional as F

from atom_attention import AtomAttentionEncoder

def reorder_encoding(dim=-1, offset=0):
    token_enc_shift = {
                          i: i for i in range(31)
                      } | {
                          23: 24,
                          24: 23,
                          26: 27,
                          27: 29,
                          28: 28,
                          29: 30,
                          30: 26,
                      }
    def f(tensor):
        new_shape = list(tensor.shape)
        new_shape[dim] -= 1
        new_tensor = torch.zeros(new_shape, device=tensor.device,  dtype=tensor.dtype)
        new_tensor[:, :offset] = tensor[:, :offset]
        new_tensor[:, offset+31:] = tensor[:, offset+32:]
        for i_old, i_new in token_enc_shift.items():
            new_tensor[:, offset+i_old] = tensor[:, offset+i_new]
        return new_tensor
    
    return f


class InputEmbedder(nn.Module):
    def __init__(self, c_z=128, c_s=384, tf_dim=449, rel_feat_dim=139):
        super().__init__()
        self.left_single = nn.Linear(tf_dim, c_z, bias=False)
        self.right_single = nn.Linear(tf_dim, c_z, bias=False)
        self.position_activations = nn.Linear(rel_feat_dim, c_z, bias=False)
        self.bond_embedding = nn.Linear(1, c_z, bias=False)
        self.single_embedding = nn.Linear(tf_dim, c_s, bias=False)
        self.atom_cross_att = AtomAttentionEncoder(c_s, c_z)

    def relative_encoding(self, batch, rmax=32, smax=2):
        token_features = batch['token_features']
        token_index = token_features['token_index']
        residue_index = token_features['residue_index']
        asym_id = token_features['asym_id']
        entity_id = token_features['entity_id']
        sym_id = token_features['sym_id']

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
        rel_feat = p.to(dtype=torch.float32)
        rel_enc = self.position_activations(rel_feat)

        return rel_enc, rel_feat

    def forward(self, batch):
        # Implements Line 1 to Line 5 from Algorithm 1
        target_feat = batch['msa_features']['target_feat']
        with ttr.Chapter('input_embedding'):
            token_act, _ = self.atom_cross_att(batch['ref_struct'])
        ttr.compare(target_feat, 'input_embedding/target_feat', input_processing=[reorder_encoding(offset=32), reorder_encoding(offset=0)])
        ttr.compare(token_act, 'input_embedding/token_act')
        s_input = torch.cat((target_feat, token_act), dim=-1)

        s_init = self.single_embedding(s_input)
        a = self.left_single(s_input)
        b = self.right_single(s_input)
        z_init = a[..., None, :] + b[..., None, :, :]
        ttr.compare(z_init, 'evoformer/z_0_init')

        rel_enc, rel_feat = self.relative_encoding(batch)
        z_init += rel_enc
        z_init += self.bond_embedding(batch['contact_matrix'])

        return s_input, s_init, z_init, rel_feat
