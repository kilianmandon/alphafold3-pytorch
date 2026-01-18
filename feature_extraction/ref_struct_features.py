from dataclasses import dataclass, fields
import math
from typing import Optional

import numpy as np
from torch.nn.attention.flex_attention import create_block_mask, BlockMask
from torch.nn import functional as F
import rdkit
import torch
from atomworks.io.tools.rdkit import atom_array_from_rdkit, ccd_code_to_rdkit
from atomworks.io.utils.ccd import get_available_ccd_codes
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms.rdkit_utils import ccd_code_to_rdkit_with_conformers, \
    sample_rdkit_conformer_for_atom_array, generate_conformers
from atomworks.ml.utils.token import get_token_count, get_token_starts
from biotite.structure import AtomArray

import utils


Array = np.ndarray | torch.Tensor

@dataclass
class RefStructFeatures:
    element: Array
    charge: Array
    atom_name_chars: Array
    positions: Array
    mask: Array
    ref_space_uid: Array
    token_index: Array

    def map_arrays(self, fn):
        field_dict = {f.name: fn(getattr(self, f.name)) for f in fields(self)}
        return RefStructFeatures(**field_dict)

    @property
    def block_mask(self) -> BlockMask:
        unpadded_atom_count = self.unpadded_atom_count
        centers = torch.arange(16, self.atom_count, 32, device=self.mask.device, dtype=torch.int32)
        left_bounds = centers - 64
        right_bounds = centers + 64
        borders = torch.stack((left_bounds, right_bounds), dim=-1)
        exceeding_left = left_bounds < 0
        exceeding_right = right_bounds >= unpadded_atom_count
        borders[exceeding_left] -= left_bounds[exceeding_left, None]
        borders[exceeding_right] -= (right_bounds[exceeding_right, None] - unpadded_atom_count)
        left_bounds, right_bounds = borders[..., 0].detach(), borders[..., 1].detach()

        def mask_mod(b, h, q, k):
            return (q < unpadded_atom_count) & (left_bounds[q//32] <= k) & (k < right_bounds[q//32])
        
        batch_size = math.prod(self.mask.shape[:-2])
        return create_block_mask(mask_mod, batch_size, None, self.atom_count, self.atom_count, self.mask.device)


    @property
    def unpadded_atom_count(self):
        if isinstance(self.mask, np.ndarray):
            return np.sum(self.mask, axis=-1)            
        else:
            return torch.sum(self.mask, dim=-1)

    @property
    def atom_count(self):
        return self.mask.shape[-1]

    @property
    def token_layout_ref_mask(self):
        token_count = self.atom_count // 24

        token_atom_counts = F.one_hot(torch.as_tensor(self.token_index[self.mask]), num_classes=token_count).sum(dim=-2)
        ref_mask = torch.arange(24, device=self.mask.device) < token_atom_counts[..., None]
        ref_mask = utils.pad_to_shape(ref_mask, (token_count, 24))

        if isinstance(self.mask, np.ndarray):
            return ref_mask.numpy()
        else:
            return ref_mask


    def to_token_layout(self, feature):
        batch_shape = self.element.shape[:-1]
        token_count = self.atom_count // 24
        token_layout_ref_mask = self.token_layout_ref_mask
        out_shape = batch_shape + (token_count, 24) + feature.shape[len(batch_shape)+1:]

        if isinstance(self.mask, np.ndarray):
            feature = np.array(feature)
            out = np.zeros(out_shape, dtype=feature.dtype)
        else:
            feature = torch.as_tensor(feature)
            out = torch.zeros(out_shape, dtype=feature.dtype, device=feature.device)
        out[token_layout_ref_mask] = feature[self.mask]

        return out


    def patch_atom_dimension(self, feature):
        batch_shape = self.element.shape[:-1]
        token_count = self.atom_count // 24
        unsqueezed_shape = batch_shape + (token_count, 1) + feature.shape[len(batch_shape) + 1:]
        broadcasted_shape = batch_shape + (token_count, 24) + feature.shape[len(batch_shape) + 1:]

        feature = feature.reshape(unsqueezed_shape)
        if isinstance(self.mask, np.ndarray):
            feature = np.array(feature)
            return np.broadcast_to(feature, broadcasted_shape)
        else:
            feature = torch.tensor(feature)
            return feature.expand(*broadcasted_shape)

    def to_atom_layout(self, feature, has_atom_dimension=True):
        batch_shape = self.element.shape[:-1]
        if not has_atom_dimension:
            feature = self.patch_atom_dimension(feature)

        out_shape = batch_shape + (self.atom_count,) + feature.shape[len(batch_shape)+2:]

        if isinstance(self.mask, np.ndarray):
            feature = np.array(feature)
            if not has_atom_dimension:
                feature = feature[len()]
            out = np.zeros(out_shape, dtype=feature.dtype)
        else:
            feature = torch.as_tensor(feature)
            out = torch.zeros(out_shape, dtype=feature.dtype, device=feature.device)

        out[self.mask] = feature[self.token_layout_ref_mask]
        return out





class CalculateRefStructFeatures(Transform):

    def __init__(self):
        ...

    def check_input(self, data):
        ...

    def prep_atom_chars(self, atom_names):
        padded = np.strings.ljust(atom_names, width=4)
        cropped = np.strings.slice(padded, 0, 4)
        encoded = np.strings.encode(cropped, encoding='ascii')
        return encoded.view(np.uint8).reshape(-1, 4) - 32

    @staticmethod
    def calculate_ref_positions(atom_array: AtomArray):
        residue_borders = get_residue_starts(atom_array, add_exclusive_stop=True)
        residue_starts, residue_ends = residue_borders[:-1], residue_borders[1:]
        res_names = atom_array.res_name[residue_starts]
        chain_iids = atom_array.chain_iid[residue_starts]
        to_generate = list(set(zip(res_names, chain_iids)))

        known_ccd_codes = get_available_ccd_codes() - { 'UNL' }

        cached_conformers = {}
        cached_unknown_conformers = {}

        ref_pos = np.zeros((len(atom_array), 3))

        for i, (res_name, chain_iid) in enumerate(zip(res_names, chain_iids)):
            if res_name not in known_ccd_codes:
                if res_name not in cached_unknown_conformers:
                    res_atom_array = atom_array[residue_starts[i]:residue_ends[i]]
                    cached_unknown_conformers[res_name] = sample_rdkit_conformer_for_atom_array(res_atom_array)

                conformer = cached_unknown_conformers[res_name]
            else:
                if (res_name, chain_iid) not in cached_conformers:
                    # mol = ccd_code_to_rdkit_with_conformers(res_name, 1, timeout=None, seed=1, optimize=False, attempts_with_distance_geometry=1)
                    mol = ccd_code_to_rdkit(res_name, hydrogen_policy='keep')
                    annotations = mol._annotations
                    order = np.argsort(annotations['atom_name'])
                    mol = rdkit.Chem.RenumberAtoms(mol, order.tolist())
                    mol._annotations = {
                        k: v[order] for k, v in annotations.items()
                    }
                    mol = generate_conformers(mol, seed=1, optimize=False, attempts_with_distance_geometry=250, hydrogen_policy='keep')
                    cached_conformers[(res_name, chain_iid)] = atom_array_from_rdkit(mol, conformer_id=0)
                conformer = cached_conformers[(res_name, chain_iid)]

            for j in range(residue_starts[i], residue_ends[i]):
                matching_atom_idx = np.nonzero(conformer.atom_name == atom_array.atom_name[j])[0]
                if len(matching_atom_idx) == 0:
                    print('Warning: could not find matching atom for residue {}'.format(res_name))
                else:
                    ref_pos[j] = conformer.coord[matching_atom_idx]


        return ref_pos


    def forward(self, data: dict):
        atom_array: AtomArray = data['atom_array']

        residue_starts = get_residue_starts(atom_array)
        ref_space_uid = np.arange(len(residue_starts))
        _, closest_start = utils.round_down_to(np.arange(len(atom_array)), residue_starts, return_indices=True)
        ref_space_uid = ref_space_uid[closest_start]

        token_starts = get_token_starts(atom_array)
        _, closest_start = utils.round_down_to(np.arange(len(atom_array)), token_starts, return_indices=True)
        token_index = np.arange(len(token_starts))
        token_index = token_index[closest_start]



        ref_struct = {
            'element': atom_array.atomic_number,
            'charge': atom_array.charge,
            'atom_name_chars': self.prep_atom_chars(atom_array.atom_name),
            'positions': self.calculate_ref_positions(atom_array).astype(np.float32),
            'mask': np.ones_like(atom_array.atomic_number).astype(bool),
            'ref_space_uid': ref_space_uid,
            'token_index': token_index,
        }

        padded_token_count = data['token_features'].restype.shape[0]
        padded_atom_count = padded_token_count * 24


        for k, v in ref_struct.items():
            padded_shape = (padded_atom_count,) + v.shape[1:]
            ref_struct[k] = utils.pad_to_shape(v, padded_shape)

        data['ref_struct'] = RefStructFeatures(**ref_struct)

        return data