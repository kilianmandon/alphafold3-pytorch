import time

import numpy as np
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
from atom_layout import AtomLayout


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
        print('Starting ref positions...')
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

        residue_borders = get_residue_starts(atom_array, add_exclusive_stop=True)
        residue_starts, residue_ends = residue_borders[:-1], residue_borders[1:]
        ref_space_uid = np.arange(len(residue_starts))
        _, closest_start = utils.round_down_to(np.arange(len(atom_array)), residue_starts, return_indices=True)
        ref_space_uid = ref_space_uid[closest_start]



        ref_struct = {
            'ref_element': atom_array.atomic_number,
            'ref_charge': atom_array.charge,
            'ref_atom_name_chars': self.prep_atom_chars(atom_array.atom_name),
            # 'atom_names': atom_array.atom_name,
            'ref_pos': self.calculate_ref_positions(atom_array).astype(np.float32),
            'ref_mask': np.ones_like(atom_array.atomic_number),
            'ref_space_uid': ref_space_uid,
        }

        padded_token_count = data['token_features']['restype'].shape[0]
        padded_atom_count = padded_token_count * 24
        atom_layout = AtomLayout.from_atom_array(atom_array, padded_token_count)

        N_blocks = padded_token_count * 24 // 32


        for k, v in ref_struct.items():
            padded_shape = (padded_atom_count,) + v.shape[1:]
            v = utils.pad_to_shape_np(v, padded_shape)

            n_feat_dims = len(v.shape) - 1
            v_new_shape = (N_blocks, 32) + v.shape[1:]
            v = atom_layout.queries_to_tokens(torch.tensor(v).reshape(v_new_shape), n_feat_dims=n_feat_dims).numpy()

            ref_struct[k] = v


        ref_struct['atom_layout'] = atom_layout

        data['ref_struct'] = ref_struct

        return data