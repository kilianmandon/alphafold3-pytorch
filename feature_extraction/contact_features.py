import numpy as np
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_token_starts

import utils


class CalculateContactMatrix(Transform):
    def forward(self, data):
        atom_array = data['atom_array']

        atom1_idxs, atom2_idxs, _ = atom_array.bonds.as_array().T
        token_starts = get_token_starts(atom_array)
        _, atom1_token_indices = utils.round_down_to(atom1_idxs, token_starts, return_indices=True)
        _, atom2_token_indices = utils.round_down_to(atom2_idxs, token_starts, return_indices=True)

        padded_token_count = data['token_features']['restype'].shape[0]
        is_atomized = utils.pad_to_shape_np(atom_array[token_starts].atomize, (padded_token_count,))

        atom1_token_indices, atom2_token_indices = np.minimum(atom1_token_indices, atom2_token_indices), np.maximum(atom1_token_indices, atom2_token_indices)
        bond_matrix = np.zeros((padded_token_count, padded_token_count))
        bond_matrix[atom1_token_indices, atom2_token_indices] = 1

        non_poly_poly = is_atomized.reshape(-1, 1) | is_atomized.reshape(1, -1)
        non_diagonal = 1 - np.eye(padded_token_count)

        bond_matrix = bond_matrix * non_poly_poly * non_diagonal

        data['contact_matrix'] = bond_matrix[..., None].astype(np.float32)

        return data