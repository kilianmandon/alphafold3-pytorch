import numpy as np
from atomworks.constants import UNKNOWN_AA, STANDARD_RNA, UNKNOWN_RNA, STANDARD_DNA, UNKNOWN_DNA, STANDARD_AA
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_token_starts

from feature_extraction.feature_extraction import round_to_bucket
from residue_constants import AF3_TOKENS_MAP
import utils


def encode_restype(restype):
    return np.vectorize(lambda x: AF3_TOKENS_MAP.get(x, AF3_TOKENS_MAP[UNKNOWN_AA]))(restype)


class CalculateTokenFeatures(Transform):
    def __init__(self):
        ...

    def forward(self, data):
        atom_array = data['atom_array']
        token_starts = get_token_starts(atom_array)

        token_array = atom_array[token_starts]

        token_index = np.arange(len(token_array))
        res_id = token_array.res_id

        entity_id = np.unique(token_array.pn_unit_entity, return_inverse=True)[1]
        asym_id = np.unique(token_array.pn_unit_iid, return_inverse=True)[1]

        entity_id_changes = np.nonzero(entity_id[1:] != entity_id[:-1])[0] + 1
        entity_id_changes = np.concatenate(([0], entity_id_changes))
        start_of_entities = entity_id_changes[entity_id]
        sym_id = asym_id - asym_id[start_of_entities]

        restype = encode_restype(token_array.res_name)

        is_rna = np.isin(token_array.res_name, STANDARD_RNA +(UNKNOWN_RNA,))
        is_dna = np.isin(token_array.res_name, STANDARD_DNA + (UNKNOWN_DNA,))
        is_protein = np.isin(token_array.res_name, STANDARD_AA + (UNKNOWN_AA,))
        is_ligand = ~(is_rna | is_dna | is_protein)


        token_features = {
            'residue_index': res_id,
            'token_index': token_index + 1,
            'asym_id': asym_id + 1,
            'entity_id': entity_id + 1,
            'sym_id': sym_id + 1,
            'single_mask': np.ones(len(token_array)),
            'restype': restype,

            'is_rna': is_rna,
            'is_dna': is_dna,
            'is_protein': is_protein,
            'is_ligand': is_ligand,
        }

        padded_token_count = round_to_bucket(len(token_array))
        for k, v in token_features.items():
            token_features[k] = utils.pad_to_shape_np(v, (padded_token_count,))

        data['token_features'] = token_features

        return data









