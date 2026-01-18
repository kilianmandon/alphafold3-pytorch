from dataclasses import dataclass, fields
import numpy as np
from atomworks.constants import UNKNOWN_AA, STANDARD_RNA, UNKNOWN_RNA, STANDARD_DNA, UNKNOWN_DNA, STANDARD_AA
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_token_starts
import torch

from feature_extraction.msa_features import MSAFeatures
from residue_constants import AF3_TOKENS_MAP
import utils

def round_to_bucket(v):
    buckets = np.array([256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072,
                        3584, 4096, 4608, 5120])

    return utils.round_up_to(v, buckets)

def encode_restype(restype):
    return np.vectorize(lambda x: AF3_TOKENS_MAP.get(x, AF3_TOKENS_MAP[UNKNOWN_AA]))(restype)

Array = np.ndarray | torch.Tensor

@dataclass
class TokenFeatures:
    residue_index: Array
    token_index: Array
    asym_id: Array
    entity_id: Array
    sym_id: Array
    mask: Array
    restype: Array
    is_rna: Array
    is_dna: Array
    is_protein: Array
    is_ligand: Array

    def map_arrays(self, fn):
        field_dict = {f.name: fn(getattr(self, f.name)) for f in fields(self)}
        return TokenFeatures(**field_dict)

    @property
    def token_count(self):
        return self.residue_index.shape[-1]
    


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
            'mask': np.ones(len(token_array), dtype=bool),
            'restype': restype,

            'is_rna': is_rna,
            'is_dna': is_dna,
            'is_protein': is_protein,
            'is_ligand': is_ligand,
        }

        padded_token_count = round_to_bucket(len(token_array))
        for k, v in token_features.items():
            token_features[k] = utils.pad_to_shape(v, (padded_token_count,))

        data['token_features'] = TokenFeatures(**token_features)

        return data









