import numpy as np
import torch
from atomworks.constants import UNKNOWN_AA
from atomworks.ml.transforms.atom_array import get_chain_instance_starts
from atomworks.ml.transforms.base import Transform, Compose
from atomworks.ml.transforms.msa._msa_constants import MSA_INTEGER_TO_THREE_LETTER
from atomworks.ml.transforms.msa.msa import LoadPolymerMSAs, PairAndMergePolymerMSAs
from atomworks.ml.utils.token import get_token_count, get_token_starts

from torch.nn import functional as F
import utils
from residue_constants import AF3_TOKENS_MAP, _PROTEIN_TO_ID

class HotfixDuplicateRowIfSingleMSA(Transform):
    def __init__(self):
        ...

    def forward(self, data):
        if len(data['polymer_msas_by_chain_id']) == 1:
            msa_data = list(data['polymer_msas_by_chain_id'].values())[0]
            for k, v in msa_data.items():
                new_shape = (v.shape[0]+1,) + v.shape[1:]
                new_val = np.zeros(new_shape, v.dtype)
                new_val[1:] = v
                new_val[0] = v[0]
                msa_data[k] = new_val
            msa_data['msa_is_padded_mask'][0] = True

        return data

class HotfixEncodeRNAAsProtein(Transform):
    def forward(self, data):
        is_rna = data['token_features']['is_rna']
        is_dna = data['token_features']['is_dna']
        is_nucleic = is_rna | is_dna

        if np.all(~is_nucleic):
            return data

        for k in data['msa_features']:
            if k == 'full_msa_mask':
                continue
            original_data = data['msa_features'][k][:, is_nucleic]
            shifted = np.roll(original_data, 1, axis=0)
            if k != 'individual_msa_mask':
                shifted[0] = shifted[1]
            data['msa_features'][k][:, is_nucleic] = shifted

        msa_first_row = data['msa_features']['msa'][0, is_nucleic]
        hotfix_map = {
            i: i for i in range(32)
        } | {
            AF3_TOKENS_MAP['A']: AF3_TOKENS_MAP['ALA'],
            AF3_TOKENS_MAP['C']: AF3_TOKENS_MAP['CYS'],
            AF3_TOKENS_MAP['G']: AF3_TOKENS_MAP['GLY'],
            AF3_TOKENS_MAP['U']: AF3_TOKENS_MAP['CYS'],
            AF3_TOKENS_MAP['DA']: AF3_TOKENS_MAP['ALA'],
            AF3_TOKENS_MAP['DC']: AF3_TOKENS_MAP['CYS'],
            AF3_TOKENS_MAP['DG']: AF3_TOKENS_MAP['GLY'],
            AF3_TOKENS_MAP['DT']: AF3_TOKENS_MAP['THR'],
        }
        msa_first_row = np.vectorize(hotfix_map.get)(msa_first_row)
        data['msa_features']['msa'][0, is_nucleic] = msa_first_row

        return data

class HotfixAF3LigandAsGap(Transform):
    def forward(self, data):
        ligand_inds = np.nonzero(data['token_features']['is_ligand'])[0]
        data['msa_features']['msa'][0, ligand_inds] = _PROTEIN_TO_ID['-']
        return data


class EncodeMSA(Transform):
    def __init__(self):
        self.lookup_table = np.zeros(len(MSA_INTEGER_TO_THREE_LETTER), dtype=np.int8)
        for i_msa, code in MSA_INTEGER_TO_THREE_LETTER.items():
            self.lookup_table[i_msa] = AF3_TOKENS_MAP.get(code, AF3_TOKENS_MAP[UNKNOWN_AA])

    def forward(self, data):
        for chain_id, msa_data in data['polymer_msas_by_chain_id'].items():
            data['polymer_msas_by_chain_id'][chain_id]['msa'] = self.lookup_table[msa_data['msa']]
        return data



class DeduplicateMSA(Transform):
    def __init__(self):
        ...

    def forward(self, data):
        polymer_msas = data['polymer_msas_by_chain_id']

        for chain_id in polymer_msas:
            _, unique_inds, inv = np.unique(polymer_msas[chain_id]['msa'], axis=0, return_index=True,
                                            return_inverse=True)
            unique_inds = np.sort(unique_inds)
            for k, v in polymer_msas[chain_id].items():
                polymer_msas[chain_id][k] = v[unique_inds]

        return data

class ConcatMSAs(Transform):
    def __init__(self, max_msa_sequences=16384):
        self.max_msa_sequences = max_msa_sequences

    def forward(self, data):
        polymer_msas = data['polymer_msas_by_chain_id']
        atom_array = data['atom_array']

        max_msa_size = max(msa_data['msa'].shape[0] for msa_data in polymer_msas.values())
        token_count = get_token_count(atom_array)
        padded_token_count = data['token_features']['restype'].shape[0]

        full_msa = np.zeros((self.max_msa_sequences, padded_token_count), dtype=np.int64)
        deletion_count = np.zeros((self.max_msa_sequences, padded_token_count), dtype=np.float32)
        full_msa_mask = np.zeros((self.max_msa_sequences, padded_token_count), dtype=np.float32)
        individual_msa_mask = np.zeros((self.max_msa_sequences, padded_token_count), dtype=np.float32)

        full_msa[:max_msa_size, :token_count] = AF3_TOKENS_MAP['<G>']
        full_msa_mask[:max_msa_size, :token_count] = 1

        full_msa[0] = data['token_features']['restype']
        full_msa_mask[0, :token_count] = 1
        individual_msa_mask[0, :token_count] = 1

        tokens = data['atom_array'][get_token_starts(data['atom_array'])]
        chain_borders = get_chain_instance_starts(tokens, add_exclusive_stop=True)
        chain_starts, chain_ends = chain_borders[:-1], chain_borders[1:]

        for chain_start, chain_end in zip(chain_starts, chain_ends):
            chain_id = tokens.chain_id[chain_start]
            if chain_id in polymer_msas:
                chain_tokens = tokens[chain_start:chain_end]

                atomizing_starts = np.nonzero(~chain_tokens.atomize[1:] | ~chain_tokens.atomize[:-1])[0] + 1
                atomizing_starts = np.concatenate(([0], atomizing_starts)) + chain_start
                msa_to_fill = polymer_msas[chain_id]['msa']
                ins_to_fill = polymer_msas[chain_id]['ins']
                msa_mask_to_fill = ~polymer_msas[chain_id]['msa_is_padded_mask']

                full_msa[:msa_to_fill.shape[0], atomizing_starts] = msa_to_fill
                deletion_count[:ins_to_fill.shape[0], atomizing_starts] = ins_to_fill
                individual_msa_mask[:msa_mask_to_fill.shape[0], atomizing_starts] = msa_mask_to_fill

        data['msa_features'] = {
            'msa': full_msa,
            'deletion_count': deletion_count,
            'full_msa_mask': full_msa_mask,
            'individual_msa_mask': individual_msa_mask,
        }

        return data

class EncodeMSAFeatures(Transform):
    def __init__(self, msa_trunc_count=1024, msa_shuffle_orders=None, n_recycling_iterations=1):
        self.msa_trunc_count = msa_trunc_count
        self.msa_shuffle_orders = msa_shuffle_orders
        self.n_recycling_iterations = n_recycling_iterations

    def forward(self, data):
        deletion_count = data['msa_features']['deletion_count']
        msa = data['msa_features']['msa']
        full_msa_mask = data['msa_features']['full_msa_mask']
        individual_msa_mask = data['msa_features']['individual_msa_mask']
        restype = data['token_features']['restype']

        deletion_mean = utils.masked_mean_np(deletion_count, individual_msa_mask, axis=0)
        deletion_value = (2 / np.pi) * np.arctan(deletion_count / 3)
        msa_one_hot = F.one_hot(torch.tensor(msa), num_classes=32).numpy()
        profile = utils.masked_mean_np(msa_one_hot, individual_msa_mask[..., None], axis=0)

        deletion_value = deletion_value[..., None]
        has_deletion = np.clip(deletion_count, a_min=0, a_max=1, dtype=np.float32)[..., None]

        full_msa_feat = np.concatenate([msa_one_hot, has_deletion, deletion_value], axis=-1)
        msa_feat, msa_mask = self.sample_msa_features(full_msa_feat, full_msa_mask, self.msa_shuffle_orders)
        target_feat = self.calculate_target_feat(restype, profile, deletion_mean)

        data['msa_features']['msa_feat'] = msa_feat.astype(np.float32)
        data['msa_features']['msa_mask'] = msa_mask.astype(np.float32)
        data['msa_features']['target_feat'] = target_feat.astype(np.float32)
        data['msa_features']['profile'] = profile.astype(np.float32)

        return data

    def calculate_target_feat(self, restype, profile, deletion_mean):
        restype_one_hot = F.one_hot(torch.tensor(restype), num_classes=32).numpy()

        target_feat = np.concatenate((restype_one_hot, profile, deletion_mean[..., None]), axis=-1)
        return target_feat

    def sample_msa_features(self, base_msa_feat, base_msa_mask, msa_shuffle_orders):
        has_entries = np.clip(np.sum(base_msa_mask, axis=-1), a_min=0, a_max=1)
        odds = (has_entries - 1) * -1e2

        if self.msa_shuffle_orders is None:
            scores_shape = (self.n_recycling_iterations,) + odds.shape
            scores = odds + torch.distributions.Gumbel(0, 1).sample(scores_shape).numpy()
            msa_shuffle_orders = np.argsort(scores, axis=-1)

        msa_indices = msa_shuffle_orders[:, :self.msa_trunc_count]
        msa_feat = base_msa_feat[msa_indices]
        msa_mask = base_msa_mask[msa_indices]

        msa_feat = np.moveaxis(msa_feat, 0, -1)
        msa_mask = np.moveaxis(msa_mask, 0, -1)

        return msa_feat, msa_mask



class CalculateMSAFeatures(Transform):
    def __init__(self, protein_msa_dirs=None, rna_msa_dirs=None, max_msa_sequences=16384, msa_trunc_count=1024, msa_shuffle_orders=None, n_recycling_iterations=1):
        self.transforms = Compose([
            LoadPolymerMSAs(
                protein_msa_dirs=protein_msa_dirs,
                rna_msa_dirs=rna_msa_dirs,
                max_msa_sequences=max_msa_sequences,
                use_paths_in_chain_info=True,
            ),
            DeduplicateMSA(),
            HotfixDuplicateRowIfSingleMSA(),
            EncodeMSA(),
            # PairAndMergePolymerMSAs(dense=True),
            ConcatMSAs(max_msa_sequences=max_msa_sequences),
            HotfixEncodeRNAAsProtein(),
            HotfixAF3LigandAsGap(),
            EncodeMSAFeatures(msa_trunc_count, msa_shuffle_orders, n_recycling_iterations)
        ])

    def forward(self, data: dict):
        data = self.transforms(data)

        return data


