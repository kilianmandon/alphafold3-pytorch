import re
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from residue_constants import _DNA_TO_ID, _PROTEIN_TO_ID, _RNA_TO_ID
import utils


id_maps = {
    'rna': _RNA_TO_ID,
    'dna': _DNA_TO_ID,
    'protein': _PROTEIN_TO_ID,
    'ligand': _PROTEIN_TO_ID,
}


def load_a3m_file(file_name: str):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    description_line_indices = [i for i, l in enumerate(lines) if l.startswith('>')]

    seqs = [lines[i+1].strip() for i in description_line_indices]

    return seqs

def map_seq_to_inds(seq, seq_type):
    id_map = id_maps[seq_type]
    sequence_inds = torch.tensor([id_map[a] for a in seq])
    return sequence_inds

def initial_data_from_seqs(seqs, seq_type):
    deletion_count_matrix = []
    unique_seqs = []
    for seq in seqs:
        deletion_count_list = []
        deletion_counter = 0
        for letter in seq:
            if letter.islower():
                deletion_counter += 1
            else:
                deletion_count_list.append(deletion_counter)
                deletion_counter = 0
        
        seq_without_deletion = re.sub('[a-z]', '', seq)

        if seq_without_deletion in unique_seqs:
            continue

        unique_seqs.append(seq_without_deletion)
        deletion_count_matrix.append(deletion_count_list)

    unique_seqs = torch.stack([map_seq_to_inds(seq, seq_type) for seq in unique_seqs], dim=0).long()
    
    if seq_type != 'ligand':
        restype = unique_seqs[0]
    else:
        restype = map_seq_to_inds(seq.replace('-', 'X'), seq_type)

    unique_seqs_onehot = F.one_hot(unique_seqs, num_classes=31)
    deletion_count_matrix = torch.tensor(deletion_count_matrix).float()
    res_distribution = unique_seqs_onehot.float().mean(dim=0)

    deletion_mean = deletion_count_matrix.mean(dim=0)
    has_deletion = torch.clip(deletion_count_matrix, min=0, max=1)
    deletion_value = (2/torch.pi) * torch.arctan(deletion_count_matrix/3)

    return { 'restype': restype, 'msa': unique_seqs, 'deletion_value': deletion_value, 'profile': res_distribution, 'deletion_mean': deletion_mean, 'has_deletion': has_deletion }

def process_msa_file(filename, seq_type):
    seqs = load_a3m_file(filename)
    data = initial_data_from_seqs(seqs, seq_type)
    return data

def empty_msa(seq, seq_type):
    return initial_data_from_seqs([seq], seq_type)
    




def deduplicate_unpaired(unpaired_msa, paired_msa):
    hashes = set(hash(a.tobytes()) for a in paired_msa['msa'].numpy().astype(np.int8))
    inds_to_keep = []
    for i, row in enumerate(unpaired_msa['msa'].numpy().astype(np.int8)):
        if not hash(row.tobytes()) in hashes:
            inds_to_keep.append(i)

    for feat in ['msa', 'deletion_value', 'has_deletion']:
        unpaired_msa[feat] = unpaired_msa[feat][inds_to_keep]

    return unpaired_msa

def merge_unpaired_paired(unpaired_msa, paired_msa, max_row_count, deduplicate=True):
    if deduplicate:
        unpaired_msa = deduplicate_unpaired(unpaired_msa, paired_msa)

    unpaired_row_count = unpaired_msa['msa'].shape[0]
    paired_row_count = paired_msa['msa'].shape[0]
    max_paired_size = min(max_row_count//2, paired_row_count)
    max_unpaired_size = min(max_row_count - max_paired_size, unpaired_row_count)

    merged_features = {**unpaired_msa}

    for feat in ['msa', 'deletion_value', 'has_deletion']:
        unpaired_feat = unpaired_msa[feat][:max_unpaired_size, ...]
        paired_feat = paired_msa[feat][:max_paired_size, ...]
        merged_features[feat] = torch.cat((paired_feat, unpaired_feat), dim=0)

    return merged_features


def merge_msa_features(msa_data_list, padded_row_count=None, padded_col_count=None):
    pad_values = {
        'msa': id_maps['protein']['-'],
        'deletion_value': 0,
        'has_deletion': 0,
    }

    row_dim_feats = ['msa', 'deletion_value', 'has_deletion']
    non_row_feats = ['profile', 'deletion_mean', 'restype']
    all_feats = row_dim_feats + non_row_feats

    max_rows = max(msa_data['msa'].shape[0] for msa_data in msa_data_list)
    token_count = sum(msa_data['profile'].shape[0] for msa_data in msa_data_list)

    if padded_row_count is None:
        padded_row_count = max_rows
    if padded_col_count is None:
        padded_col_count = token_count

    for msa_data in msa_data_list:
        for key in row_dim_feats:
            padded_shape = (max_rows,) + msa_data[key].shape[1:]
            msa_data[key] = utils.crop_pad_to_shape(msa_data[key], padded_shape, pad_values[key])

    joined_data = dict()

    for key in all_feats:
        if key in row_dim_feats:
            joined_data[key] = torch.cat([msa_data[key] for msa_data in msa_data_list], dim=1)
        else:
            joined_data[key] = torch.cat([msa_data[key] for msa_data in msa_data_list], dim=0)

    joined_data['msa_mask'] = torch.ones(joined_data['msa'].shape)
    row_dim_feats.append('msa_mask')

    for key in joined_data.keys():
        if key in row_dim_feats:
            padded_shape = (padded_row_count, padded_col_count) + joined_data[key].shape[2:]
        else:
            padded_shape = (padded_col_count,) + joined_data[key].shape[1:]

        joined_data[key] = utils.pad_to_shape(joined_data[key], padded_shape, value=0)


    return joined_data

def calculate_msa_feat(msa_features):
    msa_one_hot = F.one_hot(msa_features['msa'], num_classes=32)
    deletion_value = msa_features['deletion_value'][..., None]
    has_deletion = msa_features['has_deletion'][..., None]

    msa_feat = [msa_one_hot, has_deletion, deletion_value]
    msa_feat = torch.cat(msa_feat, dim=-1)

    return msa_feat

def sample_n_msa_feats(base_msa_feat, base_msa_mask, n, msa_trunc_count=1024, msa_shuffle_orders = None):
        msa_feats = []
        msa_masks = []

        batch_shape = base_msa_feat.shape[:-3]

        for i in range(n):
            if msa_shuffle_orders is not None:
                msa_shuffle_order = msa_shuffle_orders[i]
            else:
                has_entries = torch.clip(msa_mask.sum(dim=-1), 0, 1)
                odds = (has_entries - 1) * -1e2
                scores = odds + torch.distributions.Gumbel(0, 1).sample(odds.shape)
                msa_shuffle_order = torch.argsort(scores, dim=-1)

            msa_indices = msa_shuffle_order[..., :msa_trunc_count]
            msa_feat = utils.batched_gather(base_msa_feat, msa_indices, batch_shape)
            msa_mask = utils.batched_gather(base_msa_mask, msa_indices, batch_shape)

            msa_feats.append(msa_feat)
            msa_masks.append(msa_mask)

        msa_feat = torch.stack(msa_feats, dim=-1)
        msa_mask = torch.stack(msa_masks, dim=-1)

        return msa_feat, msa_mask

def calculate_target_feat(msa_features):
        restype = msa_features['restype']
        restype_1h = F.one_hot(restype, 31)
        profile = msa_features['profile']
        deletion_mean = msa_features['deletion_mean'][..., None]

        target_feat = torch.cat((restype_1h, profile, deletion_mean), dim=-1)
        return target_feat


