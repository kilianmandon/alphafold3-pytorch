import re
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from residue_constants import _DNA_TO_ID, _PROTEIN_TO_ID, _RNA_TO_ID


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


def onehot_encode_seq(seq, seq_type):
    id_map = id_maps[seq_type]
    sequence_inds = torch.tensor([id_map[a] for a in seq])
    encoding = nn.functional.one_hot(sequence_inds, num_classes=31)

    return encoding

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
        msa_aatype = unique_seqs[0]
    else:
        msa_aatype = map_seq_to_inds(seq.replace('-', 'X'), seq_type)

    unique_seqs_onehot = F.one_hot(unique_seqs, num_classes=31)
    deletion_count_matrix = torch.tensor(deletion_count_matrix).float()
    res_distribution = unique_seqs_onehot.float().mean(dim=0)

    deletion_mean = deletion_count_matrix.mean(dim=0)

    return { 'msa_aatype': msa_aatype, 'rows': unique_seqs, 'deletion_matrix': deletion_count_matrix, 'profile': res_distribution, 'deletion_mean': deletion_mean}
    
def crop_pad_to_shape(data, padded_shape, value=0):
    inds = tuple(slice(min(i, j)) for i, j in zip(data.shape, padded_shape))
    data = data[inds]
    return pad_to_shape(data, padded_shape, value)

def pad_to_shape(data, padded_shape, value=0):
    padded = torch.full(padded_shape, fill_value=value, dtype=data.dtype, device=data.device)
    inds = tuple(slice(i) for i in data.shape)
    padded[inds] = data
    return padded


def deduplicate_unpaired(unpaired_msa, paired_msa):
    hashes = set(hash(a.tobytes()) for a in paired_msa['rows'].numpy().astype(np.int8))
    inds_to_keep = []
    for i, row in enumerate(unpaired_msa['rows'].numpy().astype(np.int8)):
        if not hash(row.tobytes()) in hashes:
            inds_to_keep.append(i)

    for feat in ['rows', 'deletion_matrix']:
        unpaired_msa[feat] = unpaired_msa[feat][inds_to_keep]

    return unpaired_msa

def merge_unpaired_paired(unpaired_msa, paired_msa, max_row_count, deduplicate=True):
    if deduplicate:
        unpaired_msa = deduplicate_unpaired(unpaired_msa, paired_msa)

    unpaired_row_count = unpaired_msa['rows'].shape[0]
    paired_row_count = paired_msa['rows'].shape[0]
    max_paired_size = min(max_row_count//2, paired_row_count)
    max_unpaired_size = min(max_row_count - max_paired_size, unpaired_row_count)

    merged_features = {**unpaired_msa}

    for feat in ['rows', 'deletion_matrix']:
        unpaired_feat = unpaired_msa[feat][:max_unpaired_size, ...]
        paired_feat = paired_msa[feat][:max_paired_size, ...]
        merged_features[feat] = torch.cat((paired_feat, unpaired_feat), dim=0)

    return merged_features


def join_msas(msa_data_list, padded_row_count, padded_col_count):
    padded_msa_data_list = []
    max_rows = max(msa_data['rows'].shape[0] for msa_data in msa_data_list)

    pad_values = {
        'rows': id_maps['protein']['-'],
        'deletion_matrix': 0,
    }

    row_dim_feats = ['rows', 'deletion_matrix']
    non_row_feats = ['profile', 'deletion_mean', 'msa_aatype']

    for msa_data in msa_data_list:
        padded_msa_data = { key: msa_data[key] for key in non_row_feats}
        for key in row_dim_feats:
            feat = msa_data[key]
            cutoff = min(padded_row_count, feat.shape[0])
            padded_shape = (max_rows,) + feat.shape[1:]
            pad_value = pad_values[key]
            feat = feat[:cutoff, ...]
            feat = pad_to_shape(feat, padded_shape, value=pad_value)
            padded_msa_data[key] = feat
        padded_msa_data_list.append(padded_msa_data)

    joined_data = dict()

    for key in row_dim_feats:
        joined_feat = torch.cat([msa_data[key] for msa_data in padded_msa_data_list], dim=1)
        joined_data[key] = joined_feat

    joined_data['msa_mask'] = torch.ones(joined_data['rows'].shape)
    row_dim_feats.append('msa_mask')

    for key in row_dim_feats:
        joined_feat = joined_data[key]
        padded_shape = (padded_row_count, padded_col_count) + joined_feat.shape[2:]
        joined_feat = pad_to_shape(joined_feat, padded_shape, value=0)
        joined_data[key] = joined_feat

    for key in non_row_feats:
        feat = torch.cat([msa_data[key] for msa_data in padded_msa_data_list], dim=0)
        feat = pad_to_shape(feat, (padded_col_count,)+feat.shape[1:], value=0)
        joined_data[key] = feat

    return joined_data


def process_msa_file(filename, seq_type):
    seqs = load_a3m_file(filename)
    data = initial_data_from_seqs(seqs, seq_type)
    return data

def empty_msa(seq, seq_type):
    return initial_data_from_seqs([seq], seq_type)
