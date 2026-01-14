import pickle

import numpy as np
import torch
from atomworks.constants import STANDARD_AA, STANDARD_RNA, STANDARD_DNA, UNKNOWN_RNA
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from atomworks.ml.transforms.atom_array import AddWithinPolyResIdxAnnotation, AddWithinChainInstanceResIdx
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.encoding import EncodeAF3TokenLevelFeatures
from atomworks.ml.transforms.msa.msa import LoadPolymerMSAs, PairAndMergePolymerMSAs, EncodeMSA, FillFullMSAFromEncoded
from atomworks.ml.utils.token import get_token_count, get_token_starts

from feature_extraction.token_features import round_to_bucket
import utils
from atom_layout import AtomLayout
from evoformer import Evoformer
from feature_extraction.feature_extraction import load_input, custom_af3_pipeline
from model import Model
from residue_constants import _PROTEIN_TO_ID, AF3_TOKENS_MAP, AF3_TOKENS
from tests import diff


def test_input_embeddings():
    data = load_input('data/fold_inputs/fold_input_lysozyme.json')
    msa_shuffle_order = torch.load('tests/test_lysozyme/debug_inputs/msa_shuffle_order.pt').long()
    transform = custom_af3_pipeline(msa_shuffle_orders=msa_shuffle_order, n_recycling_iterations=2)
    batch = transform.forward(data)
    exp_ref_struct = torch.load('tests/test_lysozyme/ref_structure.pt', weights_only=False)
    batch['ref_struct']['ref_pos'] = torch.tensor(exp_ref_struct['positions'])

    for k1 in ['msa_features', 'token_features', 'ref_struct']:
        for k2, v2 in batch[k1].items():
            if isinstance(v2, torch.Tensor) and v2.dtype == torch.float32:
                batch[k1][k2] = v2.float()

    exp_msa_emb = torch.load(
        'tests/test_lysozyme/msa_activations_0.pt',
        weights_only=False)
    exp_target_feat_emb = torch.load(
        'tests/test_lysozyme/target_feat_activations_0.pt',
        weights_only=False)[0]

    model = Model(N_cycle=2, noise_steps=4)
    params = torch.load('data/params/af3_pytorch.pt')
    model.load_state_dict(params)

    msa_emb = model.evoformer.msa_module.linear_m(batch['msa_features']['msa_feat'][..., 0])
    token_act, _ = model.evoformer.input_embedder.atom_cross_att(batch['ref_struct'])
    s_input = torch.cat((batch['msa_features']['target_feat'], token_act), dim=-1)
    target_feat_concat_exp = torch.load('tests/test_lysozyme/target_feat_concatted.pt', weights_only=False)
    target_feat_emb = model.evoformer.msa_module.linear_s(s_input)
    mask = batch['token_features']['single_mask'][..., None]

    di, da, dr = diff(msa_emb, exp_msa_emb)
    if di.numel() > 0:
        print('MSA embedding issue')
        ...

    print(f'MSA max err: {da.max()}')

    di, da, dr = diff(target_feat_emb, exp_target_feat_emb)
    if di.numel() > 0:
        print('Target feat embedding issue')
        ...

    print(f'TF max err: {da.max()}')

    print('All done.')

def test_full_pass():
    data = load_input('data/fold_inputs/fold_input_lysozyme.json')
    msa_shuffle_order = torch.load('tests/test_lysozyme/debug_inputs/msa_shuffle_order.pt').long()
    transform = custom_af3_pipeline(msa_shuffle_orders=msa_shuffle_order, n_recycling_iterations=2)
    batch = transform.forward(data)
    exp_ref_struct = torch.load('tests/test_lysozyme/ref_structure.pt', weights_only=False)
    batch['ref_struct']['ref_pos'] = torch.tensor(exp_ref_struct['positions'])

    for k1 in ['msa_features', 'token_features', 'ref_struct']:
        for k2, v2 in batch[k1].items():
            if isinstance(v2, torch.Tensor) and v2.dtype == torch.float32:
                batch[k1][k2] = v2.float()


    exp_evo_embs = torch.load('tests/test_lysozyme/test_outputs/evoformer_embeddings.pt', weights_only=False)
    atom_layout = batch['ref_struct']['atom_layout']
    diff_noise = torch.load('tests/test_lysozyme/debug_inputs/diffusion_noise.pt', weights_only=False)
    noise_data = {
        'init_pos':
            atom_layout.tokens_to_queries(
            torch.load('tests/test_lysozyme/debug_inputs/diffusion_initial_positions.pt', weights_only=False)[0],
            n_feat_dims=1),
        'aug_rot': 
            torch.load(f'tests/test_lysozyme/debug_inputs/diffusion_randaug_rot.pt', weights_only=False),
        'aug_trans': torch.load(
            f'tests/test_lysozyme/debug_inputs/diffusion_randaug_trans.pt',
            weights_only=False),
        'noise': [
            atom_layout.tokens_to_queries(
                diff_noise[i],
                n_feat_dims=1)
            for i in range(4)
        ]
    }

    exp_final_positions = (
        atom_layout.tokens_to_queries(
            torch.load('tests/test_lysozyme/test_outputs/diffusion_final_positions.pt', weights_only=False)[0],
            n_feat_dims=1)
    )

    model = Model(N_cycle=2, noise_steps=4)
    params = torch.load('data/params/af3_pytorch.pt')
    model.load_state_dict(params)
    model.to('cuda')

    model.eval()

    ref_struct = batch['ref_struct']
    single_mask = batch['token_features']['single_mask']

    s_input_exp = exp_evo_embs['target_feat'][-1]
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
    new_shape = (s_input_exp.shape[0], s_input_exp.shape[1] + 2)
    s_input = torch.zeros(new_shape)
    s_input[:, 64:] = s_input_exp[:, 62:]
    for i_old, i_new in token_enc_shift.items():
        s_input[:, i_new] = s_input_exp[:, i_old]
        s_input[:, i_new + 32] = s_input_exp[:, i_old + 31]
    # s_trunk = exp_evo_embs['single'][-1]
    # z_trunk = exp_evo_embs['pair'][-1]
    # rel_enc = exp_evo_embs['rel_enc']

    batch = utils.move_to_device(batch, 'cuda')
    model.to('cuda')

    s_input, s_trunk, z_trunk, rel_enc = model.evoformer(batch)
    s_input, s_trunk, z_trunk, rel_enc = s_input.to('cpu'), s_trunk.to('cpu'), z_trunk.to('cpu'), rel_enc.to('cpu')
    # with open('embeddings.pkl', 'rb') as f:
        # s_input, s_trunk, z_trunk, rel_enc = pickle.load(f)

    # z_trunk = exp_evo_embs['pair'][-1]
    # s_trunk = exp_evo_embs['single'][-1]

    di, da, dr = diff(s_trunk, exp_evo_embs['single'][-1], mask=single_mask[..., None])
    if di.numel() > 0:
        max_err = da.max()
        max_rel_err = dr.max()
        ...

    di, da, dr = diff(z_trunk, exp_evo_embs['pair'][-1],
                      mask=single_mask[..., None, None] * single_mask[..., None, :, None])
    if di.numel() > 0:
        max_err = da.max()
        max_rel_err = dr.max()
        print('Problems with evoformer')
        ...

    s_input, s_trunk, z_trunk, rel_enc = s_input.to('cuda'), s_trunk.to('cuda'), z_trunk.to('cuda'), rel_enc.to('cuda')
    noise_data = utils.move_to_device(noise_data, 'cuda')

    tf_mask = torch.ones(s_input.shape[-1], device=s_input.device, dtype=bool)
    tf_mask[415] = tf_mask[447] = False
    s_input = s_input[:, tf_mask]

    x_flat = model.diffusion_sampler(model.diffusion_module,
                                    s_input, s_trunk, z_trunk, rel_enc,
                                    batch['ref_struct'], batch['token_features']['single_mask'], noise_data=noise_data)

    x_flat = x_flat.cpu()
    flat_mask = atom_layout.tokens_to_queries(batch['ref_struct']['ref_mask'], n_feat_dims=0).cpu()

    di, da, dr = diff(x_flat, exp_final_positions, mask=flat_mask[..., None])
    if di.numel() > 0:
        print('Problems with diffusion')
        ...


    # atom_layout = ref_struct['atom_layout']
    # token_mask = ref_struct['ref_mask']
    # x_out = atom_layout.queries_to_tokens(x_flat, n_feat_dims=1)
    atom_array = batch['atom_array']
    x_flat = x_flat.reshape(-1, 3).numpy()
    flat_mask = flat_mask.reshape(-1).bool().numpy()
    atom_array.coord = x_flat[flat_mask]
    token_starts = get_token_starts(atom_array)
    to_cif_file(atom_array, 'test_structure.cif')


def real_full_pass():
    data = load_input('data/fold_inputs/fold_input_protein_rna_ion.json')
    transform = custom_af3_pipeline(n_recycling_iterations=11)
    batch = transform.forward(data)
    for k1 in ['msa_features', 'token_features', 'ref_struct']:
        for k2, v2 in batch[k1].items():
            if isinstance(v2, torch.Tensor) and v2.dtype == torch.float32:
                batch[k1][k2] = v2.float()

    device = torch.device('cuda')
    batch = utils.move_to_device(batch, device)


    model = Model()
    params = torch.load('data/params/af3_pytorch.pt')
    model.load_state_dict(params)
    model.to(device)

    model.eval()

    ref_struct = batch['ref_struct']
    single_mask = batch['token_features']['single_mask']

    x_out, token_mask = model(batch)
    atom_layout = ref_struct['atom_layout']

    x_flat = atom_layout.tokens_to_queries(x_out, n_feat_dims=1).reshape(-1, 3).cpu().numpy()
    flat_mask = atom_layout.tokens_to_queries(token_mask, n_feat_dims=0).reshape(-1).bool().cpu().numpy()

    atom_array = batch['atom_array']
    atom_array.coord = x_flat[flat_mask]
    token_starts = get_token_starts(atom_array)
    to_cif_file(atom_array, 'test_structure.cif')
    print('All done.')



def test_featurization():
    data = load_input('data/fold_inputs/fold_input_lysozyme.json')
    msa_shuffle_order = torch.load('tests/test_lysozyme/debug_inputs/msa_shuffle_order.pt').long()
    transform = custom_af3_pipeline(msa_shuffle_orders=msa_shuffle_order, n_recycling_iterations=2)

    out = transform.forward(data)
    exp_msa_features = torch.load('tests/test_lysozyme/msa.pt', weights_only=False)

    # swap 23<->24, 28<->29, shift 30->29->28->27->26->30
    rna_enc_shift = {
                        i: i for i in range(31)
                    } | {
                        23: 24,
                        24: 23,
                        26:27,
                        27:29,
                        28:28,
                        29:30,
                        30:26,
                    }

    exp_profile = exp_msa_features['profile']
    patched_profile = np.zeros((exp_profile.shape[0], 32))
    for k,v in rna_enc_shift.items():
        patched_profile[:, v] = exp_profile[:, k]
    exp_msa_features['profile'] = patched_profile



    exp_ref_struct = torch.load('tests/test_lysozyme/ref_structure.pt', weights_only=False)
    exp_token_feats = torch.load('tests/test_lysozyme/token_feats.pt', weights_only=False)

    exp_token_feats['aatype'] = np.vectorize(lambda a: rna_enc_shift[a])(exp_token_feats['aatype'])
    exp_msa_features['rows'] = np.vectorize(lambda a: rna_enc_shift[a])(exp_msa_features['rows'])

    # Densify AF3 RefStruct
    order = np.argsort(-exp_ref_struct['mask'].astype(int), axis=1, stable=True)
    for k in exp_ref_struct:
        exp_ref_struct[k] = utils.batched_gather(torch.tensor(exp_ref_struct[k]), torch.tensor(order), batch_shape=exp_ref_struct[k].shape[:1]).numpy()


    for k1, k2 in zip(['restype', 'residue_index', 'token_index', 'asym_id', 'entity_id', 'sym_id'], ['aatype', 'residue_index', 'token_index', 'asym_id', 'entity_id', 'sym_id']):
        v1 = out['token_features'][k1]
        v2 = exp_token_feats[k2]
        di, da, dr = diff(v1, v2)
        if di.numel() > 0:
            print('Token issue!')

    for k1, k2 in zip(['full_msa_mask', 'msa', 'deletion_count', 'profile'], ['mask', 'rows', 'deletion_matrix', 'profile']):
        v1 = out['msa_features'][k1]
        v2 = exp_msa_features[k2]
        if k1 not in ['full_msa_mask', 'profile']:
            mask = exp_msa_features['mask']
            for _ in range(len(v1.shape)-2): mask=mask[..., None]
        else:
            mask = None

        di, da, dr = diff(v1, v2, mask=mask)

        if di.numel() > 0 or k1=='profile':
            print('MSA issue!')

    N_tokens = round_to_bucket(get_token_count(out['atom_array']))
    atom_layout = AtomLayout.from_atom_array(out['atom_array'], N_tokens)
    N_blocks = N_tokens * 24 // 32

    for k1, k2 in zip(['ref_mask', 'ref_element', 'ref_charge', 'ref_atom_name_chars', 'ref_space_uid', 'ref_pos'], ['mask', 'element', 'charge', 'atom_name_chars', 'ref_space_uid', 'positions']):
        v1 = out['ref_struct'][k1]
        n_feat_dims = len(v1.shape) - 2
        v2 = exp_ref_struct[k2]
        if k1 != 'ref_mask':
            mask = exp_ref_struct['mask']
            for _ in range(n_feat_dims): mask=mask[..., None]
        else:
            mask = None
        di, da, dr = diff(v1, v2, mask=mask)
        print(k1)
        print(da.max())
        if di.numel() > 0:
            print('Refstruct issue!')



    ...



def index_to_aa(row):
    inv_lookup = {v:k for k,v in _PROTEIN_TO_ID.items()}
    inv_lookup[6] = 'E'
    inv_lookup[3] = 'D'
    inv_lookup[4] = 'C'
    inv_lookup[20] = 'U'
    return ''.join([inv_lookup[a] for a in row])



def test():
    data = load_input('data/fold_inputs/debug_fold_input.json')
    af3_sequence_encoding = AF3SequenceEncoding()
    transform = Compose(
        [
            LoadPolymerMSAs(),
            EncodeMSA(
                encoding=af3_sequence_encoding,
                token_to_use_for_gap=af3_sequence_encoding.token_to_idx["<G>"],
            ),
        ])

    out = transform(data)
    ...



if __name__ == '__main__':
    with torch.no_grad():
        test_full_pass()
    # test()