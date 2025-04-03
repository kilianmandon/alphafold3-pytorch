

import torch
from feature_extraction import Input, move_to_device
from model import Model

def diff(v1, v2, mask=None, atol=1e-3, rtol=1e-2):
    if mask is not None:
        v1 = v1 * mask
        v2 = v2 * mask
    da = (v1-v2).abs()
    dr = da / torch.maximum(v1.abs(), v2.abs()).clip(min=1e-10)
    diff_mask = (da > atol) &  (dr > rtol)
    di = torch.nonzero(diff_mask)

    return di, da, dr
    

def broadcast_mask(mask, v):
    mask_inds = set()
    mask_len = len(mask.shape)

    for i in range(len(v.shape) - mask_len + 1):
        if v.shape[i:i+mask_len] == mask.shape:
            mask_inds.add(i)
    
    mask_inds = list(mask_inds)
    if len(mask_inds) == 0:
        raise ValueError(f'Mask shape {mask.shape} not applicable to tensor with shape {v.shape}.')
    elif len(mask_inds) > 1:
        raise ValueError(f'Mask shape {mask.shape} ambiguous for tensor with shape {v.shape}.')

    mask_ind = mask_inds[0]
    mask = mask.reshape((1,) * mask_ind + mask.shape + (1,) * (len(v.shape)-mask_ind-mask_len))
    mask = mask.broadcast_to(v.shape)

    return mask


def check_batch(batch, true_batch):
    ref_struct = batch['ref_struct']
    true_ref_struct = true_batch['ref_struct']

    batch_matching_names = { 
        'msa_features': {
            'msa_aatype': batch['msa_aatype'],
            'rows': batch['rows'],
            'deletion_matrix': batch['deletion_matrix'],
            'profile': batch['profile'],
            'deletion_mean': batch['deletion_mean'],
            'target_feat': batch['target_feat']
        },

        'token_features': {
            'residue_index': batch['residue_index'],
            'token_index': batch['token_index'],
            'asym_id': batch['asym_id'],
            'entity_id': batch['entity_id'],
            'sym_id': batch['sym_id'],
            'single_mask': batch['single_mask'],
        }, 

        'ref_struct': batch['ref_struct']
    }

    batch_masks = { 
        'msa_features': {
        },

        'token_features': {
        }, 

        'ref_struct': {
            'ref_space_uid': true_batch['ref_struct']['mask']
        }
    }

    true_batch['msa_features']['target_feat'] = true_batch['msa_features']['target_feat'][:, :63]

    print('Running tests for input features...')
    for k1 in true_batch:
        for k2 in true_batch[k1]:
            v1 = batch_matching_names[k1][k2]
            v2 = true_batch[k1][k2]
            if k2 in batch_masks[k1]:
                mask = broadcast_mask(batch_masks[k1][k2], v1)
                v1 = v1 * mask
                v2 = v2 * mask
            di, da, dr = diff(v1, v2)
            if di.numel() > 0:
                print(f'Problems with {k2}.')
                pass

    print('Done!')

def check_evoformer(evo_embeddings, true_evo_embeddings, single_mask):
    evo_embeddings = move_to_device(evo_embeddings, device='cpu')
    true_evo_embeddings = move_to_device(true_evo_embeddings, device='cpu')
    single_mask = move_to_device(single_mask, 'cpu')

    s_input, s_trunk, z_trunk, rel_enc = evo_embeddings

    true_evo_embeddings = {
        'pair': true_evo_embeddings['pair'][-1],
        'single': true_evo_embeddings['single'][-1],
        'target_feat': true_evo_embeddings['target_feat'][-1],
        'rel_enc': true_evo_embeddings['rel_enc'],
    }
    evo_embeddings = {
        'pair': z_trunk,
        'single': s_trunk,
        'target_feat': s_input,
        'rel_enc': rel_enc,
    }

    masks = {
        'pair': single_mask[..., None, :] * single_mask[..., :, None],
        'single': single_mask,
        'target_feat': single_mask,
        'rel_enc': single_mask[..., None, :] * single_mask[..., :, None],
    }

    print('Running tests for evoformer...')

    for key in evo_embeddings:
        v1 = evo_embeddings[key]
        v2 = true_evo_embeddings[key]

        mask = broadcast_mask(masks[key], v1)
        v1 = v1 * mask
        v2 = v2 * mask

        di, da, dr = diff(v1, v2)
        if di.numel() > 0:
            print(f'Problems with {key}.')
            pass

    print('Done!')

def check_diffusion(diff_x, true_diff_x, mask):
    diff_x = move_to_device(diff_x, 'cpu')
    true_diff_x = move_to_device(true_diff_x, 'cpu')
    true_diff_x = true_diff_x[-1]
    mask = move_to_device(mask, 'cpu')
    mask = broadcast_mask(mask, diff_x)
    
    diff_x = diff_x * mask
    true_diff_x = true_diff_x * mask

    di, da, dr = diff(diff_x, true_diff_x)
    print('Running tests for diffusion...')
    if di.numel() > 0:
        print(f'Problems with diffusion.')
        pass

    print('Done!')


def main():
    msa_shuffle_order = torch.load('tests/test_lysozyme/debug_inputs/msa_shuffle_order.pt').long()

    model = Model(N_cycle=2, noise_steps=4)
    params = torch.load('data/params/af3_pytorch.pt')
    model.load_state_dict(params)

    inp = Input.load_input('data/fold_input_lysozyme.json')
    batch = inp.create_batch(msa_shuffle_order=msa_shuffle_order)
    true_batch = torch.load('tests/test_lysozyme/test_outputs/batch.pt')
    check_batch(batch, true_batch)
    
    device = torch.device('mps')
    batch = move_to_device(batch, device)
    model = model.to(device=device)
    model.eval()

    evo_embeddings = model.evoformer(batch)
    true_evo_embeddings = torch.load('tests/test_lysozyme/test_outputs/evoformer_embeddings.pt')
    check_evoformer(evo_embeddings, true_evo_embeddings, batch['single_mask'])

    diffusion_init_pos = torch.load('tests/test_lysozyme/debug_inputs/diffusion_initial_pos.pt')
    diffusion_noise = torch.load('tests/test_lysozyme/debug_inputs/diffusion_noise.pt')
    diffusion_randaug_rot = torch.load('tests/test_lysozyme/debug_inputs/diffusion_randaug_rot.pt')
    diffusion_randaug_trans = torch.load('tests/test_lysozyme/debug_inputs/diffusion_randaug_trans.pt')
    diffusion_init_pos = diffusion_init_pos.to(device=device)
    diffusion_noise = diffusion_noise.to(device=device)

    atom_layout = batch['ref_struct']['atom_layout']

    diffusion_randomness = {
        'init_pos': atom_layout.tokens_to_queries(diffusion_init_pos[0], n_feat_dims=1),
        'noise': torch.stack([atom_layout.tokens_to_queries(diffusion_noise[i], n_feat_dims=1) for i in range(diffusion_noise.shape[0])], dim=0),
        'aug_rot': diffusion_randaug_rot,
        'aug_trans': diffusion_randaug_trans,
    }
    diffusion_randomness = move_to_device(diffusion_randomness, device=device)

    s_input, s_trunk, z_trunk, rel_enc = evo_embeddings
    diff_x = model.diffusion_sampler(model.diffusion_module,
                               s_input, s_trunk, z_trunk, rel_enc, 
                               batch['ref_struct'], batch['single_mask'], noise_data=diffusion_randomness)

    diff_x = atom_layout.queries_to_tokens(diff_x, n_feat_dims=1)

    true_diff_x = torch.load('tests/test_lysozyme/test_outputs/diffusion_positions.pt')
    check_diffusion(diff_x, true_diff_x, batch['ref_struct']['mask'])


if __name__=='__main__':
    with torch.no_grad():
        main()