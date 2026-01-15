import time
import torch
from feature_extraction.feature_extraction import custom_af3_pipeline, load_input
import utils
from model import Model
import tensortrace as ttr




def check_batch(batch, true_batch):
    ref_struct = batch['ref_struct']
    true_ref_struct = true_batch['ref_struct']
    msa_features = batch['msa_features']

    # Not saved in batch, only deletion_value, so not checked
    true_batch['msa_features'].pop('deletion_matrix')

    batch_matching_names = { 
        'msa_features': {
            'msa_aatype': msa_features['restype'],
            'rows': msa_features['msa'],
            # 'deletion_matrix': msa_features['deletion_value'],
            'profile': msa_features['profile'],
            'deletion_mean': msa_features['deletion_mean'],
            'target_feat': msa_features['target_feat'],
            'msa_feat': msa_features['msa_feat'],
            'msa_mask': msa_features['msa_mask'],
        },
        'token_features': batch['token_features'], 
        'ref_struct': {
          'positions': ref_struct['ref_pos'],
          'mask': ref_struct['ref_mask'],
          'element': ref_struct['ref_element'],
          'charge': ref_struct['ref_charge'],
          'atom_name_chars': ref_struct['ref_atom_name_chars'],
          'ref_space_uid': ref_struct['ref_space_uid'],
        },
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
    evo_embeddings = utils.move_to_device(evo_embeddings, device='cpu')
    true_evo_embeddings = utils.move_to_device(true_evo_embeddings, device='cpu')
    single_mask = utils.move_to_device(single_mask, 'cpu')

    s_input, s_trunk, z_trunk, rel_enc = evo_embeddings

    true_evo_embeddings = {
        'pair': true_evo_embeddings['pair'][-1],
        'single': true_evo_embeddings['single'][-1],
        # 'target_feat': true_evo_embeddings['target_feat'][-1],
        'rel_enc': true_evo_embeddings['rel_enc'],
    }

    evo_embeddings = {
        'pair': z_trunk,
        'single': s_trunk,
        # 'target_feat': s_input,
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
    diff_x = utils.move_to_device(diff_x, 'cpu')
    true_diff_x = utils.move_to_device(true_diff_x, 'cpu')
    true_diff_x = true_diff_x[-1]
    mask = utils.move_to_device(mask, 'cpu')
    mask = broadcast_mask(mask, diff_x)
    
    diff_x = diff_x * mask
    true_diff_x = true_diff_x * mask

    di, da, dr = diff(diff_x, true_diff_x)
    print('Running tests for diffusion...')
    if di.numel() > 0:
        print(f'Problems with diffusion.')
        pass

    print('Done!')

def reorder_encoding(dim=-1, offset=0):
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
    def f(tensor):
        new_shape = tensor.shape
        new_shape[dim] -= 1
        new_tensor = torch.zeros(new_shape, device=tensor.device,  dtype=tensor.dtype)
        new_tensor[:, offset+31:] = tensor[:, offset+32:]
        for i_old, i_new in token_enc_shift.items():
            new_tensor[:, offset+i_old] = tensor[:, offset+i_new]
        return new_tensor
    
    return f


def main():
    msa_shuffle_order = torch.stack(ttr.load_all('evoformer/msa_shuffle_order'), dim=0)

    model = Model(N_cycle=2, noise_steps=4)
    params = torch.load('data/params/af3_pytorch.pt')
    model.load_state_dict(params)

    t1 = time.time()
    data = load_input('data/fold_inputs/fold_input_lysozyme.json')
    transform = custom_af3_pipeline(n_recycling_iterations=2, msa_shuffle_orders=msa_shuffle_order)

    batch = transform.forward(data)
    batch['ref_struct']['ref_pos'] = ttr.load('ref_structure/positions')

    print(f'Featurization took {time.time()-t1:.1f} seconds.')

    ttr.compare({
        'mask': batch['ref_struct']['ref_mask'],
        'charge': batch['ref_struct']['ref_charge'],
        'element': batch['ref_struct']['ref_element'],
        'atom_name_chars': batch['ref_struct']['ref_atom_name_chars'],
        'ref_space_uid': batch['ref_struct']['ref_space_uid'],
    }, 'ref_structure', use_mask={'mask': False})
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    batch = utils.move_to_device(batch, device)

    batch['ref_struct']['atom_layout']

    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    evo_embeddings = model.evoformer(batch)
    # s_input, s_trunk, z_trunk, rel_enc
    ttr.compare(
        {
            'pair': evo_embeddings[2],
            'single': evo_embeddings[1],
            'target_feat': evo_embeddings[0],
        },
        'evoformer/embeddings',
        input_processing={'target_feat': [reorder_encoding(offset=32), reorder_encoding(offset=0)]}
    )

    evo_embeddings = utils.move_to_device([
        evo_embeddings[0],
        ttr.load('evoformer/embeddings/single'),
        ttr.load('evoformber/embeddings/pair'),
        evo_embeddings[3],
    ], device)

    atom_layout = batch['ref_struct']['atom_layout']

    def t2q(tensor, mask=None):
        n_feat_dims = tensor.ndim - 2
        return batch['ref_struct']['atom_layout'].tokens_to_queries(tensor, n_feat_dims=n_feat_dims)

    def indexing(*args):
        def apply_index(tensor):
            return tensor[*args]
        return apply_index
    
    def to_device(tensor):
        return tensor.to(device)


    diffusion_randomness = {
        'init_pos': ttr.load('diffusion/initial_positions', processing=[indexing(0), t2q, to_device]),
        'noise': ttr.load_all('diffusion/noise', processing=[t2q, to_device]),
        'aug_rot': ttr.load_all('diffusion/rand_aug/rot', processing=to_device),
        'aug_trans': ttr.load_all('diffusion/rand_aug/trans', processing=to_device),
    }

    s_input, s_trunk, z_trunk, rel_enc = evo_embeddings
    diff_x = model.diffusion_sampler(model.diffusion_module,
                               s_input, s_trunk, z_trunk, rel_enc, 
                               batch['ref_struct'], batch['token_features']['single_mask'], noise_data=diffusion_randomness)

    diff_x = atom_layout.queries_to_tokens(diff_x, n_feat_dims=1)

    ttr.compare(diff_x, 'diffusion/final_positions')
    # true_diff_x = torch.load('tests/test_lysozyme/test_outputs/diffusion_positions.pt')
    # check_diffusion(diff_x, true_diff_x, batch['ref_struct']['ref_mask'])


if __name__=='__main__':
    with torch.no_grad(), ttr.TensorTrace('lysozyme_trace', mode='read', framework='pytorch'):
        main()