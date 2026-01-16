import time
import torch
from feature_extraction.feature_extraction import custom_af3_pipeline, load_input
import utils
from model import Model
import tensortrace as ttr


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
        new_shape = list(tensor.shape)
        new_shape[dim] -= 1
        new_tensor = torch.zeros(new_shape, device=tensor.device,  dtype=tensor.dtype)
        new_tensor[:, :offset] = tensor[:, :offset]
        new_tensor[:, offset+31:] = tensor[:, offset+32:]
        for i_old, i_new in token_enc_shift.items():
            new_tensor[:, offset+i_old] = tensor[:, offset+i_new]
        return new_tensor
    
    return f

def to_float(tensor):
    return tensor.float()


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
    }, 'ref_structure', use_mask={'mask': False, 'ref_space_uid': False })

    token_feats = { k: batch['token_features'][k] for k in ['asym_id', 'sym_id', 'entity_id', 'is_dna', 'is_rna', 'is_protein', 'residue_index', 'restype']} 
    token_feats |= {
        'mask': batch['token_features']['single_mask']
    }
    ttr.compare(token_feats, 'token_features', use_mask={'mask': False})

    msa_feats = {
        'msa': batch['msa_features']['msa'],
        'profile': batch['msa_features']['profile'],
        'deletion_count': batch['msa_features']['deletion_count'],
        'mask': batch['msa_features']['full_msa_mask'],
    }

    ttr.compare(msa_feats, 'msa_features', use_mask={'mask': False}, input_processing={'profile': reorder_encoding(offset=0)})
    ttr.compare(batch['msa_features']['target_feat'], 'input_embedding/target_feat', input_processing=[reorder_encoding(offset=32), reorder_encoding(offset=0)])
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # device = torch.device('cpu')
    batch = utils.move_to_device(batch, device)

    batch['ref_struct']['atom_layout']

    model = model.to(device=device)
    model.eval()

    evo_embeddings = model.evoformer(batch)
    # s_input, s_trunk, z_trunk, rel_enc
    ttr.compare(evo_embeddings[1], 'evoformer/single')
    ttr.compare(evo_embeddings[2], 'evoformer/pair')

    # evo_embeddings = utils.move_to_device([
    #     evo_embeddings[0],
    #     ttr.load('evoformer/embeddings/single'),
    #     ttr.load('evoformber/embeddings/pair'),
    #     evo_embeddings[3],
    # ], device)

    atom_layout = batch['ref_struct']['atom_layout']

    def t2q(tensor, mask=None):
        n_feat_dims = tensor.ndim - 2
        atom_layout = utils.move_to_device(batch['ref_struct']['atom_layout'], 'cpu')
        return atom_layout.tokens_to_queries(tensor, n_feat_dims=n_feat_dims)

    def indexing(*args):
        def apply_index(tensor):
            return tensor[*args]
        return apply_index
    
    def to_device(tensor):
        return tensor.to(device)

    def to_float(tensor):
        return tensor.float()


    diffusion_randomness = {
        'init_pos': ttr.load('diffusion/initial_positions', processing=[indexing(0), t2q, to_device, to_float]),
        'noise': ttr.load_all('diffusion/noise', processing=[indexing(0), t2q, to_device, to_float]),
        'aug_rot': ttr.load_all('diffusion/rand_aug/rot', processing=[indexing(0), to_device, to_float]),
        'aug_trans': ttr.load_all('diffusion/rand_aug/trans', processing=[indexing(0), to_device, to_float]),
    }
    
    batch = utils.move_to_device(batch, device)

    s_input, s_trunk, z_trunk, rel_enc = evo_embeddings
    with ttr.Chapter('diffusion'):
        diff_x = model.diffusion_sampler(model.diffusion_module,
                                s_input, s_trunk, z_trunk, rel_enc, 
                                batch['ref_struct'], batch['token_features']['single_mask'], noise_data=diffusion_randomness)

    diff_x = atom_layout.queries_to_tokens(diff_x, n_feat_dims=1)

    ttr.compare(diff_x, 'diffusion/final_positions', processing=[indexing(0)])
    # true_diff_x = torch.load('tests/test_lysozyme/test_outputs/diffusion_positions.pt')
    # check_diffusion(diff_x, true_diff_x, batch['ref_struct']['ref_mask'])


if __name__=='__main__':
    with torch.no_grad(), ttr.TensorTrace('../alphafold3-pure/lysozyme_trace', mode='read', framework='pytorch'):
        main()