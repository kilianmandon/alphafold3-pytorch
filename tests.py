import math
import time
import torch
from atom_layout import AtomLayout
from feature_extraction.feature_extraction import Batch, custom_af3_pipeline, load_input, tree_map
import utils
from model import Model
import tensortrace as ttr
from torch.nn.attention.flex_attention import BlockMask


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

    data = transform.forward(data)
    batch = Batch(data['token_features'], data['msa_features'], data['ref_struct'], data['contact_matrix'])

    batch = tree_map(lambda x: torch.tensor(x), batch)

    batch.ref_struct.positions = batch.ref_struct.to_atom_layout(ttr.load('ref_structure/positions'))

    print(f'Featurization took {time.time()-t1:.1f} seconds.')

    ttr.compare(batch.ref_struct.mask, 're')
    ttr.compare({
        'mask': batch.ref_struct.mask,
        'charge': batch.ref_struct.charge,
        'element': batch.ref_struct.element,
        'atom_name_chars': batch.ref_struct.atom_name_chars,
        'ref_space_uid': batch.ref_struct.ref_space_uid,
    }, 'ref_structure', use_mask={'mask': False }, input_processing=[lambda x: batch.ref_struct.to_token_layout(x)])

    token_feats = {
        'asym_id': batch.token_features.asym_id,
        'sym_id': batch.token_features.sym_id,
        'entity_id': batch.token_features.entity_id,
        'is_dna': batch.token_features.is_dna,
        'is_rna': batch.token_features.is_rna,
        'is_protein': batch.token_features.is_protein,
        'mask': batch.token_features.mask,
    }
    ttr.compare(token_feats, 'token_features', use_mask={'mask': False})

    ttr.compare(batch.msa_features.target_feat, 'input_embedding/target_feat', input_processing=[reorder_encoding(offset=32), reorder_encoding(offset=0)])
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # device = torch.device('cpu')
    batch: Batch = tree_map(lambda x: x.to(device=device), batch)

    model = model.to(device=device)
    model.eval()

    padded_atom_count = batch.ref_struct.atom_count
    atom_count = batch.ref_struct.unpadded_atom_count
    key_inds = AtomLayout.calculate_key_inds(atom_count, padded_atom_count)
    BLOCK_SIZE=32
    num_blocks = padded_atom_count // BLOCK_SIZE

    kv_num_blocks = torch.zeros(num_blocks, dtype=int)
    kv_indices = torch.full((num_blocks, num_blocks), dtype=int, fill_value=int(1e6))


    mask_left_bound = key_inds[:, 0].float().detach()
    mask_right_bound = key_inds[:, -1].float().detach()

    def mask_mod(batch, head, q, k):
        return (q < atom_count) & (mask_left_bound[q//32] <= k) & (k <= mask_right_bound[q//32])

    for i_block in range(num_blocks):
        if i_block*BLOCK_SIZE >= atom_count:
            continue
        lower = key_inds[i_block][0] // BLOCK_SIZE
        upper = key_inds[i_block][-1] // BLOCK_SIZE + 1
        kv_num_blocks[i_block] = upper - lower
        kv_indices[i_block] = torch.arange(num_blocks).roll(-lower.item(), dims=0)

    kv_num_blocks = kv_num_blocks.reshape(1, 1, num_blocks)
    kv_indices = kv_indices.reshape(1, 1, num_blocks, num_blocks)

    batch.ref_struct.block_mask = BlockMask.from_kv_blocks(kv_num_blocks, kv_indices, BLOCK_SIZE=BLOCK_SIZE, mask_mod=mask_mod)

    s_input, s_trunk, z_trunk, rel_enc = model.evoformer(batch)
    ttr.compare(s_trunk, 'evoformer/single')
    ttr.compare(z_trunk, 'evoformer/pair')


    def t2q(tensor):
        return batch.ref_struct.to_atom_layout(tensor)

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
    
    with ttr.Chapter('diffusion'):
        diff_x = model.diffusion_sampler(model.diffusion_module,
                                s_input, s_trunk, z_trunk, rel_enc, 
                                batch.ref_struct, batch.token_features['single_mask'], noise_data=diffusion_randomness)

    diff_x = batch.ref_struct.to_token_layout(diff_x)

    ttr.compare(diff_x, 'diffusion/final_positions', processing=[indexing(0)])


if __name__=='__main__':
    with torch.no_grad(), ttr.TensorTrace('lysozyme_trace', mode='read', framework='pytorch'):
        main()