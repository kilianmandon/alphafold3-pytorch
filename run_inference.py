
import time
import torch
from feature_extraction.feature_extraction import Batch, custom_af3_pipeline, load_input, tree_map
from model import Model
from atomworks.io.utils.io_utils import to_cif_file


def main():
    model = Model(N_cycle=11)
    params = torch.load('data/params/af3_pytorch.pt')
    model.load_state_dict(params)

    pipeline = custom_af3_pipeline(n_recycling_iterations=11)
    data = load_input('data/fold_inputs/fold_input_protein_rna_ion.json')

    print('Starting featurization...')
    t = time.time()
    data = pipeline(data)
    batch = Batch(data['token_features'], data['msa_features'], data['ref_struct'], data['contact_matrix'])
    print(f'Finished featurization, took {time.time()-t} seconds.')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device=device)
    batch = tree_map(lambda x: torch.tensor(x, device=device), batch)

    print('Starting inference...')
    t = time.time()
    x_out = model(batch).cpu().numpy()
    print(f'Finished inference, took {time.time()-t} seconds.')

    atom_array = data['atom_array']
    atom_mask = batch.ref_struct.mask.cpu().numpy()
    atom_array.coord = x_out[atom_mask]
    to_cif_file(atom_array, 'test_structure.cif')    
    

if __name__=='__main__':
    with torch.no_grad():
        main()