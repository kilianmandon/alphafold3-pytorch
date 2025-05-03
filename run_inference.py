import torch
from ccd import load_ccd
from feature_extraction import Input
from model import Model
import utils


def main():
    model = Model()
    params = torch.load('data/params/af3_pytorch.pt', weights_only=False)
    model.load_state_dict(params)

    inp = Input.load_input('data/fold_input_lysozyme.json')
    batch = inp.create_batch()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    batch = utils.move_to_device(batch, device=device)
    model.to(device=device)
    model.eval()
    with torch.no_grad():
        token_positions, token_mask = model(batch)
    
    token_positions = token_positions.to(device='cpu')
    token_mask = token_mask.to(device='cpu')

    ccd = load_ccd()
    mmcif_string = utils.to_modelcif(token_positions, token_mask, inp, ccd)

    with open('data/lysozyme_model.cif', 'w') as f:
        f.write(mmcif_string)