# AlphaFold3-PyTorch

A PyTorch implementation of AlphaFold3 compatible with the official weights from DeepMind.

## Overview

This repository provides a simplified PyTorch implementation of AlphaFold3, designed to be compatible with DeepMind's official weights. The primary goal is to offer a clear, understandable codebase that makes it easy for researchers and developers to explore and build upon the AlphaFold3 architecture.

Similar to the [AlphaFold-Decoded](https://github.com/kilianmandon/alphafold-decoded) project, this implementation aims to guide users through understanding the full AlphaFold3 implementation. I'm aiming to migrate this code to an alphafold3-decoded vesion as well, that guides you through doing the implementation yourself in an exercise-like fashion.

## Important Note Regarding Weights

This repository **does not** include the AlphaFold3 model weights. Users must obtain the weights directly from DeepMind according to their policies. Using these weights with this PyTorch implementation does not exempt users from DeepMind's terms of use.

**In particular:**
- Do not distribute or leak the weights under any circumstances
- Do not use the weights for commercial purposes
- Follow all terms specified by DeepMind for the use of AlphaFold3

## Installation

### Prerequisites

1. Clone this repository:
   `git clone https://github.com/kilianmandon/alphafold3-pytorch.git`

2. Create and activate a conda environment from the provided requirements file:
    ```bash
    $ conda env create -f environment.yml
    $ conda activate alphafold3-pytorch`
    ```

### Getting the Model Weights

1. Obtain the AlphaFold3 parameters from DeepMind. These are available for academic use. The steps of the application process are described here: https://github.com/google-deepmind/alphafold3

2. Place the obtained weights in `data/params/`

### Setting Up the Chemical Component Dictionary

Download the CCD:

```bash
$ wget -P data/ccd https://files.wwpdb.org/pub/pdb/data/monomers/components.cif`
```

The first run of the model will take ~5 minutes for the conversion of the `components.cif` file to pickle.

## Converting Weights

Before using the model, you need to convert the JAX weights to PyTorch format:
```bash
$ python remap_weights.py
```


## Usage

For an example of how to use the model, refer to `run_inference.py`:


```python
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
```


## Citation

If you use this implementation in your research, please cite the original AlphaFold3 paper:

```
@article{Abramson2024-fj,
  title    = "Accurate structure prediction of biomolecular interactions with
              {AlphaFold} 3",
  author   = "Abramson, Josh and Adler, Jonas and Dunger, Jack and Evans,
              Richard and Green, Tim and Pritzel, Alexander and Ronneberger,
              Olaf and Willmore, Lindsay and Ballard, Andrew J and Bambrick,
              Joshua and Bodenstein, Sebastian W and Evans, David A and Hung,
              Chia-Chun and O'Neill, Michael and Reiman, David and
              Tunyasuvunakool, Kathryn and Wu, Zachary and {\v Z}emgulyt{\.e},
              Akvil{\.e} and Arvaniti, Eirini and Beattie, Charles and
              Bertolli, Ottavia and Bridgland, Alex and Cherepanov, Alexey and
              Congreve, Miles and Cowen-Rivers, Alexander I and Cowie, Andrew
              and Figurnov, Michael and Fuchs, Fabian B and Gladman, Hannah and
              Jain, Rishub and Khan, Yousuf A and Low, Caroline M R and Perlin,
              Kuba and Potapenko, Anna and Savy, Pascal and Singh, Sukhdeep and
              Stecula, Adrian and Thillaisundaram, Ashok and Tong, Catherine
              and Yakneen, Sergei and Zhong, Ellen D and Zielinski, Michal and
              {\v Z}{\'\i}dek, Augustin and Bapst, Victor and Kohli, Pushmeet
              and Jaderberg, Max and Hassabis, Demis and Jumper, John M",
  journal  = "Nature",
  month    = "May",
  year     =  2024
}
```

## Acknowledgements

This implementation builds on the work of DeepMind's AlphaFold3 team. I am grateful for their contributions to the field of protein structure prediction.