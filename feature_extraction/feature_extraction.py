from dataclasses import dataclass, fields
import json
from typing import Mapping, Sequence

from atomworks.ml.conditions import AtomArray
import numpy as np
import torch
from atomworks.constants import STANDARD_AA, STANDARD_RNA, STANDARD_DNA
from atomworks.enums import ChainType
from atomworks.io.parser import parse_atom_array
from atomworks.io.tools.inference import components_to_atom_array
from atomworks.io.utils import ccd
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Compose, Transform
from atomworks.ml.transforms.filters import RemoveHydrogens, RemoveTerminalOxygen

from feature_extraction.contact_features import CalculateContactMatrix
from feature_extraction.msa_features import CalculateMSAFeatures, MSAFeatures
from feature_extraction.ref_struct_features import CalculateRefStructFeatures, RefStructFeatures
from feature_extraction.token_features import CalculateTokenFeatures, TokenFeatures

            
Array = np.ndarray | torch.Tensor

def tree_map(fn, x):
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return fn(x)

    if hasattr(x, "map_arrays"):
        return x.map_arrays(lambda y: tree_map(fn, y))

    if isinstance(x, dict):
        return type(x)({k: tree_map(fn, v) for k, v in x.items()})

    if isinstance(x, (tuple, list)):
        return type(x)(tree_map(fn, v) for v in x)

    return x

@dataclass
class Batch:
    token_features: TokenFeatures
    msa_features: MSAFeatures
    ref_struct: RefStructFeatures
    contact_matrix: Array

    def map_arrays(self, fn):
        field_dict = {f.name: fn(getattr(self, f.name)) for f in fields(self)}
        return Batch(**field_dict)



def load_input(path):
    with open(path, 'r') as f:
        data = json.load(f)

    example_id = data['name']
    components = []

    for entry in data['sequences']:
        for _type, component in entry.items():
            if isinstance(component['id'], str):
                component['id'] = [component['id']]

            for i, _id in enumerate(component['id']):
                if _type == 'protein':
                    new_component = {
                        'seq': component['sequence'],
                        'chain_type': ChainType.POLYPEPTIDE_L,
                    }
                    if 'unpairedMsaPath' in component:
                        new_component['msa_path'] = component['unpairedMsaPath']
                    components.append(new_component)
                elif _type == 'rna':
                    new_component ={
                        'seq': component['sequence'],
                        'chain_type': ChainType.RNA,
                    }
                    if 'unpairedMsaPath' in component:
                        new_component['msa_path'] = component['unpairedMsaPath']
                    components.append(new_component)
                elif _type == 'dna':
                    new_component = {
                        'seq': component['sequence'],
                        'chain_type': ChainType.DNA,
                    }
                    if 'unpairedMsaPath' in component:
                        new_component['msa_path'] = component['unpairedMsaPath']
                    components.append(new_component)

                elif _type == 'ligand':
                    components.append({
                        'ccd_code': component['ccdCodes'][i]
                    })

    atom_array, components = components_to_atom_array(components, return_components=True)
    atom_array_data = parse_atom_array(atom_array)
    atom_array = atom_array_data['assemblies']['1'][0]
    chain_info = atom_array_data['chain_info']

    for component in components:
        if hasattr(component, 'msa_path'):
            chain_info[component.chain_id]['msa_path'] = component.msa_path

    return {
        "example_id": example_id,
        "atom_array": atom_array,
        "chain_info": chain_info,
    }

class HotfixDropSaccharideO1(Transform):
    def forward(self, data):
        atom_array = data['atom_array']
        res_names = np.unique(atom_array.res_name)
        saccharide_res_names = [res_name for res_name in res_names if 'D-SACCHARIDE' in ccd.get_chem_comp_type(res_name)]
        mask = np.isin(atom_array.res_name, saccharide_res_names) & (atom_array.atom_name == 'O1')
        data['atom_array'] = atom_array[~mask]

        return data

class HotfixFillRefSpaceUID(Transform):
    def forward(self, data):
        ref_struct = data['ref_struct']
        token_layout = ref_struct.to_token_layout(ref_struct.ref_space_uid)
        token_layout[...] = token_layout[:, :1]
        ref_struct.ref_space_uid = ref_struct.to_atom_layout(token_layout)
        return data




def custom_af3_pipeline(n_recycling_iterations, msa_shuffle_orders=None):
    # msa_shuffle_orders = np.stack([
    #     torch.load(f'/Users/kilianmandon/Projects/alphafold3/kilian/feature_extraction/test_outputs_lysozyme/rec_{i}_msa_shuffle_order.pt', weights_only=False).long().numpy()
    #     for i in range(2)], axis=0)
    transforms = [
        RemoveHydrogens(),
        HotfixDropSaccharideO1(),
        # RemoveTerminalOxygen(),
        AtomizeByCCDName(
            atomize_by_default=True,
            res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
        ),
        CalculateTokenFeatures(),
        CalculateRefStructFeatures(),
        # HotfixFillRefSpaceUID(),
        CalculateMSAFeatures(msa_shuffle_orders=msa_shuffle_orders, n_recycling_iterations=n_recycling_iterations),
        CalculateContactMatrix(),
        # ExpandBatchDimension(),
    ]

    return Compose(transforms)

