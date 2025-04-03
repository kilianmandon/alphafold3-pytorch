import copy
import json

import math
from pathlib import Path
import pickle
from typing import Dict, List

import numpy as np
import rdkit
import torch
from torch.nn import functional as F
from ccd import drop_atoms, load_ccd
from atom_layout import AtomLayout
from model import Model

import modelcif
import modelcif.model
import ihm

import residue_constants
import utils


from dataclasses import dataclass, field

from msa_features import crop_pad_to_shape, empty_msa, join_msas, merge_unpaired_paired, pad_to_shape, process_msa_file

base_path = Path('data')


class Config:
    max_row_count = 16384


config = Config()


class ReferenceStructure:
    ref_structures = dict()
    ref_space_uid = 0

    @classmethod
    def reserve_ref_space(cls):
        current = cls.ref_space_uid
        cls.ref_space_uid += 1
        return current

    @staticmethod
    def get_atoms(ccd, res_name, mask, positions, asym_unit, seq_id, leaving_atoms=[]):
        mol = ccd[res_name]['mol']

        # atoms = [a for a in mol.GetAtoms() if a.GetSymbol() != 'H' and a.GetProp('atom_name') not in leaving_atoms]
        atoms = drop_atoms(ccd[res_name], drop_hydrogens=True)
        atom_names = atoms['_chem_comp_atom.atom_id']
        atom_elements = atoms['_chem_comp_atom.type_symbol']

        atom_elements = [e for a, e in zip(atom_names, atom_elements) if a not in leaving_atoms]
        atom_names = [a for a in atom_names if a not in leaving_atoms]

        for idx, (atom_name, atom_type) in enumerate(zip(atom_names, atom_elements)):
            if not mask[idx]:
                continue

            x,y,z = (a.item() for a in positions[idx])

            yield modelcif.model.Atom(
                asym_unit=asym_unit,
                type_symbol=atom_type,
                seq_id=seq_id,
                atom_id=atom_name,
                x=x, y=y, z=z,
                het=False,
                occupancy=1.00,
            )


    @staticmethod
    def calculate_data(res_name, ccd, pad=True):
        mol = ccd[res_name]['mol']

        params = rdkit.Chem.AllChem.ETKDGv3()
        params.randomSeed = 0
        mol_copy = rdkit.Chem.Mol(mol)
        conformer_id = rdkit.Chem.AllChem.EmbedMolecule(mol_copy, params)
        conformer = mol_copy.GetConformer(conformer_id)

        used_atom_data = drop_atoms(ccd[res_name], drop_hydrogens=True)
        all_atom_data = dict()

        for idx, atom in enumerate(mol_copy.GetAtoms()):
            name = atom.GetProp('atom_name')
            name_chars = torch.tensor([ord(c)-32 for c in name])
            name_chars = pad_to_shape(name_chars, (4,))

            element = atom.GetAtomicNum()
            charge = atom.GetFormalCharge()
            coords = conformer.GetAtomPosition(idx)
            pos = torch.tensor([coords.x, coords.y, coords.z])

            all_atom_data[name] = {
                'element': element,
                'pos': pos,
                'charge': charge,
                'atom_name_chars': name_chars
            }

        used_names = used_atom_data['_chem_comp_atom.atom_id']
        all_elements = torch.tensor(
            [all_atom_data[name]['element'] for name in used_names])
        all_pos = torch.stack([all_atom_data[name]['pos']
                              for name in used_names], dim=0)
        all_charges = torch.tensor(
            [all_atom_data[name]['charge'] for name in used_names])
        all_atom_name_chars = torch.stack(
            [all_atom_data[name]['atom_name_chars'] for name in used_names], dim=0)

        full_data = {
            'element': all_elements,
            'positions': all_pos,
            'charge': all_charges,
            'atom_name_chars': all_atom_name_chars,
            'mask': torch.ones_like(all_elements),
        }

        if pad:
            for key, val in full_data.items():
                full_data[key] = crop_pad_to_shape(val, (24,)+val.shape[1:])

        full_data['atom_names'] = used_names

        return full_data

    @classmethod
    def get_ref_structure(cls, chain_id, res_name, ccd, pad=True, drop_atoms=[]):
        if not (chain_id, res_name) in cls.ref_structures:
            cls.ref_structures[(chain_id, res_name)] = cls.calculate_data(
                res_name, ccd, pad)

        ref_struct = copy.deepcopy(cls.ref_structures[(chain_id, res_name)])
        ref_space_uid = cls.reserve_ref_space()
        # ref_struct['ref_space_uid'] = ref_struct['mask'] * ref_space_uid
        ref_struct['ref_space_uid'] = torch.full_like(
            ref_struct['mask'], ref_space_uid)

        for i, atom_name in enumerate(ref_struct['atom_names']):
            if atom_name in drop_atoms:
                ref_struct['element'][i] = 0
                ref_struct['mask'][i] = 0
                ref_struct['charge'][i] = 0
                ref_struct['positions'][i, :] = 0
                ref_struct['atom_name_chars'][i, :] = 0
                ref_struct['atom_names'][i] = ''
                # ref_struct['ref_space_uid'][i] = 0

        filtered = {key: val for key, val in ref_struct.items()
                    if key != 'atom_names'}

        return filtered


@dataclass
class Protein:
    id: List[str]
    sequence: str
    unpairedMsaPath: str
    pairedMsa: str
    templates: List[str]

    @property
    def tokenCount(self):
        return len(self.sequence) * len(self.id)

    def three_letter_code(self):
        return [residue_constants.restypes_three_letter[residue_constants.restypes.index(a)] for a in self.sequence]

    def calculate_msa_feats(self, deduplicate=True):
        unpaired = process_msa_file(
            base_path / self.unpairedMsaPath, 'protein')
        paired = empty_msa(self.sequence, 'protein')
        merged = merge_unpaired_paired(
            unpaired, paired, config.max_row_count, deduplicate)
        return [merged] * len(self.id)

    def calculate_ref_structure(self, ccd):
        keys = ['positions', 'mask', 'element',
                'charge', 'atom_name_chars', 'ref_space_uid']
        all_data = []
        for id in self.id:
            for i, res in enumerate(self.three_letter_code()):
                drop_atoms = [] if i == len(self.sequence)-1 else ['OXT']

                data = ReferenceStructure.get_ref_structure(
                    id, res, ccd, drop_atoms=drop_atoms)
                all_data.append(data)

        all_data_stacked = dict()

        for key in keys:
            gathered = [data[key] for data in all_data]
            all_data_stacked[key] = torch.stack(gathered, dim=0)
        return all_data_stacked

    def calculate_contact_matrix(self):
        shape = (len(self.sequence), len(self.sequence))
        return [torch.zeros(shape)] * len(self.id)

    def get_model(self, ccd, token_positions: torch.Tensor, mask: torch.Tensor):
        entity = modelcif.Entity(self.sequence, description='Protein')
        asym_units = [
            modelcif.AsymUnit(entity, details='Protein Subunit {asym_id}', id=asym_id) for asym_id in self.id
        ]

        def atom_iterator():
            token = 0
            for asym_unit in asym_units:
                for i, res in enumerate(self.three_letter_code()):
                    yield from ReferenceStructure.get_atoms(ccd, res, mask[token], token_positions[token], asym_unit, i+1)
                    token += 1


        return asym_units, atom_iterator()




@dataclass
class Ligand:
    id: List[str]
    ccdCodes: List[str]
    ccdData: Dict = field(default=None)

    @property
    def sequence(self):
        return '-' * len(self.ccdData['_chem_comp_atom.atom_id'])

    @property
    def tokenCount(self):
        return len(self.sequence) * len(self.id)

    def setup(self, ccd):
        self.ccdData = drop_atoms(
            ccd[self.ccdCodes[0]], drop_hydrogens=True, drop_leaving_atoms=True)

    def calculate_msa_feats(self, deduplicate=True):
        return [empty_msa(self.sequence, 'ligand')] * len(self.id)

    def calculate_ref_structure(self, ccd):
        keys = ['positions', 'mask', 'element',
                'charge', 'atom_name_chars', 'ref_space_uid']
        all_data = []

        res = self.ccdCodes[0]
        is_saccharide = 'saccharide' in self.ccdData['_chem_comp.type'].lower()
        atoms_to_drop = ['O1'] if is_saccharide else []
        for id in self.id:
            ref_struct = ReferenceStructure.get_ref_structure(
                id, res, ccd, pad=False, drop_atoms=atoms_to_drop)

            leaving_mask = ref_struct['mask'] == 1
            for key, val in ref_struct.items():
                # TODO: Check if leaving atoms should still occupy tokens
                val = val[leaving_mask]
                ref_struct[key] = pad_to_shape(
                    val[:, None, ...], val.shape[:1]+(24,)+val.shape[1:])
            all_data.append(ref_struct)

        all_data_stacked = dict()

        for key in keys:
            gathered = [data[key] for data in all_data]
            all_data_stacked[key] = torch.cat(gathered, dim=0)
        return all_data_stacked

    def calculate_contact_matrix(self):
        shape = (len(self.sequence), len(self.sequence))
        contacts = torch.zeros(shape)
        bond_id1 = self.ccdData['_chem_comp_bond.atom_id_1']
        bond_id2 = self.ccdData['_chem_comp_bond.atom_id_2']

        for bid1, bid2 in zip(bond_id1, bond_id2):
            bid1 = self.ccdData['_chem_comp_atom.atom_id'].index(bid1)
            bid2 = self.ccdData['_chem_comp_atom.atom_id'].index(bid2)
            bid1, bid2 = min(bid1, bid2), max(bid1, bid2)
            contacts[bid1, bid2] = 1

        return [contacts] * len(self.id)

    def get_model(self, ccd, token_positions: torch.Tensor, mask: torch.Tensor):
        res_masks = torch.split(mask[:, 0, ...], len(self.sequence), dim=0)
        res_token_positions = torch.split(token_positions[:, 0, ...], len(self.sequence), dim=0)

        entity = modelcif.Entity([ihm.NonPolymerChemComp(self.ccdCodes[0])], description='Ligand')

        asym_units = [
            modelcif.AsymUnit(entity, details='Ligand Subunit {asym_id}', id=asym_id) for asym_id in self.id
        ]

        def atom_iterator():
            is_saccharide = 'saccharide' in self.ccdData['_chem_comp.type'].lower()
            leaving_atoms = ['O1'] if is_saccharide else []

            for i in range(len(self.id)):
                ccdCode = self.ccdCodes[i]
                pos = res_token_positions[i]
                mask = res_masks[i]
                yield from ReferenceStructure.get_atoms(ccd, ccdCode, mask, pos, asym_units[i], 1, leaving_atoms=leaving_atoms)



        return asym_units, atom_iterator()





@dataclass
class RNA:
    id: List[str]
    sequence: str
    unpairedMsa: str
    unpairedMsaPath: str

    @property
    def tokenCount(self):
        return len(self.sequence) * len(self.id)

    def calculate_msa_feats(self, deduplicate=True):
        unpaired = empty_msa(self.sequence, 'rna')
        paired = empty_msa(self.sequence, 'protein')
        merged = merge_unpaired_paired(
            unpaired, paired, config.max_row_count, deduplicate)
        return [merged] * len(self.id)

    def calculate_ref_structure(self, ccd):
        keys = ['positions', 'mask', 'element',
                'charge', 'atom_name_chars', 'ref_space_uid']
        all_data = []
        for id in self.id:
            for i, res in enumerate(self.sequence):
                drop_atoms = [] if i == 0 else ['OP3']
                data = ReferenceStructure.get_ref_structure(
                    id, res, ccd, drop_atoms=drop_atoms)
                all_data.append(data)

        all_data_stacked = dict()

        for key in keys:
            gathered = [data[key] for data in all_data]
            all_data_stacked[key] = torch.stack(gathered, dim=0)
        return all_data_stacked

    def calculate_contact_matrix(self):
        shape = (len(self.sequence), len(self.sequence))
        return [torch.zeros(shape)] * len(self.id)


    def get_model(self, ccd, token_positions: torch.Tensor, mask: torch.Tensor):
        entity = modelcif.Entity(self.sequence, description='RNA')
        asym_units = [
            modelcif.AsymUnit(entity, details='RNA Subunit {asym_id}', id=asym_id) for asym_id in self.id
        ]

        def atom_iterator():
            token = 0
            for asym_unit in asym_units:
                for i, res in enumerate(self.three_letter_code()):
                    yield from ReferenceStructure.get_atoms(ccd, res, mask[token], token_positions[token], asym_unit, i+1)
                    token += 1


        return asym_units, atom_iterator()


class Input:
    sequences: List[Protein | RNA | Ligand]

    @property
    def tokenCount(self):
        return sum(seq.tokenCount for seq in self.sequences)

    def __init__(self, sequences):
        self.sequences = sequences

    @staticmethod
    def load_input(input_file: str):
        with open(input_file, 'r') as f:
            data = json.load(f)
        name = data['name']
        seqs_json = data['sequences']
        seqs = []
        ccd = load_ccd()

        for seq in seqs_json:
            if 'protein' in seq:
                seqs.append(Protein(**seq['protein']))
            elif 'ligand' in seq:
                ligand = Ligand(**seq['ligand'])
                ligand.setup(ccd)
                seqs.append(ligand)
            elif 'rna' in seq:
                seqs.append(RNA(**seq['rna']))
            else:
                raise NotImplementedError(
                    f'Sequence type not implemented: {seq.keys()}')

        return Input(seqs)

    def calculate_msa_feats(self):
        msa_col_count = round_to_bucket(self.tokenCount)
        deduplicate = len([p for p in self.sequences if isinstance(p, Protein)]) > 1
        all_msa_feats = sum((seq.calculate_msa_feats(deduplicate) for seq in self.sequences), [])
        joined_msa_feats = join_msas(all_msa_feats, config.max_row_count, msa_col_count)

        msa_one_hot = F.one_hot(joined_msa_feats['rows'], num_classes=32)
        deletion_matrix = joined_msa_feats['deletion_matrix'][..., None]
        has_deletion = torch.clip(deletion_matrix, 0, 1)
        deletion_value = (2/torch.pi) * torch.arctan(deletion_matrix/3)
        msa_feat = [msa_one_hot, has_deletion, deletion_value]

        msa_feat = torch.cat(msa_feat, dim=-1)
        joined_msa_feats['msa_feat'] = msa_feat

        return joined_msa_feats

    def truncate_msa_feat(self, msa_feat, msa_mask, msa_shuffle_order=None, msa_trunc_count=1024):
        # msa_feat has shape (*, N_seqs, N_token, c)
        # msa_mask has shape (*, N_seqs, N_token)
        batch_shape = msa_feat.shape[:-3]

        if msa_shuffle_order is None:
            has_entries = torch.clip(msa_mask.sum(dim=-1), 0, 1)
            odds = (has_entries - 1) * -1e2
            scores = odds + torch.distributions.Gumbel(0, 1).sample(odds.shape)
            # sel_inds has shape (*, 1024)
            msa_shuffle_order = torch.argsort(scores, dim=-1)

        msa_shuffle_order = msa_shuffle_order[..., :msa_trunc_count]

        msa_feat = utils.batched_gather(msa_feat, msa_shuffle_order, batch_shape)
        msa_mask = utils.batched_gather(msa_mask, msa_shuffle_order, batch_shape)

        return msa_feat, msa_mask

    def sample_n_msa_feats(self, batch, n=11, msa_trunc_count=1024, msa_shuffle_order = None):
        msa_feats = []
        msa_masks = []
        for i in range(n):
            msa_feat, msa_mask = self.truncate_msa_feat(batch['msa_feat'], batch['msa_mask'], msa_shuffle_order=msa_shuffle_order[i], msa_trunc_count=msa_trunc_count)
            msa_feats.append(msa_feat)
            msa_masks.append(msa_mask)
        msa_feat = torch.stack(msa_feats, dim=-1)
        msa_mask = torch.stack(msa_masks, dim=-1)
        batch['msa_feat'] = msa_feat
        batch['msa_mask'] = msa_mask

    def calculate_target_feat(self, batch):
        aatype = batch['msa_aatype']
        aatype_1h = F.one_hot(aatype, 31)
        profile = batch['profile']
        deletion_mean = batch['deletion_mean'][..., None]

        target_feat = torch.cat((aatype_1h, profile, deletion_mean), dim=-1)
        return target_feat

    def calculate_contact_matrix(self):
        padded_token_count = round_to_bucket(self.tokenCount)
        contact_blocks = sum((seq.calculate_contact_matrix()
                             for seq in self.sequences), [])
        contact_matrix = torch.block_diag(*contact_blocks)
        contact_matrix = pad_to_shape(
            contact_matrix, (padded_token_count, padded_token_count))

        return contact_matrix[..., None]

    def calculate_token_features(self):
        # token_index = seq_features.token_index
        # residue_index = seq_features.residue_index
        # asym_id = seq_features.asym_id
        # entity_id = seq_features.entity_id
        # sym_id = seq_features.sym_id
        token_index = torch.arange(self.tokenCount) + 1
        residue_indices = []
        asym_id_counter = 1
        entity_id_counter = 1
        asym_ids = []
        entity_ids = []
        sym_ids = []
        for seq in self.sequences:
            if isinstance(seq, Ligand):
                residue_index = torch.ones((len(seq.sequence),), dtype=torch.int64)
            else:
                residue_index = torch.arange(len(seq.sequence)) + 1

            residue_indices += [residue_index] * len(seq.id)
            new_asym_ids = []
            new_sym_ids = []
            new_entity_ids = []
            sym_id_counter = 1
            for _ in seq.id:
                shape = (len(seq.sequence),)
                new_asym_ids.append(torch.full(shape, asym_id_counter))
                new_sym_ids.append(torch.full(shape, sym_id_counter))
                new_entity_ids.append(torch.full(shape, entity_id_counter))
                asym_id_counter += 1
                sym_id_counter += 1

            entity_id_counter += 1
            asym_ids += new_asym_ids
            sym_ids += new_sym_ids
            entity_ids += new_entity_ids

        residue_index = torch.cat(residue_indices, dim=0)
        asym_id = torch.cat(asym_ids, dim=0)
        entity_id = torch.cat(entity_ids, dim=0)
        sym_id = torch.cat(sym_ids, dim=0)
        single_mask = torch.ones((self.tokenCount,))

        token_features = {
            'residue_index': residue_index,
            'token_index': token_index,
            'asym_id': asym_id,
            'entity_id': entity_id,
            'sym_id': sym_id,
            'single_mask': single_mask,
        }

        padded_token_count = round_to_bucket(self.tokenCount)

        for key, val in token_features.items():
            token_features[key] = pad_to_shape(val, (padded_token_count,))

        return token_features

    def calculate_ref_structure(self):
        ccd = load_ccd()
        keys = ['positions', 'mask', 'element',
                'charge', 'atom_name_chars', 'ref_space_uid']
        all_data = []
        for seq in self.sequences:
            all_data.append(seq.calculate_ref_structure(ccd))

        ref_struct = dict()
        data_count = round_to_bucket(self.tokenCount)
        atom_count = data_count * 24

        for key in keys:
            gathered = [data[key] for data in all_data]
            stacked = torch.cat(gathered, dim=0)
            ref_struct[key] = pad_to_shape(
                stacked, (data_count,)+stacked.shape[1:])

        atom_layout = AtomLayout.from_single_mask(ref_struct['mask'])
        ref_struct['atom_layout'] = atom_layout

        return ref_struct

    def create_batch(self, msa_shuffle_order=None):
        batch = {
            'ref_struct': self.calculate_ref_structure(),
            'contact_matrix': self.calculate_contact_matrix(),
            **self.calculate_token_features(),
            **self.calculate_msa_feats(),
        }
        batch['target_feat'] = self.calculate_target_feat(batch)
        self.sample_n_msa_feats(batch, msa_shuffle_order=msa_shuffle_order)

        return batch


def round_to_bucket(v):
    buckets = np.array([256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072,
                        3584, 4096, 4608, 5120])

    return buckets[np.argmax(buckets >= v)]


def msa_list_test(inp: Input, outp):
    msa_trunc_count = 1024
    sel_inds = torch.load(Path(outp) / 'msa_shuffle_order.pt',
                          weights_only=False)[:msa_trunc_count].int()
    my_feats = inp.calculate_msa_feature_list(sel_inds=sel_inds)
    their_feats = torch.load(
        Path(outp) / 'msa_batch_03_truncated.pt', weights_only=False)

    their_feat_names = ['rows', 'mask',
                        'deletion_matrix', 'profile', 'deletion_mean']
    my_feat_names = ['rows', 'msa_mask',
                     'deletion_matrix', 'profile', 'deletion_mean']

    diff_inds = dict()
    switches = np.cumsum([seq.tokenCount for seq in inp.sequences])

    for their_key, my_key in zip(their_feat_names, my_feat_names):
        their_feat = their_feats[their_key]
        my_feat = my_feats[my_key]

        assert their_feat.shape == my_feat.shape

        diff_inds[my_key] = torch.nonzero(
            torch.abs(their_feat-my_feat) > 1e-3).numpy()
        assert torch.allclose(my_feat.float(), their_feat.float(), atol=1e-6)

    pass


def msa_feat_test(inp: Input, outp):
    msa_trunc_count = 1024
    sel_inds = torch.load(Path(outp) / 'msa_shuffle_order.pt',
                          weights_only=False)[:msa_trunc_count].int()

    my_feat = inp.calculate_msa_feat(sel_inds=sel_inds)
    their_feat = torch.load(Path(outp) / 'msa_feat.pt', weights_only=False)
    assert torch.allclose(my_feat, their_feat, atol=1e-6)


def ref_struct_test(inp: Input, output):
    msa_trunc_count = 1024
    model = Model()
    params = torch.load('data/params/af3_pytorch.pt', weights_only=False)
    res = model.load_state_dict(params, strict=False)
    if len(res.unexpected_keys) > 0 or len(res.missing_keys) > 0:
        print("------- Problems with parameter loading ----------")
        print('Missing keys:')
        print([k for k in res.missing_keys if not any(f'.{i}.' in k for i in range(1, 50))])
        print('Unexpected keys:')
        print(res.unexpected_keys)
        print()
    # model.atom_cross_att.per_atom_cond(ref_struct)
    print('Parameters loaded.')
    batch = inp.create_batch()
    model.eval()
    device = torch.device('mps')
    # device = 'cpu'
    model.to(device=device)
    batch = move_to_device(batch, device=device)
    with torch.no_grad():
        token_positions, token_mask = model(batch)
    
    token_positions = token_positions.to(device='cpu')
    token_mask = token_mask.to(device='cpu')
    ccd = load_ccd()
    mmcif_string = utils.to_modelcif(token_positions, token_mask, inp, ccd)
    with open('my_own_model.cif', 'w') as f:
        f.write(mmcif_string)
    pass

def move_to_device(obj, device):
    """Recursively move all tensors in a nested structure to a specified device."""
    if isinstance(obj, torch.Tensor):  
        return obj.to(device)  # Move tensor to device
    elif isinstance(obj, dict):  
        return {key: move_to_device(value, device) for key, value in obj.items()}  
    elif isinstance(obj, list):  
        return [move_to_device(item, device) for item in obj]  
    elif isinstance(obj, tuple):  
        return tuple(move_to_device(item, device) for item in obj)  
    elif hasattr(obj, "__dict__"):  
        # If the object has a __dict__, it's a custom class -> move its attributes
        for attr in vars(obj):  
            setattr(obj, attr, move_to_device(getattr(obj, attr), device))
        return obj  
    else:
        return obj  # Return unchanged if not a tensor, dict, list, or tuple

def tests():
    test_names = ['Multimer', 'Lysozyme']
    test_inputs = ['data/fold_input_lysozyme.json']
    test_outputs = ['kilian/feature_extraction/test_outputs_multimer']

    for name, inp_file, output in zip(test_names, test_inputs, test_outputs):
        msa_trunc_count = 1024
        print(f"Running tests for {name}...")
        inp = Input.load_input(inp_file)
        ref_struct_test(inp, None)



def main():
    tests()


if __name__ == '__main__':
    main()
