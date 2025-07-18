import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from torch.nn import functional as F
from feature_extraction.ccd import drop_atoms
from atom_layout import AtomLayout
import modelcif
import ihm
from feature_extraction.ref_struct_features import ReferenceStructure
import residue_constants

from feature_extraction.msa_features import calculate_msa_feat, calculate_target_feat, empty_msa, merge_msa_features, merge_unpaired_paired, process_msa_file, sample_n_msa_feats
import abc

import utils

base_path = Path('data')

class Config:
    max_row_count = 16384

config = Config()

def round_to_bucket(v):
    buckets = np.array([256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072,
                        3584, 4096, 4608, 5120])

    return buckets[np.argmax(buckets >= v)]


class Sequence(abc.ABC):
    @abc.abstractmethod
    def calculate_msa_features(self, deduplicate=True): pass

    @abc.abstractmethod
    def calculate_ref_structure(self, ccd: dict): pass

    @abc.abstractmethod
    def calculate_contact_matrix(self): pass
    
    @abc.abstractmethod
    def get_model(self, ccd, token_positions: torch.Tensor, mask: torch.Tensor): pass


class Protein(Sequence):

    def __init__(self, id, sequence, unpairedMsaPath, pairedMsa, templates):
        self.id = id
        self.sequence = sequence
        self.unpaired_msa_path = unpairedMsaPath
        self.paired_msa = pairedMsa
        self.templates = templates

        self.token_count = len(self.sequence) * len(self.id)


    def calculate_msa_features(self, deduplicate=True):
        unpaired = process_msa_file(base_path / self.unpaired_msa_path, 'protein')
        paired = empty_msa(self.sequence, 'protein')
        merged = merge_unpaired_paired(unpaired, paired, config.max_row_count, deduplicate)

        n = len(self.id)
        return merge_msa_features([merged] * n)


    def calculate_ref_structure(self, ccd):
        keys = ['ref_pos', 'ref_mask', 'ref_element',
                'ref_charge', 'ref_atom_name_chars', 'ref_space_uid']
        all_data = []
        three_letter_codes = [residue_constants.restypes_one_to_three[a] for a in self.sequence]
        for chain_id in self.id:
            for i, res in enumerate(three_letter_codes):
                drop_atoms = [] if i == len(self.sequence)-1 else ['OXT']
                data = ReferenceStructure.get_ref_structure(chain_id, res, ccd, drop_atoms=drop_atoms)
                all_data.append(data)

        all_data_stacked = dict()

        for key in keys:
            gathered = [data[key] for data in all_data]
            all_data_stacked[key] = torch.stack(gathered, dim=0)
        return all_data_stacked

    def calculate_contact_matrix(self):
        shape = (len(self.sequence), len(self.sequence))
        n = len(self.id)
        contact_matrices = [torch.zeros(shape)] * n

        return torch.block_diag(*contact_matrices)

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




class Ligand(Sequence):

    def __init__(self, id: List[str], ccdCodes: List[str], ccd: Dict):
        self.id = id
        self.ccd_codes = ccdCodes
        self.ccd_data = drop_atoms(ccd[self.ccd_codes[0]], drop_hydrogens=True, drop_leaving_atoms=True)
        self.sequence = '-' * len(self.ccd_data['atom_id'])
        self.token_count = len(self.sequence) * len(self.id)

    def calculate_msa_features(self, deduplicate=True):
        msa_features = empty_msa(self.sequence, 'ligand')
        n = len(self.id)
        return merge_msa_features([msa_features] * n)

    def calculate_ref_structure(self, ccd):
        keys = ['ref_pos', 'ref_mask', 'ref_element',
                'ref_charge', 'ref_atom_name_chars', 'ref_space_uid']
        all_data = []

        res = self.ccd_codes[0]
        is_saccharide = 'saccharide' in self.ccd_data['type'].lower()
        atoms_to_drop = ['O1'] if is_saccharide else []
        for id in self.id:
            ref_struct = ReferenceStructure.get_ref_structure(
                id, res, ccd, pad=False, drop_atoms=atoms_to_drop)

            leaving_mask = ref_struct['ref_mask'] == 1
            for key, val in ref_struct.items():
                # TODO: Check if leaving atoms should still occupy tokens
                val = val[leaving_mask]
                ref_struct[key] = utils.pad_to_shape(
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
        bond_id1 = self.ccd_data['bond_atom_id_1']
        bond_id2 = self.ccd_data['bond_atom_id_2']

        for bid1, bid2 in zip(bond_id1, bond_id2):
            bid1 = self.ccd_data['atom_id'].index(bid1)
            bid2 = self.ccd_data['atom_id'].index(bid2)
            bid1, bid2 = min(bid1, bid2), max(bid1, bid2)
            contacts[bid1, bid2] = 1

        n = len(self.id)
        contact_matrices = [contacts] * n

        return torch.block_diag(*contact_matrices)

    def get_model(self, ccd, token_positions: torch.Tensor, mask: torch.Tensor):
        res_masks = torch.split(mask[:, 0, ...], len(self.sequence), dim=0)
        res_token_positions = torch.split(token_positions[:, 0, ...], len(self.sequence), dim=0)

        entity = modelcif.Entity([ihm.NonPolymerChemComp(self.ccd_codes[0])], description='Ligand')

        asym_units = [
            modelcif.AsymUnit(entity, details='Ligand Subunit {asym_id}', id=asym_id) for asym_id in self.id
        ]

        def atom_iterator():
            is_saccharide = 'saccharide' in self.ccd_data['type'].lower()
            leaving_atoms = ['O1'] if is_saccharide else []

            for i in range(len(self.id)):
                ccdCode = self.ccd_codes[i]
                pos = res_token_positions[i]
                mask = res_masks[i]
                yield from ReferenceStructure.get_atoms(ccd, ccdCode, mask, pos, asym_units[i], 1, leaving_atoms=leaving_atoms)



        return asym_units, atom_iterator()


class RNA(Sequence):
    def __init__(self, id, sequence, unpairedMsa, unpairedMsaPath):
        self.id = id
        self.sequence = sequence
        self.unpaired_msa = unpairedMsa
        self.unpaired_msa_path = unpairedMsaPath
        self.token_count = len(self.sequence) * len(self.id)


    def calculate_msa_features(self, deduplicate=True):
        unpaired = empty_msa(self.sequence, 'rna')
        paired = empty_msa(self.sequence, 'protein')
        merged = merge_unpaired_paired(
            unpaired, paired, config.max_row_count, deduplicate)

        n = len(self.id)
        return merge_msa_features([merged] * n)

    def calculate_ref_structure(self, ccd):
        keys = ['ref_pos', 'ref_mask', 'ref_element',
                'ref_charge', 'ref_atom_name_chars', 'ref_space_uid']
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
        n = len(self.id)
        contact_matrices = [torch.zeros(shape)] * n

        return torch.block_diag(*contact_matrices)


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
    def __init__(self, sequences: List[Sequence], ccd):
        self.sequences = sequences
        self.token_count = sum(seq.tokenCount for seq in self.sequences)
        self.ccd = ccd

    @staticmethod
    def load_input(input_file: str, ccd):
        with open(input_file, 'r') as f:
            data = json.load(f)
        name = data['name']
        seqs_json = data['sequences']
        seqs = []

        for seq in seqs_json:
            if 'protein' in seq:
                seqs.append(Protein(**seq['protein']))
            elif 'ligand' in seq:
                seqs.append(Ligand(ccd=ccd, **seq['ligand']))
            elif 'rna' in seq:
                seqs.append(RNA(**seq['rna']))
            else:
                raise NotImplementedError(
                    f'Sequence type not implemented: {seq.keys()}')

        return Input(seqs, ccd)

    def calculate_msa_features(self, msa_shuffle_orders=None):
        msa_col_count = round_to_bucket(self.token_count)
        protein_count = len([p for p in self.sequences if isinstance(p, Protein)])
        deduplicate = protein_count > 1

        single_chain_features = [seq.calculate_msa_features(deduplicate) for seq in self.sequences]
        msa_features = merge_msa_features(single_chain_features, config.max_row_count, msa_col_count)

        msa_feat = calculate_msa_feat(msa_features)
        msa_mask = msa_features['msa_mask']

        msa_feat, msa_mask = sample_n_msa_feats(msa_feat, msa_mask, n=11, msa_shuffle_orders=msa_shuffle_orders)
        msa_features['msa_feat'] = msa_feat
        msa_features['msa_mask'] = msa_mask

        msa_features['target_feat'] = calculate_target_feat(msa_features)

        return msa_features


    def calculate_contact_matrix(self):
        padded_token_count = round_to_bucket(self.token_count)
        contact_blocks = [seq.calculate_contact_matrix() for seq in self.sequences]
        contact_matrix = torch.block_diag(*contact_blocks)
        contact_matrix = utils.pad_to_shape(contact_matrix, (padded_token_count, padded_token_count))

        return contact_matrix[..., None]

    def calculate_token_features(self):
        token_index = torch.arange(self.token_count) + 1
        residue_indices = []
        asym_id_counter = 1
        entity_id_counter = 1
        asym_ids = []
        entity_ids = []
        sym_ids = []
        for seq in self.sequences:
            if isinstance(seq, Ligand):
                new_residue_indices = torch.ones((len(seq.sequence),), dtype=torch.int64)
            else:
                new_residue_indices = torch.arange(len(seq.sequence)) + 1

            residue_indices += [new_residue_indices] * len(seq.id)
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
        single_mask = torch.ones((self.token_count,))

        token_features = {
            'residue_index': residue_index,
            'token_index': token_index,
            'asym_id': asym_id,
            'entity_id': entity_id,
            'sym_id': sym_id,
            'single_mask': single_mask,
        }

        padded_token_count = round_to_bucket(self.token_count)

        for key, val in token_features.items():
            token_features[key] = utils.pad_to_shape(val, (padded_token_count,))

        return token_features

    def calculate_ref_structure(self):
        keys = ['ref_pos', 'ref_mask', 'ref_element',
                'ref_charge', 'ref_atom_name_chars', 'ref_space_uid']

        all_data = [seq.calculate_ref_structure(self.ccd) for seq in self.sequences]

        ref_struct = dict()
        padded_token_count = round_to_bucket(self.token_count)
        atom_count = padded_token_count * 24

        for key in keys:
            gathered = [data[key] for data in all_data]
            stacked = torch.cat(gathered, dim=0)
            ref_struct[key] = utils.pad_to_shape(
                stacked, (padded_token_count,)+stacked.shape[1:])

        atom_layout = AtomLayout.from_ref_mask(ref_struct['ref_mask'])
        ref_struct['atom_layout'] = atom_layout

        return ref_struct

    def create_batch(self, msa_shuffle_orders=None):
        batch = {
            'ref_struct': self.calculate_ref_structure(),
            'token_features': self.calculate_token_features(),
            'msa_features': self.calculate_msa_features(msa_shuffle_orders=msa_shuffle_orders),
            'contact_matrix': self.calculate_contact_matrix(),
        }

        return batch


