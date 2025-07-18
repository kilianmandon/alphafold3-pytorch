import copy
import modelcif
import rdkit
import torch

from feature_extraction.ccd import drop_atoms
from feature_extraction.msa_features import crop_pad_to_shape, pad_to_shape


class ReferenceStructure:
    ref_structures = dict()
    ref_space_uid = 0

    @classmethod
    def get_and_increase_uid_counter(cls):
        current = cls.ref_space_uid
        cls.ref_space_uid += 1
        return current

    @staticmethod
    def get_atoms(ccd, res_name, mask, positions, asym_unit, seq_id, leaving_atoms=[]):
        # atoms = [a for a in mol.GetAtoms() if a.GetSymbol() != 'H' and a.GetProp('atom_name') not in leaving_atoms]
        atoms = drop_atoms(ccd[res_name], drop_hydrogens=True)
        atom_names = atoms['atom_id']
        atom_elements = atoms['atom_type']

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
                'ref_element': element,
                'pos': pos,
                'ref_charge': charge,
                'ref_atom_name_chars': name_chars
            }

        used_names = used_atom_data['atom_id']
        all_elements = torch.tensor(
            [all_atom_data[name]['ref_element'] for name in used_names])
        all_pos = torch.stack([all_atom_data[name]['pos']
                              for name in used_names], dim=0)
        all_charges = torch.tensor(
            [all_atom_data[name]['ref_charge'] for name in used_names])
        all_atom_name_chars = torch.stack(
            [all_atom_data[name]['ref_atom_name_chars'] for name in used_names], dim=0)

        full_data = {
            'ref_element': all_elements,
            'ref_pos': all_pos,
            'ref_charge': all_charges,
            'ref_atom_name_chars': all_atom_name_chars,
            'ref_mask': torch.ones_like(all_elements),
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
        ref_space_uid = cls.get_and_increase_uid_counter()
        # ref_struct['ref_space_uid'] = ref_struct['ref_mask'] * ref_space_uid
        ref_struct['ref_space_uid'] = torch.full_like(
            ref_struct['ref_mask'], ref_space_uid)

        for i, atom_name in enumerate(ref_struct['atom_names']):
            if atom_name in drop_atoms:
                ref_struct['ref_element'][i] = 0
                ref_struct['ref_mask'][i] = 0
                ref_struct['ref_charge'][i] = 0
                ref_struct['ref_pos'][i, :] = 0
                ref_struct['ref_atom_name_chars'][i, :] = 0
                ref_struct['atom_names'][i] = ''
                # ref_struct['ref_space_uid'][i] = 0

        filtered = {key: val for key, val in ref_struct.items()
                    if key != 'atom_names'}

        return filtered