import copy
from pathlib import Path
from gemmi import cif
from pdbeccdutils.core import ccd_reader
from pdbeccdutils.core.models import ConformerType
import pickle

import rdkit
import tqdm

import residue_constants

def retrieve_cif_attribute(ccd_data, attr):
    block = ccd_data.ccd_cif_block
    pair_val = block.find_value(attr)
    if pair_val is not None:
        return cif.as_string(pair_val)

    loop = block.find_loop(attr)
    if loop is not None:
        return [cif.as_string(v) for v in loop]

    return None

def load_ccd_from_file():
    ccd_path = Path('data/ccd/components.cif')
    key_map = {
        '_chem_comp.type': 'type',
        '_chem_comp_atom.atom_id': 'atom_id',
        '_chem_comp_atom.type_symbol': 'atom_type',
        '_chem_comp_bond.atom_id_1': 'bond_atom_id_1',
        '_chem_comp_bond.atom_id_2': 'bond_atom_id_2',
    }

    keys_requiring_lists = [
       'atom_id',
       'atom_type',
       'bond_atom_id_1',
       'bond_atom_id_2'
    ]

    print('Reading component file...')
    ccd_full = ccd_reader.read_pdb_components_file(str(ccd_path))

    ccd = dict()
    print("Retrieving items from file...")
    for name, entry in tqdm.tqdm(ccd_full.items()):
        vals = {
            new_key: retrieve_cif_attribute(entry.component, key) for key, new_key in key_map.items()
        }
        for key in keys_requiring_lists:
            if not isinstance(vals[key], list):
                vals[key] = [vals[key]]
        try:
            vals['mol'] = entry.component.mol
        except:
            vals['conformer'] = None
        ccd[name] = vals

    missing_entries = [any(v is None for v in entry.values()) for entry in ccd.values()]
    print(f"Values missing entries: {missing_entries.count(True)/len(missing_entries)}")
    return ccd

def load_ccd():
    pickle_path = Path('data/ccd/ccd.pickle')

    if not pickle_path.exists():
        ccd = load_ccd_from_file()
        ccd = add_atoms_to_ccd(ccd)

        rdkit.Chem.SetDefaultPickleProperties(rdkit.Chem.PropertyPickleOptions.AllProps)
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(ccd, f)
    else:
        with open(pickle_path, 'rb') as f:
            ccd = pickle.load(f)

    return ccd

def drop_atoms(res_data, drop_hydrogens=True, drop_leaving_atoms=False):
    res_data = copy.deepcopy(res_data)
    atom_ids = res_data['atom_id']
    atom_types = res_data['atom_type']
    bond_id1 = res_data['bond_atom_id_1']
    bond_id2 = res_data['bond_atom_id_2']

    atoms_to_drop = []

    if drop_hydrogens:
        atoms_to_drop = [atom_ids[i] for i, v in enumerate(atom_types) if v=='H']

    if drop_leaving_atoms:
        is_saccharide = 'saccharide' in res_data['type'].lower()
        if is_saccharide and 'O1' in atom_ids:
            atoms_to_drop.append('O1')

    new_bond_ids = [(bid1, bid2) for bid1, bid2 in zip(bond_id1, bond_id2) if not (bid1 in atoms_to_drop or bid2 in atoms_to_drop)]
    new_atom_types = [atom_type for atom_id, atom_type in zip(atom_ids, atom_types) if not atom_id in atoms_to_drop]
    new_ids = [atom_id for atom_id in atom_ids if not atom_id in atoms_to_drop]

    res_data['bond_atom_id_1'] = [bid1 for bid1, _ in new_bond_ids]
    res_data['bond_atom_id_2'] = [bid2 for _, bid2 in new_bond_ids]
    res_data['atom_id'] = new_ids
    res_data['atom_type'] = new_atom_types

    return res_data    



def add_atoms_to_ccd(ccd):
    keys = (
      'atom_id',
      'atom_type',
      'bond_atom_id_1',
      'bond_atom_id_2',
    ) 

    print('Updating ccd entries...')
    for res_name, entry in tqdm.tqdm(ccd.items()):
        old_atom_ids = entry['atom_id']
        if len(old_atom_ids) == 0:
            continue

        for atom, atom_name in zip(entry['mol'].GetAtoms(), old_atom_ids):
            atom.SetProp('atom_name', atom_name)

        new_atoms = new_atom_types = new_bonds1 = new_bonds2 = []

        all_atom_ids = old_atom_ids + new_atoms
        entry['atom_id'] = all_atom_ids
        entry['atom_type'].extend(new_atom_types)
        entry['bond_atom_id_1'].extend(new_bonds1)
        entry['bond_atom_id_2'].extend(new_bonds2)

        rw_mol = rdkit.Chem.RWMol(entry['mol'])

        for atom_id, atom_type in zip(new_atoms, new_atom_types):
            atom = rdkit.Chem.Atom(atom_type)
            atom.SetProp('atom_name', atom_id)
            atom.SetFormalCharge(0)
            rw_mol.AddAtom(atom)

        for bond_id_1, bond_id_2 in zip(new_bonds1, new_bonds2):
            rw_mol.AddBond(all_atom_ids.index(bond_id_1), all_atom_ids.index(bond_id_2))

        atom_name_map = {name: i for i, name in enumerate(all_atom_ids)}
        sorted_name_map = sorted(atom_name_map.items())
        _, new_order = zip(*sorted_name_map)
        entry['mol'] = rdkit.Chem.RenumberAtoms(rw_mol, new_order)

    return ccd
        


def main():
    ccd = load_ccd()

if __name__=='__main__':
    main()