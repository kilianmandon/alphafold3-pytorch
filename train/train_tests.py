import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from atomworks import parse
from atomworks.constants import STANDARD_AA
from atomworks.enums import ChainType
from atomworks.io.parser import parse_atom_array
from atomworks.io.tools.inference import components_to_atom_array
from atomworks.io.utils.visualize import view
from atomworks.ml.datasets import FileDataset, StructuralDatasetWrapper, ConcatDatasetWithID, PandasDataset
from atomworks.ml.datasets.loaders import create_loader_with_query_pn_units
from atomworks.ml.datasets.parsers import PNUnitsDFParser, InterfacesDFParser
from atomworks.ml.pipelines.af3 import build_af3_transform_pipeline
from atomworks.ml.samplers import calculate_af3_example_weights, get_cluster_sizes
from atomworks.ml.transforms.atom_array import AddGlobalAtomIdAnnotation
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.crop import CropSpatialLikeAF3
from torch.utils.data import WeightedRandomSampler


def simple_loading_fn(raw_data):
    parse_output = parse(raw_data)
    return {'atom_array': parse_output['assemblies']['1'][0]}

def af3_dataset():
    datasets = [
        # Single PN units
        PandasDataset(
            name='pn_units',
            id_column='example_id',
            data='data/pdb_metadata/pn_units_df_fixed_short.parquet',
            loader=create_loader_with_query_pn_units(pn_unit_iid_colnames='q_pn_unit_iid', base_path='../data/pdb_mirror',
                                                     extension='.cif.gz', sharding_pattern='/1:3/'),
            filters=[
                "deposition_date < '2022-01-01'",
                "resolution < 5.0 and ~method.str.contains('NMR')",
                "num_polymer_pn_units <= 20",
                "cluster.notnull()",
                "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']",
                # Train only on D-polypeptides:
                "q_pn_unit_type in [5, 6]",  # 5 = POLYPEPTIDE_D, 6 = POLYPEPTIDE_L
                # Exclude ligands from AF3 excluded set:
                "~(q_pn_unit_non_polymer_res_names.notnull() and q_pn_unit_non_polymer_res_names.str.contains('${af3_excluded_ligands_regex}', regex=True))",
            ],
            transform=build_af3_transform_pipeline(
                is_inference=False,
                n_recycles=5,  # This means that we will subsample 5 random sets from the MSA for each example.
                crop_size=256,
                crop_contiguous_probability=0.3333333333333333,
                crop_spatial_probability=0.6666666666666666,
                diffusion_batch_size=32,
                protein_msa_dirs=[],
                rna_msa_dirs=[]
            ),
            save_failed_examples_to_dir=None
        ),
        # Binary interfaces
        PandasDataset(
            name='interfaces',
            id_column='example_id',
            data=Path('../data/pdb_metadata/interfaces_df_fixed_short.parquet'),
            loader=create_loader_with_query_pn_units(pn_unit_iid_colnames=["pn_unit_1_iid", "pn_unit_2_iid"],
                                                     base_path='../data/pdb_mirror', extension='.cif.gz',
                                                     sharding_pattern='/1:3/'),
            transform=build_af3_transform_pipeline(
                is_inference=False,
                n_recycles=5,
                crop_size=256,
                crop_spatial_probability=1.0,
                crop_contiguous_probability=0.0,
                diffusion_batch_size=32,
                protein_msa_dirs=[],
                rna_msa_dirs=[],
            ),
            filters=[
                "deposition_date < '2022-01-01'",
                "resolution < 5.0 and ~method.str.contains('NMR')",
                "num_polymer_pn_units <= 20",
                "cluster.notnull()",
                "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']",
                # Train only on D-polypeptide interfaces:
                "pn_unit_1_type in [5, 6]",  # 5 = POLYPEPTIDE_D, 6 = POLYPEPTIDE_L
                "pn_unit_2_type in [5, 6]",  # 5 = POLYPEPTIDE_D, 6 = POLYPEPTIDE_L
                "~(pn_unit_1_non_polymer_res_names.notnull() and pn_unit_1_non_polymer_res_names.str.contains('${af3_excluded_ligands_regex}', regex=True))",
                "~(pn_unit_2_non_polymer_res_names.notnull() and pn_unit_2_non_polymer_res_names.str.contains('${af3_excluded_ligands_regex}', regex=True))"
            ],
            save_failed_examples_to_dir=None
        )
    ]

    af3_pdb_dataset = ConcatDatasetWithID(datasets)
    return af3_pdb_dataset


def fix_datasets():
    int_table = pq.read_table('data/pdb_metadata/interfaces_df.parquet')
    pn_table = pq.read_table('data/pdb_metadata/pn_units_df.parquet')

    def replace_col(table: pa.Table):
        path_idx = table.column_names.index('path')
        shortened = table.remove_column(path_idx)
        refilled = shortened.add_column(path_idx, pa.field('path', pa.string()), shortened['pdb_id'])
        return refilled

    int_table = replace_col(int_table)
    pn_table = replace_col(pn_table)

    int_table_short = int_table.slice(0, 1000)
    pn_table_short = pn_table.slice(0, 1000)

    pq.write_table(int_table, Path('../data/pdb_metadata/interfaces_df_fixed.parquet'))
    pq.write_table(int_table_short, Path('../data/pdb_metadata/interfaces_df_fixed_short.parquet'))

    pq.write_table(pn_table, Path('../data/pdb_metadata/pn_units_df_fixed.parquet'))
    pq.write_table(pn_table_short, Path('../data/pdb_metadata/pn_units_df_fixed_short.parquet'))

def build_sampler(dataset):
    for ds in dataset.datasets:
        cluster_id_to_size_map = get_cluster_sizes(ds.data, cluster_column='cluster')
        ds.data['cluster_size'] = ds.data['cluster'].map(cluster_id_to_size_map)

    alphas = {
        "a_prot": 3,
        # Choosing same as for protein,
        # but atomworks says peptides were oversampled in AF3
        "a_peptide": 3,
        "a_nuc": 3,
        "a_ligand": 1,
        "a_loi": 0
    }
    beta_chain = 0.5
    beta_interface = 1

    weights_chains = calculate_af3_example_weights(dataset.datasets[0].data, alphas, beta_chain)
    weights_interfaces = calculate_af3_example_weights(dataset.datasets[1].data, alphas, beta_interface)
    weights = np.concatenate([weights_chains.to_numpy(), weights_interfaces.to_numpy()])

    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler


def main():
    dataset = af3_dataset()
    with open('../data/fold_inputs/fold_input_multimer.json', 'r') as f:
        data = json.load(f)

    # components = af3_to_rf3_components(data['sequences'])
    # atom_array = components_to_atom_array(components)
    # chain_data = parse_atom_array(atom_array)
    # atom_array = chain_data['assemblies']['1'][0]
    # example_id = data['name']

    transform = build_af3_transform_pipeline(
        is_inference=True,
        n_recycles=5,
        crop_size=256,
        crop_spatial_probability=1.0,
        crop_contiguous_probability=0.0,
        diffusion_batch_size=32,
        protein_msa_dirs=[],
        rna_msa_dirs=[],
    )

    # dataset.datasets[0].transform = lambda x: x
    test_sample = dataset[0]

    ...


if __name__ == '__main__':
    main()
