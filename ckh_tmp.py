import os
from alphafold3_pytorch.utils.utils import default, exists, first
from alphafold3_pytorch import collate_inputs_to_batched_atom_input
from alphafold3_pytorch.alphafold3 import Alphafold3
from alphafold3_pytorch.inputs import (
    PDBDataset,
    molecule_to_atom_input,
    pdb_input_to_molecule_input,
)
from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler
# data_test = os.path.join("data", "test")
data_test = os.path.join('/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/datasets/AF3/data/pdb_data/data_caches/200_mmcif')

"""Test a PDBDataset constructed using a WeightedPDBSampler."""
interface_mapping_path = os.path.join(data_test, "interface_cluster_mapping.csv")
chain_mapping_paths = [
    os.path.join(data_test, "ligand_chain_cluster_mapping.csv"),
    os.path.join(data_test, "nucleic_acid_chain_cluster_mapping.csv"),
    os.path.join(data_test, "peptide_chain_cluster_mapping.csv"),
    os.path.join(data_test, "protein_chain_cluster_mapping.csv"),
]

sampler = WeightedPDBSampler(
    chain_mapping_paths=chain_mapping_paths,
    interface_mapping_path=interface_mapping_path,
    batch_size=1,
)

dataset = PDBDataset(
    folder=os.path.join("/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/datasets/AF3/data/pdb_data", "merged_mmcifs"), 
    sampler=sampler, sample_type="default", crop_size=128,training=True
)
len(dataset)

atom_num_cut_off = 2000
wanted_file_id = []
error_cnt=0
for i in range(len(dataset)):
    data = dataset[i]
    filepath = data.mmcif_filepath
    file_id = os.path.splitext(os.path.basename(filepath))[0] if exists(filepath) else None
    # if file_id !='2mtz-assembly1':
    #     continue
    # mol_input = pdb_input_to_molecule_input(pdb_input=data)
    # atom_input = molecule_to_atom_input(mol_input)
    # batched_atom_input = collate_inputs_to_batched_atom_input([atom_input], atoms_per_window=27)

    # print(f"pass data:{i} | {file_id}",len(mol_input.molecules),len(atom_input.atom_inputs))
    try:
        mol_input = pdb_input_to_molecule_input(pdb_input=data)
        atom_input = molecule_to_atom_input(mol_input)
        batched_atom_input = collate_inputs_to_batched_atom_input([atom_input], atoms_per_window=27)

        print(f"pass data:{i} | {file_id}",len(mol_input.molecules),len(atom_input.atom_inputs))

        if len(atom_input.atom_inputs)<atom_num_cut_off:
            wanted_file_id.append(file_id)
            
    except Exception as e:
        print(f"Error in {i}:{file_id}")
        print(f'Exception: {e}')
print(wanted_file_id)