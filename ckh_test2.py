import os

from alphafold3_pytorch import collate_inputs_to_batched_atom_input
from alphafold3_pytorch.alphafold3 import Alphafold3
from alphafold3_pytorch.inputs import (
    PDBDataset,
    molecule_to_atom_input,
    pdb_input_to_molecule_input,
)
from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler
import random
import numpy as np
import torch

def set_seed(seed):
    # Python的随机数生成器
    random.seed(seed)
    
    # Numpy的随机数生成器
    np.random.seed(seed)
    
    # PyTorch的CPU随机数生成器
    torch.manual_seed(seed)
    
    # 如果使用GPU，设置所有GPU的随机数生成器
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 为了确保生成的随机数是可预测的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子，例如42
set_seed(42)

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
    sampler=sampler, sample_type="default", crop_size=128
)
len(dataset)

from alphafold3_pytorch.utils.utils import default, exists, first
error_cnt=0
for i in range(len(dataset)):
    filepath = dataset[i].mmcif_filepath
    file_id = os.path.splitext(os.path.basename(filepath))[0] if exists(filepath) else None
    # if file_id != '4ni9-assembly1':
    #     continue
    # print(f"Processing:{file_id}")
    mol_input = pdb_input_to_molecule_input(pdb_input=dataset[i])
