import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch

import tensorboard

from omegaconf import OmegaConf

from alphafold3_pytorch.trainer import Trainer
from alphafold3_pytorch.alphafold3 import Alphafold3
from alphafold3_pytorch.inputs import PDBDataset
from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler
from alphafold3_pytorch import alphafold3_inputs_to_batched_atom_input
from alphafold3_pytorch.trainer import Trainer,DataLoader

from functools import partial
from tree import map_structure
from pathlib import Path

from Bio.PDB.mmcifio import MMCIFIO

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    data_test = os.path.join("data", "test")
    data_test = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/datasets/AF3/data/pdb_data/data_caches/200_mmcif'

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
        sampler=sampler, sample_type="default", crop_size=128, training=False
    )
    
    DataLoader_ = partial(DataLoader, atoms_per_window = 27)

    dataloader = DataLoader_(
        dataset,
        batch_size = 1,
        shuffle=False,
        drop_last=True
    )

    for data in dataloader:
        test_input = data
        break
    
    
    conf = OmegaConf.load('tests/configs/alphafold3.yaml')
    print(conf)

    conf.dim_atom_inputs = 3
    conf.dim_template_feats = 44
    # conf.num_molecule_mods = 0
    
    alphafold3 = Alphafold3(
        **conf
    )

    weights_path = 'test-folder/checkpoints/(4pi8)_af3.ckpt.130.pt'
    
    alphafold3.load(weights_path) 
    alphafold3 = alphafold3.to(device)
    alphafold3.eval()
    
    data_input = test_input.model_forward_dict()
    data_input = map_structure(lambda v:v.to(device) if torch.is_tensor(v) else v,data_input)
    
    # del(data_input['is_molecule_mod'])
    
    structure = r_ans = alphafold3.forward(
        **data_input,
        return_loss=False,
        return_confidence_head_logits=False,
        return_distogram_head_logits=False,
        num_sample_steps=1000,
        return_bio_pdb_structures=True,
    )
    
    output = 'output.cif'
    output_path = Path(output)
    output_path.parents[0].mkdir(exist_ok = True, parents = True)

    pdb_writer = MMCIFIO()
    pdb_writer.set_structure(structure[0])
    pdb_writer.save(str(output_path))
    
    print('test')
    
if __name__ == '__main__':
    main()
    
