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

import shutil

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

def main():
    data_test = os.path.join("data", "test")
    data_test = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/workspace/chenbaoyou/datasets/train_2k_mmcifs'

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
   
    dataloader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle=False,
        drop_last=True,
        atoms_per_window = 27
    )
        
    conf = OmegaConf.load('tests/configs/alphafold3.yaml')
    print(conf)

    conf.dim_atom_inputs = 3
    conf.dim_template_feats = 44
    # conf.num_molecule_mods = 0
    
    alphafold3 = Alphafold3(
        **conf
    )
    
    weights_path = 'checkpoints/wangjun2/(jyjx)_af3.ckpt.254.pt'
    
    alphafold3.load(weights_path) 
    alphafold3 = alphafold3.to(device)
    alphafold3.eval()

    skip_not_protein = True

    for iter_num,data in enumerate(dataloader):
        if iter_num > 200:
            break
        data_input = data.model_forward_dict()
        data_input = map_structure(lambda v:v.to(device) if torch.is_tensor(v) else v,data_input)
        print(data.filepath)
        
        # if data_input['molecule_ids'].max() >= 20:
        #     if skip_not_protein:
        #         continue
        
        try:
            structure = r_ans = alphafold3.forward(
                **data_input,
                return_loss=False,
                return_confidence_head_logits=False,
                return_distogram_head_logits=False,
                num_sample_steps=1000,
                return_bio_pdb_structures=True,
            )
            
            shutil.copy(data.filepath[0],'output')
            
            output = f'output/{os.path.basename(data.filepath[0])[:4]}_predict.cif'
            output_path = Path(output)
            output_path.parents[0].mkdir(exist_ok = True, parents = True)

            pdb_writer = MMCIFIO()
            pdb_writer.set_structure(structure[0])
            pdb_writer.save(str(output_path))
        except Exception as e:
            print('error')
            print(e)
            print(data_input['molecule_ids'])
            
if __name__ == '__main__':
    main()
    
