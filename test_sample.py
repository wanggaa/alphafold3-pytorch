import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch

from alphafold3_pytorch.trainer import Trainer,DataLoader
from alphafold3_pytorch.model.alphafold3_bak import Alphafold3
from alphafold3_pytorch.inputs import PDBDataset
from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler

from functools import partial

from tree import map_structure

device = 'cuda' if torch.cuda.is_available else 'cpu'

def main():
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
        sampler=sampler, sample_type="default", crop_size=128,training=True
    )
    # train dataloader

    DataLoader_ = partial(DataLoader, atoms_per_window = 27)

    dataloader = DataLoader_(
        dataset,
        batch_size = 1,
        shuffle=True,
        drop_last=True
    )

    for data in dataloader:
        test_input = data
        break
    
    test_weights_path = 'test-folder/checkpoints/(s7qo)_af3.ckpt.130.pt'
    
    alphafold3 = Alphafold3(
        dim_atom_inputs=3,
        dim_atompair_inputs=5,
        atoms_per_window=27,
        dim_template_feats=44,
        num_dist_bins=38,
        confidence_head_kwargs=dict(pairformer_depth=1),
        template_embedder_kwargs=dict(pairformer_stack_depth=1),
        msa_module_kwargs=dict(depth=1),
        pairformer_stack=dict(depth=2),
        diffusion_module_kwargs=dict(
            atom_encoder_depth=1,
            token_transformer_depth=1,
            atom_decoder_depth=1,
        ),
        
        # jwang's debug parameters
        # dim_token=128,   
    )

    alphafold3.load(test_weights_path) 
    alphafold3 = alphafold3.to(device)
    alphafold3.eval()
    
    
    
    data_input = test_input.model_forward_dict()
    data_input = map_structure(lambda v:v.to(device) if torch.is_tensor(v) else v,data_input)
    
    del data_input['atom_inputs']
    data_input['atom_inputs'] = None
    
    r_ans = alphafold3(
        **data_input,
        return_loss=False,
        return_confidence_head_logits=True,
        return_distogram_head_logits=True,
    )
    
    print('test')
    
    

if __name__ == '__main__':
    main()