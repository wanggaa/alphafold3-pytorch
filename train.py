import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import tensorboard

from omegaconf import OmegaConf
from alphafold3_pytorch import Trainer
from alphafold3_pytorch import Alphafold3
from alphafold3_pytorch.inputs import PDBDataset
from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',default='wangjun')
    parser.add_argument('--gpu_num',type=int,default=2)    
    parser.add_argument('--ckpt_path',default=\
        '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/workspace/wangjun/alphafold3-pytorch/checkpoints/liuce/(ie9q)_af3.ckpt.6858.pt')
    
    args = parser.parse_args()
    
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
        sampler=sampler, sample_type="default", crop_size=128,training=True
    )

    conf = OmegaConf.load('tests/configs/alphafold3.yaml')
    print(conf)

    conf.dim_atom_inputs = 3
    conf.dim_template_feats = 44

    alphafold3 = Alphafold3(
        **conf
    )
    weights_path = args.ckpt_path
    
    alphafold3.load(weights_path) 
    
    checkpoint_folder = f'./checkpoints/{args.name}'
    if not os.path.exists(checkpoint_folder):
        os.mkdir(checkpoint_folder)
    
    # Trainer = None
    trainer = Trainer(
        alphafold3,
        dataset = dataset,
        valid_dataset = None,
        test_dataset = None,
        accelerator = 'auto',
        num_train_steps = 2000,
        batch_size = 1,
        valid_every = 1,
        grad_accum_every = 1,
        checkpoint_every = 1,
        checkpoint_folder = checkpoint_folder,
        overwrite_checkpoints = True,
        use_ema = False,
        ema_kwargs = dict(
            use_foreach = True,
            update_after_step = 0,
            update_every = 1
        ),
        fabric_kwargs={'devices':args.gpu_num,'strategy':'ddp'},
        # jwang's additional parameters
        epochs = 50000,
        )

    trainer()

if __name__ == '__main__':
    main()