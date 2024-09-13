import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import tensorboard

from omegaconf import OmegaConf
from alphafold3_pytorch.trainer import Trainer
from alphafold3_pytorch.alphafold3 import Alphafold3
from alphafold3_pytorch.inputs import PDBDataset
from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler

def main():
    data_test = os.path.join("data", "test")
    data_test = 'tests/data/200_mmcif'


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

    # alphafold3 = Alphafold3(
    #         dim_atom_inputs=3,
    #         dim_atompair_inputs=5,
    #         atoms_per_window=27,
    #         dim_template_feats=44,
    #         num_dist_bins=64,
    #         confidence_head_kwargs=dict(pairformer_depth=1),
    #         template_embedder_kwargs=dict(pairformer_stack_depth=1),
    #         msa_module_kwargs=dict(depth=1),
    #         pairformer_stack=dict(depth=2),
    #         diffusion_module_kwargs=dict(
    #             atom_encoder_depth=1,
    #             token_transformer_depth=1,
    #             atom_decoder_depth=1,
    #         ),
            
    #         # jwang's debug parameters
    #         # dim_token=128,
            
    #     )

    conf = OmegaConf.load('tests/configs/alphafold3.yaml')
    print(conf)

    conf.dim_atom_inputs = 3
    conf.dim_template_feats = 44

    alphafold3 = Alphafold3(
        **conf
    )

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
        checkpoint_folder = './test-folder/checkpoints',
        overwrite_checkpoints = True,
        use_ema = False,
        ema_kwargs = dict(
            use_foreach = True,
            update_after_step = 0,
            update_every = 1
        ),
        fabric_kwargs={'devices':2,'strategy':'ddp'},
        # jwang's additional parameters
        epochs = 50000,
        )

    trainer()

if __name__ == '__main__':
    main()