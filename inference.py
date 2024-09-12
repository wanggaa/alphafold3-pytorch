import os
os.environ['TYPECHECK'] = 'True'
os.environ['DEBUG'] = 'True'
from shutil import rmtree
from pathlib import Path

import torch

from alphafold3_pytorch.cli import cli

from alphafold3_pytorch.alphafold3 import (
    Alphafold3
)

from omegaconf import OmegaConf
conf = OmegaConf.load('tests/configs/alphafold3.yaml')


conf.dim_atom_inputs=3
conf.dim_template_feats=44
conf.num_molecule_mods=0

print(conf)

device = 'cuda' if torch.cuda.is_available else 'cpu'
def test_cli():
    alphafold3 = Alphafold3(
        **conf
    )

    checkpoint_path = 'test-folder/checkpoints/(hc9n)_af3.ckpt.104.pt'
    alphafold3.save(checkpoint_path, overwrite = True)

    cli([
        '--checkpoint', checkpoint_path,
        '-prot', 'SALQDLLRTLKSPSSPQQQQQVLNILKSNPQLMAAFIKQRTAKYVAN',
        '-prot', 'SALQDLLRTLKSPSSPQQQQQVLNILKSNPQLMAAFIKQRTAKYVAN',
        '--output',
        './test-folder/output.mmcif'
        ], standalone_mode = False)

    # cli(['--checkpoint', checkpoint_path, 
    #      '--protein', 'SALQDLLRTLKSPSSPQQQQQVLNILKSNPQLMAAFIKQRTAKYVAN', 
    #      '--prot'
    #      '--output', './test-folder/output.cif',
    #        '--device', device,
    #        ], standalone_mode = False)

    # assert Path('./test-folder/output.pdb').exists()

    # rmtree('./test-folder')
test_cli()