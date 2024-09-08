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
device = 'cuda' if torch.cuda.is_available else 'cpu'
def test_cli():
    alphafold3 = Alphafold3(
        dim_atom_inputs = 3,
        dim_template_feats = 44,
        num_molecule_mods = 0
    )

    checkpoint_path = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/workspace/chengkaihui/code/AF3/alphafold3-pytorch/test-folder/checkpoints/(ffpy)_af3.ckpt.26.pt'
    alphafold3.save(checkpoint_path, overwrite = True)

    cli(['--checkpoint', checkpoint_path, 
         '--protein', 'SALQDLLRTLKSPSSPQQQQQVLNILKSNPQLMAAFIKQRTAKYVAN', 
         '--output', './test-folder/output.cif',
           '--device', device,
           ], standalone_mode = False)

    # assert Path('./test-folder/output.pdb').exists()

    # rmtree('./test-folder')
test_cli()