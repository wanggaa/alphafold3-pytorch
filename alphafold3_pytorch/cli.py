from __future__ import annotations

import click
from pathlib import Path
from tree import map_structure
import torch

from alphafold3_pytorch import (
    Alphafold3,
    Alphafold3Input
)

from Bio.PDB.mmcifio import MMCIFIO

# simple cli using click

@click.command()
@click.option('-ckpt', '--checkpoint', type = str, help = 'path to alphafold3 checkpoint')
@click.option('-prot', '--protein', type = str, multiple = True, help = 'protein sequences')
@click.option('-rna', '--rna', type = str, multiple = True, help = 'single stranded rna sequences')
@click.option('-dna', '--dna', type = str, multiple = True, help = 'single stranded dna sequences')
@click.option('-o', '--output', type = str, help = 'output path', default = 'output.cif')
@click.option('--device', type=str, default='cpu', help='Specify the device to use, e.g., cpu or gpu.')
def cli(
    checkpoint: str,
    protein: list[str],
    rna: list[str],
    dna: list[str],
    output: str,
    device
):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint_path = Path(checkpoint)
    assert checkpoint_path.exists(), f'AlphaFold 3 checkpoint must exist at {str(checkpoint_path)}'

    alphafold3_input = Alphafold3Input(
        proteins = protein,
        ss_rna = rna,
        ss_dna = dna,
    )

    alphafold3 = Alphafold3.init_and_load(checkpoint_path)

    alphafold3.eval()
    structure, = alphafold3.forward_with_alphafold3_inputs(alphafold3_input, return_bio_pdb_structures = True)

    output_path = Path(output)
    output_path.parents[0].mkdir(exist_ok = True, parents = True)

    pdb_writer = MMCIFIO()
    pdb_writer.set_structure(structure)
    pdb_writer.save(str(output_path))

    print(f'mmCIF file saved to {str(output_path)}')
