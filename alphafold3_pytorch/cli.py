import click
from pathlib import Path

from tree import map_structure

import torch

from alphafold3_pytorch import (
    Alphafold3,
    Alphafold3Input,
    alphafold3_inputs_to_batched_atom_input
)

from Bio.PDB.mmcifio import MMCIFIO

# simple cli using click

@click.command()
@click.option('-ckpt', '--checkpoint', type = str, help = 'path to alphafold3 checkpoint')
@click.option('-p', '--protein', type = str, help = 'one protein sequence')
@click.option('-o', '--output', type = str, help = 'output path', default = 'output.cif')
def cli(
    checkpoint: str,
    protein: str,
    output: str
):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint_path = Path(checkpoint)
    assert checkpoint_path.exists(), f'AlphaFold 3 checkpoint must exist at {str(checkpoint_path)}'

    alphafold3_input = Alphafold3Input(
        proteins = [protein],
    )

    alphafold3 = Alphafold3.init_and_load(checkpoint_path)

    batched_atom_input = alphafold3_inputs_to_batched_atom_input(alphafold3_input, atoms_per_window = alphafold3.atoms_per_window)

    alphafold3 = alphafold3.to(device)
    alphafold3.eval()
    
    data_input = batched_atom_input.model_forward_dict()
    data_input = map_structure(lambda v:v.to(device) if torch.is_tensor(v) else v)
    
    structure, = alphafold3(**data_input.model_forward_dict(), num_sample_steps=1000, return_bio_pdb_structures = True)

    output_path = Path(output)
    output_path.parents[0].mkdir(exist_ok = True, parents = True)

    pdb_writer = MMCIFIO()
    pdb_writer.set_structure(structure)
    pdb_writer.save(str(output_path))

    print(f'mmCIF file saved to {str(output_path)}')
