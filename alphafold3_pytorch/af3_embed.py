import torch
import torch.nn as nn

import einx
import einops

from af3_utils import exists,default,identity
from af3_utils import lens_to_mask,pack_one,to_pairwise_mask
from af3_basic import LinearNoBias

from attention import full_pairwise_repr_to_windowed

# todo: still use af3 code here, should be seperate later
from alphafold3 import InputFeatureEmbedder
from alphafold3 import RelativePositionEncoding

IS_BIOMOLECULE_INDICES = slice(0, 3)

class AF3Embed(nn.Module):
    def __init__(
        self,
        *,
        dim_atom_inputs,        
        dim_atom,
        
        dim_atompair_inputs,
        dim_atompair,
        
        dim_input_embedder_token,
        dim_additional_token_feats,
        
        atoms_per_window,
        
        num_atom_embeds,
        
        num_atompair_embeds,
        
        num_molecule_mods,
        num_molecule_types,
        
        dim_single,
        dim_pairwise,
        
        input_embedder_kwargs,
        relative_position_encoding_kwargs
        
        ):
        super().__init__()
        
        self.dim_atom_inputs = dim_atom_inputs
        self.dim_atompair_inputs = dim_atompair_inputs
        
        has_atom_embeds = num_atom_embeds > 0
        if has_atom_embeds:
            self.atom_embeds = nn.Embedding(num_atom_embeds, dim_atom)
        self.has_atom_embeds = has_atom_embeds

        has_atompair_embeds = num_atompair_embeds > 0
        if has_atompair_embeds:
            self.atompair_embeds = nn.Embedding(num_atompair_embeds, dim_atompair)
        self.has_atompair_embeds = has_atompair_embeds
        
        num_molecule_mods = default(num_molecule_mods, 0)
        has_molecule_mod_embeds = num_molecule_mods > 0
        if has_molecule_mod_embeds:
            self.molecule_mod_embeds = nn.Embedding(num_molecule_mods, dim_single)
        self.has_molecule_mod_embeds = has_molecule_mod_embeds
    
        self.input_embedder = InputFeatureEmbedder(
            num_molecule_types = num_molecule_types,
            dim_atom_inputs = dim_atom_inputs,
            dim_atompair_inputs = dim_atompair_inputs,
            atoms_per_window = atoms_per_window,
            dim_atom = dim_atom,
            dim_atompair = dim_atompair,
            dim_token = dim_input_embedder_token,
            dim_single = dim_single,
            dim_pairwise = dim_pairwise,
            dim_additional_token_feats = dim_additional_token_feats,
            **input_embedder_kwargs
        )

        dim_single_inputs = dim_input_embedder_token + dim_additional_token_feats
        
        self.relative_position_encoding = RelativePositionEncoding(
            dim_out = dim_pairwise,
            **relative_position_encoding_kwargs
        )
    
        self.token_bond_to_pairwise_feat = nn.Sequential(
            einops.layers.torch.Rearrange('... -> ... 1'),
            LinearNoBias(1, dim_pairwise)
        )
    
    def forward(
        self,
        atom_inputs,
        atom_ids,
        atom_mask,
        
        atompair_inputs,
        atompair_ids,
        valid_atom_indices_for_frame,
        
        token_bonds,
        additional_token_feats,
        
        molecule_atom_lens,
        molecule_atom_indices,
        molecule_ids,
        is_molecule_mod,
        is_molecule_types,
        additional_molecule_feats,
        
        distogram_atom_indices,
        atom_indices_for_frame,
        ):
        
        device = atom_inputs.device
        
        atom_seq_len = atom_inputs.shape[-2]
        single_structure_input = atom_inputs.shape[0] == 1

        dtype = atom_inputs.dtype

        # validate atom and atompair input dimensions

        assert atom_inputs.shape[-1] == self.dim_atom_inputs, f'expected {self.dim_atom_inputs} for atom_inputs feature dimension, but received {atom_inputs.shape[-1]}'
        assert atompair_inputs.shape[-1] == self.dim_atompair_inputs, f'expected {self.dim_atompair_inputs} for atompair_inputs feature dimension, but received {atompair_inputs.shape[-1]}'

        # soft validate

        valid_molecule_atom_mask = valid_atom_len_mask = molecule_atom_lens >= 0

        molecule_atom_lens = molecule_atom_lens.masked_fill(~valid_atom_len_mask, 0)

        if exists(molecule_atom_indices):
            valid_molecule_atom_mask = molecule_atom_indices >= 0 & valid_atom_len_mask
            molecule_atom_indices = molecule_atom_indices.masked_fill(~valid_molecule_atom_mask, 0)

        if exists(distogram_atom_indices):
            valid_distogram_mask = distogram_atom_indices >= 0 & valid_atom_len_mask
            distogram_atom_indices = distogram_atom_indices.masked_fill(~valid_distogram_mask, 0)

        if exists(atom_indices_for_frame):
            valid_atom_indices_for_frame = default(valid_atom_indices_for_frame, torch.ones_like(molecule_atom_lens).bool())

            valid_atom_indices_for_frame = valid_atom_indices_for_frame & (atom_indices_for_frame >= 0).all(dim = -1) & valid_atom_len_mask
            atom_indices_for_frame = einx.where('b n, b n three, -> b n three', valid_atom_indices_for_frame, atom_indices_for_frame, 0)

        assert exists(molecule_atom_lens) or exists(atom_mask)

        # if atompair inputs are not windowed, window it

        is_atompair_inputs_windowed = atompair_inputs.ndim == 5

        if not is_atompair_inputs_windowed:
            atompair_inputs = full_pairwise_repr_to_windowed(atompair_inputs, window_size = self.atoms_per_window)

        # handle atom mask

        total_atoms = molecule_atom_lens.sum(dim = -1)
        atom_mask = lens_to_mask(total_atoms, max_len = atom_seq_len)

        # get atom sequence length and molecule sequence length depending on whether using packed atomic seq

        seq_len = molecule_atom_lens.shape[-1]

        # embed inputs

        (
            single_inputs,
            single_init,
            pairwise_init,
            atom_feats,
            atompair_feats
        ) = self.input_embedder(
            atom_inputs = atom_inputs,
            atompair_inputs = atompair_inputs,
            atom_mask = atom_mask,
            additional_token_feats = additional_token_feats,
            molecule_atom_lens = molecule_atom_lens,
            molecule_ids = molecule_ids
        )

        # handle maybe atom and atompair embeddings
        # 都有，或者都没有
        assert not (exists(atom_ids) ^ self.has_atom_embeds), 'you either set `num_atom_embeds` and did not pass in `atom_ids` or vice versa'
        assert not (exists(atompair_ids) ^ self.has_atompair_embeds), 'you either set `num_atompair_embeds` and did not pass in `atompair_ids` or vice versa'

        if self.has_atom_embeds:
            atom_embeds = self.atom_embeds(atom_ids)
            atom_feats = atom_feats + atom_embeds

        if self.has_atompair_embeds:
            atompair_embeds = self.atompair_embeds(atompair_ids)

            if atompair_embeds.ndim == 4:
                atompair_embeds = full_pairwise_repr_to_windowed(atompair_embeds, window_size = self.atoms_per_window)

            atompair_feats = atompair_feats + atompair_embeds

        # handle maybe molecule modifications

        assert not (exists(is_molecule_mod) ^ self.has_molecule_mod_embeds), 'you either set `num_molecule_mods` and did not pass in `is_molecule_mod` or vice versa'

        if self.has_molecule_mod_embeds:
            single_init, seq_unpack_one = pack_one(single_init, '* ds')

            is_molecule_mod, _ = pack_one(is_molecule_mod, '* mods')

            if not is_molecule_mod.is_sparse:
                is_molecule_mod = is_molecule_mod.to_sparse()

            seq_indices, mod_id = is_molecule_mod.indices()
            scatter_values = self.molecule_mod_embeds(mod_id)

            seq_indices = einops.repeat(seq_indices, 'n -> n ds', ds = single_init.shape[-1])
            single_init = single_init.scatter_add(0, seq_indices, scatter_values)

            single_init = seq_unpack_one(single_init)

        # relative positional encoding

        relative_position_encoding = self.relative_position_encoding(
            additional_molecule_feats = additional_molecule_feats
        )

        # only apply relative positional encodings to biomolecules that are chained
        # not to ligands + metal ions

        is_chained_biomol = is_molecule_types[..., IS_BIOMOLECULE_INDICES].any(dim = -1) # first three types are chained biomolecules (protein, rna, dna)
        paired_is_chained_biomol = to_pairwise_mask(is_chained_biomol)

        relative_position_encoding = einx.where(
            'b i j, b i j d, -> b i j d',
            paired_is_chained_biomol, relative_position_encoding, 0.
        )

        # add relative positional encoding to pairwise init

        pairwise_init = pairwise_init + relative_position_encoding

        # token bond features

        if exists(token_bonds):
            # well do some precautionary standardization
            # (1) mask out diagonal - token to itself does not count as a bond
            # (2) symmetrize, in case it is not already symmetrical (could also throw an error)

            token_bonds = token_bonds | einops.rearrange(token_bonds, 'b i j -> b j i')
            diagonal = torch.eye(seq_len, device = device, dtype = torch.bool)
            token_bonds = token_bonds.masked_fill(diagonal, False)
        else:
            seq_arange = torch.arange(seq_len, device = device)
            token_bonds = einx.subtract('i, j -> i j', seq_arange, seq_arange).abs() == 1

        token_bonds_feats = self.token_bond_to_pairwise_feat(token_bonds.type(dtype))

        pairwise_init = pairwise_init + token_bonds_feats

        # molecule mask and pairwise mask

        single_mask = molecule_atom_lens > 0
        pairwise_mask = to_pairwise_mask(single_mask)

        r_ans = {
            's_init':single_init,
            's_mask':single_mask,
            'z_init':pairwise_init,
            'z_mask':pairwise_mask
        }

        return r_ans
    
# common test script
if __name__ == '__main__':
    main_dir = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/workspace/wangjun/alphafold3-pytorch'
    
    import os
    config_path = os.path.join(main_dir,'configs/af3_test.yml')
    from omegaconf import OmegaConf
    conf = OmegaConf.load(config_path)
    
    embed_model = AF3Embed(**conf.embed)
    
    import pickle
    input_data_path = os.path.join(main_dir,'.tmp/debug_data/temp.pkl')
    with open(input_data_path,'rb') as f:
        input_data = pickle.load(f)
    
    import tree
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    input_data = tree.map_structure(lambda x: x.to(device) if torch.is_tensor(x) else identity(x),input_data)
    embed_model = embed_model.to(device)
    
    # clear input data
    import inspect
    sig = inspect.signature(AF3Embed.forward)
    function_kwargs = set(sig.parameters)
    function_kwargs.discard('self')
    input_data_kwargs = set(input_data.keys())
    
    for kw in function_kwargs.difference(input_data_kwargs):
        input_data[kw] = None
    for kw in input_data_kwargs.difference(function_kwargs):
        del input_data[kw]
    
    r_ans = embed_model.forward(**input_data)
    for k,v in r_ans.items():
        print(k,v.shape)
    print('test')