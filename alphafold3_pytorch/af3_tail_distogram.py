import einops.layers
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.checkpoint as tckpt

from torch import Tensor

from af3_utils import exists,default,identity

from af3_embed import AF3Embed
from af3_trunk import AF3Trunk
from af3_utils import lens_to_mask,pack_one,to_pairwise_mask

from af3_basic import LinearNoBias

import einops

from alphafold3_pytorch.utils.model_utils import distance_to_dgram
from alphafold3 import LinearNoBiasThenOuterSum

class DistogramHead(nn.Module):

    # @typecheck
    def __init__(
        self,
        *,
        dim_pairwise = 128,
        num_dist_bins = 64,
        dim_atom = 128,
        atom_resolution = False,
        checkpoint = False,
    ):
        super().__init__()

        self.to_distogram_logits = nn.Sequential(
            LinearNoBias(dim_pairwise, num_dist_bins),
            einops.layers.torch.Rearrange('b ... l -> b l ...')
        )

        # atom resolution
        # for now, just embed per atom distances, sum to atom features, project to pairwise dimension

        self.atom_resolution = atom_resolution

        if atom_resolution:
            self.atom_feats_to_pairwise = LinearNoBiasThenOuterSum(dim_atom, dim_pairwise)

        # checkpointing

        self.checkpoint = checkpoint

        # tensor typing

        self.da = dim_atom

    # @typecheck
    def to_layers(
        self,
        pairwise_repr: Tensor,  # type: ignore
        molecule_atom_lens: Tensor | None = None,  # type: ignore
        atom_feats: Tensor | None = None,  # type: ignore
    ) -> Tensor | Tensor:  # type: ignore
        """Compute the distogram logits.

        :param pairwise_repr: The pairwise representation tensor.
        :param molecule_atom_lens: The molecule atom lengths tensor.
        :param atom_feats: The atom features tensor.
        :return: The distogram logits.
        """
        if self.atom_resolution:
            assert exists(molecule_atom_lens)
            assert exists(atom_feats)

            pairwise_repr = batch_repeat_interleave_pairwise(pairwise_repr, molecule_atom_lens)

            pairwise_repr = pairwise_repr + self.atom_feats_to_pairwise(atom_feats)

        logits = self.to_distogram_logits(symmetrize(pairwise_repr))

        return logits

    # @typecheck
    def to_checkpointed_layers(
        self,
        pairwise_repr: Tensor,  # type: ignore
        molecule_atom_lens: Tensor | None = None,  # type: ignore
        atom_feats: Tensor | None = None,  # type: ignore
    ) -> Tensor | Tensor:  # type: ignore
        """Compute the checkpointed distogram logits.

        :param pairwise_repr: The pairwise representation tensor.
        :param molecule_atom_lens: The molecule atom lengths tensor.
        :param atom_feats: The atom features tensor.
        :return: The checkpointed distogram logits.
        """
        wrapped_layers = []
        inputs = (pairwise_repr, molecule_atom_lens, atom_feats)

        def atom_resolution_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                pairwise_repr, molecule_atom_lens, atom_feats = inputs

                assert exists(molecule_atom_lens)
                assert exists(atom_feats)

                pairwise_repr = batch_repeat_interleave_pairwise(pairwise_repr, molecule_atom_lens)

                pairwise_repr = pairwise_repr + fn(atom_feats)
                return pairwise_repr, molecule_atom_lens, atom_feats

            return inner

        def distogram_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                pairwise_repr, molecule_atom_lens, atom_feats = inputs
                pairwise_repr = fn(symmetrize(pairwise_repr))
                return pairwise_repr, molecule_atom_lens, atom_feats

            return inner

        if self.atom_resolution:
            wrapped_layers.append(atom_resolution_wrapper(self.atom_feats_to_pairwise))
        wrapped_layers.append(distogram_wrapper(self.to_distogram_logits))

        for layer in wrapped_layers:
            inputs = checkpoint(layer, inputs)

        logits, *_ = inputs
        return logits

    # @typecheck
    def forward(
        self,
        pairwise_repr: Tensor,  # type: ignore
        molecule_atom_lens: Tensor | None = None,  # type: ignore
        atom_feats: Tensor | None = None,  # type: ignore
    ) -> Tensor | Tensor:  # type: ignore
        """Compute the distogram logits.
        
        :param pairwise_repr: The pairwise representation tensor.
        :param molecule_atom_lens: The molecule atom lengths tensor.
        :param atom_feats: The atom features tensor.
        :return: The distogram logits.
        """
        # going through the layers

        if should_checkpoint(self, pairwise_repr):
            to_layers_fn = self.to_checkpointed_layers
        else:
            to_layers_fn = self.to_layers

        logits = to_layers_fn(
            pairwise_repr=pairwise_repr,
            molecule_atom_lens=molecule_atom_lens,
            atom_feats=atom_feats,
        )

        return logits

def compute_distogram(
    atom_pos,
    atom_mask,
    molecule_atom_lens: Tensor,
    
    distance_bins,
    distogram_atom_indices,
    use_distogram_atom_resolution,
    
    ignore_index,
    **kwargs,
    ):
    
    distogram_pos = atom_pos
    
    valid_molecule_atom_mask = valid_atom_len_mask = molecule_atom_lens >= 0
    molecule_atom_lens = molecule_atom_lens.masked_fill(~valid_atom_len_mask, 0)
    valid_molecule_atom_mask = valid_atom_len_mask = molecule_atom_lens >= 0
    
    if exists(distogram_atom_indices):
        valid_distogram_mask = distogram_atom_indices >= 0 & valid_atom_len_mask
        distogram_atom_indices = distogram_atom_indices.masked_fill(~valid_distogram_mask, 0)
    
    if not use_distogram_atom_resolution:
        # molecule_pos = einx.get_at('b [m] c, b n -> b n c', atom_pos, distogram_atom_indices)

        distogram_atom_coords_indices = einops.repeat(
            distogram_atom_indices, "b n -> b n c", c=distogram_pos.shape[-1]
        )
        molecule_pos = distogram_pos = distogram_pos.gather(
            1, distogram_atom_coords_indices
        )
        distogram_mask = valid_distogram_mask
        
    else:
        distogram_mask = atom_mask

    distogram_dist = torch.cdist(distogram_pos, distogram_pos, p=2)
    distance_labels = distance_to_dgram(
        distogram_dist, distance_bins, return_labels = True
    )

    # account for representative distogram atom missing from residue (-1 set on distogram_atom_indices field)
    distogram_mask = to_pairwise_mask(distogram_mask)
    distance_labels.masked_fill_(~distogram_mask, ignore_index)
    return distance_labels

class DistogramTail(nn.Module):
    def __init__(
        self,
        distance_bins,
        distogram_atom_resolution,
        distogram_head_kwargs = {
            dim_pairwise = dim_pairwise,
            dim_atom = dim_atom,
            num_dist_bins = num_dist_bins,
            atom_resolution = distogram_atom_resolution,
            checkpoint = checkpoint_distogram_head
        }
        
        
        ):
        super().__init__()
        self.ignore_index = -1
        self.distance_bins # 使用原子级别的分辨率
        
        distance_bins_tensor = torch.tensor(distance_bins)
        self.register_buffer('distance_bins', distance_bins_tensor)
        num_dist_bins = default(num_dist_bins, len(distance_bins_tensor))
        
        self.distogram_head = DistogramHead(**distogram_head_kwargs)
    
        self.distance_bins = None
        self.register_buffer('distance_bins', torch.tensor(distance_bins))
    
    # main use distance_lable
    def forward(
        self,
        atom_pos,
        atom_feats,
        atom_mask,
        
        pairwise,
        pairwise_mask,
        
        distance_labels,
        distogram_atom_indices,
        molecule_atom_lens,
        ):
        

        if not exists(distance_labels):
            compute_distogram()


        if exists(distance_labels):

            distogram_mask = pairwise_mask

            if distogram_atom_resolution:
                distogram_mask = to_pairwise_mask(atom_mask)

            distance_labels = torch.where(distogram_mask, distance_labels, self.ignore_index)

            distogram_logits = self.distogram_head(
                pairwise,
                molecule_atom_lens = molecule_atom_lens,
                atom_feats = atom_feats
            )

            distogram_loss = F.cross_entropy(distogram_logits, distance_labels, ignore_index = self.ignore_index)

        pass
        
    

if __name__ == '__main__':
    from af3_debug import rebuild_inputdata_by_functions
    main_dir = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/workspace/wangjun/alphafold3-pytorch'
    
    import os
    config_path = os.path.join(main_dir,'configs/af3_test.yml')
    from omegaconf import OmegaConf
    conf = OmegaConf.load(config_path)

    from af3_embed import AF3Embed
    embed_model = AF3Embed(**conf.embed)

    import pickle
    input_data_path = os.path.join(main_dir,'.tmp/debug_data/temp.pkl')
    with open(input_data_path,'rb') as f:
        input_data = pickle.load(f)
    
    import tree
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print('embed part start')
    input_data = tree.map_structure(lambda x: x.to(device) if torch.is_tensor(x) else identity(x),input_data)
    
    embed_model = embed_model.to(device)
    
    # clear input data
    data = rebuild_inputdata_by_functions(input_data,AF3Embed.forward)
    
    embed_init = embed_model.forward(**data)
    for k,v in embed_init.items():
        print(k,v.shape)
        
    num_parameters = sum(p.numel() for p in embed_model.parameters())
    print(num_parameters)
    
    print('embed part over')
    print('-----------------------------')
    print('trunk part start')
    trunk_model = AF3Trunk(**conf.trunk)
    trunk_model = trunk_model.to(device)
    
    input_data.update(embed_init)
    trunk_forward_kwargs = {
        'num_recycling_steps': 8,
        'detach_when_recycling': True
    }
    input_data.update(trunk_forward_kwargs)
    
    data = rebuild_inputdata_by_functions(input_data,AF3Trunk.forward)
    
    r_ans = trunk_model.forward(**data)
    for k,v in r_ans.items():
        print(k,v.shape)
    # 
    num_parameters = sum(p.numel() for p in trunk_model.parameters())
    print(num_parameters)
        
    print('trunk part over')
    
    print('test')