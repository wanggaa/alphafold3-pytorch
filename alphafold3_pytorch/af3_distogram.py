import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.checkpoint as tckpt

from torch import Tensor

from typing import Tuple
from functools import wraps

from af3_utils import exists,default,identity,symmetrize

from af3_embed import AF3Embed
from af3_trunk import AF3Trunk
from af3_utils import lens_to_mask,pack_one,to_pairwise_mask,distance_to_dgram

from af3_basic import LinearNoBias

import einops

from alphafold3 import batch_repeat_interleave_pairwise
from alphafold3 import LinearNoBiasThenOuterSum,DistogramHead
  
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

class DistogramHead(nn.Module):

    # @typecheck
    def __init__(
        self,
        *,
        dim_pairwise = 128,
        num_dist_bins = 64,
        dim_atom = 128,
        atom_resolution = False,
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

    def forward(
        self,
        pairwise_repr: Tensor,  
        molecule_atom_lens: Tensor | None = None,  
        atom_feats: Tensor | None = None,
    ) -> Tensor:  
        """Compute the distogram logits.
        
        :param pairwise_repr: The pairwise representation tensor.
        :param molecule_atom_lens: The molecule atom lengths tensor.
        :param atom_feats: The atom features tensor.
        :return: The distogram logits.
        """
        # going through the layers
        
        if self.atom_resolution:
            assert exists(molecule_atom_lens)
            assert exists(atom_feats)

            pairwise_repr = batch_repeat_interleave_pairwise(pairwise_repr, molecule_atom_lens)

            pairwise_repr = pairwise_repr + self.atom_feats_to_pairwise(atom_feats)

        pred_logits = self.to_distogram_logits(symmetrize(pairwise_repr))

        return pred_logits


class DistogramLoss(nn.Module):
    def __init__(
        self,
        distance_bins,
        distogram_atom_resolution,
        ignore_index,      
        ):
        super().__init__()
        self.ignore_index = -1
      
        distance_bins_tensor = torch.tensor(distance_bins)
        
        # self.distance_bins
        self.register_buffer('distance_bins', distance_bins_tensor)
        num_dist_bins = default(num_dist_bins, len(distance_bins_tensor))
        
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
            distance_labels = compute_distogram(
                atom_pos,
                atom_mask,
                molecule_atom_lens,
                
                distance_bins,
                distogram_atom_indices,
                use_distogram_atom_resolution,
                ignore_index,
            )

        if exists(distance_labels):

            distogram_mask = pairwise_mask

            if distogram_atom_resolution:
                distogram_mask = to_pairwise_mask(atom_mask)

            distance_labels = torch.where(distogram_mask, distance_labels, self.ignore_index)

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
    
    atom_feats = embed_init['atom_feats']
    
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
    
    embed_trunk = trunk_model.forward(**data)
    for k,v in embed_trunk.items():
        print(k,v.shape)
    # 
    num_parameters = sum(p.numel() for p in trunk_model.parameters())
    print(num_parameters)
        
    print('trunk part over')
    print('-----------------------------')
    print('distogram part start')
    
    distogram_model = DistogramHead(**conf.distogram)
    distogram_model = distogram_model.to(device)
    
    pairwise_repr = embed_trunk['z']
    molecule_atom_lens = input_data['molecule_atom_lens']
    atom_feats = input_data['atom_feats']
    
    pred_logit = distogram_model.forward(        
        pairwise_repr = embed_trunk['z'],  
        molecule_atom_lens = input_data['molecule_atom_lens'],  
        atom_feats = input_data['atom_feats']
    )
    
    compute_distogram()
    
    print('test')