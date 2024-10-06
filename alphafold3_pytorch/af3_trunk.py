import torch
import torch.nn as nn

from af3_utils import exists,default,identity
from af3_basic import LinearNoBias

from alphafold3_pytorch import PairformerStack
from alphafold3_pytorch import TemplateEmbedder
from alphafold3_pytorch import MSAModule

class AF3Trunk(nn.Module):
    def __init__(
        self,
        dim_single,
        dim_pairwise,
        
        pairformer_kwargs,
        
        use_template,
        template_kwargs,
        
        use_msa,
        msa_kwargs,
        ):
        super().__init__()
        
        self.pairformer = PairformerStack(
            dim_single = dim_single,
            dim_pairwise = dim_pairwise,
            **pairformer_kwargs
        )
        
        # recycling related
        self.recycle_single = nn.Sequential(
            nn.LayerNorm(dim_single),
            LinearNoBias(dim_single, dim_single)
        )

        self.recycle_pairwise = nn.Sequential(
            nn.LayerNorm(dim_pairwise),
            LinearNoBias(dim_pairwise, dim_pairwise)
        )

        # templates
        self.use_template = use_template
        if self.use_template:
            self.template_embedder = TemplateEmbedder(
                dim_pairwise = dim_pairwise,
                **template_kwargs
            )

        # msa

        # they concat some MSA related information per MSA-token pair (`has_deletion` w/ dim=1, `deletion_value` w/ dim=1)
        self.use_msa = use_msa
        if self.use_msa:
            self.msa_module = MSAModule(
                dim_single = dim_single,
                dim_pairwise = dim_pairwise,
                **msa_kwargs,
            )
        
    def forward(
        self,
        s_init,
        s_mask,
        z_init,
        z_mask,
        
        template,
        template_mask,
        
        msa,
        msa_mask,
        additional_msa_feats,
        
        num_recycling_steps,
        detach_when_recycling,
        ):
        
        maybe_recycling_detach = torch.detach if detach_when_recycling else identity

        recycled_pairwise = recycled_single = None
        single = pairwise = None

        # for each recycling step

        for _ in range(num_recycling_steps):

            # handle recycled single and pairwise if not first step
            recycled_single = recycled_pairwise = 0.

            if exists(single):
                single = maybe_recycling_detach(single)
                recycled_single = self.recycle_single(single)

            if exists(pairwise):
                pairwise = maybe_recycling_detach(pairwise)
                recycled_pairwise = self.recycle_pairwise(pairwise)

            single = s_init + recycled_single
            pairwise = z_init + recycled_pairwise
                
            # template
            if self.use_template and exists(template):
                embedded_template = self.template_embedder(
                    templates = template,
                    template_mask = template_mask,
                    pairwise_repr = pairwise,
                )

                pairwise = embedded_template + pairwise

            # msa
            if self.use_msa and exists(msa):
                embedded_msa = self.msa_module(
                    msa = msa,
                    single_repr = single,
                    pairwise_repr = pairwise,
                    msa_mask = msa_mask,
                    additional_msa_feats = additional_msa_feats
                )

                pairwise = embedded_msa + pairwise
              
            # main attention trunk (pairformer)
            single, pairwise = self.pairformer(
                single_repr = single,
                pairwise_repr = pairwise,
                mask = s_mask,
            )
        r_ans = {
            's':single,
            'z':pairwise
        }
        return r_ans
    
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
    
    num_parameters = sum(p.numel() for p in trunk_model.parameters())
    print(num_parameters)
        
    print('trunk part over')
    
    print('test')