import torch
import torch.nn as nn

from af3_utils import exists,default,identity
from af3_basic import LinearNoBias

from alphafold3_pytorch import PairformerStack,TemplateEmbedder,MSAModule

class AF3Trunk(nn.Module):
    def __init__(
        self,
        dim_single,
        dim_pairwise,
        checkpoint_trunk_pairformer,
        pairformer_stack,
        detach_when_recycling,
        use_template,
        template_kwargs,
        use_msa,
        msa_kwargs,
        ):
        super().__init__()

        self.num_recycling_steps
        
        self.recycle_single
        self.recycle_pairwise
        
        self.pairformer = PairformerStack(
            dim_single = dim_single,
            dim_pairwise = dim_pairwise,
            checkpoint=checkpoint_trunk_pairformer,
            **pairformer_stack
        )
        
        # recycling related

        self.detach_when_recycling = detach_when_recycling

        self.recycle_single = nn.Sequential(
            nn.LayerNorm(dim_single),
            LinearNoBias(dim_single, dim_single)
        )

        self.recycle_pairwise = nn.Sequential(
            nn.LayerNorm(dim_pairwise),
            LinearNoBias(dim_pairwise, dim_pairwise)
        )

        # templates
        if use_template:
            self.template_embedder = TemplateEmbedder(
                dim_template_feats = dim_template_feats,
                dim = dim_template_model,
                dim_pairwise = dim_pairwise,
                checkpoint=checkpoint_input_embedding,
                **template_embedder_kwargs
            )

        # msa

        # they concat some MSA related information per MSA-token pair (`has_deletion` w/ dim=1, `deletion_value` w/ dim=1)
        if use_msa:
            self.msa_module = MSAModule(
                dim_single = dim_single,
                dim_pairwise = dim_pairwise,
                dim_msa_input = dim_msa_inputs,
                dim_additional_msa_feats = dim_additional_msa_feats,
                checkpoint=checkpoint_input_embedding,
                **msa_module_kwargs,
            )
            

        
    def forward(
        self,
        single_init,
        single_mask,
        pairwise_init,
        pairwise_mask,
        
        templates,
        templates_mask,
        
        msa,
        msa_mask,
        
        detach_when_recycling=True,
        ):
        detach_when_recycling = default(detach_when_recycling, self.detach_when_recycling)
        maybe_recycling_detach = torch.detach if detach_when_recycling else identity

        recycled_pairwise = recycled_single = None
        single = pairwise = None

        # for each recycling step

        for _ in range(self.num_recycling_steps):

            # handle recycled single and pairwise if not first step
            recycled_single = recycled_pairwise = 0.

            if exists(single):
                single = maybe_recycling_detach(single)
                recycled_single = self.recycle_single(single)

            if exists(pairwise):
                pairwise = maybe_recycling_detach(pairwise)
                recycled_pairwise = self.recycle_pairwise(pairwise)

            single = single_init + recycled_single
            pairwise = pairwise_init + recycled_pairwise
                
            # template
            if exists(templates):
                embedded_template = self.template_embedder(
                    templates = templates,
                    template_mask = template_mask,
                    pairwise_repr = pairwise,
                )

                pairwise = embedded_template + pairwise

            # msa
            if exists(msa):
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
                mask = mask
            )
            
if __name__ == '__main__':
    main_dir = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/workspace/wangjun/alphafold3-pytorch'
    
    import os
    config_path = os.path.join(main_dir,'configs/af3_test.yml')
    from omegaconf import OmegaConf
    conf = OmegaConf.load(config_path)

    from af3_embed import AF3Embed
    embed_model = AF3Embed(**conf.embed)
    trunk_model = AF3Trunk(**conf.trunk)

    import pickle
    input_data_path = os.path.join(main_dir,'.tmp/debug_data/temp.pkl')
    with open(input_data_path,'rb') as f:
        input_data = pickle.load(f)
    
    import tree
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    input_data = tree.map_structure(lambda x: x.to(device) if torch.is_tensor(x) else identity(x),input_data)
    input_data_bak = input_data.copy()
    
    embed_model = embed_model.to(device)
    trunk_model = trunk_model.to(device)
    
    # clear input data
    import inspect
    sig = inspect.signature(AF3Trunk.forward)
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
        
    input_data = input_data_bak
    
    
    print('test')