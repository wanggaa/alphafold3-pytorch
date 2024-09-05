

class InputFeatureEmbedder(Module):
    """ Algorithm 2 """

    def __init__(
        self,
        *,
        dim_atom_inputs,
        dim_atompair_inputs = 5,
        atoms_per_window = 27,
        dim_atom = 128,
        dim_atompair = 16,
        dim_token = 384,
        dim_single = 384,
        dim_pairwise = 128,
        dim_additional_token_feats = 33,
        num_molecule_types = NUM_MOLECULE_IDS,
        atom_transformer_blocks = 3,
        atom_transformer_heads = 4,
        atom_transformer_kwargs: dict = dict(),
    ):
        super().__init__()
        self.atoms_per_window = atoms_per_window

        self.to_atom_feats = LinearNoBias(dim_atom_inputs, dim_atom)

        self.to_atompair_feats = LinearNoBias(dim_atompair_inputs, dim_atompair)

        self.atom_repr_to_atompair_feat_cond = nn.Sequential(
            nn.LayerNorm(dim_atom),
            LinearNoBias(dim_atom, dim_atompair * 2),
            nn.ReLU()
        )

        self.atompair_feats_mlp = nn.Sequential(
            LinearNoBias(dim_atompair, dim_atompair),
            nn.ReLU(),
            LinearNoBias(dim_atompair, dim_atompair),
            nn.ReLU(),
            LinearNoBias(dim_atompair, dim_atompair),
        )

        self.atom_transformer = DiffusionTransformer(
            depth = atom_transformer_blocks,
            heads = atom_transformer_heads,
            dim = dim_atom,
            dim_single_cond = dim_atom,
            dim_pairwise = dim_atompair,
            attn_window_size = atoms_per_window,
            **atom_transformer_kwargs
        )

        self.atom_feats_to_pooled_token = AtomToTokenPooler(
            dim = dim_atom,
            dim_out = dim_token
        )

        dim_single_input = dim_token + dim_additional_token_feats

        self.dim_additional_token_feats = dim_additional_token_feats

        self.single_input_to_single_init = LinearNoBias(dim_single_input, dim_single)
        self.single_input_to_pairwise_init = LinearNoBiasThenOuterSum(dim_single_input, dim_pairwise)

        # this accounts for the `restypes` in the additional molecule features

        self.single_molecule_embed = nn.Embedding(num_molecule_types, dim_single)
        self.pairwise_molecule_embed = nn.Embedding(num_molecule_types, dim_pairwise)

    @typecheck
    def forward(
        self,
        *,
        atom_inputs: Float['b m dai'],
        atompair_inputs: Float['b m m dapi'] | Float['b nw w1 w2 dapi'],
        atom_mask: Bool['b m'],
        molecule_atom_lens: Int['b n'],
        molecule_ids: Int['b n'],
        additional_token_feats: Float['b n {self.dim_additional_token_feats}'] | None = None,

    ) -> EmbeddedInputs:

        w = self.atoms_per_window

        atom_feats = self.to_atom_feats(atom_inputs)
        atompair_feats = self.to_atompair_feats(atompair_inputs)

        # window the atom pair features before passing to atom encoder and decoder

        is_windowed = atompair_inputs.ndim == 5

        if not is_windowed:
            atompair_feats = full_pairwise_repr_to_windowed(atompair_feats, window_size = w)

        # condition atompair with atom repr

        atom_feats_cond = self.atom_repr_to_atompair_feat_cond(atom_feats)

        atom_feats_cond = pad_and_window(atom_feats_cond, w)

        atom_feats_cond_row, atom_feats_cond_col = atom_feats_cond.chunk(2, dim = -1)
        atom_feats_cond_col = concat_previous_window(atom_feats_cond_col, dim_seq = 1, dim_window = -2)

        atompair_feats = einx.add('b nw w1 w2 dap, b nw w1 dap',atompair_feats, atom_feats_cond_row)
        atompair_feats = einx.add('b nw w1 w2 dap, b nw w2 dap',atompair_feats, atom_feats_cond_col)

        # initial atom transformer

        atom_feats = self.atom_transformer(
            atom_feats,
            single_repr = atom_feats,
            pairwise_repr = atompair_feats
        )

        atompair_feats = self.atompair_feats_mlp(atompair_feats) + atompair_feats

        single_inputs = self.atom_feats_to_pooled_token(
            atom_feats = atom_feats,
            atom_mask = atom_mask,
            molecule_atom_lens = molecule_atom_lens
        )

        if exists(additional_token_feats):
            single_inputs = torch.cat((
                single_inputs,
                additional_token_feats
            ), dim = -1)

        single_init = self.single_input_to_single_init(single_inputs)
        pairwise_init = self.single_input_to_pairwise_init(single_inputs)

        # account for molecule id (restypes)

        molecule_ids = torch.where(molecule_ids >= 0, molecule_ids, 0) # account for padding

        single_molecule_embed = self.single_molecule_embed(molecule_ids)

        pairwise_molecule_embed = self.pairwise_molecule_embed(molecule_ids)
        pairwise_molecule_embed = einx.add('b i dp, b j dp -> b i j dp', pairwise_molecule_embed, pairwise_molecule_embed)

        # sum to single init and pairwise init, equivalent to one-hot in additional residue features

        single_init = single_init + single_molecule_embed
        pairwise_init = pairwise_init + pairwise_molecule_embed

        return EmbeddedInputs(single_inputs, single_init, pairwise_init, atom_feats, atompair_feats)
