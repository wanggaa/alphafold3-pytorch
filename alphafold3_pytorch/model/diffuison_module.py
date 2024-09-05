
class DiffusionModule(Module):
    """ Algorithm 20 """

    @typecheck
    def __init__(
        self,
        *,
        dim_pairwise_trunk,
        dim_pairwise_rel_pos_feats,
        atoms_per_window = 27,  # for atom sequence, take the approach of (batch, seq, atoms, ..), where atom dimension is set to the molecule or molecule with greatest number of atoms, the rest padded. atom_mask must be passed in - default to 27 for proteins, with tryptophan having 27 atoms
        dim_pairwise = 128,
        sigma_data = 16,
        dim_atom = 128,
        dim_atompair = 16,
        dim_token = 768,
        dim_single = 384,
        dim_fourier = 256,
        single_cond_kwargs: dict = dict(
            num_transitions = 2,
            transition_expansion_factor = 2,
        ),
        pairwise_cond_kwargs: dict = dict(
            num_transitions = 2
        ),
        atom_encoder_depth = 3,
        atom_encoder_heads = 4,
        token_transformer_depth = 24,
        token_transformer_heads = 16,
        atom_decoder_depth = 3,
        atom_decoder_heads = 4,
        serial = True,
        atom_encoder_kwargs: dict = dict(),
        atom_decoder_kwargs: dict = dict(),
        token_transformer_kwargs: dict = dict(),
        use_linear_attn = False,
        checkpoint_token_transformer = False,
        linear_attn_kwargs: dict = dict(
            heads = 8,
            dim_head = 16
        )
    ):
        super().__init__()

        self.atoms_per_window = atoms_per_window

        # conditioning

        self.single_conditioner = SingleConditioning(
            sigma_data = sigma_data,
            dim_single = dim_single,
            dim_fourier = dim_fourier,
            **single_cond_kwargs
        )

        self.pairwise_conditioner = PairwiseConditioning(
            dim_pairwise_trunk = dim_pairwise_trunk,
            dim_pairwise_rel_pos_feats = dim_pairwise_rel_pos_feats,
            dim_pairwise = dim_pairwise,
            **pairwise_cond_kwargs
        )

        # atom attention encoding related modules

        self.atom_pos_to_atom_feat = LinearNoBias(3, dim_atom)

        self.missing_atom_feat = nn.Parameter(torch.zeros(dim_atom))

        self.single_repr_to_atom_feat_cond = nn.Sequential(
            nn.LayerNorm(dim_single),
            LinearNoBias(dim_single, dim_atom)
        )

        self.pairwise_repr_to_atompair_feat_cond = nn.Sequential(
            nn.LayerNorm(dim_pairwise),
            LinearNoBias(dim_pairwise, dim_atompair)
        )

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

        self.atom_encoder = DiffusionTransformer(
            dim = dim_atom,
            dim_single_cond = dim_atom,
            dim_pairwise = dim_atompair,
            attn_window_size = atoms_per_window,
            depth = atom_encoder_depth,
            heads = atom_encoder_heads,
            serial = serial,
            use_linear_attn = use_linear_attn,
            linear_attn_kwargs = linear_attn_kwargs,
            **atom_encoder_kwargs
        )

        self.atom_feats_to_pooled_token = AtomToTokenPooler(
            dim = dim_atom,
            dim_out = dim_token
        )

        # token attention related modules

        self.cond_tokens_with_cond_single = nn.Sequential(
            nn.LayerNorm(dim_single),
            LinearNoBias(dim_single, dim_token)
        )

        self.token_transformer = DiffusionTransformer(
            dim = dim_token,
            dim_single_cond = dim_single,
            dim_pairwise = dim_pairwise,
            depth = token_transformer_depth,
            heads = token_transformer_heads,
            serial = serial,
            **token_transformer_kwargs
        )

        self.attended_token_norm = nn.LayerNorm(dim_token)

        # checkpointing

        self.checkpoint_token_transformer = checkpoint_token_transformer

        # atom attention decoding related modules

        self.tokens_to_atom_decoder_input_cond = LinearNoBias(dim_token, dim_atom)

        self.atom_decoder = DiffusionTransformer(
            dim = dim_atom,
            dim_single_cond = dim_atom,
            dim_pairwise = dim_atompair,
            attn_window_size = atoms_per_window,
            depth = atom_decoder_depth,
            heads = atom_decoder_heads,
            serial = serial,
            use_linear_attn = use_linear_attn,
            linear_attn_kwargs = linear_attn_kwargs,
            **atom_decoder_kwargs
        )

        self.atom_feat_to_atom_pos_update = nn.Sequential(
            nn.LayerNorm(dim_atom),
            LinearNoBias(dim_atom, 3)
        )

    @typecheck
    def forward(
        self,
        noised_atom_pos: Float['b m 3'],
        *,
        atom_feats: Float['b m da'],
        atompair_feats: Float['b m m dap'] | Float['b nw w (w*2) dap'],
        atom_mask: Bool['b m'],
        times: Float[' b'],
        mask: Bool['b n'],
        single_trunk_repr: Float['b n dst'],
        single_inputs_repr: Float['b n dsi'],
        pairwise_trunk: Float['b n n dpt'],
        pairwise_rel_pos_feats: Float['b n n dpr'],
        molecule_atom_lens: Int['b n'],
        atom_parent_ids: Int['b m'] | None = None,
        missing_atom_mask: Bool['b m']| None = None
    ):
        w = self.atoms_per_window
        device = noised_atom_pos.device

        batch_size, seq_len = single_trunk_repr.shape[:2]
        atom_seq_len = atom_feats.shape[1]

        conditioned_single_repr = self.single_conditioner(
            times = times,
            single_trunk_repr = single_trunk_repr,
            single_inputs_repr = single_inputs_repr
        )

        conditioned_pairwise_repr = self.pairwise_conditioner(
            pairwise_trunk = pairwise_trunk,
            pairwise_rel_pos_feats = pairwise_rel_pos_feats
        )

        # lines 7-14 in Algorithm 5

        atom_feats_cond = atom_feats

        # the most surprising part of the paper; no geometric biases!

        noised_atom_pos_feats = self.atom_pos_to_atom_feat(noised_atom_pos)

        # for missing atoms, replace the noise atom pos features with a missing embedding

        if exists(missing_atom_mask):
            noised_atom_pos_feats = einx.where('b m, d, b m d -> b m d', missing_atom_mask, self.missing_atom_feat, noised_atom_pos_feats)

        # sum the noised atom position features to the atom features

        atom_feats = noised_atom_pos_feats + atom_feats

        # condition atom feats cond (cl) with single repr

        single_repr_cond = self.single_repr_to_atom_feat_cond(conditioned_single_repr)

        single_repr_cond = batch_repeat_interleave(single_repr_cond, molecule_atom_lens)
        single_repr_cond = pad_or_slice_to(single_repr_cond, length = atom_feats_cond.shape[1], dim = 1)

        atom_feats_cond = single_repr_cond + atom_feats_cond

        # window the atom pair features before passing to atom encoder and decoder if necessary

        atompair_is_windowed = atompair_feats.ndim == 5

        if not atompair_is_windowed:
            atompair_feats = full_pairwise_repr_to_windowed(atompair_feats, window_size = self.atoms_per_window)

        # condition atompair feats with pairwise repr

        pairwise_repr_cond = self.pairwise_repr_to_atompair_feat_cond(conditioned_pairwise_repr)

        indices = torch.arange(seq_len, device = device)
        indices = repeat(indices, 'n -> b n', b = batch_size)

        indices = batch_repeat_interleave(indices, molecule_atom_lens)
        indices = pad_or_slice_to(indices, atom_seq_len, dim = -1)
        indices = pad_and_window(indices, w)

        row_indices = col_indices = indices
        row_indices = rearrange(row_indices, 'b n w -> b n w 1', w = w)
        col_indices = rearrange(col_indices, 'b n w -> b n 1 w', w = w)

        col_indices = concat_previous_window(col_indices, dim_seq = 1, dim_window = -1)
        row_indices, col_indices = torch.broadcast_tensors(row_indices, col_indices)

        # pairwise_repr_cond = einx.get_at('b [i j] dap, b nw w1 w2, b nw w1 w2 -> b nw w1 w2 dap', pairwise_repr_cond, row_indices, col_indices)

        row_indices, unpack_one = pack_one(row_indices, 'b *')
        col_indices, _ = pack_one(col_indices, 'b *')

        rowcol_indices = col_indices + row_indices * pairwise_repr_cond.shape[2]
        rowcol_indices = repeat(rowcol_indices, 'b rc -> b rc dap', dap = pairwise_repr_cond.shape[-1])
        pairwise_repr_cond, _ = pack_one(pairwise_repr_cond, 'b * dap')

        pairwise_repr_cond = pairwise_repr_cond.gather(1, rowcol_indices)
        pairwise_repr_cond = unpack_one(pairwise_repr_cond, 'b * dap')
        
        atompair_feats = pairwise_repr_cond + atompair_feats

        # condition atompair feats further with single atom repr

        atom_repr_cond = self.atom_repr_to_atompair_feat_cond(atom_feats)
        atom_repr_cond = pad_and_window(atom_repr_cond, w)

        atom_repr_cond_row, atom_repr_cond_col = atom_repr_cond.chunk(2, dim = -1)

        atom_repr_cond_col = concat_previous_window(atom_repr_cond_col, dim_seq = 1, dim_window = 2)

        atompair_feats = einx.add('b nw w1 w2 dap, b nw w1 dap -> b nw w1 w2 dap', atompair_feats, atom_repr_cond_row)
        atompair_feats = einx.add('b nw w1 w2 dap, b nw w2 dap -> b nw w1 w2 dap', atompair_feats, atom_repr_cond_col)

        # furthermore, they did one more MLP on the atompair feats for attention biasing in atom transformer

        atompair_feats = self.atompair_feats_mlp(atompair_feats) + atompair_feats

        # take care of restricting atom attention to be intra molecular, if the atom_parent_ids were passed in

        windowed_mask = None

        if exists(atom_parent_ids):
            atom_parent_ids_rows = pad_and_window(atom_parent_ids, w)
            atom_parent_ids_columns = concat_previous_window(atom_parent_ids_rows, dim_seq = 1, dim_window = 2)

            windowed_mask = einx.equal('b n i, b n j -> b n i j', atom_parent_ids_rows, atom_parent_ids_columns)

        # atom encoder

        atom_feats = self.atom_encoder(
            atom_feats,
            mask = atom_mask,
            windowed_mask = windowed_mask,
            single_repr = atom_feats_cond,
            pairwise_repr = atompair_feats
        )

        atom_feats_skip = atom_feats

        tokens = self.atom_feats_to_pooled_token(
            atom_feats = atom_feats,
            atom_mask = atom_mask,
            molecule_atom_lens = molecule_atom_lens
        )

        # maybe checkpoint token transformer

        token_transformer = self.token_transformer

        if should_checkpoint(self, tokens, 'checkpoint_token_transformer'):
            token_transformer = partial(checkpoint, token_transformer)

        # token transformer

        tokens = self.cond_tokens_with_cond_single(conditioned_single_repr) + tokens

        tokens = token_transformer(
            tokens,
            mask = mask,
            single_repr = conditioned_single_repr,
            pairwise_repr = conditioned_pairwise_repr,
        )

        tokens = self.attended_token_norm(tokens)

        # atom decoder

        atom_decoder_input = self.tokens_to_atom_decoder_input_cond(tokens)

        atom_decoder_input = batch_repeat_interleave(atom_decoder_input, molecule_atom_lens)
        atom_decoder_input = pad_or_slice_to(atom_decoder_input, length = atom_feats_skip.shape[1], dim = 1)

        atom_decoder_input = atom_decoder_input + atom_feats_skip

        atom_feats = self.atom_decoder(
            atom_decoder_input,
            mask = atom_mask,
            windowed_mask = windowed_mask,
            single_repr = atom_feats_cond,
            pairwise_repr = atompair_feats
        )

        atom_pos_update = self.atom_feat_to_atom_pos_update(atom_feats)

        return atom_pos_update
