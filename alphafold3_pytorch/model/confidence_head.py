
class ConfidenceHead(Module):
    """ Algorithm 31 """

    @typecheck
    def __init__(
        self,
        *,
        dim_single_inputs,
        dim_atom = 128,
        atompair_dist_bins: List[float],
        dim_single = 384,
        dim_pairwise = 128,
        num_plddt_bins = 50,
        num_pde_bins = 64,
        num_pae_bins = 64,
        pairformer_depth = 4,
        pairformer_kwargs: dict = dict()
    ):
        super().__init__()

        atompair_dist_bins = Tensor(atompair_dist_bins)

        self.register_buffer('atompair_dist_bins', atompair_dist_bins)

        num_dist_bins = atompair_dist_bins.shape[-1]
        self.num_dist_bins = num_dist_bins

        self.dist_bin_pairwise_embed = nn.Embedding(num_dist_bins, dim_pairwise)
        self.single_inputs_to_pairwise = LinearNoBiasThenOuterSum(dim_single_inputs, dim_pairwise)

        # pairformer stack

        self.pairformer_stack = PairformerStack(
            dim_single = dim_single,
            dim_pairwise = dim_pairwise,
            depth = pairformer_depth,
            **pairformer_kwargs
        )

        # to predictions

        self.to_pae_logits = nn.Sequential(
            LinearNoBias(dim_pairwise, num_pae_bins),
            Rearrange('b ... l -> b l ...')
        )

        self.to_pde_logits = nn.Sequential(
            LinearNoBias(dim_pairwise, num_pde_bins),
            Rearrange('b ... l -> b l ...')
        )

        self.to_plddt_logits = nn.Sequential(
            LinearNoBias(dim_single, num_plddt_bins),
            Rearrange('b ... l -> b l ...')
        )

        self.to_resolved_logits = nn.Sequential(
            LinearNoBias(dim_single, 2),
            Rearrange('b ... l -> b l ...')
        )

        # atom resolution

        self.atom_feats_to_single = LinearNoBias(dim_atom, dim_single)

        # tensor typing

        self.da = dim_atom

    @typecheck
    def forward(
        self,
        *,
        single_inputs_repr: Float["b n dsi"],
        single_repr: Float["b n ds"],
        pairwise_repr: Float["b n n dp"],
        pred_atom_pos: Float["b m 3"],
        atom_feats: Float["b m {self.da}"],
        molecule_atom_indices: Int["b n"],
        molecule_atom_lens: Int["b n"],
        mask: Bool["b n"] | None = None,
        return_pae_logits: bool = True,
    ) -> ConfidenceHeadLogits:
        """Compute the confidence head logits.

        :param single_inputs_repr: The single inputs representation tensor.
        :param single_repr: The single representation tensor.
        :param pairwise_repr: The pairwise representation tensor.
        :param pred_atom_pos: The predicted atom positions tensor.
        :param atom_feats: The atom features tensor.
        :param molecule_atom_indices: The molecule atom indices tensor.
        :param molecule_atom_lens: The molecule atom lengths tensor.
        :param mask: The mask tensor.
        :param return_pae_logits: Whether to return the predicted aligned error (PAE) logits.
        :return: The confidence head logits.
        """

        pairwise_repr = pairwise_repr + self.single_inputs_to_pairwise(single_inputs_repr)

        # pluck out the representative atoms for non-atomic resolution confidence head outputs

        # pred_molecule_pos = einx.get_at('b [m] c, b n -> b n c', pred_atom_pos, molecule_atom_indices)

        molecule_atom_indices = repeat(
            molecule_atom_indices, "b n -> b n c", c=pred_atom_pos.shape[-1]
        )
        pred_molecule_pos = pred_atom_pos.gather(1, molecule_atom_indices)

        # interatomic distances - embed and add to pairwise

        intermolecule_dist = torch.cdist(pred_molecule_pos, pred_molecule_pos, p=2)

        dist_bin_indices = distance_to_bins(intermolecule_dist, self.atompair_dist_bins)
        pairwise_repr = pairwise_repr + self.dist_bin_pairwise_embed(dist_bin_indices)

        # pairformer stack

        single_repr, pairwise_repr = self.pairformer_stack(
            single_repr=single_repr, pairwise_repr=pairwise_repr, mask=mask
        )

        # handle atom level resolution

        atom_single_repr = batch_repeat_interleave(single_repr, molecule_atom_lens)

        atom_single_repr = atom_single_repr + self.atom_feats_to_single(atom_feats)

        # to logits

        pde_logits = self.to_pde_logits(symmetrize(pairwise_repr))

        plddt_logits = self.to_plddt_logits(atom_single_repr)
        resolved_logits = self.to_resolved_logits(atom_single_repr)

        # they only incorporate pae at some stage of training

        pae_logits = None

        if return_pae_logits:
            pae_logits = self.to_pae_logits(pairwise_repr)

        # return all logits

        return ConfidenceHeadLogits(pae_logits, pde_logits, plddt_logits, resolved_logits)

