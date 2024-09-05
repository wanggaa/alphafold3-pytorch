
# distogram head

class DistogramHead(Module):

    @typecheck
    def __init__(
        self,
        *,
        dim_pairwise = 128,
        num_dist_bins = 38,
        dim_atom = 128,
        atom_resolution = False
    ):
        super().__init__()

        self.to_distogram_logits = nn.Sequential(
            LinearNoBias(dim_pairwise, num_dist_bins),
            Rearrange('b ... l -> b l ...')
        )

        # atom resolution
        # for now, just embed per atom distances, sum to atom features, project to pairwise dimension

        self.atom_resolution = atom_resolution

        if atom_resolution:
            self.atom_feats_to_pairwise = LinearNoBiasThenOuterSum(dim_atom, dim_pairwise)

        # tensor typing

        self.da = dim_atom

    @typecheck
    def forward(
        self,
        pairwise_repr: Float['b n n d'],
        molecule_atom_lens: Int['b n'] | None = None,
        atom_feats: Float['b m {self.da}'] | None = None,
    ) -> Float['b l n n'] | Float['b l m m']:

        if self.atom_resolution:
            assert exists(molecule_atom_lens)
            assert exists(atom_feats)

            pairwise_repr = batch_repeat_interleave_pairwise(pairwise_repr, molecule_atom_lens)
            pairwise_repr = pairwise_repr + self.atom_feats_to_pairwise(atom_feats)

        logits = self.to_distogram_logits(symmetrize(pairwise_repr))

        return logits
