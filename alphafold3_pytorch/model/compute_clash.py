
class ComputeClash(Module):
    """Compute clash score."""

    def __init__(
        self,
        atom_clash_dist: float = 1.1,
        chain_clash_count: int = 100,
        chain_clash_ratio: float = 0.5,
    ):
        super().__init__()
        self.atom_clash_dist = atom_clash_dist
        self.chain_clash_count = chain_clash_count
        self.chain_clash_ratio = chain_clash_ratio

    @typecheck
    def compute_has_clash(
        self,
        atom_pos: Float["m 3"],  
        asym_id: Int[" n"],  
        indices: Int[" m"],  
        valid_indices: Bool[" m"],
    ) -> Bool[""]:  
        """Compute if there is a clash in the chain.

        :param atom_pos: [m 3] atom positions
        :param asym_id: [n] asym_id of each residue
        :param indices: [m] indices
        :param valid_indices: [m] valid indices
        :return: [1] has_clash
        """

        # Section 5.9.2

        atom_pos = atom_pos[valid_indices]
        atom_asym_id = asym_id[indices][valid_indices]

        unique_chains = atom_asym_id.unique()
        for i in range(len(unique_chains)):
            for j in range(i + 1, len(unique_chains)):
                chain_i, chain_j = unique_chains[i], unique_chains[j]

                mask_i = atom_asym_id == chain_i
                mask_j = atom_asym_id == chain_j

                chain_i_len = mask_i.sum()
                chain_j_len = mask_j.sum()
                assert min(chain_i_len, chain_j_len) > 0

                chain_pair_dist = torch.cdist(atom_pos[mask_i], atom_pos[mask_j])
                chain_pair_clash = chain_pair_dist < self.atom_clash_dist
                clashes = chain_pair_clash.sum()
                has_clash = (clashes > self.chain_clash_count) or (
                    clashes / min(chain_i_len, chain_j_len) > self.chain_clash_ratio
                )

                if has_clash:
                    return torch.tensor(True, dtype=torch.bool, device=atom_pos.device)

        return torch.tensor(False, dtype=torch.bool, device=atom_pos.device)

    @typecheck
    def forward(
        self,
        atom_pos: Float["b m 3"] | Float["m 3"],  
        atom_mask: Bool["b m"] | Bool[" m"],
        molecule_atom_lens: Int["b n"] | Int[" n"],  
        asym_id: Int["b n"] | Int[" n"],  
    ) -> Bool[" b"]:

        """Compute if there is a clash in the chain.

        :param atom_pos: [b m 3] atom positions
        :param atom_mask: [b m] atom mask
        :param molecule_atom_lens: [b n] molecule atom lens
        :param asym_id: [b n] asym_id of each residue
        :return: [b] has_clash
        """

        if atom_pos.ndim == 2:
            atom_pos = atom_pos.unsqueeze(0)
            molecule_atom_lens = molecule_atom_lens.unsqueeze(0)
            asym_id = asym_id.unsqueeze(0)
            atom_mask = atom_mask.unsqueeze(0)

        device = atom_pos.device
        batch_size, seq_len = asym_id.shape

        indices = torch.arange(seq_len, device=device)

        indices = repeat(indices, "n -> b n", b=batch_size)
        valid_indices = torch.ones_like(indices).bool()

        # valid_indices at padding position has value False
        indices = batch_repeat_interleave(indices, molecule_atom_lens)
        valid_indices = batch_repeat_interleave(valid_indices, molecule_atom_lens)

        if exists(atom_mask):
            valid_indices = valid_indices * atom_mask

        has_clash = []
        for b in range(batch_size):
            has_clash.append(
                self.compute_has_clash(atom_pos[b], asym_id[b], indices[b], valid_indices[b])
            )

        has_clash = torch.stack(has_clash)
        return has_clash


    @typecheck
    def compute_modified_residue_score(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        atom_mask: Bool["b m"],  
        atom_is_modified_residue: Int["b m"],  
    ) -> Float[" b"]:  
        """Compute modified residue score.

        :param confidence_head_logits: ConfidenceHeadLogits
        :param atom_mask: [b m] atom mask
        :param atom_is_modified_residue: [b m] atom is modified residue
        :return: [b] score
        """

        # Section 5.9.3.4

        plddt = self.compute_confidence_score.compute_plddt(
            confidence_head_logits.plddt,
        )

        mask = atom_is_modified_residue * atom_mask
        plddt_mean = masked_average(plddt, mask, dim=-1, eps=self.eps)

        return plddt_mean