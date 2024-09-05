
class ComputeRankingScore(Module):
    """Compute ranking score."""

    def __init__(
        self,
        eps: float = 1e-8,
        score_iptm_weight: float = 0.8,
        score_ptm_weight: float = 0.2,
        score_disorder_weight: float = 0.5,
    ):
        super().__init__()
        self.eps = eps
        self.compute_clash = ComputeClash()
        self.compute_confidence_score = ComputeConfidenceScore(eps=eps)

        self.score_iptm_weight = score_iptm_weight
        self.score_ptm_weight = score_ptm_weight
        self.score_disorder_weight = score_disorder_weight

    @typecheck
    def compute_disorder(
        self,
        plddt: Float["b m"],  
        atom_mask: Bool["b m"],
        atom_is_molecule_types: Bool[f"b m {IS_MOLECULE_TYPES}"],
    ) -> Float[" b"]:  
        """Compute disorder score.

        :param plddt: [b m] plddt
        :param atom_mask: [b m] atom mask
        :param atom_is_molecule_types: [b m 2] atom is molecule types
        :return: [b] disorder
        """
        is_protein_mask = atom_is_molecule_types[..., IS_PROTEIN_INDEX]
        mask = atom_mask * is_protein_mask

        atom_rasa = 1.0 - plddt

        disorder = ((atom_rasa > 0.581) * mask).sum(dim=-1) / (self.eps + mask.sum(dim=1))
        return disorder

    @typecheck
    def compute_full_complex_metric(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        asym_id: Int["b n"],  
        has_frame: Bool["b n"],  
        molecule_atom_lens: Int["b n"],  
        atom_pos: Float["b m 3"],  
        atom_mask: Bool["b m"],  
        is_molecule_types: Bool[f"b n {IS_MOLECULE_TYPES}"],
        return_confidence_score: bool = False,
    ) -> Float[" b"] | Tuple[Float[" b"], Tuple[ConfidenceScore, Bool[" b"]]]:

        """Compute full complex metric.

        :param confidence_head_logits: ConfidenceHeadLogits
        :param asym_id: [b n] asym_id of each residue
        :param has_frame: [b n] has_frame of each residue
        :param molecule_atom_lens: [b n] molecule atom lens
        :param atom_pos: [b m 3] atom positions
        :param atom_mask: [b m] atom mask
        :param is_molecule_types: [b n 2] is_molecule_types
        :return: [b] score
        """

        # Section 5.9.3.1

        device = atom_pos.device
        batch_size, seq_len = asym_id.shape

        indices = torch.arange(seq_len, device=device)

        indices = repeat(indices, "n -> b n", b=batch_size)
        valid_indices = torch.ones_like(indices).bool()

        # valid_indices at padding position has value False
        indices = batch_repeat_interleave(indices, molecule_atom_lens)
        valid_indices = batch_repeat_interleave(valid_indices, molecule_atom_lens)

        # broadcast is_molecule_types to atom

        # einx.get_at('b [n] is_type, b m -> b m is_type', is_molecule_types, indices)

        indices = repeat(indices, "b m -> b m is_type", is_type=is_molecule_types.shape[-1])
        atom_is_molecule_types = is_molecule_types.gather(1, indices) * valid_indices[..., None]

        confidence_score = self.compute_confidence_score(
            confidence_head_logits, asym_id, has_frame, multimer_mode=True
        )
        has_clash = self.compute_clash(
            atom_pos,
            atom_mask,
            molecule_atom_lens,
            asym_id,
        )

        disorder = self.compute_disorder(confidence_score.plddt, atom_mask, atom_is_molecule_types)

        # Section 5.9.3 equation 19
        weighted_score = (
            confidence_score.iptm * self.score_iptm_weight
            + confidence_score.ptm * self.score_ptm_weight
            + disorder * self.score_disorder_weight
            - 100 * has_clash
        )

        if not return_confidence_score:
            return weighted_score

        return weighted_score, (confidence_score, has_clash)

    @typecheck
    def compute_single_chain_metric(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        asym_id: Int["b n"],  
        has_frame: Bool["b n"],  
    ) -> Float[" b"]:

        """Compute single chain metric.

        :param confidence_head_logits: ConfidenceHeadLogits
        :param asym_id: [b n] asym_id of each residue
        :param has_frame: [b n] has_frame of each residue
        :return: [b] score
        """

        # Section 5.9.3.2

        confidence_score = self.compute_confidence_score(
            confidence_head_logits, asym_id, has_frame, multimer_mode=False
        )

        score = confidence_score.ptm
        return score

    @typecheck
    def compute_interface_metric(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        asym_id: Int["b n"],  
        has_frame: Bool["b n"],  
        interface_chains: List,
    ) -> Float[" b"]:  
        """Compute interface metric.

        :param confidence_head_logits: ConfidenceHeadLogits
        :param asym_id: [b n] asym_id of each residue
        :param has_frame: [b n] has_frame of each residue
        :param interface_chains: List
        :return: [b] score
        """

        batch = asym_id.shape[0]

        # Section 5.9.3.3

        # interface_chains: List[chain_id_tuple]
        # chain_id_tuple:
        #  - correspond to the asym_id of one or two chain
        #  - compute R(C) for one chain
        #  - compute 1/2 [R(A) + R(b)] for two chain

        (
            chain_wise_iptm,
            chain_wise_iptm_mask,
            unique_chains,
        ) = self.compute_confidence_score.compute_ptm(
            confidence_head_logits.pae, asym_id, has_frame, compute_chain_wise_iptm=True
        )

        # Section 5.9.3 equation 20
        interface_metric = torch.zeros(batch).type_as(chain_wise_iptm)

        # R(c) = mean(Mij) restricted to i = c or j = c
        masked_chain_wise_iptm = chain_wise_iptm * chain_wise_iptm_mask
        iptm_sum = masked_chain_wise_iptm + rearrange(masked_chain_wise_iptm, "b i j -> b j i")
        iptm_count = chain_wise_iptm_mask.int() + rearrange(
            chain_wise_iptm_mask.int(), "b i j -> b j i"
        )

        for b, chains in enumerate(interface_chains):
            for chain in chains:
                idx = unique_chains[b].tolist().index(chain)
                interface_metric[b] += iptm_sum[b, idx].sum() / iptm_count[b, idx].sum().clamp(
                    min=1
                )
            interface_metric[b] /= len(chains)
        return interface_metric
