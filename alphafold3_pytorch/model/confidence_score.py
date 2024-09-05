

class ComputeConfidenceScore(Module):
    """Compute confidence score."""

    @typecheck
    def __init__(
        self,
        pae_breaks: Float[" pae_break"] = torch.arange(0, 31.5, 0.5),  
        pde_breaks: Float[" pde_break"] = torch.arange(0, 31.5, 0.5),  
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = eps
        self.register_buffer("pae_breaks", pae_breaks)
        self.register_buffer("pde_breaks", pde_breaks)

    @typecheck
    def _calculate_bin_centers(
        self,
        breaks: Float[" breaks"],  
    ) -> Float[" breaks+1"]:  
        """Calculate bin centers from bin edges.

        :param breaks: [num_bins -1] bin edges
        :return: bin_centers: [num_bins] bin centers
        """

        step = breaks[1] - breaks[0]

        bin_centers = breaks + step / 2
        last_bin_center = breaks[-1] + step

        bin_centers = torch.concat([bin_centers, last_bin_center.unsqueeze(0)])

        return bin_centers

    @typecheck
    def forward(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        asym_id: Int["b n"],  
        has_frame: Bool["b n"],  
        ptm_residue_weight: Float["b n"] | None = None,  
        multimer_mode: bool = True,
    ) -> ConfidenceScore:
        """Main function to compute confidence score.

        :param confidence_head_logits: ConfidenceHeadLogits
        :param asym_id: [b n] asym_id of each residue
        :param has_frame: [b n] has_frame of each residue
        :param ptm_residue_weight: [b n] weight of each residue
        :param multimer_mode: bool
        :return: Confidence score
        """
        plddt = self.compute_plddt(confidence_head_logits.plddt)

        # Section 5.9.1 equation 17
        ptm = self.compute_ptm(
            confidence_head_logits.pae, asym_id, has_frame, ptm_residue_weight, interface=False,
        )

        iptm = None

        if multimer_mode:
            # Section 5.9.2 equation 18
            iptm = self.compute_ptm(
                confidence_head_logits.pae, asym_id, has_frame, ptm_residue_weight, interface=True,
            )

        confidence_score = ConfidenceScore(plddt=plddt, ptm=ptm, iptm=iptm)
        return confidence_score

    @typecheck
    def compute_plddt(
        self,
        logits: Float["b plddt m"],  
    ) -> Float["b m"]:  
        """Compute plDDT from logits.

        :param logits: [b c m] logits
        :return: [b m] plDDT
        """
        logits = rearrange(logits, "b plddt m -> b m plddt")
        num_bins = logits.shape[-1]
        bin_width = 1.0 / num_bins
        bin_centers = torch.arange(
            0.5 * bin_width, 1.0, bin_width, dtype=logits.dtype, device=logits.device
        )
        probs = F.softmax(logits, dim=-1)

        predicted_lddt = einsum(probs, bin_centers, "b m plddt, plddt -> b m")
        return predicted_lddt * 100

    @typecheck
    def compute_ptm(
        self,
        pae_logits: Float["b pae n n"],  
        asym_id: Int["b n"],  
        has_frame: Bool["b n"],  
        residue_weights: Float["b n"] | None = None,
        interface: bool = False,
        compute_chain_wise_iptm: bool = False,
    ) -> Float[" b"] | Tuple[Float["b chains chains"], Bool["b chains chains"], Int["b chains"]]:

        """Compute pTM from logits.

        :param logits: [b c n n] logits
        :param asym_id: [b n] asym_id of each residue
        :param has_frame: [b n] has_frame of each residue
        :param residue_weights: [b n] weight of each residue
        :param interface: bool
        :param compute_chain_wise_iptm: bool
        :return: pTM
        """        
        if not exists(residue_weights):
            residue_weights = torch.ones_like(has_frame)

        residue_weights = residue_weights * has_frame

        num_batch, *_, num_res, device = *pae_logits.shape, pae_logits.device

        pae_logits = rearrange(pae_logits, "b c i j -> b i j c")

        bin_centers = self._calculate_bin_centers(self.pae_breaks)

        num_frame = torch.sum(has_frame, dim=-1)
        # Clip num_frame to avoid negative/undefined d0.
        clipped_num_frame = torch.clamp(num_frame, min=19)

        # Compute d_0(num_frame) as defined by TM-score, eqn. (5) in Yang & Skolnick
        # "Scoring function for automated assessment of protein structure template
        # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
        d0 = 1.24 * (clipped_num_frame - 15) ** (1.0 / 3) - 1.8

        # TM-Score term for every bin. [num_batch, num_bins]
        tm_per_bin = 1.0 / (1 + torch.square(bin_centers[None, :]) / torch.square(d0[..., None]))

        # Convert logits to probs.
        probs = F.softmax(pae_logits, dim=-1)

        # E_distances tm(distance).
        predicted_tm_term = einsum(probs, tm_per_bin, "b i j pae, b pae -> b i j ")

        if compute_chain_wise_iptm:
            # chain_wise_iptm[b, i, j]: iptm of chain i and chain j in batch b

            # get the max num_chains across batch
            unique_chains = [torch.unique(asym).tolist() for asym in asym_id]
            max_chains = max(len(chains) for chains in unique_chains)

            chain_wise_iptm = torch.zeros(
                (num_batch, max_chains, max_chains), device=device
            )
            chain_wise_iptm_mask = torch.zeros_like(chain_wise_iptm).bool()

            for b in range(num_batch):
                for i, chain_i in enumerate(unique_chains[b]):
                    for j, chain_j in enumerate(unique_chains[b]):
                        if chain_i != chain_j:
                            mask_i = (asym_id[b] == chain_i)[:, None]
                            mask_j = (asym_id[b] == chain_j)[None, :]
                            pair_mask = mask_i * mask_j
                            pair_residue_weights = pair_mask * einx.multiply(
                                "... i, ... j -> ... i j", residue_weights[b], residue_weights[b]
                            )

                            if pair_residue_weights.sum() == 0:
                                # chain i or chain j does not have any valid frame
                                continue

                            normed_residue_mask = pair_residue_weights / (
                                self.eps
                                + torch.sum(pair_residue_weights, dim=-1, keepdims=True)
                            )

                            masked_predicted_tm_term = predicted_tm_term[b] * pair_mask

                            per_alignment = torch.sum(
                                masked_predicted_tm_term * normed_residue_mask, dim=-1
                            )
                            weighted_argmax = (residue_weights[b] * per_alignment).argmax()
                            chain_wise_iptm[b, i, j] = per_alignment[weighted_argmax]
                            chain_wise_iptm_mask[b, i, j] = True

            return chain_wise_iptm, chain_wise_iptm_mask, torch.tensor(unique_chains)

        else:
            pair_mask = torch.ones(size=(num_batch, num_res, num_res), device=device).bool()
            if interface:
                pair_mask *= asym_id[:, :, None] != asym_id[:, None, :]

            predicted_tm_term *= pair_mask

            pair_residue_weights = pair_mask * (
                residue_weights[:, None, :] * residue_weights[:, :, None]
            )
            normed_residue_mask = pair_residue_weights / (
                self.eps + torch.sum(pair_residue_weights, dim=-1, keepdims=True)
            )

            per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)
            weighted_argmax = (residue_weights * per_alignment).argmax(dim=-1)
            return per_alignment[torch.arange(num_batch), weighted_argmax]

    @typecheck
    def compute_pde(
        self,
        pde_logits: Float["b pde n n"],  
        tok_repr_atm_mask: Bool["b n"],  
    ) -> Float["b n n"]:  
        """Compute PDE from logits."""

        pde_logits = rearrange(pde_logits, "b pde i j -> b i j pde")
        bin_centers = self._calculate_bin_centers(self.pde_breaks)
        probs = F.softmax(pde_logits, dim=-1)

        pde = einsum(probs, bin_centers, "b i j pde, pde -> b i j")

        mask = to_pairwise_mask(tok_repr_atm_mask)

        pde = pde * mask
        return pde
