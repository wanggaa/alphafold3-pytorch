class ComputeModelSelectionScore(Module):
    """Compute model selection score."""

    INITIAL_TRAINING_DICT = {
        "protein-protein": {"interface": 20, "intra-chain": 20},
        "DNA-protein": {"interface": 10},
        "RNA-protein": {"interface": 10},
        "ligand-protein": {"interface": 10},
        "DNA-ligand": {"interface": 5},
        "RNA-ligand": {"interface": 5},
        "DNA-DNA": {"interface": 4, "intra-chain": 4},
        "RNA-RNA": {"interface": 16, "intra-chain": 16},
        "ligand-ligand": {"interface": 20, "intra-chain": 20},
        "metal_ion-metal_ion": {"interface": 10, "intra-chain": 10},
        "unresolved": {"unresolved": 10},
    }

    FINETUNING_DICT = {
        "protein-protein": {"interface": 20, "intra-chain": 20},
        "DNA-protein": {"interface": 10},
        "RNA-protein": {"interface": 2},
        "ligand-protein": {"interface": 10},
        "DNA-ligand": {"interface": 5},
        "RNA-ligand": {"interface": 2},
        "DNA-DNA": {"interface": 4, "intra-chain": 4},
        "RNA-RNA": {"interface": 16, "intra-chain": 16},
        "ligand-ligand": {"interface": 20, "intra-chain": 20},
        "metal_ion-metal_ion": {"interface": 0, "intra-chain": 0},
        "unresolved": {"unresolved": 10},
    }

    TYPE_MAPPING = {
        IS_PROTEIN: "protein",
        IS_DNA: "DNA",
        IS_RNA: "RNA",
        IS_LIGAND: "ligand",
        IS_METAL_ION: "metal_ion",
    }

    @typecheck
    def __init__(
        self,
        eps: float = 1e-8,
        dist_breaks: Float[" dist_break"] = torch.linspace(  
            2.3125,
            21.6875,
            37,
        ),
        nucleic_acid_cutoff: float = 30.0,
        other_cutoff: float = 15.0,
        contact_mask_threshold: float = 8.0,
        is_fine_tuning: bool = False,
        weight_dict_config: dict = None,
        dssp_path: str = "mkdssp",
    ):
        super().__init__()
        self.compute_confidence_score = ComputeConfidenceScore(eps=eps)
        self.eps = eps
        self.nucleic_acid_cutoff = nucleic_acid_cutoff
        self.other_cutoff = other_cutoff
        self.contact_mask_threshold = contact_mask_threshold
        self.is_fine_tuning = is_fine_tuning
        self.weight_dict_config = weight_dict_config

        self.register_buffer("dist_breaks", dist_breaks)
        self.register_buffer('lddt_thresholds', torch.tensor([0.5, 1.0, 2.0, 4.0]))

        self.dssp_path = dssp_path

    @property
    def can_calculate_unresolved_protein_rasa(self):
        """Check if `mkdssp` is available.

        :return: True if `mkdssp` is available
        """
        try:
            sh.which(self.dssp_path)
            return True
        except:
            return False

    @typecheck
    def compute_gpde(
        self,
        pde_logits: Float["b pde n n"],  
        dist_logits: Float["b dist n n"],  
        dist_breaks: Float[" dist_break"],  
        tok_repr_atm_mask: Bool["b n"],  
    ) -> Float[" b"]:  
        """Compute global PDE following Section 5.7 of the AF3 supplement.

        :param pde_logits: [b pde n n] PDE logits
        :param dist_logits: [b dist n n] distance logits
        :param dist_breaks: [dist_break] distance breaks
        :param tok_repr_atm_mask: [b n] true if token representation atoms exists
        :return: [b] global PDE
        """

        dtype = pde_logits.dtype

        pde = self.compute_confidence_score.compute_pde(pde_logits, tok_repr_atm_mask)

        dist_logits = rearrange(dist_logits, "b dist i j -> b i j dist")
        dist_probs = F.softmax(dist_logits, dim=-1)

        # for distances greater than the last breaks
        dist_breaks = F.pad(dist_breaks.float(), (0, 1), value=1e6).type(dtype)
        contact_mask = dist_breaks < self.contact_mask_threshold

        contact_prob = einx.where(
            " dist, b i j dist, -> b i j dist", contact_mask, dist_probs, 0.0
        ).sum(dim=-1)

        mask = to_pairwise_mask(tok_repr_atm_mask)
        contact_prob = contact_prob * mask

        # Section 5.7 equation 16
        gpde = masked_average(pde, contact_prob, dim=(-1, -2))

        return gpde

    @typecheck
    def compute_lddt(
        self,
        pred_coords: Float["b m 3"],  
        true_coords: Float["b m 3"],  
        is_dna: Bool["b m"],  
        is_rna: Bool["b m"],  
        pairwise_mask: Bool["b m m"],  
        coords_mask: Bool["b m"] | None = None,  
    ) -> Float[" b"]:  
        """Compute lDDT.

        :param pred_coords: predicted coordinates
        :param true_coords: true coordinates
        :param is_dna: boolean tensor indicating DNA atoms
        :param is_rna: boolean tensor indicating RNA atoms
        :param pairwise_mask: boolean tensor indicating atompair for which LDDT is computed
        :param coords_mask: boolean tensor indicating valid atoms
        :return: lDDT
        """

        dtype = pred_coords.dtype
        atom_seq_len, device = pred_coords.shape[1], pred_coords.device

        # Compute distances between all pairs of atoms
        pred_dists = torch.cdist(pred_coords, pred_coords)
        true_dists = torch.cdist(true_coords, true_coords)

        # Compute distance difference for all pairs of atoms
        dist_diff = torch.abs(true_dists - pred_dists)

        lddt = einx.subtract('thresholds, ... -> ... thresholds', self.lddt_thresholds, dist_diff)
        lddt = (lddt >= 0).type(dtype).mean(dim=-1)

        # Restrict to bespoke inclusion radius
        is_nucleotide = is_dna | is_rna
        is_nucleotide_pair = to_pairwise_mask(is_nucleotide)

        inclusion_radius = torch.where(
            is_nucleotide_pair,
            true_dists < self.nucleic_acid_cutoff,
            true_dists < self.other_cutoff,
        )

        # Compute mean, avoiding self term
        mask = inclusion_radius & ~torch.eye(atom_seq_len, dtype=torch.bool, device=device)

        # Take into account variable lengthed atoms in batch
        if exists(coords_mask):
            paired_coords_mask = to_pairwise_mask(coords_mask)
            mask = mask & paired_coords_mask

        mask = mask * pairwise_mask

        # Calculate masked averaging
        lddt_mean = masked_average(lddt, mask, dim=(-1, -2))

        return lddt_mean

    @typecheck
    def compute_chain_pair_lddt(
        self,
        asym_mask_a: Bool["b m"] | Bool[" m"],  
        asym_mask_b: Bool["b m"] | Bool[" m"],  
        pred_coords: Float["b m 3"] | Float["m 3"],  
        true_coords: Float["b m 3"] | Float["m 3"],  
        is_molecule_types: Bool[f"b m {IS_MOLECULE_TYPES}"] | Bool[f"m {IS_MOLECULE_TYPES}"],  
        coords_mask: Bool["b m"] | Bool[" m"] | None = None,  
    ) -> Float[" b"]:  
        """Compute the plDDT between atoms marked by `asym_mask_a` and `asym_mask_b`.

        :param asym_mask_a: [b m] asym_mask_a
        :param asym_mask_b: [b m] asym_mask_b
        :param pred_coords: [b m 3] predicted coordinates
        :param true_coords: [b m 3] true coordinates
        :param is_molecule_types: [b m 2] is_molecule_types
        :param coords_mask: [b m] coords_mask
        :return: [b] lddt
        """

        if not exists(coords_mask):
            coords_mask = torch.ones_like(asym_mask_a)

        if asym_mask_a.ndim == 1:
            (
                asym_mask_a,
                asym_mask_b,
                pred_coords,
                true_coords,
                is_molecule_types,
                coords_mask,
            ) = map(lambda t: rearrange(t, '... -> 1 ...'), (
                asym_mask_a,
                asym_mask_b,
                pred_coords,
                true_coords,
                is_molecule_types,
                coords_mask,
            ))

        is_dna = is_molecule_types[..., IS_DNA_INDEX]
        is_rna = is_molecule_types[..., IS_RNA_INDEX]
        pairwise_mask = to_pairwise_mask(asym_mask_a)

        lddt = self.compute_lddt(
            pred_coords, true_coords, is_dna, is_rna, pairwise_mask, coords_mask
        )

        return lddt

    @typecheck
    def get_lddt_weight(
        self,
        type_chain_a: int,
        type_chain_b: int,
        lddt_type: Literal["interface", "intra-chain", "unresolved"],
        is_fine_tuning: bool = None,
    ) -> int:
        """Get a specified lDDT weight.

        :param type_chain_a: type of chain a
        :param type_chain_b: type of chain b
        :param lddt_type: lDDT type
        :param is_fine_tuning: is fine tuning
        :return: lDDT weight
        """
        is_fine_tuning = default(is_fine_tuning, self.is_fine_tuning)

        weight_dict = default(
            self.weight_dict_config,
            self.FINETUNING_DICT if is_fine_tuning else self.INITIAL_TRAINING_DICT,
        )

        if lddt_type == "unresolved":
            weight = weight_dict.get(lddt_type, {}).get(lddt_type, None)
            assert weight
            return weight

        interface_type = sorted([self.TYPE_MAPPING[type_chain_a], self.TYPE_MAPPING[type_chain_b]])
        interface_type = "-".join(interface_type)
        weight = weight_dict.get(interface_type, {}).get(lddt_type, None)
        assert weight, f"Weight not found for {interface_type} {lddt_type}"
        return weight

    @typecheck
    def compute_weighted_lddt(
        self,
        # atom level input
        pred_coords: Float["b m 3"],  
        true_coords: Float["b m 3"],  
        atom_mask: Bool["b m"] | None,  
        # token level input
        asym_id: Int["b n"],  
        is_molecule_types: Bool[f"b n {IS_MOLECULE_TYPES}"],  
        molecule_atom_lens: Int["b n"],  
        # additional input
        chains_list: List[Tuple[int, int] | Tuple[int]],
        is_fine_tuning: bool = None,
        unweighted: bool = False,
        # RASA input
        compute_rasa: bool = False,
        unresolved_cid: List[int] | None = None,
        unresolved_residue_mask: Bool["b n"] | None = None,  
        molecule_ids: Int["b n"] | None = None,  
    ) -> Float[" b"]:  
        """Compute the weighted lDDT.

        :param pred_coords: [b m 3] predicted coordinates
        :param true_coords: [b m 3] true coordinates
        :param atom_mask: [b m] atom mask
        :param asym_id: [b n] asym_id of each residue
        :param is_molecule_types: [b n 2] is_molecule_types
        :param molecule_atom_lens: [b n] molecule atom lens
        :param chains_list: List of chains
        :param is_fine_tuning: is fine tuning
        :param unweighted: unweighted lddt
        :param compute_rasa: compute RASA
        :param unresolved_cid: unresolved chain ids
        :param unresolved_residue_mask: unresolved residue mask
        :return: [b] weighted lddt
        """
        is_fine_tuning = default(is_fine_tuning, self.is_fine_tuning)

        device = pred_coords.device
        batch_size = pred_coords.shape[0]

        # broadcast asym_id and is_molecule_types to atom level
        atom_asym_id = batch_repeat_interleave(asym_id, molecule_atom_lens, output_padding_value=-1)
        atom_is_molecule_types = batch_repeat_interleave(is_molecule_types, molecule_atom_lens)

        weighted_lddt = torch.zeros(batch_size, device=device)

        for b in range(batch_size):
            chains = chains_list[b]
            if len(chains) == 2:
                asym_id_a = chains[0]
                asym_id_b = chains[1]
                lddt_type = "interface"
            elif len(chains) == 1:
                asym_id_a = asym_id_b = chains[0]
                lddt_type = "intra-chain"
            else:
                raise Exception(f"Invalid chain list {chains}")

            type_chain_a = get_cid_molecule_type(
                asym_id_a, atom_asym_id[b], atom_is_molecule_types[b], return_one_hot=False
            )
            type_chain_b = get_cid_molecule_type(
                asym_id_b, atom_asym_id[b], atom_is_molecule_types[b], return_one_hot=False
            )

            lddt_weight = self.get_lddt_weight(
                type_chain_a, type_chain_b, lddt_type, is_fine_tuning
            )

            asym_mask_a = atom_asym_id[b] == asym_id_a
            asym_mask_b = atom_asym_id[b] == asym_id_b

            lddt = self.compute_chain_pair_lddt(
                asym_mask_a,
                asym_mask_b,
                pred_coords[b],
                true_coords[b],
                atom_is_molecule_types[b],
                atom_mask[b],
            )

            weighted_lddt[b] = (1.0 if unweighted else lddt_weight) * lddt

        # Average the lDDT with the relative solvent accessible surface area (RASA) for unresolved proteins
        # NOTE: This differs from the AF3 Section 5.7 slightly, as here we compute the algebraic mean of the (batched) lDDT and RASA
        if compute_rasa:
            assert (
                exists(unresolved_cid) and exists(unresolved_residue_mask) and exists(molecule_ids)
            ), "RASA computation requires `unresolved_cid`, `unresolved_residue_mask`, and `molecule_ids` to be provided."
            weighted_rasa = self.compute_unresolved_rasa(
                unresolved_cid,
                unresolved_residue_mask,
                asym_id,
                molecule_ids,
                molecule_atom_lens,
                true_coords,
                atom_mask,
                is_fine_tuning=is_fine_tuning,
            )
            weighted_lddt = (weighted_lddt + weighted_rasa) / 2

        return weighted_lddt

    @typecheck
    def _compute_unresolved_rasa(
        self,
        unresolved_cid: int,
        unresolved_residue_mask: Bool[" n"],  
        asym_id: Int[" n"],  
        molecule_ids: Int[" n"],  
        molecule_atom_lens: Int[" n"],  
        atom_pos: Float["m 3"],  
        atom_mask: Bool[" m"],  
    ) -> Float[""]:  
        """Compute the unresolved relative solvent accessible surface area (RASA) for proteins.

        unresolved_cid: asym_id for protein chains with unresolved residues
        unresolved_residue_mask: True for unresolved residues, False for resolved residues
        asym_id: asym_id for each residue
        molecule_ids: molecule_ids for each residue
        molecule_atom_lens: number of atoms for each residue
        atom_pos: [m 3] atom positions
        atom_mask: True for valid atoms, False for missing/padding atoms
        :return: unresolved RASA
        """

        assert self.can_calculate_unresolved_protein_rasa, "`mkdssp` needs to be installed"

        residue_constants = get_residue_constants(res_chem_index=IS_PROTEIN)

        device = atom_pos.device
        dtype = atom_pos.dtype
        num_atom = atom_pos.shape[0]

        chain_mask = asym_id == unresolved_cid
        chain_unresolved_residue_mask = unresolved_residue_mask[chain_mask]
        chain_asym_id = asym_id[chain_mask]
        chain_molecule_ids = molecule_ids[chain_mask]
        chain_molecule_atom_lens = molecule_atom_lens[chain_mask]

        chain_mask_to_atom = torch.repeat_interleave(chain_mask, molecule_atom_lens)

        # if there's padding in num atom
        num_pad = num_atom - molecule_atom_lens.sum()
        if num_pad > 0:
            chain_mask_to_atom = F.pad(chain_mask_to_atom, (0, num_pad), value=False)

        chain_atom_pos = atom_pos[chain_mask_to_atom]
        chain_atom_mask = atom_mask[chain_mask_to_atom]

        structure = _protein_structure_from_feature(
            chain_asym_id,
            chain_molecule_ids,
            chain_molecule_atom_lens,
            chain_atom_pos,
            chain_atom_mask,
        )

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=True) as temp_file:
            temp_file_path = temp_file.name

            pdb_writer = PDBIO()
            pdb_writer.set_structure(structure)
            pdb_writer.save(temp_file_path)
            dssp = DSSP(structure[0], temp_file_path, dssp=self.dssp_path)
            dssp_dict = dict(dssp)

        rasa = []
        aatypes = []
        for residue in structure.get_residues():
            rsa = float(dssp_dict.get((residue.get_full_id()[2], residue.id))[3])
            rasa.append(rsa)

            aatype = dssp_dict.get((residue.get_full_id()[2], residue.id))[1]
            aatypes.append(residue_constants.restype_order[aatype])

        rasa = torch.tensor(rasa, dtype=dtype, device=device)
        aatypes = torch.tensor(aatypes, device=device).int()

        unresolved_aatypes = aatypes[chain_unresolved_residue_mask]
        unresolved_molecule_ids = chain_molecule_ids[chain_unresolved_residue_mask]

        assert torch.equal(
            unresolved_aatypes, unresolved_molecule_ids
        ), "aatype not match for input feature and structure"
        unresolved_rasa = rasa[chain_unresolved_residue_mask]

        return unresolved_rasa.mean()

    @typecheck
    def compute_unresolved_rasa(
        self,
        unresolved_cid: List[int],
        unresolved_residue_mask: Bool["b n"],  
        asym_id: Int["b n"],  
        molecule_ids: Int["b n"],  
        molecule_atom_lens: Int["b n"],  
        atom_pos: Float["b m 3"],  
        atom_mask: Bool["b m"],  
        is_fine_tuning: bool = None,
    ) -> Float[" b"]:  
        """Compute the unresolved relative solvent accessible surface area (RASA) for (batched)
        proteins.

        unresolved_cid: asym_id for protein chains with unresolved residues
        unresolved_residue_mask: True for unresolved residues, False for resolved residues
        asym_id: [b n] asym_id of each residue
        molecule_ids: [b n] molecule_ids of each residue
        molecule_atom_lens: [b n] molecule atom lens
        atom_pos: [b m 3] atom positions
        atom_mask: [b m] atom mask
        :return: [b] unresolved RASA
        """
        is_fine_tuning = default(is_fine_tuning, self.is_fine_tuning)

        weight_dict = default(
            self.weight_dict_config,
            self.FINETUNING_DICT if is_fine_tuning else self.INITIAL_TRAINING_DICT,
        )

        weight = weight_dict.get("unresolved", {}).get("unresolved", None)
        assert weight, f"Weight not found for unresolved"

        unresolved_rasa = [
            self._compute_unresolved_rasa(*args)
            for args in zip(
                unresolved_cid,
                unresolved_residue_mask,
                asym_id,
                molecule_ids,
                molecule_atom_lens,
                atom_pos,
                atom_mask,
            )
        ]
        return torch.stack(unresolved_rasa) * weight

    @typecheck
    def compute_model_selection_score(
        self,
        batch: BatchedAtomInput,
        samples: List[Sample],
        is_fine_tuning: bool = None,
        return_details: bool = False,
        return_unweighted_scores: bool = False,
        compute_rasa: bool = False,
        unresolved_cid: List[int] | None = None,
        unresolved_residue_mask: Bool["b n"] | None = None,
        missing_chain_index: int = -1,
    ) -> Float[" b"] | ScoreDetails:
        """Compute the model selection score for an input batch and corresponding (sampled) atom
        positions.

        :param batch: A batch of `AtomInput` data.
        :param samples: A list of sampled atom positions along with their predicted distance errors and labels.
        :param is_fine_tuning: is fine tuning
        :param return_details: return the top model and its score
        :param return_unweighted_scores: return the unweighted scores (i.e., lDDT)
        :param compute_rasa: compute the relative solvent accessible surface area (RASA) for unresolved proteins
        :param unresolved_cid: unresolved chain ids
        :param unresolved_residue_mask: unresolved residue mask
        :param missing_chain_index: missing chain index
        :return: [b] model selection score and optionally the top model
        """
        is_fine_tuning = default(is_fine_tuning, self.is_fine_tuning)

        if compute_rasa:
            if not (exists(unresolved_cid) and exists(unresolved_residue_mask)):
                logger.warning(
                    "RASA computation requires `unresolved_cid` and `unresolved_residue_mask` to be provided. Skipping RASA computation."
                )
                compute_rasa = False

        # collect required features

        batch_dict = batch.dict()

        atom_pos_true = batch_dict["atom_pos"]
        atom_mask = ~batch_dict["missing_atom_mask"]

        asym_id = batch_dict["additional_molecule_feats"].unbind(dim=-1)[2]
        is_molecule_types = batch_dict["is_molecule_types"]

        chains = [
            tuple(chain for chain in chains_list if chain != missing_chain_index)
            for chains_list in batch_dict["chains"].tolist()
        ]
        molecule_atom_lens = batch_dict["molecule_atom_lens"]
        molecule_ids = batch_dict["molecule_ids"]

        valid_atom_len_mask = batch_dict["molecule_atom_lens"] >= 0
        tok_repr_atm_mask = batch_dict["distogram_atom_indices"] >= 0 & valid_atom_len_mask

        # score samples

        scored_samples: List[ScoredSample] = []

        for sample_idx, sample in enumerate(samples):
            atom_pos_pred, pde_logits, plddt, dist_logits = sample

            weighted_lddt = self.compute_weighted_lddt(
                atom_pos_pred,
                atom_pos_true,
                atom_mask,
                asym_id,
                is_molecule_types,
                molecule_atom_lens,
                chains_list=chains,
                is_fine_tuning=is_fine_tuning,
                compute_rasa=compute_rasa,
                unresolved_cid=unresolved_cid,
                unresolved_residue_mask=unresolved_residue_mask,
                molecule_ids=molecule_ids,
                unweighted=return_unweighted_scores,
            )

            gpde = self.compute_gpde(
                pde_logits,
                dist_logits,
                self.dist_breaks,
                tok_repr_atm_mask,
            )

            scored_samples.append((sample_idx, atom_pos_pred, plddt, weighted_lddt, gpde))

        # quick collate

        *_, all_weighted_lddt, all_gpde = zip(*scored_samples)

        # rank by batch-averaged minimum gPDE

        best_gpde_index = torch.stack(all_gpde).mean(dim=-1).argmin().item()

        # rank by batch-averaged maximum lDDT

        best_lddt_index = torch.stack(all_weighted_lddt).mean(dim=-1).argmax().item()

        # some weighted score

        model_selection_score = (
            scored_samples[best_gpde_index][-2] + scored_samples[best_lddt_index][-2]
        ) / 2

        if not return_details:
            return model_selection_score

        score_details = ScoreDetails(
            best_gpde_index=best_gpde_index,
            best_lddt_index=best_lddt_index,
            score=model_selection_score,
            scored_samples=scored_samples,
        )

        return score_details

    @typecheck
    def forward(
        self, alphafolds: Tuple[Alphafold3], batched_atom_inputs: BatchedAtomInput, **kwargs
    ) -> Float[" b"] | ScoreDetails:
        """Make model selections by computing the model selection score.

        NOTE: Give this function a tuple of `Alphafold3` modules and a batch of atomic inputs, and it will
        select the best module via the model selection score by returning the index of the corresponding tuple.

        :param alphafolds: Tuple of `Alphafold3` modules
        :param batched_atom_inputs: A batch of `AtomInput` data
        :param kwargs: Additional keyword arguments
        :return: Model selection score
        """

        samples = []

        with torch.no_grad():
            for alphafold in alphafolds:
                alphafold.eval()

                pred_atom_pos, logits = alphafold(
                    **batched_atom_inputs.model_forward_dict(),
                    return_loss=False,
                    return_confidence_head_logits=True,
                    return_distogram_head_logits=True,
                )
                plddt = self.compute_confidence_score.compute_plddt(logits.plddt)

                samples.append((pred_atom_pos, logits.pde, plddt, logits.distance))

        scores = self.compute_model_selection_score(batched_atom_inputs, samples=samples, **kwargs)

        return scores
