class MultiChainPermutationAlignment(Module):
    """Section 4.2 of the AlphaFold 3 Supplement."""

    @typecheck
    def __init__(
        self,
        weighted_rigid_align: WeightedRigidAlign,
        **kwargs,
    ):
        super().__init__()
        self.weighted_rigid_align = weighted_rigid_align

    @staticmethod
    @typecheck
    def split_ground_truth_labels(gt_features: Dict[str, Tensor]) -> List[Dict[str, Tensor]]:
        """Split ground truth features according to chains.

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param gt_features: A dictionary within a PyTorch Dataset iteration, which is returned by
            the upstream DataLoader.iter() method. In the DataLoader pipeline, all tensors
            belonging to all the ground truth chains are concatenated. This function is needed to
            1) detect the number of chains, i.e., unique(asym_id) and 2) split the concatenated
            tensors back to individual ones that correspond to individual asym_ids.
        :return: A list of feature dictionaries with only necessary ground truth features required
            to finish multi-chain permutation. E.g., it will be a list of 5 elements if there are 5
            chains in total.
        """
        _, asym_id_counts = torch.unique(
            gt_features["asym_id"], sorted=True, return_counts=True, dim=-1
        )
        n_res = gt_features["asym_id"].shape[-1]

        def split_dim(shape):
            """Return the dimension index where the size is n_res."""
            return next(iter(i for i, size in enumerate(shape) if size == n_res), None)

        labels = list(
            map(
                dict,
                zip(
                    *[
                        [
                            (k, v)
                            for v in torch.split(
                                v_all, asym_id_counts.tolist(), dim=split_dim(v_all.shape)
                            )
                        ]
                        for k, v_all in gt_features.items()
                        if n_res in v_all.shape
                    ]
                ),
            )
        )
        return labels

    @staticmethod
    @typecheck
    def get_per_asym_token_index(features: Dict[str, Tensor], padding_value: int = -1) -> Dict[int, Int["b ..."]]:  # type: ignore
        """A function that retrieves a mapping denoting which token belong to which `asym_id`.

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param features: A dictionary that contains input features after cropping.
        :return: A dictionary that records which region of the sequence belongs to which `asym_id`.
        """
        batch_size = features["token_index"].shape[0]

        unique_asym_ids = [i for i in torch.unique(features["asym_id"]) if i != padding_value]
        per_asym_token_index = {}
        for cur_asym_id in unique_asym_ids:
            asym_mask = (features["asym_id"] == cur_asym_id).bool()
            per_asym_token_index[int(cur_asym_id)] = rearrange(
                features["token_index"][asym_mask], "(b a) -> b a", b=batch_size
            )

        return per_asym_token_index

    @staticmethod
    @typecheck
    def get_entity_to_asym_list(
        features: Dict[str, Tensor], no_gaps: bool = False
    ) -> Dict[int, Tensor]:
        """Generate a dictionary mapping unique entity IDs to lists of unique asymmetry IDs
        (asym_id) for each entity.

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param features: A dictionary containing data features, including `entity_id` and `asym_id` tensors.
        :param no_gaps: Whether to remove gaps in the `asym_id` values.
        :return: A dictionary where keys are unique entity IDs, and values are tensors of unique asymmetry IDs
            associated with each entity.
        """
        entity_to_asym_list = {}
        unique_entity_ids = torch.unique(features["entity_id"])

        # First pass: Collect all unique `cur_asym_id` values across all entities
        all_asym_ids = set()
        for cur_ent_id in unique_entity_ids:
            ent_mask = features["entity_id"] == cur_ent_id
            cur_asym_id = torch.unique(features["asym_id"][ent_mask])
            entity_to_asym_list[int(cur_ent_id)] = cur_asym_id
            all_asym_ids.update(cur_asym_id.tolist())

        # Second pass: Remap `asym_id` values to remove any gaps
        if no_gaps:
            sorted_asym_ids = sorted(all_asym_ids)
            remap_dict = {old_id: new_id for new_id, old_id in enumerate(sorted_asym_ids)}

            for cur_ent_id in entity_to_asym_list:
                cur_asym_id = entity_to_asym_list[cur_ent_id]
                remapped_asym_id = torch.tensor([remap_dict[id.item()] for id in cur_asym_id])
                entity_to_asym_list[cur_ent_id] = remapped_asym_id

        return entity_to_asym_list

    @typecheck
    def get_least_asym_entity_or_longest_length(
        self, batch: Dict[str, Tensor], input_asym_id: List[int], padding_value: int = -1
    ) -> Tuple[Tensor, List[Tensor]]:
        """Check how many subunit(s) one sequence has. Select the subunit that is less common,
        e.g., if the protein was AABBB then select one of the As as an anchor.

        If there is a tie, e.g. AABB, first check which sequence is the longest,
        then choose one of the corresponding subunits as an anchor.

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param batch: In this function, `batch` is the full ground truth features.
        :param input_asym_id: A list of `asym_ids` that are in the cropped input features.
        :param padding_value: The padding value used in the input features.
        :return: Selected ground truth `asym_ids` and a list of
            integer tensors denoting of all possible pred anchor candidates.
        """
        entity_to_asym_list = self.get_entity_to_asym_list(features=batch)
        unique_entity_ids = [i for i in torch.unique(batch["entity_id"]) if i != padding_value]
        entity_asym_count = {}
        entity_length = {}

        all_asym_ids = set()

        for entity_id in unique_entity_ids:
            asym_ids = torch.unique(batch["asym_id"][batch["entity_id"] == entity_id])

            all_asym_ids.update(asym_ids.tolist())

            # Make sure some asym IDs associated with ground truth entity ID exist in cropped prediction
            asym_ids_in_pred = [a for a in asym_ids if a in input_asym_id]
            if not asym_ids_in_pred:
                continue

            entity_asym_count[int(entity_id)] = len(asym_ids)

            # Calculate entity length
            entity_mask = batch["entity_id"] == entity_id
            entity_length[int(entity_id)] = entity_mask.sum().item()

        min_asym_count = min(entity_asym_count.values())
        least_asym_entities = [
            entity for entity, count in entity_asym_count.items() if count == min_asym_count
        ]

        # If multiple entities have the least asym_id count, return those with the longest length
        if len(least_asym_entities) > 1:
            max_length = max([entity_length[entity] for entity in least_asym_entities])
            least_asym_entities = [
                entity for entity in least_asym_entities if entity_length[entity] == max_length
            ]

        # If there are still multiple entities, return a random one
        if len(least_asym_entities) > 1:
            least_asym_entities = [random.choice(least_asym_entities)]  # nosec

        assert (
            len(least_asym_entities) == 1
        ), "There should be only one entity with the least `asym_id` count."
        least_asym_entities = least_asym_entities[0]

        anchor_gt_asym_id = random.choice(entity_to_asym_list[least_asym_entities])  # nosec
        anchor_pred_asym_ids = [
            asym_id
            for asym_id in entity_to_asym_list[least_asym_entities]
            if asym_id in input_asym_id
        ]

        # Remap `asym_id` values to remove any gaps in the ground truth asym IDs,
        # but leave the prediction asym IDs as is since they are used for masking
        sorted_asym_ids = sorted(all_asym_ids)
        remap_dict = {old_id: new_id for new_id, old_id in enumerate(sorted_asym_ids)}

        remapped_anchor_gt_asym_id = torch.tensor([remap_dict[anchor_gt_asym_id.item()]])

        return remapped_anchor_gt_asym_id, anchor_pred_asym_ids

    @staticmethod
    @typecheck
    def calculate_input_mask(
        true_masks: List[Int["b ..."]],  # type: ignore
        anchor_gt_idx: int,
        asym_mask: Bool["b n"],  # type: ignore
        pred_mask: Float["b n"],  # type: ignore
    ) -> Bool["b a"]:  # type: ignore
        """Calculate an input mask for downstream optimal transformation computation.

        :param true_masks: A list of masks from the ground truth chains. E.g., it will be a length
            of 5 if there are 5 chains in ground truth structure.
        :param anchor_gt_idx: A tensor with one integer in it (i.e., the index of selected ground
            truth anchor).
        :param asym_mask: A boolean tensor with which to mask out other elements in a tensor if
            they do not belong to a specific asym ID.
        :param pred_mask: A boolean tensor corresponding to the mask with which to mask the
            predicted features.
        :return: A boolean mask.
        """
        batch_size = pred_mask.shape[0]
        anchor_pred_mask = rearrange(
            pred_mask[asym_mask],
            "(b a) -> b a",
            b=batch_size,
        )
        anchor_true_mask = true_masks[anchor_gt_idx]
        input_mask = (anchor_true_mask * anchor_pred_mask).bool()
        return input_mask

    @typecheck
    def calculate_optimal_transform(
        self,
        true_poses: List[Float["b ... 3"]],  # type: ignore
        anchor_gt_idx: int,
        true_masks: List[Int["b ..."]],  # type: ignore
        pred_mask: Float["b n"],  # type: ignore
        asym_mask: Bool["b n"],  # type: ignore
        pred_pos: Float["b n 3"],  # type: ignore
    ) -> Tuple[Float["b 3 3"], Float["b 1 3"]]:  # type: ignore
        """Take the selected anchor ground truth token center atom positions and the selected
        predicted anchor token center atom position and then calculate the optimal rotation matrix
        to align the ground-truth anchor and predicted anchor.

        Process:
        1) Select an anchor chain from ground truth, denoted by anchor_gt_idx, and an anchor chain from the predicted structure.
            Both anchor_gt and anchor_pred have exactly the same sequence.
        2) Obtain the token center atom positions corresponding to the selected anchor_gt,
            done be slicing the true_pose according to anchor_gt_token
        3) Calculate the optimal transformation that can best align the token center atoms of anchor_pred to those of anchor_gt
            via the Kabsch algorithm (https://en.wikipedia.org/wiki/Kabsch_algorithm).

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param true_poses: A list of tensors, corresponding to the token center atom positions of the ground truth structure.
            E.g., If there are 5 chains, this list will have a length of 5.
        :param anchor_gt_idx: A tensor with one integer in it (i.e., the index of selected ground truth anchor).
        :param true_masks: list of masks from the ground truth chains. E.g., it will be a length of 5 if there are
            5 chains in ground truth structure.
        :param pred_mask: A boolean tensor corresponding to the mask with which to mask the predicted features.
        :param asym_mask: A boolean tensor with which to mask out other elements in a tensor if they do not belong
            to a specific asym ID.
        :param pred_pos: A tensor of predicted token center atom positions.
        :return: A rotation matrix that records the optimal rotation that will best align the selected anchor prediction to the
            selected anchor truth as well as a matrix that records how the atoms should be shifted after applying `r`.
            N.b., Optimal alignment requires 1) a rotation and 2) a shift of the positions.
        """
        dtype = pred_pos.dtype
        batch_size = pred_pos.shape[0]

        input_mask = self.calculate_input_mask(
            true_masks=true_masks,
            anchor_gt_idx=anchor_gt_idx,
            asym_mask=asym_mask,
            pred_mask=pred_mask,
        )
        anchor_true_pos = true_poses[anchor_gt_idx]
        anchor_pred_pos = rearrange(
            pred_pos[asym_mask],
            "(b a) ... -> b a ...",
            b=batch_size,
        )
        _, r, x = self.weighted_rigid_align(
            pred_coords=anchor_pred_pos.float(),
            true_coords=anchor_true_pos.float(),
            mask=input_mask,
            return_transforms=True,
        )

        return r.type(dtype), x.type(dtype)

    @staticmethod
    @typecheck
    def apply_transform(pose: Float["b a 3"], r: Float["b 3 3"], x: Float["b 1 3"]) -> Float["b a 3"]:  # type: ignore
        """Apply the optimal transformation to the predicted token center atom positions.

        :param pose: A tensor of predicted token center atom positions.
        :param r: A rotation matrix that records the optimal rotation that will best align the selected anchor prediction to the
            selected anchor truth.
        :param x: A matrix that records how the atoms should be shifted after applying `r`.
        :return: A tensor of predicted token center atom positions after applying the optimal transformation.
        """
        aligned_pose = einsum(r, pose, "b i j, b n j -> b n i") + x
        aligned_pose.detach_()
        return aligned_pose

    @staticmethod
    @typecheck
    def batch_compute_rmsd(
        true_pos: Float["b a 3"],  # type: ignore
        pred_pos: Float["b a 3"],  # type: ignore
        mask: Bool["b a"] | None = None,  # type: ignore
        eps: float = 1e-6,
    ) -> Float["b"]:  # type: ignore
        """Calculate the root-mean-square deviation (RMSD) between predicted and ground truth
        coordinates.

        :param true_pos: The ground truth coordinates.
        :param pred_pos: The predicted coordinates.
        :param mask: The mask tensor.
        :param eps: A small value to prevent division by zero.
        :return: The RMSD.
        """
        # Apply mask if provided
        if exists(mask):
            true_coords = einx.where("b a, b a c, -> b a c", mask, true_pos, 0.0)
            pred_coords = einx.where("b a, b a c, -> b a c", mask, pred_pos, 0.0)

        # Compute squared differences across the last dimension (which is of size 3)
        sq_diff = torch.square(true_coords - pred_coords).sum(dim=-1)  # [b, m]

        # Compute mean squared deviation per batch
        msd = torch.mean(sq_diff, dim=-1)  # [b]

        # Replace NaN values with a large number to avoid issues
        msd = torch.nan_to_num(msd, nan=1e8)

        # Return the root mean square deviation per batch
        return torch.sqrt(msd + eps)  # [b]

    @typecheck
    def greedy_align(
        self,
        batch: Dict[str, Tensor],
        entity_to_asym_list: Dict[int, Tensor],
        pred_pos: Float["b n 3"],  # type: ignore
        pred_mask: Float["b n"],  # type: ignore
        true_poses: List[Float["b ... 3"]],  # type: ignore
        true_masks: List[Int["b ..."]],  # type: ignore
        padding_value: int = -1,
    ) -> List[Tuple[int, int]]:
        """
        Implement Algorithm 4 in the Supplementary Information of the AlphaFold-Multimer paper:
            Evans, R et al., 2022 Protein complex prediction with AlphaFold-Multimer,
            bioRxiv 2021.10.04.463034; doi: https://doi.org/10.1101/2021.10.04.463034

        NOTE: The tuples in the returned list begin are zero-indexed.

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param batch: A dictionary of ground truth features.
        :param entity_to_asym_list: A dictionary recording which asym ID(s) belong to which entity ID.
        :param pred_pos: Predicted positions of token center atoms from the results of model.forward().
        :param pred_mask: A boolean tensor that masks `pred_pos`.
        :param true_poses: A list of tensors, corresponding to the token center atom positions of the ground truth structure.
            E.g., if there are 5 chains, this list will have a length of 5.
        :param true_masks: A list of tensors, corresponding to the masks of the token center atom positions of the ground truth structure.
            E.g., if there are 5 chains, this list will have a length of 5.
        :param padding_value: The padding value used in the input features.
        :return: A list of tuple(int, int) that provides instructions for how the ground truth chains should be permuted.
            E.g., if 3 chains in the input structure have the same sequences, an example return value would be:
            `[(0, 2), (1, 1), (2, 0)]`, meaning the first chain in the predicted structure should be aligned
            to the third chain in the ground truth and the second chain in the predicted structure is fine
            to stay with the second chain in the ground truth.
        """
        batch_size = pred_pos.shape[0]

        used = [
            # NOTE: This is a list the keeps a record of whether a ground truth chain has been used.
            False
            for _ in range(len(true_poses))
        ]
        alignments = []

        unique_asym_ids = [i for i in torch.unique(batch["asym_id"]) if i != padding_value]

        for cur_asym_id in unique_asym_ids:
            i = int(cur_asym_id)

            asym_mask = batch["asym_id"] == cur_asym_id
            cur_entity_ids = rearrange(
                batch["entity_id"][asym_mask],
                "(b a) -> b a",
                b=batch_size,
            )

            # NOTE: Here, we assume there can be multiple unique entity IDs associated
            # with a given asym ID. This is a valid assumption when the original batch
            # contains a single unique structure that has one or more chains spread
            # across multiple entities (e.g., in the case of ligands residing in
            # a protein-majority chain).

            unique_cur_entity_ids = torch.unique(cur_entity_ids, dim=-1).unbind(dim=-1)

            for batch_cur_entity_id in unique_cur_entity_ids:
                cur_pred_pos = rearrange(
                    pred_pos[asym_mask],
                    "(b a) ... -> b a ...",
                    b=batch_size,
                )
                cur_pred_mask = rearrange(
                    pred_mask[asym_mask],
                    "(b a) -> b a",
                    b=batch_size,
                )

                best_rmsd = torch.inf
                best_idx = None

                # NOTE: Here, we assume there is only one unique entity ID per batch,
                # which is a valid assumption only when the original batch size is 1
                # (meaning only a single unique structure is represented in the batch).

                unique_cur_entity_id = torch.unique(batch_cur_entity_id)
                assert (
                    len(unique_cur_entity_id) == 1
                ), "There should be only one unique entity ID per batch."
                cur_asym_list = entity_to_asym_list[int(unique_cur_entity_id)]

                for next_asym_id in cur_asym_list:
                    j = int(next_asym_id)

                    if not used[j]:  # NOTE: This is a possible candidate.
                        cropped_pos = true_poses[j]
                        mask = true_masks[j]

                        rmsd = self.batch_compute_rmsd(
                            true_pos=cropped_pos.mean(1, keepdim=True),
                            pred_pos=cur_pred_pos.mean(1, keepdim=True),
                            mask=(
                                cur_pred_mask.any(-1, keepdim=True) * mask.any(-1, keepdim=True)
                            ),
                        ).mean()

                        if rmsd < best_rmsd:
                            # NOTE: We choose the permutation that minimizes the batch-wise
                            # average RMSD of the predicted token center atom centroid coordinates
                            # with respect to the ground truth token center atom centroid coordinates.
                            best_rmsd = rmsd
                            best_idx = j

                if exists(best_idx):
                    # NOTE: E.g., for ligands within a protein-majority chain, we may have
                    # multiple unique entity IDs associated with a given asym ID. In this case,
                    # we need to ensure that we do not reuse a chain that has already been used
                    # in the permutation alignment process.
                    used[best_idx] = True
                    alignments.append((i, best_idx))

        assert all(used), "All chains should be used in the permutation alignment process."
        return alignments

    @staticmethod
    @typecheck
    def pad_features(feature_tensor: Tensor, num_tokens_pad: int, pad_dim: int) -> Tensor:
        """Pad an input feature tensor. Padding values will be 0 and put behind the true feature
        values.

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param feature_tensor: A feature tensor to pad.
        :param num_tokens_pad: The number of tokens to pad.
        :param pad_dim: Along which dimension of `feature_tensor` to pad.
        :return: A padded feature tensor.
        """
        pad_shape = list(feature_tensor.shape)
        pad_shape[pad_dim] = num_tokens_pad
        padding_tensor = feature_tensor.new_zeros(pad_shape, device=feature_tensor.device)
        return torch.concat((feature_tensor, padding_tensor), dim=pad_dim)

    @typecheck
    def merge_labels(
        self,
        labels: List[Dict[str, Tensor]],
        alignments: List[Tuple[int, int]],
        original_num_tokens: int,
        dimension_to_merge: int = 1,
    ) -> Dict[str, Tensor]:
        """Merge ground truth labels according to permutation results.

        Adapted from:
        https://github.com/dptech-corp/Uni-Fold/blob/b1c89a2cebd4e4ee4c47b4e443f92beeb9138fbb/unifold/losses/chain_align.py#L176C1-L176C1
        and
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param labels: A list of original ground truth feats. E.g., if there are 5 chains,
            `labels` will have a length of 5.
        :param alignments: A list of tuples, each entry specifying the corresponding label of the asym ID.
        :param original_num_tokens: An integer corresponding to the number of tokens specified
            by one's (e.g., training-time) crop size.
        :param dimension_to_merge: The dimension along which to merge the labels.
        :return: A new dictionary of permuted ground truth features.
        """
        outs = {}
        for k in labels[0].keys():
            cur_out = {}
            for i, j in alignments:
                label = labels[j][k]
                cur_out[i] = label

            cur_out = [x[1] for x in sorted(cur_out.items())]
            if len(cur_out) > 0:
                new_v = torch.concat(cur_out, dim=dimension_to_merge)

                # Check whether padding is needed.
                if new_v.shape[dimension_to_merge] != original_num_tokens:
                    num_tokens_pad = original_num_tokens - new_v.shape[dimension_to_merge]
                    new_v = self.pad_features(new_v, num_tokens_pad, pad_dim=dimension_to_merge)

                outs[k] = new_v

        return outs

    @typecheck
    def compute_permutation_alignment(
        self,
        out: Dict[str, Tensor],
        features: Dict[str, Tensor],
        ground_truth: Dict[str, Tensor],
        padding_value: int = -1,
    ) -> List[Tuple[int, int]]:
        """A method that permutes chains in ground truth before calculating the loss because the
        mapping between the predicted and ground truth will become arbitrary. The model cannot be
        assumed to predict chains in the same order as the ground truth. Thus, this function picks
        the optimal permutation of predicted chains that best matches the ground truth, by
        minimising the RMSD (i.e., the best permutation of ground truth chains is selected based on
        which permutation has the lowest RMSD calculation).

        Details are described in Section 7.3 in the Supplementary of AlphaFold-Multimer paper:
        https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param out: A dictionary of output tensors from model.forward().
        :param features: A dictionary of feature tensors that are used as input for model.forward().
        :param ground_truth: A list of dictionaries of features corresponding to chains in ground truth structure.
            E.g., it will be a length of 5 if there are 5 chains in ground truth structure.
        :param padding_value: The padding value used in the input features.
        :return: A list of tuple(int, int) that instructs how ground truth chains should be permutated.
        """
        num_tokens = features["token_index"].shape[-1]

        unique_asym_ids = set(torch.unique(features["asym_id"]).tolist())
        unique_asym_ids.discard(padding_value)  # Remove padding value
        is_monomer = len(unique_asym_ids) == 1

        per_asym_token_index = self.get_per_asym_token_index(
            features=features, padding_value=padding_value
        )

        if is_monomer:
            best_alignments = list(enumerate(range(len(per_asym_token_index))))
            return best_alignments

        best_rmsd = torch.inf
        best_alignments = None

        # 1. Choose the least ambiguous ground truth "anchor" chain.
        # For example, in an A3B2 complex an arbitrary B chain is chosen.
        # In the event of a tie e.g., A2B2 stoichiometry, the longest chain
        # is chosen, with the hope that in general the longer chains are
        # likely to have higher confidence predictions.

        # 2. Select the prediction anchor chain from the set of all prediction
        # chains with the same sequence as the ground truth anchor chain.

        anchor_gt_asym, anchor_pred_asym_ids = self.get_least_asym_entity_or_longest_length(
            batch=ground_truth,
            input_asym_id=list(unique_asym_ids),
        )
        entity_to_asym_list = self.get_entity_to_asym_list(features=ground_truth, no_gaps=True)
        labels = self.split_ground_truth_labels(gt_features=ground_truth)
        anchor_gt_idx = int(anchor_gt_asym)

        # 3. Optimally align the ground truth anchor chain to the prediction
        # anchor chain using a rigid alignment algorithm.

        pred_pos = out["pred_coords"]
        pred_mask = out["mask"].to(dtype=pred_pos.dtype)

        true_poses = [l["true_coords"] for l in labels]
        true_masks = [l["mask"].long() for l in labels]

        # Assignment Stage - Section 7.3.2 of the AlphaFold-Multimer Paper

        # 1. Greedily assign each of the predicted chains to their nearest
        # neighbour of the same sequence in the ground truth. These assignments
        # define the optimal permutation to apply to the ground truth chains.
        # Nearest neighbours are defined as the chains with the smallest distance
        # between the average of their token center atom coordinates.

        # 2. Repeat the above alignment and assignment stages for all valid choices
        # of the prediction anchor chain given the ground truth anchor chain.

        # 3. Finally, we pick the permutation that minimizes the RMSD between the
        # token center atom coordinate averages of the predicted and ground truth chains.

        for candidate_pred_anchor in anchor_pred_asym_ids:
            asym_mask = (features["asym_id"] == candidate_pred_anchor).bool()

            r, x = self.calculate_optimal_transform(
                true_poses=true_poses,
                anchor_gt_idx=anchor_gt_idx,
                true_masks=true_masks,
                pred_mask=pred_mask,
                asym_mask=asym_mask,
                pred_pos=pred_pos,
            )

            # Apply transforms.
            aligned_true_poses = [
                self.apply_transform(pose.to(r.dtype), r, x) for pose in true_poses
            ]

            alignments = self.greedy_align(
                batch=features,
                entity_to_asym_list=entity_to_asym_list,
                pred_pos=pred_pos,
                pred_mask=pred_mask,
                true_poses=aligned_true_poses,
                true_masks=true_masks,
            )

            merged_labels = self.merge_labels(
                labels=labels,
                alignments=alignments,
                original_num_tokens=num_tokens,
            )

            aligned_true_pos = self.apply_transform(merged_labels["true_coords"].to(r.dtype), r, x)

            rmsd = self.batch_compute_rmsd(
                true_pos=aligned_true_pos.mean(1, keepdim=True),
                pred_pos=pred_pos.mean(1, keepdim=True),
                mask=(
                    pred_mask.any(-1, keepdim=True) * merged_labels["mask"].any(-1, keepdim=True)
                ),
            ).mean()

            if rmsd < best_rmsd:
                # NOTE: We choose the permutation that minimizes the batch-wise
                # average RMSD of the predicted token center atom centroid coordinates
                # with respect to the ground truth token center atom centroid coordinates.
                best_rmsd = rmsd
                best_alignments = alignments

        # NOTE: The above algorithm naturally generalizes to both training and inference
        # contexts (i.e., with and without cropping) by, where applicable, pre-applying
        # cropping to the (ground truth) input coordinates and features.

        assert exists(best_alignments), "Best alignments must be found."
        return best_alignments

    @typecheck
    def forward(
        self,
        pred_coords: Float["b m 3"],  # type: ignore - predicted coordinates
        true_coords: Float["b m 3"],  # type: ignore - true coordinates
        molecule_atom_lens: Int["b n"],  # type: ignore - molecule atom lengths
        molecule_atom_indices: Int["b n"],  # type: ignore - molecule atom indices
        token_bonds: Bool["b n n"],  # type: ignore - token bonds
        additional_molecule_feats: Int[f"b n {ADDITIONAL_MOLECULE_FEATS}"] | None = None,  # type: ignore - additional molecule features
        is_molecule_types: Bool[f"b n {IS_MOLECULE_TYPES}"] | None = None,  # type: ignore - molecule types
        mask: Bool["b m"] | None = None,  # type: ignore - mask for variable lengths
    ) -> Float["b m 3"]:  # type: ignore
        """Compute the multi-chain permutation alignment.

        NOTE: This function assumes that the ground truth features are batched yet only contain
        features for the same structure. This is the case after performing data augmentation
        with a batch size of 1 in the `Alphafold3` module's forward pass. If the batched
        ground truth features represent multiple different structures, this function will not
        return correct results.

        :param pred_coords: Predicted coordinates.
        :param true_coords: True coordinates.
        :param molecule_atom_lens: The molecule atom lengths.
        :param molecule_atom_indices: The molecule atom indices.
        :param token_bonds: The token bonds.
        :param is_molecule_types: Molecule type of each atom.
        :param mask: The mask for variable lengths.
        :return: The optimally chain-permuted aligned coordinates.
        """
        num_atoms = pred_coords.shape[1]

        if not exists(additional_molecule_feats) or not exists(is_molecule_types):
            # NOTE: If no chains or no molecule types are specified,
            # we cannot perform multi-chain permutation alignment.
            true_coords.detach_()
            return true_coords

        if exists(mask):
            # Zero out all predicted and true coordinates where not an atom.
            pred_coords = einx.where("b n, b n c, -> b n c", mask, pred_coords, 0.0)
            true_coords = einx.where("b n, b n c, -> b n c", mask, true_coords, 0.0)

        # Alignment Stage - Section 7.3.1 of the AlphaFold-Multimer Paper

        _, token_index, token_asym_id, token_entity_id, _ = additional_molecule_feats.unbind(
            dim=-1
        )

        # NOTE: Ligands covalently bonded to polymer chains are to be permuted
        # in sync with the corresponding chains by assigning them the same
        # asymmetric unit ID (asym_id) to group all covalently bonded
        # components together.
        polymer_indices = [IS_PROTEIN_INDEX, IS_RNA_INDEX, IS_DNA_INDEX]
        ligand_indices = [IS_LIGAND_INDEX, IS_METAL_ION_INDEX]

        is_polymer_types = is_molecule_types[..., polymer_indices].any(-1)
        is_ligand_types = is_molecule_types[..., ligand_indices].any(-1)

        polymer_ligand_pair_mask = is_polymer_types[..., None] & is_ligand_types[..., None, :]
        polymer_ligand_pair_mask = polymer_ligand_pair_mask | polymer_ligand_pair_mask.transpose(
            -1, -2
        )

        covalent_bond_mask = polymer_ligand_pair_mask & token_bonds

        is_covalent_residue_mask = covalent_bond_mask.any(-1)
        is_covalent_ligand_mask = is_ligand_types & is_covalent_residue_mask

        # NOTE: Covalent ligand-polymer bond pairs may be many-to-many, so
        # we need to group them together by assigning covalent ligands the same
        # asym IDs as the polymer chains to which they are most frequently bonded.
        covalent_bonded_asym_id = torch.where(
            covalent_bond_mask, token_asym_id[..., None], torch.tensor(float("nan"))
        )

        covalent_bond_mode_values, _ = covalent_bonded_asym_id.mode(dim=-1, keepdim=False)
        mapped_token_asym_id = torch.where(
            is_covalent_ligand_mask, covalent_bond_mode_values, token_asym_id
        )
        mapped_atom_asym_id = batch_repeat_interleave(mapped_token_asym_id, molecule_atom_lens)

        # Move ligand coordinates to be adjacent to their covalently bonded polymer chains.
        _, mapped_atom_sorted_indices = torch.sort(mapped_atom_asym_id, dim=1)
        mapped_atom_true_coords = torch.gather(
            true_coords, dim=1, index=mapped_atom_sorted_indices.unsqueeze(-1).expand(-1, -1, 3)
        )

        # Segment the ground truth coordinates into chains.
        labels = self.split_ground_truth_labels(
            dict(asym_id=mapped_atom_asym_id, true_coords=mapped_atom_true_coords)
        )

        # Pool atom-level features into token-level features.
        mol_atom_indices = repeat(molecule_atom_indices, "b m -> b m d", d=true_coords.shape[-1])

        token_pred_coords = torch.gather(pred_coords, 1, mol_atom_indices)
        token_true_coords = torch.gather(true_coords, 1, mol_atom_indices)
        token_mask = torch.gather(mask, 1, molecule_atom_indices)

        # Permute ground truth chains.
        out = {"pred_coords": token_pred_coords, "mask": token_mask}
        features = {
            "asym_id": token_asym_id,
            "entity_id": token_entity_id,
            "token_index": token_index,
        }
        ground_truth = {
            "true_coords": token_true_coords,
            "mask": token_mask,
            "asym_id": token_asym_id,
            "entity_id": token_entity_id,
        }

        alignments = self.compute_permutation_alignment(
            out=out,
            features=features,
            ground_truth=ground_truth,
        )

        # Reorder ground truth coordinates according to permutation results.
        labels = self.merge_labels(
            labels=labels,
            alignments=alignments,
            original_num_tokens=num_atoms,
        )

        permuted_true_coords = labels["true_coords"].detach()
        return permuted_true_coords
