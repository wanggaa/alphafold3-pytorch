
class WeightedRigidAlign(Module):
    """Algorithm 28."""

    @typecheck
    @autocast("cuda", enabled=False)
    def forward(
        self,
        pred_coords: Float["b m 3"],  # type: ignore - predicted coordinates
        true_coords: Float["b m 3"],  # type: ignore - true coordinates
        weights: Float["b m"] | None = None,  # type: ignore - weights for each atom
        mask: Bool["b m"] | None = None,  # type: ignore - mask for variable lengths
        return_transforms: bool = False,
    ) -> Union[Float["b m 3"], Tuple[Float["b m 3"], Float["b 3 3"], Float["b 1 3"]]]:  # type: ignore
        """Compute the weighted rigid alignment.

        The check for ambiguous rotation and low rank of cross-correlation between aligned point
        clouds is inspired by
        https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/points_alignment.html.

        :param pred_coords: Predicted coordinates.
        :param true_coords: True coordinates.
        :param weights: Weights for each atom.
        :param mask: The mask for variable lengths.
        :param return_transform: Whether to return the transformation matrix.
        :return: The optimally aligned coordinates.
        """

        batch_size, num_points, dim = pred_coords.shape

        if not exists(weights):
            # if no weights are provided, assume uniform weights
            weights = torch.ones_like(pred_coords[..., 0])

        if exists(mask):
            # zero out all predicted and true coordinates where not an atom
            pred_coords = einx.where("b n, b n c, -> b n c", mask, pred_coords, 0.0)
            true_coords = einx.where("b n, b n c, -> b n c", mask, true_coords, 0.0)
            weights = einx.where("b n, b n, -> b n", mask, weights, 0.0)

        # Take care of weights broadcasting for coordinate dimension
        weights = rearrange(weights, "b n -> b n 1")

        # Compute weighted centroids
        true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
            dim=1, keepdim=True
        )
        pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
            dim=1, keepdim=True
        )

        # Center the coordinates
        true_coords_centered = true_coords - true_centroid
        pred_coords_centered = pred_coords - pred_centroid

        if num_points < (dim + 1):
            logger.warning(
                "Warning: The size of one of the point clouds is <= dim+1. "
                + "`WeightedRigidAlign` cannot return a unique rotation."
            )

        # Compute the weighted covariance matrix
        cov_matrix = einsum(
            weights * true_coords_centered, pred_coords_centered, "b n i, b n j -> b i j"
        )

        # Compute the SVD of the covariance matrix
        U, S, V = torch.svd(cov_matrix)
        U_T = U.transpose(-2, -1)

        # Catch ambiguous rotation by checking the magnitude of singular values
        if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
            logger.warning(
                "Warning: Excessively low rank of "
                + "cross-correlation between aligned point clouds. "
                + "`WeightedRigidAlign` cannot return a unique rotation."
            )

        det = torch.det(einsum(V, U_T, "b i j, b j k -> b i k"))

        # Ensure proper rotation matrix with determinant 1
        diag = torch.eye(dim, dtype=det.dtype, device=det.device)
        diag = repeat(diag, "i j -> b i j", b=batch_size).clone()

        diag[:, -1, -1] = det
        rot_matrix = einsum(V, diag, U_T, "b i j, b j k, b k l -> b i l")

        # Apply the rotation and translation
        true_aligned_coords = (
            einsum(rot_matrix, true_coords_centered, "b i j, b n j -> b n i") + pred_centroid
        )
        true_aligned_coords.detach_()

        if return_transforms:
            translation = true_centroid - einsum(
                rot_matrix, pred_centroid, "b i j, b ... j -> b ... i"
            )
            return true_aligned_coords, rot_matrix, translation

        return true_aligned_coords
