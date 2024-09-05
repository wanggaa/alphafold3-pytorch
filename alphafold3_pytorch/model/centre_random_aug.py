
class CentreRandomAugmentation(Module):
    """ Algorithm 19 """

    @typecheck
    def __init__(self, trans_scale: float = 1.0):
        super().__init__()
        self.trans_scale = trans_scale
        self.register_buffer('dummy', torch.tensor(0), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    @typecheck
    def forward(
        self,
        coords: Float['b n 3'],
        mask: Bool['b n'] | None = None
    ) -> Float['b n 3']:
        """
        coords: coordinates to be augmented
        """
        batch_size = coords.shape[0]

        # Center the coordinates
        # Accounting for masking

        if exists(mask):
            coords = einx.where('b n, b n c, -> b n c', mask, coords, 0.)
            num = reduce(coords, 'b n c -> b c', 'sum')
            den = reduce(mask.float(), 'b n -> b', 'sum')
            coords_mean = einx.divide('b c, b -> b 1 c', num, den.clamp(min = 1.))
        else:
            coords_mean = coords.mean(dim = 1, keepdim = True)

        centered_coords = coords - coords_mean

        # Generate random rotation matrix
        rotation_matrix = self._random_rotation_matrix(batch_size)

        # Generate random translation vector
        translation_vector = self._random_translation_vector(batch_size)
        translation_vector = rearrange(translation_vector, 'b c -> b 1 c')

        # Apply rotation and translation
        augmented_coords = einsum(centered_coords, rotation_matrix, 'b n i, b j i -> b n j') + translation_vector

        return augmented_coords

    @typecheck
    def _random_rotation_matrix(self, batch_size: int) -> Float['b 3 3']:
        # Generate random rotation angles
        angles = torch.rand((batch_size, 3), device = self.device) * 2 * torch.pi

        # Compute sine and cosine of angles
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)

        # Construct rotation matrix
        eye = torch.eye(3, device = self.device)
        rotation_matrix = repeat(eye, 'i j -> b i j', b = batch_size).clone()

        rotation_matrix[:, 0, 0] = cos_angles[:, 0] * cos_angles[:, 1]
        rotation_matrix[:, 0, 1] = cos_angles[:, 0] * sin_angles[:, 1] * sin_angles[:, 2] - sin_angles[:, 0] * cos_angles[:, 2]
        rotation_matrix[:, 0, 2] = cos_angles[:, 0] * sin_angles[:, 1] * cos_angles[:, 2] + sin_angles[:, 0] * sin_angles[:, 2]
        rotation_matrix[:, 1, 0] = sin_angles[:, 0] * cos_angles[:, 1]
        rotation_matrix[:, 1, 1] = sin_angles[:, 0] * sin_angles[:, 1] * sin_angles[:, 2] + cos_angles[:, 0] * cos_angles[:, 2]
        rotation_matrix[:, 1, 2] = sin_angles[:, 0] * sin_angles[:, 1] * cos_angles[:, 2] - cos_angles[:, 0] * sin_angles[:, 2]
        rotation_matrix[:, 2, 0] = -sin_angles[:, 1]
        rotation_matrix[:, 2, 1] = cos_angles[:, 1] * sin_angles[:, 2]
        rotation_matrix[:, 2, 2] = cos_angles[:, 1] * cos_angles[:, 2]

        return rotation_matrix

    @typecheck
    def _random_translation_vector(self, batch_size: int) -> Float['b 3']:
        # Generate random translation vector
        translation_vector = torch.randn((batch_size, 3), device = self.device) * self.trans_scale
        return translation_vector
