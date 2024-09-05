
class ComputeAlignmentError(Module):
    """ Algorithm 30 """

    @typecheck
    def __init__(
        self,
        eps: float = 1e-8
    ):
        super().__init__()
        self.eps = eps
        self.express_coordinates_in_frame = ExpressCoordinatesInFrame()

    @typecheck
    def forward(
        self,
        pred_coords: Float['b n 3'],
        true_coords: Float['b n 3'],
        pred_frames: Float['b n 3 3'],
        true_frames: Float['b n 3 3'],
        mask: Bool['b n'] | None = None,
    ) -> Float['b n n']:
        """
        pred_coords: predicted coordinates
        true_coords: true coordinates
        pred_frames: predicted frames
        true_frames: true frames
        """
        # to pairs

        seq = pred_coords.shape[1]
        
        pair2seq = partial(rearrange, pattern='b n m ... -> b (n m) ...')
        seq2pair = partial(rearrange, pattern='b (n m) ... -> b n m ...', n = seq, m = seq)
        
        pair_pred_coords = pair2seq(repeat(pred_coords, 'b n d -> b n m d', m = seq))
        pair_true_coords = pair2seq(repeat(true_coords, 'b n d -> b n m d', m = seq))
        pair_pred_frames = pair2seq(repeat(pred_frames, 'b n d e -> b m n d e', m = seq))
        pair_true_frames = pair2seq(repeat(true_frames, 'b n d e -> b m n d e', m = seq))
        
        # Express predicted coordinates in predicted frames
        pred_coords_transformed = self.express_coordinates_in_frame(pair_pred_coords, pair_pred_frames)

        # Express true coordinates in true frames
        true_coords_transformed = self.express_coordinates_in_frame(pair_true_coords, pair_true_frames)

        # Compute alignment errors
        alignment_errors = F.pairwise_distance(pred_coords_transformed, true_coords_transformed, eps = self.eps)

        alignment_errors = seq2pair(alignment_errors)

        # Masking
        if exists(mask):
            pair_mask = to_pairwise_mask(mask)
            alignment_errors = einx.where('b i j, b i j, -> b i j', pair_mask, alignment_errors, 0.)

        return alignment_errors