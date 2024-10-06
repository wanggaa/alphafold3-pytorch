import torch
import torch.nn.functional as F
from torch import Tensor

import einx
import einops

exists = lambda x: x is not None
default = lambda x,d: x if x is not None else d
identity = lambda x: x

def pack_one(t, pattern):
    packed, ps = einops.pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern = None):
        unpacked, = einops.unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one

def lens_to_mask(
    lens: Tensor,
    max_len: int | None = None
) -> Tensor:

    device = lens.device
    if not exists(max_len):
        max_len = lens.amax()
    arange = torch.arange(max_len, device = device)
    return einx.less('m, ... -> ... m', arange, lens)

def to_pairwise_mask( 
    mask_i: Tensor,
    mask_j: Tensor | None = None
) -> Tensor:

    mask_j = default(mask_j, mask_i)
    assert mask_i.shape == mask_j.shape
    return einx.logical_and('... i, ... j -> ... i j', mask_i, mask_j)

def symmetrize(t: Tensor) -> Tensor:
    return t + einops.rearrange(t, 'b i j ... -> b j i ...')

def distance_to_dgram(
    distance: Tensor,  # type: ignore
    bins: Tensor,  # type: ignore
    return_labels: bool = False,
) -> Tensor:  # type: ignore
    """Converting from distance to discrete bins, e.g., for distance_labels and pae_labels using
    the same logic as OpenFold.

    :param distance: The distance tensor.
    :param bins: The bins tensor.
    :param return_labels: Whether to return the labels.
    :return: The one-hot bins tensor or the bin labels.
    """

    distance = distance.abs()

    bins = F.pad(bins, (0, 1), value = float('inf'))
    low, high = bins[:-1], bins[1:]

    one_hot = (
        einx.greater_equal("..., bin_low -> ... bin_low", distance, low)
        & einx.less("..., bin_high -> ... bin_high", distance, high)
    ).long()

    if return_labels:
        return one_hot.argmax(dim=-1)

    return one_hot