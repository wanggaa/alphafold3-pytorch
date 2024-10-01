import torch
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