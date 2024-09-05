from __future__ import annotations

import random
import sh
from math import pi, sqrt
from pathlib import Path
from itertools import product
from functools import partial, wraps
from collections import namedtuple

import torch
from torch import nn
from torch import Tensor
from torch.amp import autocast
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from torch.nn import (
    Module,
    ModuleList,
    Linear,
    Sequential,
)

from typing import Callable, Dict, List, Literal, NamedTuple, Tuple

from alphafold3_pytorch.tensor_typing import (
    Float,
    Int,
    Bool,
    Shaped,
    typecheck,
    IS_DEBUGGING
)

from alphafold3_pytorch.attention import (
    Attention,
    pad_at_dim,
    slice_at_dim,
    pad_or_slice_to,
    pad_to_multiple,
    concat_previous_window,
    full_attn_bias_to_windowed,
    full_pairwise_repr_to_windowed
)

from alphafold3_pytorch.inputs import (
    IS_MOLECULE_TYPES,
    IS_PROTEIN_INDEX,
    IS_DNA_INDEX,
    IS_RNA_INDEX,
    IS_LIGAND_INDEX,
    IS_METAL_ION_INDEX,
    IS_BIOMOLECULE_INDICES,
    IS_PROTEIN,
    IS_DNA,
    IS_RNA,
    IS_LIGAND,
    IS_METAL_ION,
    NUM_MOLECULE_IDS,
    NUM_MSA_ONE_HOT,
    DEFAULT_NUM_MOLECULE_MODS,
    ADDITIONAL_MOLECULE_FEATS,
    BatchedAtomInput,
    hard_validate_atom_indices_ascending
)

from alphafold3_pytorch.common.biomolecule import (
    get_residue_constants,
)

from alphafold3_pytorch.utils.model_utils import (
    ExpressCoordinatesInFrame,
    RigidFrom3Points,
    calculate_weighted_rigid_align_weights,
)

from frame_averaging_pytorch import FrameAverage

from taylor_series_linear_attention import TaylorSeriesLinearAttn

from colt5_attention import ConditionalRoutedAttention

import einx
from einops import rearrange, repeat, reduce, einsum, pack, unpack
from einops.layers.torch import Rearrange

from tqdm import tqdm

from loguru import logger

from importlib.metadata import version

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.DSSP import DSSP
import tempfile

"""
global ein notation:

a - number of tokens in a given chain (asym_id)
b - batch
ba - batch with augmentation
bt - batch with templates dimension merged
h - heads
n - molecule sequence length
i - molecule sequence length (source)
j - molecule sequence length (target)
l - number of distogram bins
m - atom sequence length
nw - windowed sequence length
d - feature dimension
ds - feature dimension (single)
dp - feature dimension (pairwise)
dap - feature dimension (atompair)
dapi - feature dimension (atompair input)
da - feature dimension (atom)
dai - feature dimension (atom input)
dmi - feature dimension (msa input)
dmf - additional msa feats derived from msa (has_deletion and deletion_value)
dtf - additional token feats derived from msa (profile and deletion_mean)
t - templates
s - msa
r - registers
ts - diffusion timesteps
"""

"""
additional_msa_feats: [*, 2]:
- concatted to the msa single rep

0: has_deletion
1: deletion_value
"""

"""
additional_token_feats: [*, 33]:
- concatted to the single rep

0: profile
1: deletion_mean
"""

"""
additional_molecule_feats: [*, 5]:
- used for deriving relative positions

0: molecule_index
1: token_index
2: asym_id
3: entity_id
4: sym_id
"""

"""
is_molecule_types: [*, 5]

0: is_protein
1: is_rna
2: is_dna
3: is_ligand
4: is_metal_ions_or_misc
"""

# constants

LinearNoBias = partial(Linear, bias = False)

# always use non reentrant checkpointing

checkpoint = partial(checkpoint, use_reentrant = False)
checkpoint_sequential = partial(checkpoint_sequential, use_reentrant = False)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(x, *args, **kwargs):
    return x

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def divisible_by(num, den):
    return (num % den) == 0

def compact(*args):
    return tuple(filter(exists, args))

# tensor helpers

def l2norm(t, eps = 1e-20, dim = -1):
    return F.normalize(t, p = 2, eps = eps, dim = dim)

def max_neg_value(t: Tensor):
    return -torch.finfo(t.dtype).max

def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern = None):
        unpacked, = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one

def exclusive_cumsum(t, dim = -1):
    return t.cumsum(dim = dim) - t

@typecheck
def symmetrize(t: Float['b n n ...']) -> Float['b n n ...']:
    return t + rearrange(t, 'b i j ... -> b j i ...')

@typecheck
def masked_average(
    t: Shaped['...'],
    mask: Shaped['...'],
    *,
    dim: int | Tuple[int, ...],
    eps = 1.
) -> Float['...']:

    num = (t * mask).sum(dim = dim)
    den = mask.sum(dim = dim)
    return num / den.clamp(min = eps)

# checkpointing utils

@typecheck
def should_checkpoint(
    self: Module,
    inputs: Tensor | Tuple[Tensor, ...],
    check_instance_variable: str | None = 'checkpoint'
) -> bool:
    if torch.is_tensor(inputs):
        inputs = (inputs,)

    return (
        self.training and
        any([i.requires_grad for i in inputs]) and
        (not exists(check_instance_variable) or getattr(self, check_instance_variable, False))
    )

# decorators

def maybe(fn):
    @wraps(fn)
    def inner(t, *args, **kwargs):
        if not exists(t):
            return None
        return fn(t, *args, **kwargs)
    return inner

def save_args_and_kwargs(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        self._args_and_kwargs = (args, kwargs)
        self._version = version('alphafold3_pytorch')

        return fn(self, *args, **kwargs)
    return inner

@typecheck
def pad_and_window(
    t: Float['b n ...'] | Int['b n ...'],
    window_size: int
):
    t = pad_to_multiple(t, window_size, dim = 1)
    t = rearrange(t, 'b (n w) ... -> b n w ...', w = window_size)
    return t

# packed atom representation functions

@typecheck
def lens_to_mask(
    lens: Int['b ...'],
    max_len: int | None = None
) -> Bool['... m']:

    device = lens.device
    if not exists(max_len):
        max_len = lens.amax()
    arange = torch.arange(max_len, device = device)
    return einx.less('m, ... -> ... m', arange, lens)

@typecheck
def to_pairwise_mask(
    mask_i: Bool['... n'],
    mask_j: Bool['... n'] | None = None
) -> Bool['... n n']:

    mask_j = default(mask_j, mask_i)
    assert mask_i.shape == mask_j.shape
    return einx.logical_and('... i, ... j -> ... i j', mask_i, mask_j)

@typecheck
def mean_pool_with_lens(
    feats: Float['b m d'],
    lens: Int['b n']
) -> Float['b n d']:

    seq_len = feats.shape[1]

    mask = lens > 0
    assert (lens.sum(dim = -1) <= seq_len).all(), 'one of the lengths given exceeds the total sequence length of the features passed in'

    cumsum_feats = feats.cumsum(dim = 1)
    cumsum_feats = F.pad(cumsum_feats, (0, 0, 1, 0), value = 0.)

    cumsum_indices = lens.cumsum(dim = 1)
    cumsum_indices = F.pad(cumsum_indices, (1, 0), value = 0)

    # sel_cumsum = einx.get_at('b [m] d, b n -> b n d', cumsum_feats, cumsum_indices)

    cumsum_indices = repeat(cumsum_indices, 'b n -> b n d', d = cumsum_feats.shape[-1])
    sel_cumsum = cumsum_feats.gather(-2, cumsum_indices)

    # subtract cumsum at one index from the previous one
    summed = sel_cumsum[:, 1:] - sel_cumsum[:, :-1]

    avg = einx.divide('b n d, b n', summed, lens.clamp(min = 1))
    avg = einx.where('b n, b n d, -> b n d', mask, avg, 0.)
    return avg

@typecheck
def mean_pool_fixed_windows_with_mask(
    feats: Float['b m d'],
    mask: Bool['b m'],
    window_size: int,
    return_mask_and_inverse: bool = False,
) -> Float['b n d'] | Tuple[Float['b n d'], Bool['b n'], Callable[[Float['b m d']], Float['b n d']]]:

    seq_len = feats.shape[-2]
    assert divisible_by(seq_len, window_size)

    feats = einx.where('b m, b m d, -> b m d', mask, feats, 0.)

    num = reduce(feats, 'b (n w) d -> b n d', 'sum', w = window_size)
    den = reduce(mask.float(), 'b (n w) -> b n 1', 'sum', w = window_size)

    avg = num / den.clamp(min = 1.)

    if not return_mask_and_inverse:
        return avg

    pooled_mask = reduce(mask, 'b (n w) -> b n', 'any', w = window_size)

    @typecheck
    def inverse_fn(pooled: Float['b n d']) -> Float['b m d']:
        unpooled = repeat(pooled, 'b n d -> b (n w) d', w = window_size)
        unpooled = einx.where('b m, b m d, -> b m d', mask, unpooled, 0.)
        return unpooled

    return avg, pooled_mask, inverse_fn

@typecheck
def batch_repeat_interleave(
    feats: Float['b n ...'] | Bool['b n ...'] | Bool['b n'] | Int['b n'],
    lens: Int['b n'],
    output_padding_value: float | int | bool | None = None, # this value determines what the output padding value will be
) -> Float['b m ...'] | Bool['b m ...'] | Bool['b m'] | Int['b m']:

    device, dtype = feats.device, feats.dtype

    batch, seq, *dims = feats.shape

    # get mask from lens

    mask = lens_to_mask(lens)

    # derive arange

    window_size = mask.shape[-1]
    arange = torch.arange(window_size, device = device)

    offsets = exclusive_cumsum(lens)
    indices = einx.add('w, b n -> b n w', arange, offsets)

    # create output tensor + a sink position on the very right (index max_len)

    total_lens = lens.clamp(min = 0).sum(dim = -1)
    output_mask = lens_to_mask(total_lens)

    max_len = total_lens.amax()

    output_indices = torch.zeros((batch, max_len + 1), device = device, dtype = torch.long)

    indices = indices.masked_fill(~mask, max_len) # scatter to sink position for padding
    indices = rearrange(indices, 'b n w -> b (n w)')

    # scatter

    seq_arange = torch.arange(seq, device = device)
    seq_arange = repeat(seq_arange, 'n -> b (n w)', b = batch, w = window_size)

    # output_indices = einx.set_at('b [m], b nw, b nw -> b [m]', output_indices, indices, seq_arange)

    output_indices = output_indices.scatter(1, indices, seq_arange)

    # remove sink

    output_indices = output_indices[:, :-1]

    # gather

    # output = einx.get_at('b [n] ..., b m -> b m ...', feats, output_indices)

    feats, unpack_one = pack_one(feats, 'b n *')
    output_indices = repeat(output_indices, 'b m -> b m d', d = feats.shape[-1])
    output = feats.gather(1, output_indices)
    output = unpack_one(output)

    # set output padding value

    output_padding_value = default(output_padding_value, False if dtype == torch.bool else 0)

    output = einx.where(
        'b n, b n ..., -> b n ...',
        output_mask, output, output_padding_value
    )

    return output

@typecheck
def batch_repeat_interleave_pairwise(
    pairwise: Float['b n n d'],
    molecule_atom_lens: Int['b n']
) -> Float['b m m d']:

    pairwise = batch_repeat_interleave(pairwise, molecule_atom_lens)

    molecule_atom_lens = repeat(molecule_atom_lens, 'b ... -> (b r) ...', r = pairwise.shape[1])
    pairwise, unpack_one = pack_one(pairwise, '* n d')
    pairwise = batch_repeat_interleave(pairwise, molecule_atom_lens)
    return unpack_one(pairwise)

@typecheck
def distance_to_bins(
    distance: Float['... dist'],
    bins: Float[' bins']
) -> Int['... dist']:
    """
    converting from distance to discrete bins, for distance_labels and pae_labels
    """

    dist_from_dist_bins = einx.subtract('... dist, dist_bins -> ... dist dist_bins', distance, bins).abs()
    return dist_from_dist_bins.argmin(dim = -1)

# linear and outer sum
# for single repr -> pairwise pattern throughout this architecture

class LinearNoBiasThenOuterSum(Module):
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.proj = LinearNoBias(dim, dim_out * 2)

    @typecheck
    def forward(
        self,
        t: Float['b n ds']
    ) -> Float['b n n dp']:

        single_i, single_j = self.proj(t).chunk(2, dim = -1)
        out = einx.add('b i d, b j d -> b i j d', single_i, single_j)
        return out

# classic feedforward, SwiGLU variant
# they name this 'transition' in their paper
# Algorithm 11

class SwiGLU(Module):
    @typecheck
    def forward(
        self,
        x: Float['... d']
    ) -> Float[' ... (d//2)']:

        x, gates = x.chunk(2, dim = -1)
        return F.silu(gates) * x

class Transition(Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor = 4
    ):
        super().__init__()
        dim_inner = int(dim * expansion_factor)

        self.ff = Sequential(
            LinearNoBias(dim, dim_inner * 2),
            SwiGLU(),
            LinearNoBias(dim_inner, dim)
        )

    @typecheck
    def forward(
        self,
        x: Float['... d']
    ) -> Float['... d']:

        return self.ff(x)

# dropout
# they seem to be using structured dropout - row / col wise in triangle modules

class Dropout(Module):
    @typecheck
    def __init__(
        self,
        prob: float,
        *,
        dropout_type: Literal['row', 'col'] | None = None
    ):
        super().__init__()
        self.dropout = nn.Dropout(prob)
        self.dropout_type = dropout_type

    @typecheck
    def forward(
        self,
        t: Tensor
    ) -> Tensor:

        if self.dropout_type in {'row', 'col'}:
            assert t.ndim == 4, 'tensor must be 4 dimensions for row / col structured dropout'

        if not exists(self.dropout_type):
            return self.dropout(t)

        if self.dropout_type == 'row':
            batch, _, col, dim = t.shape
            ones_shape = (batch, 1, col, dim)

        elif self.dropout_type == 'col':
            batch, row, _, dim = t.shape
            ones_shape = (batch, row, 1, dim)

        ones = t.new_ones(ones_shape)
        dropped = self.dropout(ones)
        return t * dropped

# normalization
# both pre layernorm as well as adaptive layernorm wrappers

class PreLayerNorm(Module):
    @typecheck
    def __init__(
        self,
        fn: Attention | Transition | TriangleAttention | TriangleMultiplication | AttentionPairBias,
        *,
        dim,
    ):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    @typecheck
    def forward(
        self,
        x: Float['... n d'],
        **kwargs
    ) -> Float['... n d']:

        x = self.norm(x)
        return self.fn(x, **kwargs)

class AdaptiveLayerNorm(Module):
    """ Algorithm 26 """

    def __init__(
        self,
        *,
        dim,
        dim_cond
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine = False)
        self.norm_cond = nn.LayerNorm(dim_cond, bias = False)

        self.to_gamma = nn.Sequential(
            Linear(dim_cond, dim),
            nn.Sigmoid()
        )

        self.to_beta = LinearNoBias(dim_cond, dim)

    @typecheck
    def forward(
        self,
        x: Float['b n d'],
        cond: Float['b n dc']
    ) -> Float['b n d']:

        normed = self.norm(x)
        normed_cond = self.norm_cond(cond)

        gamma = self.to_gamma(normed_cond)
        beta = self.to_beta(normed_cond)
        return normed * gamma + beta

class ConditionWrapper(Module):
    """ Algorithm 25 """

    @typecheck
    def __init__(
        self,
        fn: Attention | Transition | TriangleAttention |  AttentionPairBias,
        *,
        dim,
        dim_cond,
        adaln_zero_bias_init_value = -2.
    ):
        super().__init__()
        self.fn = fn
        self.adaptive_norm = AdaptiveLayerNorm(dim = dim, dim_cond = dim_cond)

        adaln_zero_gamma_linear = Linear(dim_cond, dim)
        nn.init.zeros_(adaln_zero_gamma_linear.weight)
        nn.init.constant_(adaln_zero_gamma_linear.bias, adaln_zero_bias_init_value)

        self.to_adaln_zero_gamma = nn.Sequential(
            adaln_zero_gamma_linear,
            nn.Sigmoid()
        )

    @typecheck
    def forward(
        self,
        x: Float['b n d'],
        *,
        cond: Float['b n dc'],
        **kwargs
    ) -> Float['b n d']:
        x = self.adaptive_norm(x, cond = cond)

        out = self.fn(x, **kwargs)

        gamma = self.to_adaln_zero_gamma(cond)
        return out * gamma

# triangle multiplicative module
# seems to be unchanged from alphafold2

class TriangleMultiplication(Module):

    @typecheck
    def __init__(
        self,
        *,
        dim,
        dim_hidden = None,
        mix: Literal["incoming", "outgoing"] = 'incoming',
        dropout = 0.,
        dropout_type: Literal['row', 'col'] | None = None
    ):
        super().__init__()

        dim_hidden = default(dim_hidden, dim)

        self.left_right_proj = nn.Sequential(
            LinearNoBias(dim, dim_hidden * 4),
            nn.GLU(dim = -1)
        )

        self.out_gate = LinearNoBias(dim, dim_hidden)

        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'incoming':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(dim_hidden)

        self.to_out = Sequential(
            LinearNoBias(dim_hidden, dim),
            Dropout(dropout, dropout_type = dropout_type)
        )

    @typecheck
    def forward(
        self,
        x: Float['b n n d'],
        mask: Bool['b n'] | None = None
    ) -> Float['b n n d']:

        if exists(mask):
            mask = to_pairwise_mask(mask)
            mask = rearrange(mask, '... -> ... 1')

        left, right = self.left_right_proj(x).chunk(2, dim = -1)

        if exists(mask):
            left = left * mask
            right = right * mask

        out = einsum(left, right, self.mix_einsum_eq)

        out = self.to_out_norm(out)

        out_gate = self.out_gate(x).sigmoid()

        return self.to_out(out) * out_gate

# there are two types of attention in this paper, triangle and attention-pair-bias
# they differ by how the attention bias is computed
# triangle is axial attention w/ itself projected for bias

class AttentionPairBias(Module):
    def __init__(
        self,
        *,
        heads,
        dim_pairwise,
        window_size = None,
        num_memory_kv = 0,
        **attn_kwargs
    ):
        super().__init__()

        self.window_size = window_size

        self.attn = Attention(
            heads = heads,
            window_size = window_size,
            num_memory_kv = num_memory_kv,
            **attn_kwargs
        )

        # line 8 of Algorithm 24

        to_attn_bias_linear = LinearNoBias(dim_pairwise, heads)
        nn.init.zeros_(to_attn_bias_linear.weight)

        self.to_attn_bias = nn.Sequential(
            nn.LayerNorm(dim_pairwise),
            to_attn_bias_linear,
            Rearrange('b ... h -> b h ...')
        )

    @typecheck
    def forward(
        self,
        single_repr: Float['b n ds'],
        *,
        pairwise_repr: Float['b n n dp'] | Float['b nw w (w*2) dp'],
        attn_bias: Float['b n n'] | Float['b nw w (w*2)'] | None = None,
        **kwargs
    ) -> Float['b n ds']:

        w, has_window_size = self.window_size, exists(self.window_size)

        # take care of windowing logic
        # for sequence-local atom transformer

        windowed_pairwise = pairwise_repr.ndim == 5

        windowed_attn_bias = None

        if exists(attn_bias):
            windowed_attn_bias = attn_bias.shape[-1] != attn_bias.shape[-2]

        if has_window_size:
            if not windowed_pairwise:
                pairwise_repr = full_pairwise_repr_to_windowed(pairwise_repr, window_size = w)
            if exists(attn_bias):
                attn_bias = full_attn_bias_to_windowed(attn_bias, window_size = w)
        else:
            assert not windowed_pairwise, 'cannot pass in windowed pairwise repr if no window_size given to AttentionPairBias'
            assert not exists(windowed_attn_bias) or not windowed_attn_bias, 'cannot pass in windowed attention bias if no window_size set for AttentionPairBias'

        # attention bias preparation with further addition from pairwise repr

        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, 'b ... -> b 1 ...')
        else:
            attn_bias = 0.

        attn_bias = self.to_attn_bias(pairwise_repr) + attn_bias

        out = self.attn(
            single_repr,
            attn_bias = attn_bias,
            **kwargs
        )

        return out

class TriangleAttention(Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        node_type: Literal['starting', 'ending'],
        dropout = 0.,
        dropout_type: Literal['row', 'col'] | None = None,
        **attn_kwargs
    ):
        super().__init__()
        self.need_transpose = node_type == 'ending'

        self.attn = Attention(dim = dim, heads = heads, **attn_kwargs)

        self.dropout = Dropout(dropout, dropout_type = dropout_type)

        self.to_attn_bias = nn.Sequential(
            LinearNoBias(dim, heads),
            Rearrange('... i j h -> ... h i j')
        )

    @typecheck
    def forward(
        self,
        pairwise_repr: Float['b n n d'],
        mask: Bool['b n'] | None = None,
        **kwargs
    ) -> Float['b n n d']:

        if self.need_transpose:
            pairwise_repr = rearrange(pairwise_repr, 'b i j d -> b j i d')

        attn_bias = self.to_attn_bias(pairwise_repr)

        batch_repeat = pairwise_repr.shape[1]
        attn_bias = repeat(attn_bias, 'b ... -> (b repeat) ...', repeat = batch_repeat)

        if exists(mask):
            mask = repeat(mask, 'b ... -> (b repeat) ...', repeat = batch_repeat)

        pairwise_repr, unpack_one = pack_one(pairwise_repr, '* n d')

        out = self.attn(
            pairwise_repr,
            mask = mask,
            attn_bias = attn_bias,
            **kwargs
        )

        out = unpack_one(out)

        if self.need_transpose:
            out = rearrange(out, 'b j i d -> b i j d')

        return self.dropout(out)

# PairwiseBlock
# used in both MSAModule and Pairformer
# consists of all the "Triangle" modules + Transition

class PairwiseBlock(Module):
    def __init__(
        self,
        *,
        dim_pairwise = 128,
        tri_mult_dim_hidden = None,
        tri_attn_dim_head = 32,
        tri_attn_heads = 4,
        dropout_row_prob = 0.25,
        dropout_col_prob = 0.25,
    ):
        super().__init__()

        pre_ln = partial(PreLayerNorm, dim = dim_pairwise)

        tri_mult_kwargs = dict(
            dim = dim_pairwise,
            dim_hidden = tri_mult_dim_hidden
        )

        tri_attn_kwargs = dict(
            dim = dim_pairwise,
            heads = tri_attn_heads,
            dim_head = tri_attn_dim_head
        )

        self.tri_mult_outgoing = pre_ln(TriangleMultiplication(mix = 'outgoing', dropout = dropout_row_prob, dropout_type = 'row', **tri_mult_kwargs))
        self.tri_mult_incoming = pre_ln(TriangleMultiplication(mix = 'incoming', dropout = dropout_row_prob, dropout_type = 'row', **tri_mult_kwargs))
        self.tri_attn_starting = pre_ln(TriangleAttention(node_type = 'starting', dropout = dropout_row_prob, dropout_type = 'row', **tri_attn_kwargs))
        self.tri_attn_ending = pre_ln(TriangleAttention(node_type = 'ending', dropout = dropout_col_prob, dropout_type = 'col', **tri_attn_kwargs))
        self.pairwise_transition = pre_ln(Transition(dim = dim_pairwise))

    @typecheck
    def forward(
        self,
        *,
        pairwise_repr: Float['b n n d'],
        mask: Bool['b n'] | None = None
    ):
        pairwise_repr = self.tri_mult_outgoing(pairwise_repr, mask = mask) + pairwise_repr
        pairwise_repr = self.tri_mult_incoming(pairwise_repr, mask = mask) + pairwise_repr
        pairwise_repr = self.tri_attn_starting(pairwise_repr, mask = mask) + pairwise_repr
        pairwise_repr = self.tri_attn_ending(pairwise_repr, mask = mask) + pairwise_repr

        pairwise_repr = self.pairwise_transition(pairwise_repr) + pairwise_repr
        return pairwise_repr

# msa module

class OuterProductMean(Module):
    """ Algorithm 9 """

    def __init__(
        self,
        *,
        dim_msa = 64,
        dim_pairwise = 128,
        dim_hidden = 32,
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(dim_msa)
        self.to_hidden = LinearNoBias(dim_msa, dim_hidden * 2)
        self.to_pairwise_repr = nn.Linear(dim_hidden ** 2, dim_pairwise)

    @typecheck
    def forward(
        self,
        msa: Float['b s n d'],
        *,
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None
    ) -> Float['b n n dp']:
        
        dtype = msa.dtype

        msa = self.norm(msa)

        # line 2

        a, b = self.to_hidden(msa).chunk(2, dim = -1)

        # maybe masked mean for outer product

        if exists(msa_mask):
            a = einx.multiply('b s i d, b s -> b s i d', a, msa_mask.type(dtype))
            b = einx.multiply('b s j e, b s -> b s j e', b, msa_mask.type(dtype))

            outer_product = einsum(a, b, 'b s i d, b s j e -> b i j d e')

            num_msa = reduce(msa_mask.type(dtype), '... s -> ...', 'sum')

            outer_product_mean = einx.divide('b i j d e, b', outer_product, num_msa.clamp(min = self.eps))
        else:
            num_msa = msa.shape[1]
            outer_product = einsum(a, b, 'b s i d, b s j e -> b i j d e')
            outer_product_mean = outer_product / num_msa

        # flatten

        outer_product_mean = rearrange(outer_product_mean, '... d e -> ... (d e)')

        # masking for pairwise repr

        if exists(mask):
            mask = to_pairwise_mask(mask)
            outer_product_mean = einx.multiply(
                'b i j d, b i j', outer_product_mean, mask.type(dtype)
            )

        pairwise_repr = self.to_pairwise_repr(outer_product_mean)
        return pairwise_repr


class MSAPairWeightedAveraging(Module):
    """ Algorithm 10 """

    def __init__(
        self,
        *,
        dim_msa = 64,
        dim_pairwise = 128,
        dim_head = 32,
        heads = 8,
        dropout = 0.,
        dropout_type: Literal['row', 'col'] | None = None
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.msa_to_values_and_gates = nn.Sequential(
            nn.LayerNorm(dim_msa),
            LinearNoBias(dim_msa, dim_inner * 2),
            Rearrange('b s n (gv h d) -> gv b h s n d', gv = 2, h = heads)
        )

        self.pairwise_repr_to_attn = nn.Sequential(
            nn.LayerNorm(dim_pairwise),
            LinearNoBias(dim_pairwise, heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            Rearrange('b h s n d -> b s n (h d)'),
            LinearNoBias(dim_inner, dim_msa),
            Dropout(dropout, dropout_type = dropout_type)
        )

    @typecheck
    def forward(
        self,
        *,
        msa: Float['b s n d'],
        pairwise_repr: Float['b n n dp'],
        mask: Bool['b n'] | None = None,
    ) -> Float['b s n d']:

        values, gates = self.msa_to_values_and_gates(msa)
        gates = gates.sigmoid()

        # line 3

        b = self.pairwise_repr_to_attn(pairwise_repr)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            b = b.masked_fill(~mask, max_neg_value(b))

        # line 5

        weights = b.softmax(dim = -1)

        # line 6

        out = einsum(weights, values, 'b h i j, b h s j d -> b h s i d')

        out = out * gates

        # combine heads

        return self.to_out(out)

class MSAModule(Module):
    """ Algorithm 8 """

    def __init__(
        self,
        *,
        dim_single = 384,
        dim_pairwise = 128,
        depth = 4,
        dim_msa = 64,
        dim_msa_input=NUM_MSA_ONE_HOT,
        dim_additional_msa_feats=2,
        outer_product_mean_dim_hidden = 32,
        msa_pwa_dropout_row_prob = 0.15,
        msa_pwa_heads = 8,
        msa_pwa_dim_head = 32,
        checkpoint = False,
        checkpoint_segments = 1,
        pairwise_block_kwargs: dict = dict(),
        max_num_msa: int | None = None,
        layerscale_output: bool = True
    ):
        super().__init__()

        self.max_num_msa = default(
            max_num_msa, float('inf')
        )  # cap the number of MSAs, will do sample without replacement if exceeds

        self.msa_init_proj = LinearNoBias(dim_msa_input + dim_additional_msa_feats, dim_msa)

        self.single_to_msa_feats = LinearNoBias(dim_single, dim_msa)

        layers = ModuleList([])

        for _ in range(depth):

            msa_pre_ln = partial(PreLayerNorm, dim = dim_msa)

            outer_product_mean = OuterProductMean(
                dim_msa = dim_msa,
                dim_pairwise = dim_pairwise,
                dim_hidden = outer_product_mean_dim_hidden
            )

            msa_pair_weighted_avg = MSAPairWeightedAveraging(
                dim_msa = dim_msa,
                dim_pairwise = dim_pairwise,
                heads = msa_pwa_heads,
                dim_head = msa_pwa_dim_head,
                dropout = msa_pwa_dropout_row_prob,
                dropout_type = 'row'
            )

            msa_transition = Transition(dim = dim_msa)

            pairwise_block = PairwiseBlock(
                dim_pairwise = dim_pairwise,
                **pairwise_block_kwargs
            )

            layers.append(ModuleList([
                outer_product_mean,
                msa_pair_weighted_avg,
                msa_pre_ln(msa_transition),
                pairwise_block
            ]))

        self.checkpoint = checkpoint
        self.checkpoint_segments = checkpoint_segments

        self.layers = layers

        self.layerscale_output = nn.Parameter(torch.zeros(dim_pairwise)) if layerscale_output else 1.

        # msa related

        self.dmi = dim_additional_msa_feats

    @typecheck
    def to_layers(
        self,
        *,
        pairwise_repr: Float['b n n dp'],
        msa: Float['b s n dm'],
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
    ) -> Float['b n n dp']:

        for (
            outer_product_mean,
            msa_pair_weighted_avg,
            msa_transition,
            pairwise_block
        ) in self.layers:

            # communication between msa and pairwise rep

            pairwise_repr = outer_product_mean(msa, mask = mask, msa_mask = msa_mask) + pairwise_repr

            msa = msa_pair_weighted_avg(msa = msa, pairwise_repr = pairwise_repr, mask = mask) + msa
            msa = msa_transition(msa) + msa

            # pairwise block

            pairwise_repr = pairwise_block(pairwise_repr = pairwise_repr, mask = mask)

        return pairwise_repr

    @typecheck
    def to_checkpointed_layers(
        self,
        *,
        pairwise_repr: Float['b n n dp'],
        msa: Float['b s n dm'],
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
    ) -> Float['b n n dp']:

        inputs = (pairwise_repr, mask, msa, msa_mask)

        wrapped_layers = []

        def outer_product_mean_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                pairwise_repr, mask, msa, msa_mask = inputs
                pairwise_repr = fn(msa = msa, mask = mask, msa_mask = msa_mask) + pairwise_repr
                return pairwise_repr, mask, msa, msa_mask
            return inner

        def msa_pair_weighted_avg_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                pairwise_repr, mask, msa, msa_mask = inputs
                msa = fn(msa = msa, pairwise_repr = pairwise_repr, mask = mask) + msa
                return pairwise_repr, mask, msa, msa_mask
            return inner

        def pairwise_block_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                pairwise_repr, mask, msa, msa_mask = inputs
                pairwise_repr = fn(pairwise_repr = pairwise_repr, mask = mask)
                return pairwise_repr, mask, msa, msa_mask
            return inner

        def msa_transition_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                pairwise_repr, mask, msa, msa_mask = inputs
                msa = fn(msa) + msa
                return pairwise_repr, mask, msa, msa_mask
            return inner

        for (
            outer_product_mean,
            msa_pair_weighted_avg,
            msa_transition,
            pairwise_block
        ) in self.layers:

            wrapped_layers.append(outer_product_mean_wrapper(outer_product_mean))
            wrapped_layers.append(msa_pair_weighted_avg_wrapper(msa_pair_weighted_avg))
            wrapped_layers.append(msa_transition_wrapper(msa_transition))
            wrapped_layers.append(pairwise_block_wrapper(pairwise_block))

        pairwise_repr, *_ = checkpoint_sequential(wrapped_layers, self.checkpoint_segments, inputs)

        return pairwise_repr

    @typecheck
    def forward(
        self,
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'],
        msa: Float['b s n dm'],
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
        additional_msa_feats: Float['b s n {self.dmi}'] | None = None,
    ) -> Float['b n n dp']:

        batch, num_msa, device = *msa.shape[:2], msa.device

        # sample without replacement

        if num_msa > self.max_num_msa:
            rand = torch.randn((batch, num_msa), device = device)

            if exists(msa_mask):
                rand.masked_fill_(~msa_mask, max_neg_value(msa))

            indices = rand.topk(self.max_num_msa, dim = -1).indices

            # msa = einx.get_at('b [s] n dm, b sampled -> b sampled n dm', msa, indices)

            msa, unpack_one = pack_one(msa, 'b s *')
            msa_indices = repeat(indices, 'b sampled -> b sampled d', d = msa.shape[-1])
            msa = msa.gather(1, msa_indices)
            msa = unpack_one(msa)

            if exists(msa_mask):
                # msa_mask = einx.get_at('b [s], b sampled -> b sampled', msa_mask, indices)
                msa_mask = msa_mask.gather(1, indices)

            if exists(additional_msa_feats):
                # additional_msa_feats = einx.get_at('b s 2, b sampled -> b sampled 2', additional_msa_feats, indices)

                additional_msa_feats, unpack_one = pack_one(additional_msa_feats, 'b s *')
                additional_msa_indices = repeat(
                    indices, 'b sampled -> b sampled d', d=additional_msa_feats.shape[-1]
                )
                additional_msa_feats = additional_msa_feats.gather(1, additional_msa_indices)
                additional_msa_feats = unpack_one(additional_msa_feats)

        # account for no msa

        if exists(msa_mask):
            has_msa = reduce(msa_mask, 'b s -> b', 'any')

        # account for additional msa features

        if exists(additional_msa_feats):
            msa = torch.cat((msa, additional_msa_feats), dim=-1)

        # process msa

        msa = self.msa_init_proj(msa)

        single_msa_feats = self.single_to_msa_feats(single_repr)

        msa = rearrange(single_msa_feats, 'b n d -> b 1 n d') + msa

        # going through the layers

        if should_checkpoint(self, (pairwise_repr, msa)):
            to_layers_fn = self.to_checkpointed_layers
        else:
            to_layers_fn = self.to_layers

        pairwise_repr = to_layers_fn(
            msa = msa,
            mask = mask,
            pairwise_repr = pairwise_repr,
            msa_mask = msa_mask
        )

        # final masking and then layer scale

        if exists(msa_mask):
            pairwise_repr = einx.where(
                'b, b ..., -> b ...',
                has_msa, pairwise_repr, 0.
            )

        return pairwise_repr * self.layerscale_output

# pairformer stack

class PairformerStack(Module):
    """ Algorithm 17 """

    def __init__(
        self,
        *,
        dim_single = 384,
        dim_pairwise = 128,
        depth = 48,
        recurrent_depth = 1, # effective depth will be depth * recurrent_depth
        pair_bias_attn_dim_head = 64,
        pair_bias_attn_heads = 16,
        dropout_row_prob = 0.25,
        num_register_tokens = 0,
        checkpoint = False,
        checkpoint_segments = 1,
        pairwise_block_kwargs: dict = dict(),
        pair_bias_attn_kwargs: dict = dict()
    ):
        super().__init__()
        layers = ModuleList([])

        pair_bias_attn_kwargs = dict(
            dim = dim_single,
            dim_pairwise = dim_pairwise,
            heads = pair_bias_attn_heads,
            dim_head = pair_bias_attn_dim_head,
            dropout = dropout_row_prob,
            **pair_bias_attn_kwargs
        )

        for _ in range(depth):

            single_pre_ln = partial(PreLayerNorm, dim = dim_single)

            pairwise_block = PairwiseBlock(
                dim_pairwise = dim_pairwise,
                **pairwise_block_kwargs
            )

            pair_bias_attn = AttentionPairBias(**pair_bias_attn_kwargs)
            single_transition = Transition(dim = dim_single)

            layers.append(ModuleList([
                pairwise_block,
                single_pre_ln(pair_bias_attn),
                single_pre_ln(single_transition),
            ]))

        self.layers = layers

        # checkpointing

        self.checkpoint = checkpoint
        self.checkpoint_segments = checkpoint_segments

        # https://arxiv.org/abs/2405.16039 and https://arxiv.org/abs/2405.15071
        # although possibly recycling already takes care of this

        assert recurrent_depth > 0
        self.recurrent_depth = recurrent_depth

        self.num_registers = num_register_tokens
        self.has_registers = num_register_tokens > 0

        if self.has_registers:
            self.single_registers = nn.Parameter(torch.zeros(num_register_tokens, dim_single))
            self.pairwise_row_registers = nn.Parameter(torch.zeros(num_register_tokens, dim_pairwise))
            self.pairwise_col_registers = nn.Parameter(torch.zeros(num_register_tokens, dim_pairwise))

    @typecheck
    def to_layers(
        self,
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'],
        mask: Bool['b n'] | None = None

    ) -> Tuple[Float['b n ds'], Float['b n n dp']]:

        for _ in range(self.recurrent_depth):
            for (
                pairwise_block,
                pair_bias_attn,
                single_transition
            ) in self.layers:

                pairwise_repr = pairwise_block(pairwise_repr = pairwise_repr, mask = mask)

                single_repr = pair_bias_attn(single_repr, pairwise_repr = pairwise_repr, mask = mask) + single_repr
                single_repr = single_transition(single_repr) + single_repr

        return single_repr, pairwise_repr

    @typecheck
    def to_checkpointed_layers(
        self,
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'],
        mask: Bool['b n'] | None = None

    ) -> Tuple[Float['b n ds'], Float['b n n dp']]:

        inputs = (single_repr, pairwise_repr, mask)

        def pairwise_block_wrapper(layer):
            @wraps(layer)
            def inner(inputs, *args, **kwargs):
                single_repr, pairwise_repr, mask = inputs
                pairwise_repr = layer(pairwise_repr = pairwise_repr, mask = mask)
                return single_repr, pairwise_repr, mask
            return inner

        def pair_bias_attn_wrapper(layer):
            @wraps(layer)
            def inner(inputs, *args, **kwargs):
                single_repr, pairwise_repr, mask = inputs
                single_repr = layer(single_repr, pairwise_repr = pairwise_repr, mask = mask) + single_repr
                return single_repr, pairwise_repr, mask
            return inner

        def single_transition_wrapper(layer):
            @wraps(layer)
            def inner(inputs, *args, **kwargs):
                single_repr, pairwise_repr, mask = inputs
                single_repr = layer(single_repr) + single_repr
                return single_repr, pairwise_repr, mask
            return inner

        wrapped_layers = []

        for _ in range(self.recurrent_depth):
            for (
                pairwise_block,
                pair_bias_attn,
                single_transition
            ) in self.layers:

                wrapped_layers.append(pairwise_block_wrapper(pairwise_block))
                wrapped_layers.append(pair_bias_attn_wrapper(pair_bias_attn))
                wrapped_layers.append(single_transition_wrapper(single_transition))

        single_repr, pairwise_repr, _ = checkpoint_sequential(wrapped_layers, self.checkpoint_segments, inputs)

        return single_repr, pairwise_repr

    @typecheck
    def forward(
        self,
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'],
        mask: Bool['b n'] | None = None

    ) -> Tuple[Float['b n ds'], Float['b n n dp']]:

        # prepend register tokens

        if self.has_registers:
            batch_size, num_registers = single_repr.shape[0], self.num_registers
            single_registers = repeat(self.single_registers, 'r d -> b r d', b = batch_size)
            single_repr = torch.cat((single_registers, single_repr), dim = 1)

            row_registers = repeat(self.pairwise_row_registers, 'r d -> b r n d', b = batch_size, n = pairwise_repr.shape[-2])
            pairwise_repr = torch.cat((row_registers, pairwise_repr), dim = 1)
            col_registers = repeat(self.pairwise_col_registers, 'r d -> b n r d', b = batch_size, n = pairwise_repr.shape[1])
            pairwise_repr = torch.cat((col_registers, pairwise_repr), dim = 2)

            if exists(mask):
                mask = F.pad(mask, (num_registers, 0), value = True)

        # maybe checkpoint

        if should_checkpoint(self, (single_repr, pairwise_repr)):
            to_layers_fn = self.to_checkpointed_layers
        else:
            to_layers_fn = self.to_layers

        # main transformer block layers

        single_repr, pairwise_repr = to_layers_fn(
            single_repr = single_repr,
            pairwise_repr = pairwise_repr,
            mask = mask
        )

        # splice out registers

        if self.has_registers:
            single_repr = single_repr[:, num_registers:]
            pairwise_repr = pairwise_repr[:, num_registers:, num_registers:]

        return single_repr, pairwise_repr

# embedding related

class RelativePositionEncoding(Module):
    """ Algorithm 3 """
    
    def __init__(
        self,
        *,
        r_max = 32,
        s_max = 2,
        dim_out = 128
    ):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        
        dim_input = (2*r_max+2) + (2*r_max+2) + 1 + (2*s_max+2)
        self.out_embedder = LinearNoBias(dim_input, dim_out)

    @typecheck
    def forward(
        self,
        *,
        additional_molecule_feats: Int[f'b n {ADDITIONAL_MOLECULE_FEATS}']
    ) -> Float['b n n dp']:

        dtype = self.out_embedder.weight.dtype
        device = additional_molecule_feats.device

        res_idx, token_idx, asym_id, entity_id, sym_id = additional_molecule_feats.unbind(dim = -1)
        
        diff_res_idx = einx.subtract('b i, b j -> b i j', res_idx, res_idx)
        diff_token_idx = einx.subtract('b i, b j -> b i j', token_idx, token_idx)
        diff_sym_id = einx.subtract('b i, b j -> b i j', sym_id, sym_id)

        mask_same_chain = einx.subtract('b i, b j -> b i j', asym_id, asym_id) == 0
        mask_same_res = diff_res_idx == 0
        mask_same_entity = einx.subtract('b i, b j -> b i j 1', entity_id, entity_id) == 0
        
        d_res = torch.where(
            mask_same_chain, 
            torch.clip(diff_res_idx + self.r_max, 0, 2*self.r_max),
            2*self.r_max + 1
        )

        d_token = torch.where(
            mask_same_chain * mask_same_res, 
            torch.clip(diff_token_idx + self.r_max, 0, 2*self.r_max),
            2*self.r_max + 1
        )

        d_chain = torch.where(
            ~mask_same_chain, 
            torch.clip(diff_sym_id + self.s_max, 0, 2*self.s_max),
            2*self.s_max + 1
        )
        
        def onehot(x, bins):
            dist_from_bins = einx.subtract('... i, j -> ... i j', x, bins)
            indices = dist_from_bins.abs().min(dim = -1, keepdim = True).indices
            one_hots = F.one_hot(indices.long(), num_classes = len(bins))
            return one_hots.type(dtype)

        r_arange = torch.arange(2*self.r_max + 2, device = device)
        s_arange = torch.arange(2*self.s_max + 2, device = device)

        a_rel_pos = onehot(d_res, r_arange)
        a_rel_token = onehot(d_token, r_arange)
        a_rel_chain = onehot(d_chain, s_arange)

        out, _ = pack((
            a_rel_pos,
            a_rel_token,
            mask_same_entity,
            a_rel_chain
        ), 'b i j *')

        return self.out_embedder(out)

class TemplateEmbedder(Module):
    """ Algorithm 16 """

    def __init__(
        self,
        *,
        dim_template_feats,
        dim = 64,
        dim_pairwise = 128,
        pairformer_stack_depth = 2,
        pairwise_block_kwargs: dict = dict(),
        eps = 1e-5,
        checkpoint = False,
        checkpoint_segments = 1,
        layerscale_output = True
    ):
        super().__init__()
        self.eps = eps

        self.template_feats_to_embed_input = LinearNoBias(dim_template_feats, dim)

        self.pairwise_to_embed_input = nn.Sequential(
            nn.LayerNorm(dim_pairwise),
            LinearNoBias(dim_pairwise, dim)
        )

        layers = ModuleList([])
        for _ in range(pairformer_stack_depth):
            block = PairwiseBlock(
                dim_pairwise = dim,
                **pairwise_block_kwargs
            )

            layers.append(block)

        self.pairformer_stack = layers

        self.checkpoint = checkpoint
        self.checkpoint_segments = checkpoint_segments

        self.final_norm = nn.LayerNorm(dim)

        # final projection of mean pooled repr -> out

        self.to_out = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(dim, dim_pairwise)
        )

        self.layerscale = nn.Parameter(torch.zeros(dim_pairwise)) if layerscale_output else 1.

    @typecheck
    def to_layers(
        self,
        templates: Float['bt n n dt'],
        *,
        mask: Bool['bt n'] | None = None
    ) -> Float['bt n n dt']:

        for block in self.pairformer_stack:
            templates = block(
                pairwise_repr = templates,
                mask = mask
            ) + templates

        return templates

    @typecheck
    def to_checkpointed_layers(
        self,
        templates: Float['bt n n dt'],
        *,
        mask: Bool['bt n'] | None = None
    ) -> Float['bt n n dt']:

        wrapped_layers = []
        inputs = (templates, mask)

        def block_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                templates, mask = inputs
                templates = fn(pairwise_repr = templates, mask = mask)
                return templates, mask
            return inner

        for block in self.pairformer_stack:
            wrapped_layers.append(block_wrapper(block))

        templates, _ = checkpoint_sequential(wrapped_layers, self.checkpoint_segments, inputs)

        return templates

    @typecheck
    def forward(
        self,
        *,
        templates: Float['b t n n dt'],
        template_mask: Bool['b t'],
        pairwise_repr: Float['b n n dp'],
        mask: Bool['b n'] | None = None,
    ) -> Float['b n n dp']:

        dtype = templates.dtype
        num_templates = templates.shape[1]

        pairwise_repr = self.pairwise_to_embed_input(pairwise_repr)
        pairwise_repr = rearrange(pairwise_repr, 'b i j d -> b 1 i j d')

        templates = self.template_feats_to_embed_input(templates) + pairwise_repr

        templates, unpack_one = pack_one(templates, '* i j d')

        has_templates = reduce(template_mask, 'b t -> b', 'any')

        if exists(mask):
            mask = repeat(mask, 'b n -> (b t) n', t = num_templates)

        # going through the pairformer stack

        if should_checkpoint(self, templates):
            to_layers_fn = self.to_checkpointed_layers
        else:
            to_layers_fn = self.to_layers

        # layers

        templates = to_layers_fn(templates, mask = mask)

        # final norm

        templates = self.final_norm(templates)

        templates = unpack_one(templates)

        # masked mean pool template repr

        templates = einx.where(
            'b t, b t ..., -> b t ...',
            template_mask, templates, 0.
        )

        num = reduce(templates, 'b t i j d -> b i j d', 'sum')
        den = reduce(template_mask.type(dtype), 'b t -> b', 'sum')

        avg_template_repr = einx.divide('b i j d, b -> b i j d', num, den.clamp(min = self.eps))

        out = self.to_out(avg_template_repr)

        out = einx.where(
            'b, b ..., -> b ...',
            has_templates, out, 0.
        )

        return out * self.layerscale

# diffusion related
# both diffusion transformer as well as atom encoder / decoder

class FourierEmbedding(Module):
    """ Algorithm 22 """

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(1, dim)
        self.proj.requires_grad_(False)

    @typecheck
    def forward(
        self,
        times: Float[' b'],
    ) -> Float['b d']:

        times = rearrange(times, 'b -> b 1')
        rand_proj = self.proj(times)
        return torch.cos(2 * pi * rand_proj)

class PairwiseConditioning(Module):
    """ Algorithm 21 """

    def __init__(
        self,
        *,
        dim_pairwise_trunk,
        dim_pairwise_rel_pos_feats,
        dim_pairwise = 128,
        num_transitions = 2,
        transition_expansion_factor = 2,
    ):
        super().__init__()

        self.dim_pairwise_init_proj = nn.Sequential(
            LinearNoBias(dim_pairwise_trunk + dim_pairwise_rel_pos_feats, dim_pairwise),
            nn.LayerNorm(dim_pairwise)
        )

        transitions = ModuleList([])
        for _ in range(num_transitions):
            transition = PreLayerNorm(Transition(dim = dim_pairwise, expansion_factor = transition_expansion_factor), dim = dim_pairwise)
            transitions.append(transition)

        self.transitions = transitions

    @typecheck
    def forward(
        self,
        *,
        pairwise_trunk: Float['b n n dpt'],
        pairwise_rel_pos_feats: Float['b n n dpr'],
    ) -> Float['b n n dp']:

        pairwise_repr = torch.cat((pairwise_trunk, pairwise_rel_pos_feats), dim = -1)

        pairwise_repr = self.dim_pairwise_init_proj(pairwise_repr)

        for transition in self.transitions:
            pairwise_repr = transition(pairwise_repr) + pairwise_repr

        return pairwise_repr

class SingleConditioning(Module):
    """ Algorithm 21 """

    def __init__(
        self,
        *,
        sigma_data: float,
        dim_single = 384,
        dim_fourier = 256,
        num_transitions = 2,
        transition_expansion_factor = 2,
        eps = 1e-20
    ):
        super().__init__()
        self.eps = eps

        self.dim_single = dim_single
        self.sigma_data = sigma_data

        self.norm_single = nn.LayerNorm(dim_single)

        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.norm_fourier = nn.LayerNorm(dim_fourier)
        self.fourier_to_single = LinearNoBias(dim_fourier, dim_single)

        transitions = ModuleList([])
        for _ in range(num_transitions):
            transition = PreLayerNorm(Transition(dim = dim_single, expansion_factor = transition_expansion_factor), dim = dim_single)
            transitions.append(transition)

        self.transitions = transitions

    @typecheck
    def forward(
        self,
        *,
        times: Float[' b'],
        single_trunk_repr: Float['b n dst'],
        single_inputs_repr: Float['b n dsi'],
    ) -> Float['b n (dst+dsi)']:

        single_repr = torch.cat((single_trunk_repr, single_inputs_repr), dim = -1)

        assert single_repr.shape[-1] == self.dim_single

        single_repr = self.norm_single(single_repr)

        fourier_embed = self.fourier_embed(0.25 * log(times / self.sigma_data, eps = self.eps))

        normed_fourier = self.norm_fourier(fourier_embed)

        fourier_to_single = self.fourier_to_single(normed_fourier)

        single_repr = rearrange(fourier_to_single, 'b d -> b 1 d') + single_repr

        for transition in self.transitions:
            single_repr = transition(single_repr) + single_repr

        return single_repr


class AtomToTokenPooler(Module):
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.proj = nn.Sequential(
            LinearNoBias(dim, dim_out),
            nn.ReLU()
        )

    @typecheck
    def forward(
        self,
        *,
        atom_feats: Float['b m da'],
        atom_mask: Bool['b m'],
        molecule_atom_lens: Int['b n']
    ) -> Float['b n ds']:

        atom_feats = self.proj(atom_feats)
        tokens = mean_pool_with_lens(atom_feats, molecule_atom_lens)
        return tokens

# elucidated diffusion model adapted for atom position diffusing
# from Karras et al.
# https://arxiv.org/abs/2206.00364

class DiffusionLossBreakdown(NamedTuple):
    diffusion_mse: Float['']
    diffusion_bond: Float['']
    diffusion_smooth_lddt: Float['']

class ElucidatedAtomDiffusionReturn(NamedTuple):
    loss: Float['']
    denoised_atom_pos: Float['ba m 3']
    loss_breakdown: DiffusionLossBreakdown
    noise_sigmas: Float[' ba']

# modules todo

class SmoothLDDTLoss(Module):
    """ Algorithm 27 """

    @typecheck
    def __init__(
        self,
        nucleic_acid_cutoff: float = 30.0,
        other_cutoff: float = 15.0
    ):
        super().__init__()
        self.nucleic_acid_cutoff = nucleic_acid_cutoff
        self.other_cutoff = other_cutoff

        self.register_buffer('lddt_thresholds', torch.tensor([0.5, 1.0, 2.0, 4.0]))

    @typecheck
    def forward(
        self,
        pred_coords: Float['b n 3'],
        true_coords: Float['b n 3'],
        is_dna: Bool['b n'],
        is_rna: Bool['b n'],
        coords_mask: Bool['b n'] | None = None,
    ) -> Float['']:
        """
        pred_coords: predicted coordinates
        true_coords: true coordinates
        is_dna: boolean tensor indicating DNA atoms
        is_rna: boolean tensor indicating RNA atoms
        """
        # Compute distances between all pairs of atoms
        device = pred_coords.device

        pred_dists = torch.cdist(pred_coords, pred_coords)
        true_dists = torch.cdist(true_coords, true_coords)

        # Compute distance difference for all pairs of atoms
        dist_diff = torch.abs(true_dists - pred_dists)

        # Compute epsilon values

        eps = einx.subtract('thresholds, ... -> ... thresholds', self.lddt_thresholds, dist_diff)
        eps = eps.sigmoid().mean(dim = -1)

        # Restrict to bespoke inclusion radius
        is_nucleotide = is_dna | is_rna
        is_nucleotide_pair = to_pairwise_mask(is_nucleotide)

        inclusion_radius = torch.where(
            is_nucleotide_pair,
            true_dists < self.nucleic_acid_cutoff,
            true_dists < self.other_cutoff
        )

        # Compute mean, avoiding self term
        mask = inclusion_radius & ~torch.eye(pred_coords.shape[1], dtype=torch.bool, device=pred_coords.device)

        # Take into account variable lengthed atoms in batch
        if exists(coords_mask):
            paired_coords_mask = to_pairwise_mask(coords_mask)
            mask = mask & paired_coords_mask

        # Calculate masked averaging
        lddt = masked_average(eps, mask = mask, dim = (-1, -2), eps = 1)

        return 1. - lddt.mean()



# input embedder

class EmbeddedInputs(NamedTuple):
    single_inputs: Float['b n ds']
    single_init: Float['b n ds']
    pairwise_init: Float['b n n dp']
    atom_feats: Float['b m da']
    atompair_feats: Float['b m m dap']


# more confidence / clash calculations

class ConfidenceScore(NamedTuple):
    """The ConfidenceScore class."""

    plddt: Float["b m"]
    ptm: Float[" b"]  
    iptm: Float[" b"] | None  


# confidence head

class ConfidenceHeadLogits(NamedTuple):
    pae: Float['b pae n n'] |  None
    pde: Float['b pde n n']
    plddt: Float['b plddt m']
    resolved: Float['b 2 m']

class Alphafold3Logits(NamedTuple):
    pae: Float['b pae n n'] |  None
    pde: Float['b pde n n']
    plddt: Float['b plddt m']
    resolved: Float['b 2 m']
    distance: Float['b dist m m'] | Float['b dist n n'] | None




# model selection

@typecheck
def get_cid_molecule_type(
    cid: int,
    asym_id: Int[" n"],  
    is_molecule_types: Bool[f"n {IS_MOLECULE_TYPES}"],  
    return_one_hot: bool = False,
) -> int | Bool[f" {IS_MOLECULE_TYPES}"]:  
    """Get the (majority) molecule type for where `asym_id == cid`.

    NOTE: Several PDB chains contain multiple molecule types, so
    we must choose a single molecule type for the chain. We choose
    the molecule type that is most common (i.e., the mode) in the chain.

    :param cid: chain id
    :param asym_id: [n] asym_id of each residue
    :param is_molecule_types: [n 2] is_molecule_types
    :param return_one_hot: return one hot
    :return: molecule type
    """

    cid_is_molecule_types = is_molecule_types[asym_id == cid]

    molecule_types = cid_is_molecule_types.int().argmax(1)
    molecule_type_mode = molecule_types.mode()
    molecule_type = cid_is_molecule_types[molecule_type_mode.indices.item()]

    if not return_one_hot:
        molecule_type = molecule_type_mode.values.item()
    return molecule_type


@typecheck
def _protein_structure_from_feature(
    asym_id: Int[" n"],  
    molecule_ids: Int[" n"],  
    molecule_atom_lens: Int[" n"],  
    atom_pos: Float["m 3"],  
    atom_mask: Bool[" m"],  
) -> Bio.PDB.Structure.Structure:
    """Create structure for unresolved proteins.

    :param atom_mask: True for valid atoms, False for missing/padding atoms
    return: A Biopython Structure object
    """

    num_atom = atom_pos.shape[0]
    num_res = molecule_ids.shape[0]

    residue_constants = get_residue_constants(res_chem_index=IS_PROTEIN)

    molecule_atom_indices = exclusive_cumsum(molecule_atom_lens)

    builder = StructureBuilder()
    builder.init_structure("structure")
    builder.init_model(0)

    cur_cid = None
    cur_res_id = None

    for res_idx in range(num_res):
        num_atom = molecule_atom_lens[res_idx]
        cid = str(asym_id[res_idx].detach().cpu().item())

        if cid != cur_cid:
            builder.init_chain(cid)
            builder.init_seg(segid=" ")
            cur_cid = cid
            cur_res_id = 0

        restype = residue_constants.restypes[molecule_ids[res_idx]]
        resname = residue_constants.restype_1to3[restype]
        atom_names = residue_constants.restype_name_to_compact_atom_names[resname]
        atom_names = list(filter(lambda x: x, atom_names))
        # assume residues for unresolved protein are standard
        assert (
            len(atom_names) == num_atom
        ), f"Molecule atom lens {num_atom} doesn't match with residue constant {len(atom_names)}"

        # skip if all atom of the residue is missing
        atom_idx_offset = molecule_atom_indices[res_idx]
        if not torch.any(atom_mask[atom_idx_offset : atom_idx_offset + num_atom]):
            continue

        builder.init_residue(resname, " ", cur_res_id + 1, " ")
        cur_res_id += 1

        for atom_idx in range(num_atom):
            if not atom_mask[atom_idx]:
                continue

            atom_coord = atom_pos[atom_idx + atom_idx_offset].detach().cpu().numpy()
            atom_name = atom_names[atom_idx]
            builder.init_atom(
                name=atom_name,
                coord=atom_coord,
                b_factor=1.0,
                occupancy=1.0,
                fullname=atom_name,
                altloc=" ",
                # only N, C, O in restype_name_to_compact_atom_names for protein
                # so just take the first char
                element=atom_name[0],
            )

    return builder.get_structure()

Sample = Tuple[Float["b m 3"], Float["b pde n n"], Float["b m"], Float["b dist n n"]]
ScoredSample = Tuple[int, Float["b m 3"], Float["b m"], Float[" b"], Float[" b"]]

class ScoreDetails(NamedTuple):
    best_gpde_index: int
    best_lddt_index: int
    score: Float[' b']
    scored_samples: List[ScoredSample]


# main class

class LossBreakdown(NamedTuple):
    total_loss: Float['']
    total_diffusion: Float['']
    distogram: Float['']
    pae: Float['']
    pde: Float['']
    plddt: Float['']
    resolved: Float['']
    confidence: Float['']
    diffusion_mse: Float['']
    diffusion_bond: Float['']
    diffusion_smooth_lddt: Float['']

# an alphafold3 that can download pretrained weights from huggingface

class Alphafold3WithHubMixin(Alphafold3, PyTorchModelHubMixin):
    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        proxies: Dict | None,
        resume_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        map_location: str = 'cpu',
        strict: bool = False,
        model_filename: str = 'alphafold3.bin',
        **model_kwargs,
    ):
        model_file = Path(model_id) / model_filename

        if not model_file.exists():
            model_file = hf_hub_download(
                repo_id = model_id,
                filename = model_filename,
                revision = revision,
                cache_dir = cache_dir,
                force_download = force_download,
                proxies = proxies,
                resume_download = resume_download,
                token = token,
                local_files_only = local_files_only,
            )

        model = cls.init_and_load(
            model_file,
            strict = strict,
            map_location = map_location
        )

        return model
