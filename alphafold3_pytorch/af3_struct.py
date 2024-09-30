# from lucidrain code
# basic diffusion with edm

import torch
import torch.nn as nn
import torch.nn.functional as F

import functools

from torch import Tensor
from beartype.typing import NamedTuple



class DiffusionLossBreakdown(NamedTuple):
    diffusion_mse: Tensor
    diffusion_bond: Tensor
    diffusion_smooth_lddt: Tensor

class DiffusionModule(nn.Module):
    """ Algorithm 20 """
    def __init__(
        self,
        *,
        dim_pairwise_trunk,
        dim_pairwise_rel_pos_feats,
        atoms_per_window = 27,  # for atom sequence, take the approach of (batch, seq, atoms, ..), where atom dimension is set to the molecule or molecule with greatest number of atoms, the rest padded. atom_mask must be passed in - default to 27 for proteins, with tryptophan having 27 atoms
        dim_pairwise = 128,
        sigma_data = 16,
        dim_atom = 128,
        dim_atompair = 16,
        dim_token = 768,
        dim_single = 384,
        dim_fourier = 256,
        single_cond_kwargs: dict = dict(
            num_transitions = 2,
            transition_expansion_factor = 2,
        ),
        pairwise_cond_kwargs: dict = dict(
            num_transitions = 2
        ),
        atom_encoder_depth = 3,
        atom_encoder_heads = 4,
        token_transformer_depth = 24,
        token_transformer_heads = 16,
        atom_decoder_depth = 3,
        atom_decoder_heads = 4,
        serial = True,
        atom_encoder_kwargs: dict = dict(),
        atom_decoder_kwargs: dict = dict(),
        token_transformer_kwargs: dict = dict(),
        use_linear_attn = False,
        checkpoint = False,
        linear_attn_kwargs: dict = dict(
            heads = 8,
            dim_head = 16
        )
    ):
        super().__init__()

        self.atoms_per_window = atoms_per_window

        # conditioning

        self.single_conditioner = SingleConditioning(
            sigma_data = sigma_data,
            dim_single = dim_single,
            dim_fourier = dim_fourier,
            **single_cond_kwargs
        )

        self.pairwise_conditioner = PairwiseConditioning(
            dim_pairwise_trunk = dim_pairwise_trunk,
            dim_pairwise_rel_pos_feats = dim_pairwise_rel_pos_feats,
            dim_pairwise = dim_pairwise,
            **pairwise_cond_kwargs
        )

        # atom attention encoding related modules

        self.atom_pos_to_atom_feat = LinearNoBias(3, dim_atom)

        self.missing_atom_feat = nn.Parameter(torch.zeros(dim_atom))

        self.single_repr_to_atom_feat_cond = nn.Sequential(
            nn.LayerNorm(dim_single),
            LinearNoBias(dim_single, dim_atom)
        )

        self.pairwise_repr_to_atompair_feat_cond = nn.Sequential(
            nn.LayerNorm(dim_pairwise),
            LinearNoBias(dim_pairwise, dim_atompair)
        )

        self.atom_repr_to_atompair_feat_cond = nn.Sequential(
            nn.LayerNorm(dim_atom),
            LinearNoBias(dim_atom, dim_atompair * 2),
            nn.ReLU()
        )

        self.atompair_feats_mlp = nn.Sequential(
            LinearNoBias(dim_atompair, dim_atompair),
            nn.ReLU(),
            LinearNoBias(dim_atompair, dim_atompair),
            nn.ReLU(),
            LinearNoBias(dim_atompair, dim_atompair),
        )

        self.atom_encoder = DiffusionTransformer(
            dim = dim_atom,
            dim_single_cond = dim_atom,
            dim_pairwise = dim_atompair,
            attn_window_size = atoms_per_window,
            depth = atom_encoder_depth,
            heads = atom_encoder_heads,
            serial = serial,
            use_linear_attn = use_linear_attn,
            linear_attn_kwargs = linear_attn_kwargs,
            checkpoint = checkpoint,
            **atom_encoder_kwargs
        )

        self.atom_feats_to_pooled_token = AtomToTokenPooler(
            dim = dim_atom,
            dim_out = dim_token
        )

        # token attention related modules

        self.cond_tokens_with_cond_single = nn.Sequential(
            nn.LayerNorm(dim_single),
            LinearNoBias(dim_single, dim_token)
        )

        self.token_transformer = DiffusionTransformer(
            dim = dim_token,
            dim_single_cond = dim_single,
            dim_pairwise = dim_pairwise,
            depth = token_transformer_depth,
            heads = token_transformer_heads,
            serial = serial,
            checkpoint = checkpoint,
            **token_transformer_kwargs
        )

        self.attended_token_norm = nn.LayerNorm(dim_token)

        # atom attention decoding related modules

        self.tokens_to_atom_decoder_input_cond = LinearNoBias(dim_token, dim_atom)

        self.atom_decoder = DiffusionTransformer(
            dim = dim_atom,
            dim_single_cond = dim_atom,
            dim_pairwise = dim_atompair,
            attn_window_size = atoms_per_window,
            depth = atom_decoder_depth,
            heads = atom_decoder_heads,
            serial = serial,
            use_linear_attn = use_linear_attn,
            linear_attn_kwargs = linear_attn_kwargs,
            checkpoint = checkpoint,
            **atom_decoder_kwargs
        )

        self.atom_feat_to_atom_pos_update = nn.Sequential(
            nn.LayerNorm(dim_atom),
            LinearNoBias(dim_atom, 3)
        )

    def forward(
        self,
        noised_atom_pos,
        *,
        atom_feats,
        atompair_feats ,
        atom_mask ,
        times ,
        mask ,
        single_trunk_repr ,
        single_inputs_repr ,
        pairwise_trunk ,
        pairwise_rel_pos_feats ,
        molecule_atom_lens ,
        atom_parent_ids ,
        missing_atom_mask
    ):
        w = self.atoms_per_window
        device = noised_atom_pos.device

        batch_size, seq_len = single_trunk_repr.shape[:2]
        atom_seq_len = atom_feats.shape[1]

        conditioned_single_repr = self.single_conditioner(
            times = times,
            single_trunk_repr = single_trunk_repr,
            single_inputs_repr = single_inputs_repr
        )

        conditioned_pairwise_repr = self.pairwise_conditioner(
            pairwise_trunk = pairwise_trunk,
            pairwise_rel_pos_feats = pairwise_rel_pos_feats
        )

        # lines 7-14 in Algorithm 5

        atom_feats_cond = atom_feats

        # the most surprising part of the paper; no geometric biases!

        noised_atom_pos_feats = self.atom_pos_to_atom_feat(noised_atom_pos)

        # for missing atoms, replace the noise atom pos features with a missing embedding

        if exists(missing_atom_mask):
            noised_atom_pos_feats = einx.where('b m, d, b m d -> b m d', missing_atom_mask, self.missing_atom_feat, noised_atom_pos_feats)

        # sum the noised atom position features to the atom features

        atom_feats = noised_atom_pos_feats + atom_feats

        # condition atom feats cond (cl) with single repr

        single_repr_cond = self.single_repr_to_atom_feat_cond(conditioned_single_repr)

        single_repr_cond = batch_repeat_interleave(single_repr_cond, molecule_atom_lens)
        single_repr_cond = pad_or_slice_to(single_repr_cond, length = atom_feats_cond.shape[1], dim = 1)

        atom_feats_cond = single_repr_cond + atom_feats_cond

        # window the atom pair features before passing to atom encoder and decoder if necessary

        atompair_is_windowed = atompair_feats.ndim == 5

        if not atompair_is_windowed:
            atompair_feats = full_pairwise_repr_to_windowed(atompair_feats, window_size = self.atoms_per_window)

        # condition atompair feats with pairwise repr

        pairwise_repr_cond = self.pairwise_repr_to_atompair_feat_cond(conditioned_pairwise_repr)

        indices = torch.arange(seq_len, device = device)
        indices = repeat(indices, 'n -> b n', b = batch_size)

        indices = batch_repeat_interleave(indices, molecule_atom_lens)
        indices = pad_or_slice_to(indices, atom_seq_len, dim = -1)
        indices = pad_and_window(indices, w)

        row_indices = col_indices = indices
        row_indices = rearrange(row_indices, 'b n w -> b n w 1', w = w)
        col_indices = rearrange(col_indices, 'b n w -> b n 1 w', w = w)

        col_indices = concat_previous_window(col_indices, dim_seq = 1, dim_window = -1)
        row_indices, col_indices = torch.broadcast_tensors(row_indices, col_indices)

        # pairwise_repr_cond = einx.get_at('b [i j] dap, b nw w1 w2, b nw w1 w2 -> b nw w1 w2 dap', pairwise_repr_cond, row_indices, col_indices)

        row_indices, unpack_one = pack_one(row_indices, 'b *')
        col_indices, _ = pack_one(col_indices, 'b *')

        rowcol_indices = col_indices + row_indices * pairwise_repr_cond.shape[2]
        rowcol_indices = repeat(rowcol_indices, 'b rc -> b rc dap', dap = pairwise_repr_cond.shape[-1])
        pairwise_repr_cond, _ = pack_one(pairwise_repr_cond, 'b * dap')

        pairwise_repr_cond = pairwise_repr_cond.gather(1, rowcol_indices)
        pairwise_repr_cond = unpack_one(pairwise_repr_cond, 'b * dap')
        
        atompair_feats = pairwise_repr_cond + atompair_feats

        # condition atompair feats further with single atom repr

        atom_repr_cond = self.atom_repr_to_atompair_feat_cond(atom_feats)
        atom_repr_cond = pad_and_window(atom_repr_cond, w)

        atom_repr_cond_row, atom_repr_cond_col = atom_repr_cond.chunk(2, dim = -1)

        atom_repr_cond_col = concat_previous_window(atom_repr_cond_col, dim_seq = 1, dim_window = 2)

        atompair_feats = einx.add('b nw w1 w2 dap, b nw w1 dap -> b nw w1 w2 dap', atompair_feats, atom_repr_cond_row)
        atompair_feats = einx.add('b nw w1 w2 dap, b nw w2 dap -> b nw w1 w2 dap', atompair_feats, atom_repr_cond_col)

        # furthermore, they did one more MLP on the atompair feats for attention biasing in atom transformer

        atompair_feats = self.atompair_feats_mlp(atompair_feats) + atompair_feats

        # take care of restricting atom attention to be intra molecular, if the atom_parent_ids were passed in

        windowed_mask = None

        if exists(atom_parent_ids):
            atom_parent_ids_rows = pad_and_window(atom_parent_ids, w)
            atom_parent_ids_columns = concat_previous_window(atom_parent_ids_rows, dim_seq = 1, dim_window = 2)

            windowed_mask = einx.equal('b n i, b n j -> b n i j', atom_parent_ids_rows, atom_parent_ids_columns)

        # atom encoder

        atom_feats = self.atom_encoder(
            atom_feats,
            mask = atom_mask,
            windowed_mask = windowed_mask,
            single_repr = atom_feats_cond,
            pairwise_repr = atompair_feats
        )

        atom_feats_skip = atom_feats

        tokens = self.atom_feats_to_pooled_token(
            atom_feats = atom_feats,
            atom_mask = atom_mask,
            molecule_atom_lens = molecule_atom_lens
        )

        # token transformer

        tokens = self.cond_tokens_with_cond_single(conditioned_single_repr) + tokens

        tokens = self.token_transformer(
            tokens,
            mask = mask,
            single_repr = conditioned_single_repr,
            pairwise_repr = conditioned_pairwise_repr,
        )

        tokens = self.attended_token_norm(tokens)

        # atom decoder

        atom_decoder_input = self.tokens_to_atom_decoder_input_cond(tokens)

        atom_decoder_input = batch_repeat_interleave(atom_decoder_input, molecule_atom_lens)
        atom_decoder_input = pad_or_slice_to(atom_decoder_input, length = atom_feats_skip.shape[1], dim = 1)

        atom_decoder_input = atom_decoder_input + atom_feats_skip

        atom_feats = self.atom_decoder(
            atom_decoder_input,
            mask = atom_mask,
            windowed_mask = windowed_mask,
            single_repr = atom_feats_cond,
            pairwise_repr = atompair_feats
        )

        atom_pos_update = self.atom_feat_to_atom_pos_update(atom_feats)

        return atom_pos_update


class ElucidatedAtomDiffusionReturn(NamedTuple):
    loss: Tensor
    denoised_atom_pos: Tensor
    loss_breakdown: DiffusionLossBreakdown
    noise_sigmas: Tensor

class ElucidatedAtomDiffusion(nn.Module):
    # @typecheck
    def __init__(
        self,
        net: DiffusionModule,
        *,
        num_sample_steps = 32, # number of sampling steps
        sigma_min = 0.002,     # min noise level
        sigma_max = 80,        # max noise level
        sigma_data = 0.5,      # standard deviation of data distribution
        rho = 7,               # controls the sampling schedule
        P_mean = -1.2,         # mean of log-normal distribution from which noise is drawn for training
        P_std = 1.5,           # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in paper
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 1.003,
        step_scale = 1.5,
        augment_during_sampling = True,
        lddt_mask_kwargs: dict = dict(),
        smooth_lddt_loss_kwargs: dict = dict(),
        weighted_rigid_align_kwargs: dict = dict(),
        multi_chain_permutation_alignment_kwargs: dict = dict(),
        centre_random_augmentation_kwargs: dict = dict(),
        karras_formulation = True,  # use the original EDM formulation from Karras et al. Table 1 in https://arxiv.org/abs/2206.00364 - differences are that the noise and sampling schedules are scaled by sigma data, as well as loss weight adds the sigma data instead of multiply in denominator
    ):
        super().__init__()
        self.net = net

        # parameters

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper
        self.step_scale = step_scale

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

        # centre random augmenter

        self.augment_during_sampling = augment_during_sampling
        self.centre_random_augmenter = CentreRandomAugmentation(**centre_random_augmentation_kwargs)

        # weighted rigid align

        self.weighted_rigid_align = WeightedRigidAlign(**weighted_rigid_align_kwargs)

        # multi-chain permutation alignment

        self.multi_chain_permutation_alignment = MultiChainPermutationAlignment(
            **multi_chain_permutation_alignment_kwargs,
            weighted_rigid_align=self.weighted_rigid_align,
        )

        # smooth lddt loss

        self.smooth_lddt_loss = SmoothLDDTLoss(**smooth_lddt_loss_kwargs)

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # whether to use original karras formulation or not

        self.karras_formulation = karras_formulation

    @property
    def device(self):
        return next(self.net.parameters()).device

    @property
    def dtype(self):
        return next(self.net.parameters()).dtype

    # derived preconditioning params - Table 1

    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # preconditioned network output
    # equation (7) in the paper

    # @typecheck
    def preconditioned_network_forward(
        self,
        noised_atom_pos ,
        sigma:   float,
        network_condition_kwargs: dict,
        clamp = False,
    ):
        batch, dtype, device = (
            noised_atom_pos.shape[0],
            noised_atom_pos.dtype,
            noised_atom_pos.device,
        )

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, dtype=dtype, device=device)

        padded_sigma = rearrange(sigma, 'b -> b 1 1')

        net_out = self.net.forward(
            self.c_in(padded_sigma) * noised_atom_pos,
            times = sigma,
            **network_condition_kwargs
        )

        out = self.c_skip(padded_sigma) * noised_atom_pos +  self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1., 1.)

        return out

    # sampling

    # sample schedule
    # equation (5) in the paper

    def sample_schedule(self, num_sample_steps = None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device=self.device, dtype=self.dtype)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value = 0.) # last step is sigma value of 0.

        return sigmas * self.sigma_data

    @torch.no_grad()
    def sample(
        self,
        atom_mask ,
        num_sample_steps = None,
        clamp = False,
        use_tqdm_pbar = True,
        tqdm_pbar_title = 'sampling time step',
        return_all_timesteps = False,
        **network_condition_kwargs
    )  :

        dtype = self.dtype

        step_scale, num_sample_steps = self.step_scale, default(num_sample_steps, self.num_sample_steps)

        shape = (*atom_mask.shape, 3)

        network_condition_kwargs.update(atom_mask = atom_mask)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(num_sample_steps)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # atom position is noise at the beginning

        init_sigma = sigmas[0]

        atom_pos = init_sigma * torch.randn(shape, dtype = dtype, device = self.device)

        # gradually denoise

        maybe_tqdm_wrapper = tqdm if use_tqdm_pbar else identity

        maybe_augment_fn = self.centre_random_augmenter if self.augment_during_sampling else identity

        all_atom_pos = [atom_pos]

        for sigma, sigma_next, gamma in maybe_tqdm_wrapper(sigmas_and_gammas, desc = tqdm_pbar_title):
            sigma, sigma_next, gamma = tuple(t.item() for t in (sigma, sigma_next, gamma))

            atom_pos = maybe_augment_fn(atom_pos.float()).type(dtype)

            eps = self.S_noise * torch.randn(
                shape, dtype = dtype, device = self.device
            )  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            atom_pos_hat = atom_pos + sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            model_output = self.preconditioned_network_forward(atom_pos_hat, sigma_hat, clamp = clamp, network_condition_kwargs = network_condition_kwargs)
            denoised_over_sigma = (atom_pos_hat - model_output) / sigma_hat

            atom_pos_next = atom_pos_hat + (sigma_next - sigma_hat) * denoised_over_sigma * step_scale

            # second order correction, if not the last timestep

            if self.karras_formulation and sigma_next != 0:
                model_output_next = self.preconditioned_network_forward(atom_pos_next, sigma_next, clamp = clamp, network_condition_kwargs = network_condition_kwargs)
                denoised_prime_over_sigma = (atom_pos_next - model_output_next) / sigma_next
                atom_pos_next = atom_pos_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma) * step_scale

            atom_pos = atom_pos_next

            all_atom_pos.append(atom_pos)

        # if returning atom positions across all timesteps for visualization
        # then stack the `all_atom_pos`

        if return_all_timesteps:
            atom_pos = torch.stack(all_atom_pos)

        if clamp:
            atom_pos = atom_pos.clamp(-1., 1.)

        return atom_pos

    # training

    def karras_loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def loss_weight(self, sigma):
        """ for some reason, in paper they add instead of multiply as in original paper """
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma + self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device = self.device)).exp() * self.sigma_data

    def forward(
        self,
        atom_pos_ground_truth ,
        atom_mask ,
        atom_feats ,
        atompair_feats ,
        mask ,
        single_trunk_repr ,
        single_inputs_repr ,
        pairwise_trunk ,
        pairwise_rel_pos_feats ,
        molecule_atom_lens ,
        molecule_atom_indices ,
        token_bonds ,
        missing_atom_mask ,
        atom_parent_ids ,
        is_molecule_types,
        filepaths,
        additional_molecule_feats,
        return_denoised_pos = False,
        add_smooth_lddt_loss = False,
        add_bond_loss = False,
        nucleotide_loss_weight = 5.,
        ligand_loss_weight = 10.,
        return_loss_breakdown = False,
        single_structure_input=False,
    ) -> ElucidatedAtomDiffusionReturn:

        # diffusion loss

        dtype = atom_pos_ground_truth.dtype
        batch_size = atom_pos_ground_truth.shape[0]

        sigmas = self.noise_distribution(batch_size).type(dtype)
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1')

        noise = torch.randn_like(atom_pos_ground_truth)

        noised_atom_pos = atom_pos_ground_truth + padded_sigmas * noise  # alphas are 1. in the paper

        denoised_atom_pos = self.preconditioned_network_forward(
            noised_atom_pos,
            sigmas,
            network_condition_kwargs = dict(
                atom_feats = atom_feats,
                atom_mask = atom_mask,
                missing_atom_mask = missing_atom_mask,
                atompair_feats = atompair_feats,
                atom_parent_ids = atom_parent_ids,
                mask = mask,
                single_trunk_repr = single_trunk_repr,
                single_inputs_repr = single_inputs_repr,
                pairwise_trunk = pairwise_trunk,
                pairwise_rel_pos_feats = pairwise_rel_pos_feats,
                molecule_atom_lens = molecule_atom_lens
            )
        )

        # total loss, for accumulating all auxiliary losses

        total_loss = 0.

        # section 3.7.1 equation 2 - weighted rigid aligned ground truth

        align_weights = calculate_weighted_rigid_align_weights(
            atom_pos_ground_truth=atom_pos_ground_truth,
            molecule_atom_lens=molecule_atom_lens,
            is_molecule_types=is_molecule_types,
            nucleotide_loss_weight=nucleotide_loss_weight,
            ligand_loss_weight=ligand_loss_weight,
        )

        atom_pos_aligned_ground_truth = self.weighted_rigid_align(
            pred_coords=denoised_atom_pos.float(),
            true_coords=atom_pos_ground_truth.float(),
            weights=align_weights.float(),
            mask=atom_mask,
        ).type(dtype)

        # section 4.2 - multi-chain permutation alignment

        if single_structure_input:
            try:
                atom_pos_aligned_ground_truth = self.multi_chain_permutation_alignment(
                    pred_coords=denoised_atom_pos,
                    true_coords=atom_pos_aligned_ground_truth,
                    molecule_atom_lens=molecule_atom_lens,
                    molecule_atom_indices=molecule_atom_indices,
                    token_bonds=token_bonds,
                    additional_molecule_feats=additional_molecule_feats,
                    is_molecule_types=is_molecule_types,
                    mask=atom_mask,
                )
            except Exception as e:
                # NOTE: For many (random) unit test inputs, permutation alignment can be unstable
                logger.warning(f"Skipping multi-chain permutation alignment {f'for {filepaths}' if exists(filepaths) else ''} due to: {e}")

        # main diffusion mse loss

        losses = F.mse_loss(denoised_atom_pos, atom_pos_aligned_ground_truth, reduction = 'none') / 3.
        losses = einx.multiply('b m c, b m -> b m c',  losses, align_weights)

        # regular loss weight as defined in EDM paper

        loss_weight_fn = self.karras_loss_weight if self.karras_formulation else self.loss_weight

        loss_weights = loss_weight_fn(padded_sigmas)

        losses = losses * loss_weights

        # if there are missing atoms, update the atom mask to not include them in the loss

        if exists(missing_atom_mask):
            atom_mask = atom_mask & ~ missing_atom_mask

        # account for atom mask

        mse_loss = losses[atom_mask].mean()

        total_loss = total_loss + mse_loss

        # proposed extra bond loss during finetuning

        bond_loss = self.zero

        if add_bond_loss:
            atompair_mask = to_pairwise_mask(atom_mask)

            denoised_cdist = torch.cdist(denoised_atom_pos, denoised_atom_pos, p = 2)
            normalized_cdist = torch.cdist(atom_pos_ground_truth, atom_pos_ground_truth, p = 2)

            bond_losses = F.mse_loss(denoised_cdist, normalized_cdist, reduction = 'none')
            bond_losses = bond_losses * loss_weights

            bond_loss = bond_losses[atompair_mask].mean()

            total_loss = total_loss + bond_loss

        # proposed auxiliary smooth lddt loss

        smooth_lddt_loss = self.zero

        if add_smooth_lddt_loss:
            assert exists(is_molecule_types)

            is_nucleotide_or_ligand_fields = is_molecule_types.unbind(dim=-1)

            is_nucleotide_or_ligand_fields = tuple(
                batch_repeat_interleave(t, molecule_atom_lens)
                for t in is_nucleotide_or_ligand_fields
            )
            is_nucleotide_or_ligand_fields = tuple(
                pad_or_slice_to(t, length=align_weights.shape[-1], dim=-1)
                for t in is_nucleotide_or_ligand_fields
            )

            _, atom_is_dna, atom_is_rna, _, _ = is_nucleotide_or_ligand_fields

            smooth_lddt_loss = self.smooth_lddt_loss(
                denoised_atom_pos,
                atom_pos_ground_truth,
                atom_is_dna,
                atom_is_rna,
                coords_mask = atom_mask
            )

            total_loss = total_loss + smooth_lddt_loss

        # calculate loss breakdown

        loss_breakdown = DiffusionLossBreakdown(mse_loss, bond_loss, smooth_lddt_loss)

        return ElucidatedAtomDiffusionReturn(total_loss, denoised_atom_pos, loss_breakdown, sigmas)


# 在输入前需要重复倍数，num_augs
class StructureModel(nn.Module):
    def __init__(self,edm_kwargs):
        super(self,StructureModel).__init__()
        
        
        self.edm = ElucidatedAtomDiffusion(
            self.diffusion_module,
            sigma_data = sigma_data,
            smooth_lddt_loss_kwargs = dict(
                nucleic_acid_cutoff = lddt_mask_nucleic_acid_cutoff,
                other_cutoff = lddt_mask_other_cutoff,
            ),
            **edm_kwargs
        )
        
        pass
    
    def sample(self):
        sampled_atom_pos = self.edm.sample(
            num_sample_steps = num_sample_steps,
            atom_feats = atom_feats,
            atompair_feats = atompair_feats,
            atom_parent_ids = atom_parent_ids,
            atom_mask = atom_mask,
            mask = mask,
            single_trunk_repr = single,
            single_inputs_repr = single_inputs,
            pairwise_trunk = pairwise,
            pairwise_rel_pos_feats = relative_position_encoding,
            molecule_atom_lens = molecule_atom_lens,
            return_all_timesteps = return_all_diffused_atom_pos
        )
        if atom_mask is not None:
            sampled_atom_pos = einx.where('b m, ... b m c, -> ... b m c', atom_mask, sampled_atom_pos, 0.)
                    if return_bio_pdb_structures:
                assert not return_all_diffused_atom_pos

                sampled_atom_pos = [
                    protein_structure_from_feature(*args)
                    for args in zip(
                        additional_molecule_feats[..., 2],
                        molecule_ids,
                        molecule_atom_lens,
                        sampled_atom_pos,
                        atom_mask
                    )
                ]

    
    def forward(self,x):
        diffusion_loss, denoised_atom_pos, diffusion_loss_breakdown, _ = self.edm.forward(
            atom_pos,
            additional_molecule_feats = additional_molecule_feats,
            is_molecule_types = is_molecule_types,
            add_smooth_lddt_loss = diffusion_add_smooth_lddt_loss,
            add_bond_loss = diffusion_add_bond_loss,
            atom_feats = atom_feats,
            atompair_feats = atompair_feats,
            atom_parent_ids = atom_parent_ids,
            missing_atom_mask = missing_atom_mask,
            atom_mask = atom_mask,
            mask = mask,
            single_trunk_repr = single,
            single_inputs_repr = single_inputs,
            pairwise_trunk = pairwise,
            pairwise_rel_pos_feats = relative_position_encoding,
            molecule_atom_lens = molecule_atom_lens,
            molecule_atom_indices = molecule_atom_indices,
            token_bonds = token_bonds,
            return_denoised_pos = True,
            nucleotide_loss_weight = self.nucleotide_loss_weight,
            ligand_loss_weight = self.ligand_loss_weight,
            single_structure_input = single_structure_input,
            filepaths = filepaths,
        )
        if exists(atom_mask):
            sampled_atom_pos = einx.where('b m, ... b m c, -> ... b m c', atom_mask, sampled_atom_pos, 0.)
        if return_bio_pdb_structures:
            assert not return_all_diffused_atom_pos

            sampled_atom_pos = [
                protein_structure_from_feature(*args)
                for args in zip(
                    additional_molecule_feats[..., 2],
                    molecule_ids,
                    molecule_atom_lens,
                    sampled_atom_pos,
                    atom_mask
                )
            ]

        if not return_confidence_head_logits:
            return sampled_atom_pos

        diffusion_loss, denoised_atom_pos, diffusion_loss_breakdown, _ = self.edm.forward(
            atom_pos,
            additional_molecule_feats = additional_molecule_feats,
            is_molecule_types = is_molecule_types,
            add_smooth_lddt_loss = diffusion_add_smooth_lddt_loss,
            add_bond_loss = diffusion_add_bond_loss,
            atom_feats = atom_feats,
            atompair_feats = atompair_feats,
            atom_parent_ids = atom_parent_ids,
            missing_atom_mask = missing_atom_mask,
            atom_mask = atom_mask,
            mask = mask,
            single_trunk_repr = single,
            single_inputs_repr = single_inputs,
            pairwise_trunk = pairwise,
            pairwise_rel_pos_feats = relative_position_encoding,
            molecule_atom_lens = molecule_atom_lens,
            molecule_atom_indices = molecule_atom_indices,
            token_bonds = token_bonds,
            return_denoised_pos = True,
            nucleotide_loss_weight = self.nucleotide_loss_weight,
            ligand_loss_weight = self.ligand_loss_weight,
            single_structure_input = single_structure_input,
            filepaths = filepaths,
        )
        denoised_atom_pos = self.edm.sample(
            num_sample_steps=num_rollout_steps,
            atom_feats=atom_feats,
            atompair_feats=atompair_feats,
            atom_mask=atom_mask,
            mask=mask,
            single_trunk_repr=single,
            single_inputs_repr=single_inputs,
            pairwise_trunk=pairwise,
            pairwise_rel_pos_feats=relative_position_encoding,
            molecule_atom_lens=molecule_atom_lens,
            use_tqdm_pbar=rollout_show_tqdm_pbar,
            tqdm_pbar_title="Training rollout",
        )
        