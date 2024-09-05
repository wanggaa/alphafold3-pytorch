
class ElucidatedAtomDiffusion(Module):
    @typecheck
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

    @typecheck
    def preconditioned_network_forward(
        self,
        noised_atom_pos: Float['b m 3'],
        sigma: Float[' b'] | Float[' '] | float,
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

        net_out = self.net(
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
        atom_mask: Bool['b m'] | None = None,
        num_sample_steps = None,
        clamp = False,
        use_tqdm_pbar = True,
        tqdm_pbar_title = 'sampling time step',
        return_all_timesteps = False,
        **network_condition_kwargs
    ) -> Float['b m 3'] | Float['ts b m 3']:

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
        atom_pos_ground_truth: Float['b m 3'],
        atom_mask: Bool['b m'],
        atom_feats: Float['b m da'],
        atompair_feats: Float['b m m dap'],
        mask: Bool['b n'],
        single_trunk_repr: Float['b n dst'],
        single_inputs_repr: Float['b n dsi'],
        pairwise_trunk: Float['b n n dpt'],
        pairwise_rel_pos_feats: Float['b n n dpr'],
        molecule_atom_lens: Int['b n'],
        molecule_atom_indices: Int['b n'],
        token_bonds: Bool['b n n'],
        missing_atom_mask: Bool['b m'] | None = None,
        atom_parent_ids: Int['b m'] | None = None,
        return_denoised_pos = False,
        is_molecule_types: Bool[f'b n {IS_MOLECULE_TYPES}'] | None = None,
        additional_molecule_feats: Int[f'b n {ADDITIONAL_MOLECULE_FEATS}'] | None = None,
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
                logger.warning(f"Skipping multi-chain permutation alignment due to: {e}")

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
