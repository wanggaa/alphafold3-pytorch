
class DiffusionTransformer(Module):
    """ Algorithm 23 """

    def __init__(
        self,
        *,
        depth,
        heads,
        dim = 384,
        dim_single_cond = None,
        dim_pairwise = 128,
        attn_window_size = None,
        attn_pair_bias_kwargs: dict = dict(),
        attn_num_memory_kv = False,
        trans_expansion_factor = 2,
        num_register_tokens = 0,
        serial = True,
        add_residual = True,
        use_linear_attn = False,
        checkpoint = False,
        checkpoint_segments = 1,
        linear_attn_kwargs = dict(
            heads = 8,
            dim_head = 16
        ),
        use_colt5_attn = False,
        colt5_attn_kwargs = dict(
            heavy_dim_head = 64,
            heavy_heads = 8,
            num_heavy_tokens_q = 512,
            num_heavy_tokens_kv = 512
        )

    ):
        super().__init__()
        self.attn_window_size = attn_window_size

        dim_single_cond = default(dim_single_cond, dim)

        layers = ModuleList([])

        for _ in range(depth):

            linear_attn = None

            if use_linear_attn:
                linear_attn = TaylorSeriesLinearAttn(
                    dim = dim,
                    prenorm = True,
                    gate_value_heads = True,
                    remove_even_power_dups = True,
                    **linear_attn_kwargs
                )

            colt5_attn = None

            if use_colt5_attn:
                colt5_attn = ConditionalRoutedAttention(
                    dim = dim,
                    has_light_attn = False,
                    **colt5_attn_kwargs
                )

            pair_bias_attn = AttentionPairBias(
                dim = dim,
                dim_pairwise = dim_pairwise,
                heads = heads,
                window_size = attn_window_size,
                num_memory_kv = attn_num_memory_kv,
                **attn_pair_bias_kwargs
            )

            transition = Transition(
                dim = dim,
                expansion_factor = trans_expansion_factor
            )

            conditionable_pair_bias = ConditionWrapper(
                pair_bias_attn,
                dim = dim,
                dim_cond = dim_single_cond
            )

            conditionable_transition = ConditionWrapper(
                transition,
                dim = dim,
                dim_cond = dim_single_cond
            )

            layers.append(ModuleList([
                linear_attn,
                colt5_attn,
                conditionable_pair_bias,
                conditionable_transition
            ]))

        assert not (not serial and checkpoint), 'checkpointing can only be used for serial version of diffusion transformer'

        self.checkpoint = checkpoint
        self.checkpoint_segments = checkpoint_segments

        self.layers = layers

        self.serial = serial
        self.add_residual = add_residual

        self.has_registers = num_register_tokens > 0
        self.num_registers = num_register_tokens

        if self.has_registers:
            assert not exists(attn_window_size), 'register tokens disabled for windowed attention'

            self.registers = nn.Parameter(torch.zeros(num_register_tokens, dim))

    @typecheck
    def to_checkpointed_serial_layers(
        self,
        noised_repr: Float['b n d'],
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'] | Float['b nw w (w*2) dp'],
        mask: Bool['b n'] | None = None,
        windowed_mask: Bool['b nw w (w*2)'] | None = None
    ):

        inputs = (noised_repr, single_repr, pairwise_repr, mask, windowed_mask)

        wrapped_layers = []

        def efficient_attn_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                noised_repr, single_repr, pairwise_repr, mask, windowed_mask = inputs
                noised_repr = fn(noised_repr, mask = mask) + noised_repr
                return noised_repr, single_repr, pairwise_repr, mask, windowed_mask
            return inner

        def attn_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                noised_repr, single_repr, pairwise_repr, mask, windowed_mask = inputs
                noised_repr = fn(noised_repr, cond = single_repr, pairwise_repr = pairwise_repr, mask = mask, windowed_mask = windowed_mask) + noised_repr
                return noised_repr, single_repr, pairwise_repr, mask, windowed_mask
            return inner

        def transition_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                noised_repr, single_repr, pairwise_repr, mask, windowed_mask = inputs
                noised_repr = fn(noised_repr, cond = single_repr) + noised_repr
                return noised_repr, single_repr, pairwise_repr, mask, windowed_mask
            return inner

        for linear_attn, colt5_attn, attn, transition in self.layers:

            if exists(linear_attn):
                wrapped_layers.append(efficient_attn_wrapper(linear_attn))

            if exists(colt5_attn):
                wrapped_layers.append(efficient_attn_wrapper(colt5_attn))

            wrapped_layers.append(attn_wrapper(attn))
            wrapped_layers.append(transition_wrapper(transition))

        out = checkpoint_sequential(wrapped_layers, self.checkpoint_segments, inputs)

        noised_repr, *_ = out
        return noised_repr

    @typecheck
    def to_serial_layers(
        self,
        noised_repr: Float['b n d'],
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'] | Float['b nw w (w*2) dp'],
        mask: Bool['b n'] | None = None,
        windowed_mask: Bool['b nw w (w*2)'] | None = None
    ):

        for linear_attn, colt5_attn, attn, transition in self.layers:

            if exists(linear_attn):
                noised_repr = linear_attn(noised_repr, mask = mask) + noised_repr

            if exists(colt5_attn):
                noised_repr = colt5_attn(noised_repr, mask = mask) + noised_repr

            noised_repr = attn(
                noised_repr,
                cond = single_repr,
                pairwise_repr = pairwise_repr,
                mask = mask,
                windowed_mask = windowed_mask
            ) + noised_repr

            noised_repr = transition(
                noised_repr,
                cond = single_repr
            ) + noised_repr

        return noised_repr

    @typecheck
    def to_parallel_layers(
        self,
        noised_repr: Float['b n d'],
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'] | Float['b nw w (w*2) dp'],
        mask: Bool['b n'] | None = None,
        windowed_mask: Bool['b nw w (w*2)'] | None = None
    ):

        for linear_attn, colt5_attn, attn, transition in self.layers:

            if exists(linear_attn):
                noised_repr = linear_attn(noised_repr, mask = mask) + noised_repr

            if exists(colt5_attn):
                noised_repr = colt5_attn(noised_repr, mask = mask) + noised_repr

            attn_out = attn(
                noised_repr,
                cond = single_repr,
                pairwise_repr = pairwise_repr,
                mask = mask,
                windowed_mask = windowed_mask
            )

            ff_out = transition(
                noised_repr,
                cond = single_repr
            )

            # in the algorithm, they omitted the residual, but it could be an error
            # attn + ff + residual was used in GPT-J and PaLM, but later found to be unstable configuration, so it seems unlikely attn + ff would work
            # but in the case they figured out something we have not, you can use their exact formulation by setting `serial = False` and `add_residual = False`

            residual = noised_repr if self.add_residual else 0.

            noised_repr = ff_out + attn_out + residual

        return noised_repr

    @typecheck
    def forward(
        self,
        noised_repr: Float['b n d'],
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'] | Float['b nw w (w*2) dp'],
        mask: Bool['b n'] | None = None,
        windowed_mask: Bool['b nw w (w*2)'] | None = None
    ):
        w, serial = self.attn_window_size, self.serial
        has_windows = exists(w)

        # handle windowing

        pairwise_is_windowed = pairwise_repr.ndim == 5

        if has_windows and not pairwise_is_windowed:
            pairwise_repr = full_pairwise_repr_to_windowed(pairwise_repr, window_size = w)

        # register tokens

        if self.has_registers:
            num_registers = self.num_registers
            registers = repeat(self.registers, 'r d -> b r d', b = noised_repr.shape[0])
            noised_repr, registers_ps = pack((registers, noised_repr), 'b * d')

            single_repr = F.pad(single_repr, (0, 0, num_registers, 0), value = 0.)
            pairwise_repr = F.pad(pairwise_repr, (0, 0, num_registers, 0, num_registers, 0), value = 0.)

            if exists(mask):
                mask = F.pad(mask, (num_registers, 0), value = True)

        # main transformer

        if serial and should_checkpoint(self, (noised_repr, single_repr, pairwise_repr)):
            to_layers_fn = self.to_checkpointed_serial_layers
        elif serial:
            to_layers_fn = self.to_serial_layers
        else:
            to_layers_fn = self.to_parallel_layers

        noised_repr = to_layers_fn(
            noised_repr,
            single_repr = single_repr,
            pairwise_repr = pairwise_repr,
            mask = mask,
            windowed_mask = windowed_mask,
        )

        # splice out registers

        if self.has_registers:
            _, noised_repr = unpack(noised_repr, registers_ps, 'b * d')

        return noised_repr