class Gaussian:
    r"""A joint Gaussian distribution over named vector blocks.

    Each block ``b`` has a mean ``m_b``, a row ``F_b`` of a **shared factor**
    and an optional **independent term** ``D_b``:

    .. math::

        x_b = m_b + F_b\,\xi + e_b, \qquad
        \xi \sim \mathcal N(0, I_k), \qquad
        e_b \sim \mathcal N(0, D_b) \text{ independent of everything else,}

    so that

    .. math::

        \operatorname{cov}(x_a, x_b) = F_a F_b^\top \ (a \ne b), \qquad
        \operatorname{cov}(x_b, x_b) = F_b F_b^\top + D_b .

    ``F_b`` is an operator of shape ``(d_b, k)``; every block shares the latent
    width ``k``, which may be zero. ``D_b`` is a positive semi-definite
    operator of shape ``(d_b, d_b)``. A block may lack either: no factor row
    means the block is independent of the latent vector, no independent term
    means ``D_b = 0``.

    This one representation covers every Gaussian an ensemble method meets:

    ================================== ============================ ===================
    distribution                       shared factor                independent terms
    ================================== ============================ ===================
    a prior :math:`\mathcal N(m, C)`   none (:math:`k = 0`)         :math:`C`
    an ensemble's moment match [*]_    :math:`A^\top/\sqrt{J-1}`    none
    the same, plus known noise on y    :math:`A^\top/\sqrt{J-1}`    :math:`R` on ``y``
    a linear-Gaussian joint (u, y)     :math:`[L;\ AL]`, LL^T = C   :math:`R` on ``y``
    ================================== ============================ ===================

    .. [*] An :class:`EnsembleGaussian`, the subclass whose latent coordinates
       are an ensemble's particles.

    Parameters
    ----------
    means : Mapping[str, Array]
        One exactly 1-D ``(d_b,)`` array per block. Its keys are the block
        names, and its order is the block order.
    factors : Mapping[str, LinOp], optional
        Factor rows, each of shape ``(d_b, k)`` with one shared ``k``, keyed by
        a subset of the block names. Arrays are wrapped in
        :class:`~enskit.linalg.Dense`.
    block_covs : Mapping[str, PSDLinOp], optional
        Independent terms, each of shape ``(d_b, d_b)``, keyed by a subset of
        the block names.
    latent_dim : int, optional
        Keyword-only. The latent width ``k``; inferred from the factors when
        omitted. Stored as static metadata, so a marginal over blocks that have
        no factor row keeps the joint's ``k``.

    Raises
    ------
    ValueError
        If a mean is not 1-D, a factor or independent term disagrees with its
        block's dimension, the factors disagree on ``k``, or a key names no
        block.
    TypeError
        If a factor is not a :class:`~enskit.linalg.LinOp` or array, or an
        independent term is not a :class:`~enskit.linalg.PSDLinOp`.

    Notes
    -----
    Methods return a Gaussian of the same kind when they keep the latent space
    (so an :class:`EnsembleGaussian` stays one) and a plain :class:`Gaussian`
    when they change it (:meth:`absorb`, :meth:`compress`).

    The representation is a factor rather than three covariance blocks
    because a factor keeps every block's covariance and every cross-covariance
    consistent by construction, keeps sample covariances low rank, and gives
    conditioning a form that never squares the factor. The independent terms
    are kept apart because they are what carries known structure: a noise
    covariance used only through ``whiten``, or a prior with a Kronecker or
    diagonal form.
    """

    # -- construction ------------------------------------------------------------
    @classmethod
    def independent(cls, blocks: Mapping[str, tuple[Array, PSDLinOp]] | None = None, /,
                    **block_specs: tuple[Array, PSDLinOp]) -> Gaussian:
        r"""Mutually independent blocks, each given as ``(mean, cov)``.

        .. math::

            x_b \sim \mathcal N(m_b, C_b) \text{ independently for each } b .

        The result has no shared factor; each covariance becomes that block's
        independent term. Blocks may be given as a mapping or as keywords.

        Examples
        --------
        >>> prior = Gaussian.independent(u=(jnp.zeros(4), DensePSD(C)))
        """

    # -- introspection -------------------------------------------------------------
    @property
    def names(self) -> tuple[str, ...]:
        """The block names, in order."""

    @property
    def dims(self) -> dict[str, int]:
        """Each block's dimension, keyed by name, in order."""

    @property
    def latent_dim(self) -> int:
        """The shared factor's width ``k`` (static); unchanged by :meth:`marginal`."""

    def mean(self, name: str) -> Array:
        """Block ``name``'s mean, ``(d,)``."""

    def factor(self, name: str) -> LinOp | None:
        """Block ``name``'s row of the shared factor, ``(d, k)``, or ``None``."""

    def block_cov(self, name: str) -> PSDLinOp | None:
        """Block ``name``'s independent term, ``(d, d)``, or ``None``."""

    def cov(self, a: str, b: str | None = None) -> LinOp:
        """The covariance of block ``a`` (with ``b``, if given), as an operator.

        ``cov(a)`` is ``F_a F_a^T + D_a`` as a
        :class:`~enskit.linalg.LowRankUpdate` (or :class:`~enskit.linalg.PSDLowRank`,
        or ``D_a`` itself, when one part is absent); ``cov(a, b)`` is
        ``F_a F_b^T`` as a product operator. Nothing of size ``(d_a, d_b)`` is
        formed by this call.
        """

    # -- structure ---------------------------------------------------------------
    def marginal(self, *names: str) -> Gaussian:
        """The marginal over ``names``, in the order given.

        Dropping blocks from a Gaussian is exact and cheap: their means,
        factor rows and independent terms are discarded, and the latent space
        is unchanged.
        """

    def drop(self, *names: str) -> Gaussian:
        """The marginal over every block not in ``names``."""

    def rename(self, mapping: Mapping[str, str]) -> Gaussian:
        """Rename blocks; positions and alignment are kept."""

    def pipe(self, f: Callable[Concatenate[Gaussian, P], R], *args: P.args,
             **kwargs: P.kwargs) -> R:
        """Return ``f(self, *args, **kwargs)``; see :meth:`Ensemble.pipe`."""

    def add_noise(self, covs: Mapping[str, PSDLinOp] | None = None, /,
                  **block_covs: PSDLinOp) -> Gaussian:
        r"""Add independent Gaussian noise to the named blocks, in place.

        .. math::

            x_b \mapsto x_b + e_b, \qquad e_b \sim \mathcal N(0, R_b)
            \text{ independent of everything else,}

        for each named block :math:`b` with :math:`R_b` = ``covs[b]`` (given as
        a mapping or as keywords: ``g.add_noise(y=R)``). On a block with no independent term
        this sets ``D_b = covs[b]`` and changes nothing else, so alignment is
        preserved and the noise covariance is used only through ``whiten``
        afterwards. On a block that already has one, the existing term is first
        moved into the shared factor (see :meth:`absorb`).

        To keep the noise-free block as well, push it through
        :class:`~enskit.maps.AdditiveNoise` to a new block instead.
        """

    def absorb(self, *names: str) -> Gaussian:
        r"""Move the independent terms of ``names`` into the shared factor.

        For each named block with an independent term ``D_b``, appends the
        columns of ``D_b.factor()`` to the latent space: ``F_b`` becomes
        ``[F_b, L_b]`` and every other block's row gains zero columns
        (:class:`~enskit.linalg.Zero`, never materialized). The distribution is
        unchanged; its representation now lets a map of block ``b`` stay
        correlated with ``b``. The result is a plain :class:`Gaussian`.

        Raises
        ------
        UnsupportedOpError
            If an independent term has no cheap ``factor``.
        """

    def compress(self, *, max_dim: int = 4096) -> Gaussian:
        """Re-factor the shared factor to width at most the total dimension.

        Stacks every factor row into one ``(D, k)`` matrix ``F`` and, when
        ``k > D``, replaces it by ``R^T`` from the thin QR ``F^T = Q R``, so that
        ``F F^T = R^T R``. The distribution is unchanged. Use it when repeated :meth:`absorb` — every step of an exact
        Kalman filter, for instance — would otherwise grow the latent width
        without bound. The result is a plain :class:`Gaussian`.

        Raises
        ------
        ValueError
            If ``D > max_dim``, before allocating.
        """

    # -- conditioning ------------------------------------------------------------
    def condition(self, values: Mapping[str, Array] | None = None, /,
                  **block_values: Array) -> Gaussian:
        r"""The exact conditional distribution of the other blocks given ``values``.

        Two cases, chosen by the conditioned blocks' structure:

        **Every conditioned block has an independent term** (the usual case:
        noisy data). Write ``F_c`` and ``D_c`` for the conditioned blocks'
        stacked factor rows and block-diagonal independent term, ``W`` for a
        whitener of ``D_c``, and

        .. math::

            S = (W F_c)^\top \in \mathbb R^{k \times N}, \qquad
            S = U \Sigma V^\top \ \text{(thin SVD)}.

        For each other block ``x`` the conditional mean and factor are

        .. math::

            m_x' = m_x + F_x\, w, \quad
            w = U \operatorname{diag}\!\Big(\tfrac{\sigma_i}{1+\sigma_i^2}\Big)
                V^\top W (y - m_c), \qquad
            F_x' = F_x T, \quad T = (I_k + S S^\top)^{-1/2},

        and its independent term is unchanged. One SVD serves both. ``T`` is
        held in its thin form ``I + U((I + Sigma^2)^{-1/2} - I)U^T``, an operator
        costing ``O(k rank(S))`` to apply, so ``F_x T`` is a product operator and
        a structured ``F_x`` (a Kronecker prior factor, say) is never densified.

        **No conditioned block has an independent term** (conditioning on an
        exact value). With ``F_c^T = Q R`` (thin QR), ``w = Q R^{-T}(y - m_c)``
        and ``T = I - Q Q^T``. Requires ``F_c`` of full row rank ``N``, hence
        ``k >= N``, and ``n_particles > N`` for an :class:`EnsembleGaussian`
        (whose centered columns have rank at most ``n_particles - 1``); both size
        conditions are shape checks and always run.

        Parameters
        ----------
        values : Mapping[str, Array], optional
            Positional-only. The conditioned blocks and their values, each
            exactly ``(d_b,)``. A family of values is a :func:`jax.vmap` over
            this method.
        **block_values : Array
            The same, as keywords: ``g.condition(y=y_obs)``. Both forms may be
            combined; a block given twice raises ``TypeError``.

        Returns
        -------
        Gaussian
            Over the remaining blocks, in their order. Particle alignment is
            preserved: conditioning right-multiplies the factor by ``T``, and
            ``T 1 = 1`` when the factor's columns are centered.

        Raises
        ------
        ValueError
            If a value's shape is not its block's, some but not all conditioned
            blocks have independent terms, or (exact case) the size conditions
            fail. In debug mode, also if the exact case is rank deficient or a
            result is not finite.
        UnsupportedOpError
            If an independent term cannot ``whiten``.

        Notes
        -----
        The residual and the factor are whitened together, centered first:
        ``W [F_c | y - m_c]`` in one ``whiten_mat`` call. Whitening before
        centering loses accuracy in proportion to ``sqrt(cond(D_c))``. Neither
        ``S S^T`` nor ``S^T S`` is formed: forming either rounds away every
        singular value below ``sqrt(eps) * sigma_max``, which are the ones
        with the largest gain multipliers. The multiplier
        ``sigma / (1 + sigma**2)`` is at most 1/2, so the gain is bounded
        however collapsed the factor is. ``T`` is assembled with the identity
        completion ``I + U((I + Sigma^2)^{-1/2} - I)U^T``, exact when
        ``rank(S) < k``. Gradients use the custom rules in the
        differentiability contract, finite at repeated and zero singular
        values.
        """

    def conditional_map(self, given: str | Sequence[str]) -> MatheronMap:
        r"""The exact conditional of this Gaussian as a transport map.

        Returns, for the given blocks :math:`y` and the remaining blocks
        :math:`x`, the map

        .. math::

            T_{y^*}(x, y) = x + K\,(y^* - y), \qquad
            K = \operatorname{cov}(x, y)\,\operatorname{cov}(y, y)^{-1},

        built now from the block *names*; the value :math:`y^*` is supplied
        each time the map is called. Applied to joint samples of this
        Gaussian it returns exact samples of the conditional at ``y*``
        (Matheron's rule); applied to the particles of a projected ensemble it
        moves each particle toward the conditional. The gain is never formed; the
        map holds the whitened factor and applies the same
        :class:`~enskit.linalg.IdentityPlusGram` as :meth:`condition`.

        Parameters
        ----------
        given : str or sequence of str
            The names of the conditioned blocks. They must all have independent terms; there
            is no exact-value (noise-free) conditional map.
        """

    def log_density(self, values: Mapping[str, Array] | None = None, /,
                    **block_values: Array) -> Array:
        r"""Log density of the marginal over exactly the blocks in ``values``.

        With the conditioned-block notation of :meth:`condition` and
        ``b = W(y - m_c)``,

        .. math::

            \log p(y) = -\tfrac12\Big(\lVert b\rVert^2
              - (S b)^\top (I + S S^\top)^{-1} S b
              + \log\det D_c + \textstyle\sum_i \log(1+\sigma_i^2)
              + N \log 2\pi\Big),

        and the QR form when no block has an independent term. This is the
        evidence of data under a joint, and the objective for tuning
        hyperparameters (see the differentiability contract). With blocks
        ``a`` and ``b``, ``g.log_density(a=v)`` is the density of the ``a``
        marginal and ``g.log_density(a=v, b=w)`` that of the joint.

        Parameters
        ----------
        values : Mapping[str, Array], optional
            Positional-only. Each ``(*batch, d_b)`` with one shared batch shape.
        **block_values : Array
            The same, as keywords.

        Returns
        -------
        Array
            Shape ``batch``; a 0-d array when unbatched. Never a Python float.
        """

    # -- sampling --------------------------------------------------------------------
    def sample(self, key, n_particles: int) -> Ensemble:
        r"""Draw ``n_particles`` independent samples as an unweighted :class:`Ensemble`.

        .. math::

            x_j^{(b)} = m_b + F_b\,\xi_j + L_b\,\eta_j^{(b)}, \qquad
            \xi_j \sim \mathcal N(0, I_k),\ \ \eta_j^{(b)} \sim \mathcal N(0, I),

        with :math:`L_b` = ``D_b.factor()``.
        The draw consumes ``key`` whole: one normal array for the latent
        vector, then one per block with an independent term, in block order.
        """
