class Ensemble:
    r"""An empirical distribution over named blocks, optionally weighted.

    Holds :math:`J` *particles* :math:`x_1, \dots, x_J`. Block ``b`` is stored
    as a ``(J, d_b)`` array whose row ``j`` is particle ``j``'s value of that
    block; every block has the same particles in the same order. The
    distribution is

    .. math::

        \hat p = \sum_{j=1}^{J} w_j\, \delta_{x_j}, \qquad
        w_j = \frac{\exp(\ell_j)}{\sum_{i=1}^{J} \exp(\ell_i)},

    with log weights :math:`\ell_j`, or :math:`w_j = 1/J` when unweighted.

    Parameters
    ----------
    blocks : Mapping[str, Array], optional
        Positional-only. One exactly 2-D ``(J, d_b)`` array per block, all with
        the same ``J >= 2`` and the same real floating dtype. The mapping's
        order is the block order.
    log_weights : Array, optional
        Keyword-only. ``(J,)`` unnormalized log weights :math:`\ell`. ``None``
        means equal weights, and differs from an array of equal values only in
        that :attr:`is_weighted` reports it.
    **block_arrays : Array
        Blocks given as keywords, appended after those in ``blocks`` in the
        order written. A block named ``log_weights`` must be given in
        ``blocks``.

    Raises
    ------
    ValueError
        If a block is not 2-D, the blocks disagree on ``J``, there are no
        blocks, or ``J < 2``.
    TypeError
        If a block is not a real floating array, the blocks' dtypes differ, or
        a block is given twice.

    Examples
    --------
    >>> ens = Ensemble(x=x, theta=theta)           # (J, 40), (J, 1)
    >>> ens["x"].shape
    (J, 40)
    >>> ens.marginal("theta")
    Ensemble(n_particles=J, blocks={'theta': 1}, weighted=False)
    """

    # -- introspection -------------------------------------------------------
    @property
    def names(self) -> tuple[str, ...]:
        """The block names, in order."""

    @property
    def dims(self) -> dict[str, int]:
        """Each block's dimension, keyed by name, in order."""

    @property
    def n_particles(self) -> int:
        """The number of particles :math:`J` (rows of every block)."""

    @property
    def is_weighted(self) -> bool:
        """Whether this ensemble carries ``log_weights``."""

    @property
    def log_weights(self) -> Array | None:
        """The stored unnormalized log weights, or ``None``."""

    @property
    def weights(self) -> Array:
        r"""Normalized weights :math:`w`, ``(J,)``; :math:`1/J` each when unweighted.

        Computed as a max-shifted softmax of the log weights, so log weights far
        outside the floating-point range normalize correctly.
        """

    @property
    def all_finite(self) -> Array:
        """``(J,)`` boolean: particle ``j`` is finite in every block.

        Non-finite rows are how the layers above mark failed particles; this is
        how such particles are found.
        """

    def __getitem__(self, name: str) -> Array:
        """Block ``name`` as a ``(J, d)`` array.

        Raises
        ------
        KeyError
            If there is no such block; the message lists the blocks.
        """

    # -- structure -------------------------------------------------------------
    def marginal(self, *names: str) -> Ensemble:
        """The ensemble restricted to ``names``, in the order given.

        Weights are kept. Marginalizing an empirical distribution is dropping
        coordinates, so the particles are unchanged.
        """

    def drop(self, *names: str) -> Ensemble:
        """The ensemble without ``names``; the complement of :meth:`marginal`."""

    def assign(self, blocks: Mapping[str, Array] | None = None, /,
               **block_arrays: Array) -> Ensemble:
        """Replace existing blocks or append new ones; weights are kept.

        Blocks may be given as a mapping, as keywords, or both
        (``ens.assign(x_prev=ens["x"])``). Each array must be ``(J, d)``. A
        replaced block keeps its position; new blocks are appended in the order
        given.
        """

    def rename(self, mapping: Mapping[str, str] | None = None, /,
               **new_names: str) -> Ensemble:
        """Rename blocks (``old=new``); positions are kept.

        Raises
        ------
        ValueError
            If a new name collides with an existing block that is not itself
            being renamed.
        """

    def pipe(self, f: Callable[Concatenate[Ensemble, P], R], *args: P.args,
             **kwargs: P.kwargs) -> R:
        """Return ``f(self, *args, **kwargs)``; see the note on ``pipe`` in §4.4."""

    # -- moments ----------------------------------------------------------------
    def mean(self, name: str) -> Array:
        r"""The (weighted) mean of block ``name``, :math:`\bar x = \sum_j w_j x_j`, ``(d,)``."""

    def anomalies(self, name: str) -> Array:
        r"""Raw deviations from the mean, :math:`a_j = x_j - \bar x`, as a ``(J, d)`` array.

        Computed by subtracting the first particle before the mean, so particles
        that are identical give exactly zero anomalies however large their
        magnitude.
        """

    def cov(self, a: str, b: str | None = None) -> LinOp:
        r"""The sample covariance of block ``a`` (with ``b``, if given), as an operator.

        .. math::

            \hat C_{ab} = \frac{1}{1 - \sum_j w_j^2}
                \sum_{j=1}^{J} w_j\, a_j^{(a)} \big(a_j^{(b)}\big)^\top,

        which with :math:`w_j = 1/J` is the usual :math:`\frac{1}{J-1}\sum_j a_j a_j^\top`. Never forms a
        ``(d_a, d_b)`` matrix: the result applies the anomalies.

        Returns
        -------
        LinOp
            :class:`~enskit.linalg.PSDLowRank` for ``cov(a)``; for ``cov(a, b)``
            with ``a != b``, a product operator of shape ``(d_a, d_b)``.
        """

    def project(self) -> Gaussian | EnsembleGaussian:
        r"""The moment-matching Gaussian: the Gaussian with this ensemble's mean and covariance.

        For an unweighted ensemble the result is an :class:`EnsembleGaussian`
        with :math:`J` particles and no independent terms:

        .. math::

            m_b = \bar x^{(b)}, \qquad
            F_b = \frac{1}{\sqrt{J-1}} \big[a_1^{(b)}, \dots, a_J^{(b)}\big]
            \in \mathbb R^{d_b \times J},

        so its latent coordinate :math:`j` belongs to particle :math:`j`, which
        is what lets :meth:`EnsembleGaussian.realize_particles` return these particles
        exactly.

        For a weighted ensemble,

        .. math::

            m_b = \sum_j w_j x_j^{(b)}, \qquad
            (F_b)_{:,j} = \frac{\sqrt{w_j}\,\big(x_j^{(b)} - m_b\big)}{\sqrt{1 - \sum_i w_i^2}},

        with :math:`1 - \sum_i w_i^2` computed as
        :math:`-\operatorname{expm1}\big(\operatorname{lse}(2\ell) - 2\operatorname{lse}(\ell)\big)`;
        the result is a plain :class:`Gaussian`.

        Returns
        -------
        EnsembleGaussian or Gaussian
            Over the same blocks, in the same order; an
            :class:`EnsembleGaussian` exactly when the ensemble is unweighted.

        Raises
        ------
        ValueError
            If the weights are concentrated on one particle to working precision
            (:math:`1 - \sum_i w_i^2` rounds to zero), where the weighted
            covariance is undefined. Checked eagerly; under a trace the result is
            ``nan``.

        Notes
        -----
        Projection is the one place a Gaussian approximation enters an
        ensemble rule, and it is always written at the call site (or inside
        :func:`enskit.kalman.gaussian_approximation`, which says so) rather than hidden inside a
        conditioning routine.
        """
