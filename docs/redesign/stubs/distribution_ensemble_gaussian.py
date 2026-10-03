class EnsembleGaussian(Gaussian):
    r"""A Gaussian whose latent coordinates are an ensemble's particles.

    The Gaussian of :class:`Gaussian`, with :math:`k = J` and

    .. math::

        F_b = \frac{1}{\sqrt{J-1}}\big[a_1^{(b)}, \dots, a_J^{(b)}\big],
        \qquad F_b \mathbf 1 = 0,

    so latent coordinate :math:`j` belongs to particle :math:`j`.
    :meth:`Ensemble.project` returns one for an unweighted ensemble. Two things
    are possible here and nowhere else, because they need that
    correspondence: reading the particles back out
    (:meth:`realize_particles`), and moving them as a set
    (:meth:`square_root_map`). A Gaussian built any other way (from a prior,
    by exact algebra, by a hybrid estimator) is a plain :class:`Gaussian` and
    has neither method.

    Every :class:`Gaussian` method that keeps the latent space returns an
    :class:`EnsembleGaussian`: :meth:`marginal`, :meth:`drop`,
    :meth:`rename`, :meth:`condition`, :meth:`add_noise` on a block with no
    independent term, and :func:`enskit.maps.pushforward` through a
    :class:`~enskit.maps.Linear` map of a block with no independent term.
    :meth:`absorb` and :meth:`compress` return a plain :class:`Gaussian`.

    Parameters
    ----------
    means, factors, block_covs
        As for :class:`Gaussian`.
    n_particles : int
        Keyword-only. :math:`J`; must equal the factors' width. The centering
        :math:`F_b\mathbf 1 = 0` is a value precondition, checked in debug mode
        only; :meth:`Ensemble.project` centers by construction and is the
        normal way to make one.
    """

    @property
    def n_particles(self) -> int:
        """:math:`J`, the number of particles (static)."""

    def realize_particles(self, *, key=None, exclude_block_covs: Sequence[str] = ()) -> Ensemble:
        r"""The particles this Gaussian's latent coordinates belong to.

        .. math::

            x_j^{(b)} = m_b + \sqrt{J-1}\,F_b e_j \;\big[\, + L_b \eta_j^{(b)} \big],
            \qquad j = 1, \dots, J,

        with :math:`e_j` the :math:`j`-th unit vector; the bracketed draw from
        :math:`D_b = L_b L_b^\top` is added for each block that has an
        independent term and is not in ``exclude_block_covs``. For the Gaussian returned by
        :meth:`Ensemble.project` this reproduces the ensemble's particles
        exactly; after :meth:`condition` it gives the conditioned particles, the
        deterministic square-root reading of the conditional.

        Parameters
        ----------
        key : jax.random key, optional
            Keyword-only. Required when a realized block has an independent
            term that is not excluded. Split once per such block, in order.
        exclude_block_covs : sequence of str
            Keyword-only. Blocks whose independent terms are left out, so that
            only their mean-plus-factor part is returned: for instance given
            blocks whose noise a :class:`MatheronMap` will draw
            itself.

        Raises
        ------
        ValueError
            If a key is required and missing.
        """

    def square_root_map(self, given: str | Sequence[str]) -> SquareRootMap:
        r"""The square-root update of these particles, as a map from given values.

        Built from the given block *names*; called with their values,
        ``smap(y=y_star)`` returns ``self.condition(y=y_star).realize_particles()``.
        Everything that does not depend on the value is computed here, once:
        :math:`S = (WF_y)^\top`, its :class:`~enskit.linalg.IdentityPlusGram`,
        and :math:`F_x T`.
        """


class SquareRootMap:
    r"""Moves the particle set an :class:`EnsembleGaussian` was built from, as a whole.

    Built by :meth:`EnsembleGaussian.square_root_map`. Called with
    :math:`y^*`, it returns, for each target block :math:`x`,

    .. math::

        x_j' = m_x + F_x\, (I + SS^\top)^{-1} S\, W(y^* - m_y)
               + \sqrt{J-1}\, F_x (I + SS^\top)^{-1/2} e_j ,
        \qquad j = 1, \dots, J .

    Unlike a :class:`MatheronMap`, which moves any samples of the joint one at
    a time, this map has no sample argument: it acts through each particle's
    latent coordinate :math:`e_j`, so it is defined only for the particles
    the Gaussian was built from. :math:`(I+SS^\top)^{-1/2}` does not depend on
    :math:`y^*` and is built once.
    """

    given: tuple[str, ...]
    targets: tuple[str, ...]

    def __call__(self, values: Mapping[str, Array] | None = None, /, *, key=None,
                 **block_values: Array) -> Ensemble:
        """The updated particles over the target blocks.

        ``key`` is needed only when a target block has an independent term,
        which is then sampled.
        """
