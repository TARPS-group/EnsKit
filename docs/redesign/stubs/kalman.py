r"""Ensemble Kalman updates: approximate conditioning of an ensemble's particles.

An ensemble Kalman update takes the particles of an ensemble over target and
given blocks, and a value :math:`y^*` for the given blocks, and returns
particles approximating the conditional of the targets at :math:`y^*`. Every
update has the same three stages:

1. **Approximate.** Fit a joint Gaussian to the particles,
   :func:`gaussian_approximation` by default.
2. **Build.** An :class:`UpdateRule` builds a :class:`ParticleUpdate` from the
   particles, the approximation and the *names* of the given blocks. Nothing
   here depends on the given values.
3. **Call.** The particle update is called with the given *values* and returns
   the updated particles.

:func:`update` runs all three in one call.

================================ ================================================
object                           is
================================ ================================================
:func:`update`                   one ensemble Kalman update, one call
:func:`gaussian_approximation`   the default joint Gaussian approximation
:class:`UpdateRule`              protocol: builds a particle update
:class:`ParticleUpdate`          protocol: given values in, updated particles out
:class:`SymmetricSquareRoot`     the deterministic rule (ETKF form)
:class:`Matheron`                the stochastic rule (Matheron's rule)
:class:`LocalizedUpdateRule`               domain localization around either rule
:class:`DomainLocalization`      locations, taper and neighborhood size
:func:`gaspari_cohn`             the compactly supported taper
:func:`inflate_multiplicative`   multiplicative inflation: scale the anomalies
:func:`inflate_additive`         additive inflation: add centered Gaussian draws
:func:`relax_to_prior_spread`    RTPS: relax posterior spread toward the prior's
:func:`relax_to_prior_perturbations` RTPP: blend posterior and prior anomalies
================================ ================================================

Conventions: the *given* blocks are the ones conditioned on; every other
block of the approximation is a *target* and is updated. The layer knows
nothing about where the given values came from, which is why it has no
notion of observations, steps or time. Particles are unweighted.

Notes
-----
Two kinds of map sit underneath, and the distinction matters. A
:class:`~enskit.distribution.MatheronMap` is *pointwise*: it moves any
samples of the joint, one at a time. A
:class:`~enskit.distribution.SquareRootMap` moves *the particle set it was
built from*, as a whole, because it acts through each particle's own latent
coordinate. A particle update is always bound to particular particles; the
:class:`Matheron` rule obtains one by applying a pointwise map to them, and
:class:`SymmetricSquareRoot` by taking the square-root map itself.
"""


class ParticleUpdate(Protocol):
    """A built update: called with the given values, returns the updated particles.

    Bound to the particles, approximation and given block names it was built
    from. Calling it again with other values reuses everything that does not
    depend on the values (factorizations, gains, transforms).
    """

    def __call__(self, values: Mapping[str, Array] | None = None, /, *, key=None,
                 **block_values: Array) -> Ensemble:
        """Return the updated particles over the target blocks.

        Parameters
        ----------
        values : Mapping[str, Array], optional
            Positional-only. :math:`y^*` for every given block, each exactly
            ``(d_b,)``.
        key : jax.random key, optional
            Keyword-only. Required by stochastic rules; ignored otherwise.
        **block_values : Array
            The values as keywords.

        Returns
        -------
        Ensemble
            Unweighted, with the bound particles' count, over the target blocks
            in the approximation's order.
        """


class UpdateRule(Protocol):
    """Builds a :class:`ParticleUpdate`. Writing one is how a new update is added.

    :func:`enskit.testing.check_update_rule` checks a user's rule.

    Contract between caller and rule: ``approximation`` is a Gaussian over the
    particles' target and given blocks, whose given blocks carry their known
    noise as independent terms. **If** it is an
    :class:`~enskit.distribution.EnsembleGaussian` with the particles' count,
    it is aligned with *these* particles, as :func:`gaussian_approximation`
    guarantees, and a rule may rely on that for a faster path. A modified
    approximation (a hybrid covariance, say) is generally a plain
    :class:`~enskit.distribution.Gaussian`, and rules that need alignment
    raise.
    """

    def build(self, particles: Ensemble, approximation: Gaussian,
              given: tuple[str, ...]) -> ParticleUpdate:
        """Build the update for ``particles`` conditioned on the blocks named ``given``."""


class SymmetricSquareRoot:
    r"""The deterministic square-root rule (ETKF form): realize after condition.

    Builds ``approximation.square_root_map(given)``. Calling it with
    :math:`y^*` returns, for each target block :math:`x`,

    .. math::

        x_j' = m_x + F_x w + \sqrt{J-1}\, F_x T e_j, \qquad
        w = (I + SS^\top)^{-1} S\, W(y^* - m_y), \qquad
        T = (I + SS^\top)^{-1/2},

    with :math:`S = (WF_y)^\top` and :math:`W` a whitener of the given blocks'
    noise. :math:`T` does not depend on :math:`y^*` and is built once.

    Exact for a linear-Gaussian problem: the returned particles' sample mean
    and covariance equal the exact conditional's whenever the input particles'
    moments equal the prior's. Needs a key only if a target block has an
    independent term, which is then sampled. "Symmetric" names the choice of
    square root, :math:`T = T^\top`; other members of the family
    (left-multiplied adjustment, random rotations) would be separate rules.

    Raises
    ------
    ValueError
        At build, if the approximation is not an
        :class:`~enskit.distribution.EnsembleGaussian` aligned with the
        particles.

    Notes
    -----
    Exact moments do not mean the right shape. On nonlinear problems the
    square-root update keeps the particles' arrangement, rotated and scaled,
    and can leave a few particles carrying most of the spread; the stochastic
    rule does not. Which to use is a modeling choice, so :func:`update` has
    no default.

    References
    ----------
    .. [1] Bishop, C. H., Etherton, B. J. & Majumdar, S. J. (2001). Adaptive
       sampling with the ensemble transform Kalman filter. Part I:
       Theoretical aspects. Monthly Weather Review, 129(3), 420–436.
    .. [2] Tippett, M. K., Anderson, J. L., Bishop, C. H., Hamill, T. M. &
       Whitaker, J. S. (2003). Ensemble square root filters. Monthly Weather
       Review, 131(7), 1485–1490.
    .. [3] Hunt, B. R., Kostelich, E. J. & Szunyogh, I. (2007). Efficient
       data assimilation for spatiotemporal chaos: a local ensemble transform
       Kalman filter. Physica D, 230(1–2), 112–126.
    .. [4] Wang, X., Bishop, C. H. & Julier, S. J. (2004). Which is better, an
       ensemble of positive-negative pairs or a centered spherical simplex
       ensemble? Monthly Weather Review, 132(7), 1590-1605.
    """


class Matheron:
    r"""The stochastic rule (perturbed values, Matheron's rule): transport the particles.

    Builds the approximation's pointwise
    :class:`~enskit.distribution.MatheronMap` and applies it to the particles.
    Calling the update with :math:`y^*` moves each particle by

    .. math::

        x_j' = x_j + e_{x,j} + K\big(y^* - g_j - e_j\big),
        \qquad K = \operatorname{cov}(x, y)\operatorname{cov}(y, y)^{-1},

    where :math:`g_j` is the particle's noise-free given value, :math:`e_j` the
    given blocks' noise and :math:`e_{x,j}` a draw from any target block's
    independent term (zero when it has none). ``key`` is split into
    ``(targets, noise)``; :math:`e_j` is drawn in whitened coordinates, one
    ``(J, N)`` standard normal array, so noise covariances need only
    ``whiten``. Works with any approximation:

    - **aligned** (the default approximation): the whitened residuals come
      from the approximation's own factor,
      :math:`W(y^* - g_j) = W(y^* - \bar g) - \sqrt{J-1}\, S_{j\cdot}`, for
      :math:`J + 1` whitenings in all;
    - **any other** (a hybrid covariance, say): through the map, whitening each
      particle's residual, :math:`2J` whitenings.

    For a linear-Gaussian problem the result is, over the noise draws, an
    unbiased sample of the conditional of the *fitted* Gaussian; its spread
    carries sampling error of order :math:`1/\sqrt{J}`.

    Raises
    ------
    ValueError
        At call, if no key is given; at build, if a given block has no
        independent term (the perturbations are that noise).

    References
    ----------
    .. [1] Burgers, G., van Leeuwen, P. J. & Evensen, G. (1998). Analysis
       scheme in the ensemble Kalman filter. Monthly Weather Review, 126(6),
       1719–1724.
    .. [2] Houtekamer, P. L. & Mitchell, H. L. (1998). Data assimilation
       using an ensemble Kalman filter technique. Monthly Weather Review,
       126(3), 796–811.
    .. [3] Wilson, J. T., Borovitskiy, V., Terenin, A., Mostowsky, P. &
       Deisenroth, M. P. (2021). Pathwise conditioning of Gaussian processes.
       Journal of Machine Learning Research, 22(105), 1–47.
    """


def gaussian_approximation(ensemble: Ensemble,
                           noise: Mapping[str, PSDLinOp] | None = None) -> EnsembleGaussian:
    r"""The joint Gaussian approximation an update conditions: moment match plus known noise.

    .. math::

        \hat p(x, y) = \mathcal N\big(\bar z,\ \hat C + \operatorname{blockdiag}(0, R)\big),

    the Gaussian with the particles' sample mean :math:`\bar z` and sample
    covariance :math:`\hat C` (:meth:`Ensemble.project`), with the known noise
    :math:`R_b` = ``noise[b]`` added to each named block *as a covariance*,
    never as samples (the lower-variance estimate; see Example 6). The result
    is an :class:`~enskit.distribution.EnsembleGaussian` aligned with the
    particles. It is the default of every ``approximation=`` hook, and the
    starting point for a modified approximation.
    """


def update(ensemble: Ensemble, given: Mapping[str, Array] | None = None, /, *,
           update_rule: UpdateRule, noise: Mapping[str, PSDLinOp] | None = None,
           approximation: Callable[[Ensemble, Mapping[str, PSDLinOp]], Gaussian] | None = None,
           key=None, **given_values: Array) -> Ensemble:
    r"""One ensemble Kalman update: approximate, build, call.

    ``update_rule.build(ensemble, approximation(ensemble, noise), names)(values,
    key=key)``. Returns particles approximating the conditional of every
    non-given block of ``ensemble`` at the given values. With ``noise``, each
    given value :math:`y^*_b` is understood as a realization of
    :math:`x_b + e_b`, :math:`e_b \sim \mathcal N(0, R_b)`, while
    ``ensemble[b]`` holds the noise-free :math:`x_b`.

    Parameters
    ----------
    ensemble : Ensemble
        Unweighted, finite. Contains every given block.
    given : Mapping[str, Array], optional
        Positional-only. Each value exactly ``(d_b,)``.
    update_rule : UpdateRule
        Keyword-only and required: :class:`SymmetricSquareRoot`,
        :class:`Matheron`, :class:`LocalizedUpdateRule`, or a user rule.
    noise : Mapping[str, PSDLinOp], optional
        Keyword-only. Known noise covariances, used only through ``whiten``.
        Without it the given blocks are conditioned on exactly; only
        :class:`SymmetricSquareRoot` supports that, and it needs more particles
        than the given blocks' total dimension.
    approximation : callable, optional
        Keyword-only. ``(ensemble, noise) -> Gaussian``, building the joint
        approximation the rule conditions; :func:`gaussian_approximation` by
        default. The extension point for other covariance estimators (hybrid,
        shrinkage, multilevel).
    key : jax.random key, optional
        Keyword-only. Passed to the built update.
    **given_values : Array
        The values as keywords: ``update(ens, g=y, update_rule=..., noise=...)``. A
        block whose name is one of this function's own parameters must go in
        ``given``.

    Returns
    -------
    Ensemble
        Over the non-given blocks, in their order.

    Raises
    ------
    ValueError
        If the ensemble is weighted (resample first, with
        :func:`enskit.distribution.resample`), a given block is missing, or a
        value has the wrong shape. In debug mode, also if a particle is not
        finite.

    Examples
    --------
    >>> ens = pushforward(ens, forward, inputs="u", output="g")
    >>> post = update(ens, g=y, noise={"g": R}, update_rule=Matheron(), key=key)

    The same, in three stages, reusing the build for several values:

    >>> approx = gaussian_approximation(ens, {"g": R})
    >>> step = Matheron().build(ens, approx, ("g",))
    >>> post_a, post_b = step(g=y_a, key=k1), step(g=y_b, key=k2)
    """


class DomainLocalization:
    r"""The geometry of domain localization: where coordinates live and which data each sees.

    Domain localization replaces one global update by many local ones. Each
    target coordinate :math:`p` is updated using only nearby given
    coordinates, with the influence of each decaying with distance. This
    object holds what that requires:

    - a location :math:`c_p \in \mathbb R^c` for every coordinate of every
      located target block, and :math:`c_i` for every coordinate of the given
      blocks;
    - a distance :math:`d(c_p, c_i)` and a taper :math:`\rho`;
    - a radius :math:`L` and a neighborhood size :math:`K`.

    The local update for coordinate :math:`p` uses the :math:`K` given
    coordinates nearest :math:`c_p`, :math:`\mathcal N_p`, with weights

    .. math::

        \rho_{pi} = \rho\big(d(c_p, c_i) / L\big), \qquad i \in \mathcal N_p,

    applied by scaling the whitened given rows by :math:`\sqrt{\rho_{pi}}`. For
    diagonal noise this is the noise variance :math:`r_i` inflated to
    :math:`r_i / \rho_{pi}`, so a coordinate at distance :math:`\ge L` (where
    :math:`\rho = 0`) has no influence. :class:`LocalizedUpdateRule` consumes it.

    Parameters
    ----------
    target_coords : Mapping[str, Array | None]
        For each target block, a ``(d_b, c)`` array of locations, one row per
        coordinate. ``None`` (or omitting a block) marks a block without a
        location, such as a global parameter: it gets the global, untapered
        update.
    given_coords : Mapping[str, Array] or Array
        ``(d_b, c)`` locations of the given blocks' coordinates. A bare array
        is accepted when exactly one block is given, which spares callers of a
        driver from naming the driver's internal block.
    radius : float
        Keyword-only. :math:`L`; the taper's support ends at distance
        :math:`L`.
    max_neighbors : int
        Keyword-only. :math:`K`. Every local problem has the same shape
        ``(J, K)``, so the local updates vectorize; neighbors beyond the radius
        get weight zero.
    taper : callable
        Keyword-only. :math:`\rho`, on :math:`r = d/L`; default
        :func:`gaspari_cohn`.
    distance : callable
        Keyword-only. ``(point (c,), points (m, c)) -> (m,)``; Euclidean by
        default. Pass a periodic distance for a periodic domain.

    References
    ----------
    .. [1] Ott, E., Hunt, B. R., Szunyogh, I., Zimin, A. V., Kostelich, E.
       J., Corazza, M., Kalnay, E., Patil, D. J. & Yorke, J. A. (2004). A
       local ensemble Kalman filter for atmospheric data assimilation. Tellus
       A, 56(5), 415–428.
    .. [2] Hunt, B. R., Kostelich, E. J. & Szunyogh, I. (2007). Efficient
       data assimilation for spatiotemporal chaos: a local ensemble transform
       Kalman filter. Physica D, 230(1–2), 112–126.
    """


class LocalizedUpdateRule:
    """Domain localization around :class:`SymmetricSquareRoot` or :class:`Matheron`.

    An :class:`UpdateRule`. The built update performs, for every located
    target coordinate, the wrapped rule's update on the local problem
    described by :class:`DomainLocalization`: one ``(J, K)`` call to
    :class:`~enskit.linalg.IdentityPlusGram`, vectorized over coordinates
    with :func:`jax.vmap`. Unlocated blocks get the global update. The local
    gains and transforms do not depend on the given values and are built
    once. The stochastic variant draws its whitened perturbations once, and
    every local and global update uses that one draw.

    Parameters
    ----------
    update_rule : SymmetricSquareRoot or Matheron
    localization : DomainLocalization

    Raises
    ------
    TypeError
        If ``update_rule`` is not one of the two, or a given block's noise is not
        row-local: a :class:`~enskit.linalg.PSDDiagonal`, possibly scaled (as
        the tempered noise ``R / delta`` of an EKI step is).
    ValueError
        At build, if the approximation is not aligned with the particles, or a
        target block has an independent term.

    Notes
    -----
    Domain rather than covariance localization: tapering the sample
    covariance entrywise destroys its low rank, and with it the whitened-SVD
    conditioning. Zero-weight neighbors give exactly zero singular values,
    which is why :class:`~enskit.linalg.IdentityPlusGram` needs custom
    derivatives.

    References
    ----------
    .. [1] Hunt, B. R., Kostelich, E. J. & Szunyogh, I. (2007). Efficient
       data assimilation for spatiotemporal chaos: a local ensemble transform
       Kalman filter. Physica D, 230(1–2), 112–126.
    .. [2] Sakov, P. & Bertino, L. (2011). Relation between two common
       localisation methods for the EnKF. Computational Geosciences, 15(2),
       225–237.
    """


def gaspari_cohn(r: Array) -> Array:
    r"""The Gaspari-Cohn fifth-order taper on :math:`r` = distance / radius.

    With :math:`z = 2r`,

    .. math::

        \rho(z) = \begin{cases}
          -\tfrac14 z^5 + \tfrac12 z^4 + \tfrac58 z^3 - \tfrac53 z^2 + 1, & 0 \le z \le 1,\\[2pt]
          \tfrac1{12} z^5 - \tfrac12 z^4 + \tfrac58 z^3 + \tfrac53 z^2 - 5z + 4 - \tfrac{2}{3z}, & 1 < z \le 2,\\[2pt]
          0, & z > 2 .
        \end{cases}

    Equal to 1 at :math:`r = 0`, positive definite as a correlation function,
    and exactly 0 for :math:`r \ge 1`.

    References
    ----------
    .. [1] Gaspari, G. & Cohn, S. E. (1999). Construction of correlation
       functions in two and three dimensions. Quarterly Journal of the Royal
       Meteorological Society, 125(554), 723–757.
    """


def inflate_multiplicative(ensemble: Ensemble, anomaly_scale,
                           names: Sequence[str] | None = None) -> Ensemble:
    r"""Multiplicative inflation of each named block's anomalies.

    .. math::

        x_j \mapsto \bar x + \lambda\,(x_j - \bar x), \qquad \hat C \mapsto \lambda^2 \hat C,

    with :math:`\lambda` = ``anomaly_scale``. ``names`` restricts it to some
    blocks; all by default.

    References
    ----------
    .. [1] Anderson, J. L. & Anderson, S. L. (1999). A Monte Carlo
       implementation of the nonlinear filtering problem to produce ensemble
       assimilations and forecasts. Monthly Weather Review, 127(12),
       2741–2758.
    """


def inflate_additive(key, ensemble: Ensemble, covs: Mapping[str, PSDLinOp]) -> Ensemble:
    r"""Additive inflation with centered Gaussian draws.

    .. math::

        x_j^{(b)} \mapsto x_j^{(b)} + \varepsilon_j - \bar\varepsilon, \qquad
        \varepsilon_j \sim \mathcal N(0, Q_b),

    for each block :math:`b` with :math:`Q_b` = ``covs[b]``. The mean is
    unchanged exactly and the covariance grows by :math:`Q_b` in expectation.
    Unlike an update, which moves particles within the span of their
    anomalies, this adds variance in new directions. Needs ``factor`` of each
    covariance.
    """


def relax_to_prior_spread(prior: Ensemble, posterior: Ensemble, alpha,
                          names: Sequence[str] | None = None) -> Ensemble:
    r"""RTPS: relax the posterior spread toward the prior's, coordinate by coordinate.

    .. math::

        a_j^{\text{post}} \mapsto a_j^{\text{post}} \Big(1 + \alpha\,
          \frac{s^{\text{prior}} - s^{\text{post}}}{s^{\text{post}}}\Big),

    with :math:`a` the anomalies and :math:`s` the per-coordinate sample
    standard deviations. ``prior`` is the ensemble the update started from;
    both must have the same particles in the same order.

    References
    ----------
    .. [1] Whitaker, J. S. & Hamill, T. M. (2012). Evaluating methods to
       account for system errors in ensemble data assimilation. Monthly
       Weather Review, 140(9), 3078–3089.
    """


def relax_to_prior_perturbations(prior: Ensemble, posterior: Ensemble, alpha,
                                 names: Sequence[str] | None = None) -> Ensemble:
    r"""RTPP: blend posterior and prior anomalies, particle by particle.

    .. math::

        a_j^{\text{post}} \mapsto (1 - \alpha)\, a_j^{\text{post}} + \alpha\, a_j^{\text{prior}} .

    References
    ----------
    .. [1] Zhang, F., Snyder, C. & Sun, J. (2004). Impacts of initial
       estimate and observation availability on convective-scale data
       assimilation with an ensemble Kalman filter. Monthly Weather Review,
       132(5), 1238–1253.
    """
