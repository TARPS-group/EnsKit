class ConditionalMap(Protocol):
    """A map carrying samples of a joint distribution to samples of a conditional.

    Built from a joint and the *names* of the given blocks; called with samples
    of the joint (an :class:`Ensemble`, whose particles are such samples) and a
    value :math:`y^*` for the given blocks. Returns samples whose distribution
    approximates (or, for a Gaussian joint, equals) the conditional of the
    targets at ``y*``. :class:`MatheronMap` is the affine instance;
    a nonlinear triangular transport map is the same protocol.

    Attributes
    ----------
    given : tuple[str, ...]
        The blocks conditioned on.
    targets : tuple[str, ...]
        The blocks the map moves.
    """

    given: tuple[str, ...]
    targets: tuple[str, ...]

    def __call__(self, samples: Ensemble, values: Mapping[str, Array] | None = None, /, *,
                 key=None, **block_values: Array) -> Ensemble:
        """Transport ``samples`` of the joint to samples of the conditional at ``values``.

        Values may be given as a mapping or as keywords (``cmap(ens, y=y_obs)``).
        Returns the ensemble with every target block replaced, the given blocks
        dropped, and any other block passed through unchanged. Weights are
        kept.
        """


class MatheronMap:
    r"""The exact conditional of a :class:`Gaussian` as a transport map.

    .. math::

        T_{y^*}(x, y) = x + K\,(y^* - y)
          = x + F_x\,(I + S S^\top)^{-1} S\, W (y^* - y),

    with :math:`S = (W F_y)^\top` and :math:`W` a whitener of the given blocks'
    independent terms.


    Built by :meth:`Gaussian.conditional_map`; not constructed directly. Holds
    the whitened factor ``S`` of the given blocks, their whiteners and the
    targets' factor rows. Applying it to ``n`` particles whitens their ``n``
    residuals and calls :meth:`~enskit.linalg.IdentityPlusGram.solve_factor`
    once on the batch; ``K`` is never
    formed. (When the samples are the particles of the EnsembleGaussian it came from, the
    residuals can be read off ``S`` instead; :class:`enskit.kalman.Matheron`
    does so.)

    Notes
    -----
    Matheron's rule: if ``(x, y)`` is a sample of the Gaussian, then
    ``x + K (y* - y)`` is a sample of its conditional at ``y*``. Applied to
    ensemble particles, ``y`` must carry the given blocks' independent noise for
    the result to have the right spread. When the particles hold only the
    noise-free part, pass ``key`` and the map draws that noise itself, in
    whitened coordinates: ``W(y* - (g + e)) = W(y* - g) - eps`` with
    ``eps ~ N(0, I)``. This needs only ``whiten`` of each independent term,
    never its ``factor``.

    References
    ----------
    .. [1] Journel, A. G. & Huijbregts, C. J. (1978). Mining Geostatistics.
       Academic Press.
    .. [2] Wilson, J. T., Borovitskiy, V., Terenin, A., Mostowsky, P. &
       Deisenroth, M. P. (2021). Pathwise conditioning of Gaussian processes.
       Journal of Machine Learning Research, 22(105), 1–47.
    """

    given: tuple[str, ...]
    targets: tuple[str, ...]

    def __call__(self, samples: Ensemble, values: Mapping[str, Array] | None = None, /, *,
                 key=None, **block_values: Array) -> Ensemble:
        """Apply the map to every sample.

        Parameters
        ----------
        samples : Ensemble
            Must contain every given block and every target block present in
            the Gaussian; other blocks pass through.
        values : Mapping[str, Array], optional
            Positional-only. :math:`y^*` for each given block, exactly ``(d_b,)``.
        **block_values : Array
            The same, as keywords.
        key : jax.random key, optional
            Keyword-only. When given, the given blocks' independent noise is
            drawn (whitened, one ``(n_particles, N)`` normal draw) and subtracted
            from the whitened residuals. Omit it when ``particles`` already carry
            the noise.
        """

    def coefficients(self, samples: Ensemble, values: Mapping[str, Array] | None = None, /, *,
                     key=None, **block_values: Array) -> Array:
        r"""The latent coefficients :math:`w_j` with :math:`K(y^* - y_j) = F_x w_j`, ``(n, k)``.

        The escape hatch for code that works in latent space, for example to
        apply the map to some blocks with a modified factor.
        """


def reweight(ensemble: Ensemble, log_weight_increments: Array) -> Ensemble:
    r"""Multiply each particle's weight by :math:`\exp(\Delta\ell_j)`.

    .. math::

        \ell_j \mapsto \ell_j + \Delta\ell_j,

    with :math:`\Delta\ell` = ``log_weight_increments`` (the incremental log
    weights of importance sampling and SMC).

    Adds to the log weights (treating an unweighted ensemble as all zeros), so
    importance weights, likelihoods and tempering factors compose by repeated
    calls without leaving log space.

    Parameters
    ----------
    log_weight_increments : Array
        ``(J,)``. Non-finite entries give that particle weight zero
        (``-inf``) or raise (``nan``, ``+inf``).
    """


def effective_sample_size(ensemble: Ensemble) -> Array:
    r"""The effective sample size of the weights, as a 0-d array.

    .. math::

        \mathrm{ESS} = \frac{1}{\sum_j w_j^2}
          = \exp\!\big(2\operatorname{lse}(\ell) - \operatorname{lse}(2\ell)\big),

    computed in the second form, which stays finite when the weights span
    hundreds of orders of magnitude. Equals :math:`J` when unweighted.

    References
    ----------
    .. [1] Kong, A., Liu, J. S. & Wong, W. H. (1994). Sequential imputations
       and Bayesian missing data problems. Journal of the American
       Statistical Association, 89(425), 278–288.
    """


def resample(key, ensemble: Ensemble, n_particles: int | None = None, *,
             scheme: str = "systematic") -> Ensemble:
    """Draw an unweighted ensemble from a weighted one.

    Parameters
    ----------
    n_particles : int, optional
        Defaults to the input's ``n_particles``.
    scheme : {"systematic", "multinomial"}
        Keyword-only. Systematic resampling has the lower variance.

    Returns
    -------
    Ensemble
        Unweighted, with duplicated particles where weights were large.
    """


def exact_moment_ensemble(key, gaussian: Gaussian, n_particles: int) -> Ensemble:
    """Particles whose sample mean and covariance equal ``gaussian``'s exactly.

    Exact *jointly*: every block's sample mean and covariance and every
    sample cross-covariance match. Draws one standard normal array for all the
    Gaussian's sources at once (the latent vector and each independent term's
    factor columns), removes its sample mean, makes its columns orthonormal
    with a thin QR (never a Cholesky of a sample covariance), scales by
    ``sqrt(n_particles - 1)`` and maps it through the Gaussian's factors. Used for
    exactness tests and for documentation whose numbers must not depend on
    sampling error.

    Raises
    ------
    ValueError
        If ``n_particles`` does not exceed the number of sources (``latent_dim``
        plus the widths of the independent terms' factors): exact moments need
        one more particle than sources.

    References
    ----------
    .. [1] Pham, D. T. (2001). Stochastic methods for sequential data
       assimilation in strongly nonlinear systems. Monthly Weather Review,
       129(5), 1194–1207.
    """
