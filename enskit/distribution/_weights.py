r"""Operations on weights, and the exact-moment fixture."""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import Array

from . import _common as c
from ._ensemble import Ensemble, _ess
from ._gaussian import Gaussian

__all__ = ["reweight", "effective_sample_size", "resample", "exact_moment_ensemble"]

_SCHEMES = ("systematic", "multinomial")


def reweight(ensemble: Ensemble, log_weight_increments) -> Ensemble:
    r"""Multiply each particle's weight by :math:`\exp(\Delta\ell_j)`.

    .. math::

        \ell_j \mapsto \ell_j + \Delta\ell_j ,

    treating an unweighted ensemble's log weights as all zeros, so importance
    weights and likelihood factors compose by repeated calls without leaving
    log space.

    Parameters
    ----------
    ensemble : Ensemble
        The particles; unchanged.
    log_weight_increments : Array
        Exactly ``(J,)``, real, converted to the ensemble's dtype. An entry
        of :math:`-\infty` gives that particle weight zero.

    Returns
    -------
    Ensemble
        The same particles, always weighted.

    Raises
    ------
    TypeError
        If ``ensemble`` is not an :class:`Ensemble`, or the increments are
        not real.
    ValueError
        If the increments are not ``(J,)``. In debug mode, also if an
        increment is ``nan`` or :math:`+\infty`, or if every resulting log
        weight is :math:`-\infty`; otherwise the weights are then ``nan``.
    """
    where = "reweight"
    _check_ensemble(where, ensemble)
    c.guard(ensemble, "reweight")
    inc = jnp.asarray(log_weight_increments)
    c.check_real(where, "log_weight_increments", inc)
    J = ensemble.n_particles
    if inc.shape != (J,):
        raise ValueError(
            f"{where}: log_weight_increments must have shape ({J},), got {inc.shape}"
        )
    inc = inc.astype(ensemble._dtype)
    c.check_finite(
        where,
        "log_weight_increments other than -inf",
        jnp.where(inc == -jnp.inf, 0.0, inc),
    )
    base = ensemble.log_weights
    new = inc if base is None else base + inc
    c.lazy_value_check(
        new,
        lambda lw: jnp.any(lw > -jnp.inf),
        lambda: f"{where}: every log weight is -inf, so the weights are undefined",
    )
    return c.build(
        Ensemble, names=ensemble.names, _blocks=ensemble._blocks, _log_weights=new
    )


def effective_sample_size(ensemble: Ensemble) -> Array:
    r"""The effective sample size of an ensemble's weights, as a 0-d array.

    .. math::

        \mathrm{ESS} = \frac{1}{\sum_j w_j^2}
          = \exp\big(2\operatorname{lse}(\ell) - \operatorname{lse}(2\ell)\big),

    computed in the second form, which stays finite when the log weights span
    hundreds of units. It is :math:`J` when unweighted.

    Raises
    ------
    TypeError
        If ``ensemble`` is not an :class:`Ensemble`.

    References
    ----------
    Kong, A., Liu, J. S. & Wong, W. H. (1994). Sequential imputations and
    Bayesian missing data problems. *Journal of the American Statistical
    Association*, 89(425), 278–288.
    """
    _check_ensemble("effective_sample_size", ensemble)
    c.guard(ensemble, "effective_sample_size")
    if ensemble.log_weights is None:
        return jnp.asarray(float(ensemble.n_particles), ensemble._dtype)
    return _ess(ensemble.log_weights)


def resample(
    key, ensemble: Ensemble, n_particles: int | None = None, *, scheme: str = "systematic"
) -> Ensemble:
    r"""Draw an unweighted ensemble from a weighted one.

    Particles are selected with probability proportional to their weights,
    so heavily weighted particles are duplicated. With normalized weights
    :math:`w` and :math:`n` = ``n_particles``, the selected indices are

    ``"systematic"`` (Kitagawa, 1996)
        :math:`i_m = \min\{\, i : c_i > c_J (u + m)/n \,\}` with
        :math:`c_i = \sum_{l \le i} w_l`, :math:`m = 0, \dots, n - 1`, for
        one :math:`u \sim \mathcal U[0, 1)`: ``u = uniform(key, (), dtype)``,
        ``c = cumsum(w)`` and
        ``minimum(searchsorted(c, c[-1] * (u + arange(n)) / n, side="right"),
        last)``, with ``last`` the index of the last particle of positive
        weight. Scaling by ``c[-1]`` and the bound by ``last`` ensure a
        particle of weight zero is never selected, however the cumulative
        sum rounds.
    ``"multinomial"`` (Gordon et al., 1993)
        :math:`n` independent draws, ``categorical(key, log_weights,
        shape=(n,))``, with zeros for the log weights when unweighted.

    Systematic resampling has the lower variance. The draws are pinned.

    Parameters
    ----------
    key
        A typed key, consumed whole.
    ensemble : Ensemble
        The particles; an unweighted one is resampled with equal weights.
    n_particles : int, optional
        A Python ``int >= 2``; defaults to the input's.
    scheme : {"systematic", "multinomial"}
        Keyword-only.

    Returns
    -------
    Ensemble
        Unweighted.

    Raises
    ------
    TypeError
        If ``ensemble`` is not an :class:`Ensemble`, ``n_particles`` is not an
        ``int``, or ``key`` is not a typed key.
    ValueError
        If ``n_particles < 2``, ``scheme`` is not one of the two, or ``key``
        is ``None``.

    Notes
    -----
    The selected indices are discrete, so the result's derivative with
    respect to the weights is zero.

    References
    ----------
    Gordon, N. J., Salmond, D. J. & Smith, A. F. M. (1993). Novel approach to
    nonlinear/non-Gaussian Bayesian state estimation. *IEE Proceedings F
    (Radar and Signal Processing)*, 140(2), 107–113.

    Kitagawa, G. (1996). Monte Carlo filter and smoother for non-Gaussian
    nonlinear state space models. *Journal of Computational and Graphical
    Statistics*, 5(1), 1–25.
    """
    where = "resample"
    _check_ensemble(where, ensemble)
    c.guard(ensemble, "resample")
    J = ensemble.n_particles
    n = J if n_particles is None else n_particles
    c.check_n_particles(where, n)
    if scheme not in _SCHEMES:
        raise ValueError(f"{where}: scheme must be one of {_SCHEMES}, got {scheme!r}")
    c.check_key(where, key, required=True)
    dtype = ensemble._dtype
    if scheme == "systematic":
        w = ensemble.weights
        u = jax.random.uniform(key, (), dtype)
        cw = jnp.cumsum(w)
        positions = cw[-1] * (u + jnp.arange(n, dtype=dtype)) / n
        # the last particle of positive weight: rounding can put a position
        # at or past cw[-1], and searchsorted then returns J
        last = J - 1 - jnp.argmax(w[::-1] > 0)
        idx = jnp.minimum(jnp.searchsorted(cw, positions, side="right"), last)
    else:
        lw = ensemble.log_weights
        if lw is None:
            lw = jnp.zeros((J,), dtype)
        idx = jax.random.categorical(key, lw, shape=(n,))
    blocks = tuple(jnp.take(a, idx, axis=0) for a in ensemble._blocks)
    return c.build(Ensemble, names=ensemble.names, _blocks=blocks, _log_weights=None)


def exact_moment_ensemble(key, gaussian: Gaussian, n_particles: int) -> Ensemble:
    r"""Particles whose sample mean and covariance equal ``gaussian``'s exactly.

    Exact *jointly*: every block's sample mean and covariance (divisor
    :math:`J - 1`) and every sample cross-covariance match. The Gaussian has
    :math:`s = k + \sum_b w_b` *sources*: the latent vector and the columns
    of each independent term's factor :math:`L_b` = ``D_b.factor()``, of
    width :math:`w_b`.

    1. Draw :math:`Z`, a ``(J, s)`` standard normal array,
       ``normal(key, (J, s))``, its columns the latent vector then each
       term's factor columns in block order.
    2. Center its columns, :math:`Z \mapsto Z - \mathbf 1\bar z^\top`.
    3. Orthonormalize with a thin QR decomposition, :math:`Z = QR`, so that
       :math:`Q^\top Q = I_s` and :math:`Q^\top \mathbf 1 = 0`.
    4. With :math:`Q_\xi` the first :math:`k` columns of :math:`Q` and
       :math:`Q_b` block :math:`b`'s,

       .. math::

           x_j^{(b)} = m_b + \sqrt{J-1}\,\big(F_b\, (Q_\xi)_{j\cdot}^\top
             + L_b\,(Q_b)_{j\cdot}^\top\big).

    The sample mean is then :math:`m_b` and the sample covariance
    :math:`FF^\top + \operatorname{blockdiag}(L_bL_b^\top)`, to round-off.

    A fixture: tests use it to build inputs whose moments are known exactly,
    and documentation where numbers must not depend on sampling error.

    Parameters
    ----------
    key
        A typed key, consumed whole.
    gaussian : Gaussian
        The target distribution; every independent term must support
        ``factor``.
    n_particles : int
        :math:`J`, a Python ``int``; must exceed :math:`s`.

    Returns
    -------
    Ensemble
        Unweighted, over every block, in block order.

    Raises
    ------
    TypeError
        If ``gaussian`` is not a :class:`Gaussian`, ``n_particles`` is not an
        ``int``, or ``key`` is not a typed key.
    ValueError
        If ``n_particles <= s``: at :math:`J = s` the centered columns have
        rank at most :math:`s - 1` and the construction fails silently. Also
        if ``key`` is ``None``.
    UnsupportedOpError
        If an independent term cannot ``factor``.

    Notes
    -----
    The orthonormalization is a QR decomposition of the draw, never a
    Cholesky factorization of a sample covariance, which fails on the
    rank-deficient targets that matter.

    References
    ----------
    Pham, D. T. (2001). Stochastic methods for sequential data assimilation
    in strongly nonlinear systems. *Monthly Weather Review*, 129(5),
    1194–1207.
    """
    where = "exact_moment_ensemble"
    if not isinstance(gaussian, Gaussian):
        raise TypeError(
            f"{where}: gaussian must be a Gaussian, got {type(gaussian).__name__}"
        )
    c.guard(gaussian, "exact_moment_ensemble")
    c.check_n_particles(where, n_particles)
    c.check_key(where, key, required=True)
    for D in gaussian._block_covs:
        if D is not None:
            c.require(D, "factor")
    Ls = [None if D is None else D.factor() for D in gaussian._block_covs]
    k = gaussian.latent_dim
    s = k + sum(L.shape[1] for L in Ls if L is not None)
    J = n_particles
    if J <= s:
        raise ValueError(
            f"{where}: n_particles must exceed the number of sources s = {s} (the "
            f"latent width plus the independent terms' factor widths), got {J}"
        )
    dtype = gaussian._dtype
    Z = jax.random.normal(key, (J, s), dtype)
    Z = Z - jnp.mean(Z, axis=-2, keepdims=True)
    Q, _ = jnp.linalg.qr(Z)
    scale = math.sqrt(J - 1)
    blocks, offset = [], k
    for m, F, L in zip(gaussian._means, gaussian._factors, Ls, strict=True):
        dev = jnp.zeros((J, m.shape[-1]), dtype)
        if F is not None:
            dev = dev + F.matvec(Q[:, :k])
        if L is not None:
            w = L.shape[1]
            dev = dev + L.matvec(Q[:, offset : offset + w])
            offset += w
        blocks.append(m + scale * dev)
    return c.build(
        Ensemble, names=gaussian.names, _blocks=tuple(blocks), _log_weights=None
    )


def _check_ensemble(where: str, ensemble) -> None:
    if not isinstance(ensemble, Ensemble):
        raise TypeError(f"{where}: expected an Ensemble, got {type(ensemble).__name__}")
