r"""Inflation and relaxation: functions on ensembles applied around an update."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..distribution import Ensemble
from ..linalg import PSDLinOp, value_check
from . import _common as c

__all__ = [
    "inflate_multiplicative",
    "inflate_additive",
    "relax_to_prior_spread",
    "relax_to_prior_perturbations",
]


def inflate_multiplicative(ensemble: Ensemble, anomaly_scale, names=None) -> Ensemble:
    r"""Multiplicative inflation: scale each named block's anomalies.

    .. math::

        x_j \mapsto \bar x + \lambda\,(x_j - \bar x), \qquad
        \hat C \mapsto \lambda^2 \hat C,

    with :math:`\lambda` = ``anomaly_scale`` and :math:`\bar x` the (weighted)
    mean, which is preserved. The covariance grows by :math:`\lambda^2`, so a
    variance inflation of 1.2 is ``anomaly_scale=math.sqrt(1.2)``.

    Parameters
    ----------
    ensemble : Ensemble
        The particles, weighted or not; the weights are kept.
    anomaly_scale : float or Array
        :math:`\lambda`, the factor on the anomalies: a real scalar, possibly
        traced, converted to the ensemble's dtype.
    names : str or sequence of str, optional
        The blocks to inflate; every block by default.

    Returns
    -------
    Ensemble
        The named blocks replaced, every other block unchanged.

    Raises
    ------
    ValueError
        If ``anomaly_scale`` is not a scalar (an array would inflate each
        coordinate by a different factor). In debug mode, also if it is not
        finite and positive, or a particle of positive weight is not finite.
    KeyError
        If a name is not a block.

    References
    ----------
    Anderson, J. L. & Anderson, S. L. (1999). A Monte Carlo implementation of
    the nonlinear filtering problem to produce ensemble assimilations and
    forecasts. *Monthly Weather Review*, 127(12), 2741–2758.
    """
    where = "kalman.inflate_multiplicative"
    c.guard(ensemble, where)
    c.check_ensemble(where, ensemble)
    names = _names(where, ensemble, names)
    lam = c.check_scalar(where, "anomaly_scale", anomaly_scale, _dtype(ensemble))
    value_check(
        lam,
        lambda r: bool(jnp.isfinite(r) & (r > 0)),
        f"{where}: anomaly_scale must be finite and positive",
    )
    c.check_particles_finite(where, ensemble, names)
    return ensemble.assign(
        {n: ensemble.mean(n) + lam * ensemble.anomalies(n) for n in names}
    )


def inflate_additive(key, ensemble: Ensemble, covs=None, /, **block_covs) -> Ensemble:
    r"""Additive inflation: add centered Gaussian draws to the named blocks.

    .. math::

        x_j^{(b)} \mapsto x_j^{(b)} + \varepsilon_j^{(b)} - \bar\varepsilon^{(b)},
        \qquad \varepsilon_j^{(b)} = L_b\eta_j^{(b)} \sim \mathcal N(0, Q_b),
        \qquad \bar\varepsilon^{(b)} = \sum_j w_j \varepsilon_j^{(b)},

    for each block :math:`b` with :math:`Q_b = L_bL_b^\top` given, and the
    ensemble's weights :math:`w_j` (:math:`1/J` when unweighted). The mean is
    unchanged to round-off, and the covariance grows by :math:`Q_b` in
    expectation. Unlike an update, which moves particles within the span of
    their anomalies, this adds variance in new directions.

    The draw is pinned: ``keys = jax.random.split(key, n)`` over the
    :math:`n` named blocks in the ensemble's block order, and
    :math:`\varepsilon = L_b` ``normal(keys[i], (J, w_b))``.

    Parameters
    ----------
    key : jax.random key
        Typed key, consumed whole.
    ensemble : Ensemble
        The particles, weighted or not; the weights are kept.
    covs : Mapping[str, PSDLinOp], optional
        Positional-only. :math:`Q_b` by block name, each of its block's side
        and supporting ``factor``.
    **block_covs : PSDLinOp
        The covariances as keywords. At least one in all.

    Returns
    -------
    Ensemble
        The named blocks replaced, every other block unchanged.

    Raises
    ------
    TypeError
        If ``key`` is not a typed key, a covariance is not a
        :class:`~enskit.linalg.PSDLinOp` or is of another dtype than the
        ensemble, or a block is given twice.
    KeyError
        If a name is not a block.
    ValueError
        If no covariance is given, one has the wrong side or is a vmapped
        family, or the key is missing. In debug mode, also if a particle of
        positive weight is not finite.
    UnsupportedOpError
        If a covariance cannot ``factor``, before any work.

    References
    ----------
    Hamill, T. M. & Whitaker, J. S. (2005). Accounting for the error due to
    unresolved scales in ensemble data assimilation: a comparison of
    different approaches. *Monthly Weather Review*, 133(11), 3132–3147.
    """
    where = "kalman.inflate_additive"
    c.guard(ensemble, where)
    c.check_ensemble(where, ensemble)
    covs = c.merge_blocks(where, covs, block_covs)
    if not covs:
        raise ValueError(f"{where}: at least one covariance is required")
    c.name_list(where, ensemble.names, tuple(covs))
    for name, Q in covs.items():
        if not isinstance(Q, PSDLinOp):
            raise TypeError(
                f"{where}: the covariance of block {name!r} must be a PSDLinOp, got "
                f"{type(Q).__name__}"
            )
        if Q.batch_shape != ():
            raise ValueError(
                f"{where}: the covariance of block {name!r} is a vmapped family"
            )
        if Q.shape[0] != ensemble.dims[name]:
            raise ValueError(
                f"{where}: the covariance of block {name!r} has side {Q.shape[0]}; the "
                f"block has dimension {ensemble.dims[name]}"
            )
    c.check_key(where, key, required=True)
    ordered = tuple(n for n in ensemble.names if n in covs)
    for name in ordered:
        covs[name]._require("factor")
    c.check_particles_finite(where, ensemble, ordered)
    J, dtype = ensemble.n_particles, _dtype(ensemble)
    keys = jax.random.split(key, len(ordered))
    out = {}
    for i, name in enumerate(ordered):
        L = covs[name].factor()
        eps = L.matvec(jax.random.normal(keys[i], (J, L.shape[1]), dtype))
        if eps.dtype != dtype:
            raise TypeError(
                f"{where}: the covariance of block {name!r} has dtype {eps.dtype}; the "
                f"ensemble has {dtype}"
            )
        if ensemble.is_weighted:
            center = jnp.sum(ensemble.weights[:, None] * eps, axis=0, keepdims=True)
        else:
            center = jnp.mean(eps, axis=0, keepdims=True)
        out[name] = ensemble[name] + (eps - center)
    return ensemble.assign(out)


def relax_to_prior_spread(
    prior: Ensemble, posterior: Ensemble, alpha, names=None
) -> Ensemble:
    r"""RTPS: relax the posterior spread toward the prior's, coordinate by coordinate.

    .. math::

        a_j^{\text{post}} \mapsto a_j^{\text{post}} \Big(1 + \alpha\,
          \frac{s^{\text{prior}} - s^{\text{post}}}{s^{\text{post}}}\Big),
        \qquad x_j^{\text{post}} \mapsto \bar x^{\text{post}} + a_j^{\text{post}},

    elementwise, with :math:`a` the anomalies and
    :math:`s_i = \big(\sum_j a_{ji}^2/(J-1)\big)^{1/2}` the per-coordinate
    sample standard deviations. A coordinate whose posterior spread is exactly
    zero is left unchanged.

    Parameters
    ----------
    prior : Ensemble
        The ensemble the update started from; unweighted.
    posterior : Ensemble
        The update's result; unweighted, with the prior's particle count and
        dtype, particle :math:`j` corresponding to the prior's particle
        :math:`j`.
    alpha : float or Array
        :math:`\alpha`, a real scalar, in :math:`[0, 1]`.
    names : str or sequence of str, optional
        The blocks to relax, each in both ensembles; the posterior's blocks
        by default.

    Returns
    -------
    Ensemble
        ``posterior`` with the named blocks replaced.

    Raises
    ------
    ValueError
        If either ensemble is weighted, their counts or a named block's
        dimensions differ, or ``alpha`` is not a scalar. In debug mode, also if
        ``alpha`` is outside :math:`[0, 1]` or a particle is not finite.
    KeyError
        If a name is not a block of both ensembles.
    TypeError
        If an argument is not an ensemble or their dtypes differ.

    References
    ----------
    Whitaker, J. S. & Hamill, T. M. (2012). Evaluating methods to account for
    system errors in ensemble data assimilation. *Monthly Weather Review*,
    140(9), 3078–3089.
    """
    where = "kalman.relax_to_prior_spread"
    names, a = _check_pair(where, prior, posterior, alpha, names)
    out = {}
    for name in names:
        ap, aq = prior.anomalies(name), posterior.anomalies(name)
        J = posterior.n_particles
        s_prior = _safe_sqrt(jnp.sum(ap * ap, axis=0) / (J - 1))
        v_post = jnp.sum(aq * aq, axis=0) / (J - 1)
        positive = v_post > 0
        s_post = jnp.sqrt(jnp.where(positive, v_post, 1))
        factor = jnp.where(positive, 1 + a * (s_prior - s_post) / s_post, 1)
        out[name] = posterior.mean(name) + aq * factor
    return posterior.assign(out)


def relax_to_prior_perturbations(
    prior: Ensemble, posterior: Ensemble, alpha, names=None
) -> Ensemble:
    r"""RTPP: blend the posterior and prior anomalies, particle by particle.

    .. math::

        a_j^{\text{post}} \mapsto (1 - \alpha)\, a_j^{\text{post}}
            + \alpha\, a_j^{\text{prior}},
        \qquad x_j^{\text{post}} \mapsto \bar x^{\text{post}} + a_j^{\text{post}},

    with :math:`a` the anomalies. The arguments, result and errors are those
    of :func:`relax_to_prior_spread`.

    References
    ----------
    Zhang, F., Snyder, C. & Sun, J. (2004). Impacts of initial estimate and
    observation availability on convective-scale data assimilation with an
    ensemble Kalman filter. *Monthly Weather Review*, 132(5), 1238–1253.
    """
    where = "kalman.relax_to_prior_perturbations"
    names, a = _check_pair(where, prior, posterior, alpha, names)
    return posterior.assign(
        {
            n: posterior.mean(n)
            + (1 - a) * posterior.anomalies(n)
            + a * prior.anomalies(n)
            for n in names
        }
    )


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------


def _dtype(ensemble: Ensemble):
    return ensemble[ensemble.names[0]].dtype


def _names(where: str, ensemble: Ensemble, names) -> tuple[str, ...]:
    if names is None:
        return ensemble.names
    names = c.name_list(where, ensemble.names, names)
    if not names:
        raise ValueError(f"{where}: names must name at least one block")
    return names


def _check_pair(where, prior, posterior, alpha, names):
    """The shared checks of the two relaxations; returns ``(names, alpha)``."""
    c.guard(prior, where)
    c.guard(posterior, where)
    c.check_ensemble(where, prior, "prior")
    c.check_ensemble(where, posterior, "posterior")
    if names is None:
        names = posterior.names
    names = c.name_list(where, posterior.names, names)
    for name in names:
        if name not in prior.names:
            raise KeyError(
                f"{where}: block {name!r} is not in the prior; its blocks are "
                f"{prior.names}"
            )
    if not names:
        raise ValueError(f"{where}: names must name at least one block")
    a = c.check_scalar(where, "alpha", alpha, _dtype(posterior))
    why = "they are an update's input and output, which are unweighted"
    if prior.is_weighted or posterior.is_weighted:
        raise ValueError(f"{where}: the prior and posterior must be unweighted; {why}")
    if prior.n_particles != posterior.n_particles:
        raise ValueError(
            f"{where}: the prior has {prior.n_particles} particles and the posterior "
            f"{posterior.n_particles}"
        )
    for name in names:
        if prior.dims[name] != posterior.dims[name]:
            raise ValueError(
                f"{where}: block {name!r} has dimension {prior.dims[name]} in the prior "
                f"and {posterior.dims[name]} in the posterior"
            )
    if _dtype(prior) != _dtype(posterior):
        raise TypeError(
            f"{where}: the prior has dtype {_dtype(prior)}, the posterior "
            f"{_dtype(posterior)}"
        )
    value_check(
        a,
        lambda r: bool(jnp.isfinite(r) & (r >= 0) & (r <= 1)),
        f"{where}: alpha must be finite and in [0, 1]",
    )
    c.check_particles_finite(where, prior, names)
    c.check_particles_finite(where, posterior, names)
    return names, a


def _safe_sqrt(v):
    """``sqrt(v)``, with a finite derivative where ``v`` is exactly zero."""
    positive = v > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, v, 1)), 0)
