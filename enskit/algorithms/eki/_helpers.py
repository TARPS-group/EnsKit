r"""The array-level pieces of an EKI run: misfits, the tempering ESS, the repair."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp

from ...distribution import Ensemble
from ...linalg import PSDLinOp

__all__ = ["misfits", "effective_sample_size", "repair_failed_particles"]


def misfits(y, predictions, noise_cov) -> Array:
    r"""The misfit of each prediction: ``(N,), (..., N) -> (...)``.

    .. math::

        \Phi(g) \;=\; \tfrac12 \bigl\lVert W(y - g) \bigr\rVert^2
        \;=\; \tfrac12 (y - g)^\top R^{-1} (y - g) ,

    with :math:`y` the data, :math:`g` a prediction, :math:`R` the noise
    covariance and :math:`W` any whitener of it, :math:`W^\top W = R^{-1}`.
    The value does not depend on which whitener the operator uses.

    Parameters
    ----------
    y : Array
        The data, ``(N,)``.
    predictions : Array
        Predictions, ``(..., N)``: vectors along the trailing axis, with any
        leading batch axes. An ensemble's ``(J, N)`` predictions give ``(J,)``
        misfits.
    noise_cov : PSDLinOp
        :math:`R`, of side ``N``, supporting ``whiten``. Pass the base
        covariance: a tempered :math:`R/\delta` would scale every misfit by
        :math:`\delta`.

    Returns
    -------
    Array
        Shape ``(...)``, the batch axes of ``predictions``.

    Raises
    ------
    TypeError
        If ``noise_cov`` is not a :class:`~enskit.linalg.PSDLinOp`.
    ValueError
        If ``y`` is not ``(N,)``, the trailing axis of ``predictions`` is not
        ``N``, or ``noise_cov`` is a vmapped family.
    UnsupportedOpError
        If ``noise_cov`` does not support ``whiten``.

    Notes
    -----
    With the factor :math:`\tfrac12`, :math:`e^{-\beta\Phi}` is the tempered
    likelihood, and the Gaussian log-likelihood is
    :math:`\log\mathcal N(y \mid g, R) = -\Phi(g) - \tfrac12(\log\det R +
    N\log 2\pi)`.
    """
    if not isinstance(noise_cov, PSDLinOp):
        raise TypeError(
            f"misfits: noise_cov must be an enskit.linalg.PSDLinOp, got "
            f"{type(noise_cov).__name__}"
        )
    data_dim = noise_cov.shape[0]
    y = jnp.asarray(y)
    if y.ndim != 1 or y.shape[0] != data_dim:
        raise ValueError(
            f"misfits: expected y of shape ({data_dim},) to match {noise_cov!r}, "
            f"got shape {y.shape}"
        )
    predictions = jnp.asarray(predictions)
    if predictions.ndim < 1 or predictions.shape[-1] != data_dim:
        raise ValueError(
            f"misfits: expected predictions of shape (..., {data_dim}), got shape "
            f"{predictions.shape}"
        )
    return _misfits_from_residuals(noise_cov.whiten(y - predictions))


def effective_sample_size(misfits, increment) -> Array:
    r"""The effective sample size of the tempering weights: ``(J,), scalar -> 0-d``.

    For misfits :math:`\Phi_j` and an increment :math:`\delta`, with weights
    :math:`w_j = e^{-\delta\Phi_j}`,

    .. math::

        \mathrm{ESS}(\delta)
        = \frac{\bigl(\sum_j w_j\bigr)^2}{\sum_j w_j^2}
        = \exp\bigl(2\,\mathrm{lse}(-\delta\Phi) - \mathrm{lse}(-2\delta\Phi)\bigr),

    computed by the second form, with :math:`\mathrm{lse}` a max-shifted
    log-sum-exp. The value lies in :math:`[1, J]` up to round-off (at
    :math:`\delta = 0` it is ``exp(log J)``), and is non-increasing in
    :math:`\delta`.

    Parameters
    ----------
    misfits : Array
        :math:`\Phi_j`, ``(J,)``.
    increment : float or Array
        :math:`\delta`, a scalar; may be traced.

    Returns
    -------
    Array
        0-d.

    Raises
    ------
    ValueError
        If ``misfits`` is not ``(J,)`` with :math:`J \ge 1`, or ``increment``
        is not a scalar.

    Notes
    -----
    :func:`enskit.distribution.effective_sample_size` is the same quantity
    for an ensemble's own normalized weights; this one is of the weights an
    increment *would* give, and is what the adaptive schedules bisect.

    The log-space form is needed: :math:`\delta\Phi_j` is often in the
    hundreds early in a run, where the direct form underflows every weight to
    zero and returns ``nan``.
    """
    misfits = jnp.asarray(misfits)
    if misfits.ndim != 1 or misfits.shape[0] < 1:
        raise ValueError(
            f"effective_sample_size: expected misfits of shape (J,) with J at least "
            f"1, got shape {misfits.shape}"
        )
    increment = jnp.asarray(increment)
    if increment.ndim != 0:
        raise ValueError(
            f"effective_sample_size: expected a scalar increment, got shape "
            f"{increment.shape}"
        )
    return _ess_from_misfits(misfits, increment)


def repair_failed_particles(*, ensemble: Ensemble, valid) -> Ensemble:
    r"""Move the failed particles to the valid particles' center, in every block.

    With :math:`m_j \in \{0, 1\}` the validity of particle :math:`j`,
    :math:`J_v = \sum_j m_j`, and :math:`\hat x = J_v^{-1}\sum_j m_j x_j` a
    block's mean over the valid particles,

    .. math::

        x_j \;\longmapsto\; m_j\, x_j + (1 - m_j)\, \hat x

    for every block. Valid particles are left exactly as they are, bit for
    bit, and failed ones are moved to the valid center.

    Parameters
    ----------
    ensemble : Ensemble
        Unweighted. Keyword-only.
    valid : Array
        ``(J,)`` boolean, ``True`` for a valid particle, at least two of
        them. Keyword-only.

    Returns
    -------
    Ensemble
        The same blocks, particle count and dtype.

    Raises
    ------
    TypeError
        If ``ensemble`` is not an :class:`~enskit.distribution.Ensemble`.
    ValueError
        If ``ensemble`` is weighted or a vmapped family, ``valid`` is not a
        ``(J,)`` boolean array, or fewer than two particles are valid (a
        single particle has no anomalies). The last check reads a value, so
        it is skipped when ``valid`` is traced.

    Notes
    -----
    Three identities hold exactly. The mean over all :math:`J` particles is
    :math:`\hat x`. The sample covariance and cross-covariance over all
    :math:`J` (divisor :math:`J - 1`) are the valid particles' (divisor
    :math:`J_v - 1`) times :math:`c = (J_v - 1)/(J - 1)`. And a repaired
    particle carries the valid center in every block, so it rejoins the
    ensemble rather than being lost: its anomaly is zero in every block.

    Conditioning with both covariance blocks scaled by :math:`c` gives the
    gain at the increment :math:`c\,\delta`: a failed particle costs a
    slightly shorter step. Rescaling the valid particles' anomalies by
    :math:`\sqrt{(J-1)/(J_v-1)}` would make the moments exact instead, at
    the cost of moving every valid particle outward, a silent inflation; this
    function does not do that.

    Keyword-only, so that ``ensemble`` and ``valid`` cannot be swapped.
    """
    where = "repair_failed_particles"
    if not isinstance(ensemble, Ensemble):
        raise TypeError(
            f"{where}: ensemble must be an enskit.distribution.Ensemble, got "
            f"{type(ensemble).__name__}"
        )
    if ensemble.batch_shape != ():
        raise ValueError(f"{where}: {ensemble!r} is a vmapped family")
    if ensemble.is_weighted:
        raise ValueError(
            f"{where}: the ensemble is weighted; the repair moves particles of an "
            f"unweighted ensemble. A weighted ensemble drops a failed particle by "
            f"giving it weight zero instead."
        )
    valid = jnp.asarray(valid)
    n_particles = ensemble.n_particles
    if valid.shape != (n_particles,) or valid.dtype != jnp.bool_:
        raise ValueError(
            f"{where}: valid must be a boolean array of shape ({n_particles},), got "
            f"shape {valid.shape} of dtype {valid.dtype}"
        )
    n_valid = _concrete_int(jnp.sum(valid))
    if n_valid is not None and n_valid < 2:
        raise ValueError(
            f"{where}: at least 2 valid particles are required, got {n_valid}. A "
            f"single particle has no anomalies."
        )
    return ensemble.assign(
        {name: _repair_block(ensemble[name], valid) for name in ensemble.names}
    )


# ---------------------------------------------------------------------------
# private
# ---------------------------------------------------------------------------


@jax.jit
def _misfits_from_residuals(whitened_residuals: Array) -> Array:
    """The misfit of each row of an already-whitened residual block."""
    return 0.5 * jnp.sum(whitened_residuals**2, axis=-1)


@jax.jit
def _ess_from_misfits(misfits: Array, increment: Array) -> Array:
    """The log-space effective sample size, with no shape checking."""
    log_w = -increment * misfits
    return jnp.exp(2.0 * logsumexp(log_w) - logsumexp(2.0 * log_w))


@jax.jit
def _repair_block(x: Array, valid: Array) -> Array:
    """``jnp.where`` on the rows, so a valid row is returned bit for bit."""
    mask = valid[:, None]
    center = jnp.sum(jnp.where(mask, x, 0), axis=0) / jnp.sum(valid).astype(x.dtype)
    return jnp.where(mask, x, center)


def _concrete_int(x) -> int | None:
    """``int(x)`` where that can be read, ``None`` under a trace."""
    try:
        return int(x)
    except (jax.errors.ConcretizationTypeError, TypeError):
        return None
