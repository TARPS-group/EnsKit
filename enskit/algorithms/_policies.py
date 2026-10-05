r"""Inflation and relaxation policies, shared by every driver."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from jax import Array

from .. import kalman
from ..distribution import Ensemble
from ..linalg import PSDLinOp
from . import _common as c

__all__ = [
    "Inflation",
    "Relaxation",
    "MultiplicativeInflation",
    "AdditiveInflation",
    "RelaxToPriorSpread",
    "RelaxToPriorPerturbations",
]


@runtime_checkable
class Inflation(Protocol):
    """A transformation of the particles a driver applies before using them.

    Any callable with this signature is an inflation; nothing subclasses
    this class. A driver calls it on the particles just before they are used
    (:mod:`enskit.algorithms.eki`: before each evaluation of the forward
    model) and passes the context it has as keywords, which an inflation
    ignores when it does not use them.

    Obligations, which :func:`enskit.testing.check_inflation` checks:

    - it returns an unweighted :class:`~enskit.distribution.Ensemble` with the
      same blocks in the same order, the same particle count and the same
      dtype as ``ensemble``;
    - it consumes ``key`` whole and is a deterministic function of its
      arguments, the key included, holding no state across calls;
    - it accepts and ignores context keywords it does not use.
    """

    def __call__(self, key, *, ensemble: Ensemble, **context) -> Ensemble:
        """Return the inflated particles."""


@runtime_checkable
class Relaxation(Protocol):
    """A transformation a driver applies to an update's result.

    Any callable with this signature is a relaxation. A driver calls it just
    after each update, with ``prior`` the particles the update started from
    and ``posterior`` its result, particle :math:`j` of one corresponding to
    particle :math:`j` of the other, and passes its context as keywords.

    Obligations, which :func:`enskit.testing.check_relaxation` checks: it
    returns an unweighted :class:`~enskit.distribution.Ensemble` with the
    blocks, particle count and dtype of ``posterior``; it is a deterministic
    function of its arguments, holding no state across calls; and it accepts
    and ignores context keywords it does not use.
    """

    def __call__(self, *, prior: Ensemble, posterior: Ensemble, **context) -> Ensemble:
        """Return the relaxed posterior particles."""


@c.pytree_class(data=("anomaly_scale",), meta=("names",))
class MultiplicativeInflation:
    r"""Multiplicative inflation: scale each named block's anomalies.

    Calls :func:`enskit.kalman.inflate_multiplicative`:

    .. math::

        x_j \mapsto \bar x + \lambda\,(x_j - \bar x), \qquad
        \hat C \mapsto \lambda^2 \hat C,

    with :math:`\lambda` = ``anomaly_scale``, :math:`x_j` particle
    :math:`j`'s value of a named block and :math:`\bar x` the block's mean,
    which is preserved. The key and the context are not used.

    Parameters
    ----------
    anomaly_scale : float or Array
        :math:`\lambda`, the factor on the *anomalies*; the covariance grows by
        its square, so an intended variance inflation of 1.2 is
        ``math.sqrt(1.2)``. A real scalar, stored as a 0-d array, so a traced
        value flows through. Positive.
    names : str or sequence of str, optional
        Keyword-only. The blocks to inflate; every block by default.

    Raises
    ------
    ValueError
        If ``anomaly_scale`` is not a scalar, or a name is repeated.
    TypeError
        If ``anomaly_scale`` is not a real number, or a name is not a ``str``.

    References
    ----------
    Anderson, J. L. & Anderson, S. L. (1999). A Monte Carlo implementation of
    the nonlinear filtering problem to produce ensemble assimilations and
    forecasts. *Monthly Weather Review*, 127(12), 2741–2758.
    """

    anomaly_scale: Array
    names: tuple[str, ...] | None

    def __init__(self, anomaly_scale, *, names=None) -> None:
        where = "MultiplicativeInflation"
        c.set_fields(
            self,
            anomaly_scale=_real_scalar(where, "anomaly_scale", anomaly_scale),
            names=_names(where, names),
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The family's batch shape, ``()`` for a directly constructed object."""
        return tuple(self.anomaly_scale.shape)

    def __call__(self, key, *, ensemble: Ensemble, **context) -> Ensemble:
        """Return ``ensemble`` with the named blocks' anomalies scaled."""
        c.guard(self, f"{self!r}")
        return kalman.inflate_multiplicative(ensemble, self.anomaly_scale, self.names)

    def __repr__(self) -> str:
        """As ``MultiplicativeInflation(anomaly_scale=1.05)``; never raises."""
        return c.safe_repr(
            lambda: _repr(self, _scalar("anomaly_scale", self.anomaly_scale)),
            "MultiplicativeInflation",
            lambda: self.batch_shape,
        )


@c.pytree_class(data=("covs",), meta=("names",))
class AdditiveInflation:
    r"""Additive inflation: add centered Gaussian draws to the named blocks.

    Calls :func:`enskit.kalman.inflate_additive` with the key:

    .. math::

        x_j^{(b)} \mapsto x_j^{(b)} + \varepsilon_j^{(b)} - \bar\varepsilon^{(b)},
        \qquad \varepsilon_j^{(b)} \sim \mathcal N(0, Q_b),

    for each block :math:`b` with a covariance :math:`Q_b`, where
    :math:`\bar\varepsilon^{(b)}` is the draws' mean. The mean is preserved
    and the sample covariance grows by :math:`Q_b` in expectation. Unlike an
    update, this adds variance in directions the particles do not span. The
    context is not used.

    Parameters
    ----------
    covs : Mapping[str, PSDLinOp], optional
        Positional-only. :math:`Q_b` for each block to inflate, each
        supporting ``factor``. A scale is folded into the operator:
        ``AdditiveInflation(u=0.01 * prior_cov)``.
    **block_covs : PSDLinOp
        The same, as keywords. At least one covariance in all.

    Raises
    ------
    TypeError
        If a covariance is not a :class:`~enskit.linalg.PSDLinOp`, or a block
        is given twice.
    ValueError
        If no covariance is given, or one is a vmapped family.

    References
    ----------
    Hamill, T. M. & Whitaker, J. S. (2005). Accounting for the error due to
    unresolved scales in ensemble data assimilation: a comparison of different
    approaches. *Monthly Weather Review*, 133(11), 3132–3147.
    """

    covs: tuple[PSDLinOp, ...]
    names: tuple[str, ...]

    def __init__(self, covs=None, /, **block_covs) -> None:
        where = "AdditiveInflation"
        merged = _merge(where, covs, block_covs)
        if not merged:
            raise ValueError(f"{where}: at least one covariance is required")
        for name, cov in merged.items():
            if not isinstance(cov, PSDLinOp):
                raise TypeError(
                    f"{where}: the covariance of block {name!r} must be an "
                    f"enskit.linalg.PSDLinOp, got {type(cov).__name__}"
                )
            if cov.batch_shape != ():
                raise ValueError(
                    f"{where}: the covariance of block {name!r} is a vmapped "
                    f"family, {cov!r}; build a family of inflations with jax.vmap "
                    f"over the constructor"
                )
        c.set_fields(self, covs=tuple(merged.values()), names=tuple(merged))

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The family's batch shape, ``()`` for a directly constructed object."""
        return c.broadcast_batch(
            "AdditiveInflation", *(tuple(q.batch_shape) for q in self.covs)
        )

    def __call__(self, key, *, ensemble: Ensemble, **context) -> Ensemble:
        """Return ``ensemble`` with centered draws added to the named blocks."""
        c.guard(self, f"{self!r}")
        covs = dict(zip(self.names, self.covs, strict=True))
        return kalman.inflate_additive(key, ensemble, covs)

    def __repr__(self) -> str:
        """As ``AdditiveInflation(u=PSDDiagonal(2, 2))``; never raises."""
        return c.safe_repr(
            lambda: "AdditiveInflation("
            + ", ".join(f"{n}={q!r}" for n, q in zip(self.names, self.covs, strict=True))
            + ")",
            "AdditiveInflation",
            lambda: self.batch_shape,
        )


@c.pytree_class(data=("alpha",), meta=("names",))
class RelaxToPriorSpread:
    r"""Relaxation to the prior spread (RTPS), coordinate by coordinate.

    Calls :func:`enskit.kalman.relax_to_prior_spread`:

    .. math::

        a_j^{\text{post}} \mapsto a_j^{\text{post}} \Big(1 + \alpha\,
          \frac{s^{\text{prior}} - s^{\text{post}}}{s^{\text{post}}}\Big),
        \qquad x_j^{\text{post}} \mapsto \bar x^{\text{post}} + a_j^{\text{post}},

    elementwise, with :math:`a_j` particle :math:`j`'s anomaly,
    :math:`\bar x^{\text{post}}` the posterior mean and :math:`s` the
    per-coordinate sample standard deviations (divisor :math:`J - 1`). A
    coordinate of zero posterior spread is left unchanged. The context is not
    used.

    Parameters
    ----------
    alpha : float or Array
        :math:`\alpha \in [0, 1]`: 0 leaves the posterior as it is, 1 restores
        the prior spread. A real scalar, stored as a 0-d array.
    names : str or sequence of str, optional
        Keyword-only. The blocks to relax; every block of the posterior by
        default.

    Raises
    ------
    ValueError
        If ``alpha`` is not a scalar, or a name is repeated.
    TypeError
        If ``alpha`` is not a real number, or a name is not a ``str``.

    References
    ----------
    Whitaker, J. S. & Hamill, T. M. (2012). Evaluating methods to account for
    system errors in ensemble data assimilation. *Monthly Weather Review*,
    140(9), 3078–3089.
    """

    alpha: Array
    names: tuple[str, ...] | None

    def __init__(self, alpha, *, names=None) -> None:
        where = "RelaxToPriorSpread"
        c.set_fields(
            self, alpha=_real_scalar(where, "alpha", alpha), names=_names(where, names)
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The family's batch shape, ``()`` for a directly constructed object."""
        return tuple(self.alpha.shape)

    def __call__(self, *, prior: Ensemble, posterior: Ensemble, **context) -> Ensemble:
        """Return ``posterior`` with the named blocks' spread relaxed."""
        c.guard(self, f"{self!r}")
        return kalman.relax_to_prior_spread(prior, posterior, self.alpha, self.names)

    def __repr__(self) -> str:
        """As ``RelaxToPriorSpread(alpha=0.5)``; never raises."""
        return c.safe_repr(
            lambda: _repr(self, _scalar("alpha", self.alpha)),
            "RelaxToPriorSpread",
            lambda: self.batch_shape,
        )


@c.pytree_class(data=("alpha",), meta=("names",))
class RelaxToPriorPerturbations:
    r"""Relaxation to the prior perturbations (RTPP), particle by particle.

    Calls :func:`enskit.kalman.relax_to_prior_perturbations`:

    .. math::

        a_j^{\text{post}} \mapsto (1 - \alpha)\, a_j^{\text{post}}
            + \alpha\, a_j^{\text{prior}},
        \qquad x_j^{\text{post}} \mapsto \bar x^{\text{post}} + a_j^{\text{post}},

    with :math:`a_j` particle :math:`j`'s anomaly in the posterior and in the
    prior, and :math:`\bar x^{\text{post}}` the posterior mean. The context is
    not used.

    Parameters
    ----------
    alpha : float or Array
        :math:`\alpha \in [0, 1]`: 0 leaves the posterior as it is, 1 restores
        the prior anomalies. A real scalar, stored as a 0-d array.
    names : str or sequence of str, optional
        Keyword-only. The blocks to relax; every block of the posterior by
        default.

    Raises
    ------
    ValueError
        If ``alpha`` is not a scalar, or a name is repeated.
    TypeError
        If ``alpha`` is not a real number, or a name is not a ``str``.

    References
    ----------
    Zhang, F., Snyder, C. & Sun, J. (2004). Impacts of initial estimate and
    observation availability on convective-scale data assimilation with an
    ensemble Kalman filter. *Monthly Weather Review*, 132(5), 1238–1253.
    """

    alpha: Array
    names: tuple[str, ...] | None

    def __init__(self, alpha, *, names=None) -> None:
        where = "RelaxToPriorPerturbations"
        c.set_fields(
            self, alpha=_real_scalar(where, "alpha", alpha), names=_names(where, names)
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The family's batch shape, ``()`` for a directly constructed object."""
        return tuple(self.alpha.shape)

    def __call__(self, *, prior: Ensemble, posterior: Ensemble, **context) -> Ensemble:
        """Return ``posterior`` with the named blocks' anomalies blended."""
        c.guard(self, f"{self!r}")
        return kalman.relax_to_prior_perturbations(
            prior, posterior, self.alpha, self.names
        )

    def __repr__(self) -> str:
        """As ``RelaxToPriorPerturbations(alpha=0.5)``; never raises."""
        return c.safe_repr(
            lambda: _repr(self, _scalar("alpha", self.alpha)),
            "RelaxToPriorPerturbations",
            lambda: self.batch_shape,
        )


def _real_scalar(where: str, name: str, value) -> Array:
    """A real scalar as a 0-d array; a ``(d,)`` array would broadcast silently."""
    if isinstance(value, bool):
        raise TypeError(f"{where}: {name} must be a real number, got the bool {value!r}")
    try:
        arr = jnp.asarray(value)
    except TypeError as exc:
        raise TypeError(
            f"{where}: {name} must be a real number, got {type(value).__name__}"
        ) from exc
    real = jnp.issubdtype(arr.dtype, jnp.floating) or jnp.issubdtype(
        arr.dtype, jnp.integer
    )
    if not real:
        raise TypeError(f"{where}: {name} must be a real number, got dtype {arr.dtype}")
    if arr.ndim != 0:
        raise ValueError(
            f"{where}: {name} must be a scalar, got shape {arr.shape}; an array would "
            f"broadcast and treat each coordinate differently"
        )
    if not jnp.issubdtype(arr.dtype, jnp.floating):
        arr = arr.astype(jnp.result_type(float))
    return arr


def _names(where: str, names) -> tuple[str, ...] | None:
    """``None``, or a tuple of distinct block names."""
    if names is None:
        return None
    if isinstance(names, str):
        names = (names,)
    if not isinstance(names, Sequence):
        raise TypeError(
            f"{where}: names must be a str or a sequence of str, got "
            f"{type(names).__name__}"
        )
    names = tuple(names)
    for name in names:
        if not isinstance(name, str):
            raise TypeError(
                f"{where}: block names must be str, got {type(name).__name__}"
            )
    if len(set(names)) != len(names):
        raise ValueError(f"{where}: a block name is repeated in {names}")
    if not names:
        raise ValueError(f"{where}: names must name at least one block, or be None")
    return names


def _merge(where: str, mapping, kwargs: dict) -> dict:
    """Merge a positional mapping and keywords into one ordered dict."""
    out: dict = {}
    if mapping is not None:
        if not isinstance(mapping, Mapping):
            raise TypeError(
                f"{where}: the positional argument must be a mapping from block "
                f"name to covariance, got {type(mapping).__name__}"
            )
        for name, value in mapping.items():
            if not isinstance(name, str):
                raise TypeError(
                    f"{where}: block names must be str, got {type(name).__name__}"
                )
            out[name] = value
    for name, value in kwargs.items():
        if name in out:
            raise TypeError(
                f"{where}: block {name!r} is given twice, in the mapping and as a keyword"
            )
        out[name] = value
    return out


def _scalar(name: str, value) -> str:
    """``name=value``, or ``name=...`` for a family's batched scalar."""
    return f"{name}={float(value):g}" if value.ndim == 0 else f"{name}=..."


def _repr(policy, shown: str) -> str:
    names = "" if policy.names is None else f", names={policy.names!r}"
    return f"{type(policy).__name__}({shown}{names})"
