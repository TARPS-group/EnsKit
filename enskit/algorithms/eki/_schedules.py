r"""Schedules and stopping rules: how far each step moves, and when a run ends."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from functools import partial
from typing import Protocol

import jax
import jax.numpy as jnp
from jax import Array, lax

from .. import _common as c
from ._helpers import _ess_from_misfits
from ._values import Evaluation

__all__ = [
    "Schedule",
    "StoppingRule",
    "FixedSchedule",
    "AdaptiveESSSchedule",
    "AdaptiveMisfitSchedule",
    "DiscrepancyStop",
]


#: A remainder of the budget below this fraction of ``beta_target`` is taken
#: with the step before it rather than left for a step of its own: it is the
#: round-off of accumulating many floor-sized increments, and a step for it
#: would cost an evaluation for nothing.
_SNAP_RELATIVE = 1e-9


class Schedule(Protocol):
    """How large the next increment is: one method and two attributes.

    Any object with these members is a schedule; nothing subclasses this
    class. A schedule must be pure: :meth:`next_increment` depends on its
    argument and the schedule's own fields only, so a run resumes from a
    state alone.

    ``n_steps`` is the ladder's length in steps, an ``int``, or ``None`` if it
    is not bounded by a count; ``beta_target`` is the level the ladder ends
    at, a ``float``, or ``None`` for an unbounded ladder.

    Notes
    -----
    The driver, not the schedule, decides that a ladder is finished, before
    each evaluation of the forward model and from the two attributes alone:
    when ``step >= n_steps``, or when
    ``beta >= beta_target * (1 - 1e-12)``. A finished ladder costs no
    evaluation. :meth:`next_increment` may also return ``None``, ending the
    run on evidence only the evaluation carries; that costs the evaluation
    that produced it.
    """

    n_steps: int | None
    beta_target: float | None

    def next_increment(self, evaluation: Evaluation) -> Array | float | None:
        """The next increment, finite and strictly positive, or ``None`` to stop.

        Pure: no trial updates and no trial evaluations of the forward model.
        """


class StoppingRule(Protocol):
    """Whether to stop before the next step: any callable of an evaluation.

    Consulted after each evaluation and before the increment is chosen. It
    must return a Python ``bool`` and be pure, holding no state across steps.
    """

    def __call__(self, evaluation: Evaluation) -> bool:
        """``True`` to end the run before taking another step."""


@c.pytree_class(data=(), meta=("increments",))
@dataclass(frozen=True, eq=False, repr=False)
class FixedSchedule:
    r"""A ladder given in advance: the increments, in order.

    Step :math:`t` takes ``increments[t]``, with :math:`t` the state's
    cumulative ``step``, so a fixed ladder interrupted part-way resumes where
    it stopped. ``n_steps`` is the number of increments and ``beta_target``
    is ``None``.

    Parameters
    ----------
    increments : tuple of float
        At least one, each finite and strictly positive; stored as Python
        floats (static). Increments that sum to exactly 1 in binary floating
        point, such as powers of two, end at the posterior exactly.

    Raises
    ------
    TypeError
        If ``increments`` is not a tuple of real numbers.
    ValueError
        If it is empty, or an increment is not finite and strictly positive.

    Notes
    -----
    The ensemble smoother with multiple data assimilation (ES-MDA) is this
    schedule with ``increments = (1/alpha_1, ..., 1/alpha_T)``, its inflation
    factors' reciprocals.

    References
    ----------
    Emerick, A. A. & Reynolds, A. C. (2013). Ensemble smoother with multiple
    data assimilation. *Computers & Geosciences*, 55, 3–15.
    """

    increments: tuple[float, ...]

    def __post_init__(self) -> None:
        where = "FixedSchedule"
        if not isinstance(self.increments, tuple):
            raise TypeError(
                f"{where}: increments must be a tuple, got "
                f"{type(self.increments).__name__}"
            )
        if not self.increments:
            raise ValueError(f"{where}: increments must not be empty")
        values = []
        for index, increment in enumerate(self.increments):
            if isinstance(increment, bool):
                raise TypeError(f"{where}: increments[{index}] is the bool {increment!r}")
            try:
                value = float(increment)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{where}: increments[{index}] must be a real number, got "
                    f"{increment!r}"
                ) from exc
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{where}: every increment must be finite and strictly positive, "
                    f"got increments[{index}] = {increment!r}"
                )
            values.append(value)
        object.__setattr__(self, "increments", tuple(values))

    @classmethod
    def uniform(cls, n_steps: int) -> FixedSchedule:
        r"""``n_steps`` equal increments of ``1 / n_steps``: a ladder to the posterior.

        The increments sum to 1 only to round-off when ``1 / n_steps`` is not
        exact in binary, and no correction is applied to the last one.
        """
        n_steps = _positive_int("FixedSchedule.uniform", "n_steps", n_steps)
        return cls((1.0 / n_steps,) * n_steps)

    @classmethod
    def constant(cls, increment: float, n_steps: int) -> FixedSchedule:
        """``n_steps`` equal increments of ``increment``.

        ``constant(1.0, n)`` with a stopping rule is the optimization form;
        ``constant(1.0, 1)`` is a single Kalman update.
        """
        n_steps = _positive_int("FixedSchedule.constant", "n_steps", n_steps)
        return cls((increment,) * n_steps)

    @property
    def n_steps(self) -> int:
        """The number of increments."""
        return len(self.increments)

    @property
    def beta_target(self) -> None:
        """``None``: a fixed ladder is bounded by its count."""
        return None

    def next_increment(self, evaluation: Evaluation) -> float:
        """``increments[evaluation.step]``.

        Raises
        ------
        IndexError
            If the step is outside the ladder, which the driver's exhaustion
            check never lets happen.
        """
        step = evaluation.step
        if not 0 <= step < len(self.increments):
            raise IndexError(
                f"{self!r}.next_increment: step {step} is outside a ladder of "
                f"{len(self.increments)} steps"
            )
        return self.increments[step]

    def __repr__(self) -> str:
        """As ``FixedSchedule(n_steps=200, total=200.0)``, never every increment."""
        try:
            total = float(f"{math.fsum(self.increments):.12g}")
            return f"FixedSchedule(n_steps={len(self.increments)}, total={total!r})"
        except Exception:
            return "<FixedSchedule (unprintable)>"


@c.pytree_class(
    data=(),
    meta=("beta_target", "min_increment", "max_increment", "ess_fraction", "n_bisect"),
)
@dataclass(frozen=True, eq=False, repr=False)
class AdaptiveESSSchedule:
    r"""The increment that keeps the tempering weights' ESS at ``ess_fraction * J``.

    For an increment :math:`\delta`, the weights
    :math:`w_j = e^{-\delta \Phi_j}` that would carry the particles from one
    level to the next have effective sample size
    :math:`\mathrm{ESS}(\delta)` (see
    :func:`~enskit.algorithms.eki.effective_sample_size`), and the criterion
    is

    .. math::

        \delta^\star = \sup\{\delta \ge 0 : \mathrm{ESS}(\delta) \ge f J\},

    with :math:`f` = ``ess_fraction``, :math:`\Phi_j` the misfits and
    :math:`J` the number of particles. It is found by bisection and then
    clamped:

    .. math::

        \delta = \min\bigl(\max(\delta^\star, \delta_{\min}),\ \delta_{\max},\
            \beta_{\text{target}} - \beta\bigr),

    the last term present only when ``beta_target`` is not ``None``. The floor
    beats the criterion, so a step is always taken; the budget beats the
    floor, so the ladder never passes ``beta_target``.

    Parameters
    ----------
    beta_target : float or None
        The level the ladder ends at; 1.0, the posterior, by default. ``None``
        for an unbounded ladder, which a stopping rule must end. Positive.
    min_increment, max_increment : float
        :math:`\delta_{\min}` and :math:`\delta_{\max}`; defaults ``1e-3`` and
        ``1.0``. Positive, the first no greater than the second, the second
        finite.
    ess_fraction : float
        :math:`f`, in :math:`(0, 1 - 10^{-6}]`; default 0.5.
    n_bisect : int
        The number of bisections, at least 1; default 50. The returned
        increment is within :math:`2^{-n}` of the bracket's top of the
        criterion.

    Raises
    ------
    ValueError
        If a field is outside its domain.

    Notes
    -----
    :math:`\mathrm{ESS}` is non-increasing in :math:`\delta` with zero
    derivative at :math:`\delta = 0`, where a derivative-based root finder
    would stall, so the method is bisection, on
    :math:`[0, \min(\delta_{\max}, \beta_{\text{target}} - \beta)]`,
    returning the end that meets the target. When every misfit is equal no
    increment changes the weights, and the largest allowed step is taken.
    The construction is the adaptive tempering of sequential Monte Carlo,
    used here only to choose a step: no weights are carried and nothing is
    resampled.

    References
    ----------
    Jasra, A., Stephens, D. A., Doucet, A. & Tsagaris, T. (2011). Inference
    for Lévy-driven stochastic volatility models via adaptive sequential Monte
    Carlo. *Scandinavian Journal of Statistics*, 38(1), 1–22.
    """

    beta_target: float | None = 1.0
    min_increment: float = 1e-3
    max_increment: float = 1.0
    ess_fraction: float = 0.5
    n_bisect: int = 50

    def __post_init__(self) -> None:
        _check_adaptive_fields(self)
        fraction = self.ess_fraction
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
            raise ValueError(
                f"AdaptiveESSSchedule: ess_fraction must be a real number, got "
                f"{fraction!r}"
            )
        if not 0.0 < fraction <= 1.0 - 1e-6:
            raise ValueError(
                f"AdaptiveESSSchedule: ess_fraction must lie in (0, 1 - 1e-6], got "
                f"{fraction}. It is bounded away from 1 because ESS(0) evaluates to "
                f"exp(log J), not exactly J."
            )
        object.__setattr__(self, "ess_fraction", float(fraction))
        if type(self.n_bisect) is not int or self.n_bisect < 1:
            raise ValueError(
                f"AdaptiveESSSchedule: n_bisect must be a Python int of at least 1, "
                f"got {self.n_bisect!r}"
            )

    @property
    def n_steps(self) -> None:
        """``None``: an adaptive ladder is bounded by its level."""
        return None

    def next_increment(self, evaluation: Evaluation) -> Array:
        """The bisected increment, clamped by the floor, the ceiling and the budget."""
        delta_hi = _bracket_top(self, evaluation.beta)
        target = self.ess_fraction * evaluation.n_particles
        unclamped = _bisect_ess(evaluation.misfits, delta_hi, target, self.n_bisect)
        return _clamp(self, unclamped, evaluation.beta)

    def __repr__(self) -> str:
        """The fields, as ``AdaptiveESSSchedule(beta_target=1.0, ...)``."""
        return _policy_repr(self)


@c.pytree_class(
    data=(),
    meta=("beta_target", "min_increment", "max_increment", "divergence_budget"),
)
@dataclass(frozen=True, eq=False, repr=False)
class AdaptiveMisfitSchedule:
    r"""The increment from the misfits' mean and spread: the data misfit controller.

    With :math:`\overline\Phi` and :math:`\sigma^2_\Phi` the mean and the
    sample variance (divisor :math:`J - 1`) of the particles' misfits,

    .. math::

        \delta^\star = \max\Big(\frac{\theta}{\overline{\Phi}},\
            \sqrt{\frac{\theta}{\sigma^2_\Phi}}\Big),

    with :math:`\theta` = ``divergence_budget``, :math:`N/2` by default for
    data dimension :math:`N`, and then clamped as in
    :class:`AdaptiveESSSchedule`. A vanishing denominator gives
    :math:`+\infty`, so the largest allowed step; a ``nan`` misfit gives
    ``nan``, which the driver refuses as an increment.

    Parameters
    ----------
    beta_target : float or None
        As in :class:`AdaptiveESSSchedule`; default 1.0.
    min_increment, max_increment : float
        As in :class:`AdaptiveESSSchedule`.
    divergence_budget : float or None
        :math:`\theta`, the bound on the Jeffreys divergence between
        consecutive tempered targets; ``None`` for :math:`N/2`, at which the
        schedule has no tuning parameter. Finite and positive when given.

    Raises
    ------
    ValueError
        If a field is outside its domain.

    Notes
    -----
    The criterion is the larger of the two bounds, not the smaller: each
    approximates the divergence in a different regime, and the bound of the
    valid regime is the larger. The mean bound applies when the misfits'
    coefficient of variation exceeds :math:`1/\sqrt\theta`, the common case.

    This takes far longer steps than :class:`AdaptiveESSSchedule`: at the
    increment it chooses, the tempering weights' effective sample size is near
    its floor of 1. Prefer it when the fit, rather than the ensemble, is what
    the run is for, or when evaluations of the forward model are scarce.

    References
    ----------
    Iglesias, M. & Yang, Y. (2021). Adaptive regularisation for ensemble
    Kalman inversion. *Inverse Problems*, 37(2), 025008.
    """

    beta_target: float | None = 1.0
    min_increment: float = 1e-3
    max_increment: float = 1.0
    divergence_budget: float | None = None

    def __post_init__(self) -> None:
        _check_adaptive_fields(self)
        budget = self.divergence_budget
        if budget is not None:
            if not isinstance(budget, (int, float)) or isinstance(budget, bool):
                raise ValueError(
                    f"AdaptiveMisfitSchedule: divergence_budget must be None or a "
                    f"real number, got {budget!r}"
                )
            if not math.isfinite(budget) or budget <= 0.0:
                raise ValueError(
                    f"AdaptiveMisfitSchedule: divergence_budget must be finite and "
                    f"strictly positive, got {budget}"
                )
            object.__setattr__(self, "divergence_budget", float(budget))

    @property
    def n_steps(self) -> None:
        """``None``: an adaptive ladder is bounded by its level."""
        return None

    def next_increment(self, evaluation: Evaluation) -> Array:
        """The closed-form increment, clamped by the floor, the ceiling and the budget."""
        theta = (
            0.5 * evaluation.data_dim
            if self.divergence_budget is None
            else self.divergence_budget
        )
        return _clamp(
            self, _misfit_criterion(evaluation.misfits, theta), evaluation.beta
        )

    def __repr__(self) -> str:
        """The fields, as ``AdaptiveMisfitSchedule(beta_target=1.0, ...)``."""
        return _policy_repr(self)


@c.pytree_class(data=(), meta=("tau",))
@dataclass(frozen=True, eq=False, repr=False)
class DiscrepancyStop:
    r"""Stop by the discrepancy principle, once the mean prediction fits to the noise.

    Fires when

    .. math::

        2\,\Phi(\bar g) \le \tau^2 N,

    with :math:`\bar g` the mean prediction, :math:`\Phi(\bar g)` the
    evaluation's ``center_misfit`` and :math:`N` the data dimension.

    Parameters
    ----------
    tau : float
        :math:`\tau > 0`; default 1.0. At the true parameters :math:`2\Phi` is
        a :math:`\chi^2_N` variate, of mean :math:`N` and standard deviation
        :math:`\sqrt{2N}`, so :math:`\tau^2 = 1 + k\sqrt{2/N}` sets the
        threshold :math:`k` standard deviations above the mean.

    Raises
    ------
    ValueError
        If ``tau`` is not finite and strictly positive.

    Notes
    -----
    The residual is that of the mean *prediction*, not of the prediction at
    the mean parameters, which would cost another evaluation of the forward
    model. Paired with a schedule that has a ``beta_target``, it can end a
    sampling run at an intermediate level; :attr:`EKIResult.stop_fired
    <enskit.algorithms.eki.EKIResult.stop_fired>` and
    :attr:`~enskit.algorithms.eki.EKIResult.budget_complete` tell the cases
    apart.

    References
    ----------
    Iglesias, M. A. (2016). A regularizing iterative ensemble Kalman method for
    PDE-constrained inverse problems. *Inverse Problems*, 32(2), 025002.
    """

    tau: float = 1.0

    def __post_init__(self) -> None:
        tau = self.tau
        if not isinstance(tau, (int, float)) or isinstance(tau, bool):
            raise ValueError(f"DiscrepancyStop: tau must be a real number, got {tau!r}")
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError(
                f"DiscrepancyStop: tau must be finite and strictly positive, got {tau}"
            )
        object.__setattr__(self, "tau", float(tau))

    def __call__(self, evaluation: Evaluation) -> bool:
        r"""Whether :math:`2\Phi(\bar g) \le \tau^2 N`."""
        return bool(2.0 * evaluation.center_misfit <= self.tau**2 * evaluation.data_dim)

    def __repr__(self) -> str:
        """As ``DiscrepancyStop(tau=1.0)``."""
        return _policy_repr(self)


# ---------------------------------------------------------------------------
# private
# ---------------------------------------------------------------------------


def _bracket_top(schedule, beta: Array) -> Array:
    """The largest increment the ceiling and the budget allow."""
    ceiling = jnp.asarray(schedule.max_increment, dtype=beta.dtype)
    if schedule.beta_target is None:
        return ceiling
    return jnp.minimum(ceiling, schedule.beta_target - beta)


def _clamp(schedule, unclamped: Array, beta: Array) -> Array:
    """Floor, then ceiling, then budget, in that order.

    The floor beats the criterion, so a step is always taken; the budget beats
    the floor, so the ladder cannot overshoot ``beta_target``. The budget term
    is present only when there is a budget. A step that would leave less than
    ``_SNAP_RELATIVE * beta_target`` of the budget takes the rest of it, so
    accumulated round-off never costs a step of its own.
    """
    delta = jnp.maximum(unclamped, schedule.min_increment)
    delta = jnp.minimum(delta, schedule.max_increment)
    if schedule.beta_target is not None:
        remaining = schedule.beta_target - beta
        delta = jnp.minimum(delta, remaining)
        snap = _SNAP_RELATIVE * schedule.beta_target
        delta = jnp.where(remaining - delta <= snap, remaining, delta)
    return delta


@partial(jax.jit, static_argnums=(3,))
def _bisect_ess(misfits: Array, delta_hi: Array, target, n_bisect: int) -> Array:
    """Bisect ``[0, delta_hi]`` for the largest increment meeting the target.

    When the bracket's top already meets the target, ``lo`` starts there and
    the loop runs unchanged, so a degenerate ensemble takes the largest
    allowed step exactly rather than to within ``2**-n_bisect``. A ``nan``
    effective sample size propagates, so a poisoned ensemble reaches the
    driver's increment check instead of taking the floor step.
    """
    ess_hi = _ess_from_misfits(misfits, delta_hi)
    lo = jnp.where(ess_hi >= target, delta_hi, jnp.zeros_like(delta_hi))

    def body(_, bracket):
        low, high = bracket
        mid = 0.5 * (low + high)
        ok = _ess_from_misfits(misfits, mid) >= target
        return jnp.where(ok, mid, low), jnp.where(ok, high, mid)

    low, _ = lax.fori_loop(0, n_bisect, body, (lo, delta_hi))
    return jnp.where(jnp.isnan(ess_hi), jnp.nan, low)


@jax.jit
def _misfit_criterion(misfits: Array, theta) -> Array:
    """The larger of the mean and variance bounds, both guarded."""
    mean = jnp.mean(misfits)
    variance = jnp.var(misfits, ddof=1)
    return jnp.maximum(
        _guarded_ratio(theta, mean), jnp.sqrt(_guarded_ratio(theta, variance))
    )


def _guarded_ratio(theta, denominator: Array) -> Array:
    """``theta / denominator``: ``inf`` at zero, ``nan`` at ``nan`` or below zero.

    The inner ``where`` keeps ``theta / 0`` from being formed, whose
    derivative would be ``nan``. The outer one keeps a ``nan`` denominator
    ``nan``: ``nan > 0`` is ``False``, so a single fallback to ``inf`` would
    turn a poisoned misfit into the largest allowed step.
    """
    positive = denominator > 0
    safe = jnp.where(positive, denominator, 1.0)
    return jnp.where(
        positive, theta / safe, jnp.where(denominator == 0, jnp.inf, jnp.nan)
    )


def _check_adaptive_fields(schedule) -> None:
    """Validate the three fields both adaptive schedules share."""
    name = type(schedule).__name__
    target = schedule.beta_target
    if target is not None:
        if not isinstance(target, (int, float)) or isinstance(target, bool):
            raise ValueError(
                f"{name}: beta_target must be None or a real number, got {target!r}"
            )
        if not math.isfinite(target) or target <= 0.0:
            raise ValueError(
                f"{name}: beta_target must be finite and strictly positive when "
                f"given, got {target}"
            )
        object.__setattr__(schedule, "beta_target", float(target))
    for field_name in ("min_increment", "max_increment"):
        value = getattr(schedule, field_name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name}: {field_name} must be a real number, got {value!r}")
        object.__setattr__(schedule, field_name, float(value))
    if not math.isfinite(schedule.max_increment):
        raise ValueError(
            f"{name}: max_increment must be finite, got {schedule.max_increment}. An "
            f"unbounded ladder with an infinite ceiling has no upper clamp."
        )
    if not schedule.min_increment > 0.0:
        raise ValueError(
            f"{name}: min_increment must be strictly positive, got "
            f"{schedule.min_increment}"
        )
    if schedule.min_increment > schedule.max_increment:
        raise ValueError(
            f"{name}: min_increment {schedule.min_increment} exceeds max_increment "
            f"{schedule.max_increment}"
        )


def _positive_int(where: str, name: str, value) -> int:
    """A positive Python ``int`` that sets a static ladder length."""
    if type(value) is not int:
        raise TypeError(
            f"{where}: {name} must be a Python int, got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"{where}: {name} must be at least 1, got {value}")
    return value


def _policy_repr(policy) -> str:
    """Type name and fields, in declaration order; never raises."""
    try:
        shown = ", ".join(
            f"{f.name}={getattr(policy, f.name)!r}" for f in dataclasses.fields(policy)
        )
        return f"{type(policy).__name__}({shown})"
    except Exception:
        return f"<{type(policy).__name__} (unprintable)>"
