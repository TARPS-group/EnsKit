r"""The value classes of a run, and the error a run raises."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from ...distribution import Ensemble
from ...linalg import value_check
from .. import _common as c
from ._helpers import _ess_from_misfits, _misfits_from_residuals

__all__ = [
    "PREDICTION",
    "SCHEDULE_EXHAUSTED",
    "STOPPING_RULE",
    "INTERRUPTED",
    "EKIState",
    "Evaluation",
    "HistoryRecord",
    "EKIResult",
    "EKIError",
]

#: The name of the block the forward model's output becomes. A state's
#: ensemble may not use it.
PREDICTION = "prediction"

#: The ladder finished, by the schedule's attributes or by ``next_increment``
#: returning ``None``.
SCHEDULE_EXHAUSTED = "schedule_exhausted"

#: The stopping rule fired; the last record is terminal.
STOPPING_RULE = "stopping_rule"

#: The run was ended by its caller, not by a policy. Never produced by
#: :func:`~enskit.algorithms.eki.run`.
INTERRUPTED = "interrupted"

Status = Literal["schedule_exhausted", "stopping_rule", "interrupted"]

_STATUSES = (SCHEDULE_EXHAUSTED, STOPPING_RULE, INTERRUPTED)

#: The declaration order of :class:`HistoryRecord`'s fields.
_RECORD_FIELDS = (
    "step",
    "n_valid",
    "beta",
    "increment",
    "beta_next",
    "misfit_mean",
    "misfit_min",
    "misfit_max",
    "center_misfit",
    "spread",
    "ess",
)


@c.pytree_class(data=("ensemble", "beta", "key"), meta=("step",))
class EKIState:
    r"""The state carried from step to step: everything a run needs to continue.

    A state is the unit of checkpointing: :func:`~enskit.algorithms.eki.run`
    on a state returned by an earlier run continues that run, and the rest of
    it is bit-identical to an uninterrupted one.

    Parameters
    ----------
    ensemble : Ensemble
        The particles, an unweighted :class:`~enskit.distribution.Ensemble`
        whose blocks are the parameters. It may not have a block named
        :data:`PREDICTION`.
    key : jax.random key
        Keyword-only and required. A typed key of shape ``()``, the run's only
        source of randomness; split once per step into
        ``(next, inflate, evaluate, update)``.
    beta : float or Array
        Keyword-only. The tempering level :math:`\beta \ge 0` reached; 0, the
        prior, by default. Stored as a 0-d floating array.
    step : int
        Keyword-only. The number of steps completed, a non-negative Python
        ``int`` (static). It counts across resumed runs, and position-dependent
        schedules read it.

    Raises
    ------
    TypeError
        If ``ensemble`` is not an :class:`~enskit.distribution.Ensemble`,
        ``key`` is not a typed key, or ``step`` is not a Python ``int``.
    ValueError
        If ``ensemble`` is weighted, a vmapped family, or has a block named
        :data:`PREDICTION`; if ``beta`` is not a scalar; or if ``step`` is
        negative. In debug mode, also if a particle is not finite or ``beta``
        is negative or not finite.

    Notes
    -----
    **``step`` is cumulative across runs.** That is what lets a ten-step
    :class:`~enskit.algorithms.eki.FixedSchedule` interrupted after four steps
    resume at its fifth increment. It also makes chaining a *new* ladder onto
    a finished state a no-op: the new schedule finds itself exhausted, and the
    run returns at once. :meth:`restart` resets the counters for that.
    """

    ensemble: Ensemble
    beta: Array
    key: Array
    step: int

    def __init__(self, ensemble, *, key, beta=0.0, step: int = 0) -> None:
        where = "EKIState"
        _check_parameter_ensemble(where, ensemble)
        if type(step) is not int:
            raise TypeError(
                f"{where}: step must be a Python int, got {type(step).__name__}. It "
                f"is static: a schedule indexes a tuple with it."
            )
        if step < 0:
            raise ValueError(f"{where}: step must not be negative, got {step}")
        c.check_key(where, key)
        c.set_fields(
            self,
            ensemble=ensemble,
            beta=_as_level(where, beta),
            key=key,
            step=step,
        )
        first = ensemble[ensemble.names[0]]
        value_check(
            first,
            lambda _: bool(jnp.all(ensemble.all_finite)),
            f"{where}: every particle must be finite.",
        )

    @classmethod
    def from_prior(cls, key, prior, n_particles: int) -> EKIState:
        """Draw the initial particles from a prior; ``beta = 0``, ``step = 0``.

        The draw is pinned as

        .. code-block:: python

            key_sample, key_state = jax.random.split(key)
            EKIState(prior.sample(key_sample, n_particles), key=key_state)

        so the particles are :meth:`~enskit.distribution.Gaussian.sample`'s
        draw, and the state's own key is independent of it.

        Parameters
        ----------
        key : jax.random key
            A typed key, consumed whole.
        prior : Gaussian
            A :class:`~enskit.distribution.Gaussian` over the parameter
            blocks.
        n_particles : int
            :math:`J \\ge 2`.

        Returns
        -------
        EKIState

        Raises
        ------
        UnsupportedOpError
            If a block's covariance cannot be sampled (``factor``), from the
            operator layer, unmodified.
        """
        c.check_key("EKIState.from_prior", key)
        key_sample, key_state = jax.random.split(key)
        return cls(prior.sample(key_sample, n_particles), key=key_state)

    def restart(self) -> EKIState:
        """The same particles and key at ``beta = 0`` and ``step = 0``.

        For chaining a fresh ladder onto a finished run, which otherwise
        returns at once: a step-bounded schedule is exhausted by ``step`` and
        a budgeted one by ``beta``, so this resets both.
        """
        c.guard(self, f"{self!r}.restart")
        return EKIState(self.ensemble, key=self.key)

    @property
    def n_particles(self) -> int:
        """The number of particles :math:`J`."""
        return self.ensemble.n_particles

    @property
    def dims(self) -> dict[str, int]:
        """Each parameter block's dimension, keyed by name, in order."""
        return self.ensemble.dims

    def mean(self, name: str) -> Array:
        """The particles' mean of block ``name``, ``(d,)``."""
        c.guard(self, f"{self!r}.mean")
        return self.ensemble.mean(name)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The family's batch shape, ``()`` for a directly constructed object."""
        return c.broadcast_batch(
            "EKIState",
            tuple(self.ensemble.batch_shape),
            tuple(self.beta.shape),
            tuple(self.key.shape),
        )

    def __repr__(self) -> str:
        """As ``EKIState(n_particles=64, blocks={'u': 2}, step=3)``; never raises."""
        return c.safe_repr(
            lambda: (
                f"EKIState(n_particles={self.n_particles}, blocks={self.dims}, "
                f"step={self.step})"
            ),
            "EKIState",
            lambda: self.batch_shape,
        )


@c.pytree_class(
    data=("beta", "ensemble", "whitened_residuals", "n_valid"), meta=("step",)
)
class Evaluation:
    r"""What one evaluation of the forward model produced, and its summaries.

    What a schedule and a stopping rule are shown, what
    :func:`~enskit.algorithms.eki.assimilate` consumes, and what a run reports
    as its ``last_evaluation``. Returned by
    :func:`~enskit.algorithms.eki.evaluate`.

    Parameters
    ----------
    step : int
        Keyword-only. The index of the step, a non-negative Python ``int``.
    beta : float or Array
        Keyword-only. The level entering the step.
    ensemble : Ensemble
        Keyword-only. The particles that were evaluated, after inflation and
        after repair, with the forward model's output appended as the block
        :data:`PREDICTION`. Unweighted, with at least one other block.
    whitened_residuals : Array
        Keyword-only. ``(J, N)``: row :math:`j` is :math:`b_j = W(y - g_j)`,
        with :math:`g_j` particle :math:`j`'s prediction and :math:`W` a
        whitener of the base noise covariance.
    n_valid : int or Array
        Keyword-only. How many particles' predictions were finite, an integer
        in :math:`[2, J]`, stored as a 0-d array.

    Raises
    ------
    TypeError
        If ``step`` is not a Python ``int``, ``ensemble`` not an
        :class:`~enskit.distribution.Ensemble`, or ``n_valid`` not an integer.
    ValueError
        If ``ensemble`` is weighted or lacks a :data:`PREDICTION` block or a
        parameter block, ``whitened_residuals`` is not ``(J, N)``, or a field
        has the wrong rank. In debug mode, also if ``n_valid`` is outside
        :math:`[2, J]`.

    Notes
    -----
    The whitened residuals determine three quantities:

    .. math::

        \Phi_j = \tfrac12\lVert b_j\rVert^2, \qquad
        \Phi(\bar g) = \tfrac12\lVert \bar b\rVert^2, \qquad
        b_j - \bar b = -W(g_j - \bar g),

    the misfits, the misfit of the mean prediction, and the whitened
    prediction anomalies, with :math:`\bar b` and :math:`\bar g` means over
    the particles. The last is a diagnostic, never a substitute for the
    update's own whitening, which centers before it whitens.
    """

    step: int
    beta: Array
    ensemble: Ensemble
    whitened_residuals: Array
    n_valid: Array

    def __init__(self, *, step, beta, ensemble, whitened_residuals, n_valid) -> None:
        where = "Evaluation"
        if type(step) is not int:
            raise TypeError(
                f"{where}: step must be a Python int, got {type(step).__name__}"
            )
        if step < 0:
            raise ValueError(f"{where}: step must not be negative, got {step}")
        if not isinstance(ensemble, Ensemble):
            raise TypeError(
                f"{where}: ensemble must be an enskit.distribution.Ensemble, got "
                f"{type(ensemble).__name__}"
            )
        if ensemble.batch_shape != ():
            raise ValueError(f"{where}: {ensemble!r} is a vmapped family")
        if ensemble.is_weighted:
            raise ValueError(f"{where}: the ensemble must be unweighted")
        if PREDICTION not in ensemble.names or len(ensemble.names) < 2:
            raise ValueError(
                f"{where}: the ensemble must hold the block {PREDICTION!r} and at "
                f"least one parameter block, got blocks {ensemble.names}"
            )
        residuals = jnp.asarray(whitened_residuals)
        shape = (ensemble.n_particles, ensemble.dims[PREDICTION])
        if residuals.shape != shape:
            raise ValueError(
                f"{where}: whitened_residuals must have shape {shape}, one row per "
                f"particle and one column per datum, got {residuals.shape}"
            )
        n_valid = jnp.asarray(n_valid)
        if n_valid.ndim != 0 or not jnp.issubdtype(n_valid.dtype, jnp.integer):
            raise TypeError(
                f"{where}: n_valid must be a 0-d integer, got shape {n_valid.shape} "
                f"of dtype {n_valid.dtype}"
            )
        J = ensemble.n_particles
        value_check(
            n_valid,
            lambda n: bool(2 <= n <= J),
            f"{where}: n_valid must lie in [2, {J}].",
        )
        c.set_fields(
            self,
            step=step,
            beta=_as_level(where, beta),
            ensemble=ensemble,
            whitened_residuals=residuals,
            n_valid=n_valid,
        )

    @property
    def misfits(self) -> Array:
        r"""The particles' misfits :math:`\Phi_j = \tfrac12\lVert b_j\rVert^2`, ``(J,)``.

        The quantity :func:`~enskit.algorithms.eki.misfits` computes from the
        data and the predictions.
        """
        c.guard(self, f"{self!r}.misfits")
        return _misfits_from_residuals(self.whitened_residuals)

    @property
    def center_misfit(self) -> Array:
        r"""The misfit of the mean prediction, :math:`\Phi(\bar g)`, 0-d.

        Not the mean of :attr:`misfits`: the two differ by half the whitened
        prediction spread,

        .. math::

            \overline{\Phi_j} = \Phi(\bar g)
            + \tfrac{J-1}{2J}\operatorname{tr}\bigl(W \widehat C_{gg} W^\top\bigr),

        with :math:`\widehat C_{gg}` the predictions' sample covariance
        (divisor :math:`J - 1`). A discrepancy principle asks about the
        center; a tempering criterion about the particles.
        """
        c.guard(self, f"{self!r}.center_misfit")
        return _misfits_from_residuals(jnp.mean(self.whitened_residuals, axis=-2))

    @property
    def rms_parameter_spread(self) -> Array:
        r"""The root-mean-square per-coordinate spread of the parameters, 0-d.

        .. math::

            \Big(\frac{1}{(J-1)P}\sum_b \lVert A_b\rVert_F^2\Big)^{1/2},

        over the parameter blocks :math:`b`, with :math:`A_b` a block's
        ``(J, d_b)`` anomalies and :math:`P = \sum_b d_b`. It depends on the
        parameters' units, so it is dominated by the largest-magnitude
        coordinate when they differ.
        """
        c.guard(self, f"{self!r}.rms_parameter_spread")
        names = [n for n in self.ensemble.names if n != PREDICTION]
        total = sum(jnp.sum(self.ensemble.anomalies(n) ** 2) for n in names)
        P = sum(self.ensemble.dims[n] for n in names)
        return jnp.sqrt(total / ((self.n_particles - 1) * P))

    @property
    def n_particles(self) -> int:
        """The number of particles :math:`J`."""
        return self.ensemble.n_particles

    @property
    def data_dim(self) -> int:
        """The data dimension :math:`N`."""
        return int(self.whitened_residuals.shape[-1])

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The family's batch shape, ``()`` for a directly constructed object."""
        return c.broadcast_batch(
            "Evaluation",
            tuple(self.beta.shape),
            tuple(self.ensemble.batch_shape),
            tuple(self.whitened_residuals.shape[:-2]),
            tuple(self.n_valid.shape),
        )

    def __repr__(self) -> str:
        """As ``Evaluation(step=3, n_particles=64)``; never raises."""
        return c.safe_repr(
            lambda: f"Evaluation(step={self.step}, n_particles={self.n_particles})",
            "Evaluation",
            lambda: self.batch_shape,
        )


@c.pytree_class(data=_RECORD_FIELDS, meta=())
class HistoryRecord:
    r"""The record of one step: where it started, what it did, and its summaries.

    Every field is a 0-d array, so the records of a run stack into arrays
    (:attr:`EKIResult.stacked`). Build one from an :class:`Evaluation` with
    :meth:`from_evaluation`, as the driver does.

    Parameters
    ----------
    step, n_valid : Array
        Keyword-only. The step's index and its count of valid particles, 0-d
        integers.
    beta, increment, beta_next : Array
        Keyword-only. The level entering the step, the increment taken, and
        their sum. A *terminal* record, of an evaluation after which the run
        ended, has ``increment`` exactly 0.
    misfit_mean, misfit_min, misfit_max, center_misfit : Array
        Keyword-only. Summaries of the evaluation's misfits.
    spread : Array
        Keyword-only. The evaluation's
        :attr:`~Evaluation.rms_parameter_spread`.
    ess : Array
        Keyword-only. The effective sample size of the tempering weights at
        the increment taken; :math:`J` on a terminal record.

    Raises
    ------
    ValueError
        If a field is not a 0-d array.

    Notes
    -----
    No field is static: two records with different static fields would be
    different pytree types, and the history would not stack.
    """

    def __init__(self, **fields) -> None:
        where = "HistoryRecord"
        missing = [n for n in _RECORD_FIELDS if n not in fields]
        extra = [n for n in fields if n not in _RECORD_FIELDS]
        if missing or extra:
            raise TypeError(
                f"{where}: expected exactly the keywords {_RECORD_FIELDS}; missing "
                f"{missing}, unexpected {extra}"
            )
        for name in _RECORD_FIELDS:
            value = jnp.asarray(fields[name])
            if value.ndim != 0:
                raise ValueError(
                    f"{where}: {name} must be a 0-d array, got shape {value.shape}"
                )
            object.__setattr__(self, name, value)

    @classmethod
    def from_evaluation(cls, evaluation: Evaluation, increment=None) -> HistoryRecord:
        """The record of a step that took ``increment`` from ``evaluation``.

        Parameters
        ----------
        evaluation : Evaluation
            The step's evaluation, the record's single source of truth.
        increment : float or Array, optional
            The increment taken. ``None`` gives the *terminal* record of an
            evaluation after which the run ended: ``increment`` exactly 0,
            ``beta_next == beta``, and ``ess`` exactly :math:`J`.

        Returns
        -------
        HistoryRecord
        """
        if not isinstance(evaluation, Evaluation):
            raise TypeError(
                f"HistoryRecord.from_evaluation: expected an Evaluation, got "
                f"{type(evaluation).__name__}"
            )
        c.guard(evaluation, "HistoryRecord.from_evaluation")
        misfits = evaluation.misfits
        beta = evaluation.beta
        if increment is None:
            increment = jnp.zeros_like(beta)
            ess = jnp.asarray(float(evaluation.n_particles), dtype=beta.dtype)
        else:
            increment = jnp.asarray(increment, dtype=beta.dtype)
            ess = _ess_from_misfits(misfits, increment)
        return cls(
            step=jnp.asarray(evaluation.step),
            n_valid=evaluation.n_valid,
            beta=beta,
            increment=increment,
            beta_next=beta + increment,
            misfit_mean=jnp.mean(misfits),
            misfit_min=jnp.min(misfits),
            misfit_max=jnp.max(misfits),
            center_misfit=evaluation.center_misfit,
            spread=evaluation.rms_parameter_spread,
            ess=ess,
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The family's batch shape, ``()`` for a directly constructed object."""
        return c.broadcast_batch(
            "HistoryRecord", *[tuple(getattr(self, n).shape) for n in _RECORD_FIELDS]
        )

    def __repr__(self) -> str:
        """As ``HistoryRecord(step=3)``; never raises."""
        return c.safe_repr(
            lambda: (
                "HistoryRecord"
                if self.batch_shape != ()
                else f"HistoryRecord(step={int(self.step)})"
            ),
            "HistoryRecord",
            lambda: self.batch_shape,
        )


@dataclass(frozen=True, eq=False, repr=False)
class EKIResult:
    r"""What a run produced, and why it ended. A frozen dataclass, not a pytree.

    Parameters
    ----------
    state : EKIState
        Keyword-only. The final state.
    history : tuple of HistoryRecord
        Keyword-only. One record per evaluation of the forward model.
    status : str
        Keyword-only. :data:`SCHEDULE_EXHAUSTED`, :data:`STOPPING_RULE`, or
        :data:`INTERRUPTED` for a run ended by its caller.
    last_evaluation : Evaluation or None
        Keyword-only. The final evaluation, or ``None`` if the run made none.

    Raises
    ------
    ValueError
        If ``status`` is not one of the three.
    TypeError
        If a field has the wrong type.

    Notes
    -----
    **The returned ensemble has never been evaluated** on a run that ended
    with an exhausted schedule: the last update produced it and the run then
    ended, so ``last_evaluation.ensemble`` holds the particles *before* that
    update. On a :data:`STOPPING_RULE` run the state is unchanged after the
    last evaluation, so they hold the same particles.
    """

    state: EKIState = field(kw_only=True)
    history: tuple[HistoryRecord, ...] = field(kw_only=True)
    status: Status = field(kw_only=True)
    last_evaluation: Evaluation | None = field(kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.state, EKIState):
            raise TypeError(
                f"EKIResult.state: must be an EKIState, got {type(self.state).__name__}"
            )
        if not isinstance(self.history, tuple) or not all(
            isinstance(record, HistoryRecord) for record in self.history
        ):
            raise TypeError("EKIResult.history: must be a tuple of HistoryRecord")
        if self.status not in _STATUSES:
            raise ValueError(
                f"EKIResult.status: must be one of {_STATUSES}, got {self.status!r}. "
                f"Compare against the exported constants rather than a literal."
            )
        if self.last_evaluation is not None and not isinstance(
            self.last_evaluation, Evaluation
        ):
            raise TypeError(
                f"EKIResult.last_evaluation: must be an Evaluation or None, got "
                f"{type(self.last_evaluation).__name__}"
            )

    @property
    def ensemble(self) -> Ensemble:
        """The final particles, ``state.ensemble``."""
        return self.state.ensemble

    @property
    def beta(self) -> Array:
        """The level reached, ``state.beta``."""
        return self.state.beta

    def mean(self, name: str) -> Array:
        """The final particles' mean of block ``name``, ``(d,)``."""
        return self.state.mean(name)

    @property
    def n_evaluations(self) -> int:
        """How many times this run called the forward model: one per record.

        Each call evaluates every particle, so the run's cost in particle
        evaluations is :math:`J` times this.
        """
        return len(self.history)

    @property
    def n_completed_steps(self) -> int:
        """How many steps this run completed, that is, updates it made.

        :attr:`n_evaluations`, or one less when the run ended on a terminal
        evaluation (a stopping rule fired, or a schedule returned ``None``).
        A terminal record is the one whose ``increment`` is exactly 0; there
        is at most one, and it is last.
        """
        if not self.history:
            return 0
        return len(self.history) - int(float(self.history[-1].increment) == 0.0)

    @property
    def min_n_valid(self) -> int | None:
        """The smallest ``n_valid`` over the history, or ``None`` if it is empty."""
        if not self.history:
            return None
        return min(int(record.n_valid) for record in self.history)

    @property
    def stacked(self) -> HistoryRecord:
        """The history as one record whose fields are ``(T,)`` arrays.

        A vmapped family of records, for plotting:
        ``plt.plot(result.stacked.beta, result.stacked.misfit_mean)``. An
        empty history gives ``(0,)`` fields rather than raising.
        """
        if not self.history:
            return jax.tree.map(
                lambda x: jnp.zeros((0, *x.shape), x.dtype), _zero_record()
            )
        return jax.tree.map(lambda *xs: jnp.stack(xs), *self.history)

    @property
    def stop_fired(self) -> bool:
        """Whether the run ended because its stopping rule fired."""
        return self.status == STOPPING_RULE

    @property
    def budget_complete(self) -> bool:
        """Whether the run ended because its schedule was exhausted.

        Together with :attr:`stop_fired` this tells a finished sampling ladder
        from one a stopping rule ended early, at an intermediate level.
        """
        return self.status == SCHEDULE_EXHAUSTED

    def __repr__(self) -> str:
        """As ``EKIResult(status='schedule_exhausted', n_evaluations=17, beta=1)``."""
        try:
            return (
                f"EKIResult(status={self.status!r}, "
                f"n_evaluations={self.n_evaluations}, beta={float(self.beta):g})"
            )
        except Exception:
            return "<EKIResult (unprintable)>"


class EKIError(RuntimeError):
    """A run cannot continue.

    Raised when ``max_steps`` is exceeded, fewer than two particles are
    valid, a particle failed under ``on_failure="raise"``, or an update
    returned a non-finite particle.

    Attributes
    ----------
    state : EKIState
        The last good state, set on every raise.
    history : tuple of HistoryRecord
        The records accumulated before the failure.

    Notes
    -----
    The attributes make a caught error resumable::

        try:
            result = eki.run(state, forward, y, noise_cov, update_rule=rule,
                             schedule=schedule)
        except eki.EKIError as exc:
            checkpoint(exc.state)        # eki.run(exc.state, ...) continues exactly
    """

    def __init__(self, message: str, *, state=None, history=()) -> None:
        super().__init__(message)
        self.state = state
        self.history = tuple(history)


# ---------------------------------------------------------------------------
# private
# ---------------------------------------------------------------------------


def _check_parameter_ensemble(where: str, ensemble) -> None:
    """The particles a state may carry: an unweighted ensemble of parameters."""
    if not isinstance(ensemble, Ensemble):
        raise TypeError(
            f"{where}: ensemble must be an enskit.distribution.Ensemble, got "
            f"{type(ensemble).__name__}. Wrap an array as Ensemble(u=array)."
        )
    if ensemble.batch_shape != ():
        raise ValueError(
            f"{where}: {ensemble!r} is a vmapped family; a state holds one ensemble"
        )
    if ensemble.is_weighted:
        raise ValueError(
            f"{where}: the ensemble is weighted; an update takes unweighted "
            f"particles. Resample first, with enskit.distribution.resample."
        )
    if PREDICTION in ensemble.names:
        raise ValueError(
            f"{where}: the block name {PREDICTION!r} is reserved for the forward "
            f"model's output; rename the parameter block"
        )


def _zero_record() -> HistoryRecord:
    """A record of zeros, the prototype an empty ``stacked`` maps over."""
    zero_int = jnp.zeros((), dtype=jnp.result_type(int))
    zero = jnp.zeros((), dtype=jnp.result_type(float))
    return HistoryRecord(
        step=zero_int,
        n_valid=zero_int,
        **{n: zero for n in _RECORD_FIELDS if n not in ("step", "n_valid")},
    )


def _as_level(where: str, value) -> Array:
    """A tempering level as a 0-d floating array."""
    if isinstance(value, bool):
        raise TypeError(f"{where}: beta must be a real scalar, got the bool {value!r}")
    level = jnp.asarray(value)
    if level.ndim != 0:
        raise ValueError(f"{where}: beta must be a scalar, got shape {level.shape}")
    if not jnp.issubdtype(level.dtype, jnp.floating):
        level = level.astype(jnp.result_type(float))
    value_check(
        level,
        lambda x: bool(jnp.isfinite(x) & (x >= 0.0)),
        f"{where}: beta must be finite and not negative.",
    )
    return level

