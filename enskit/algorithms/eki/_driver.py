r"""One step, as its two phases, and the two loops over them."""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import Array

from ... import kalman, maps
from ...linalg import PSDLinOp
from .. import _common as c
from ._helpers import repair_failed_particles
from ._schedules import _SNAP_RELATIVE
from ._values import (
    PREDICTION,
    SCHEDULE_EXHAUSTED,
    STOPPING_RULE,
    EKIError,
    EKIResult,
    EKIState,
    Evaluation,
    HistoryRecord,
)

__all__ = ["evaluate", "assimilate", "advance", "iterate", "run"]

#: The layer's logger. No handler is installed; a caller who wants progress
#: reports adds one.
logger = logging.getLogger("enskit.algorithms.eki")

#: Relative slack on a budget's exhaustion check: relative, so that a small
#: ``beta_target`` is not swallowed whole.
_BUDGET_TOL_RELATIVE = 1e-12

_ON_FAILURE = ("raise", "repair")



def evaluate(
    state: EKIState,
    forward,
    y,
    noise_cov,
    *,
    inflation=None,
    inputs=None,
    on_failure: str = "raise",
) -> Evaluation:
    r"""Phase 1 of a step: inflate, evaluate the forward model once, repair, summarize.

    Takes no increment and moves nothing: ``state`` is unchanged, and the
    level and the step index are carried into the result. In order:

    1. split ``state.key`` into ``(next, inflate, evaluate, update)``;
    2. inflate: ``inflation(key_inflate, ensemble=state.ensemble,
       step=state.step, beta=state.beta)``, or the particles unchanged;
    3. evaluate: ``maps.pushforward(particles, forward, inputs=inputs,
       output=PREDICTION, key=key_evaluate)``;
    4. find the particles whose prediction has a non-finite entry, and raise
       or repair them;
    5. whiten the residuals :math:`y - g_j` against the base noise covariance.

    Parameters
    ----------
    state : EKIState
        The state to evaluate.
    forward : callable or StructuredMap
        The forward model, a simulator in the sense of
        :func:`enskit.maps.pushforward`: called once with every particle,
        receiving one ``(J, d_b)`` array per input block and returning
        ``(J, N)``. Never traced. A simulator declaring ``needs_key`` receives
        the step's evaluation key.
    y : Array
        The data, ``(N,)``, finite.
    noise_cov : PSDLinOp
        The base noise covariance :math:`R`, of side ``N``, supporting
        ``whiten``.
    inflation : Inflation, optional
        Keyword-only. Applied to the particles before they are evaluated;
        see :class:`enskit.algorithms.Inflation`. ``None`` passes them
        through unchanged.
    inputs : str or sequence of str, optional
        Keyword-only. The parameter blocks passed to ``forward``, in order;
        every block in block order by default.
    on_failure : {"raise", "repair"}
        Keyword-only. ``"raise"`` (default) raises :class:`EKIError` naming
        the failed particles; ``"repair"`` moves them to the valid particles'
        center with :func:`repair_failed_particles`.

    Returns
    -------
    Evaluation
        Its ensemble is the evaluated particles, after inflation and repair,
        with the block :data:`PREDICTION` appended.

    Raises
    ------
    EKIError
        If fewer than two particles are valid, or any is invalid under
        ``on_failure="raise"``. Carries ``state``.
    ValueError
        If ``y`` or ``noise_cov`` is malformed, ``y`` is not finite, the
        forward model's output is not ``(J, N)`` or has a refused dtype, the
        inflation's output does not match the particles, or ``on_failure`` is
        not one of the two values.
    TypeError
        If an argument has the wrong type.
    UnsupportedOpError
        If ``noise_cov`` does not support ``whiten``.

    Warns
    -----
    UserWarning
        From :func:`enskit.maps.pushforward`, when the forward model returned
        a narrower floating dtype than the particles', which is promoted.

    Notes
    -----
    When every particle is valid, nothing is repaired: the particles pass
    through bit for bit. Both phases split the key independently from
    ``state.key``; the split is deterministic, so they agree, and no key is
    put on the evaluation that schedules and stopping rules receive.
    """
    where = "eki.evaluate"
    _check_state(where, state)
    y, data_dim = _check_problem(where, y, noise_cov)
    _check_on_failure(where, on_failure)
    _check_optional_callable(where, "inflation", inflation)
    inputs = _check_inputs(where, state, inputs)
    evaluation, _ = _evaluate(
        where, state, forward, y, noise_cov, inflation, inputs, on_failure, data_dim
    )
    return evaluation


def assimilate(
    state: EKIState,
    evaluation: Evaluation,
    increment,
    y,
    noise_cov,
    *,
    update_rule,
    approximation=None,
    relaxation=None,
) -> EKIState:
    r"""Phase 2 of a step: one Kalman update with noise ``noise_cov / increment``.

    The update is

    .. code-block:: python

        kalman.update(evaluation.ensemble, {PREDICTION: y},
                      noise={PREDICTION: noise_cov / increment},
                      update_rule=update_rule, approximation=approximation,
                      key=key_update)

    since, with :math:`\Phi` the misfit and :math:`G` the forward model,
    :math:`e^{-\delta\Phi(G(u))} \propto \mathcal N(y;\, G(u),\, R/\delta)`:
    moving from level :math:`\beta` to :math:`\beta + \delta` is
    conditioning on :math:`y` with noise :math:`R/\delta`. The relaxation, if
    any, is then applied, the result checked to be finite, and the new state
    built at ``beta + increment`` and ``step + 1``.

    Parameters
    ----------
    state : EKIState
        The state to move.
    evaluation : Evaluation
        An evaluation of ``state``.
    increment : float or Array
        :math:`\delta`, a finite, strictly positive scalar.
    y : Array
        The data, ``(N,)``, finite.
    noise_cov : PSDLinOp
        The base noise covariance :math:`R`.
    update_rule : UpdateRule
        Keyword-only and required: :class:`~enskit.kalman.SymmetricSquareRoot`,
        :class:`~enskit.kalman.Matheron`, or any object with a ``build``
        method.
    approximation : callable, optional
        Keyword-only. ``(ensemble, noise) -> Gaussian``, the joint Gaussian
        the update conditions; :func:`enskit.kalman.gaussian_approximation`
        by default.
    relaxation : Relaxation, optional
        Keyword-only. Called as ``relaxation(prior=evaluation.ensemble,
        posterior=..., step=..., beta=...)`` on the update's result; see
        :class:`enskit.algorithms.Relaxation`.

    Returns
    -------
    EKIState

    Raises
    ------
    ValueError
        If ``increment`` is not a finite, strictly positive scalar; if
        ``evaluation`` is not of this state's step and level; or if the
        relaxation's output does not match the particles.
    TypeError
        If an argument has the wrong type, or ``update_rule`` has no ``build``.
    EKIError
        If the updated particles are not finite. Carries ``state``.

    Notes
    -----
    The checks on the increment and the evaluation run before anything else.
    The evaluation check compares the step and the level only, so it catches
    an evaluation the state has moved past but not one from another problem
    at the same position.
    """
    where = "eki.assimilate"
    _check_state(where, state)
    y, _ = _check_problem(where, y, noise_cov)
    dbeta = _check_increment(where, increment)
    _check_provenance(where, state, evaluation)
    _check_update_rule(where, update_rule)
    _check_optional_callable(where, "approximation", approximation)
    _check_optional_callable(where, "relaxation", relaxation)
    return _assimilate(
        where, state, evaluation, dbeta, y, noise_cov, update_rule, approximation,
        relaxation,
    )


def advance(
    state: EKIState,
    forward,
    y,
    noise_cov,
    increment,
    *,
    update_rule,
    inflation=None,
    relaxation=None,
    inputs=None,
    on_failure: str = "raise",
    approximation=None,
) -> EKIState:
    """One step at a known increment: :func:`evaluate`, then :func:`assimilate`.

    Exactly ``assimilate(state, evaluate(state, forward, y, noise_cov,
    inflation=..., inputs=..., on_failure=...), increment, y, noise_cov,
    update_rule=..., approximation=..., relaxation=...)``, except that the
    increment, the update rule and the callables are checked first, before
    the forward model is called.

    Parameters
    ----------
    state, forward, y, noise_cov, increment
        As :func:`evaluate` and :func:`assimilate` take them.
    update_rule, inflation, relaxation, inputs, on_failure, approximation
        Keyword-only, as :func:`evaluate` and :func:`assimilate` take them.

    Returns
    -------
    EKIState

    Raises
    ------
    EKIError, ValueError, TypeError
        As :func:`evaluate` and :func:`assimilate` raise them.
    """
    where = "eki.advance"
    _check_increment(where, increment)
    _check_update_rule(where, update_rule)
    _check_optional_callable(where, "approximation", approximation)
    _check_optional_callable(where, "relaxation", relaxation)
    evaluation = evaluate(
        state, forward, y, noise_cov, inflation=inflation, inputs=inputs,
        on_failure=on_failure,
    )
    return assimilate(
        state, evaluation, increment, y, noise_cov, update_rule=update_rule,
        approximation=approximation, relaxation=relaxation,
    )


def iterate(
    state: EKIState,
    forward,
    y,
    noise_cov,
    *,
    update_rule,
    schedule,
    stop=None,
    inflation=None,
    relaxation=None,
    inputs=None,
    on_failure: str = "raise",
    approximation=None,
    max_steps: int = 1000,
):
    """The driver as a generator, yielding after each step.

    Takes the arguments of :func:`run`. Yields after every evaluation,
    including a terminal one after which no step is taken, and returns the
    terminating status as its ``StopIteration`` value. For anything that
    observes or interrupts a run between steps: checkpointing, logging, a
    wall-clock budget, an early ``break``. Exceptions propagate; abandoning
    the generator is safe.

    Yields
    ------
    tuple
        ``(state, record, evaluation)``: the state after the step (unchanged
        after a terminal evaluation), the step's :class:`HistoryRecord`, and
        the :class:`Evaluation` it was built from.

    Raises
    ------
    EKIError, ValueError, TypeError
        As :func:`run` raises them, from the first ``next`` rather than at the
        call, since a generator's body does not run until then.

    Notes
    -----
    A caller who ends the loop has what a result needs::

        records = []
        for state, record, evaluation in eki.iterate(state, ...):
            records.append(record)
            if too_late():
                break
        result = eki.EKIResult(state=state, history=tuple(records),
                               status=eki.INTERRUPTED, last_evaluation=evaluation)
    """
    return (
        yield from _drive(
            "eki.iterate",
            state,
            forward,
            y,
            noise_cov,
            update_rule=update_rule,
            schedule=schedule,
            stop=stop,
            inflation=inflation,
            relaxation=relaxation,
            inputs=inputs,
            on_failure=on_failure,
            approximation=approximation,
            max_steps=max_steps,
        )
    )


def run(
    state: EKIState,
    forward,
    y,
    noise_cov,
    *,
    update_rule,
    schedule,
    stop=None,
    inflation=None,
    relaxation=None,
    inputs=None,
    on_failure: str = "raise",
    approximation=None,
    max_steps: int = 1000,
) -> EKIResult:
    r"""Run EKI until the schedule is exhausted or the stopping rule fires.

    Before each step, in order: end the run if the schedule is exhausted;
    raise if ``max_steps`` steps have been taken in this call; evaluate
    (:func:`evaluate`); end the run if the stopping rule fires; ask the
    schedule for the increment, ending the run on ``None``; and assimilate
    (:func:`assimilate`).

    Parameters
    ----------
    state : EKIState
        The state to start or resume from.
    forward : callable or StructuredMap
        The forward model; see :func:`evaluate`. Called once per step with
        every particle.
    y : Array
        The data, ``(N,)``, finite.
    noise_cov : PSDLinOp
        The base noise covariance :math:`R`, used through ``whiten``.
    update_rule : UpdateRule
        Keyword-only and required. There is no default: the square-root rule
        is exact in moments and the stochastic rule keeps the shape of a
        nonlinear posterior better.
    schedule : Schedule
        Keyword-only. :class:`FixedSchedule`, :class:`AdaptiveESSSchedule`,
        :class:`AdaptiveMisfitSchedule`, or your own.
    stop : StoppingRule, optional
        Keyword-only. Consulted after each evaluation.
    inflation : Inflation, optional
        Keyword-only. Applied before each evaluation.
    relaxation : Relaxation, optional
        Keyword-only. Applied after each update.
    inputs : str or sequence of str, optional
        Keyword-only. The parameter blocks passed to ``forward``; all, in
        order, by default.
    on_failure : {"raise", "repair"}
        Keyword-only. What a particle whose prediction is not finite does;
        see :func:`evaluate`.
    approximation : callable, optional
        Keyword-only. ``(ensemble, noise) -> Gaussian``, the joint Gaussian
        each update conditions; see :func:`enskit.kalman.update`.
    max_steps : int
        Keyword-only. A bound on the steps taken by this call, default 1000.
        Exceeding it raises.

    Returns
    -------
    EKIResult
        With status :data:`SCHEDULE_EXHAUSTED` or :data:`STOPPING_RULE`.

    Raises
    ------
    EKIError
        If ``max_steps`` is reached, or on a failure :func:`evaluate` or
        :func:`assimilate` raises. It carries ``state``, the last good state,
        and ``history``, so ``run(exc.state, ...)`` continues the run.
    ValueError
        On an invalid argument, including a ``max_steps`` below the number of
        steps the schedule's floor can require.
    TypeError
        If an argument has the wrong type.

    Warns
    -----
    UserWarning
        From :func:`enskit.maps.pushforward`, at each evaluation whose forward
        model returned a narrower dtype than the particles'; and once per run
        in which a particle was repaired.

    Notes
    -----
    Running on a state returned by an earlier run continues it. Starting a new
    ladder from a finished state needs :meth:`EKIState.restart`, since
    ``step`` and ``beta`` carry over.
    """
    driver = _drive(
        "eki.run",
        state,
        forward,
        y,
        noise_cov,
        update_rule=update_rule,
        schedule=schedule,
        stop=stop,
        inflation=inflation,
        relaxation=relaxation,
        inputs=inputs,
        on_failure=on_failure,
        approximation=approximation,
        max_steps=max_steps,
    )
    records: list[HistoryRecord] = []
    last_evaluation = None
    final_state = state
    while True:
        try:
            final_state, record, last_evaluation = next(driver)
        except StopIteration as finished:
            status = finished.value
            break
        records.append(record)
    result = EKIResult(
        state=final_state,
        history=tuple(records),
        status=status,
        last_evaluation=last_evaluation,
    )
    worst = result.min_n_valid
    if worst is not None and worst < state.n_particles:
        warnings.warn(
            f"eki.run: some particles' predictions were not finite and were "
            f"repaired; the worst step had {worst} of {state.n_particles} valid. "
            f"Each such step conditioned on a covariance damped by "
            f"(n_valid - 1) / (J - 1). Inspect result.stacked.n_valid.",
            stacklevel=2,
        )
    return result


# ---------------------------------------------------------------------------
# private: the one driver both loops wrap
# ---------------------------------------------------------------------------


def _drive(
    where,
    state,
    forward,
    y,
    noise_cov,
    *,
    update_rule,
    schedule,
    stop,
    inflation,
    relaxation,
    inputs,
    on_failure,
    approximation,
    max_steps,
):
    """The loop that knows why it stopped, as a generator returning the status."""
    _check_state(where, state)
    y, data_dim = _check_problem(where, y, noise_cov)
    _check_update_rule(where, update_rule)
    _check_on_failure(where, on_failure)
    _check_max_steps(where, max_steps)
    _check_schedule(where, schedule)
    for name, value in (
        ("stop", stop),
        ("inflation", inflation),
        ("relaxation", relaxation),
        ("approximation", approximation),
    ):
        _check_optional_callable(where, name, value)
    inputs = _check_inputs(where, state, inputs)
    _check_budget_against_bound(where, schedule, state.beta, max_steps)

    records: list[HistoryRecord] = []
    completed = 0
    while True:
        if _ladder_finished(schedule, state.step, state.beta):
            status = SCHEDULE_EXHAUSTED
            break
        if completed >= max_steps:
            raise EKIError(
                f"{where}: max_steps={max_steps} reached at step {state.step}, beta "
                f"{float(state.beta):g}, with schedule {schedule!r} and "
                f"{'a' if stop is not None else 'no'} stopping rule. An unbounded "
                f"schedule with no stopping rule is the usual cause.",
                state=state,
                history=records,
            )
        try:
            evaluation, n_valid = _evaluate(
                where, state, forward, y, noise_cov, inflation, inputs,
                on_failure, data_dim,
            )
        except EKIError as failure:
            failure.history = tuple(records)
            raise
        if n_valid < state.n_particles:
            logger.warning(
                "step %d: %d of %d particles' predictions were finite; the rest "
                "were repaired to the valid center",
                evaluation.step,
                n_valid,
                state.n_particles,
            )

        if stop is not None and bool(stop(evaluation)):
            record = HistoryRecord.from_evaluation(evaluation)
            records.append(record)
            _log_step(record)
            yield state, record, evaluation
            status = STOPPING_RULE
            break

        increment = schedule.next_increment(evaluation)
        if increment is None:
            record = HistoryRecord.from_evaluation(evaluation)
            records.append(record)
            _log_step(record)
            yield state, record, evaluation
            status = SCHEDULE_EXHAUSTED
            break

        dbeta = _check_increment(where, increment)
        try:
            new_state = _assimilate(
                where, state, evaluation, dbeta, y, noise_cov, update_rule,
                approximation, relaxation,
            )
        except EKIError as failure:
            failure.history = tuple(records)
            raise
        record = HistoryRecord.from_evaluation(evaluation, dbeta)
        records.append(record)
        completed += 1
        state = new_state
        _log_step(record)
        yield state, record, evaluation

    if not records:
        logger.warning(
            "the run performed no evaluations: its ladder was already finished on "
            "entry, at step %d and beta %g. A new ladder on a finished state needs "
            "EKIState.restart().",
            state.step,
            float(state.beta),
        )
    return status


def _check_schedule(where: str, schedule) -> None:
    """Require the two attributes the exhaustion check reads, and the method."""
    missing = [n for n in ("n_steps", "beta_target") if not hasattr(schedule, n)]
    if missing or not callable(getattr(schedule, "next_increment", None)):
        raise TypeError(
            f"{where}: a schedule must provide next_increment and the attributes "
            f"n_steps and beta_target; {schedule!r} lacks "
            f"{', '.join(missing) or 'next_increment'}. Either attribute may be "
            f"None."
        )


def _ladder_finished(schedule, step: int, beta) -> bool:
    """The driver's exhaustion check, from the schedule's attributes alone."""
    n_steps = schedule.n_steps
    if n_steps is not None and step >= n_steps:
        return True
    beta_target = schedule.beta_target
    if beta_target is None:
        return False
    return bool(beta >= beta_target - _BUDGET_TOL_RELATIVE * beta_target)


# ---------------------------------------------------------------------------
# private: the array work of one step
# ---------------------------------------------------------------------------


def _evaluate(
    where, state, forward, y, noise_cov, inflation, inputs, on_failure, data_dim
):
    """The first phase, with its arguments checked; returns the count read too."""
    _, key_inflate, key_evaluate, _ = _split_key(state.key)
    particles = state.ensemble
    if inflation is not None:
        particles = inflation(
            key_inflate, ensemble=particles, step=state.step, beta=state.beta
        )
        c.check_policy_output(
            where, f"the inflation {inflation!r}", particles, state.ensemble
        )
    evaluated = maps.pushforward(
        particles, forward, inputs=inputs, output=PREDICTION, key=key_evaluate
    )
    got = evaluated.dims[PREDICTION]
    if got != data_dim:
        raise ValueError(
            f"{where}: the forward model returned {got} values per particle, but "
            f"the data have {data_dim}"
        )
    valid = _valid_rows(evaluated[PREDICTION])
    n_valid = int(jnp.sum(valid))
    J = state.n_particles
    if n_valid < J:
        if n_valid < 2:
            raise EKIError(
                f"{where}: only {n_valid} of {J} particles' predictions were finite "
                f"at step {state.step}, beta {float(state.beta):g}. At least 2 are "
                f"required: a single particle has no anomalies.",
                state=state,
            )
        if on_failure == "raise":
            failed = [int(i) for i in jnp.flatnonzero(~valid)]
            raise EKIError(
                f"{where}: {J - n_valid} of {J} particles' predictions were not "
                f"finite at step {state.step}, beta {float(state.beta):g}, with "
                f"on_failure='raise'. Failed particles: {failed}. Pass "
                f"on_failure='repair' to move them to the valid particles' center.",
                state=state,
            )
        evaluated = repair_failed_particles(ensemble=evaluated, valid=valid)
    residuals = _whitened_residuals(noise_cov, y, evaluated[PREDICTION])
    evaluation = Evaluation(
        step=state.step,
        beta=state.beta,
        ensemble=evaluated,
        whitened_residuals=residuals,
        n_valid=n_valid,
    )
    return evaluation, n_valid


def _assimilate(
    where, state, evaluation, dbeta, y, noise_cov, update_rule, approximation,
    relaxation,
):
    """The second phase, with its arguments checked."""
    key_next, _, _, key_update = _split_key(state.key)
    names = state.ensemble.names
    dtype = state.ensemble[names[0]].dtype
    posterior = kalman.update(
        evaluation.ensemble,
        {PREDICTION: y},
        update_rule=update_rule,
        noise={PREDICTION: noise_cov / dbeta.astype(dtype)},
        approximation=approximation,
        key=key_update,
    )
    if posterior.names != names:
        posterior = posterior.marginal(*names)
    _check_finite_result(where, state, posterior, f"the update {update_rule!r}")
    if relaxation is not None:
        posterior = relaxation(
            prior=evaluation.ensemble, posterior=posterior, step=state.step,
            beta=state.beta,
        )
        c.check_policy_output(
            where, f"the relaxation {relaxation!r}", posterior, state.ensemble
        )
        _check_finite_result(where, state, posterior, f"the relaxation {relaxation!r}")
    return EKIState(
        posterior, key=key_next, beta=state.beta + dbeta, step=state.step + 1
    )


def _check_finite_result(where, state, posterior, who) -> None:
    """Raise :class:`EKIError`, carrying the state, on a non-finite particle.

    Checked on the update's result before a relaxation sees it, so that a
    relaxation's own debug-mode check cannot raise first without the state.
    """
    if not bool(jnp.all(posterior.all_finite)):
        raise EKIError(
            f"{where}: {who} returned a non-finite particle at step {state.step}, "
            f"beta {float(state.beta):g}. A non-finite particle would make every "
            f"later step nan, so the run stops here.",
            state=state,
        )


def _split_key(key):
    """The pinned split, ``(next, inflate, evaluate, update)``, whatever the policies."""
    key_next, key_inflate, key_evaluate, key_update = jax.random.split(key, 4)
    return key_next, key_inflate, key_evaluate, key_update


@jax.jit
def _valid_rows(predictions: Array) -> Array:
    """``True`` where a particle's whole prediction is finite."""
    return jnp.all(jnp.isfinite(predictions), axis=-1)


@jax.jit
def _whitened_residuals(noise_cov, y: Array, predictions: Array) -> Array:
    """``W (y - g_j)``, one row per particle."""
    return noise_cov.whiten(y - predictions)


def _log_step(record: HistoryRecord) -> None:
    """One ``INFO`` record per step: the step, the level, the increment, the misfit."""
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "step %d: beta %g -> %g, increment %g, mean misfit %g",
            int(record.step),
            float(record.beta),
            float(record.beta_next),
            float(record.increment),
            float(record.misfit_mean),
        )


# ---------------------------------------------------------------------------
# private: call-time validation
# ---------------------------------------------------------------------------


def _check_state(where: str, state) -> None:
    if not isinstance(state, EKIState):
        raise TypeError(f"{where}: state must be an EKIState, got {type(state).__name__}")
    if state.batch_shape != ():
        raise ValueError(
            f"{where}: {state!r} is a vmapped family. A run cannot be traced, so a "
            f"family of runs is a Python loop over eki.run, not a jax.vmap."
        )


def _check_problem(where: str, y, noise_cov):
    """Validate the data and the noise covariance, and the data's finiteness."""
    if not isinstance(noise_cov, PSDLinOp):
        raise TypeError(
            f"{where}: noise_cov must be an enskit.linalg.PSDLinOp, got "
            f"{type(noise_cov).__name__}"
        )
    if noise_cov.batch_shape != ():
        raise ValueError(
            f"{where}: {noise_cov!r} is a vmapped family; a run binds one noise "
            f"covariance"
        )
    # Demanded here rather than at the first whitening, so that a covariance
    # without a whitener costs no evaluation of the forward model.
    noise_cov._require("whiten")
    data_dim = noise_cov.shape[0]
    y = jnp.asarray(y)
    if y.ndim != 1 or y.shape[0] != data_dim:
        raise ValueError(
            f"{where}: expected y of shape ({data_dim},) to match {noise_cov!r}, got "
            f"shape {y.shape}"
        )
    if not bool(jnp.all(jnp.isfinite(y))):
        raise ValueError(
            f"{where}: y must be finite. A non-finite datum otherwise surfaces as a "
            f"run of nan updates or as a refused increment, neither of which names y."
        )
    return y, data_dim


def _check_inputs(where: str, state: EKIState, inputs) -> tuple[str, ...]:
    """The parameter blocks the forward model receives, as a tuple."""
    names = state.ensemble.names
    if inputs is None:
        return names
    if isinstance(inputs, str):
        inputs = (inputs,)
    if not isinstance(inputs, Sequence):
        raise TypeError(
            f"{where}: inputs must be a str or a sequence of str, got "
            f"{type(inputs).__name__}"
        )
    inputs = tuple(inputs)
    for name in inputs:
        if not isinstance(name, str):
            raise TypeError(f"{where}: inputs must be str, got {type(name).__name__}")
        if name not in names:
            raise KeyError(
                f"{where}: input {name!r} is not a parameter block; the blocks are "
                f"{names}"
            )
    if not inputs:
        raise ValueError(f"{where}: inputs must name at least one block")
    if len(set(inputs)) != len(inputs):
        raise ValueError(f"{where}: a block is repeated in inputs {inputs}")
    return inputs


def _check_update_rule(where: str, update_rule) -> None:
    if not callable(getattr(update_rule, "build", None)):
        raise TypeError(
            f"{where}: update_rule must have a build method (an "
            f"enskit.kalman.UpdateRule), got {type(update_rule).__name__}. Pass "
            f"kalman.SymmetricSquareRoot() or kalman.Matheron()."
        )


def _check_optional_callable(where: str, name: str, value) -> None:
    if value is not None and not callable(value):
        raise TypeError(
            f"{where}: {name} must be callable or None, got {type(value).__name__}"
        )


def _check_increment(where: str, increment) -> Array:
    """A finite, strictly positive scalar, as a 0-d floating array."""
    if isinstance(increment, bool):
        raise TypeError(f"{where}: the increment is the bool {increment!r}")
    value = jnp.asarray(increment)
    if value.ndim != 0:
        raise ValueError(
            f"{where}: the increment must be a scalar, got shape {value.shape}"
        )
    if not (
        jnp.issubdtype(value.dtype, jnp.floating)
        or jnp.issubdtype(value.dtype, jnp.integer)
    ):
        raise TypeError(f"{where}: the increment must be real, got dtype {value.dtype}")
    value = value.astype(jnp.result_type(float))
    if not bool(jnp.isfinite(value) & (value > 0.0)):
        raise ValueError(
            f"{where}: the increment must be finite and strictly positive, got "
            f"{value}. A zero increment leaves the particles unchanged while beta "
            f"never advances, so an adaptive ladder would spin until max_steps."
        )
    return value


def _check_provenance(where: str, state: EKIState, evaluation) -> None:
    """Require that the evaluation is of this state's step and level."""
    if not isinstance(evaluation, Evaluation):
        raise TypeError(
            f"{where}: evaluation must be an Evaluation, got {type(evaluation).__name__}"
        )
    c.guard(evaluation, where)
    if evaluation.step != state.step or not bool(evaluation.beta == state.beta):
        raise ValueError(
            f"{where}: the evaluation is of another state: step {evaluation.step}, "
            f"beta {float(evaluation.beta):g}, against the state's step "
            f"{state.step}, beta {float(state.beta):g}"
        )
    if evaluation.n_particles != state.n_particles:
        raise ValueError(
            f"{where}: the evaluation has {evaluation.n_particles} particles, the "
            f"state {state.n_particles}"
        )


def _check_on_failure(where: str, on_failure) -> None:
    if on_failure not in _ON_FAILURE:
        raise ValueError(
            f"{where}: on_failure must be one of {_ON_FAILURE}, got {on_failure!r}. "
            f"An unrecognized value raises rather than falling back to a default."
        )


def _check_max_steps(where: str, max_steps) -> None:
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError(f"{where}: max_steps must be a positive int, got {max_steps!r}")


def _check_budget_against_bound(where: str, schedule, beta, max_steps: int) -> None:
    """Refuse a bound too small for the schedule's own floor-bound worst case.

    A schedule exposing ``beta_target`` and ``min_increment`` needs at most
    ``_steps_needed`` further steps; this raises before the first evaluation
    when ``max_steps`` is below that. The remaining budget is used, not the
    whole, so a resumed run is bounded exactly.
    """
    beta_target = getattr(schedule, "beta_target", None)
    floor = getattr(schedule, "min_increment", None)
    if beta_target is None or floor is None:
        return
    try:
        remaining = float(beta_target) - float(beta)
        worst_case = _steps_needed(remaining, float(floor), float(beta_target))
    except (TypeError, ValueError, ZeroDivisionError):
        return
    if worst_case > max_steps:
        raise ValueError(
            f"{where}: max_steps={max_steps} cannot accommodate {schedule!r}, whose "
            f"floor of {floor} against a remaining budget of {remaining:g} needs up "
            f"to {worst_case} steps. Raise max_steps to at least {worst_case}, or "
            f"raise min_increment."
        )


def _steps_needed(remaining: float, floor: float, beta_target: float) -> int:
    """The most steps of at least ``floor`` that a remaining budget can take.

    The exhaustion check ends the ladder within ``_BUDGET_TOL_RELATIVE`` of the
    target, and the clamp takes a remainder below ``_SNAP_RELATIVE`` with the
    step before it, so the count is ``ceil((remaining - snap) / floor)``, and
    one step for a remainder between the two tolerances. Subtracting the snap
    before dividing also makes a quotient that is an integer up to round-off
    count as that integer.
    """
    if remaining <= _BUDGET_TOL_RELATIVE * beta_target:
        return 0
    snap = _SNAP_RELATIVE * beta_target
    return max(1, math.ceil((remaining - snap) / floor))
