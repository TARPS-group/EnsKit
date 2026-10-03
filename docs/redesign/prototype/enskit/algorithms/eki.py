"""Prototype EKI driver, written entirely on top of maps + kalman.

Protocols follow the design's stubs: schedules and stopping rules read an
Evaluation; inflation is called as inflation(key, *, ensemble, step, beta);
each step splits the key into (next, inflate, update).
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from .. import kalman
from ..distribution import Ensemble, Gaussian, _pytree
from ..maps import pushforward

PREDICTION = "prediction"
SCHEDULE_EXHAUSTED = "schedule_exhausted"
STOPPING_RULE = "stopping_rule"


@_pytree(data=("ensemble", "beta", "key"), meta=("step",))
class EKIState:
    def __init__(self, ensemble: Ensemble, *, beta=0.0, key=None, step: int = 0):
        if PREDICTION in ensemble.names:
            raise ValueError(f"EKIState: block name {PREDICTION!r} is reserved")
        object.__setattr__(self, "ensemble", ensemble)
        object.__setattr__(self, "beta", jnp.asarray(beta, dtype=float))
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "step", step)

    @classmethod
    def from_prior(cls, key, prior: Gaussian, n_particles: int) -> EKIState:
        k1, k2 = jax.random.split(key)
        return cls(prior.sample(k1, n_particles), key=k2)

    def replace(self, **kw) -> EKIState:
        d = dict(ensemble=self.ensemble, beta=self.beta, key=self.key, step=self.step)
        d.update(kw)
        return EKIState(d.pop("ensemble"), **d)

    def restart(self) -> EKIState:
        return self.replace(beta=0.0, step=0)


@_pytree(data=("beta", "ensemble", "whitened_residuals"), meta=("step",))
class Evaluation:
    """The evaluated members with the prediction block, and whitened residuals."""

    def __init__(self, *, step, beta, ensemble, whitened_residuals):
        for k, v in dict(step=step, beta=beta, ensemble=ensemble,
                         whitened_residuals=whitened_residuals).items():
            object.__setattr__(self, k, v)

    @property
    def misfits(self):
        return 0.5 * jnp.sum(self.whitened_residuals**2, axis=-1)

    @property
    def center_misfit(self):  # misfit of the mean prediction (residuals are linear in g)
        r = jnp.mean(self.whitened_residuals, axis=0)
        return 0.5 * jnp.sum(r * r)

    @property
    def n_particles(self):
        return self.ensemble.n_particles


class EKIError(RuntimeError):
    def __init__(self, message, *, state, history):
        super().__init__(message)
        self.state, self.history = state, history


def misfits(y, predictions, noise_cov) -> Array:
    r = noise_cov.whiten(y - predictions)
    return 0.5 * jnp.sum(r * r, axis=-1)


def effective_sample_size(misfit_values, increment) -> Array:
    lse = jax.scipy.special.logsumexp
    logw = -increment * misfit_values
    return jnp.exp(2 * lse(logw) - lse(2 * logw))


# -- schedules ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FixedSchedule:
    increments: tuple
    beta_target = None

    @property
    def n_steps(self):
        return len(self.increments)

    @classmethod
    def constant(cls, increment, n_steps):
        return cls((increment,) * n_steps)

    def next_increment(self, evaluation):
        return jnp.asarray(self.increments[evaluation.step])


@dataclasses.dataclass(frozen=True)
class AdaptiveESSSchedule:
    ess_fraction: float = 0.5
    beta_target: float | None = 1.0
    min_increment: float = 1e-3
    max_increment: float = 1.0
    n_bisect: int = 50
    n_steps = None

    def next_increment(self, evaluation):
        phi = evaluation.misfits
        J = phi.shape[0]
        hi = jnp.asarray(self.max_increment)
        if self.beta_target is not None:
            hi = jnp.minimum(hi, self.beta_target - evaluation.beta)
        target = self.ess_fraction * J
        lo = jnp.where(effective_sample_size(phi, hi) >= target, hi, 0.0)

        def body(_, lohi):
            lo, hi = lohi
            mid = 0.5 * (lo + hi)
            ok = effective_sample_size(phi, mid) >= target
            return jnp.where(ok, mid, lo), jnp.where(ok, hi, mid)

        lo, _ = jax.lax.fori_loop(0, self.n_bisect, body, (lo, hi))
        delta = jnp.maximum(lo, self.min_increment)  # the floor beats the criterion
        if self.beta_target is not None:              # the budget beats the floor
            delta = jnp.minimum(delta, self.beta_target - evaluation.beta)
        return jax.lax.stop_gradient(delta)


@dataclasses.dataclass(frozen=True)
class DiscrepancyStop:
    tau: float = 1.0

    def __call__(self, evaluation) -> bool:
        n_data = evaluation.whitened_residuals.shape[-1]
        return bool(2 * evaluation.center_misfit <= self.tau**2 * n_data)


# -- one step -------------------------------------------------------------------------


def _keys(state):
    return jax.random.split(state.key, 3)  # (next, inflate, update), whatever the policies


def evaluate(state: EKIState, forward: Callable, y, noise_cov, *, inflation=None,
             inputs=None) -> Evaluation:
    _, k_inflate, _ = _keys(state)
    ens = state.ensemble
    if inflation is not None:
        ens = inflation(k_inflate, ensemble=ens, step=state.step, beta=state.beta)
    ens = pushforward(ens, forward, inputs=inputs or ens.names, output=PREDICTION)
    return Evaluation(step=state.step, beta=state.beta, ensemble=ens,
                      whitened_residuals=noise_cov.whiten(y - ens[PREDICTION]))


def assimilate(state: EKIState, evaluation: Evaluation, increment, y, noise_cov, *,
               update_rule, approximation=None) -> EKIState:
    increment = jnp.asarray(increment)
    if not isinstance(increment, jax.core.Tracer) and not bool(
            jnp.isfinite(increment) & (increment > 0)):
        raise ValueError(f"assimilate: increment must be finite and > 0, got {increment}")
    if evaluation.step != state.step:
        raise ValueError("assimilate: evaluation is from a different step")
    k_next, _, k_update = _keys(state)
    post = kalman.update(evaluation.ensemble, {PREDICTION: y}, update_rule=update_rule,
                         noise={PREDICTION: noise_cov / increment}, approximation=approximation, key=k_update)
    return state.replace(ensemble=post, beta=state.beta + increment, key=k_next,
                         step=state.step + 1)


def advance(state, forward, y, noise_cov, increment, *, update_rule, inflation=None, inputs=None,
            approximation=None):
    ev = evaluate(state, forward, y, noise_cov, inflation=inflation, inputs=inputs)
    return assimilate(state, ev, increment, y, noise_cov, update_rule=update_rule, approximation=approximation)


# -- the run ------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EKIResult:
    state: EKIState
    history: tuple
    status: str
    last_evaluation: Evaluation | None
    n_evaluations: int

    @property
    def ensemble(self):
        return self.state.ensemble


def _exhausted(schedule, state) -> bool:
    if schedule.n_steps is not None and state.step >= schedule.n_steps:
        return True
    bt = schedule.beta_target
    return bt is not None and bool(state.beta >= bt - 1e-12 * bt)


def run(state: EKIState, forward: Callable, y, noise_cov, *, update_rule, schedule,
        stop=None, inflation=None, inputs=None, approximation=None, max_steps: int = 1000) -> EKIResult:
    history, n_eval, ev = [], 0, None
    for taken in range(max_steps + 1):
        if _exhausted(schedule, state):  # before evaluating: a T-step ladder costs T evaluations
            return EKIResult(state, tuple(history), SCHEDULE_EXHAUSTED, ev, n_eval)
        if taken == max_steps:
            raise EKIError(f"run: max_steps={max_steps} exceeded", state=state,
                           history=tuple(history))
        ev = evaluate(state, forward, y, noise_cov, inflation=inflation, inputs=inputs)
        n_eval += 1
        if stop is not None and stop(ev):
            return EKIResult(state, tuple(history), STOPPING_RULE, ev, n_eval)
        inc = schedule.next_increment(ev)
        history.append(dict(step=state.step, beta=float(state.beta), increment=float(inc),
                            mean_misfit=float(jnp.mean(ev.misfits)),
                            ess=float(effective_sample_size(ev.misfits, inc))))
        state = assimilate(state, ev, inc, y, noise_cov, update_rule=update_rule, approximation=approximation)
