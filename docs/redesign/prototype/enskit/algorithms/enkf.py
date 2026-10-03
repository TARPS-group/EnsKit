"""Prototype EnKF driver: forecast with a transition, analysis with a Kalman update."""
from __future__ import annotations

import dataclasses
from typing import Callable, Sequence

import jax
import jax.numpy as jnp

from .. import kalman
from ..distribution import Ensemble
from ..maps import AdditiveNoise, pushforward

PREDICTION = "prediction"


def forecast(key, ensemble: Ensemble, transition: Callable, *, state: str,
             inputs: Sequence[str] | None = None, transition_noise=None) -> Ensemble:
    """Push the ``state`` block through the transition (optionally reading other blocks)."""
    ens = pushforward(ensemble, transition, inputs=inputs or (state,), output=state)
    if transition_noise is not None:
        ens = pushforward(ens, AdditiveNoise(transition_noise), inputs=state, output=state, key=key)
    return ens


def analysis(key, ensemble: Ensemble, observation, *, observe: Callable, noise_cov, update_rule,
             inputs: Sequence[str], approximation: Callable | None = None):
    """One analysis. Returns (analysis ensemble, log predictive density of the observation)."""
    ens = pushforward(ensemble, observe, inputs=inputs, output=PREDICTION)
    build = kalman.gaussian_approximation if approximation is None else approximation
    g = build(ens, {PREDICTION: noise_cov})
    log_evidence = g.marginal(PREDICTION).log_density({PREDICTION: observation})
    return update_rule.build(ens, g, (PREDICTION,))({PREDICTION: observation}, key=key), log_evidence


@dataclasses.dataclass(frozen=True)
class FilterResult:
    ensemble: Ensemble
    means: dict            # block -> (T, d) analysis means
    log_evidence: jnp.ndarray  # (T,)


def filter(key, ensemble: Ensemble, observations, *, transition, observe, noise_cov, update_rule,
           state: str | None = None, transition_inputs=None, observe_inputs=None,
           transition_noise=None, inflation: Callable | None = None,
           relaxation: Callable | None = None, approximation: Callable | None = None) -> FilterResult:
    state = state or ensemble.names[0]
    observe_inputs = observe_inputs or (state,)
    means = {n: [] for n in ensemble.names}
    logz = []
    for t, y in enumerate(observations):
        key, k_fc, k_inf, k_an = jax.random.split(key, 4)
        ensemble = forecast(k_fc, ensemble, transition, state=state, inputs=transition_inputs,
                            transition_noise=transition_noise)
        if inflation is not None:
            ensemble = inflation(k_inf, ensemble=ensemble, time=t)
        prior = ensemble
        ensemble, lz = analysis(k_an, ensemble, y, observe=observe, noise_cov=noise_cov,
                                update_rule=update_rule, inputs=observe_inputs, approximation=approximation)
        if relaxation is not None:
            ensemble = relaxation(prior=prior.marginal(*ensemble.names), posterior=ensemble, time=t)
        for n in ensemble.names:
            means[n].append(ensemble.mean(n))
        logz.append(lz)
    return FilterResult(ensemble, {n: jnp.stack(v) for n, v in means.items()}, jnp.stack(logz))
