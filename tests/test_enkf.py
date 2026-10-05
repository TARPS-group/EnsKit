"""Tests for ``enskit.algorithms.enkf``, by the numbered obligations of its contract.

``docs/enkf-contract.md`` lists the obligations under *Conformance*; each
test name starts with the obligation's number. The toy problems' tests are
at the end, then the user guide's page.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.stats as st
import numpy as np
import pytest

from enskit import kalman, maps, toy
from enskit.algorithms import (
    AdditiveInflation,
    MultiplicativeInflation,
    RelaxToPriorPerturbations,
    RelaxToPriorSpread,
    enkf,
)
from enskit.distribution import Ensemble, Gaussian, exact_moment_ensemble
from enskit.linalg import Dense, DensePSD, PSDDiagonal

SSR = kalman.SymmetricSquareRoot()
MATHERON = kalman.Matheron()


def _linear(**kwargs):
    """A small linear problem, without transition noise unless asked for."""
    defaults = dict(state_dim=3, data_dim=2, n_times=6, transition_noise_std=0.0)
    return toy.linear_state_space(**{**defaults, **kwargs})


def _noisy(**kwargs):
    return _linear(transition_noise_std=0.3, **kwargs)


def _particles(problem, n_particles=8, seed=0):
    return problem.initial.sample(jax.random.key(seed), n_particles=n_particles)


def _filter(problem, ensemble=None, **kwargs):
    ensemble = _particles(problem) if ensemble is None else ensemble
    arguments = dict(
        transition=problem.transition,
        observe=problem.observe,
        noise_cov=problem.noise_cov,
        update_rule=SSR,
    )
    if getattr(problem, "transition_noise", None) is not None:
        arguments["transition_noise"] = problem.transition_noise
    arguments.update(kwargs)
    observations = arguments.pop("observations", problem.observations)
    return enkf.filter(ensemble, observations, **arguments)


def _same(a: Ensemble, b: Ensemble) -> None:
    assert a.names == b.names
    for name in a.names:
        assert np.array_equal(np.asarray(a[name]), np.asarray(b[name])), name


def _rmse(means, truth):
    return jnp.sqrt(jnp.mean((means - truth) ** 2, axis=1))


class _Identity:
    """An inflation that returns its ensemble unchanged and draws nothing."""

    def __call__(self, key, *, ensemble, **context):
        return ensemble


class _Recorder:
    """A policy recording every call's arguments, passing the particles through."""

    def __init__(self, relaxation=False):
        self.calls = []
        self.relaxation = relaxation

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return kwargs["posterior"] if self.relaxation else kwargs["ensemble"]


class _CompileCounter:
    """Counts backend compilations, through JAX's own monitoring events."""

    def __init__(self):
        self.count = 0

    def __call__(self, event, duration, **kwargs):
        if event == "/jax/core/compile/backend_compile_duration":
            self.count += 1


# ===========================================================================
# 1-2: forecast
# ===========================================================================


def test_1_forecast_is_its_two_pushforwards_bit_for_bit():
    problem = _noisy()
    ens = _particles(problem).assign(theta=jnp.ones((8, 2)))
    key = jax.random.key(3)
    got = enkf.forecast(
        ens, problem.transition, state="x", transition_noise=problem.transition_noise,
        key=key,
    )
    key_transition, key_noise = jax.random.split(key)
    want = maps.pushforward(
        ens, problem.transition, inputs=("x",), output="x", key=key_transition
    )
    want = maps.pushforward(
        want, maps.AdditiveNoise(problem.transition_noise), inputs="x", output="x",
        key=key_noise,
    )
    _same(got, want)
    assert got.names == ("x", "theta")
    assert np.array_equal(np.asarray(got["theta"]), np.asarray(ens["theta"]))


def test_1_forecast_without_noise_needs_no_key_and_keeps_weights():
    problem = _linear()
    ens = _particles(problem)
    weighted = Ensemble(x=ens["x"], log_weights=jnp.linspace(-1.0, 0.0, 8))
    got = enkf.forecast(weighted, problem.transition, state="x")
    assert got.is_weighted
    assert np.array_equal(np.asarray(got.log_weights), np.asarray(weighted.log_weights))
    assert np.allclose(got["x"], problem.transition(ens["x"]), atol=1e-14)


def test_1_a_transition_that_needs_a_key_gets_the_transition_key():
    seen = []

    def transition(key, x):
        seen.append(key)
        return x + jax.random.normal(key, x.shape)

    transition.needs_key = True
    ens = Ensemble(x=jnp.zeros((4, 2)))
    key = jax.random.key(5)
    enkf.forecast(ens, transition, state="x", key=key)
    want, _ = jax.random.split(key)
    assert np.array_equal(
        np.asarray(jax.random.key_data(seen[0])), np.asarray(jax.random.key_data(want))
    )


def test_2_forecast_refuses_what_it_cannot_do():
    problem = _noisy()
    ens = _particles(problem)
    with pytest.raises(ValueError, match="key is required"):
        enkf.forecast(
            ens, problem.transition, state="x",
            transition_noise=problem.transition_noise,
        )
    with pytest.raises(ValueError, match="side 2"):
        enkf.forecast(
            ens, problem.transition, state="x", transition_noise=PSDDiagonal(jnp.ones(2)),
            key=jax.random.key(0),
        )
    with pytest.raises(ValueError, match="returned 2 values"):
        enkf.forecast(ens, lambda x: x[:, :2], state="x")
    with pytest.raises(KeyError, match="'y' is not a block"):
        enkf.forecast(ens, problem.transition, state="y")
    with pytest.raises(KeyError, match="not a block"):
        enkf.forecast(ens, problem.transition, state="x", inputs=("x", "z"))
    with pytest.raises(TypeError, match="Ensemble"):
        enkf.forecast(ens["x"], problem.transition, state="x")
    with pytest.raises(TypeError, match="typed key"):
        enkf.forecast(ens, problem.transition, state="x", key=jax.random.PRNGKey(0))


# ===========================================================================
# 3-5: analysis
# ===========================================================================


@pytest.mark.parametrize("rule", [SSR, MATHERON], ids=["ssr", "matheron"])
def test_3_analysis_is_kalman_update_on_the_predicted_ensemble(rule):
    problem = _linear()
    ens = _particles(problem).assign(theta=jnp.arange(16.0).reshape(8, 2))
    y = problem.observations[0]
    key = jax.random.key(4)
    got, _ = enkf.analysis(
        ens, y, observe=problem.observe, noise_cov=problem.noise_cov, update_rule=rule,
        inputs="x", key=key,
    )
    predicted = maps.pushforward(ens, problem.observe, inputs="x", output=enkf.PREDICTION)
    want = kalman.update(
        predicted, {enkf.PREDICTION: y}, noise={enkf.PREDICTION: problem.noise_cov},
        update_rule=rule, key=key,
    )
    _same(got, want)
    assert got.names == ("x", "theta")


def test_3_analysis_takes_a_localized_rule_unchanged():
    problem = toy.lorenz96(state_dim=12, n_times=1, obs_every=3)
    ens = problem.initial.sample(jax.random.key(0), n_particles=6)
    localization = kalman.DomainLocalization(
        target_coords={"x": problem.coords}, given_coords=problem.obs_coords,
        radius=4.0, max_neighbors=3,
    )
    rule = kalman.LocalizedUpdateRule(SSR, localization)
    got, evidence = enkf.analysis(
        ens, problem.observations[0], observe=problem.observe,
        noise_cov=problem.noise_cov, update_rule=rule, inputs="x",
    )
    predicted = maps.pushforward(ens, problem.observe, inputs="x", output=enkf.PREDICTION)
    want = kalman.update(
        predicted, {enkf.PREDICTION: problem.observations[0]},
        noise={enkf.PREDICTION: problem.noise_cov}, update_rule=rule,
    )
    _same(got, want)
    assert evidence.shape == ()


def test_3_the_block_order_is_restored_after_a_reordering_approximation():
    problem = _linear()
    ens = _particles(problem).assign(theta=jnp.arange(16.0).reshape(8, 2))

    def reordered(ensemble, noise):
        moved = ensemble.marginal(enkf.PREDICTION, "theta", "x")
        return kalman.gaussian_approximation(moved, noise)

    got, _ = enkf.analysis(
        ens, problem.observations[0], observe=problem.observe,
        noise_cov=problem.noise_cov, update_rule=MATHERON, inputs="x",
        approximation=reordered, key=jax.random.key(0),
    )
    assert got.names == ("x", "theta")


def test_4_the_log_evidence_is_the_dense_density_and_the_same_for_both_rules():
    problem = _linear()
    ens = _particles(problem, n_particles=12)
    y = problem.observations[2]
    h = problem.observe(ens["x"])
    mean = h.mean(axis=0)
    centered = h - mean
    cov = centered.T @ centered / (12 - 1) + problem.noise_cov.to_dense()
    want = st.multivariate_normal.logpdf(y, mean, cov)
    for rule, key in ((SSR, None), (MATHERON, jax.random.key(1))):
        _, got = enkf.analysis(
            ens, y, observe=problem.observe, noise_cov=problem.noise_cov,
            update_rule=rule, inputs="x", key=key,
        )
        assert abs(float(got) - float(want)) < 1e-12


def test_4_the_approximation_is_called_once_and_serves_both():
    problem = _linear()
    ens = _particles(problem)
    calls = []

    def inflated_noise(ensemble, noise):
        calls.append(ensemble.names)
        return ensemble.project().add_noise(
            {name: PSDDiagonal(cov.diag() + 3.0) for name, cov in noise.items()}
        )

    y = problem.observations[0]
    got, evidence = enkf.analysis(
        ens, y, observe=problem.observe, noise_cov=problem.noise_cov,
        update_rule=SSR, inputs="x", approximation=inflated_noise,
    )
    assert calls == [("x", enkf.PREDICTION)]
    bigger = PSDDiagonal(problem.noise_cov.diag() + 3.0)
    want, want_evidence = enkf.analysis(
        ens, y, observe=problem.observe, noise_cov=bigger, update_rule=SSR, inputs="x"
    )
    _same(got, want)
    assert float(evidence) == float(want_evidence)


def test_5_analysis_refuses_what_it_cannot_do():
    problem = _linear()
    ens = _particles(problem)
    common = dict(observe=problem.observe, noise_cov=problem.noise_cov, update_rule=SSR)
    y = problem.observations[0]
    with pytest.raises(TypeError, match="inputs is required"):
        enkf.analysis(ens, y, inputs=None, **common)
    with pytest.raises(ValueError, match="named 'prediction'"):
        enkf.analysis(ens.rename(x=enkf.PREDICTION), y, inputs=enkf.PREDICTION, **common)
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        enkf.analysis(ens, jnp.zeros(3), inputs="x", **common)
    with pytest.raises(ValueError, match="returned 3 values"):
        enkf.analysis(
            ens, y, inputs="x", observe=lambda x: x, noise_cov=problem.noise_cov,
            update_rule=SSR,
        )

    def forgetful(ensemble, noise):
        return kalman.gaussian_approximation(ensemble.drop("theta"), noise)

    with pytest.raises(ValueError, match=r"has blocks \('x', 'prediction'\)"):
        enkf.analysis(
            ens.assign(theta=jnp.ones((8, 1))), y, inputs="x", approximation=forgetful,
            key=jax.random.key(0), observe=problem.observe,
            noise_cov=problem.noise_cov, update_rule=MATHERON,
        )
    with pytest.raises(TypeError, match="build method"):
        enkf.analysis(
            ens, y, inputs="x", observe=problem.observe, noise_cov=problem.noise_cov,
            update_rule=object(),
        )
    with pytest.raises(ValueError, match="requires a key|key is required|needs a key"):
        enkf.analysis(
            ens, y, inputs="x", observe=problem.observe, noise_cov=problem.noise_cov,
            update_rule=MATHERON,
        )


# ===========================================================================
# 6-11: filter
# ===========================================================================


def test_6_the_filter_is_its_loop_written_out():
    problem = _noisy(n_times=5)
    ens = _particles(problem)
    inflation = MultiplicativeInflation(1.1)
    relaxation = RelaxToPriorPerturbations(0.3)
    key = jax.random.key(7)
    result = _filter(
        problem, ens, update_rule=MATHERON, inflation=inflation, relaxation=relaxation,
        key=key, keep_ensembles=True,
    )
    means, evidence = [], []
    for t, y in enumerate(problem.observations):
        key, k_forecast, k_inflate, k_analysis = jax.random.split(key, 4)
        forecast = enkf.forecast(
            ens, problem.transition, state="x", transition_noise=problem.transition_noise,
            key=k_forecast,
        )
        background = inflation(k_inflate, ensemble=forecast, time=t)
        analyzed, lz = enkf.analysis(
            background, y, observe=problem.observe, noise_cov=problem.noise_cov,
            update_rule=MATHERON, inputs="x", key=k_analysis,
        )
        ens = relaxation(prior=background, posterior=analyzed, time=t)
        means.append(ens.mean("x"))
        evidence.append(lz)
    _same(result.ensemble, ens)
    assert np.array_equal(np.asarray(result.means["x"]), np.asarray(jnp.stack(means)))
    assert np.array_equal(
        np.asarray(result.log_evidence), np.asarray(jnp.stack(evidence))
    )


@pytest.mark.parametrize("n_particles", [4, 8])
def test_7_the_filter_is_exact_on_a_linear_gaussian_problem(n_particles):
    """J = d + 1 and 2d + 2: the means, covariance and evidence of the Kalman filter."""
    problem = _linear(n_times=20, transition_noise_std=0.0)
    assert problem.transition_noise is None
    ens = exact_moment_ensemble(jax.random.key(0), problem.initial, n_particles)
    result = _filter(problem, ens)
    exact, exact_evidence = problem.exact_filter()
    exact_means = jnp.stack([g.mean("x") for g in exact])
    assert float(jnp.max(jnp.abs(result.means["x"] - exact_means))) < 1e-12
    assert float(jnp.max(jnp.abs(result.log_evidence - exact_evidence))) < 1e-12
    cov_gap = result.ensemble.cov("x").to_dense() - exact[-1].cov("x").to_dense()
    assert float(jnp.max(jnp.abs(cov_gap))) < 1e-12


def test_7_exactness_needs_more_particles_than_dimensions():
    """At J = d the sample covariance is rank deficient, and the filter is not exact."""
    problem = _linear(n_times=20, transition_noise_std=0.0)
    ens = problem.initial.sample(jax.random.key(0), n_particles=3)
    result = _filter(problem, ens)
    exact, _ = problem.exact_filter()
    exact_means = jnp.stack([g.mean("x") for g in exact])
    assert float(jnp.max(jnp.abs(result.means["x"] - exact_means))) > 1e-3


def test_8_an_inflation_that_draws_nothing_shifts_no_draw():
    problem = _linear()
    plain = _filter(problem, update_rule=MATHERON, key=jax.random.key(2))
    inflated = _filter(
        problem, update_rule=MATHERON, key=jax.random.key(2), inflation=_Identity()
    )
    _same(plain.ensemble, inflated.ensemble)
    assert np.array_equal(
        np.asarray(plain.log_evidence), np.asarray(inflated.log_evidence)
    )


def test_8_a_short_stochastic_filter_is_snapshotted():
    """A JAX-side PRNG change, or a change of the split, is detected."""
    problem = _linear(n_times=3)
    result = _filter(
        problem, _particles(problem, n_particles=4), update_rule=MATHERON,
        key=jax.random.key(11),
    )
    assert np.abs(np.asarray(result.ensemble["x"]) - np.array(_SNAPSHOT)).max() < 1e-12


_SNAPSHOT = [
    [-0.43319949585667095, -0.11944692858820022, 1.1015945658142456],
    [-0.5911968420579755, 0.004510092578616481, 1.0221780663228037],
    [-0.792053639824885, 0.2530890767174613, 0.04955122187815511],
    [-0.3140243196739727, 0.05492858848270424, 0.5929676022537959],
]


def test_9_the_policies_are_called_as_specified():
    problem = _linear(n_times=3)
    inflation = _Recorder()
    relaxation = _Recorder(relaxation=True)
    key = jax.random.key(9)
    result = _filter(
        problem, inflation=inflation, relaxation=relaxation, key=key,
        update_rule=MATHERON, keep_ensembles=True,
    )
    assert len(inflation.calls) == len(relaxation.calls) == 3
    for t, ((args, kwargs), (_, relax_kwargs)) in enumerate(
        zip(inflation.calls, relaxation.calls, strict=True)
    ):
        key, _, k_inflate, _ = jax.random.split(key, 4)
        assert np.array_equal(
            np.asarray(jax.random.key_data(args[0])),
            np.asarray(jax.random.key_data(k_inflate)),
        )
        assert kwargs["time"] == t and relax_kwargs["time"] == t
        # the relaxation's prior is the background, here the forecast itself
        _same(relax_kwargs["prior"], kwargs["ensemble"])
        _same(relax_kwargs["posterior"], result.ensembles[t])


def test_9_relaxation_relaxes_toward_the_inflated_background():
    """Settles #71: RTPS at alpha = 1 restores the *inflated* forecast spread."""
    problem = _linear(n_times=1, transition_noise_std=0.0)
    ens = _particles(problem, n_particles=16)
    lam = 1.3
    result = _filter(
        problem, ens, inflation=MultiplicativeInflation(lam),
        relaxation=RelaxToPriorSpread(1.0), keep_ensembles=True,
    )
    forecast = enkf.forecast(ens, problem.transition, state="x")
    want = lam * jnp.std(forecast["x"], axis=0, ddof=1)
    got = jnp.std(result.ensemble["x"], axis=0, ddof=1)
    assert float(jnp.max(jnp.abs(got - want))) < 1e-12


def test_9_a_policy_changing_the_dtype_is_named():
    problem = _linear()

    def demote(key, *, ensemble, **context):
        return Ensemble(x=ensemble["x"].astype(jnp.float32))

    with pytest.raises(TypeError, match="the inflation .*dtype float32"):
        _filter(problem, inflation=demote, key=jax.random.key(0))

    def demote_relaxed(*, prior, posterior, **context):
        return Ensemble(x=posterior["x"].astype(jnp.float32))

    with pytest.raises(TypeError, match="the relaxation .*dtype float32"):
        _filter(problem, relaxation=demote_relaxed, key=jax.random.key(0))

    def drop(*, prior, posterior, **context):
        return Ensemble(x=posterior["x"][:-1])

    with pytest.raises(ValueError, match="the relaxation .*particles"):
        _filter(problem, relaxation=drop, key=jax.random.key(0))


class _FailingTransition:
    """The problem's transition, returning ``nan`` rows on call ``fail_at``."""

    def __init__(self, transition, fail_at):
        self.transition = transition
        self.fail_at = fail_at
        self.calls = 0

    def __call__(self, x):
        out = self.transition(x)
        self.calls += 1
        if self.calls - 1 == self.fail_at:
            out = out.at[0].set(jnp.nan)
        return out


def test_10_a_failed_transition_raises_and_the_error_resumes_the_filter():
    problem = _linear(n_times=8)
    ens = _particles(problem)
    key = jax.random.key(13)
    common = dict(update_rule=MATHERON, inflation=MultiplicativeInflation(1.05))
    whole = _filter(problem, ens, key=key, keep_ensembles=True, **common)
    failing = _FailingTransition(problem.transition, fail_at=5)
    with pytest.raises(enkf.EnKFError, match="transition .* at time 5") as caught:
        _filter(problem, ens, key=key, transition=failing, keep_ensembles=True, **common)
    exc = caught.value
    assert exc.time == 5 and exc.result.n_times == 5
    _same(exc.result.ensemble, whole.ensembles[4])
    rest = _filter(
        problem, exc.result.ensemble, key=exc.key,
        observations=problem.observations[exc.time:], **common,
    )
    _same(rest.ensemble, whole.ensemble)
    joined = jnp.concatenate([exc.result.means["x"], rest.means["x"]])
    assert np.array_equal(np.asarray(joined), np.asarray(whole.means["x"]))


class _TimedInflation:
    """Multiplicative inflation by ``1 + 0.01 * time``: a policy that reads ``time``."""

    def __init__(self):
        self.times = []

    def __call__(self, key, *, ensemble, time, **context):
        self.times.append(time)
        return kalman.inflate_multiplicative(ensemble, 1.0 + 0.01 * time)


def test_10_resuming_reproduces_draws_and_times_with_every_source_of_randomness():
    """Transition noise, additive inflation, a stochastic rule, a timed policy."""
    problem = _noisy(n_times=8)
    ens = _particles(problem)
    key = jax.random.key(17)

    def run(transition, observations, start, ensemble, k, inflation):
        return _filter(
            problem, ensemble, key=k, transition=transition, update_rule=MATHERON,
            observations=observations, start_time=start, keep_ensembles=True,
            inflation=lambda kk, *, ensemble, **c: AdditiveInflation(
                x=PSDDiagonal(jnp.full(3, 0.01))
            )(kk, ensemble=inflation(kk, ensemble=ensemble, **c)),
        )

    timed_whole = _TimedInflation()
    whole = run(problem.transition, problem.observations, 0, ens, key, timed_whole)
    timed_first = _TimedInflation()
    failing = _FailingTransition(problem.transition, fail_at=5)
    with pytest.raises(enkf.EnKFError) as caught:
        run(failing, problem.observations, 0, ens, key, timed_first)
    exc = caught.value
    assert exc.time == 5
    timed_rest = _TimedInflation()
    rest = run(
        problem.transition, problem.observations[exc.time:], exc.time,
        exc.result.ensemble, exc.key, timed_rest,
    )
    assert timed_first.times + timed_rest.times == timed_whole.times == list(range(8))
    _same(rest.ensemble, whole.ensemble)
    joined = jnp.concatenate([exc.result.log_evidence, rest.log_evidence])
    assert np.array_equal(np.asarray(joined), np.asarray(whole.log_evidence))


def test_10_without_start_time_a_timed_policy_sees_other_times():
    """Why ``start_time`` exists: the draws agree, the times do not."""
    problem = _linear(n_times=4)
    timed = _TimedInflation()
    _filter(problem, observations=problem.observations[2:], inflation=timed)
    assert timed.times == [0, 1]
    timed = _TimedInflation()
    _filter(problem, observations=problem.observations[2:], inflation=timed,
            start_time=2)
    assert timed.times == [2, 3]


def test_10_a_failure_at_the_first_time_carries_the_initial_ensemble():
    problem = _linear()
    ens = _particles(problem)
    with pytest.raises(enkf.EnKFError) as caught:
        _filter(problem, ens, transition=_FailingTransition(problem.transition, 0))
    exc = caught.value
    assert exc.time == 0 and exc.key is None
    _same(exc.result.ensemble, ens)
    assert exc.result.means["x"].shape == (0, 3) and exc.result.log_evidence.shape == (0,)


@pytest.mark.parametrize(
    "stage", ["inflation", "observation model", "update", "relaxation"]
)
def test_10_each_later_stage_is_named_when_it_fails(stage):
    problem = _linear()

    def nan_inflation(key, *, ensemble, **context):
        return ensemble.assign(x=ensemble["x"].at[0, 0].set(jnp.nan))

    def nan_relaxation(*, prior, posterior, **context):
        return posterior.assign(x=posterior["x"].at[0, 0].set(jnp.inf))

    class NanRule:
        def build(self, particles, approximation, given):
            def call(values=None, /, *, key=None, **block_values):
                x = particles["x"].at[1, 1].set(jnp.nan)
                return Ensemble(x=x)

            return call

        def __repr__(self):
            return "NanRule()"

    def nan_observe(x):
        return problem.observe(x).at[2, 0].set(jnp.nan)

    kwargs = {
        "inflation": dict(inflation=nan_inflation),
        "observation model": dict(observe=nan_observe),
        "update": dict(update_rule=NanRule()),
        "relaxation": dict(relaxation=nan_relaxation),
    }[stage]
    with pytest.raises(enkf.EnKFError, match=f"the {stage}"):
        _filter(problem, **kwargs)


def test_11_the_filter_refuses_bad_input_before_calling_the_transition():
    problem = _linear()
    ens = _particles(problem)
    never = _FailingTransition(problem.transition, fail_at=-1)
    bad_rows = problem.observations.at[2, 0].set(jnp.nan)
    with pytest.raises(ValueError, match=r"rows \[2\]"):
        _filter(problem, ens, observations=bad_rows, transition=never)
    with pytest.raises(ValueError, match=r"shape \(T, 2\)"):
        _filter(problem, ens, observations=problem.observations[:, :1], transition=never)
    with pytest.raises(ValueError, match="at least one row"):
        _filter(problem, ens, observations=problem.observations[:0], transition=never)
    weighted = Ensemble(x=ens["x"], log_weights=jnp.zeros(8))
    with pytest.raises(ValueError, match="weighted"):
        _filter(problem, weighted, transition=never)
    with pytest.raises(ValueError, match="non-finite"):
        _filter(problem, ens.assign(x=ens["x"].at[0, 0].set(jnp.inf)), transition=never)
    with pytest.raises(ValueError, match="named 'prediction'"):
        _filter(problem, ens.assign(prediction=jnp.ones((8, 1))), transition=never)
    with pytest.raises(TypeError, match="keep_ensembles"):
        _filter(problem, ens, keep_ensembles=1, transition=never)
    with pytest.raises(KeyError, match="observe_inputs"):
        _filter(problem, ens, observe_inputs="y", transition=never)
    with pytest.raises(TypeError, match="observe must be callable"):
        _filter(problem, ens, observe=3, transition=never)
    with pytest.raises(ValueError, match="key is required"):
        _filter(problem, ens, transition_noise=PSDDiagonal(jnp.ones(3)), transition=never)
    for bad in (-1, 1.0, True):
        with pytest.raises(TypeError, match="start_time"):
            _filter(problem, ens, start_time=bad, transition=never)
    assert never.calls == 0


# ===========================================================================
# 12-15: JAX, dtypes, blocks
# ===========================================================================


def test_12_compilations_do_not_grow_with_the_number_of_times():
    """Nothing compiles per time; a new length compiles the same few operations.

    Counted by JAX's own compilation events. A few operations depend on the
    length of ``observations`` (its finiteness check, its rows, the stacked
    means), so each new length compiles them once. A compilation per time
    would make the step from 30 to 60 times cost more than the step from 3
    to 30.
    """
    problem = _noisy(n_times=60)
    configuration = dict(
        update_rule=MATHERON,
        inflation=MultiplicativeInflation(1.01),
        relaxation=RelaxToPriorSpread(0.2),
        key=jax.random.key(0),
    )
    prefixes = [problem.observations[:n] for n in (3, 30, 60)]
    counter = _CompileCounter()
    jax.monitoring.register_event_duration_secs_listener(counter)
    try:
        counts = []
        for observations in prefixes:
            _filter(problem, observations=observations, **configuration)
            counts.append(counter.count)
    finally:
        jax.monitoring.unregister_event_duration_listener(counter)
    per_length = counts[1] - counts[0]
    assert counts[2] - counts[1] == per_length, f"compilations by run: {counts}"
    assert per_length <= 8, f"compilations by run: {counts}"


@pytest.mark.parametrize("rule", [SSR, MATHERON], ids=["ssr", "matheron"])
def test_13_a_float32_filter_stays_float32(rule):
    problem = _linear()
    f32 = jnp.float32
    ens = Ensemble(x=_particles(problem)["x"].astype(f32))
    result = enkf.filter(
        ens, problem.observations.astype(f32),
        transition=maps.Linear(Dense(problem.transition.op.to_dense().astype(f32))),
        observe=maps.Linear(Dense(problem.observe.op.to_dense().astype(f32))),
        noise_cov=PSDDiagonal(jnp.full(2, 0.25, f32)),
        transition_noise=PSDDiagonal(jnp.full(3, 0.09, f32)),
        update_rule=rule,
        inflation=MultiplicativeInflation(1.05),
        relaxation=RelaxToPriorSpread(0.5),
        key=jax.random.key(0),
    )
    assert result.ensemble["x"].dtype == f32
    assert result.means["x"].dtype == f32
    assert result.log_evidence.dtype == f32
    mixed = enkf.filter(
        ens, problem.observations,               # float64 observations and noise
        transition=maps.Linear(Dense(problem.transition.op.to_dense().astype(f32))),
        observe=maps.Linear(Dense(problem.observe.op.to_dense().astype(f32))),
        noise_cov=problem.noise_cov, update_rule=rule, key=jax.random.key(0),
    )
    assert mixed.ensemble["x"].dtype == f32 and mixed.log_evidence.dtype == f32


def test_14_a_parameter_the_transition_reads_is_estimated():
    """State augmentation: the scale of a linear transition is learned from data.

    The data were generated with scale 1; the prior is 0.8 +- 0.1. The lagged
    copy of Example 13, the smoother, is test 16's.
    """
    problem = _noisy(n_times=200)
    A = problem.transition.op.to_dense()

    def transition(x, scale):
        return scale * (x @ A.T)

    prior_scale = 0.8 + 0.1 * jax.random.normal(jax.random.key(1), (200, 1))
    ens = _particles(problem, n_particles=200).assign(scale=prior_scale)
    result = enkf.filter(
        ens, problem.observations, transition=transition, observe=problem.observe,
        noise_cov=problem.noise_cov, update_rule=MATHERON,
        transition_inputs=("x", "scale"), transition_noise=problem.transition_noise,
        key=jax.random.key(2),
    )
    assert tuple(result.means) == ("x", "scale")
    assert abs(float(result.means["scale"][-1, 0]) - 1.0) < 0.05
    assert float(jnp.std(result.ensemble["scale"])) < 0.1 / 3


def test_15_keep_ensembles_keeps_every_analysis():
    problem = _linear(n_times=4)
    result = _filter(problem, keep_ensembles=True)
    assert len(result.ensembles) == 4
    _same(result.ensembles[-1], result.ensemble)
    for t, ens in enumerate(result.ensembles):
        assert np.array_equal(np.asarray(ens.mean("x")), np.asarray(result.means["x"][t]))
    assert _filter(problem).ensembles is None


def test_15_the_result_validates_and_prints():
    problem = _linear(n_times=4)
    result = _filter(problem)
    assert repr(result) == "FilterResult(n_times=4, blocks=('x',))"
    assert result.n_times == 4
    assert float(result.total_log_evidence) == float(jnp.sum(result.log_evidence))
    with pytest.raises(ValueError, match="blocks"):
        enkf.FilterResult(
            ensemble=result.ensemble, means={}, log_evidence=result.log_evidence
        )
    with pytest.raises(ValueError, match=r"means\['x'\]"):
        enkf.FilterResult(
            ensemble=result.ensemble, means={"x": jnp.zeros((3, 3))},
            log_evidence=result.log_evidence,
        )


# ===========================================================================
# 16: the design's examples
# ===========================================================================


@pytest.fixture(scope="module")
def lorenz():
    return toy.lorenz96(state_dim=40, n_times=300, obs_every=2, noise_std=1.0)


def _lorenz_error(problem, n_particles, **kwargs):
    ens = problem.initial.sample(jax.random.key(0), n_particles=n_particles)
    result = enkf.filter(
        ens, problem.observations, transition=problem.transition,
        observe=problem.observe, noise_cov=problem.noise_cov,
        inflation=MultiplicativeInflation(1.05), key=jax.random.key(1), **kwargs,
    )
    return float(_rmse(result.means["x"], problem.truth)[50:].mean())


def test_16_example_2_tracks_the_lorenz96_truth(lorenz):
    assert _lorenz_error(lorenz, 40, update_rule=MATHERON) < 0.6


def _periodic(point, points):
    d = jnp.abs(points[:, 0] - point[0])
    return jnp.minimum(d, 40.0 - d)


def test_16_example_12_localization_beats_the_global_filter(lorenz):
    localization = kalman.DomainLocalization(
        target_coords={"x": lorenz.coords}, given_coords=lorenz.obs_coords,
        radius=8.0, max_neighbors=10, distance=_periodic,
    )
    error_global = _lorenz_error(lorenz, 10, update_rule=SSR)
    for rule in (SSR, MATHERON):
        local = _lorenz_error(
            lorenz, 10, update_rule=kalman.LocalizedUpdateRule(rule, localization)
        )
        # 0.35 to 0.54 over the problem's seeds 0 to 5; CI's trajectory gave 0.54
        assert local < 0.7 and local < 0.5 * error_global


def test_16_example_11_a_hybrid_beats_the_plain_filter(lorenz):
    d = jnp.abs(jnp.arange(40)[:, None] - jnp.arange(40)[None, :])
    d = jnp.minimum(d, 40 - d)
    B = DensePSD(0.3 * jnp.exp(-0.5 * d**2) + 1e-6 * jnp.eye(40))

    def hybrid(ensemble, noise, alpha=0.2):
        ((given, R),) = noise.items()
        fit = ensemble.marginal("x").project()
        blended = Gaussian(
            {"x": fit.mean("x")},
            factors={"x": fit.factor("x") * math.sqrt(alpha)},
            block_covs={"x": B * (1.0 - alpha)},
        )
        return blended.pipe(
            maps.pushforward, lorenz.observe, inputs="x", output=given
        ).add_noise({given: R})

    plain = _lorenz_error(lorenz, 10, update_rule=MATHERON)
    blended = _lorenz_error(lorenz, 10, update_rule=MATHERON, approximation=hybrid)
    assert blended < 0.5 * plain
    with pytest.raises(ValueError):
        _lorenz_error(lorenz, 10, update_rule=SSR, approximation=hybrid)


def test_16_example_13_augmentation_recovers_the_forcing_and_smooths():
    problem = toy.lorenz96(n_times=200)
    key = jax.random.key(0)
    k1, k2, key = jax.random.split(key, 3)
    ens = Ensemble(
        x=problem.initial.sample(k1, 40)["x"],
        forcing=6.0 + jax.random.normal(k2, (40, 1)),
    )
    filtered, smoothed = [], []
    for y in problem.observations:
        key, k_analysis = jax.random.split(key)
        ens = ens.assign(x_prev=ens["x"])
        ens = enkf.forecast(ens, toy.lorenz96_step, state="x", inputs=("x", "forcing"))
        ens = kalman.inflate_multiplicative(ens, 1.05)
        ens, _ = enkf.analysis(
            ens, y, observe=problem.observe, noise_cov=problem.noise_cov,
            update_rule=MATHERON, inputs=("x",), key=k_analysis,
        )
        filtered.append(ens.mean("x"))
        smoothed.append(ens.mean("x_prev"))
    filtered, smoothed = jnp.stack(filtered), jnp.stack(smoothed)
    filter_error = _rmse(filtered[:-1], problem.truth[:-1])[50:].mean()
    smoother_error = _rmse(smoothed[1:], problem.truth[:-1])[50:].mean()
    assert smoother_error < filter_error
    assert abs(float(ens.mean("forcing")[0]) - 8.0) < 0.3


# ===========================================================================
# 17: the toy problems
# ===========================================================================


def test_17_lorenz96_is_reproducible_and_consistent():
    a = toy.lorenz96(state_dim=8, n_times=5, obs_every=3, seed=4)
    b = toy.lorenz96(state_dim=8, n_times=5, obs_every=3, seed=4)
    assert np.array_equal(np.asarray(a.observations), np.asarray(b.observations))
    other = toy.lorenz96(state_dim=8, n_times=5, obs_every=3, seed=5)
    assert not np.array_equal(np.asarray(a.observations), np.asarray(other.observations))
    assert repr(a) == "Lorenz96(state_dim=8, data_dim=3, n_times=5)"
    assert a.state_dim == 8 and a.data_dim == 3 and a.n_times == 5
    assert np.array_equal(np.asarray(a.obs_coords[:, 0]), [0.0, 3.0, 6.0])
    assert np.array_equal(np.asarray(a.coords[:, 0]), np.arange(8.0))
    x0 = a.initial.mean("x")
    assert np.allclose(a.truth[0], toy.lorenz96_step(x0), atol=1e-14)
    assert np.allclose(a.truth[1:], toy.lorenz96_step(a.truth[:-1]), atol=1e-14)
    assert np.allclose(a.transition(a.truth[:-1]), a.truth[1:], atol=1e-14)


def test_17_lorenz96_observations_carry_noise_of_the_stated_size():
    problem = toy.lorenz96(state_dim=40, n_times=500, noise_std=0.5)
    residual = problem.observations - problem.truth[:, ::2]
    assert abs(float(jnp.std(residual)) - 0.5) < 0.01
    assert np.allclose(problem.noise_cov.diag(), 0.25)


def test_17_lorenz96_step_matches_the_equations_and_broadcasts_the_forcing():
    x = jax.random.normal(jax.random.key(0), (3, 6))
    forcing = jnp.array([[7.0], [8.0], [9.0]])

    def f(z, F):
        return (jnp.roll(z, -1) - jnp.roll(z, 2)) * jnp.roll(z, 1) - z + F

    for j in range(3):
        F = float(forcing[j, 0])
        dt = 0.05
        k1 = f(x[j], F)
        k2 = f(x[j] + dt / 2 * k1, F)
        k3 = f(x[j] + dt / 2 * k2, F)
        k4 = f(x[j] + dt * k3, F)
        want = x[j] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        assert np.allclose(toy.lorenz96_step(x, forcing)[j], want, atol=1e-14)
    # the stencil, at one site: dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F
    z = jnp.arange(6.0)
    assert float(f(z, 0.0)[3]) == (4.0 - 1.0) * 2.0 - 3.0


def test_17_the_toy_keys_do_not_coincide_with_a_split_of_the_same_seed():
    """#73's hazard: a caller's ``split(key(seed), n)`` never reproduces a toy draw.

    The observation matrix is ``normal(k, (N, d)) / sqrt(d)`` for the toy's own
    key ``k``; no key from a split of the same seed gives it.
    """
    problem = toy.linear_state_space(state_dim=4, data_dim=3, n_times=1, seed=0)
    H = np.asarray(problem.observe.op.to_dense()) * 2.0
    for n in range(1, 7):
        for k in jax.random.split(jax.random.key(0), n):
            assert not np.allclose(np.asarray(jax.random.normal(k, (3, 4))), H)
    assert not np.allclose(np.asarray(jax.random.normal(jax.random.key(0), (3, 4))), H)


def test_17_exact_filter_matches_the_dense_recursion():
    problem = _noisy(n_times=20)
    exact, evidence = problem.exact_filter()
    A = problem.transition.op.to_dense()
    H = problem.observe.op.to_dense()
    Q = problem.transition_noise.to_dense()
    R = problem.noise_cov.to_dense()
    m, P = jnp.zeros(3), jnp.eye(3)
    for t, y in enumerate(problem.observations):
        m, P = A @ m, A @ P @ A.T + Q
        S = H @ P @ H.T + R
        want = st.multivariate_normal.logpdf(y, H @ m, S)
        assert abs(float(evidence[t]) - float(want)) < 1e-12
        K = P @ H.T @ jnp.linalg.inv(S)
        m, P = m + K @ (y - H @ m), P - K @ S @ K.T
        assert np.allclose(exact[t].mean("x"), m, atol=1e-12)
        assert np.allclose(exact[t].cov("x").to_dense(), P, atol=1e-12)
        assert exact[t].latent_dim <= 3


def test_17_linear_state_space_is_consistent_and_stable():
    problem = toy.linear_state_space(state_dim=5, data_dim=4, n_times=30, seed=2)
    A = np.asarray(problem.transition.op.to_dense())
    assert np.allclose(np.abs(np.linalg.eigvals(A)), 0.95)
    assert repr(problem) == "LinearStateSpace(state_dim=5, data_dim=4, n_times=30)"
    H = problem.observe.op.to_dense()
    residual = problem.observations - problem.truth @ H.T
    assert residual.shape == (30, 4)
    large = toy.linear_state_space(state_dim=4, data_dim=3, n_times=4000, noise_std=0.5)
    residual = large.observations - large.truth @ large.observe.op.to_dense().T
    assert abs(float(jnp.std(residual)) - 0.5) < 0.01
    shocks = large.truth[1:] - large.truth[:-1] @ large.transition.op.to_dense().T
    assert abs(float(jnp.std(shocks)) - 0.3) < 0.01
    assert toy.linear_state_space(transition_noise_std=0.0).transition_noise is None


def test_17_the_classes_validate_their_fields():
    problem = _linear()
    fields = dict(
        transition=problem.transition, transition_noise=problem.transition_noise,
        observe=problem.observe, noise_cov=problem.noise_cov, initial=problem.initial,
        observations=problem.observations, truth=problem.truth,
    )
    toy.LinearStateSpace(**fields)
    with pytest.raises(ValueError, match="truth"):
        toy.LinearStateSpace(**{**fields, "truth": problem.truth[:-1]})
    with pytest.raises(ValueError, match="transition_noise|side"):
        toy.LinearStateSpace(**{**fields, "transition_noise": PSDDiagonal(jnp.ones(2))})
    with pytest.raises(TypeError, match="maps.Linear"):
        toy.LinearStateSpace(**{**fields, "transition": lambda x: x})
    with pytest.raises(ValueError, match="'x'"):
        toy.LinearStateSpace(**{**fields, "initial": problem.initial.rename(x="u")})
    lorenz = toy.lorenz96(state_dim=8, n_times=3)
    with pytest.raises(ValueError, match="obs_coords"):
        toy.Lorenz96(
            forcing=8.0, dt=0.05, initial=lorenz.initial, observe=lorenz.observe,
            noise_cov=lorenz.noise_cov, observations=lorenz.observations,
            truth=lorenz.truth, obs_coords=lorenz.obs_coords[:-1],
        )
    with pytest.raises(ValueError, match="transition"):
        lorenz.transition(jnp.zeros(8))
    with pytest.raises(ValueError, match="at least 4"):
        toy.lorenz96(state_dim=3)
    with pytest.raises(ValueError, match="obs_every"):
        toy.lorenz96(state_dim=8, obs_every=9)
    with pytest.raises(ValueError, match="transition_noise_std"):
        toy.linear_state_space(transition_noise_std=-0.1)


# ===========================================================================
# 18: the user guide's page
# ===========================================================================


def test_18_the_user_guide_page_runs_and_says_what_it_does():
    """Every Python block of ``docs/user-guide/filtering.md``, in order.

    Lorenz-96 is chaotic, so its digits are not portable: a last-bit
    difference in the arithmetic (CI's Linux against a developer's macOS)
    changes the toy trajectory. The page's claims are checked in bands that
    held at the problem's seeds 0 to 5; the linear problem's exactness is
    portable and checked exactly.
    """
    page = Path(__file__).parents[1] / "docs" / "user-guide" / "filtering.md"
    blocks = re.findall(r"```python\n(.*?)```", page.read_text(), re.S)
    assert len(blocks) == 6
    ns: dict = {}
    for block in blocks:
        exec(compile(block, str(page), "exec"), ns)
    # "well below the observation noise" (0.31 to 0.42 over seeds 0 to 5)
    assert float(ns["error"]) < 0.6
    # "ranks the four the same way as the error" (all six seeds)
    assert np.argsort(ns["evidence"])[::-1].tolist() == np.argsort(ns["errors"]).tolist()
    assert float(ns["mean_gap"]) < 1e-14 and float(ns["evidence_gap"]) < 1e-13
    # "loses the truth" (3.6 to 4.6) and localization tracks it (0.35 to 0.41)
    assert float(ns["error_global"]) > 2.0 and float(ns["error_local"]) < 0.6
    # "close to 8" (7.88 to 8.03)
    assert abs(float(ns["forcing"]) - 8.0) < 0.2
