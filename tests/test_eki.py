"""Conformance and regression tests for ``enskit.algorithms.eki``.

The file has two sections. The first works through the numbered conformance
obligations of the "Ensemble Kalman Inversion contract"; the second holds one
targeted regression test per class of silent failure that contract names.
Each documents why a rule of the contract exists, and deleting one as
redundant loses that. The contract's section *Ported regression tests* maps
every test of the old ``enskit.eki`` suite to its port here.

Two rules govern the reference throughout:

- **The dense reference is hand-written here**, in plain NumPy over means,
  anomalies and materialized operators, never routed through the layers under
  test, so every comparison is between two independent paths.
- **Exactness tests compare against closed forms**, at a tolerance of a few
  machine epsilons times the natural scale of the quantity, never one chosen
  to make the test pass.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import warnings

import jax
import jax.monitoring
import jax.numpy as jnp
import numpy as np
import pytest

import enskit  # noqa: F401  -- enables x64 before any array exists
from enskit import kalman, maps
from enskit.algorithms import (
    AdditiveInflation,
    MultiplicativeInflation,
    RelaxToPriorPerturbations,
    RelaxToPriorSpread,
    eki,
)
from enskit.algorithms.eki import (
    INTERRUPTED,
    PREDICTION,
    SCHEDULE_EXHAUSTED,
    STOPPING_RULE,
    AdaptiveESSSchedule,
    AdaptiveMisfitSchedule,
    DiscrepancyStop,
    EKIError,
    EKIResult,
    EKIState,
    Evaluation,
    FixedSchedule,
    HistoryRecord,
    advance,
    assimilate,
    effective_sample_size,
    evaluate,
    iterate,
    misfits,
    repair_failed_particles,
    run,
)
from enskit.distribution import Ensemble, Gaussian
from enskit.linalg import (
    DensePSD,
    Identity,
    PSDDiagonal,
    PSDLowRank,
    UnsupportedOpError,
    block_diag,
    debug_checks,
)
from enskit.testing import check_schedule

RNG = np.random.default_rng(0)
EPS = float(np.finfo(np.float64).eps)
SQRT = kalman.SymmetricSquareRoot()
MATHERON = kalman.Matheron()


# ---------------------------------------------------------------------------
# fixtures: an affine problem, and its dense posterior
# ---------------------------------------------------------------------------


def _psd(n: int, seed: int = 3) -> np.ndarray:
    """A well-conditioned dense PSD matrix, as a NumPy array."""
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    return M @ M.T + n * np.eye(n)


def _exact_moment_ensemble(J: int, mu: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Particles whose sample moments are exactly ``mu`` and ``F @ F.T``.

    The QR-of-ones construction: take the complete QR of the all-ones vector
    in R^J, let E be its next k columns, orthonormal and each orthogonal to
    the ones vector, and set the particles to ``mu + sqrt(J - 1) E F^T``. Only
    ``J >= k + 1`` binds.
    """
    k = F.shape[1]
    assert J >= k + 1, "the construction needs J >= k + 1"
    Q, _ = np.linalg.qr(np.ones((J, 1)), mode="complete")
    E = Q[:, 1 : k + 1]
    return mu + np.sqrt(J - 1) * E @ F.T


class _AffineProblem:
    """An affine forward model, its prior, and the dense posterior it implies."""

    def __init__(
        self,
        P: int = 3,
        N: int = 5,
        J: int = 12,
        seed: int = 7,
        prior_rank: int | None = None,
        noise_scale: float = 1.0,
    ):
        rng = np.random.default_rng(seed)
        self.P, self.N, self.J = P, N, J
        self.G = rng.normal(size=(N, P))
        self.m0 = rng.normal(size=P)
        self.C0 = _psd(P, seed=seed + 1)
        self.R = noise_scale * _psd(N, seed=seed + 2)
        self.y = rng.normal(size=N)
        self.factor = np.linalg.cholesky(self.C0)
        if prior_rank is not None:
            self.factor = self.factor[:, :prior_rank]
            self.C0 = self.factor @ self.factor.T
        self.members = _exact_moment_ensemble(J, self.m0, self.factor)
        self.noise_cov = DensePSD(jnp.asarray(self.R))
        self.calls: list[np.ndarray] = []

    def forward(self, u):
        """The model, recording every input it is given."""
        u = jnp.asarray(u)
        self.calls.append(np.asarray(u))
        return u @ jnp.asarray(self.G).T

    def state(self, seed: int = 0) -> EKIState:
        return EKIState(Ensemble(u=jnp.asarray(self.members)), key=jax.random.key(seed))

    def prior(self) -> Gaussian:
        return Gaussian(
            {"u": jnp.asarray(self.m0)}, block_covs={"u": DensePSD(jnp.asarray(self.C0))}
        )

    def posterior(self, level: float = 1.0):
        """The exact Gaussian posterior at tempering level ``level``."""
        prior_precision = np.linalg.inv(self.C0)
        data_precision = self.G.T @ np.linalg.solve(self.R, self.G)
        cov = np.linalg.inv(prior_precision + level * data_precision)
        mean = cov @ (
            prior_precision @ self.m0
            + level * self.G.T @ np.linalg.solve(self.R, self.y)
        )
        return mean, cov


def _u(x) -> np.ndarray:
    """The ``"u"`` block of a state's, a result's or an ensemble's particles."""
    ensemble = x if isinstance(x, Ensemble) else x.ensemble
    return np.asarray(ensemble["u"])


def _moments(members) -> tuple[np.ndarray, np.ndarray]:
    """Sample mean and covariance, divisor J - 1, of row-wise particles."""
    members = np.asarray(members)
    J = members.shape[0]
    A = members - members.mean(axis=0)
    return members.mean(axis=0), A.T @ A / (J - 1)


def _run(problem, **kwargs):
    """``run`` on the problem's own state, data and noise, square-root by default."""
    kwargs.setdefault("update_rule", SQRT)
    state = kwargs.pop("state", None) or problem.state()
    forward = kwargs.pop("forward", problem.forward)
    return run(state, forward, jnp.asarray(problem.y), problem.noise_cov, **kwargs)


def _recovered_whitener(noise_cov, n: int) -> np.ndarray:
    """The operator's own whitener, recovered by whitening the identity."""
    return np.asarray(noise_cov.whiten(jnp.eye(n))).T


def _keys(key):
    """The pinned per-step split, ``(next, inflate, evaluate, update)``."""
    return jax.random.split(key, 4)


#: Ladders whose increments sum to 1 *exactly* in binary floating point.
_EXACT_LADDERS = [
    (1.0,),
    (0.5, 0.5),
    (0.25, 0.25, 0.25, 0.25),
    (0.125,) * 8,
    (0.5, 0.25, 0.25),
    (0.75, 0.125, 0.0625, 0.0625),
    (0.0625, 0.4375, 0.25, 0.25),
]


# ===========================================================================
# Section 1 -- the conformance obligations
# ===========================================================================


@pytest.mark.parametrize("increments", _EXACT_LADDERS)
def test_1_the_ladder_telescopes_to_one_shot_conditioning(increments):
    """A ladder summing to 1 reproduces the exact posterior, to round-off.

    Every clause of the claim is supplied: an affine model, a Gaussian prior,
    particles whose sample moments equal the prior's exactly, the square-root
    rule, no inflation or relaxation, no failed particles, and increments
    summing exactly to 1.
    """
    problem = _AffineProblem()
    result = _run(problem, schedule=FixedSchedule(increments))
    assert result.n_evaluations == len(increments)
    got_mean, got_cov = _moments(_u(result))
    want_mean, want_cov = problem.posterior()

    scale = max(np.abs(want_mean).max(), np.abs(want_cov).max())
    tolerance = 1e3 * EPS * scale
    assert np.abs(got_mean - want_mean).max() < tolerance
    assert np.abs(got_cov - want_cov).max() < tolerance


def test_2_the_level_mis_scaling_is_caught_by_that_tolerance():
    """``R / beta`` instead of ``R / increment`` fails test 1 by a wide margin.

    The mis-scaling is written out locally, against ``kalman.update``
    directly. On a uniform T-step ladder it accumulates ``(T + 1)/2`` times
    the data precision instead of one: the result lands on the exact
    posterior at that level, orders of magnitude outside test 1's tolerance.
    """
    problem = _AffineProblem()
    y = jnp.asarray(problem.y)
    for n_steps in (2, 5, 10):
        ens = Ensemble(u=jnp.asarray(problem.members))
        beta = 0.0
        for _ in range(n_steps):
            beta += 1.0 / n_steps
            ens = maps.pushforward(ens, problem.forward, inputs="u", output="g")
            ens = kalman.update(
                ens, g=y, noise={"g": problem.noise_cov / beta},  # the bug: level
                update_rule=SQRT,
            )

        got_mean, got_cov = _moments(_u(ens))
        want_mean, want_cov = problem.posterior(level=(n_steps + 1) / 2)
        scale = max(np.abs(want_mean).max(), np.abs(want_cov).max())
        assert np.abs(got_mean - want_mean).max() < 1e3 * EPS * scale
        assert np.abs(got_cov - want_cov).max() < 1e3 * EPS * scale

        correct_mean, correct_cov = problem.posterior()
        correct_scale = max(np.abs(correct_mean).max(), np.abs(correct_cov).max())
        disagreement = max(
            np.abs(got_mean - correct_mean).max(),
            np.abs(got_cov - correct_cov).max(),
        )
        assert disagreement > 1e9 * EPS * correct_scale


def test_3_the_stochastic_update_composes_to_the_same_posterior():
    """``Matheron`` reproduces a dense reference, and telescopes in expectation.

    Elementwise, for a fixed key, a ladder reproduces a hand-written dense
    perturbed-observation step with the pinned draws: the update key is the
    fourth of the step's split, and the noise is its second half's normal
    draw, whitened by the tempered noise covariance. In expectation over many
    keys the mean matches the posterior, to a tolerance derived from the
    ``K R K^T / J`` scale.
    """
    problem = _AffineProblem(J=16)
    increments = (0.25, 0.25, 0.5)

    state = problem.state(seed=5)
    members = np.asarray(problem.members)
    key = state.key
    for increment in increments:
        key_next, _, _, key_update = _keys(key)
        predictions = members @ problem.G.T
        members = _dense_matheron_step(
            members, predictions, problem, increment=increment, key=key_update
        )
        key = key_next
    result = _run(
        problem, state=state, schedule=FixedSchedule(increments), update_rule=MATHERON
    )
    scale = np.abs(members).max()
    assert np.abs(_u(result) - members).max() < 1e4 * EPS * scale

    want_mean, want_cov = problem.posterior()
    gain = want_cov @ problem.G.T @ np.linalg.inv(problem.R)
    noise_scale = np.sqrt(np.diag(gain @ problem.R @ gain.T).max() / problem.J)

    n_keys = 400
    means = np.zeros((n_keys, problem.P))
    for index in range(n_keys):
        drawn = _run(
            problem, state=problem.state(seed=1000 + index),
            schedule=FixedSchedule(increments), update_rule=MATHERON,
        )
        means[index] = _u(drawn).mean(axis=0)
    assert np.abs(means.mean(axis=0) - want_mean).max() < 3 * noise_scale / np.sqrt(
        n_keys
    )


def _dense_matheron_step(members, predictions, problem, *, increment, key):
    """One perturbed-observation step, in plain dense linear algebra."""
    J = members.shape[0]
    Au = members - members.mean(axis=0)
    Av = predictions - predictions.mean(axis=0)
    R = problem.R / increment
    gain = (Au.T @ Av / (J - 1)) @ np.linalg.inv(Av.T @ Av / (J - 1) + R)
    W = _recovered_whitener(problem.noise_cov / increment, problem.N)
    _, k_noise = jax.random.split(key)
    eps = np.asarray(jax.random.normal(k_noise, (J, problem.N)))
    perturbed = problem.y - predictions - eps @ np.linalg.inv(W).T
    return members + perturbed @ gain.T


def test_4_fixed_schedule_takes_its_increments_and_completes_under_its_own_bound():
    """Exactly its increments, exactly T evaluations, and no off-by-one bound."""
    problem = _AffineProblem()
    increments = (0.1, 0.4, 0.2, 0.3)
    result = _run(problem, schedule=FixedSchedule(increments), max_steps=len(increments))
    assert result.status == SCHEDULE_EXHAUSTED
    assert result.n_evaluations == len(increments)
    assert len(problem.calls) == len(increments)
    assert np.allclose(np.asarray(result.stacked.increment), increments)
    assert float(result.beta) == pytest.approx(sum(increments), abs=8 * EPS)


def test_4_a_schedule_returning_none_ends_the_run_with_a_terminal_record():
    """``next_increment`` may end the ladder on evidence only the evaluation has."""

    class _GiveUpAfterTwo:
        n_steps = None
        beta_target = None

        def next_increment(self, evaluation):
            return None if evaluation.step >= 2 else 0.25

    problem = _AffineProblem(J=10)
    result = _run(problem, schedule=_GiveUpAfterTwo())
    assert result.status == SCHEDULE_EXHAUSTED
    assert result.n_evaluations == 3
    assert float(result.stacked.increment[-1]) == 0.0
    assert float(result.stacked.beta_next[-1]) == float(result.stacked.beta[-1])
    # The literal float(J), at a J where exp(log J) differs from J.
    assert float(np.exp(np.log(problem.J))) != float(problem.J)
    assert float(result.stacked.ess[-1]) == float(problem.J)


@pytest.mark.parametrize("schedule_type", [AdaptiveESSSchedule, AdaptiveMisfitSchedule])
def test_4_both_adaptive_schedules_reach_their_budget_without_exceeding_it(
    schedule_type,
):
    problem = _AffineProblem()
    schedule = schedule_type(beta_target=1.0)
    result = _run(problem, schedule=schedule)
    beta = float(result.beta)
    assert beta <= 1.0 + 8 * EPS
    assert beta >= 1.0 - 1e-12
    assert np.all(np.asarray(result.stacked.increment) > 0.0)
    again = _run(problem, state=result.state, schedule=schedule)
    assert again.n_evaluations == 0
    assert again.n_completed_steps == 0


@pytest.mark.parametrize("schedule_type", [AdaptiveESSSchedule, AdaptiveMisfitSchedule])
def test_4_the_clamp_precedence_holds_in_all_three_regimes(schedule_type):
    """Floor beats criterion, ceiling binds, and the budget cap beats the floor."""
    evaluation = _evaluation_with_misfits(np.array([1.0, 1.0001, 0.9999, 1.00005]))

    spread_out = _evaluation_with_misfits(np.array([1.0, 400.0, 0.5, 900.0]))
    tiny = schedule_type(beta_target=None, min_increment=0.5, max_increment=2.0)
    unclamped = float(
        schedule_type(
            beta_target=None, min_increment=1e-12, max_increment=2.0
        ).next_increment(spread_out)
    )
    assert unclamped < 0.5, "the fixture must put the criterion below the floor"
    assert float(tiny.next_increment(spread_out)) == pytest.approx(0.5)

    capped = schedule_type(beta_target=None, min_increment=1e-6, max_increment=0.3)
    assert float(capped.next_increment(evaluation)) == pytest.approx(0.3)

    near_budget = _evaluation_with_misfits(
        np.array([1.0, 1.0001, 0.9999, 1.00005]), beta=0.99
    )
    budgeted = schedule_type(beta_target=1.0, min_increment=0.5, max_increment=2.0)
    assert float(budgeted.next_increment(near_budget)) == pytest.approx(0.01)


@pytest.mark.parametrize("schedule_type", [AdaptiveESSSchedule, AdaptiveMisfitSchedule])
def test_4_a_degenerate_ensemble_takes_the_largest_allowed_step(schedule_type):
    evaluation = _evaluation_with_misfits(np.full(5, 2.5))
    unbounded = schedule_type(beta_target=None, max_increment=0.4)
    assert float(unbounded.next_increment(evaluation)) == pytest.approx(0.4)

    part_way = _evaluation_with_misfits(np.full(5, 2.5), beta=0.75)
    budgeted = schedule_type(beta_target=1.0, max_increment=0.4)
    assert float(budgeted.next_increment(part_way)) == pytest.approx(0.25)


@pytest.mark.parametrize("schedule_type", [AdaptiveESSSchedule, AdaptiveMisfitSchedule])
def test_4_an_unbounded_ladder_runs_without_raising(schedule_type):
    """The executable form of the conditional budget term in the clamp."""
    problem = _AffineProblem()
    result = _run(
        problem,
        schedule=schedule_type(beta_target=None),
        stop=DiscrepancyStop(tau=50.0),
        max_steps=20,
    )
    assert result.status == STOPPING_RULE


def test_4_step_counts_are_exact_under_a_capped_criterion():
    """A budget of 1 under a ceiling of 0.3 takes exactly four steps.

    One test pins the ``>=`` in the exhaustion check, ``budget_tol``,
    cap-beats-floor, and the absence of a trailing dribble step. Every misfit
    is identical, so the ceiling alone chooses the increment.
    """

    class _ConstantModel:
        def __init__(self, n_particles, data_dim):
            self.shape = (n_particles, data_dim)
            self.calls = 0

        def __call__(self, u):
            self.calls += 1
            return jnp.ones(self.shape)

    problem = _AffineProblem()
    model = _ConstantModel(problem.J, problem.N)
    for schedule_type in (AdaptiveESSSchedule, AdaptiveMisfitSchedule):
        model.calls = 0
        result = _run(
            problem, forward=model,
            schedule=schedule_type(beta_target=1.0, max_increment=0.3),
        )
        assert result.n_evaluations == 4, schedule_type
        assert model.calls == 4
        got = np.asarray(result.stacked.increment)
        assert np.allclose(got, [0.3, 0.3, 0.3, 0.1], atol=1e-12)


def test_4_ess_bisection_returns_the_safe_end_and_matches_a_hand_written_one():
    """At a small ``n_bisect``, where returning ``lo`` and ``hi`` differ."""
    misfit_values = np.array([0.5, 2.0, 4.5, 9.0, 1.25, 3.0])
    evaluation = _evaluation_with_misfits(misfit_values)
    n_bisect = 8
    schedule = AdaptiveESSSchedule(
        beta_target=None, min_increment=1e-9, max_increment=4.0, n_bisect=n_bisect
    )
    got = float(schedule.next_increment(evaluation))
    target = 0.5 * misfit_values.size
    assert _reference_ess(misfit_values, got) >= target
    assert 1e-9 < got < 4.0

    lo, hi = 0.0, 4.0
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if _reference_ess(misfit_values, mid) >= target:
            lo = mid
        else:
            hi = mid
    assert got == pytest.approx(lo, abs=1e-14)


def test_4_the_misfit_schedule_returns_the_larger_of_its_two_bounds():
    """One test per regime, decided by the misfits' coefficient of variation."""
    theta = 2.0
    spread_out = np.array([0.05, 4.0, 0.1, 8.0, 0.2])
    mean, variance = spread_out.mean(), spread_out.var(ddof=1)
    assert np.sqrt(variance) / mean > 1 / np.sqrt(theta)
    schedule = AdaptiveMisfitSchedule(
        beta_target=None, divergence_budget=theta, max_increment=1e6
    )
    got = float(schedule.next_increment(_evaluation_with_misfits(spread_out)))
    assert got == pytest.approx(theta / mean, rel=1e-12)
    assert theta / mean > np.sqrt(theta / variance)

    clustered = np.array([10.0, 10.4, 9.7, 10.1, 9.9])
    mean, variance = clustered.mean(), clustered.var(ddof=1)
    assert np.sqrt(variance) / mean < 1 / np.sqrt(theta)
    got = float(schedule.next_increment(_evaluation_with_misfits(clustered)))
    assert got == pytest.approx(np.sqrt(theta / variance), rel=1e-12)
    assert np.sqrt(theta / variance) > theta / mean


def test_4_the_misfit_schedule_guards_both_of_its_divisions():
    """``inf`` at a vanishing denominator, ``nan`` at a ``nan`` one."""
    schedule = AdaptiveMisfitSchedule(beta_target=None, max_increment=0.7)
    assert float(
        schedule.next_increment(_evaluation_with_misfits(np.full(4, 3.0)))
    ) == pytest.approx(0.7)
    assert float(
        schedule.next_increment(_evaluation_with_misfits(np.zeros(4)))
    ) == pytest.approx(0.7)
    poisoned = _evaluation_with_misfits(np.array([1.0, np.nan, 2.0, 3.0]))
    assert np.isnan(float(schedule.next_increment(poisoned)))


def test_4_the_entry_time_budget_check_raises_before_any_evaluation():
    """The bound is checked against the schedule's own floor-bound worst case."""
    problem = _AffineProblem()
    scaled_down = AdaptiveESSSchedule(beta_target=0.01, min_increment=1e-3)
    with pytest.raises(ValueError, match="cannot accommodate"):
        _run(problem, schedule=scaled_down, max_steps=9)
    assert problem.calls == []
    _run(problem, schedule=scaled_down, max_steps=10)

    # The shipped defaults satisfy the relation with no slack: 1 / 1e-3 == 1000.
    _run(problem, schedule=AdaptiveESSSchedule())
    with pytest.raises(ValueError, match="cannot accommodate"):
        _run(problem, schedule=AdaptiveESSSchedule(beta_target=2.0))


def test_4_a_round_off_remainder_never_costs_a_step():
    """Ten steps of 0.1 leave about 1e-16 of the budget; no eleventh step takes it.

    The clamp gives a remainder below ``1e-9 * beta_target`` to the step
    before it, and the exhaustion check's own tolerance is ``1e-12``. Both
    are needed: the remainder of many floor-sized steps grows with their
    number, past the exhaustion tolerance at a hundred thousand of them.
    """
    problem = _AffineProblem()
    accumulated = 0.0
    for _ in range(10):
        accumulated += 0.1
    assert accumulated < 1.0, "the fixture depends on this being inexact"
    schedule = AdaptiveMisfitSchedule(
        beta_target=1.0, min_increment=0.1, max_increment=0.1
    )
    result = _run(problem, schedule=schedule, max_steps=10)
    assert result.n_evaluations == 10
    assert float(result.beta) == pytest.approx(1.0, abs=4 * EPS)

    # The entry check counts with the same tolerances, so a budget a hair
    # above an integer number of floors is accepted and completes.
    hair = AdaptiveMisfitSchedule(
        beta_target=1.0000000001, min_increment=0.25, max_increment=0.25
    )
    done = _run(problem, schedule=hair, max_steps=4)
    assert done.budget_complete and done.n_evaluations == 4

    # And the arithmetic, where running it would take too long: the same
    # loop in Python floats, against the count the entry check uses.
    from enskit.algorithms.eki._driver import _steps_needed

    for target, floor in ((1.0, 1e-5), (0.7, 1e-6), (1.0000000001, 1e-3)):
        beta, steps = 0.0, 0
        while not beta >= target - 1e-12 * target:
            delta = min(floor, target - beta)
            if target - beta - delta <= 1e-9 * target:
                delta = target - beta
            beta += delta
            steps += 1
        assert steps == _steps_needed(target, floor, target)


def test_4_the_entry_budget_check_measures_the_remaining_budget():
    """A resumed run gets the bound the caller asked for."""
    problem = _AffineProblem()
    schedule = AdaptiveESSSchedule(beta_target=1.0, min_increment=1e-3)
    part_way = EKIState(
        Ensemble(u=jnp.asarray(problem.members)), key=jax.random.key(0), beta=0.99
    )
    resumed = _run(problem, state=part_way, schedule=schedule, max_steps=10)
    assert resumed.budget_complete
    with pytest.raises(ValueError, match="cannot accommodate"):
        _run(problem, state=part_way, schedule=schedule, max_steps=9)
    with pytest.raises(ValueError, match="cannot accommodate"):
        _run(problem, schedule=schedule, max_steps=999)

    # The quotient is robust to its own round-off: 1e-9 / 1e-12 is 1000 steps.
    from enskit.algorithms.eki._driver import _steps_needed

    assert _steps_needed(1e-9, 1e-12, 1e-9) == 1000
    assert _steps_needed(1.0, 1e-3, 1.0) == 1000
    assert _steps_needed(0.55, 0.1, 1.0) == 6
    assert _steps_needed(-0.5, 1e-3, 1.0) == 0


def _reference_ess(misfit_values: np.ndarray, increment: float) -> float:
    """ESS of ``exp(-increment * misfits)``, in plain NumPy from the definition."""
    shifted = -increment * misfit_values
    weights = np.exp(shifted - shifted.max())
    return float(weights.sum() ** 2 / (weights**2).sum())


def _evaluation_with_misfits(
    misfit_values: np.ndarray, *, beta: float = 0.0, step: int = 0, data_dim: int = 4
) -> Evaluation:
    """An ``Evaluation`` whose misfits are exactly the values given.

    Row ``j`` of the whitened residuals is placed on the first coordinate at
    ``sqrt(2 * phi_j)``, so ``0.5 * ||b_j||**2`` is exactly ``phi_j``.
    """
    J = misfit_values.size
    residuals = np.zeros((J, data_dim))
    residuals[:, 0] = np.sqrt(2.0 * misfit_values)
    members = _exact_moment_ensemble(J, np.zeros(2), np.eye(2))
    return Evaluation(
        step=step,
        beta=beta,
        ensemble=Ensemble(
            {"u": jnp.asarray(members), PREDICTION: jnp.zeros((J, data_dim))}
        ),
        whitened_residuals=jnp.asarray(residuals),
        n_valid=J,
    )


def test_5_the_effective_sample_size_matches_its_definition():
    """``J`` at zero, monotone, and equal to ``J / (1 + cv^2)`` of the weights."""
    misfit_values = np.array([0.25, 1.0, 3.5, 7.0, 0.75, 2.0, 12.0])
    J = misfit_values.size
    at_zero = float(effective_sample_size(jnp.asarray(misfit_values), 0.0))
    assert at_zero == pytest.approx(J, abs=8 * EPS * J)

    grid = np.linspace(0.0, 3.0, 40)
    values = np.array(
        [float(effective_sample_size(jnp.asarray(misfit_values), d)) for d in grid]
    )
    assert np.all(np.diff(values) <= 8 * EPS * J)
    assert values[-1] < values[0]

    for increment in (0.1, 0.7, 2.5):
        weights = np.exp(-increment * misfit_values)
        got = float(effective_sample_size(jnp.asarray(misfit_values), increment))
        assert got == pytest.approx(_reference_ess(misfit_values, increment), rel=1e-12)
        cv_squared = weights.var(ddof=0) / weights.mean() ** 2
        assert got == pytest.approx(J / (1.0 + cv_squared), rel=1e-12)
        assert 1.0 - 1e-12 <= got <= J + 1e-9


def test_5_the_effective_sample_size_survives_misfits_the_naive_form_cannot():
    """Misfits of order 1e4 give a finite value where naive weights give ``nan``."""
    misfit_values = np.array([1.0e4, 1.2e4, 1.5e4, 1.1e4])
    got = float(effective_sample_size(jnp.asarray(misfit_values), 1.0))
    assert np.isfinite(got)
    assert 1.0 <= got <= misfit_values.size
    with np.errstate(invalid="ignore", under="ignore"):
        naive_weights = np.exp(-1.0 * misfit_values)
        naive = naive_weights.sum() ** 2 / (naive_weights**2).sum()
    assert np.isnan(naive)


def test_6_the_two_adaptive_criteria_are_distinct_and_measurably_so():
    """The misfit criterion drives the ESS to its floor and takes far longer steps."""
    rng = np.random.default_rng(11)
    misfit_values = 50.0 + 8.0 * rng.normal(size=40)
    evaluation = _evaluation_with_misfits(misfit_values, data_dim=80)
    unbounded = dict(beta_target=None, min_increment=1e-9, max_increment=1e6)
    misfit_step = float(AdaptiveMisfitSchedule(**unbounded).next_increment(evaluation))
    ess_step = float(AdaptiveESSSchedule(**unbounded).next_increment(evaluation))
    at_misfit_step = float(effective_sample_size(jnp.asarray(misfit_values), misfit_step))
    assert at_misfit_step < 4.0
    assert misfit_step >= 5.0 * ess_step


def test_7_the_discrepancy_stop_fires_on_its_threshold():
    """Fires exactly when ``2 Phi(gbar) <= tau^2 N``."""
    data_dim = 4
    for tau in (0.5, 1.0, 2.0):
        rule = DiscrepancyStop(tau=tau)
        for center_misfit in (0.1, 0.9, 2.0, 4.5, 9.0):
            evaluation = _evaluation_with_misfits(
                np.full(4, center_misfit), data_dim=data_dim
            )
            assert float(evaluation.center_misfit) == pytest.approx(center_misfit)
            assert rule(evaluation) is (2.0 * center_misfit <= tau**2 * data_dim)


def test_7_a_fired_stop_ends_the_run_with_a_zero_increment_terminal_record():
    problem = _AffineProblem()
    result = _run(
        problem, schedule=FixedSchedule.constant(0.5, 50), stop=DiscrepancyStop(tau=1e6)
    )
    assert result.status == STOPPING_RULE
    assert result.n_evaluations == 1
    assert float(result.stacked.increment[0]) == 0.0
    assert len(problem.calls) == 1
    assert float(result.beta) == 0.0
    assert np.array_equal(_u(result), problem.members)
    assert np.array_equal(_u(result.last_evaluation), _u(result))


def test_8_max_steps_raises_with_a_payload_that_makes_the_run_resumable():
    problem = _AffineProblem()
    schedule = FixedSchedule.constant(0.25, 40)
    with pytest.raises(EKIError, match="max_steps") as caught:
        _run(problem, schedule=schedule, max_steps=6)
    failure = caught.value
    assert "FixedSchedule" in str(failure)
    assert "no stopping rule" in str(failure)
    assert isinstance(failure.state, EKIState)
    assert len(failure.history) == 6
    assert failure.state.step == 6

    resumed = _run(problem, state=failure.state, schedule=schedule, max_steps=40)
    uninterrupted = _run(problem, schedule=schedule, max_steps=40)
    assert np.array_equal(_u(resumed), _u(uninterrupted))
    assert resumed.n_evaluations == 34


def test_8_a_budgeted_schedule_with_a_positive_floor_never_reaches_the_bound():
    problem = _AffineProblem()
    result = _run(problem, schedule=AdaptiveESSSchedule(), max_steps=1000)
    assert result.status == SCHEDULE_EXHAUSTED


def test_8_every_eki_error_path_carries_the_history_accumulated_so_far():
    """All three raise paths of a step, not just the bound."""
    problem = _AffineProblem(J=8)
    ladder = FixedSchedule.constant(0.2, 30)

    def fails_at(step, indices):
        state = {"n": 0}

        def forward(u):
            v = jnp.asarray(u) @ jnp.asarray(problem.G).T
            hit = state["n"] == step
            state["n"] += 1
            return v.at[jnp.asarray(indices), 0].set(jnp.nan) if hit else v

        return forward

    with pytest.raises(EKIError, match="At least 2 are required") as caught:
        _run(problem, forward=fails_at(3, list(range(1, problem.J))), schedule=ladder,
             on_failure="repair")
    assert len(caught.value.history) == 3
    assert caught.value.state.step == 3

    with pytest.raises(EKIError, match="on_failure='raise'") as caught:
        _run(problem, forward=fails_at(1, [2]), schedule=ladder)
    assert len(caught.value.history) == 1
    assert caught.value.state.step == 1

    with pytest.raises(EKIError, match="non-finite particle") as caught:
        _run(problem, schedule=ladder, update_rule=_PoisonAt(3))
    assert len(caught.value.history) == 2
    assert caught.value.state.step == 2
    assert isinstance(caught.value.history, tuple)
    resumed = _run(problem, state=caught.value.state, schedule=ladder, max_steps=30)
    assert resumed.n_evaluations == 28


class _PoisonAt:
    """An update rule whose ``n``-th update returns non-finite particles."""

    def __init__(self, n):
        self.n, self.calls = n, 0

    def build(self, particles, approximation, given):
        self.calls += 1
        if self.calls != self.n:
            return SQRT.build(particles, approximation, given)
        targets = [name for name in particles.names if name not in given]

        def call(values=None, /, *, key=None, **kwargs):
            return Ensemble({n: jnp.full_like(particles[n], jnp.nan) for n in targets})

        return call


def test_9_the_repair_moves_only_the_failed_particles_and_damps_the_moments():
    """The three exact identities, pinned as equalities rather than tolerances."""
    rng = np.random.default_rng(19)
    J, P, N = 9, 3, 4
    u = jnp.asarray(rng.normal(size=(J, P)))
    g = jnp.asarray(rng.normal(size=(J, N)))
    valid = jnp.asarray([True, True, False, True, False, True, True, True, False])
    n_valid = int(np.asarray(valid).sum())

    repaired = repair_failed_particles(ensemble=Ensemble(u=u, g=g), valid=valid)
    ru, rg = np.asarray(repaired["u"]), np.asarray(repaired["g"])
    mask = np.asarray(valid)
    u_hat = np.asarray(u)[mask].mean(axis=0)
    g_hat = np.asarray(g)[mask].mean(axis=0)

    assert np.array_equal(ru[mask], np.asarray(u)[mask])
    assert np.array_equal(rg[mask], np.asarray(g)[mask])
    assert np.abs(ru[~mask] - u_hat).max() < 8 * EPS
    assert np.abs(rg[~mask] - g_hat).max() < 8 * EPS

    got_mean = ru.mean(axis=0)
    assert np.abs(got_mean - u_hat).max() < 16 * EPS * max(1.0, np.abs(u_hat).max())

    damping = (n_valid - 1) / (J - 1)
    valid_u = np.asarray(u)[mask] - u_hat
    valid_g = np.asarray(g)[mask] - g_hat
    for got, want in (
        (_moments(ru)[1], damping * (valid_u.T @ valid_u) / (n_valid - 1)),
        (
            (ru - got_mean).T @ (rg - g_hat) / (J - 1),
            damping * (valid_u.T @ valid_g) / (n_valid - 1),
        ),
    ):
        scale = max(1.0, np.abs(want).max())
        assert np.abs(got - want).max() < 64 * EPS * scale


def test_9_a_no_failure_step_skips_the_repair_entirely():
    """When nothing fails the particles pass through untouched, bit for bit."""
    problem = _AffineProblem()
    state = problem.state()
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    evaluation = evaluate(state, problem.forward, y, noise)
    assert evaluation.ensemble["u"] is state.ensemble["u"]
    assert int(evaluation.n_valid) == problem.J

    predictions = jnp.asarray(problem.members @ problem.G.T)
    whole = Ensemble(u=state.ensemble["u"], g=predictions)
    all_valid = jnp.ones(problem.J, dtype=bool)
    repaired = repair_failed_particles(ensemble=whole, valid=all_valid)
    assert np.array_equal(np.asarray(repaired["u"]), np.asarray(whole["u"]))
    assert np.array_equal(np.asarray(repaired["g"]), np.asarray(whole["g"]))


def test_9_the_failure_modes_raise_where_the_contract_says_they_do():
    problem = _AffineProblem()
    state = problem.state()
    y, noise = jnp.asarray(problem.y), problem.noise_cov

    def fail(indices):
        def forward(u):
            v = jnp.asarray(u) @ jnp.asarray(problem.G).T
            return v.at[jnp.asarray(indices), 0].set(jnp.nan)

        return forward

    # "raise" is the default.
    with pytest.raises(EKIError, match="on_failure='raise'"):
        evaluate(state, fail([2]), y, noise)
    with pytest.raises(EKIError, match=r"\[2\]"):
        evaluate(state, fail([2]), y, noise, on_failure="raise")

    all_but_one = list(range(1, problem.J))
    for mode in ("repair", "raise"):
        with pytest.raises(EKIError, match="At least 2 are required"):
            evaluate(state, fail(all_but_one), y, noise, on_failure=mode)

    with pytest.raises(ValueError, match="on_failure"):
        evaluate(state, problem.forward, y, noise, on_failure="Raise")
    with pytest.raises(ValueError, match="on_failure"):
        _run(problem, schedule=FixedSchedule.uniform(2), on_failure="skip")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _run(
            problem, forward=fail([1, 4]), schedule=FixedSchedule.uniform(3),
            on_failure="repair",
        )
    stacked = result.stacked
    for name in ("misfit_mean", "misfit_min", "misfit_max", "center_misfit", "ess"):
        assert np.all(np.isfinite(np.asarray(getattr(stacked, name)))), name
    assert int(stacked.n_valid[0]) == problem.J - 2
    assert result.min_n_valid == problem.J - 2


def test_10_the_inflation_policies_are_the_kalman_functions():
    """Each shipped inflation equals the function it wraps, bit for bit."""
    rng = np.random.default_rng(23)
    ens = Ensemble(
        u=jnp.asarray(rng.normal(size=(10, 4))), v=jnp.asarray(rng.normal(size=(10, 2)))
    )
    key = jax.random.key(4)
    for factor in (1.02, 1.2, 2.0):
        got = MultiplicativeInflation(factor)(key, ensemble=ens, step=0, beta=0.0)
        want = kalman.inflate_multiplicative(ens, factor)
        for n in ens.names:
            assert np.array_equal(np.asarray(got[n]), np.asarray(want[n]))
    got = MultiplicativeInflation(1.5, names="v")(key, ensemble=ens)
    assert np.array_equal(np.asarray(got["u"]), np.asarray(ens["u"]))

    cov = DensePSD(jnp.asarray(_psd(4, seed=31) * 0.01))
    got = AdditiveInflation(u=cov)(key, ensemble=ens, step=0, beta=0.0)
    want = kalman.inflate_additive(key, ens, u=cov)
    for n in ens.names:
        assert np.array_equal(np.asarray(got[n]), np.asarray(want[n]))


def test_10_multiplicative_inflation_stays_in_the_span_and_additive_leaves_it():
    """The executable form of the subspace property, in both directions."""
    problem = _AffineProblem(P=5, N=6, J=4, seed=17, prior_rank=3)
    basis = _span_basis(problem.members)
    assert basis.shape[1] < problem.P

    plain = _run(problem, schedule=FixedSchedule.uniform(4))
    assert _leaves_span(_u(plain), problem.members, basis) < 1e-9

    multiplicative = _run(
        problem, schedule=FixedSchedule.uniform(4),
        inflation=MultiplicativeInflation(1.05),
    )
    assert _leaves_span(_u(multiplicative), problem.members, basis) < 1e-9

    additive = _run(
        problem, schedule=FixedSchedule.uniform(4),
        inflation=AdditiveInflation(u=DensePSD(jnp.eye(problem.P) * 0.05)),
    )
    assert _leaves_span(_u(additive), problem.members, basis) > 1e-3


def _span_basis(members) -> np.ndarray:
    anomalies = np.asarray(members) - np.asarray(members).mean(axis=0)
    basis, *_ = np.linalg.svd(anomalies.T, full_matrices=False)
    return basis[:, : int(np.linalg.matrix_rank(anomalies))]


def _leaves_span(members, initial, basis: np.ndarray) -> float:
    moved = np.asarray(members) - np.asarray(initial).mean(axis=0)
    return float(np.abs(moved - moved @ basis @ basis.T).max())


@pytest.mark.parametrize("batch", [(), (3,), (2, 3)])
def test_11_misfits_matches_a_dense_quadratic_form_at_every_batch_rank(batch):
    rng = np.random.default_rng(41)
    N = 5
    R = _psd(N, seed=43)
    y = rng.normal(size=N)
    predictions = rng.normal(size=(*batch, N))
    got = misfits(jnp.asarray(y), jnp.asarray(predictions), DensePSD(jnp.asarray(R)))
    residual = y - predictions
    want = 0.5 * np.einsum("...i,ij,...j->...", residual, np.linalg.inv(R), residual)
    assert np.asarray(got).shape == batch
    assert np.abs(np.asarray(got) - want).max() < 1e-9 * max(1.0, np.abs(want).max())


def test_11_misfits_is_whitener_invariant_and_carries_the_half():
    diagonal = np.array([2.0, 0.5, 4.0])
    y = jnp.asarray([1.0, 2.0, 3.0])
    predictions = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    as_diagonal = PSDDiagonal(jnp.asarray(diagonal))
    as_dense = DensePSD(jnp.diag(jnp.asarray(diagonal)))
    got = misfits(y, predictions, as_diagonal)
    dense = np.asarray(misfits(y, predictions, as_dense))
    assert np.abs(np.asarray(got) - dense).max() < 1e-12
    assert float(got[0]) == pytest.approx(0.5 * (0.5 + 8.0 + 2.25))
    assert float(got[1]) == 0.0


def _noise_operators() -> dict[str, object]:
    diagonal = PSDDiagonal(jnp.asarray([0.5, 2.0, 1.5, 3.0, 0.25, 1.0]))
    return {
        "identity": Identity(6),
        "diagonal": diagonal,
        "dense": DensePSD(jnp.asarray(_psd(6, seed=47))),
        "block_diag": block_diag(
            Identity(2),
            PSDDiagonal(jnp.asarray([2.0, 0.5])),
            DensePSD(jnp.asarray(_psd(2, seed=53))),
        ),
    }


@pytest.mark.parametrize("name", ["identity", "diagonal", "dense", "block_diag"])
@pytest.mark.parametrize("batch", [(), (4,), (3, 4)])
def test_11_misfits_is_the_gaussian_log_likelihood_less_its_normalizer(name, batch):
    """``log N(y | g, R) + Phi(g) == -1/2 (log det R + N log 2 pi)``, every row.

    The normalized log-likelihood is the log density of the Gaussian with mean
    ``y`` and covariance ``R``, at ``g``.
    """
    noise_cov = _noise_operators()[name]
    rng = np.random.default_rng(59)
    y = jnp.asarray(rng.normal(size=6))
    predictions = jnp.asarray(rng.normal(size=(*batch, 6)))
    noise = Gaussian({"g": y}, block_covs={"g": noise_cov})
    log_likelihood = noise.log_density(g=predictions)
    total = log_likelihood + misfits(y, predictions, noise_cov)
    R = np.asarray(noise_cov.to_dense())
    want = -0.5 * (np.linalg.slogdet(R)[1] + 6 * np.log(2.0 * np.pi))
    assert total.shape == batch
    assert np.abs(np.asarray(total) - want).max() < 1e-12 * abs(want)


def test_11_the_center_misfit_differs_from_the_mean_by_exactly_the_spread_term():
    """``mean(Phi_j) - Phi(gbar) == (J-1)/(2J) tr(W Chat_gg W^T)``, exactly."""
    problem = _AffineProblem(J=10)
    evaluation = evaluate(
        problem.state(), problem.forward, jnp.asarray(problem.y), problem.noise_cov
    )
    J = problem.J
    whitened = np.asarray(evaluation.whitened_residuals)
    anomalies = whitened - whitened.mean(axis=0)
    trace = float(np.sum(anomalies**2) / (J - 1))
    gap = float(np.mean(np.asarray(evaluation.misfits))) - float(evaluation.center_misfit)
    assert gap == pytest.approx((J - 1) / (2 * J) * trace, rel=1e-11)

    assert np.abs(
        np.asarray(evaluation.misfits) - 0.5 * np.sum(whitened**2, axis=1)
    ).max() < 1e-12
    assert float(evaluation.center_misfit) == pytest.approx(
        0.5 * float(np.sum(whitened.mean(axis=0) ** 2)), rel=1e-12
    )

    W = _recovered_whitener(problem.noise_cov, problem.N)
    predictions = np.asarray(evaluation.ensemble[PREDICTION])
    want_anomalies = -(predictions - predictions.mean(axis=0)) @ W.T
    assert np.abs(anomalies - want_anomalies).max() < 1e-8

    assert np.abs(
        np.asarray(evaluation.misfits)
        - np.asarray(misfits(jnp.asarray(problem.y), predictions, problem.noise_cov))
    ).max() < 1e-10


def test_12_runs_are_reproducible_resumable_and_agree_with_iterate():
    problem = _AffineProblem()
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    common = dict(schedule=FixedSchedule.uniform(8), update_rule=MATHERON)

    first = _run(problem, **common)
    second = _run(problem, **common)
    assert np.array_equal(_u(first), _u(second))

    partial_state = problem.state()
    taken = 0
    for yielded in iterate(problem.state(), problem.forward, y, noise, **common):
        partial_state = yielded[0]
        taken += 1
        if taken == 4:
            break
    assert taken == 4 and partial_state.step == 4
    resumed = _run(problem, state=partial_state, **common)
    assert np.array_equal(_u(resumed), _u(first))
    for got, want in zip(resumed.history, first.history[4:], strict=True):
        for name in ("beta", "increment", "misfit_mean", "ess"):
            assert float(getattr(got, name)) == float(getattr(want, name))

    generator = iterate(problem.state(), problem.forward, y, noise, **common)
    records, last_state, last_evaluation = [], None, None
    while True:
        try:
            last_state, record, last_evaluation = next(generator)
        except StopIteration as finished:
            status = finished.value
            break
        records.append(record)
    assert status == first.status
    assert len(records) == first.n_evaluations
    assert np.array_equal(_u(last_state), _u(first))
    assert np.array_equal(
        np.asarray(last_evaluation.ensemble[PREDICTION]),
        np.asarray(first.last_evaluation.ensemble[PREDICTION]),
    )


def test_13_the_optimization_form_approaches_the_restricted_least_squares_fit():
    """A monotone misfit, and a center approaching the subspace-restricted fit."""
    problem = _AffineProblem(P=5, N=6, J=4, seed=13, prior_rank=3, noise_scale=1e-2)
    basis = _span_basis(problem.members)
    origin = np.asarray(problem.members).mean(axis=0)
    truth = origin + basis @ np.array([1.5, -2.0, 1.0])
    perturbation = np.linalg.cholesky(problem.R) @ np.random.default_rng(3).normal(
        size=problem.N
    )
    problem.y = problem.G @ truth + perturbation
    design = problem.G @ basis
    precision = np.linalg.inv(problem.R)
    coefficients = np.linalg.solve(
        design.T @ precision @ design,
        design.T @ precision @ (problem.y - problem.G @ origin),
    )
    restricted = origin + basis @ coefficients
    unrestricted = np.linalg.lstsq(problem.G, problem.y, rcond=None)[0]
    assert np.abs(restricted - unrestricted).max() > 1e-2

    result = _run(
        problem, schedule=FixedSchedule.constant(1.0, 200),
        stop=DiscrepancyStop(tau=1.0), max_steps=200,
    )
    assert result.stop_fired
    center_misfits = np.asarray(result.stacked.center_misfit)
    assert np.all(np.diff(center_misfits) <= 1e-9 * max(1.0, center_misfits[0]))
    got = _u(result).mean(axis=0)
    assert np.abs(got - restricted).max() < 1e-2 * max(1.0, np.abs(restricted).max())
    assert np.abs(got - unrestricted).max() > 1e-2


class _CompileCounter:
    """Counts backend compilations, through JAX's own monitoring events."""

    def __init__(self):
        self.count = 0

    def __call__(self, event, duration, **kwargs):
        if event == "/jax/core/compile/backend_compile_duration":
            self.count += 1


def test_14_a_run_compiles_a_bounded_number_of_times_whatever_its_length():
    """A thirty-step run adds no compilation a three-step run has not paid for.

    Counted by JAX's own compilation events, so the whole run is covered:
    the driver, the update and every eager operation. A static field on an
    object crossing a ``jit`` boundary, or a Python float baked into a traced
    function, would compile once per step.
    """
    problem = _AffineProblem()
    configuration = dict(
        update_rule=MATHERON,
        inflation=MultiplicativeInflation(1.01),
        relaxation=RelaxToPriorSpread(0.2),
    )
    counter = _CompileCounter()
    jax.monitoring.register_event_duration_secs_listener(counter)
    try:
        # Distinct increments, so a compilation keyed on the increment's value
        # would show.
        _run(problem, schedule=FixedSchedule((0.05, 0.06, 0.07)), **configuration)
        after_short = counter.count
        long_ladder = tuple(0.01 * (1.0 + i / 30) for i in range(30))
        _run(problem, schedule=FixedSchedule(long_ladder), **configuration)
        after_long = counter.count
        _run(problem, schedule=AdaptiveESSSchedule(beta_target=0.5), **configuration)
        after_adaptive = counter.count
        _run(problem, schedule=AdaptiveESSSchedule(beta_target=0.9), **configuration)
        after_second_adaptive = counter.count
    finally:
        jax.monitoring.unregister_event_duration_listener(counter)
    assert after_long == after_short, (
        f"compilations grew with the number of steps: {after_short} -> {after_long}"
    )
    assert after_second_adaptive == after_adaptive


def test_14_the_pytree_classes_round_trip_and_families_are_inert():
    problem = _AffineProblem()
    state = problem.state()
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    evaluation = evaluate(state, problem.forward, y, noise)
    record = HistoryRecord.from_evaluation(evaluation, 0.25)

    for obj in (state, evaluation, record):
        leaves, treedef = jax.tree.flatten(obj)
        rebuilt = jax.tree.unflatten(treedef, leaves)
        assert type(rebuilt) is type(obj)
        assert rebuilt.batch_shape == ()
        sentinel = jax.tree.unflatten(treedef, [object()] * len(leaves))
        assert type(sentinel) is type(obj)
        text = repr(sentinel)
        assert "unprintable" in text or text.startswith(type(obj).__name__)

    family = jax.tree.map(lambda x: jnp.stack([x, x]), evaluation)
    assert family.batch_shape == (2,)
    assert repr(family).startswith("vmapped(")
    with pytest.raises(ValueError, match="vmapped family"):
        _ = family.misfits
    assert family.n_particles == evaluation.n_particles

    states = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    assert states.batch_shape == (2,)
    with pytest.raises(ValueError, match="vmapped family"):
        _run(problem, state=states, schedule=FixedSchedule.uniform(1))


def test_14_n_valid_is_data_so_a_jitted_policy_does_not_retrace_per_step():
    """A static ``n_valid`` would give each step its own treedef."""
    counts = {"traces": 0}

    @jax.jit
    def criterion(evaluation):
        counts["traces"] += 1
        return jnp.mean(evaluation.misfits) + evaluation.n_valid

    base = _evaluation_with_misfits(np.arange(6.0))
    for n_valid in (6, 5, 4, 3, 2):
        criterion(
            Evaluation(
                step=0, beta=0.0, ensemble=base.ensemble,
                whitened_residuals=base.whitened_residuals, n_valid=n_valid,
            )
        )
    assert counts["traces"] == 1
    with debug_checks(), pytest.raises(ValueError, match="n_valid"):
        Evaluation(
            step=0, beta=0.0, ensemble=base.ensemble,
            whitened_residuals=base.whitened_residuals, n_valid=1,
        )


def test_15_the_history_stacks_including_its_two_integer_fields():
    """A targeted regression test for the treedef trap."""
    problem = _AffineProblem()
    result = _run(problem, schedule=FixedSchedule.uniform(5))
    stacked = result.stacked
    assert isinstance(stacked, HistoryRecord)
    assert stacked.batch_shape == (5,)
    for name in ("step", "n_valid", "beta", "increment", "ess"):
        assert np.asarray(getattr(stacked, name)).shape == (5,)
    assert list(np.asarray(stacked.step)) == [0, 1, 2, 3, 4]

    empty = EKIResult(
        state=problem.state(), history=(), status=SCHEDULE_EXHAUSTED, last_evaluation=None
    ).stacked
    assert empty.batch_shape == (0,)
    for name in ("step", "n_valid", "beta", "increment", "ess"):
        assert np.asarray(getattr(empty, name)).shape == (0,)


def test_15_records_keep_the_default_float_and_refuse_a_zero_increment():
    """A float32 level would round a tiny increment to a zero, terminal-looking one."""
    problem = _AffineProblem()
    state = EKIState(Ensemble(u=jnp.asarray(problem.members)), key=jax.random.key(0),
                     beta=jnp.float32(0.0))
    assert state.beta.dtype == jnp.result_type(float)
    result = _run(problem, state=state, schedule=FixedSchedule((1e-50, 1.0)))
    assert float(result.history[0].increment) == 1e-50
    assert result.n_completed_steps == 2

    evaluation = evaluate(problem.state(), problem.forward, jnp.asarray(problem.y),
                          problem.noise_cov)
    for bad in (0.0, -1.0, np.inf):
        with pytest.raises(ValueError, match="increment=None"):
            HistoryRecord.from_evaluation(evaluation, bad)
    with pytest.raises(TypeError, match="real"):
        EKIState(Ensemble(u=jnp.asarray(problem.members)), key=jax.random.key(0),
                 beta=1j)


def test_16_the_two_phases_compose_and_one_evaluation_serves_two_increments():
    problem = _AffineProblem()
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    state = problem.state()

    composed = advance(state, problem.forward, y, noise, 0.3, update_rule=SQRT)
    assert len(problem.calls) == 1
    evaluation = evaluate(state, problem.forward, y, noise)
    applied = assimilate(state, evaluation, 0.3, y, noise, update_rule=SQRT)
    assert np.array_equal(_u(composed), _u(applied))

    before = len(problem.calls)
    small = assimilate(state, evaluation, 0.1, y, noise, update_rule=SQRT)
    large = assimilate(state, evaluation, 0.9, y, noise, update_rule=SQRT)
    assert len(problem.calls) == before
    assert float(small.beta) == pytest.approx(0.1)
    assert float(large.beta) == pytest.approx(0.9)
    assert not np.array_equal(_u(small), _u(large))
    assert np.array_equal(_u(state), problem.members)
    assert evaluation.step == 0 and float(evaluation.beta) == 0.0

    moved = advance(state, problem.forward, y, noise, 0.2, update_rule=SQRT)
    with pytest.raises(ValueError, match="another state"):
        assimilate(moved, evaluation, 0.1, y, noise, update_rule=SQRT)
    before = len(problem.calls)
    for bad in (0.0, -0.5, np.inf, np.nan):
        with pytest.raises(ValueError, match="strictly positive"):
            assimilate(state, evaluation, bad, y, noise, update_rule=SQRT)
    assert len(problem.calls) == before


def test_16_iterate_yields_what_a_hand_written_two_phase_loop_yields():
    """The ensembles and the records, from ``HistoryRecord.from_evaluation``."""
    problem = _AffineProblem()
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    increments = (0.2, 0.3, 0.5)

    by_hand, hand_records = problem.state(), []
    for increment in increments:
        evaluation = evaluate(by_hand, problem.forward, y, noise)
        hand_records.append(HistoryRecord.from_evaluation(evaluation, increment))
        by_hand = assimilate(by_hand, evaluation, increment, y, noise, update_rule=SQRT)

    yielded = list(
        iterate(problem.state(), problem.forward, y, noise, update_rule=SQRT,
                schedule=FixedSchedule(increments))
    )
    assert np.array_equal(_u(yielded[-1][0]), _u(by_hand))
    for (_, record, _), want in zip(yielded, hand_records, strict=True):
        for name in ("step", "beta", "increment", "misfit_mean", "ess", "spread"):
            assert float(getattr(record, name)) == float(getattr(want, name)), name


def test_16_the_provenance_check_catches_a_stale_evaluation_not_a_foreign_run():
    """What the check establishes, and what it deliberately does not."""
    first = _AffineProblem(seed=7)
    second = _AffineProblem(seed=77)
    y, noise = jnp.asarray(first.y), first.noise_cov
    state = first.state()
    evaluation = evaluate(state, first.forward, y, noise)

    moved = assimilate(state, evaluation, 0.25, y, noise, update_rule=SQRT)
    with pytest.raises(ValueError, match="another state"):
        assimilate(moved, evaluation, 0.25, y, noise, update_rule=SQRT)

    foreign = evaluate(
        second.state(), second.forward, jnp.asarray(second.y), second.noise_cov
    )
    assert foreign.step == state.step and float(foreign.beta) == float(state.beta)
    mixed = assimilate(state, foreign, 0.25, y, noise, update_rule=SQRT)
    expected = assimilate(second.state(), foreign, 0.25, y, noise, update_rule=SQRT)
    assert np.array_equal(_u(mixed), _u(expected))
    assert not np.array_equal(_u(mixed), _u(moved))


@pytest.mark.parametrize("P,N,J", [(1, 1, 2), (3, 1, 2), (1, 4, 5), (2, 2, 2)])
def test_17_the_degenerate_shapes_all_work(P, N, J):
    problem = _AffineProblem(P=P, N=N, J=max(J, P + 1), seed=53)
    for rule in (SQRT, MATHERON):
        result = _run(problem, schedule=FixedSchedule.uniform(3), update_rule=rule)
        assert np.all(np.isfinite(_u(result)))


def test_17_a_collapsed_ensemble_neither_moves_nor_produces_nan():
    problem = _AffineProblem()
    collapsed = jnp.tile(jnp.asarray(problem.members[:1]), (problem.J, 1))
    state = EKIState(Ensemble(u=collapsed), key=jax.random.key(0))
    for rule in (SQRT, MATHERON):
        result = _run(
            problem, state=state, schedule=FixedSchedule.uniform(3), update_rule=rule
        )
        assert np.all(np.isfinite(_u(result)))
        assert np.array_equal(_u(result), np.asarray(collapsed))


def test_16_advance_checks_its_arguments_before_calling_the_model():
    """A bad increment or rule costs no evaluation, through ``advance`` too."""
    problem = _AffineProblem()
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    for increment, rule, error in (
        (0.0, SQRT, ValueError),
        (np.nan, SQRT, ValueError),
        (0.5, object(), TypeError),
        (1j, SQRT, TypeError),
    ):
        with pytest.raises(error):
            advance(problem.state(), problem.forward, y, noise, increment,
                    update_rule=rule)
    assert problem.calls == []


def test_17_a_non_finite_update_raises_eki_error_before_a_relaxation_sees_it():
    """In debug mode the relaxation would refuse it first, without the state."""
    problem = _AffineProblem()
    for relaxation in (RelaxToPriorSpread(0.5), RelaxToPriorPerturbations(0.5)):
        with debug_checks(), pytest.raises(EKIError, match="non-finite") as caught:
            _run(problem, schedule=FixedSchedule.uniform(3), update_rule=_PoisonAt(2),
                 relaxation=relaxation)
        assert caught.value.state.step == 1
        assert len(caught.value.history) == 1


def test_17_a_wholly_failing_model_and_a_nan_update_both_raise():
    problem = _AffineProblem()
    with pytest.raises(EKIError, match="At least 2 are required"):
        _run(problem, forward=lambda u: jnp.full((problem.J, problem.N), jnp.nan),
             schedule=FixedSchedule.uniform(3))
    with pytest.raises(EKIError, match="non-finite particle at step 0"):
        _run(problem, schedule=FixedSchedule.uniform(3), update_rule=_PoisonAt(1))


def test_18_every_tier_two_and_tier_three_rule_raises_as_specified():
    """The validation table, rule by rule."""
    key = jax.random.key(0)
    ens = Ensemble(u=jnp.zeros((4, 3)))

    # -- EKIState ------------------------------------------------------------
    with pytest.raises(TypeError, match="Ensemble"):
        EKIState(jnp.zeros((4, 3)), key=key)
    with pytest.raises(ValueError, match="reserved"):
        EKIState(Ensemble({PREDICTION: jnp.zeros((4, 3))}), key=key)
    with pytest.raises(ValueError, match="weighted"):
        EKIState(Ensemble(u=jnp.zeros((4, 3)), log_weights=jnp.zeros(4)), key=key)
    with pytest.raises(ValueError, match="must not be negative"):
        EKIState(ens, key=key, step=-1)
    with pytest.raises(TypeError, match="Python int"):
        EKIState(ens, key=key, step=1.0)
    with pytest.raises(ValueError, match="scalar"):
        EKIState(ens, key=key, beta=jnp.zeros((2,)))
    with pytest.raises(TypeError, match="typed key"):
        EKIState(ens, key=jnp.zeros(()))
    with pytest.raises(TypeError, match="typed key"):
        EKIState(ens, key=jax.random.PRNGKey(0))
    with pytest.raises(TypeError):
        EKIState(ens)
    with debug_checks():
        with pytest.raises(ValueError, match="not negative"):
            EKIState(ens, key=key, beta=-0.5)
        with pytest.raises(ValueError, match="finite"):
            EKIState(Ensemble(u=jnp.full((4, 3), jnp.nan)), key=key)

    # -- policies ------------------------------------------------------------
    with pytest.raises(ValueError, match="must not be empty"):
        FixedSchedule(())
    with pytest.raises(ValueError, match="strictly positive"):
        FixedSchedule((0.5, 0.0, 0.5))
    with pytest.raises(ValueError, match="strictly positive"):
        FixedSchedule((0.5, float("inf")))
    with pytest.raises(TypeError, match="tuple"):
        FixedSchedule([0.5, 0.5])
    with pytest.raises(ValueError, match="ess_fraction"):
        AdaptiveESSSchedule(ess_fraction=1.0)
    with pytest.raises(ValueError, match="ess_fraction"):
        AdaptiveESSSchedule(ess_fraction=0.0)
    with pytest.raises(ValueError, match="n_bisect"):
        AdaptiveESSSchedule(n_bisect=0)
    with pytest.raises(ValueError, match="max_increment"):
        AdaptiveESSSchedule(max_increment=float("inf"))
    with pytest.raises(ValueError, match="min_increment"):
        AdaptiveESSSchedule(min_increment=2.0, max_increment=1.0)
    with pytest.raises(ValueError, match="beta_target"):
        AdaptiveESSSchedule(beta_target=0.0)
    with pytest.raises(ValueError, match="divergence_budget"):
        AdaptiveMisfitSchedule(divergence_budget=0.0)
    with pytest.raises(ValueError, match="divergence_budget"):
        AdaptiveMisfitSchedule(divergence_budget=float("inf"))
    with pytest.raises(ValueError, match="tau"):
        DiscrepancyStop(tau=0.0)

    # -- call-time problem and policy outputs ---------------------------------
    problem = _AffineProblem()
    ladder = FixedSchedule.uniform(2)
    with pytest.raises(ValueError, match="shape"):
        run(problem.state(), problem.forward, jnp.zeros((problem.N + 1,)),
            problem.noise_cov, update_rule=SQRT, schedule=ladder)
    with pytest.raises(ValueError, match="must be finite"):
        run(problem.state(), problem.forward, jnp.full((problem.N,), jnp.nan),
            problem.noise_cov, update_rule=SQRT, schedule=ladder)
    with pytest.raises(TypeError, match="PSDLinOp"):
        run(problem.state(), problem.forward, jnp.asarray(problem.y), jnp.eye(problem.N),
            update_rule=SQRT, schedule=ladder)
    with pytest.raises(TypeError, match="build"):
        _run(problem, schedule=ladder, update_rule=lambda *a, **k: None)
    with pytest.raises(TypeError, match="update_rule"):
        run(problem.state(), problem.forward, jnp.asarray(problem.y), problem.noise_cov,
            schedule=ladder)
    with pytest.raises(TypeError, match="next_increment"):
        _run(problem, schedule=object())
    with pytest.raises(TypeError, match="callable"):
        _run(problem, schedule=ladder, inflation=1.05)
    with pytest.raises(ValueError, match="max_steps"):
        _run(problem, schedule=ladder, max_steps=0)
    with pytest.raises(KeyError, match="not a parameter block"):
        _run(problem, schedule=ladder, inputs="v")
    with pytest.raises(ValueError, match="values per particle"):
        _run(problem, schedule=ladder,
             forward=lambda u: jnp.zeros((problem.J, problem.N + 1)))
    with pytest.raises(ValueError, match="int32"):
        _run(problem, schedule=ladder,
             forward=lambda u: jnp.zeros((problem.J, problem.N), dtype=jnp.int32))
    with pytest.raises(ValueError, match="particles, expected"):
        _run(problem, schedule=ladder,
             inflation=lambda key, *, ensemble, **_: Ensemble(u=ensemble["u"][:-1]))
    with pytest.raises(TypeError, match="dtype"):
        _run(problem, schedule=ladder,
             inflation=lambda key, *, ensemble, **_: Ensemble(
                 u=ensemble["u"].astype(jnp.float32)))
    with pytest.raises(ValueError, match="blocks"):
        _run(problem, schedule=ladder,
             relaxation=lambda *, prior, posterior, **_: posterior.assign(
                 w=posterior["u"]))
    with pytest.raises(ValueError, match="side"):
        _run(problem, schedule=ladder,
             inflation=AdditiveInflation(u=DensePSD(jnp.eye(problem.P + 1))))

    class _NonPositive:
        n_steps, beta_target = 4, None

        def next_increment(self, evaluation):
            return -0.5

    with pytest.raises(ValueError, match="strictly positive"):
        _run(problem, schedule=_NonPositive())

    # -- unsupported operations propagate unmodified, before any evaluation ---
    before = len(problem.calls)
    with pytest.raises(UnsupportedOpError):
        run(problem.state(), problem.forward, jnp.asarray(problem.y),
            PSDLowRank(jnp.eye(problem.N)), update_rule=SQRT, schedule=ladder)
    assert len(problem.calls) == before

    # -- repair_failed_particles ------------------------------------------------
    with pytest.raises(ValueError, match="at least 2 valid"):
        repair_failed_particles(
            ensemble=Ensemble(u=jnp.zeros((4, 3))),
            valid=jnp.asarray([True, False, False, False]),
        )
    with pytest.raises(ValueError, match="boolean"):
        repair_failed_particles(ensemble=Ensemble(u=jnp.zeros((4, 3))), valid=jnp.ones(4))
    with pytest.raises(ValueError, match="weighted"):
        repair_failed_particles(
            ensemble=Ensemble(u=jnp.zeros((4, 3)), log_weights=jnp.zeros(4)),
            valid=jnp.ones(4, dtype=bool),
        )
    with pytest.raises(TypeError, match="Ensemble"):
        repair_failed_particles(ensemble=jnp.zeros((4, 3)), valid=jnp.ones(4, dtype=bool))

    # -- Evaluation and EKIResult ---------------------------------------------
    with pytest.raises(ValueError, match=PREDICTION):
        Evaluation(step=0, beta=0.0, ensemble=Ensemble(u=jnp.zeros((4, 3))),
                   whitened_residuals=jnp.zeros((4, 3)), n_valid=4)
    with pytest.raises(ValueError, match="whitened_residuals"):
        Evaluation(
            step=0, beta=0.0,
            ensemble=Ensemble({"u": jnp.zeros((4, 3)), PREDICTION: jnp.zeros((4, 2))}),
            whitened_residuals=jnp.zeros((4, 3)), n_valid=4,
        )
    with pytest.raises(ValueError, match="status"):
        EKIResult(state=problem.state(), history=(), status="stopping-rule",
                  last_evaluation=None)


def test_18_reprs_are_types_and_static_sizes_with_no_array_data():
    problem = _AffineProblem()
    state = problem.state()
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    evaluation = evaluate(state, problem.forward, y, noise)
    record = HistoryRecord.from_evaluation(evaluation, 0.5)
    result = _run(problem, schedule=FixedSchedule.uniform(2))
    assert repr(state) == (
        f"EKIState(n_particles={problem.J}, blocks={{'u': {problem.P}}}, step=0)"
    )
    assert repr(evaluation) == f"Evaluation(step=0, n_particles={problem.J})"
    assert repr(record) == "HistoryRecord(step=0)"
    assert repr(result) == (
        "EKIResult(status='schedule_exhausted', n_evaluations=2, beta=1)"
    )
    assert repr(DiscrepancyStop(tau=2.0)) == "DiscrepancyStop(tau=2.0)"
    assert repr(MultiplicativeInflation(1.02)) == (
        "MultiplicativeInflation(anomaly_scale=1.02)"
    )
    assert repr(RelaxToPriorSpread(0.5, names="u")) == (
        "RelaxToPriorSpread(alpha=0.5, names=('u',))"
    )
    long_ladder = repr(FixedSchedule.constant(1.0, 200))
    assert long_ladder == "FixedSchedule(n_steps=200, total=200.0)"


def test_18_the_pinned_prior_draw_and_a_short_run_are_snapshotted():
    """A JAX-side PRNG change is detected rather than absorbed."""
    prior = Gaussian(
        {"u": jnp.zeros(2)}, block_covs={"u": PSDDiagonal(jnp.asarray([1.0, 4.0]))}
    )
    state = EKIState.from_prior(jax.random.key(0), prior, 3)
    key_sample, key_state = jax.random.split(jax.random.key(0))
    assert np.array_equal(_u(state), np.asarray(prior.sample(key_sample, 3)["u"]))
    assert np.array_equal(
        np.asarray(jax.random.key_data(state.key)),
        np.asarray(jax.random.key_data(key_state)),
    )
    assert float(state.beta) == 0.0 and state.step == 0

    problem = _AffineProblem(P=2, N=2, J=4, seed=61)
    result = _run(
        problem, state=problem.state(seed=2), schedule=FixedSchedule.uniform(3),
        update_rule=MATHERON,
    )
    snapshot = np.array(_SNAPSHOT)
    assert np.abs(_u(result) - snapshot).max() < 1e-12


_SNAPSHOT = [
    [-2.061281709980334, -0.002336154421720818],
    [-0.1474200778291348, 1.2674414446457547],
    [-0.984980961245447, -0.6212018038214422],
    [-1.9752766112428959, -1.6984052122640743],
]


def test_19_the_result_reports_the_run_on_four_fixtures():
    """``stop_fired`` and ``budget_complete`` are each true exactly on their status."""
    problem = _AffineProblem()
    completed = _run(problem, schedule=FixedSchedule.uniform(4))
    assert completed.status == SCHEDULE_EXHAUSTED
    assert completed.budget_complete and not completed.stop_fired

    unfitted = _run(
        problem, schedule=FixedSchedule.constant(1.0, 5),
        stop=DiscrepancyStop(tau=1e-6), max_steps=5,
    )
    assert unfitted.budget_complete and not unfitted.stop_fired

    early = _run(problem, schedule=FixedSchedule.uniform(10), stop=_StopAtStepThree())
    assert early.stop_fired and not early.budget_complete
    assert float(early.beta) < 1.0

    interrupted = EKIResult(
        state=problem.state(), history=(), status=INTERRUPTED, last_evaluation=None
    )
    assert not interrupted.stop_fired and not interrupted.budget_complete


class _StopAtStepThree:
    """A stopping rule that fires on position, never on the misfits."""

    def __call__(self, evaluation) -> bool:
        return evaluation.step >= 3


def test_19_min_n_valid_is_the_minimum_over_the_history():
    """The worst step, not the last one and not the best."""
    problem = _AffineProblem(J=8)
    failures = iter([1, 3, 0, 2])

    def failing(u):
        v = jnp.asarray(u) @ jnp.asarray(problem.G).T
        return v.at[: next(failures)].set(jnp.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _run(problem, forward=failing, schedule=FixedSchedule.uniform(4),
                      on_failure="repair")
    per_step = [int(r.n_valid) for r in result.history]
    assert per_step == [problem.J - 1, problem.J - 3, problem.J, problem.J - 2]
    assert result.min_n_valid == min(per_step) == problem.J - 3


def test_19_last_evaluation_is_the_final_forward_call_and_off_by_one_where_it_should_be():
    problem = _AffineProblem()
    exhausted = _run(problem, schedule=FixedSchedule.uniform(4))
    assert exhausted.n_evaluations == len(problem.calls) == 4
    assert np.array_equal(_u(exhausted.last_evaluation), problem.calls[-1])
    assert np.array_equal(
        np.asarray(exhausted.last_evaluation.ensemble[PREDICTION]),
        np.asarray(problem.forward(problem.calls[-1])),
    )
    assert not np.array_equal(_u(exhausted), _u(exhausted.last_evaluation))

    stopped = _run(problem, schedule=FixedSchedule.uniform(10), stop=_StopAtStepThree())
    assert np.array_equal(_u(stopped), _u(stopped.last_evaluation))

    finished = _run(problem, state=stopped.state, schedule=FixedSchedule.uniform(3))
    assert finished.n_evaluations == 0 and finished.last_evaluation is None


def test_20_inflation_and_relaxation_see_the_true_ladder_and_are_applied_in_place():
    """Placement is asserted against an instrumented model, not assumed."""
    problem = _AffineProblem()
    seen: dict[str, list] = {"inflate": [], "relax": []}
    shift = 0.125

    def recording_inflation(key, *, ensemble, step, beta, **_):
        seen["inflate"].append((step, float(beta)))
        return ensemble.assign(u=ensemble["u"] + shift)

    def recording_relaxation(*, prior, posterior, step, beta, **_):
        seen["relax"].append((step, float(beta), prior.names, posterior.names))
        return posterior

    increments = (0.2, 0.3, 0.5)
    result = _run(
        problem, schedule=FixedSchedule(increments),
        inflation=recording_inflation, relaxation=recording_relaxation,
    )
    levels = [0.0, 0.2, 0.5]
    assert seen["inflate"] == [(t, levels[t]) for t in range(3)]
    assert seen["relax"] == [(t, levels[t], ("u", PREDICTION), ("u",)) for t in range(3)]

    # The model's first input is the initial particles plus the shift: the
    # inflation runs before the evaluation, so the ensemble the caller
    # supplied is never itself evaluated.
    assert np.array_equal(problem.calls[0], np.asarray(problem.members + shift))
    assert not np.array_equal(_u(result), _u(result.last_evaluation))

    # A relaxation that varies with beta still gives an exactly resumable run.
    def beta_dependent(*, prior, posterior, beta, **_):
        return posterior.assign(u=posterior["u"] * (1.0 + 0.01 * beta))

    whole = _run(problem, schedule=FixedSchedule(increments), relaxation=beta_dependent)
    part = _run(
        problem, schedule=FixedSchedule(increments[:1]), relaxation=beta_dependent
    )
    rest = _run(problem, state=part.state, schedule=FixedSchedule(increments),
                relaxation=beta_dependent)
    assert np.array_equal(_u(rest), _u(whole))


def test_20_the_relaxations_relax_the_update_toward_the_evaluated_particles():
    """``prior`` is the update's input: the inflated, evaluated particles."""
    problem = _AffineProblem()
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    state = problem.state()
    inflation = MultiplicativeInflation(1.3)
    evaluation = evaluate(state, problem.forward, y, noise, inflation=inflation)
    plain = assimilate(state, evaluation, 0.5, y, noise, update_rule=SQRT)
    for relaxation, function in (
        (RelaxToPriorSpread(0.4), kalman.relax_to_prior_spread),
        (RelaxToPriorPerturbations(0.4), kalman.relax_to_prior_perturbations),
    ):
        relaxed = assimilate(
            state, evaluation, 0.5, y, noise, update_rule=SQRT, relaxation=relaxation
        )
        want = function(evaluation.ensemble, plain.ensemble, 0.4)
        assert np.array_equal(_u(relaxed), np.asarray(want["u"]))


def test_20_the_reported_spread_is_of_the_ensemble_that_was_evaluated():
    """``spread`` describes the post-inflation, post-repair particles."""
    problem = _AffineProblem(J=8)
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    plain = evaluate(problem.state(), problem.forward, y, noise)
    before = float(plain.rms_parameter_spread)

    factor = 3.0
    inflated = evaluate(
        problem.state(), problem.forward, y, noise,
        inflation=MultiplicativeInflation(factor),
    )
    got = float(inflated.rms_parameter_spread)
    assert got == pytest.approx(factor * before, rel=1e-11)

    def failing(u):
        v = jnp.asarray(u) @ jnp.asarray(problem.G).T
        return v.at[2, 0].set(jnp.nan)

    repaired = evaluate(problem.state(), failing, y, noise, on_failure="repair")
    valid = jnp.asarray([True, True, False, True, True, True, True, True])
    members = repair_failed_particles(
        ensemble=Ensemble(u=jnp.asarray(problem.members)), valid=valid
    )
    anomalies = _u(members) - _u(members).mean(axis=0)
    want = np.linalg.norm(anomalies) / np.sqrt((problem.J - 1) * problem.P)
    assert float(repaired.rms_parameter_spread) == pytest.approx(want, rel=1e-11)
    assert float(repaired.rms_parameter_spread) < before


def test_21_every_record_field_agrees_with_the_evaluation_it_came_from():
    """The only guard against a plausible-scalar bug in any of eleven fields."""
    problem = _AffineProblem(J=9)
    increments = (0.15, 0.35, 0.2, 0.3)
    result = _run(problem, schedule=FixedSchedule(increments))
    W = _recovered_whitener(problem.noise_cov, problem.N)
    level = 0.0
    pairs = zip(result.history, problem.calls, strict=True)
    for index, (record, members) in enumerate(pairs):
        predictions = np.asarray(members) @ problem.G.T
        residuals = (problem.y - predictions) @ W.T
        phi = 0.5 * np.sum(residuals**2, axis=1)
        anomalies = members - members.mean(axis=0)
        spread = np.linalg.norm(anomalies) / np.sqrt((problem.J - 1) * problem.P)
        weights = np.exp(-increments[index] * (phi - phi.min()))

        assert int(record.step) == index
        assert int(record.n_valid) == problem.J
        assert float(record.beta) == pytest.approx(level, abs=8 * EPS)
        assert float(record.increment) == pytest.approx(increments[index])
        want_next = level + increments[index]
        assert float(record.beta_next) == pytest.approx(want_next, abs=8 * EPS)
        assert float(record.misfit_mean) == pytest.approx(phi.mean(), rel=1e-11)
        assert float(record.misfit_min) == pytest.approx(phi.min(), rel=1e-11)
        assert float(record.misfit_max) == pytest.approx(phi.max(), rel=1e-11)
        assert float(record.center_misfit) == pytest.approx(
            0.5 * np.sum(residuals.mean(axis=0) ** 2), rel=1e-10
        )
        assert float(record.spread) == pytest.approx(spread, rel=1e-11)
        assert float(record.ess) == pytest.approx(
            weights.sum() ** 2 / (weights**2).sum(), rel=1e-11
        )
        level += increments[index]


def test_22_the_parameter_spread_is_exact_against_a_closed_form():
    """A divisor of J and a missing 1/sqrt(P) are separately distinguishable."""
    J, P = 5, 3
    for c in (1.0, 0.25, 7.0):
        members = _exact_moment_ensemble(J, np.zeros(P), c * np.eye(P))
        assert _spread_of({"u": members}) == pytest.approx(c, rel=1e-12)

    a, b, d = 1.0, 2.0, 4.0
    members = _exact_moment_ensemble(J, np.zeros(P), np.diag([a, b, d]))
    got = _spread_of({"u": members})
    assert got == pytest.approx(np.sqrt((a**2 + b**2 + d**2) / 3), rel=1e-12)
    assert got != pytest.approx(np.sqrt((a**2 + b**2 + d**2) / 3 * (J - 1) / J))
    assert got != pytest.approx(np.sqrt(a**2 + b**2 + d**2))

    # Over several blocks, the same root mean square over every coordinate.
    split = _spread_of({"a": members[:, :1], "b": members[:, 1:]})
    assert split == pytest.approx(got, rel=1e-12)


def _spread_of(blocks: dict) -> float:
    J = next(iter(blocks.values())).shape[0]
    evaluation = Evaluation(
        step=0, beta=0.0,
        ensemble=Ensemble({**{n: jnp.asarray(b) for n, b in blocks.items()},
                           PREDICTION: jnp.zeros((J, 2))}),
        whitened_residuals=jnp.zeros((J, 2)), n_valid=J,
    )
    return float(evaluation.rms_parameter_spread)


def test_23_a_finished_ladder_is_a_no_op_and_restart_gives_the_full_one():
    problem = _AffineProblem()
    for schedule in (FixedSchedule.uniform(4), AdaptiveESSSchedule(beta_target=1.0)):
        finished = _run(problem, schedule=schedule)
        assert finished.n_evaluations > 0
        calls_before = len(problem.calls)

        again = _run(problem, state=finished.state, schedule=schedule)
        assert again.status == SCHEDULE_EXHAUSTED
        assert again.n_evaluations == 0
        assert again.n_completed_steps == 0
        assert again.history == ()
        assert again.last_evaluation is None
        assert np.array_equal(_u(again), _u(finished))
        assert len(problem.calls) == calls_before

        restarted = _run(problem, state=finished.state.restart(), schedule=schedule)
        assert restarted.n_evaluations == finished.n_evaluations
        assert float(restarted.state.beta) == pytest.approx(float(finished.beta))


def test_23_a_no_op_run_says_so_at_warning_level(caplog):
    problem = _AffineProblem()
    finished = _run(problem, schedule=FixedSchedule.uniform(2))
    with caplog.at_level(logging.WARNING, logger="enskit.algorithms.eki"):
        _run(problem, state=finished.state, schedule=FixedSchedule.uniform(2))
    assert "no evaluations" in caplog.text
    assert "restart" in caplog.text


def test_24_the_driver_hands_the_update_both_repaired_blocks():
    """Repairing one block and not the other is a silent wrong answer."""
    problem = _AffineProblem(J=6)
    shift = 0.0625
    received: dict[str, Ensemble] = {}

    class _Recording:
        def build(self, particles, approximation, given):
            received["particles"] = particles
            return SQRT.build(particles, approximation, given)

    def failing(u):
        v = jnp.asarray(u) @ jnp.asarray(problem.G).T
        return v.at[2, 0].set(jnp.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _run(
            problem, forward=failing, schedule=FixedSchedule.uniform(1),
            inflation=lambda key, *, ensemble, **_: ensemble.assign(
                u=ensemble["u"] + shift
            ),
            update_rule=_Recording(), on_failure="repair",
        )

    inflated = jnp.asarray(problem.members + shift)
    raw = (inflated @ jnp.asarray(problem.G).T).at[2, 0].set(jnp.nan)
    valid = jnp.all(jnp.isfinite(raw), axis=-1)
    want = repair_failed_particles(
        ensemble=Ensemble({"u": inflated, PREDICTION: raw}), valid=valid
    )
    got = received["particles"]
    assert got.names == ("u", PREDICTION)
    for n in got.names:
        assert np.array_equal(np.asarray(got[n]), np.asarray(want[n]))


@pytest.mark.parametrize(
    "schedule",
    [
        FixedSchedule.uniform(3),
        FixedSchedule.constant(0.3, 3),
        AdaptiveESSSchedule(beta_target=0.6, min_increment=0.2, max_increment=0.2),
        AdaptiveMisfitSchedule(beta_target=0.6, min_increment=0.2, max_increment=0.2),
    ],
)
@pytest.mark.parametrize("update_rule", [SQRT, MATHERON])
@pytest.mark.parametrize("inflation_kind", ["none", "multiplicative", "additive"])
@pytest.mark.parametrize("relaxation_kind", ["none", "rtps"])
@pytest.mark.parametrize("with_stop", [False, True])
def test_25_the_axes_compose(schedule, update_rule, inflation_kind, relaxation_kind,
                             with_stop):
    """Every run terminates, reports a permitted status, stacks and stays finite."""
    problem = _AffineProblem(P=2, N=3, J=5, seed=71)
    inflation = {
        "none": None,
        "multiplicative": MultiplicativeInflation(1.01),
        "additive": AdditiveInflation(u=DensePSD(jnp.eye(problem.P) * 0.01)),
    }[inflation_kind]
    relaxation = {"none": None, "rtps": RelaxToPriorSpread(0.3)}[relaxation_kind]
    result = _run(
        problem, schedule=schedule, update_rule=update_rule, inflation=inflation,
        relaxation=relaxation, stop=DiscrepancyStop(tau=1e-8) if with_stop else None,
        max_steps=20,
    )
    assert result.status in (SCHEDULE_EXHAUSTED, STOPPING_RULE)
    assert result.n_evaluations == len(problem.calls)
    assert np.all(np.isfinite(_u(result)))
    stacked = result.stacked
    assert stacked.batch_shape == (result.n_evaluations,)
    assert np.all(np.isfinite(np.asarray(stacked.ess)))
    assert np.all(np.asarray(stacked.ess) >= 1.0 - 1e-9)


# ---------------------------------------------------------------------------
# 26. every runnable block of the contract page
# ---------------------------------------------------------------------------


def test_26_the_two_form_example_runs():
    problem = _AffineProblem()
    key = jax.random.key(0)
    prior = problem.prior()
    forward, y, noise_cov = problem.forward, jnp.asarray(problem.y), problem.noise_cov

    state = eki.EKIState.from_prior(key, prior, n_particles=64)

    sampled = eki.run(state, forward, y, noise_cov,
                      update_rule=kalman.Matheron(),
                      schedule=eki.AdaptiveESSSchedule())
    particles = sampled.ensemble["u"]
    center = sampled.mean("u")
    assert particles.shape == (64, problem.P)
    assert center.shape == (problem.P,)

    fit = eki.run(state, forward, y, noise_cov,
                  update_rule=kalman.SymmetricSquareRoot(),
                  schedule=eki.FixedSchedule.constant(1.0, n_steps=200),
                  stop=eki.DiscrepancyStop(tau=1.0), max_steps=200)
    assert isinstance(fit.stop_fired, bool)


def test_26_the_pinned_prior_draw_and_restart_blocks_run():
    problem = _AffineProblem()
    key = jax.random.key(9)
    prior = problem.prior()

    key_sample, key_state = jax.random.split(key)
    pinned = EKIState(prior.sample(key_sample, 8), key=key_state)
    assert np.array_equal(_u(pinned), _u(EKIState.from_prior(key, prior, 8)))

    state = _run(problem, state=pinned, schedule=FixedSchedule.uniform(10)).state
    phase2 = state.restart()  # step = 0, beta = 0.0, same particles and key
    assert phase2.step == 0 and float(phase2.beta) == 0.0
    assert np.array_equal(_u(phase2), _u(state))


def test_26_the_backtracking_loop_runs_and_costs_what_the_contract_says():
    """One evaluation per step plus one per rejection."""
    problem = _AffineProblem()
    forward, y, noise_cov = problem.forward, jnp.asarray(problem.y), problem.noise_cov
    rule = kalman.SymmetricSquareRoot()
    accepted, rejections = 0, 0

    def done(_evaluation):
        return accepted >= 3

    s, delta = problem.state(), 1.0
    current = eki.evaluate(s, forward, y, noise_cov)
    while not done(current):
        trial = eki.assimilate(s, current, delta, y, noise_cov, update_rule=rule)
        probe = eki.evaluate(trial, forward, y, noise_cov)
        if probe.center_misfit < current.center_misfit:
            s, current, delta = trial, probe, delta * 1.5     # accept, lengthen
            accepted += 1
        else:
            delta = delta / 2                                 # reject, reuse current
            rejections += 1
        assert rejections < 20
    assert len(problem.calls) == 1 + accepted + rejections


def test_26_the_stacked_and_moments_blocks_run():
    problem = _AffineProblem()
    result = _run(problem, schedule=FixedSchedule.uniform(4))
    xs, ys = result.stacked.step, result.stacked.misfit_mean
    assert xs.shape == ys.shape == (4,)

    fit = result.ensemble.project()
    assert fit.cov("u").diag().shape == (problem.P,)
    assert fit.sample(jax.random.key(1), 1000)["u"].shape == (1000, problem.P)


def test_26_the_eki_error_checkpoint_and_interrupted_result_patterns_run():
    problem = _AffineProblem()
    forward, y, noise_cov = problem.forward, jnp.asarray(problem.y), problem.noise_cov
    sched = FixedSchedule.constant(0.25, 40)
    rule = kalman.SymmetricSquareRoot()
    checkpointed, diagnosed = [], []

    try:
        eki.run(problem.state(), forward, y, noise_cov, update_rule=rule,
                schedule=sched, max_steps=3)
    except eki.EKIError as exc:
        checkpointed.append(exc.state)
        diagnosed.append(exc.history)
    assert len(checkpointed) == 1 and len(diagnosed[0]) == 3

    records = []
    for state, record, evaluation in eki.iterate(  # noqa: B007
        problem.state(), forward, y, noise_cov, update_rule=rule, schedule=sched
    ):
        records.append(record)
        if len(records) >= 2:
            break
    result = eki.EKIResult(state=state, history=tuple(records),
                           status=eki.INTERRUPTED, last_evaluation=evaluation)
    assert result.status == INTERRUPTED
    assert not result.stop_fired and not result.budget_complete
    assert result.n_evaluations == 2


def test_26_the_tikhonov_augmentation_needs_no_new_code_and_double_counts():
    """The augmentation runs, and its documented hazard is asserted exactly.

    Started from a prior ensemble and run to beta = 1, the prior enters twice,
    and the result is over-concentrated by exactly one extra copy of the
    prior precision.
    """
    problem = _AffineProblem()
    forward, y = problem.forward, jnp.asarray(problem.y)
    noise_cov = problem.noise_cov
    prior_mean, prior_cov = jnp.asarray(problem.m0), DensePSD(jnp.asarray(problem.C0))

    def forward_aug(u):
        return jnp.concatenate([forward(u), u], axis=-1)

    y_aug = jnp.concatenate([y, prior_mean])
    noise_aug = block_diag(noise_cov, prior_cov)

    result = run(problem.state(), forward_aug, y_aug, noise_aug, update_rule=SQRT,
                 schedule=FixedSchedule((0.5, 0.25, 0.25)))
    _, got_cov = _moments(_u(result))
    prior_precision = np.linalg.inv(problem.C0)
    data_precision = problem.G.T @ np.linalg.solve(problem.R, problem.G)
    doubled = np.linalg.inv(2 * prior_precision + data_precision)
    _, honest = problem.posterior()
    scale = np.abs(doubled).max()
    assert np.abs(got_cov - doubled).max() < 1e3 * EPS * scale
    assert np.abs(got_cov - honest).max() > 1e6 * EPS * scale
    assert np.trace(got_cov) < np.trace(honest)


def test_26_the_external_executable_wrapper_runs(tmp_path):
    """A forward model that runs a subprocess per particle and catches its failures.

    The wrapper obligation (catch your own failures and return a non-finite
    row) is exercised against a real process that exits non-zero on a
    negative decay rate, which the prior puts mass on.
    """
    solver = tmp_path / "solver.py"
    solver.write_text(
        "import sys, math\n"
        "u = [float(x) for x in open(sys.argv[1])]\n"
        "if u[1] < 0.0:\n"
        "    sys.exit('solver diverged: negative decay rate')\n"
        "with open(sys.argv[2], 'w') as out:\n"
        "    for t in (0.5, 1.0, 2.0):\n"
        "        out.write(repr(u[0] * math.exp(-u[1] * t)) + '\\n')\n"
    )
    data_dim = 3

    def forward(u):
        particles = np.asarray(u)
        predictions = np.full((particles.shape[0], data_dim), np.nan)
        for j, particle in enumerate(particles):
            path_in, path_out = tmp_path / f"in_{j}.txt", tmp_path / f"out_{j}.txt"
            path_out.unlink(missing_ok=True)
            np.savetxt(path_in, particle)
            try:
                subprocess.run(
                    [sys.executable, str(solver), str(path_in), str(path_out)],
                    check=True, capture_output=True, timeout=60,
                )
                row = np.loadtxt(path_out)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                    OSError, ValueError):
                continue
            if row.shape == (data_dim,):
                predictions[j] = row
        return predictions

    times = jnp.array([0.5, 1.0, 2.0])
    truth = jnp.array([2.0, 0.7])
    y = truth[0] * jnp.exp(-truth[1] * times) + jnp.array([0.02, -0.01, 0.015])
    noise = PSDDiagonal(jnp.full(data_dim, 0.01))
    prior = Gaussian({"u": jnp.array([1.0, 1.0])},
                     block_covs={"u": PSDDiagonal(jnp.array([1.0, 0.5]))})
    state = EKIState.from_prior(jax.random.key(0), prior, n_particles=32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run(state, forward, y, noise, update_rule=MATHERON,
                     schedule=AdaptiveESSSchedule(), on_failure="repair")
    assert result.min_n_valid < state.n_particles
    assert [w for w in caught if "were not finite and were repaired" in str(w.message)]
    assert result.last_evaluation.ensemble[PREDICTION].dtype == jnp.float64
    assert np.abs(np.asarray(result.mean("u")) - np.asarray(truth)).max() < 0.2


def test_27_the_forward_model_receives_what_the_contract_promises():
    """Concrete, one array per input block, in the particles' dtype, post-inflation."""
    problem = _AffineProblem(J=8)
    state = problem.state()
    seen = []

    def recording(u):
        assert isinstance(u, jax.Array)
        assert not isinstance(u, jax.core.Tracer)
        assert u.shape == (problem.J, problem.P)
        assert u.dtype == state.ensemble["u"].dtype
        assert not np.asarray(u).flags.writeable
        seen.append(np.asarray(u))
        return u @ jnp.asarray(problem.G).T

    _run(problem, forward=recording, schedule=FixedSchedule.uniform(2))
    assert len(seen) == 2
    assert np.array_equal(seen[0], _u(state))

    seen.clear()
    factor = 3.0
    _run(problem, forward=recording, schedule=FixedSchedule.uniform(1),
         inflation=MultiplicativeInflation(factor))
    members = _u(state)
    center = members.mean(axis=0, keepdims=True)
    expected = center + factor * (members - center)
    assert np.abs(seen[0] - expected).max() < 64 * EPS * np.abs(expected).max()


def test_28_the_accepted_containers_are_a_promise_not_a_tolerance():
    """A jax array, a numpy array and a nested list give bit-identical runs."""
    problem = _AffineProblem(J=8)
    G = np.asarray(problem.G)

    def as_jax(u):
        return jnp.asarray(np.asarray(u) @ G.T)

    def as_numpy(u):
        return np.asarray(u) @ G.T

    def as_list(u):
        return (np.asarray(u) @ G.T).tolist()

    runs = [_run(problem, forward=f, schedule=FixedSchedule.uniform(3))
            for f in (as_jax, as_numpy, as_list)]
    reference = _u(runs[0])
    for other in runs[1:]:
        assert np.array_equal(_u(other), reference)
        assert other.last_evaluation.ensemble[PREDICTION].dtype == jnp.float64


def test_29_a_narrow_forward_model_is_promoted_and_warned_about():
    """The maps layer's rule, through the driver, which leaves warnings alone.

    Every evaluation that promotes warns, at the caller's line, and Python's
    filters decide what is shown: under ``always`` every one, under the
    default once per location. The driver neither records nor re-issues
    warnings, so an ``error`` filter raises at its origin and a failing run
    still raises its ``EKIError``, carrying the state.
    """
    problem = _AffineProblem(J=8)
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    state = problem.state()

    def coarse(u):
        return (jnp.asarray(u) @ jnp.asarray(problem.G).T).astype(jnp.float32)

    def promotions(caught):
        return [w for w in caught if "promoted" in str(w.message)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _run(problem, forward=coarse, schedule=FixedSchedule.uniform(4))
    assert len(promotions(caught)) == 4
    assert "float32" in str(promotions(caught)[0].message)
    assert __file__.endswith(promotions(caught)[0].filename.rsplit("/", 1)[-1])
    assert result.last_evaluation.ensemble[PREDICTION].dtype == jnp.float64
    assert result.ensemble["u"].dtype == jnp.float64

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        _run(problem, forward=coarse, schedule=FixedSchedule.uniform(4))
    assert len(promotions(caught)) == 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(problem, schedule=FixedSchedule.uniform(4))
    assert not promotions(caught)

    # Under an error filter the warning raises where it is issued, at the
    # first evaluation; and a failure is still an EKIError with its state.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="promoted"):
            _run(problem, forward=coarse, schedule=FixedSchedule.uniform(4))

        def failing(u):
            return problem.forward(u).at[1, 0].set(jnp.nan)

        with pytest.raises(EKIError) as caught_error:
            _run(problem, forward=failing, schedule=FixedSchedule.uniform(4))
        assert caught_error.value.state is not None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = evaluate(state, coarse, y, noise)
        evaluate(state, coarse, y, noise)
    assert len(promotions(caught)) == 2
    assert first.ensemble[PREDICTION].dtype == jnp.float64

    with pytest.raises(ValueError, match="int64"):
        _run(problem, forward=lambda u: np.asarray(np.asarray(u) @ problem.G.T, np.int64),
             schedule=FixedSchedule.uniform(2))


@pytest.mark.xfail(strict=True, raises=TypeError, reason="#68: R / delta is float64")
@pytest.mark.parametrize("update_rule", [SQRT, MATHERON])
def test_29_a_float32_run_stays_float32(update_rule):
    """A float32 run with float32 data and noise stays float32.

    It does not yet: scaling an operator promotes it to float64 (#68), so the
    tempered noise is float64 and the Kalman layer's dtype check refuses the
    update. Strict, so that fixing #68 turns this into a failure, and the
    contract's sentence about float32 runs is removed with it.
    """
    problem = _AffineProblem(J=8)
    members = jnp.asarray(problem.members, jnp.float32)
    G = jnp.asarray(problem.G, jnp.float32)
    state = EKIState(Ensemble(u=members), key=jax.random.key(0))
    noise = PSDDiagonal(jnp.full(problem.N, 0.5, jnp.float32))
    result = run(state, lambda u: u @ G.T, jnp.asarray(problem.y, jnp.float32), noise,
                 update_rule=update_rule, schedule=FixedSchedule.uniform(2))
    assert result.ensemble["u"].dtype == jnp.float32


def test_30_the_evaluation_and_update_counts_hold_on_every_exit():
    """One evaluation per step, plus one when stopping needed a look."""
    problem = _AffineProblem(J=8)

    class _Counted:
        def __init__(self):
            self.calls = 0

        def __call__(self, u):
            self.calls += 1
            return jnp.asarray(u) @ jnp.asarray(problem.G).T

    class _StopAt:
        def __init__(self, k):
            self.k = k

        def __call__(self, evaluation):
            return int(evaluation.step) >= self.k

    class _NoneAt:
        n_steps, beta_target = None, None

        def __init__(self, k):
            self.k = k

        def next_increment(self, evaluation):
            return None if int(evaluation.step) >= self.k else 0.1

    cases = [
        ("fixed ladder", dict(schedule=FixedSchedule.uniform(5)), 0),
        ("budgeted adaptive", dict(schedule=AdaptiveESSSchedule()), 0),
        ("stopping rule", dict(schedule=FixedSchedule.constant(0.1, n_steps=50),
                               stop=_StopAt(3)), 1),
        ("increment None", dict(schedule=_NoneAt(3)), 1),
    ]
    for label, kwargs, terminal in cases:
        state = problem.state()
        model = _Counted()
        result = _run(problem, state=state, forward=model, **kwargs)
        assert result.n_evaluations == model.calls, label
        assert result.n_completed_steps == result.state.step - state.step, label
        assert result.n_evaluations - result.n_completed_steps == terminal, label
        zeros = [i for i, r in enumerate(result.history) if float(r.increment) == 0.0]
        assert zeros == ([len(result.history) - 1] if terminal else []), label

    class _Unbounded:
        n_steps, beta_target = None, None

        def next_increment(self, evaluation):
            return 0.1

    for bound in (1, 3, 7):
        model = _Counted()
        with pytest.raises(EKIError, match="max_steps"):
            _run(problem, forward=model, schedule=_Unbounded(), max_steps=bound)
        assert model.calls == bound


def test_31_several_parameter_blocks_and_inputs():
    """Named blocks: order kept, ``inputs`` chooses and orders the arguments.

    The same affine problem split into two blocks gives the single-block run's
    particles, column for column, to round-off.
    """
    problem = _AffineProblem()
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    whole = _run(problem, schedule=FixedSchedule((0.25, 0.75)))

    G = jnp.asarray(problem.G)

    def split_forward(b, a):
        return a @ G[:, :1].T + b @ G[:, 1:].T

    members = jnp.asarray(problem.members)
    state = EKIState(Ensemble(a=members[:, :1], b=members[:, 1:]), key=jax.random.key(0))
    result = run(state, split_forward, y, noise, update_rule=SQRT,
                 schedule=FixedSchedule((0.25, 0.75)), inputs=("b", "a"))
    assert result.ensemble.names == ("a", "b")
    joined = np.concatenate([np.asarray(result.ensemble["a"]),
                             np.asarray(result.ensemble["b"])], axis=1)
    assert np.abs(joined - _u(whole)).max() < 1e3 * EPS * np.abs(_u(whole)).max()
    assert result.last_evaluation.ensemble.names == ("a", "b", PREDICTION)

    # A block the forward model does not read is still updated through its
    # correlation with the others, and the model never sees it.
    seen = []

    def only_a(a):
        seen.append(a.shape)
        return jnp.tile(a, (1, problem.N))

    run(state, only_a, y, noise, update_rule=SQRT, schedule=FixedSchedule((1.0,)),
        inputs="a")
    assert seen == [(problem.J, 1)]


def test_32_the_approximation_hook_reaches_every_update():
    """``approximation`` is called once per step with the tempered noise."""
    problem = _AffineProblem()
    calls = []

    def recording(ensemble, noise):
        calls.append((ensemble.names, tuple(noise), noise[PREDICTION]))
        return kalman.gaussian_approximation(ensemble, noise)

    increments = (0.25, 0.75)
    with_hook = _run(problem, schedule=FixedSchedule(increments), approximation=recording)
    default = _run(problem, schedule=FixedSchedule(increments))
    assert np.array_equal(_u(with_hook), _u(default))
    assert [c[:2] for c in calls] == [(("u", PREDICTION), (PREDICTION,))] * 2
    for (_, _, tempered), increment in zip(calls, increments, strict=True):
        assert np.abs(
            np.asarray(tempered.to_dense()) - problem.R / increment
        ).max() < 1e-12 * np.abs(problem.R).max() / increment

    # A plain Gaussian takes Matheron's general path and agrees to round-off.
    def plain(ensemble, noise):
        approx = kalman.gaussian_approximation(ensemble, noise)
        return Gaussian(
            {n: approx.mean(n) for n in approx.names},
            factors={n: approx.factor(n) for n in approx.names},
            block_covs={n: approx.block_cov(n) for n in approx.names
                        if approx.block_cov(n) is not None},
        )

    general = _run(problem, schedule=FixedSchedule(increments), update_rule=MATHERON,
                   approximation=plain)
    aligned = _run(problem, schedule=FixedSchedule(increments), update_rule=MATHERON)
    assert np.abs(_u(general) - _u(aligned)).max() < 1e-10 * np.abs(_u(aligned)).max()
    with pytest.raises(ValueError):
        _run(problem, schedule=FixedSchedule(increments), approximation=plain)


def test_33_the_key_split_is_four_way_and_a_keyed_simulator_gets_the_third():
    """``(next, inflate, evaluate, update)``, and ``needs_key`` simulators."""
    problem = _AffineProblem()
    received = []

    def stochastic(key, u):
        received.append(np.asarray(jax.random.key_data(key)))
        return problem.forward(u) + 0.01 * jax.random.normal(key, (problem.J, problem.N))

    stochastic.needs_key = True
    state = problem.state(seed=4)
    result = _run(
        problem, state=state, forward=stochastic, schedule=FixedSchedule.uniform(2)
    )
    key = state.key
    for step in range(2):
        key_next, _, key_evaluate, _ = _keys(key)
        want = np.asarray(jax.random.key_data(key_evaluate))
        assert np.array_equal(received[step], want)
        key = key_next
    assert np.array_equal(
        np.asarray(jax.random.key_data(result.state.key)),
        np.asarray(jax.random.key_data(key)),
    )


def test_26_the_running_an_inversion_page_runs():
    """Every Python block of ``docs/user-guide/running-an-inversion.md``, in order.

    The blocks run in one namespace, and the page's claims are then checked
    against what they built, so a block that runs but no longer shows what
    the prose says fails here.
    """
    import re
    from pathlib import Path

    page = Path(__file__).parents[1] / "docs" / "user-guide" / "running-an-inversion.md"
    blocks = re.findall(r"```python\n(.*?)```", page.read_text(), re.S)
    assert len(blocks) >= 15
    ns: dict = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for block in blocks:
            exec(compile(block, str(page), "exec"), ns)

    assert ns["result"].budget_complete and float(ns["result"].beta) == 1.0
    assert ns["fit"].stop_fired and not ns["fit"].budget_complete
    assert np.isfinite(float(ns["log_target"](ns["problem"].u_true)))
    # The failure example raises by default, and repair is reported.
    assert "were not finite" in str(ns["failure"])
    assert ns["repaired"].min_n_valid < ns["start"].n_particles
    assert [w for w in caught if "were not finite and were repaired" in str(w.message)]
    # Relaxation keeps more spread than the plain optimization form.
    relaxed, fit = ns["relaxed"].stacked.spread, ns["fit"].stacked.spread
    assert float(relaxed[len(fit) - 1]) > float(fit[-1])
    assert ns["split"].ensemble.names == ("amplitude", "rate")
    assert ns["final"].ensemble[PREDICTION].shape == (64, ns["problem"].data_dim)
    assert ns["phase2"].step == 0 and float(ns["phase2"].beta) == 0.0
    assert np.array_equal(_u(ns["moved"]), np.asarray(ns["by_hand"]["u"]))
    assert ns["interrupted"].n_evaluations == 3
    assert np.array_equal(_u(ns["resumed"]), _u(ns["whole"]))
    assert ns["geometric"].budget_complete


def test_26_the_landing_page_example_runs():
    """Every Python block of ``docs/index.md``, in order, in one namespace."""
    import re
    from pathlib import Path

    page = Path(__file__).parents[1] / "docs" / "index.md"
    blocks = re.findall(r"```python\n(.*?)```", page.read_text(), re.S)
    assert len(blocks) >= 1
    ns: dict = {}
    for block in blocks:
        exec(compile(block, str(page), "exec"), ns)
    assert ns["result"].budget_complete
    # The posterior mean's prediction fits the data to within the noise.
    fitted = np.asarray(ns["forward"](ns["result"].mean("u")[None])[0])
    assert np.abs(fitted - np.asarray(ns["y"])).max() < 0.3


def test_35_a_localized_rule_runs_through_the_driver_unchanged():
    """``LocalizedUpdateRule`` plugs into ``update_rule=`` with no driver change.

    At a radius so large that every taper weight is 1 to round-off and every
    datum is in every neighborhood, the localized run is the plain one. At a
    small radius it differs, and stays finite.
    """
    problem = _AffineProblem(P=6, N=4, J=8, seed=23)
    # A localized rule needs row-local noise: restricting a correlated block
    # to a neighborhood is not an operator-layer operation.
    problem.R = np.diag(np.diag(problem.R))
    problem.noise_cov = PSDDiagonal(jnp.asarray(np.diag(problem.R)))
    sites = jnp.arange(problem.P, dtype=float)[:, None]
    data_sites = jnp.asarray([[0.5], [2.0], [3.5], [5.0]])
    plain = _run(problem, schedule=FixedSchedule((0.5, 0.5)))
    for rule in (SQRT, MATHERON):
        wide = kalman.DomainLocalization(
            {"u": sites}, data_sites, radius=1e9, max_neighbors=problem.N
        )
        localized = _run(problem, schedule=FixedSchedule((0.5, 0.5)),
                         update_rule=kalman.LocalizedUpdateRule(rule, wide))
        reference = _run(problem, schedule=FixedSchedule((0.5, 0.5)), update_rule=rule)
        scale = np.abs(_u(reference)).max()
        assert np.abs(_u(localized) - _u(reference)).max() < 1e-10 * scale
    narrow = kalman.DomainLocalization(
        {"u": sites}, data_sites, radius=1.5, max_neighbors=2
    )
    local = _run(problem, schedule=FixedSchedule((0.5, 0.5)),
                 update_rule=kalman.LocalizedUpdateRule(SQRT, narrow))
    assert np.all(np.isfinite(_u(local)))
    assert np.abs(_u(local) - _u(plain)).max() > 1e-3


# ===========================================================================
# Section 2 -- one targeted regression test per silent-failure class
#
# Several classes are pinned by a conformance test above and are named here
# rather than duplicated: the R/beta mis-scaling (test 2), the non-log-space
# ESS (test 5), a repair applied when nothing failed (test 9), a HistoryRecord
# field declared static (test 15), the safety bound checked before ladder
# exhaustion (test 4), the min/max inversion and the clamp-precedence
# inversions (test 4), the bisection returning `hi` (test 4), a record field
# disagreeing with its evaluation (test 21), two evaluations per step
# (tests 4, 16 and 19), chaining a fresh ladder onto a finished state
# (test 23), DiscrepancyStop on a budgeted ladder (test 19), and the Tikhonov
# augmentation at beta = 1 (test 26).
# ===========================================================================


def test_regression_inflation_scales_the_anomalies_not_the_covariance():
    """``anomaly_scale`` multiplies the anomalies: 1.2 scales the covariance by 1.44."""
    rng = np.random.default_rng(83)
    ens = Ensemble(u=jnp.asarray(rng.normal(size=(12, 3))))
    _, before = _moments(_u(ens))
    inflated = MultiplicativeInflation(1.2)(
        jax.random.key(0), ensemble=ens, step=0, beta=0.0
    )
    _, after = _moments(_u(inflated))
    assert np.abs(after - 1.44 * before).max() < 64 * EPS * np.abs(before).max()
    assert np.abs(after - 1.2 * before).max() > 1e-3 * np.abs(before).max()


def test_regression_the_repair_does_not_rescale_the_surviving_particles():
    """The moment-exact variant would inflate silently, by sqrt(99/89) here."""
    rng = np.random.default_rng(89)
    J, P = 100, 4
    u = jnp.asarray(rng.normal(size=(J, P)))
    mask = np.ones(J, dtype=bool)
    mask[:10] = False
    repaired = repair_failed_particles(ensemble=Ensemble(u=u), valid=jnp.asarray(mask))
    assert np.array_equal(_u(repaired)[mask], np.asarray(u)[mask])
    n_valid = int(mask.sum())
    rescaling = np.sqrt((J - 1) / (n_valid - 1))
    assert rescaling == pytest.approx(np.sqrt(99 / 89), rel=1e-12)
    u_hat = np.asarray(u)[mask].mean(axis=0)
    moment_exact = u_hat + (np.asarray(u)[mask] - u_hat) * rescaling
    assert np.abs(_u(repaired)[mask] - moment_exact).max() > 1e-3


def test_regression_misfits_are_computed_after_the_repair():
    """A failed particle would otherwise poison every statistic and criterion."""
    problem = _AffineProblem(J=8)
    y, noise = jnp.asarray(problem.y), problem.noise_cov

    def failing(u):
        v = jnp.asarray(u) @ jnp.asarray(problem.G).T
        return v.at[1, 0].set(jnp.nan)

    evaluation = evaluate(problem.state(), failing, y, noise, on_failure="repair")
    assert np.all(np.isfinite(np.asarray(evaluation.misfits)))
    assert int(evaluation.n_valid) == problem.J - 1

    mask = np.ones(problem.J, dtype=bool)
    mask[1] = False
    valid_predictions = (problem.members @ problem.G.T)[mask]
    g_hat = valid_predictions.mean(axis=0)
    W = _recovered_whitener(noise, problem.N)
    want_repaired = 0.5 * float(np.sum((W @ (problem.y - g_hat)) ** 2))
    assert float(evaluation.misfits[1]) == pytest.approx(want_repaired, rel=1e-10)
    valid_misfits = 0.5 * np.sum(((problem.y - valid_predictions) @ W.T) ** 2, axis=1)
    assert want_repaired < valid_misfits.mean()


def test_regression_a_schedule_that_counts_its_own_calls_is_caught():
    """Purity is what makes a run resumable, and the check is what catches it."""

    class _CountsItsCalls:
        n_steps, beta_target = None, 1.0

        def __init__(self):
            self.seen = 0

        def next_increment(self, evaluation):
            self.seen += 1
            return 0.1 * self.seen

    with pytest.raises(AssertionError, match="not pure"):
        check_schedule(_CountsItsCalls())


def test_regression_turning_inflation_on_does_not_shift_the_update_stream():
    """No numeric test of the default rule can catch a changed split.

    ``SymmetricSquareRoot`` consumes no randomness, so the guard is a
    ``Matheron`` run with and without an identity inflation, and a snapshot of
    the state's key after three steps.
    """
    problem = _AffineProblem(P=2, N=2, J=4, seed=61)
    common = dict(schedule=FixedSchedule.uniform(3), update_rule=MATHERON)
    without = _run(problem, state=problem.state(seed=2), **common)
    with_inflation = _run(
        problem, state=problem.state(seed=2),
        inflation=lambda key, *, ensemble, **_: ensemble, **common,
    )
    assert np.array_equal(_u(without), _u(with_inflation))

    key = problem.state(seed=2).key
    for _ in range(3):
        key = _keys(key)[0]
    assert np.array_equal(
        np.asarray(jax.random.key_data(without.state.key)),
        np.asarray(jax.random.key_data(key)),
    )
    key_data = np.asarray(jax.random.key_data(without.state.key))
    assert [int(x) for x in key_data] == _KEY_SNAPSHOT


_KEY_SNAPSHOT = [916975276, 3797780651]


def test_regression_a_fill_value_model_stalls_an_adaptive_ladder_silently():
    """Finite nonsense is invisible to the layer, and the ladder crawls."""
    problem = _AffineProblem()
    schedule = AdaptiveMisfitSchedule(beta_target=0.05, min_increment=1e-3)
    clean = _run(problem, schedule=schedule, max_steps=50)
    assert clean.n_evaluations == 1

    def with_fill_value(u):
        v = jnp.asarray(u) @ jnp.asarray(problem.G).T
        return v.at[0].set(-9999.0)

    stalled = _run(problem, forward=with_fill_value, schedule=schedule, max_steps=50)
    assert stalled.min_n_valid == problem.J
    assert stalled.n_evaluations == 50
    assert np.all(np.asarray(stalled.stacked.increment) == pytest.approx(1e-3))


def test_regression_a_systematically_failing_particle_is_visible_only_in_n_valid():
    problem = _AffineProblem(J=8)

    def always_fails_particle_three(u):
        v = jnp.asarray(u) @ jnp.asarray(problem.G).T
        return v.at[3, 0].set(jnp.nan)

    with pytest.warns(UserWarning, match="not finite and were repaired"):
        result = _run(problem, forward=always_fails_particle_three,
                      schedule=FixedSchedule.uniform(4), on_failure="repair")
    assert result.status == SCHEDULE_EXHAUSTED
    assert np.all(np.isfinite(_u(result)))
    assert result.min_n_valid == problem.J - 1
    assert list(np.asarray(result.stacked.n_valid)) == [problem.J - 1] * 4


def test_regression_a_failing_step_logs_at_warning_level(caplog):
    problem = _AffineProblem(J=8)

    def failing(u):
        v = jnp.asarray(u) @ jnp.asarray(problem.G).T
        return v.at[3, 0].set(jnp.nan)

    with caplog.at_level(logging.WARNING, logger="enskit.algorithms.eki"), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _run(problem, forward=failing, schedule=FixedSchedule.uniform(2),
             on_failure="repair")
    assert "were finite" in caplog.text


def test_regression_a_float32_update_cannot_quietly_demote_a_run():
    """Every downstream test would still pass at its own tolerance."""
    problem = _AffineProblem()

    class _Demoting:
        def build(self, particles, approximation, given):
            built = SQRT.build(particles, approximation, given)

            def call(values=None, /, *, key=None, **kw):
                out = built(values, key=key, **kw)
                return Ensemble({n: out[n].astype(jnp.float32) for n in out.names})

            return call

    with pytest.raises(TypeError, match="float32"):
        _run(problem, schedule=FixedSchedule.uniform(2), update_rule=_Demoting())

    def coarse(u):
        return (jnp.asarray(u) @ jnp.asarray(problem.G).T).astype(jnp.float32)

    with pytest.warns(UserWarning, match="promoted"):
        result = _run(problem, forward=coarse, schedule=FixedSchedule.uniform(2))
    assert result.ensemble["u"].dtype == jnp.float64


def test_regression_a_collapsed_ensemble_has_exactly_zero_spread_at_any_magnitude():
    """Identical particles give exactly zero anomalies, not round-off of the mean."""
    for magnitude in (1.0, 6e23):
        collapsed = jnp.full((7, 3), magnitude)
        naive = np.asarray(collapsed) - np.asarray(collapsed).mean(axis=0)
        if magnitude > 1.0:
            assert np.abs(naive).max() > 0.0, "the naive form is not exactly zero"
        assert _spread_of({"u": collapsed}) == 0.0


def test_regression_a_vmapped_policy_refuses_rather_than_broadcasting():
    """A family whose batch size equals P would otherwise inflate per coordinate."""
    family = jax.vmap(MultiplicativeInflation)(jnp.asarray([1.0, 2.0, 3.0]))
    assert family.batch_shape == (3,)
    with pytest.raises(ValueError, match="vmapped family"):
        family(jax.random.key(0), ensemble=Ensemble(u=jnp.arange(18.0).reshape(6, 3)))
    relax = jax.vmap(RelaxToPriorSpread)(jnp.asarray([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError, match="vmapped family"):
        zeros = Ensemble(u=jnp.zeros((6, 3)))
        relax(prior=zeros, posterior=zeros)


def test_regression_a_nan_misfit_does_not_become_the_floor_step():
    """Both adaptive schedules propagate a nan, and the driver then refuses it."""
    poisoned = _evaluation_with_misfits(np.array([1.0, np.nan, 2.0, 3.0]))
    for schedule in (AdaptiveESSSchedule(), AdaptiveMisfitSchedule()):
        got = float(schedule.next_increment(poisoned))
        assert np.isnan(got), f"{schedule!r} returned {got} on a nan misfit"

    problem = _AffineProblem()
    with pytest.raises(EKIError, match="At least 2 are required"):
        _run(problem, forward=lambda u: jnp.full((problem.J, problem.N), jnp.inf),
             schedule=AdaptiveESSSchedule())
