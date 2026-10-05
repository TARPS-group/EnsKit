"""The toy problems: their models, their closed form, and their failures.

Four kinds of test, in the order the module's claims are made:

1. **Contract conformance** — every model returns the documented shape at the
   documented input shape, in the run's dtype, is ``jit``-able and
   ``vmap``-pable, and is row-independent and deterministic. That every model
   passes :func:`enskit.testing.check_simulator`, and the checker's own
   negative tests, live in ``test_maps.py`` beside the simulator contract.
2. **Exactness** — the linear problem's closed form against a dense reference
   written here, at three tempering levels; and a full run from an
   exact-moment ensemble reaching that closed form to floating point.
3. **The properties the documentation depends on** — the ladder's advantage
   over a single unit step, the failure fraction's determinism, and the
   subspace confinement at :math:`P \\gg J`.
4. **The user guide's runnable blocks**, with their printed numbers pinned.

These tests do not replace the local forward models in ``test_eki.py``: those
are instrumented — one records every argument it is given — and their
references are written locally on purpose, which is what makes them regression
tests for the layer rather than for this module.
"""
from __future__ import annotations

import dataclasses
import math
import subprocess
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import prints_as

import enskit  # noqa: F401  -- enables x64 before any array exists
from enskit import kalman, maps, toy
from enskit.algorithms import eki
from enskit.distribution import Ensemble, Gaussian
from enskit.linalg import DensePSD, PSDDiagonal, PSDLowRank, UnsupportedOpError
from enskit.testing import check_simulator

EPS = float(np.finfo(np.float64).eps)

#: One instance of each model, with the sizes it should answer at.
PROBLEMS = [
    ("linear_gaussian", toy.linear_gaussian(), 4, 8),
    ("exponential_decay", toy.exponential_decay(), 2, 12),
    ("restricted_decay", toy.restricted_decay(), 2, 12),
]
IDS = [name for name, *_ in PROBLEMS]


def _ensemble(n_particles: int, parameter_dim: int, seed: int = 0):
    """A pseudo-random ``(J, P)`` array of parameters, drawn outside the models."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(n_particles, parameter_dim)))


def _identical(got, want) -> bool:
    """Bit-identity, counting two ``nan`` s as equal: failed rows are legal."""
    return np.array_equal(np.asarray(got), np.asarray(want), equal_nan=True)


def _exact_moment_ensemble(J: int, mu: np.ndarray, F: np.ndarray) -> np.ndarray:
    """An ensemble whose empirical moments are exactly ``mu`` and ``F @ F.T``.

    The QR-of-ones construction: the complete QR of the all-ones vector in
    R^J gives columns that are orthonormal and orthogonal to it, and
    ``mu + sqrt(J - 1) E F.T`` then has mean ``mu`` and empirical covariance
    ``F F.T`` under the package's J - 1 divisor. Only J >= k + 1 binds.
    Written out here rather than imported, as the package's conformance rules
    require of a reference.
    """
    k = F.shape[1]
    assert J >= k + 1, "the construction needs J >= k + 1"
    Q, _ = np.linalg.qr(np.ones((J, 1)), mode="complete")
    return mu + np.sqrt(J - 1) * Q[:, 1 : k + 1] @ F.T


# ===========================================================================
# 1. every model against the forward-model contract
# ===========================================================================


@pytest.mark.parametrize("name, problem, parameter_dim, data_dim", PROBLEMS, ids=IDS)
def test_1_every_model_returns_the_documented_shape_and_dtype(
    name, problem, parameter_dim, data_dim
):
    """(J, P) in, (J, N) out, in the run's working dtype, at three sizes of J.

    A model is independent of the ensemble size, so one instance answers at
    every J. The dtype matters because a narrower return is promoted with a
    warning rather than rejected, so no other test would catch a float32
    model here.
    """
    assert (problem.parameter_dim, problem.data_dim) == (parameter_dim, data_dim)
    for n_particles in (2, 5, 64):
        predictions = problem.forward(_ensemble(n_particles, parameter_dim))
        assert predictions.shape == (n_particles, data_dim)
        assert predictions.dtype == jnp.float64
    assert jnp.shape(problem.y) == (data_dim,)
    assert jnp.shape(problem.u_true) == (parameter_dim,)
    assert problem.prior.dims == {"u": parameter_dim}
    assert problem.noise_cov.shape == (data_dim, data_dim)


@pytest.mark.parametrize("name, problem, parameter_dim, data_dim", PROBLEMS, ids=IDS)
def test_1_no_problem_is_callable(name, problem, parameter_dim, data_dim):
    """A problem is not a forward model, and must not be mistakable for one.

    ``run`` takes the callable, the observation and the noise covariance as
    three arguments, and the EKI contract excludes a container accepted in
    their place. A ``__call__`` here would make the container look like the
    interface, and a reader's own model would then be a class they pass to
    ``run``, which does not work.
    """
    assert not callable(problem), (
        f"{name} became callable; pass problem.forward, problem.y and "
        f"problem.noise_cov as three arguments"
    )


@pytest.mark.parametrize("name, problem, parameter_dim, data_dim", PROBLEMS, ids=IDS)
def test_2_every_model_is_jittable_and_vmappable(name, problem, parameter_dim, data_dim):
    """A convenience of the toy models, so it is tested as one.

    Not a property of forward models in general — the contract requires
    neither — but these are used across the suite and the documentation, so a
    change that broke tracing would be one to notice.
    """
    ensemble = _ensemble(6, parameter_dim)
    eager = problem.forward(ensemble)
    assert _identical(jax.jit(problem.forward)(ensemble), eager)

    stacked = jnp.stack([ensemble, ensemble + 0.25])
    mapped = jax.vmap(problem.forward)(stacked)
    assert mapped.shape == (2, 6, data_dim)
    assert _identical(mapped[0], eager)
    assert _identical(mapped[1], problem.forward(ensemble + 0.25))


@pytest.mark.parametrize("name, problem, parameter_dim, data_dim", PROBLEMS, ids=IDS)
def test_3_every_model_is_row_independent_and_deterministic(
    name, problem, parameter_dim, data_dim
):
    """Row j of the return depends only on row j of the argument.

    The one requirement beyond the shapes that nothing inside a run detects.
    Checked by permuting the particles — for a row-independent model the
    predictions permute with them, bit-exactly, since the rows are the same
    set of particles either way — and by re-evaluating a subset of them, which
    catches the symmetric couplings a permutation cannot. The subset
    comparison is to a tolerance: a differently shaped batch legitimately
    takes a different matmul kernel and rounds differently in the last bits.
    """
    ensemble = _ensemble(7, parameter_dim)
    predictions = problem.forward(ensemble)
    permutation = jnp.array([3, 1, 0, 6, 5, 4, 2])
    assert _identical(
        problem.forward(ensemble[permutation, :]), predictions[permutation, :]
    )
    for subset in (jnp.array([0, 4]), jnp.array([2, 3, 6])):
        np.testing.assert_allclose(
            problem.forward(ensemble[subset, :]),
            predictions[subset, :],
            rtol=0,
            atol=1e-12,
        )
    assert _identical(problem.forward(ensemble), predictions)


# The five `test_4_*` tests of `enskit.eki.testing.check_forward_model` -- every
# toy model passing it, and the checker's own negative cases -- moved to
# `tests/test_maps.py` with the checker, now `enskit.testing.check_simulator`.
# The maps contract's "Ported regression tests" table maps each old name to
# its new one.


# ===========================================================================
# 2. exactness, for the linear problem
# ===========================================================================


def _dense_posterior(problem, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """The precision-form posterior, in plain dense NumPy.

    Written here rather than routed through any package code, so that the
    comparison is between two independent paths. Valid only for an invertible
    prior covariance, which is what these tests build.
    """
    C0 = np.asarray(problem.prior.cov("u").to_dense())
    G = np.asarray(problem.G.to_dense())
    R = np.asarray(problem.noise_cov.to_dense())
    prior_precision = np.linalg.inv(C0)
    data_precision = G.T @ np.linalg.solve(R, G)
    cov = np.linalg.inv(prior_precision + beta * data_precision)
    mean = cov @ (
        prior_precision @ np.asarray(problem.prior.mean("u"))
        + beta * G.T @ np.linalg.solve(R, np.asarray(problem.y))
    )
    return mean, cov


def _general_linear_problem():
    """A `LinearGaussian` with a non-zero prior mean and correlated noise.

    Every problem `linear_gaussian` builds has prior mean exactly zero and
    diagonal noise, which makes two of the three terms of the conditioning
    mean invisible: dropping the prior-mean term, or the ``G m_0`` residual
    term, changes nothing that the shipped fixture can see. This fixture
    restores both, so the exactness test below covers the whole formula.
    """
    base = toy.linear_gaussian(parameter_dim=4, data_dim=8, seed=5)
    rng = np.random.default_rng(11)
    factor = np.tril(rng.normal(size=(4, 4))) + 4.0 * np.eye(4)
    noise = rng.normal(size=(8, 8))
    return dataclasses.replace(
        base,
        prior=Gaussian(
            {"u": jnp.asarray(rng.normal(size=4))},
            block_covs={"u": DensePSD(jnp.asarray(factor @ factor.T))},
        ),
        noise_cov=DensePSD(
            jnp.asarray(noise @ noise.T / 8.0 + np.eye(8))
        ),
    )


@pytest.mark.parametrize("beta", [0.25, 1.0, 2.0])
def test_5_the_closed_form_posterior_covers_the_whole_conditioning_formula(beta):
    """The same check on a problem whose prior mean and noise are general.

    The tolerance is `1e3 * EPS * scale` as in the sibling test, and it is
    not portable to arbitrary sizes: at `parameter_dim > data_dim` and level 2
    the *reference* loses accuracy, since it inverts the precision matrix. Widen
    the parametrization and this tolerance must be revisited.
    """
    problem = _general_linear_problem()
    posterior = problem.posterior(beta)
    mean_ref, cov_ref = _dense_posterior(problem, beta)
    scale = max(np.abs(mean_ref).max(), np.abs(cov_ref).max())

    assert np.abs(np.asarray(problem.prior.mean("u"))).min() > 0.1, (
        "fixture is vacuous"
    )
    np.testing.assert_allclose(
        posterior.mean("u"), mean_ref, rtol=0, atol=1e3 * EPS * scale
    )
    np.testing.assert_allclose(
        posterior.cov("u").to_dense(), cov_ref, rtol=0, atol=1e3 * EPS * scale
    )


def test_5_the_prior_mean_and_the_residual_term_are_both_load_bearing():
    """Both terms the shipped fixture cannot see, shown to matter.

    Without this, `test_5` would pass an implementation that conditioned on
    `y` alone, ignoring the prior mean and the `G m_0` residual — because
    every `linear_gaussian` problem has prior mean zero.
    """
    problem = _general_linear_problem()
    zero_mean = dataclasses.replace(
        problem,
        prior=Gaussian(
            {"u": jnp.zeros(problem.parameter_dim)},
            block_covs={"u": problem.prior.block_cov("u")},
        ),
    )
    difference = np.abs(
        np.asarray(problem.posterior().mean("u"))
        - np.asarray(zero_mean.posterior().mean("u"))
    ).max()
    assert difference > 0.05, difference


@pytest.mark.parametrize("beta", [0.25, 1.0, 2.0])
def test_5_the_closed_form_posterior_matches_a_dense_reference(beta):
    """`posterior(beta)` is the posterior at noise R / beta, exactly.

    The distribution layer's own conditioning is already checked against a
    dense reference in its own tests. What this pins is the toy
    model's composition: that it divides the *noise* by beta, and that it
    divides it at all. The mis-scaling this guards is the layer's signature
    silent bug; the next test measures how far off it would be.
    """
    problem = toy.linear_gaussian()
    posterior = problem.posterior(beta)
    mean_ref, cov_ref = _dense_posterior(problem, beta)
    scale = max(np.abs(mean_ref).max(), np.abs(cov_ref).max())

    P = problem.parameter_dim
    assert isinstance(posterior, Gaussian)
    assert posterior.names == ("u",)
    # The (P, k) factor the docstring promises, and no independent term: the
    # prior's diagonal term is absorbed into the factor by the pushforward.
    assert posterior.factor("u").shape == (P, P)
    assert posterior.block_cov("u") is None
    assert isinstance(posterior.cov("u"), PSDLowRank)
    np.testing.assert_allclose(
        posterior.mean("u"), mean_ref, rtol=0, atol=1e3 * EPS * scale
    )
    np.testing.assert_allclose(
        posterior.cov("u").to_dense(), cov_ref, rtol=0, atol=1e3 * EPS * scale
    )


def test_5_beta_is_load_bearing_and_the_tolerance_would_catch_it():
    """The two betas must be far apart at the tolerance test 5 uses.

    Without this, test 5 would pass an implementation that ignored ``beta``
    altogether, since its default agrees.
    """
    problem = toy.linear_gaussian()
    gap = np.abs(
        np.asarray(problem.posterior(0.5).mean("u"))
        - np.asarray(problem.posterior(1.0).mean("u"))
    ).max()
    # Test 5 compares at roughly 1e3 * EPS * scale, about 2e-13 here, so a
    # gap of 0.01 is ten orders of magnitude above what it would accept.
    assert gap > 0.01, "the two betas are indistinguishable, so test 5 has no teeth"


@pytest.mark.parametrize(
    "increments", [(1.0,), (0.5, 0.5), (0.25,) * 4, (0.5, 0.25, 0.25)]
)
def test_6_a_ladder_from_an_exact_moment_ensemble_reaches_the_closed_form(
    increments,
):
    """A run converges on the toy model's own closed form, to floating point.

    ``test_eki.py``'s telescoping test pins the *layer* against a reference
    written there. This pins that ``LinearGaussian.posterior`` is the same
    posterior a run of ``LinearGaussian.forward`` reaches — the claim every
    comparison in the documentation rests on, and the one that would break if
    the closed form were built from a different problem than the model.
    Not redundant with that test; do not delete it as such.
    """
    problem = toy.linear_gaussian()
    J = 12
    factor = np.asarray(problem.prior.cov("u").factor().to_dense())
    particles = jnp.asarray(
        _exact_moment_ensemble(J, np.asarray(problem.prior.mean("u")), factor)
    )
    state = eki.EKIState(Ensemble(u=particles), key=jax.random.key(0))

    result = eki.run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        update_rule=kalman.SymmetricSquareRoot(),
        schedule=eki.FixedSchedule(increments),
        max_steps=len(increments),
    )

    closed = problem.posterior()
    ensemble = np.asarray(result.ensemble["u"])
    anomalies = ensemble - ensemble.mean(axis=0)
    closed_mean = np.asarray(closed.mean("u"))
    closed_cov = np.asarray(closed.cov("u").to_dense())
    scale = max(np.abs(closed_mean).max(), np.abs(closed_cov).max())
    np.testing.assert_allclose(
        ensemble.mean(axis=0),
        closed_mean,
        rtol=0,
        atol=1e3 * EPS * scale,
    )
    np.testing.assert_allclose(
        anomalies.T @ anomalies / (J - 1),
        closed_cov,
        rtol=0,
        atol=1e3 * EPS * scale,
    )


def test_6_the_posterior_size_guard_raises_before_allocating():
    """A P-by-k factor above the budget is refused, naming both sizes.

    The run has no such limit, so the guard must not read as one on the
    problem: a high-dimensional problem is invertible where its closed form
    cannot be written down.
    """
    big = toy.linear_gaussian(parameter_dim=5000, data_dim=10)
    with pytest.raises(ValueError, match=r"5000-by-5000"):
        big.posterior()
    assert big.forward(_ensemble(3, 5000)).shape == (3, 10)


@pytest.mark.parametrize("beta", [0.0, -1.0, float("inf"), float("nan")])
def test_6_a_non_positive_beta_raises(beta):
    with pytest.raises(ValueError, match="positive and finite"):
        toy.linear_gaussian().posterior(beta)


# ===========================================================================
# 3. the properties the documentation depends on
# ===========================================================================


@pytest.mark.parametrize("seed", range(8))
def test_7_the_ladder_beats_a_single_unit_step_on_the_decay_problem(seed):
    """The property tutorials 2 and 3 and notebook 02 are built on.

    Asserted over **every** observation seed the factory's docstring claims,
    not only the default: the claim is that the two answers differ reliably
    rather than coincidentally, and a single-seed test cannot distinguish
    those. Measured over seeds 0 to 7 with the symmetric square-root update:
    the rate gap is 0.088 to 0.354 and the ladder is nearer u_true by a
    factor of 2.70 to 26.8, so the thresholds below carry margins of 1.10x,
    1.35x and 1.65x on the worst seed.

    An earlier version of this test asserted `ladder_error < one_step_error /
    3` and `< 0.05` at seed 0 alone, where they hold with room to spare; seeds
    4 and 7 respectively break both. The docstring claiming eight-seed
    robustness was therefore false, and this is what makes it true.
    """
    problem = toy.exponential_decay(seed=seed)
    state = eki.EKIState.from_prior(jax.random.key(0), problem.prior, n_particles=64)
    common = (problem.forward, problem.y, problem.noise_cov)
    rule = kalman.SymmetricSquareRoot()

    one_step = eki.run(
        state, *common, update_rule=rule, schedule=eki.FixedSchedule((1.0,))
    )
    ladder = eki.run(
        state, *common, update_rule=rule, schedule=eki.AdaptiveESSSchedule()
    )

    u_true = np.asarray(problem.u_true)
    one_step_mean = np.asarray(one_step.mean("u"))
    ladder_mean = np.asarray(ladder.mean("u"))
    one_step_error = np.abs(one_step_mean - u_true).max()
    ladder_error = np.abs(ladder_mean - u_true).max()
    gap = np.abs(one_step_mean - ladder_mean).max()

    assert ladder.n_completed_steps > 1
    assert gap > 0.08, gap
    assert ladder_error < one_step_error / 2, (ladder_error, one_step_error)
    assert ladder_error < 0.08, ladder_error


def test_8_the_restricted_model_fails_exactly_its_out_of_domain_particles():
    """The failure is a deterministic function of the parameters, not a rate.

    Which particles fail is decided by the rate alone, and the whole row goes
    non-finite when it does — a partially finite row would be read as a valid
    particle with a huge misfit, which is the failure mode that stalls an
    adaptive ladder instead of flagging itself.
    """
    problem = toy.restricted_decay()
    ensemble = _ensemble(64, 2)
    predictions = np.asarray(problem.forward(ensemble))
    finite_rows = np.isfinite(predictions).all(axis=1)
    expected = np.asarray(ensemble)[:, 1] > 0.0

    assert np.array_equal(finite_rows, expected)
    assert np.array_equal(np.isfinite(predictions).any(axis=1), expected), (
        "a partially finite row would count as a valid particle"
    )
    assert 0 < (~finite_rows).sum() < 64, "the fixture must have both kinds"


def test_8_the_failure_fraction_is_monotone_in_the_rate_floor():
    """The knob a notebook sweeps, and the direction it moves in."""
    ensemble = _ensemble(64, 2)
    counts = []
    for rate_floor in (-1.0, -0.5, 0.0, 0.5, 1.0):
        problem = toy.restricted_decay(rate_floor=rate_floor)
        predictions = np.asarray(problem.forward(ensemble))
        counts.append(int((~np.isfinite(predictions).all(axis=1)).sum()))
    assert counts == sorted(counts) and counts[0] < counts[-1], counts
    # Only the floor moves: everything else is the exponential_decay problem.
    plain = toy.exponential_decay()
    restricted = toy.restricted_decay()
    for field in ("times", "y", "u_true"):
        assert _identical(getattr(plain, field), getattr(restricted, field))
    # "the same problem in every other respect" includes these two.
    assert _identical(plain.prior.mean("u"), restricted.prior.mean("u"))
    assert _identical(plain.prior.cov("u").diag(), restricted.prior.cov("u").diag())
    assert _identical(plain.noise_cov.diag(), restricted.noise_cov.diag())


def test_8_a_run_against_the_restricted_model_repairs_and_reports():
    """Raising is the default; with repair, a failed run is still a completed one.

    The numbers are pinned because the user guide prints them.
    """
    problem = toy.restricted_decay()
    state = eki.EKIState.from_prior(jax.random.key(0), problem.prior, n_particles=64)
    common = (problem.forward, problem.y, problem.noise_cov)
    options = dict(
        update_rule=kalman.SymmetricSquareRoot(), schedule=eki.AdaptiveESSSchedule()
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = eki.run(state, *common, on_failure="repair", **options)

    assert result.min_n_valid == 51
    assert np.array_equal(np.asarray(result.stacked.n_valid), [51, 63, 64, 64, 64])
    assert [w for w in caught if "were not finite and were repaired" in str(w.message)]
    mean = np.asarray(result.mean("u"))
    assert np.abs(mean - np.asarray(problem.u_true)).max() < 0.05

    prints_as(mean, [1.9796, 1.4739])  # the page prints this too
    with pytest.raises(eki.EKIError, match="finite"):
        eki.run(state, *common, **options)


def test_9_the_high_dimensional_problem_is_confined_and_over_confident():
    """The subspace bound, against the closed form rather than against nothing.

    Every iterate lies in the affine span of the initial ensemble, of
    dimension at most J - 1 — so at P = 2000 with J = 40 the run reports a
    spread that the exact posterior contradicts by a factor of three, with
    nothing raised and no history field flagging it. Both halves are the
    lesson; the closed form is what makes the second half sayable.

    The factor is a property of the problem rather than of this draw: over
    initial-ensemble keys 0 to 19 it is between 3.1 and 3.9 (3.28 at key 0),
    which the threshold below holds with a margin of 1.24x on the worst key.
    It was once pinned at seventy, which was an artifact: the initial
    ensemble's key aliased the key that drew `G`, so the particles were `G`'s
    rows and spanned exactly the observed directions. The pinned spread
    catches a return of that aliasing, which would move it to 0.014.
    """
    problem = toy.linear_gaussian(parameter_dim=2000, data_dim=40)
    state = eki.EKIState.from_prior(jax.random.key(0), problem.prior, n_particles=40)
    result = eki.run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        update_rule=kalman.SymmetricSquareRoot(),
        schedule=eki.AdaptiveESSSchedule(),
    )

    initial = np.asarray(state.ensemble["u"])
    displacement = np.asarray(result.ensemble["u"]) - initial.mean(axis=0)
    # The bound, not the realized value. The rank is 39 and the run takes 30
    # steps today, but pinning either makes an intentional schedule change
    # look like a regression, and tutorial 7 hedges them for that reason.
    assert np.linalg.matrix_rank(displacement) <= state.n_particles - 1
    assert result.status == eki.SCHEDULE_EXHAUSTED

    ensemble_sd = float(result.ensemble["u"].std(axis=0, ddof=1).mean())
    exact_sd = float((problem.posterior().cov("u").diag() ** 0.5).mean())
    assert exact_sd / ensemble_sd > 2.5, (ensemble_sd, exact_sd)
    # The numbers the user guide prints, to the precision it prints them.
    prints_as(ensemble_sd, 0.3022)
    prints_as(exact_sd, 0.9900)


# ===========================================================================
# 4. construction, reproducibility, and the guide's blocks
# ===========================================================================


def test_10_a_problem_is_reproducible_from_its_seed_and_varies_with_it():
    """A docs build must see the same numbers every time it runs."""
    for factory in (toy.linear_gaussian, toy.exponential_decay, toy.restricted_decay):
        first, again = factory(seed=3), factory(seed=3)
        different = factory(seed=4)
        assert _identical(first.y, again.y)
        assert _identical(first.u_true, again.u_true)
        assert not _identical(first.y, different.y)


def test_10_the_factories_and_classes_validate_as_documented():
    problem = toy.linear_gaussian()
    with pytest.raises(ValueError, match="at least 1"):
        toy.linear_gaussian(parameter_dim=0)
    with pytest.raises(ValueError, match="must be positive"):
        toy.linear_gaussian(noise_std=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        toy.exponential_decay(t_max=-1.0)
    with pytest.raises(ValueError, match="at least 1"):
        toy.restricted_decay(n_times=0)
    with pytest.raises(ValueError, match="outside the valid domain"):
        toy.restricted_decay(rate_floor=2.0)  # above the true rate of 1.5

    with pytest.raises(TypeError, match="must be an enskit.linalg.LinOp"):
        dataclasses.replace(problem, G=np.zeros((8, 4)))
    with pytest.raises(ValueError, match=r"LinearGaussian.y: expected an array"):
        dataclasses.replace(problem, y=jnp.zeros(3))
    with pytest.raises(ValueError, match="the prior has dimension"):
        dataclasses.replace(problem, prior=toy.linear_gaussian(parameter_dim=5).prior)


def test_11_the_forward_model_pages_checker_block_runs():
    """The `check_simulator` block of the forward-model guide.

    Its ``forward`` is that page's own opening example, so the call has to
    hold at the sizes the page states.
    """
    times = jnp.array([0.5, 1.0, 2.0])

    def forward(ensemble):  # (J, 2) in
        return ensemble[:, 0:1] * jnp.exp(-ensemble[:, 1:2] * times)  # (J, 3) out

    check_simulator(forward, 2, 3)


def test_11_the_toy_models_page_blocks_run():
    """Every runnable block of the user guide's toy-models page, in order.

    Its printed values are pinned here, so the page cannot rot unnoticed.
    Conformance obligation 26 of the EKI contract is the rule; the
    forward-model page's own example is the pattern.
    """
    problem = toy.linear_gaussian(parameter_dim=4, data_dim=8)
    state = eki.EKIState.from_prior(jax.random.key(0), problem.prior, n_particles=32)
    result = eki.run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        update_rule=kalman.SymmetricSquareRoot(),
        schedule=eki.AdaptiveESSSchedule(),
    )
    exact = problem.posterior()
    fitted = result.ensemble.project()

    prints_as(result.mean("u"), [-1.3258, 1.3698, 0.6372, 0.133])
    prints_as(exact.mean("u"), [-1.3303, 1.3749, 0.6281, 0.1409])
    prints_as(problem.u_true, [-1.4009, 1.4321, 0.6248, 0.2005])
    # The page says "within 0.01"; measured 0.00909, so the threshold is
    # stated loosely enough that a change of a few percent does not fail it.
    assert np.abs(np.asarray(result.mean("u") - exact.mean("u"))).max() < 0.01
    fitted_sd = fitted.cov("u").diag() ** 0.5
    exact_sd = exact.cov("u").diag() ** 0.5
    prints_as(fitted_sd, [0.0686, 0.1273, 0.1168, 0.0876])
    prints_as(exact_sd, [0.0687, 0.1276, 0.1165, 0.0876])
    # The page says "within 0.0004"; measured 0.00034.
    assert np.abs(np.asarray(fitted_sd - exact_sd)).max() < 0.0004

    # The three operations the page says `posterior()` is, as it writes them.
    beta = 1.0
    joint = problem.prior.pipe(
        maps.pushforward, maps.Linear(problem.G), inputs="u", output="g"
    )
    by_hand = joint.add_noise(g=problem.noise_cov / beta).condition(g=problem.y)
    assert _identical(by_hand.mean("u"), exact.mean("u"))
    assert _identical(by_hand.cov("u").to_dense(), exact.cov("u").to_dense())

    # The block that checks a model of your own; `forward` stands in for the
    # page's `my_forward`, at sizes cheap enough for a test.
    check_simulator(problem.forward, 4, 8)

    # The correlated-noise variant, whose R the page names.
    rng = np.random.default_rng(1)
    M = rng.normal(size=(8, 8))
    R = jnp.asarray(M @ M.T / 8 + 0.01 * np.eye(8))
    correlated = dataclasses.replace(problem, noise_cov=DensePSD(R))
    prints_as(correlated.posterior().mean("u"), [-1.4093, 0.8248, 0.4646, 0.0692])


def _dense_posterior_from_blocks(problem, beta: float):
    """The block form of the posterior, which never inverts the prior.

    ``_dense_posterior`` inverts ``C0``, so it cannot check the claim that a
    *singular* prior covariance is fine — the case a precision-form posterior
    cannot express at all. This reference uses only the observation-side
    solve, so it holds at any prior rank.
    """
    C0 = np.asarray(problem.prior.cov("u").to_dense())
    G = np.asarray(problem.G.to_dense())
    R = np.asarray(problem.noise_cov.to_dense()) / beta
    m0 = np.asarray(problem.prior.mean("u"))
    cross = C0 @ G.T
    solved = np.linalg.solve(
        G @ cross + R, np.column_stack([np.asarray(problem.y) - G @ m0, cross.T])
    )
    return m0 + cross @ solved[:, 0], C0 - cross @ solved[:, 1:]


@pytest.mark.parametrize("beta", [0.5, 1.0])
def test_5_a_singular_prior_covariance_works_and_narrows_the_factor(beta):
    """The one claim the closed form makes that the test file's own reference
    cannot check, so it gets a reference that can.

    A rank-2 prior in four dimensions: the posterior factor is ``(4, 2)``, not
    ``(4, 4)``, and the answer matches the block form of the conditioning
    identity. Every problem `linear_gaussian` builds has a full-rank diagonal
    prior, so without this `k` and `P` are the same number everywhere and
    `latent_dim = self.parameter_dim` would pass the whole suite.
    """
    base = toy.linear_gaussian(parameter_dim=4, data_dim=6, seed=2)
    factor = jnp.asarray(np.random.default_rng(4).normal(size=(4, 2)))
    problem = dataclasses.replace(
        base,
        prior=Gaussian(
            {"u": jnp.asarray([0.5, -1.0, 0.25, 2.0])},
            block_covs={"u": PSDLowRank(factor)},
        ),
    )

    posterior = problem.posterior(beta)
    assert posterior.factor("u").shape == (4, 2), (
        "the factor must narrow with the prior"
    )
    mean_ref, cov_ref = _dense_posterior_from_blocks(problem, beta)
    scale = max(np.abs(mean_ref).max(), np.abs(cov_ref).max())
    np.testing.assert_allclose(
        posterior.mean("u"), mean_ref, rtol=0, atol=1e3 * EPS * scale
    )
    np.testing.assert_allclose(
        posterior.cov("u").to_dense(), cov_ref, rtol=0, atol=1e3 * EPS * scale
    )


def test_5_the_two_standard_deviations_reach_the_arrays_they_document():
    """`prior_std` and `noise_std` have to do what their names say.

    Nothing else in the suite calls either factory at a non-default scale, so
    hardcoding the defaults inside `linear_gaussian` passed every test.
    """
    problem = toy.linear_gaussian(
        parameter_dim=3, data_dim=5, prior_std=4.0, noise_std=0.25
    )
    np.testing.assert_allclose(problem.prior.cov("u").diag(), np.full(3, 16.0))
    np.testing.assert_allclose(problem.noise_cov.diag(), np.full(5, 0.0625))
    decay = toy.exponential_decay(noise_std=0.5)
    np.testing.assert_allclose(decay.noise_cov.diag(), np.full(12, 0.25))


def test_5_the_prior_and_the_noise_must_support_what_the_posterior_needs():
    """The two documented `UnsupportedOpError` paths."""
    problem = toy.linear_gaussian(parameter_dim=4, data_dim=6)
    # PSDLowRank withholds `whiten`, which conditioning needs of the noise.
    low_rank = PSDLowRank(jnp.asarray(np.random.default_rng(0).normal(size=(6, 2))))
    with pytest.raises(UnsupportedOpError):
        dataclasses.replace(problem, noise_cov=low_rank).posterior()


def test_6_the_decay_problems_answer_at_a_size_they_were_not_built_at():
    """`n_times` and `t_max` reach `times`, and the grid is half-open.

    A constant `data_dim`, and a grid starting at 0 rather than at
    `t_max / n_times`, both passed the suite: the second is a materially
    different problem, since at `t = 0` the prediction is the amplitude
    exactly and carries no information about the rate.
    """
    problem = toy.exponential_decay(n_times=5, t_max=10.0)
    assert problem.data_dim == 5
    np.testing.assert_allclose(problem.times, [2.0, 4.0, 6.0, 8.0, 10.0])
    assert problem.forward(_ensemble(3, 2)).shape == (3, 5)
    assert float(problem.times[0]) > 0.0, "t = 0 carries no rate information"
    assert toy.restricted_decay(n_times=7).data_dim == 7


def test_6_the_documented_true_parameters_are_what_the_factories_build():
    """Every accuracy assertion is relative to `problem.u_true` itself, so the
    documented values `(2.0, 1.5)` were unpinned and free to drift."""
    for factory in (toy.exponential_decay, toy.restricted_decay):
        np.testing.assert_array_equal(np.asarray(factory().u_true), [2.0, 1.5])
    problem = toy.linear_gaussian()
    # The linear problem's u_true is a prior draw, so it is pinned by digits.
    prints_as(problem.u_true, [-1.4009, 1.4321, 0.6248, 0.2005])


def test_8_the_failure_fraction_matches_its_closed_form_under_the_prior():
    """The Notes give Phi((rate_floor - m1) / sigma1); check against it.

    A closed form exists, so the package's rules say compare against it rather
    than against a tolerance. The prior is N((1, 1), I) and the default floor
    is 0, so the expected fraction is Phi(-1) = 0.1587.
    """
    problem = toy.restricted_decay()
    particles = problem.prior.sample(jax.random.key(7), 20_000)["u"]
    predictions = np.asarray(problem.forward(particles))
    failed = (~np.isfinite(predictions).all(axis=1)).mean()

    mean = float(problem.prior.mean("u")[1])
    sd = float(problem.prior.cov("u").diag()[1]) ** 0.5
    expected = 0.5 * math.erfc(-(problem.rate_floor - mean) / (sd * math.sqrt(2.0)))
    assert expected == pytest.approx(0.15866, abs=1e-5)
    # Three standard errors of a 20,000-sample binomial is about 0.008.
    assert failed == pytest.approx(expected, abs=0.01), (failed, expected)


def test_8_the_domain_boundary_is_strict_in_both_places():
    """`>` not `>=`, in the model and in the u_true guard.

    Both were untested: the model's mask is computed from a continuous draw
    where exact equality has probability zero, and the u_true guard was only
    exercised at a floor strictly above the true rate.
    """
    problem = toy.restricted_decay(rate_floor=0.5)
    on_boundary = jnp.array([[2.0, 0.5], [2.0, 0.5 + 1e-12]])
    finite = np.isfinite(np.asarray(problem.forward(on_boundary))).all(axis=1)
    assert not finite[0], "a rate exactly at the floor is outside the domain"
    assert finite[1]
    # And a floor exactly at the true rate is refused, not only one above it.
    with pytest.raises(ValueError, match="outside the valid domain"):
        toy.restricted_decay(rate_floor=1.5)


# ===========================================================================
# 5. regressions for the defects an adversarial review found
# ===========================================================================


# The three `test_12_regression_*` tests of the checker -- a coupling in a
# small observable, an ensemble too small to check, and a permutation that is
# never the identity -- moved to `tests/test_maps.py` with the checker; the
# maps contract's "Ported regression tests" table maps each to its new name.


def test_12_regression_the_restricted_model_differentiates():
    """`jnp.where` evaluates both branches, so the discarded one must be safe.

    Without clamping the rate inside the valid branch, a particle below the
    floor computes `exp(-rate * t)` in the discarded branch, overflows to
    `inf`, and the derivative returns `nan` from `0 * inf`. The threshold is
    a rate of about -236 at the default `t_max` — but only about -29 in
    float32, and it scales with `t_max`, so it is not a remote corner.
    """
    problem = toy.restricted_decay()
    total = lambda ensemble: jnp.nansum(problem.forward(ensemble))  # noqa: E731
    for rate in (-10.0, -300.0, -1e5):
        ensemble = jnp.array([[2.0, rate], [2.0, 1.5]])
        gradient = np.asarray(jax.grad(total)(ensemble))
        assert np.isfinite(gradient).all(), (rate, gradient)
    # And the failing row is still wholly non-finite, which is the signal.
    predictions = np.asarray(problem.forward(jnp.array([[2.0, -300.0]])))
    assert not np.isfinite(predictions).any()


def test_12_regression_the_size_guard_bounds_the_transform_too():
    """The conditioning forms a (k, k) array, so a wide prior factor is bound.

    A guard on `P * k` alone let a `PSDLowRank` prior of width 9000 in four
    dimensions through — 36,000 guarded elements while building a 9000-by-9000
    transform.
    """
    problem = toy.linear_gaussian(parameter_dim=4, data_dim=3)
    factor = jnp.asarray(np.random.default_rng(0).normal(size=(4, 9000)))
    wide = Gaussian({"u": jnp.zeros(4)}, block_covs={"u": PSDLowRank(factor)})
    with pytest.raises(ValueError, match="9000-by-9000"):
        dataclasses.replace(problem, prior=wide).posterior()


def test_12_regression_every_problem_field_is_keyword_only():
    """`times` and `y` are both (N,) arrays, so a positional swap is silent.

    The same hazard for which `enskit.distribution.Gaussian` makes its factor
    rows and independent terms keyword-only — and worse here, since `times`
    and `y` collide at every N rather than only when two dimensions agree.
    """
    problem = toy.exponential_decay()
    with pytest.raises(TypeError, match="positional"):
        toy.ExponentialDecay(
            problem.y,  # would be `times`
            problem.times,
            problem.prior,
            problem.noise_cov,
            problem.u_true,
        )
    with pytest.raises(TypeError, match="positional"):
        toy.LinearGaussian(
            problem.prior, problem.noise_cov, problem.y, problem.u_true, problem.times
        )


@pytest.mark.parametrize("name, problem, parameter_dim, data_dim", PROBLEMS, ids=IDS)
def test_12_regression_a_problem_hashes_and_compares_without_raising(
    name, problem, parameter_dim, data_dim
):
    """Array fields make a synthesized `__eq__`/`__hash__` raise, not answer.

    Every other value class in the package is `eq=False, repr=False`, and its
    `repr` shows type and static sizes rather than array data.
    """
    assert isinstance(hash(problem), int)
    assert problem == problem
    assert (problem == dataclasses.replace(problem)) is False
    text = repr(problem)
    assert text.startswith(type(problem).__name__)
    assert "Array" not in text and len(text) < 80, text


def test_12_regression_a_field_that_cannot_be_inspected_is_rejected():
    """A Python list passes a shape check and then fails inside the model.

    `jnp.shape` accepts anything shape-like — and is deprecated for
    non-arrays — so validation waved lists through and the model raised an
    error naming a tracer instead of the field.
    """
    problem = toy.exponential_decay()
    for field, value in [
        ("y", [0.0] * 12),
        ("u_true", [2.0, 1.5]),
        ("times", [0.5, 1.0]),
    ]:
        with pytest.raises(TypeError, match="no shape to check"):
            dataclasses.replace(problem, **{field: value})


def test_12_regression_a_vmapped_family_field_is_rejected_at_construction():
    """Otherwise it is diagnosed later, by the operator, not by the problem."""
    problem = toy.linear_gaussian()
    family = jax.vmap(PSDDiagonal)(jnp.ones((3, 8)))
    with pytest.raises(ValueError, match="vmapped family"):
        dataclasses.replace(problem, noise_cov=family)


def test_12_regression_rate_floor_is_type_checked_not_coerced():
    """`float()` on a bool moves the domain boundary; on a str it detonates.

    `True` silently became a floor of 1.0 — failing about 84% of the shipped
    prior rather than 16% — while the validator's message reported the
    coerced value. A string passed validation and raised from inside `vmap`.
    """
    problem = toy.restricted_decay()
    for value in ("0.0", True, jnp.float64(0.0)):
        with pytest.raises(TypeError, match="must be a Python float"):
            dataclasses.replace(problem, rate_floor=value)
    with pytest.raises(ValueError, match="must be finite"):
        dataclasses.replace(problem, rate_floor=float("inf"))
    assert isinstance(problem.rate_floor, float)


def test_12_regression_the_factories_reject_infinite_scales():
    """`not (x > 0)` catches nan and lets inf through, building a nan problem."""
    with pytest.raises(ValueError, match="positive and finite"):
        toy.linear_gaussian(prior_std=float("inf"))
    with pytest.raises(ValueError, match="positive and finite"):
        toy.linear_gaussian(noise_std=float("inf"))
    with pytest.raises(ValueError, match="positive and finite"):
        toy.exponential_decay(t_max=float("inf"))
    with pytest.raises(TypeError, match="must be an int"):
        toy.linear_gaussian(parameter_dim=True)


def test_12_regression_no_layer_imports_the_toy_module():
    """The architectural rule `CLAUDE.md` calls permanent, in one line.

    The same holds for `enskit.testing`, which may import every layer.
    `enskit.toy` depends on the layers, so an import in the other direction
    would make toy problems load-bearing for the library. Checked in a fresh
    interpreter, since this one has already imported the module, and over
    every layer: `enskit.linalg`, `enskit.distribution`, `enskit.maps`,
    `enskit.kalman` and `enskit.algorithms` with its EKI driver.
    """
    program = (
        "import sys; import enskit, enskit.linalg, enskit.distribution, "
        "enskit.maps, enskit.kalman, enskit.algorithms, enskit.algorithms.eki, "
        "enskit.linalg.testing; "
        "print('enskit.toy' in sys.modules, 'enskit.testing' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False False", result.stdout


@pytest.mark.parametrize("name, problem, parameter_dim, data_dim", PROBLEMS, ids=IDS)
def test_12_regression_a_problem_is_not_a_pytree(name, problem, parameter_dim, data_dim):
    """Deliberately a plain frozen dataclass, so `jit` sees it as one leaf.

    Registering one would change what crosses a trace boundary, silently, and
    would import every question the operator layer had to answer about
    vmapped families and constructor-bypassing unflatten.
    """
    leaves = jax.tree.leaves(problem)
    assert len(leaves) == 1 and leaves[0] is problem, (
        "a problem became a pytree; jit and vmap semantics change silently"
    )


@pytest.mark.parametrize("name, problem, parameter_dim, data_dim", PROBLEMS, ids=IDS)
def test_12_regression_forward_refuses_anything_but_one_ensemble(
    name, problem, parameter_dim, data_dim
):
    """A single parameter vector returned a plausible ``(N,)``, silently.

    The generalized-ufunc convention carried any leading rank through, so
    ``problem.forward(problem.u_true)`` answered — and passing one particle
    instead of the ensemble is the mistake the forward-model guide calls the
    most common one. The two decay models raised an ``IndexError`` from
    inside JAX, naming neither the ensemble nor the model. All three now
    agree, and ``vmap`` over the method still works.
    """
    with pytest.raises(ValueError, match=f"expected a .J, {parameter_dim}. ensemble"):
        problem.forward(problem.u_true)
    with pytest.raises(ValueError, match="never with a further leading axis"):
        problem.forward(jnp.zeros((2, 3, parameter_dim)))
    with pytest.raises(ValueError, match="expected a"):
        problem.forward(jnp.zeros((3, parameter_dim + 1)))
    mapped = jax.vmap(problem.forward)(jnp.zeros((2, 3, parameter_dim)))
    assert mapped.shape == (2, 3, data_dim)
