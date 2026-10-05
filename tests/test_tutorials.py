"""The tutorial series: its runnable blocks, its figures, and its prose numbers.

Conformance obligation 26 of the EKI contract requires every runnable block in
the documentation to be executed by a test with its printed output pinned.
Figures make that harder rather than optional, so there are three kinds of test
here:

1. **The blocks**, in the order each page runs them, with every value the page
   prints asserted to the digits it shows.
2. **The figures**, through :func:`figures.build` into a temporary directory,
   and then through the data dictionary each figure function returns. Those are
   the values the figure *plotted*, so a figure drawing the wrong array fails
   here rather than merely looking wrong. Pixels are deliberately not compared;
   ``docs/figures.py`` records why.
3. **The prose numbers that are not in any block** — the counts and ratios the
   pages state in a sentence. Each one is derived here from the same data the
   figure plotted, so the two cannot drift apart.

Two claims the pages rest on get their own tests: that an evaluation and the
record from the same step carry the same tempering level, which is what makes
an ensemble pairable with the distribution it belongs to; and that the grid
reference the pages compare against is converged, which is the only reason a
nonlinear problem has a reference at all.

The identity behind the ``center_misfit`` gap is *not* re-tested here — it is
``tests/test_eki.py::test_11_the_center_misfit_differs_from_the_mean_by_exactly_the_spread_term``.
This file pins only the numbers tutorial 2 prints.

Section 5 holds the assertions whose page claims no longer hold since PR 7
changed the prior draw (``EKIState.from_prior`` now splits its key, and
``Gaussian.sample`` splits it again). Each is a strict ``xfail`` naming the
sentence it pins, so the claim is neither deleted nor silently passing, and
the test fails once the page or the draw makes it true again.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import prints_as

import enskit  # noqa: F401  -- enables x64 before any array exists
from enskit import kalman, maps, toy
from enskit.algorithms import eki
from enskit.algorithms.eki import (
    AdaptiveESSSchedule,
    DiscrepancyStop,
    EKIState,
    FixedSchedule,
    effective_sample_size,
    iterate,
    run,
)
from enskit.distribution import Gaussian
from enskit.linalg import PSDDiagonal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs"))
import figures  # noqa: E402  -- docs/figures.py, on the path just above

#: The pages this file covers, for the reference scan below.
TUTORIAL_DIR = Path(__file__).resolve().parents[1] / "docs" / "tutorials"

#: The update rules the pages run: tutorial 1's, and tutorials 2 and 3's.
MATHERON = kalman.Matheron()
SQUARE_ROOT = kalman.SymmetricSquareRoot()


def _state(n_particles=64):
    """The initial state every page draws: key 0, from the problem's prior."""
    problem = toy.exponential_decay()
    return EKIState.from_prior(
        jax.random.key(0), problem.prior, n_particles=n_particles
    )


def _sd(ensemble):
    """Block ``"u"``'s per-coordinate standard deviation, as the pages print it."""
    return np.asarray(ensemble["u"]).std(axis=0, ddof=1)


# ===========================================================================
# 1. the runnable blocks of each page
# ===========================================================================


def test_1_tutorial_1_blocks_run():
    """Every runnable block of "From approximate conditioning to EKI", in order."""
    problem = toy.exponential_decay()

    assert (problem.parameter_dim, problem.data_dim) == (2, 12)
    prints_as(problem.u_true, [2.0, 1.5])
    prints_as(problem.y[:4], [1.3705, 0.929, 0.6856, 0.45])

    # The hand-written forward model, and its agreement with the problem's.
    times = problem.times
    prints_as([float(times[0]), float(times[-1])], [0.25, 3.0], decimals=2)

    def forward(ensemble):
        amplitude = ensemble[:, 0:1]
        rate = ensemble[:, 1:2]
        return amplitude * jnp.exp(-rate * times)

    ensemble = jnp.array([[2.0, 1.5], [1.0, 0.5]])
    assert forward(ensemble).shape == (2, 12)
    assert jnp.array_equal(forward(ensemble), problem.forward(ensemble))

    # The vmap wrapper, which the page offers as the alternative.
    def one_member(member):
        return member[0] * jnp.exp(-member[1] * times)

    vmapped = jax.vmap(one_member)
    assert jnp.array_equal(vmapped(ensemble), problem.forward(ensemble))

    # The prior and the noise the page writes out are the problem's own.
    prior = Gaussian.independent(
        u=(jnp.array([1.0, 1.0]), PSDDiagonal(jnp.array([1.0, 1.0])))
    )
    noise_cov = PSDDiagonal(jnp.full(12, 0.02**2))
    assert prior.names == problem.prior.names == ("u",)
    assert jnp.array_equal(prior.mean("u"), problem.prior.mean("u"))
    assert jnp.array_equal(
        prior.cov("u").to_dense(), problem.prior.cov("u").to_dense()
    )
    assert jnp.array_equal(noise_cov.to_dense(), problem.noise_cov.to_dense())
    # The page states these as m0 = [1, 1], C0 = I, Sigma = 0.02^2 I.
    assert jnp.array_equal(prior.cov("u").to_dense(), jnp.eye(2))
    assert jnp.array_equal(noise_cov.to_dense(), 0.02**2 * jnp.eye(12))

    state = eki.EKIState.from_prior(
        jax.random.key(0), problem.prior, n_particles=64
    )
    assert state.ensemble["u"].shape == (64, 2)
    predictions = problem.forward(state.ensemble["u"])
    assert predictions.shape == (64, 12)

    one_step = eki.run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        update_rule=kalman.Matheron(),
        schedule=eki.FixedSchedule.constant(1.0, n_steps=1),
    )
    prints_as(one_step.mean("u"), [1.9875, 1.5678])
    prints_as(one_step.ensemble["u"].std(axis=0, ddof=1), [0.0567, 0.6194])

    # The Gaussian the update conditions, which the particles are drawn from.
    joint = state.ensemble.pipe(maps.pushforward, problem.forward, output="g")
    conditioned = (
        joint.project().add_noise(g=problem.noise_cov).condition(g=problem.y)
    )
    prints_as(conditioned.mean("u"), [1.9883, 1.5709])
    prints_as(conditioned.cov("u").diag() ** 0.5, [0.0584, 0.6448])

    # "mean and covariance that should approximately match the sample mean and
    # covariance of the ensemble above (they would match exactly if we had
    # instead used `TransformUpdate`)." Both halves are asserted: approximate
    # under the Matheron update, exact under the symmetric square root.
    gaussian_sd = np.sqrt(np.asarray(conditioned.cov("u").diag()))
    ensemble_sd = _sd(one_step.ensemble)
    relative = np.abs(ensemble_sd - gaussian_sd) / gaussian_sd
    assert relative.max() < 0.08, relative
    assert relative.max() > 1e-3, relative  # approximate, not exact

    deterministic = run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        update_rule=SQUARE_ROOT,
        schedule=FixedSchedule.constant(1.0, n_steps=1),
    )
    exact_cov = np.cov(np.asarray(deterministic.ensemble["u"]), rowvar=False, ddof=1)
    gaussian_cov = np.asarray(conditioned.cov("u").to_dense())
    assert (
        float(jnp.abs(conditioned.mean("u") - deterministic.mean("u")).max())
        < 1e-14
    )
    assert np.abs(gaussian_cov - exact_cov).max() < 1e-15

    result = eki.run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        update_rule=kalman.Matheron(),
        schedule=eki.AdaptiveESSSchedule(),
    )
    assert result.status == "schedule_exhausted"
    assert result.n_evaluations == 7
    assert float(result.beta) == 1.0
    # "This run required seven calls ... 448 parameter evaluations in total."
    assert result.n_evaluations * 64 == 448
    assert result.ensemble["u"].shape == (64, 2)
    prints_as(result.mean("u"), [1.9745, 1.4737])
    prints_as(result.ensemble["u"].std(axis=0, ddof=1), [0.0346, 0.0288])

    # The grid block the page closes on, and the table it fills in.
    amp = jnp.linspace(1.70, 2.25, 400)
    rate = jnp.linspace(1.25, 1.70, 400)
    A, R = jnp.meshgrid(amp, rate, indexing="ij")
    grid = jnp.stack([A.ravel(), R.ravel()], axis=-1)
    assert grid.shape == (160_000, 2)

    log_density = problem.prior.log_density(u=grid) - eki.misfits(
        problem.y, problem.forward(grid), problem.noise_cov
    )
    weights = jnp.exp(log_density - log_density.max())
    weights = weights / weights.sum()

    exact_mean = (weights[:, None] * grid).sum(axis=0)
    centered = grid - exact_mean
    exact_cov = (weights[:, None] * centered).T @ centered

    prints_as(exact_mean, [1.9769, 1.4719])
    prints_as(jnp.sqrt(jnp.diag(exact_cov)), [0.0366, 0.0317])

    # The block is a single grid where `figures._tempered_moments` refines its
    # box; the page uses the simple version, so it has to agree with the
    # refined one it is standing in for.
    refined_mean, refined_sd = figures._tempered_moments(1.0)
    assert np.abs(np.asarray(exact_mean) - refined_mean).max() < 1e-6
    assert (
        np.abs(np.asarray(jnp.sqrt(jnp.diag(exact_cov))) - refined_sd).max() < 1e-6
    )

    exact_sd = np.asarray(jnp.sqrt(jnp.diag(exact_cov)))
    exact_corr = float(
        exact_cov[0, 1] / jnp.sqrt(exact_cov[0, 0] * exact_cov[1, 1])
    )
    ensemble = np.asarray(result.ensemble["u"])
    prints_as(exact_corr, 0.822, 3)
    prints_as(float(np.corrcoef(ensemble.T)[0, 1]), 0.727, 3)

    # "The mean agrees to within 0.003 in both parameters, and the spreads
    # come out about 5% too narrow in the amplitude and 9% too narrow in the
    # rate."
    mean_error = np.abs(np.asarray(result.mean("u")) - np.asarray(exact_mean))
    assert mean_error.max() < 0.003, mean_error
    sd_ratio = _sd(result.ensemble) / exact_sd
    prints_as(1.0 - sd_ratio, [0.05, 0.09], 2)

    # The truth sits inside the ensemble's spread in both parameters, which is
    # the reason the page can show `u_true` on the figure without apology.
    error = np.abs(np.asarray(result.mean("u")) - np.asarray(problem.u_true))
    assert np.all(error < _sd(result.ensemble))


def test_1_tutorial_2_blocks_run():
    """Every runnable block of "Reading a run", in order."""
    problem = toy.exponential_decay()
    state = eki.EKIState.from_prior(
        jax.random.key(0), problem.prior, n_particles=64
    )
    result = eki.run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        update_rule=kalman.SymmetricSquareRoot(),
        schedule=eki.AdaptiveESSSchedule(),
    )

    assert result.status == "schedule_exhausted"
    assert result.budget_complete is True
    assert result.stop_fired is False
    assert float(result.beta) == 1.0
    assert result.min_n_valid == 64
    assert result.n_evaluations == 6
    assert result.n_completed_steps == 6
    # "in particle evaluations `n_evaluations * n_particles` — 384 here".
    assert result.n_evaluations * state.n_particles == 384

    history = result.stacked
    prints_as(history.beta, [0.0, 0.001, 0.0029, 0.0112, 0.0571, 0.3152])
    prints_as(history.increment, [0.001, 0.0019, 0.0083, 0.0459, 0.2581, 0.6848])

    # The eleven fields the page's table groups.
    for field in (
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
    ):
        assert np.asarray(getattr(history, field)).shape == (6,)

    prints_as(history.misfit_min[-1], 4.5889)
    prints_as(history.misfit_mean[-1], 8.1181)
    prints_as(history.misfit_max[-1], 69.4678)
    # "The worst particle fits eight times worse than the average one."
    assert float(history.misfit_max[-1] / history.misfit_mean[-1]) > 8.0
    # "the first step's misfit is 5.9e5"
    prints_as(history.misfit_mean[0] / 1e5, 5.9, 1)

    evaluation = result.last_evaluation
    prints_as(evaluation.misfits[:4], [5.5138, 28.2639, 4.721, 5.1447])
    prints_as(evaluation.center_misfit, 4.6114)
    prints_as(evaluation.misfits.mean(), 8.1181)
    prints_as(evaluation.misfits.mean() - evaluation.center_misfit, 3.5066)

    # The page's plotting block. matplotlib is on the Agg backend, since
    # importing `figures` above set it.
    import matplotlib.pyplot as plt

    figure = plt.figure()
    plt.plot(history.step, history.misfit_mean)
    plt.yscale("log")
    plt.close(figure)

    fitted = result.ensemble.project()
    prints_as(fitted.mean("u"), [1.9794, 1.4737])
    prints_as(fitted.cov("u").diag() ** 0.5, [0.0374, 0.0338])
    correlation = float(np.corrcoef(np.asarray(result.ensemble["u"]).T)[0, 1])
    prints_as(correlation, 0.80, decimals=2)

    phi = eki.misfits(
        problem.y, problem.forward(result.ensemble["u"]), problem.noise_cov
    )
    prints_as(phi.mean(), 5.6556)
    prints_as(eki.effective_sample_size(phi, 0.1), 62.458)
    prints_as(eki.effective_sample_size(phi, 1.0), 52.945)

    # The note on the first step's effective sample size and the increment
    # floor: the schedule wanted a shorter step than 1e-3 and could not take
    # one, which is why 24.6 sits below the floor of 32 without being a bug.
    first = next(
        iter(
            iterate(
                state,
                problem.forward,
                problem.y,
                problem.noise_cov,
                update_rule=SQUARE_ROOT,
                schedule=AdaptiveESSSchedule(),
            )
        )
    )
    _, record, first_evaluation = first
    assert float(record.increment) == pytest.approx(1e-3)
    prints_as(record.ess, 24.5511)
    prints_as(effective_sample_size(first_evaluation.misfits, 1e-4), 47.03, 2)


def test_1_tutorial_3_blocks_run():
    """Every runnable block of "Sampling or optimizing", in order."""
    problem = toy.exponential_decay()
    state = eki.EKIState.from_prior(
        jax.random.key(0), problem.prior, n_particles=64
    )

    sampled = eki.run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        update_rule=kalman.SymmetricSquareRoot(),
        schedule=eki.AdaptiveESSSchedule(),
    )
    assert float(sampled.beta) == 1.0
    assert sampled.budget_complete is True
    # "Here it took six forward evaluations."
    assert sampled.n_evaluations == 6
    prints_as(sampled.mean("u"), [1.9794, 1.4737])
    prints_as(sampled.ensemble["u"].std(axis=0, ddof=1), [0.0374, 0.0338])
    assert eki.AdaptiveESSSchedule().beta_target == 1.0

    fit = eki.run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        update_rule=kalman.SymmetricSquareRoot(),
        schedule=eki.FixedSchedule.constant(1.0, n_steps=200),
        stop=eki.DiscrepancyStop(tau=1.0),
    )
    assert fit.status == "stopping_rule"
    assert float(fit.beta) == 2.0
    assert fit.n_evaluations == 3
    assert fit.n_completed_steps == 2
    prints_as(fit.mean("u"), [1.9942, 1.5117])
    prints_as(fit.stacked.center_misfit, [35993.2811, 257.3198, 5.6266])
    # The page reads the threshold off those values: 257.32 above, 5.63 below.
    threshold = 1.0**2 * problem.data_dim / 2
    assert threshold == 6.0
    assert float(fit.stacked.center_misfit[-2]) > threshold
    assert float(fit.stacked.center_misfit[-1]) <= threshold

    trap = eki.run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        update_rule=kalman.SymmetricSquareRoot(),
        schedule=eki.AdaptiveESSSchedule(),
        stop=eki.DiscrepancyStop(tau=1.0),
    )
    prints_as(trap.beta, 0.0571)
    # "The run ended at beta = 0.057"
    prints_as(trap.beta, 0.057, 3)
    assert trap.stop_fired is True
    assert trap.budget_complete is False
    prints_as(trap.ensemble["u"].std(axis=0, ddof=1), [0.1461, 0.1454])


def test_2_tutorial_3s_comparison_table():
    """The four rows of tutorial 3's table, and the sentences reading them.

    The table is the page's evidence for both of its claims -- that the
    optimization form gives an excellent point estimate and a spread that is
    not an uncertainty, and that the stopped run's plausible-looking spread is
    an accident of where it stopped. Every cell is asserted, since a row
    moving would invert one of those readings without changing the prose.
    """
    problem = toy.exponential_decay()
    state = _state()
    reference_mean, reference_sd = figures._tempered_moments(1.0)
    prints_as(reference_mean, [1.9769, 1.4719])
    prints_as(reference_sd, [0.0366, 0.0317])

    rows = {
        "sampling": dict(schedule=AdaptiveESSSchedule()),
        "tau2": dict(
            schedule=FixedSchedule.constant(1.0, n_steps=200),
            stop=DiscrepancyStop(tau=2.0),
        ),
        "tau1": dict(
            schedule=FixedSchedule.constant(1.0, n_steps=200),
            stop=DiscrepancyStop(tau=1.0),
        ),
        "beta30": dict(schedule=FixedSchedule.constant(1.0, n_steps=30)),
    }
    reached, calls, mean_error, sd_ratio = {}, {}, {}, {}
    for name, kwargs in rows.items():
        result = run(
            state,
            problem.forward,
            problem.y,
            problem.noise_cov,
            update_rule=SQUARE_ROOT,
            **kwargs,
        )
        reached[name] = float(result.beta)
        calls[name] = result.n_evaluations
        mean_error[name] = float(
            np.abs(np.asarray(result.mean("u")) - reference_mean).max()
        )
        sd_ratio[name] = _sd(result.ensemble) / reference_sd

    assert reached == {"sampling": 1.0, "tau2": 2.0, "tau1": 2.0, "beta30": 30.0}
    assert calls == {"sampling": 6, "tau2": 3, "tau1": 3, "beta30": 30}
    prints_as(mean_error["sampling"], 0.0025, 4)
    prints_as(mean_error["tau2"], 0.0397, 4)
    prints_as(mean_error["tau1"], 0.0397, 4)
    prints_as(mean_error["beta30"], 0.0006, 4)
    prints_as(sd_ratio["sampling"], [1.02, 1.07], 2)
    prints_as(sd_ratio["tau2"], [1.15, 2.94], 2)
    prints_as(sd_ratio["tau1"], [1.15, 2.94], 2)
    prints_as(sd_ratio["beta30"], [0.19, 0.19], 2)

    # "the smallest error in the mean of the four rows, four times smaller
    # than the sampling form's".
    assert mean_error["beta30"] == min(mean_error.values())
    assert 3.5 < mean_error["sampling"] / mean_error["beta30"] < 4.5
    # "a spread five times too small". The `tau=1` row's "within 13% of the
    # target's" no longer holds: see `test_8_...tau_1_spread...`.
    assert 4.5 < 1.0 / sd_ratio["beta30"].max() < 5.5
    # "nearly three times too large in the rate" at `tau=2`: the point is that
    # the spread is set by where the run stopped, not by the target.
    assert sd_ratio["tau2"][1] > 2.8
    # "On this problem the optimization form is also the cheaper of the two,
    # at three forward evaluations against six."
    assert calls["tau2"] == calls["tau1"] == 3 < calls["sampling"] == 6
    # "the trap's spread is about four times the target's".
    trap = run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        update_rule=SQUARE_ROOT,
        schedule=AdaptiveESSSchedule(),
        stop=DiscrepancyStop(tau=1.0),
    )
    trap_ratio = _sd(trap.ensemble) / reference_sd
    prints_as(trap_ratio, [3.99, 4.58], 2)


# ===========================================================================
# 2. the figures
# ===========================================================================


def test_3_every_figure_builds(tmp_path):
    """Every figure is written, and is a plausible PNG rather than an empty file."""
    plotted = figures.build(tmp_path)
    assert set(plotted) == set(figures.FIGURES)
    for name in figures.FIGURES:
        path = tmp_path / f"{name}.png"
        assert path.exists(), name
        assert path.stat().st_size > 20_000, (name, path.stat().st_size)
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_3_every_figure_a_page_references_is_generated():
    """No page references a figure nobody builds, and no figure is an orphan.

    A missing figure would already fail the documentation build, since it runs
    with warnings as errors. An *orphan* would not: a figure generated on every
    build and referenced by nothing is cost with no reader.
    """
    referenced = set()
    for page in sorted(TUTORIAL_DIR.glob("*.md")):
        for line in page.read_text().splitlines():
            marker = "../_generated/figures/"
            if marker in line:
                referenced.add(line.split(marker, 1)[1].removesuffix(".png").strip())
    assert referenced == set(figures.FIGURES)


def test_4_the_prior_predictive_figure_plots_what_tutorial_1_says():
    """Tutorial 1's opening figure, and the two counts in its caption."""
    data = figures.prior_predictive()[1]
    assert data["n_particles"] == 64
    # The caption says "some curves leave the panel, and some have a negative
    # decay rate", so that is what is asserted; the exact counts follow as a
    # regression pin on the picture rather than on the prose, since a figure
    # in which every curve or no curve left the panel would be a different
    # figure making a different point.
    assert 0 < data["prior_curves_leaving_panel"] < data["n_particles"]
    assert 0 < data["prior_negative_rates"] < data["n_particles"]
    assert data["prior_curves_leaving_panel"] == 12
    assert data["prior_negative_rates"] == 13
    # The panel's own limits, which decide that first count.
    prints_as(data["prediction_ylim"], [-0.6, 2.4], 2)
    # "the noise in this problem is quite small, so the bars are obscured by
    # the observation markers" -- true only because the panel spans 3 units.
    assert data["noise_sd"] == 0.02
    # The prior predictive is wide, which is the panel's whole point.
    assert data["prior_predictive_sd"][figures.JOINT_INDEX] > 0.9


def test_4_the_one_step_figure_plots_what_tutorial_1_says():
    """Tutorial 1's conditioning figure, and the sentences that read it."""
    data = figures.one_step()[1]
    assert data["n_particles"] == 64
    prints_as(data["one_step_mean"], [1.9875, 1.5678])
    prints_as(data["one_step_sd"], [0.0567, 0.6194])
    prints_as(data["posterior_mean"], [1.9769, 1.4719])
    prints_as(data["posterior_sd"], [0.0366, 0.0317])

    # The section reads the failure off this figure qualitatively: the
    # ensemble finds the right region and misrepresents the shape. These pin
    # what "misrepresents" amounts to, so a change that quietly fixed it -- or
    # made it worse -- would not pass unnoticed. (Bands until PR 7, whose new
    # prior draw moved both ratios below them; pinned to the new draw rather
    # than widened.)
    ratio = data["one_step_sd"] / data["posterior_sd"]
    prints_as(ratio, [1.55, 19.51], 2)
    # The rate is narrowed only to 0.62 of the prior's own spread of one.
    prior_sd = np.sqrt(np.asarray(toy.exponential_decay().prior.cov("u").diag()))
    prints_as(data["one_step_sd"][1] / prior_sd[1], 0.62, 2)
    # In prediction space the fan is still far wider than the error bars.
    prints_as(data["one_step_predictive_sd"][figures.JOINT_INDEX] / 0.02, 22.3, 1)
    # The panel is sized to the ensemble, so almost every curve stays in the
    # prediction panel; the parameter panel is where the failure shows.
    assert data["one_step_curves_leaving_panel"] == 2


def test_4_the_tempering_bridge_figure_plots_what_tutorial_1_says():
    """Tutorial 1's closing figure: the exact tempered family, five levels."""
    data = figures.tempering_bridge()[1]
    prints_as(data["levels"], [0.0, 0.001, 0.01, 0.1, 1.0], 3)
    prints_as(
        data["sd"],
        [
            [1.0, 1.0],
            [0.6186, 0.6016],
            [0.3249, 0.2906],
            [0.1145, 0.0997],
            [0.0366, 0.0317],
        ],
    )
    # The two ends are the prior and the posterior, which is what makes it a
    # bridge; beta = 0 is checkable against the prior in closed form.
    problem = toy.exponential_decay()
    prints_as(data["mean"][0], np.asarray(problem.prior.mean("u")), 4)
    prints_as(data["sd"][0], np.sqrt(np.asarray(problem.prior.cov("u").diag())), 4)
    prints_as(data["mean"][-1], [1.9769, 1.4719])
    prints_as(data["sd"][-1], [0.0366, 0.0317])

    # "the distribution narrows by a factor of about thirty along the way",
    # monotonically, which is why each panel needs its own scale.
    assert np.all(np.diff(data["sd"], axis=0) < 0.0)
    narrowing = data["sd"][0] / data["sd"][-1]
    assert 25.0 < narrowing.min() and narrowing.max() < 35.0, narrowing


def test_4_the_tracked_bridge_figure_plots_what_tutorial_1_says():
    """Tutorial 1's figure of the run against the exact tempered family.

    The page makes two claims about it: the particles follow the exact
    distribution closely at most levels, and the second level is the
    exception because its target is a curved ridge. Both are ratios of
    plotted quantities. The parts of them that no longer hold since PR 7 are
    ``test_8_the_tracked_bridge_...``.
    """
    data = figures.bridge_tracked()[1]
    assert data["n_particles"] == 64
    prints_as(
        data["levels"],
        [0.0, 0.001, 0.0035, 0.0083, 0.0271, 0.0998, 0.4138, 1.0],
    )

    ratio = data["cloud_sd"] / data["exact_sd"]
    # "the ensemble tracks the distributions fairly well".
    others = np.delete(ratio, 1, axis=0)
    assert others.max() < 1.35, ratio
    # "the curvature present at the second distribution presents a challenge"
    # -- it is the most over-dispersed level in the rate.
    assert ratio[1, 1] == ratio[:, 1].max(), ratio
    # "the final posterior approximation is far superior to the one-step
    # result": within about 15% at beta = 1, against a factor of 20.
    assert np.abs(ratio[-1] - 1.0).max() < 0.15, ratio
    # The panels hold the clouds: a handful of particles outside is a scatter
    # plot, a third of them outside is a badly sized panel.
    assert data["particles_outside_panel"].max() <= 6, data[
        "particles_outside_panel"
    ]


def test_4_the_answer_figure_plots_what_tutorial_1_says():
    """Tutorial 1's closing figure, and the contrast it draws with one step."""
    data = figures.answer()[1]
    assert data["n_particles"] == 64
    assert data["status"] == "schedule_exhausted"
    assert data["n_evaluations"] == 7
    prints_as(data["mean"], [1.9745, 1.4737])
    prints_as(data["sd"], [0.0346, 0.0288])
    prints_as(data["posterior_sd"], [0.0366, 0.0317])
    # The right-hand panel's fan is inside the observation error bars, which
    # is what distinguishes it from the one-step figure's. No page quotes this
    # number any more, so the pin guards the figure rather than the prose.
    prints_as(data["predictive_sd"][figures.JOINT_INDEX], 0.009)
    assert data["predictive_sd"][figures.JOINT_INDEX] < 0.02
    ratio = data["sd"] / data["posterior_sd"]
    assert ratio.max() < 1.2, ratio
    # The panel is sized to the ensemble, so no prediction curve escapes.
    assert data["curves_leaving_panel"] == 0


def test_4_the_trajectories_figure_plots_what_tutorial_2_says():
    """Tutorial 2's three panels, and every ratio the page reads off them."""
    data = figures.trajectories()[1]
    assert data["n_particles"] == 64
    assert data["noise_floor"] == 6.0

    adaptive_ess = data["adaptive_ess"]
    prints_as(adaptive_ess, [24.5511, 32.0, 32.0, 32.0, 32.0, 44.567])
    # "sits on 32 ... the last step is the exception, at 44.6, because by then
    # only 0.6848 of budget remained".
    assert np.allclose(adaptive_ess[1:-1], 32.0)
    prints_as(adaptive_ess[-1], 44.6, 1)
    prints_as(1.0 - data["adaptive_beta"][-1], 0.6848)
    # "Its misfit falls from 5.9e5 to 8.12".
    prints_as(data["adaptive_misfit_mean"][0] / 1e5, 5.9, 1)
    prints_as(data["adaptive_misfit_mean"][-1], 8.1181)

    coarse_ess = data["three_equal_steps_ess"]
    prints_as(coarse_ess[0], 1.0)
    assert coarse_ess[0] < 1.001
    # "its last recorded misfit is 17.6, three times the reference of 6".
    prints_as(data["three_equal_steps_misfit_mean"][-1], 17.6488)
    assert round(float(data["three_equal_steps_misfit_mean"][-1]) / 6.0) == 3

    # "its spread falls smoothly, by a factor between 1.5 and 2.3 per step"
    # against "a factor of 4.8 in one step".
    adaptive_ratios = data["adaptive_spread"][:-1] / data["adaptive_spread"][1:]
    coarse_ratios = (
        data["three_equal_steps_spread"][:-1]
        / data["three_equal_steps_spread"][1:]
    )
    prints_as([adaptive_ratios.min(), adaptive_ratios.max()], [1.5, 2.3], 1)
    prints_as(coarse_ratios.max(), 4.84, 2)

    # "19% larger in the amplitude and 45% larger in the rate".
    inflation = data["three_equal_steps_sd"] / data["adaptive_sd"]
    prints_as(inflation, [1.19, 1.45], 2)


def test_4_the_two_forms_figure_plots_what_tutorial_3_says():
    """Tutorial 3's figure, and the caption's stopping level."""
    data = figures.two_forms()[1]
    assert data["n_particles"] == 64
    assert data["stopped_status"] == "stopping_rule"
    assert data["stopped_beta"] == 2.0
    assert data["stopped_evaluations"] == 3
    assert data["unstopped_beta"] == 30.0
    prints_as(data["sampled_sd"], [0.0374, 0.0338])
    prints_as(data["stopped_sd"], [0.042, 0.0934])
    prints_as(data["unstopped_sd"], [0.0069, 0.006])
    prints_as(data["reference_mean"], [1.9769, 1.4719])
    prints_as(data["reference_sd"], [0.0366, 0.0317])
    # The right panel's message: the spread keeps falling past the stop.
    assert data["unstopped_sd"].max() < data["stopped_sd"].min()


# ===========================================================================
# 3. the two claims the pages rest on
# ===========================================================================


@pytest.mark.parametrize(
    "n_particles",
    [
        pytest.param(
            64,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "since PR 7's prior draw the ratio at 64 particles is "
                    "[1.55, 19.51], below the band; at 2048 it is [2.05, 22.45], "
                    "inside it, so the page's claim holds but the band does not"
                ),
            ),
        ),
        2048,
    ],
)
def test_5_the_one_step_error_is_the_gaussian_fit_not_the_ensemble_size(
    n_particles,
):
    """Tutorial 1: "the discrepancy does not go away with a larger ensemble".

    The page attributes the one-step failure to the Gaussian approximation
    rather than to sampling error, which is a claim about what happens as the
    ensemble grows. Thirty-two times as many particles leaves the overspread
    in the decay rate essentially unchanged, so the attribution holds.

    Stated as a band rather than a pinned value: the point is that the ratio
    stays large, not that it takes a particular value at a particular size.
    """
    problem = toy.exponential_decay()
    state = _state(n_particles)
    result = run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        update_rule=MATHERON,
        schedule=FixedSchedule.constant(1.0, n_steps=1),
    )
    _, posterior_sd = figures._tempered_moments(1.0)
    ratio = _sd(result.ensemble) / posterior_sd
    assert 20.0 < ratio[1] < 26.0, (n_particles, ratio)
    assert 1.9 < ratio[0] < 2.5, (n_particles, ratio)


def test_5_an_evaluation_and_its_record_carry_the_same_level():
    """``Evaluation.beta`` is bit-identical to the ``beta`` of its own record.

    Every figure that overlays an ensemble on the distribution it belongs to
    depends on this, and getting it wrong is silent: pairing each cloud with
    ``beta_next`` instead puts all of them one contour set out of step, which
    reads as the method tracking badly rather than as a plotting bug. Tutorial
    4's figure is built on it, and ``docs/figures.py`` pairs on
    ``evaluation.beta`` for this reason.

    Bit-identity is asserted rather than a tolerance because the two fields
    are the same value carried two ways, not two computations of it.
    """
    problem = toy.exponential_decay()
    state = _state(32)
    levels = []
    for _, record, evaluation in iterate(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        update_rule=SQUARE_ROOT,
        schedule=AdaptiveESSSchedule(),
    ):
        assert jnp.array_equal(evaluation.beta, record.beta)
        assert int(evaluation.step) == int(record.step)
        # The distinct field, so that the assertion above is not vacuous.
        if float(record.increment) > 0.0:
            assert not jnp.array_equal(evaluation.beta, record.beta_next)
        levels.append(float(record.beta))
    assert levels[0] == 0.0
    assert levels == sorted(levels)


@pytest.mark.parametrize("n", [150, 200, 300, 400])
@pytest.mark.parametrize("refinements", [2, 3])
def test_6_the_grid_reference_is_converged_at_beta_one(n, refinements):
    """Tutorial 3's reference, across three grid resolutions.

    The nonlinear problem has no closed form, so the target's moments are
    quadrature. Tutorial 3 states them to four decimals and says they are
    converged; this is that claim, and it is why ``_tempered_moments`` refines
    its box rather than using one grid.
    """
    mean, sd = figures._tempered_moments(1.0, n=n, refinements=refinements)
    prints_as(mean, [1.9769, 1.4719])
    prints_as(sd, [0.0366, 0.0317])


@pytest.mark.parametrize("n", [200, 400])
def test_6_the_grid_reference_recovers_the_prior_at_beta_zero(n):
    """At ``beta = 0`` the target *is* the prior, which is known exactly.

    This is the one level where the quadrature can be checked against a
    closed form rather than against itself, so it is the strongest available
    evidence that the estimator is right and not merely self-consistent.
    """
    problem = toy.exponential_decay()
    mean, sd = figures._tempered_moments(0.0, n=n)
    prints_as(mean, np.asarray(problem.prior.mean("u")), decimals=4)
    prints_as(sd, np.sqrt(np.asarray(problem.prior.cov("u").diag())), decimals=4)


def test_6_regression_one_unrefined_grid_is_not_enough():
    """Why the box is refined, stated as the failure it avoids.

    A single grid wide enough for the prior spans eight units in each
    parameter, and the target's standard deviation is 0.037 — about five grid
    points across its whole width at ``n = 200``. It reports a mean wrong in
    the fourth decimal, which is the precision tutorial 3 prints. And a box
    sized for the target truncates the prior: at ``beta = 0`` it puts the
    prior's mean at ``[1.4839, 1.3053]`` rather than ``[1, 1]``.

    Neither failure raises, and each is invisible in a contour plot.
    """
    unrefined = figures._tempered_moments(1.0, n=200, refinements=1)[0]
    prints_as(unrefined, [1.9772, 1.4719])
    assert not np.array_equal(np.round(unrefined, 4), [1.9769, 1.4719])

    narrow = figures._grid((0.5, 3.5, 0.2, 3.0), 160)[2]
    truncated, _ = figures._moments(narrow, figures._log_terms(narrow)[0])
    prints_as(truncated, [1.4839, 1.3053])


# ===========================================================================
# 4. regressions
# ===========================================================================


def test_7_regression_the_figure_cache_notices_a_changed_source(tmp_path):
    """``is_current`` is false for a missing figure and for a stale one.

    The build skips regeneration when every output postdates every source, so
    a cache that answered ``True`` too readily would let a figure rot through a
    documentation build that reported success.
    """
    assert figures.is_current(tmp_path) is False  # nothing written yet
    figures.build(tmp_path)
    assert figures.is_current(tmp_path) is True

    # A source file newer than the outputs must invalidate them.
    for path in tmp_path.glob("*.png"):
        stale = figures._newest_source_time() - 60.0
        import os

        os.utime(path, (stale, stale))
    assert figures.is_current(tmp_path) is False


def test_7_regression_a_toy_problem_is_not_passed_to_run():
    """The three-argument call every tutorial writes is the real signature.

    Every block in the series calls ``run(state, problem.forward, problem.y,
    problem.noise_cov, ...)``, so a reader who tries handing the container to
    ``run`` instead must meet an error rather than a wrong answer. The pages
    teach this by example rather than stating it, which is why it is asserted
    here.
    """
    problem = toy.exponential_decay()
    state = _state(8)
    assert not callable(problem)
    with pytest.raises(TypeError):
        run(
            state,
            problem,
            problem.y,
            problem.noise_cov,
            update_rule=SQUARE_ROOT,
            schedule=FixedSchedule.constant(1.0, n_steps=1),
        )


# ===========================================================================
# 5. claims that stopped holding
# ===========================================================================
#
# PR 7 moved the tutorials onto the new layers, and the prior draw changed
# with it, so every number on the pages moved. The numbers are re-pinned
# above; the claims below are qualitative readings of them that the new draw
# no longer supports. Each is kept as a strict xfail that names its sentence,
# for the prose rewrite (PR 11) to resolve: rewrite the sentence and delete
# the test, or change the example until the claim holds again, at which point
# the strict xfail fails and must be removed.


def _tracked_bridge_ratio():
    data = figures.bridge_tracked()[1]
    return data["cloud_sd"] / data["exact_sd"]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "tutorial 1, 'The ensemble tracks the distributions fairly well': at "
        "beta = 0.0998 the rate's spread is 0.73 of the exact one"
    ),
)
def test_8_the_tracked_bridge_tracks_every_other_level_within_ten_percent():
    """Tutorial 1: the ensemble is at most 10% too narrow at any level but one."""
    others = np.delete(_tracked_bridge_ratio(), 1, axis=0)
    assert others.min() > 0.9, others


@pytest.mark.xfail(
    strict=True,
    reason=(
        "tutorial 1, 'the curvature present at the second distribution "
        "presents a challenge': its rate ratio is 1.29, against 0.73 at "
        "beta = 0.0998, so it is no longer the worst-tracked level"
    ),
)
def test_8_the_tracked_bridge_second_level_is_the_exception():
    """Tutorial 1: the second level is over-dispersed by more than 40%."""
    ratio = _tracked_bridge_ratio()
    assert ratio[1, 1] > 1.4, ratio


def _tutorial_3_optimization_rows():
    problem = toy.exponential_decay()
    reference_sd = figures._tempered_moments(1.0)[1]
    rows = {}
    for tau in (2.0, 1.0):
        result = run(
            _state(),
            problem.forward,
            problem.y,
            problem.noise_cov,
            update_rule=SQUARE_ROOT,
            schedule=FixedSchedule.constant(1.0, n_steps=200),
            stop=DiscrepancyStop(tau=tau),
        )
        rows[tau] = (float(result.beta), _sd(result.ensemble) / reference_sd)
    return rows


@pytest.mark.xfail(
    strict=True,
    reason=(
        "tutorial 3, '`tau=2.0` ... on this problem, stops one step earlier' "
        "and 'Stopping one step earlier, at `tau=2`': both rules now stop at "
        "beta = 2, so the two table rows are identical"
    ),
)
def test_8_tau_2_stops_one_step_before_tau_1():
    """Tutorial 3: the discrepancy principle at ``tau=2`` stops one step earlier."""
    rows = _tutorial_3_optimization_rows()
    assert rows[2.0][0] == rows[1.0][0] - 1.0, rows


@pytest.mark.xfail(
    strict=True,
    reason=(
        "tutorial 3, 'Its spread happens to land within 13% of the "
        "target's': the `tau=1` row's spread is now [1.15, 2.94] of it"
    ),
)
def test_8_tau_1_spread_lands_within_13_percent_of_the_target():
    """Tutorial 3: the ``tau=1`` row's spread is within 13% of the target's."""
    _, ratio = _tutorial_3_optimization_rows()[1.0]
    assert np.abs(ratio - 1.0).max() < 0.13, ratio
