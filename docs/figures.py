"""The documentation's figures, and the Sphinx extension that generates them.

Every figure in the tutorials is produced by a function here and written into
``docs/_generated/figures`` when the documentation is built. Nothing is
committed: a figure cannot disagree with the code that made it, because it is
remade from that code on every build whose inputs have changed.

Contents
--------

============================== ===========================================
name                           page
============================== ===========================================
``01-prior-predictive``        :doc:`tutorials/01-first-inversion`
``01-one-step``                :doc:`tutorials/01-first-inversion`
``01-tempering-bridge``        :doc:`tutorials/01-first-inversion`
``01-bridge-tracked``          :doc:`tutorials/01-first-inversion`
``01-answer``                  :doc:`tutorials/01-first-inversion`
``02-trajectories``            :doc:`tutorials/02-reading-a-run`
``03-two-forms``               :doc:`tutorials/03-sampling-or-optimizing`
``04-one-step``                :doc:`tutorials/04-tempering-schedules`
``04-uniform``                 :doc:`tutorials/04-tempering-schedules`
``04-geometric``               :doc:`tutorials/04-tempering-schedules`
``04-adaptive``                :doc:`tutorials/04-tempering-schedules`
``04-refinement``              :doc:`tutorials/04-tempering-schedules`
============================== ===========================================

Each figure function takes no arguments, is deterministic, and returns the
figure together with a dictionary of the numbers it plotted. Every figure is
drawn in two themes, light and dark (``THEMES``), and a page shows the one
matching the reader's theme: :func:`add_dark_variants` pairs each figure with
its dark variant when the page is read, so a page names only the light one.
:func:`build` writes the figures; ``python docs/figures.py <dir>`` does the
same from the command line, for looking at one while writing a page.

Notes
-----
**How a figure is kept from rotting.** Two mechanisms, because the obvious one
is not enough on its own.

*The build regenerates it.* ``setup`` registers a ``builder-inited`` handler,
so a documentation build with warnings as errors fails on a figure whose code
raises. Regeneration is skipped when every output is newer than both this
module and every source file of the package, so an unchanged build is not
slowed by it; set ``ENSKIT_DOCS_FIGURES=force`` to override.

*A test pins the numbers.* ``tests/test_tutorials.py`` calls each function and
asserts the values in the returned dictionary, which is why the dictionary
exists. That catches the case a build cannot: a figure that draws the wrong
array, or a change in the library that moves the answer. The values are the
ones *plotted*, not values recomputed by the test, so drawing the wrong array
fails the test rather than merely looking wrong. Conformance obligation 26 of
:doc:`eki-contract` is the rule this satisfies.

Pixels are deliberately not compared. Two machines with different fonts
rasterize the same figure differently, so a pixel comparison would be either
flaky or vacuous, and it would not catch a wrong number in a legible plot.

**A grid, not a closed form.** :func:`enskit.toy.exponential_decay` has no
closed-form posterior, but it has two parameters, so its tempered densities
can be evaluated on a grid up to a constant, which is all a contour plot
needs. Grid quadrature also gives the reference moments the tutorials compare
against, through :func:`_tempered_moments`, which refines its box because no
single grid serves every level: one wide enough for the prior spans eight
units per parameter and resolves a target of standard deviation 0.037 with
about five points, while one sized for the target truncates the prior. After
two refinements the moments agree to six digits across three resolutions, and
at :math:`\\beta = 0` they recover the prior's own moments exactly, which is
the one level where the quadrature can be checked against a closed form.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import enskit  # noqa: E402, F401  -- enables float64 before any array exists
from enskit import kalman, toy  # noqa: E402
from enskit.algorithms import eki  # noqa: E402

#: Where a build writes the figures, relative to this file.
OUTPUT_DIR = Path(__file__).parent / "_generated" / "figures"

#: The problem the tutorials work. Deterministic in its arguments.
PROBLEM = toy.exponential_decay()

#: The index of :math:`t = 1` among the observation times. The figures return
#: the whole predictive spread; this names the entry the tests read off it.
JOINT_INDEX = 3

#: The update rule tutorial 1 runs with. On this problem the deterministic
#: square-root update reproduces the target's covariance but leaves the
#: ensemble strung along one direction, with the across-ridge variance carried
#: by two or three particles; the Matheron update draws a perturbation per
#: particle instead, which refills the cloud. The measurements are in
#: ``HANDOFF.md``. The page names the rule and defers the choice to
#: :doc:`tutorials/05-transform-or-pathwise`.
TUTORIAL_1_UPDATE = kalman.Matheron()

#: The update rule tutorials 2 to 4 run with: the deterministic square-root
#: update. A run takes its update rule explicitly; there is no default.
TUTORIALS_2_TO_4_UPDATE = kalman.SymmetricSquareRoot()

# Colors. Deliberately few, and the same meaning on every figure. The prior
# and everything derived from it before conditioning is blue; the answer, once
# the data has been used, is green; the observation is red; a competing ladder
# is purple and a collapsed fit indigo. Each figure is drawn once per theme:
# `_use_theme` rebinds these names from `THEMES`, so a figure function reads
# them when it is called, never in a default argument.
C_PRIOR = "#3d6f9e"
C_PRIOR_LIGHT = "#9dbcd8"
C_ENSEMBLE = "#12795a"
C_ALT = "#8a4fa8"
C_DATA = "#c8382b"
C_TRUTH = "#1c2126"
C_COLLAPSED = "#2f3d63"
C_TARGET = "#8f979f"   # a reference density: neither prior nor answer
C_TEXT = "#22282e"
C_AXIS = "#c2c8ce"
C_ARROW = "#7b848d"
C_PAPER = "#ffffff"    # the background, and the halo around a marker

#: The palette of each theme. ``light`` is the one above; ``dark`` keeps each
#: color's meaning and hue, lightened to read against Furo's dark background,
#: and swaps the text, axis and paper colors.
THEMES = {
    "light": {
        "C_PRIOR": C_PRIOR,
        "C_PRIOR_LIGHT": C_PRIOR_LIGHT,
        "C_ENSEMBLE": C_ENSEMBLE,
        "C_ALT": C_ALT,
        "C_DATA": C_DATA,
        "C_TRUTH": C_TRUTH,
        "C_COLLAPSED": C_COLLAPSED,
        "C_TARGET": C_TARGET,
        "C_TEXT": C_TEXT,
        "C_AXIS": C_AXIS,
        "C_ARROW": C_ARROW,
        "C_PAPER": C_PAPER,
    },
    "dark": {
        "C_PRIOR": "#74a7da",
        "C_PRIOR_LIGHT": "#4f7aa3",
        "C_ENSEMBLE": "#4cc79a",
        "C_ALT": "#c08fdb",
        "C_DATA": "#f0766a",
        "C_TRUTH": "#eceef0",
        "C_COLLAPSED": "#97a8de",
        "C_TARGET": "#9ba3ab",
        "C_TEXT": "#d9dde1",
        "C_AXIS": "#4d555c",
        "C_ARROW": "#8b949c",
        "C_PAPER": "#131416",
    },
}

#: Contour levels for a density scaled to a maximum of one. For a Gaussian
#: these are the one-, two- and three-standard-deviation ellipses.
LEVELS = (float(np.exp(-4.5)), float(np.exp(-2.0)), float(np.exp(-0.5)))


# ---------------------------------------------------------------------------
# shared machinery
# ---------------------------------------------------------------------------


def _use_theme(name: str) -> None:
    """Rebind the module's color names to the palette of theme ``name``."""
    globals().update(THEMES[name])


def _style() -> None:
    """Apply the shared look. Called once per figure, before any axes exist.

    Notes
    -----
    Two rules the earlier version of these figures broke. No grid, because a
    grid competes with contour lines and with a scatter of particles for the
    same ink. And no legend inside the data area: labels are placed against
    the thing they name, or below the axes, so that a legend cannot cover the
    part of the plot the caption is talking about.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "figure.facecolor": C_PAPER,
            "savefig.facecolor": C_PAPER,
            "axes.facecolor": C_PAPER,
            "font.size": 9.0,
            "text.color": C_TEXT,
            "axes.titlesize": 9.5,
            "axes.titlepad": 7.0,
            "axes.titlelocation": "left",
            "axes.labelsize": 9.0,
            "axes.labelpad": 4.0,
            "axes.labelcolor": C_TEXT,
            "axes.edgecolor": C_AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.color": C_AXIS,
            "ytick.color": C_AXIS,
            "xtick.labelcolor": C_TEXT,
            "ytick.labelcolor": C_TEXT,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.handletextpad": 0.6,
            "legend.borderaxespad": 0.0,
            "lines.linewidth": 1.4,
            "lines.solid_capstyle": "round",
        }
    )


def _tint(color, n=3):
    """A pale-to-``color`` colormap, for filling contours without shouting.

    The palest stop is a visible tint rather than the paper color: a
    ``contourf`` whose first band is the paper color leaves the outermost
    contour with no fill inside it, which reads as though the density stopped
    there.
    """
    from matplotlib.colors import LinearSegmentedColormap, to_rgb

    rgb, paper = np.asarray(to_rgb(color)), np.asarray(to_rgb(C_PAPER))
    stops = [tuple(rgb + (paper - rgb) * fade) for fade in (0.86, 0.55, 0.22)]
    return LinearSegmentedColormap.from_list("tint", stops, N=n)


def _label_below(ax, ncol=3, y=-0.30, handles=None):
    """Put the axes' legend under the axes, where it cannot cover anything.

    Pass ``handles`` when the panel draws something that carries no handle of
    its own, such as a contour set.
    """
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        columnspacing=1.4,
        handlelength=1.4,
    )


def _reference_line(ax, level, label):
    """A dashed reference level, labeled at its right-hand end.

    Labeled in place rather than in a legend: two panels carry different
    reference levels, and one shared entry would have to describe both.
    """
    ax.axhline(level, color=C_TARGET, lw=1.1, ls="--", zorder=1)
    ax.annotate(
        label,
        (1.0, level),
        xycoords=("axes fraction", "data"),
        textcoords="offset points",
        xytext=(-2, 4),
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=C_TARGET,
    )


def _log_terms(grid):
    """The prior log-density and the misfit of every point of ``grid``.

    Both terms are independent of the tempering level, so one call serves
    every level: the tempered log-density at ``beta`` is
    ``log_prior - beta * phi``, up to an additive constant.
    """
    log_prior = PROBLEM.prior.log_density(u=grid)
    phi = eki.misfits(PROBLEM.y, PROBLEM.forward(grid), PROBLEM.noise_cov)
    return np.asarray(log_prior), np.asarray(phi)


def _grid(box, n):
    """A ``(n, n)`` grid over ``box = (amp_lo, amp_hi, rate_lo, rate_hi)``."""
    amp = jnp.linspace(box[0], box[1], n)
    rate = jnp.linspace(box[2], box[3], n)
    A, R = jnp.meshgrid(amp, rate, indexing="ij")
    return np.asarray(amp), np.asarray(rate), jnp.stack([A.ravel(), R.ravel()], -1)


def _moments(grid, log_pi):
    """Quadrature moments of an unnormalized log-density on a grid.

    Trustworthy only where the grid holds essentially all of the mass — see
    the module's notes. Used to size panels, and to report a reference at
    ``beta = 1``.
    """
    weights = np.exp(log_pi - log_pi.max())
    weights /= weights.sum()
    points = np.asarray(grid)
    mean = (weights[:, None] * points).sum(axis=0)
    sd = np.sqrt((weights[:, None] * (points - mean) ** 2).sum(axis=0))
    return mean, sd


#: A box wide enough for the prior, where every level's mass search starts.
_WIDE_BOX = (-3.0, 5.0, -3.0, 5.0)


def _tempered_moments(beta, n=200, refinements=2):
    """Mean and standard deviation of the tempered density at ``beta``.

    A single grid cannot serve every level: the target at ``beta = 1`` is
    thirty times narrower than the prior, so a box wide enough for the prior
    resolves it with about five points across its width and reports a mean
    wrong in the fourth decimal. So the box is refined — a pass on a box wide
    enough for the prior locates the mass, and each further pass re-grids
    ``mean +/- 6 sd``. Two refinements converge at every level used here;
    ``tests/test_tutorials.py`` asserts that against three resolutions.
    """
    box = _WIDE_BOX
    for _ in range(refinements):
        _, _, grid = _grid(box, n)
        log_prior, phi = _log_terms(grid)
        mean, sd = _moments(grid, log_prior - beta * phi)
        box = _box_around(mean, sd, 6.0)
    return mean, sd


def _tempered_contours(beta, box, n=160):
    """The tempered density at ``beta`` on ``box``, scaled to a maximum of 1."""
    amp, rate, grid = _grid(box, n)
    log_prior, phi = _log_terms(grid)
    log_pi = log_prior - beta * phi
    density = np.exp(log_pi - log_pi.max()).reshape(n, n)
    return amp, rate, density


def _box_around(mean, sd, half_widths=3.6):
    """A plotting box of ``mean +/- half_widths * sd``, in the grid's order."""
    lo = mean - half_widths * sd
    hi = mean + half_widths * sd
    return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))


def _square_box(mean, sd, half_widths=3.5):
    """A plotting box with equal sides, centered on ``mean``.

    A row of panels drawn at equal aspect lines up only if their boxes have
    the same shape, so a sequence of them is sized from the larger of the two
    standard deviations and squared. The cost is some unused margin in the
    narrower coordinate; the alternative is panels of different heights.
    """
    half = half_widths * float(np.max(sd))
    return (
        float(mean[0] - half),
        float(mean[0] + half),
        float(mean[1] - half),
        float(mean[1] + half),
    )


def _draw_contours(ax, beta, box, n=160, color=None, fill=True):
    """Contours of the tempered density at ``beta``, on the axes' own box.

    Filled, faintly, with thin lines over the fill. The fill is what makes a
    density read as a density rather than as three unexplained rings, and the
    lines are still what a reader measures the ensemble against.
    """
    color = C_PRIOR_LIGHT if color is None else color
    amp, rate, density = _tempered_contours(beta, box, n)
    if fill:
        ax.contourf(
            amp,
            rate,
            density.T,
            levels=(*LEVELS, 1.0),
            cmap=_tint(color),
            alpha=0.3,
        )
    ax.contour(amp, rate, density.T, levels=LEVELS, colors=color, linewidths=0.7)
    # A contour set carries no legend handle of its own -- and reaching into
    # its artists to attach one broke on matplotlib 3.8, which removed
    # `QuadContourSet.collections`. Callers that need an entry build a proxy
    # with `_handle` instead.
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])


def _cloud(ax, ensemble, color=None, label=None, size=9.0, alpha=0.85):
    """Scatter an ensemble in the parameter plane, with a halo of the paper color.

    The halo is what keeps 64 overlapping particles legible as particles rather
    than as one blob.
    """
    color = C_ENSEMBLE if color is None else color
    e = np.asarray(ensemble)
    ax.scatter(
        e[:, 0],
        e[:, 1],
        s=size,
        c=color,
        alpha=alpha,
        linewidths=0.35,
        edgecolors=C_PAPER,
        label=label,
        zorder=3,
    )


def _mark_truth(ax, annotate=None):
    """The parameters the observation was generated from.

    ``annotate`` places the label against the marker rather than in a legend;
    pass an ``(dx, dy)`` offset in points, or ``None`` for a legend entry.
    """
    x, y = float(PROBLEM.u_true[0]), float(PROBLEM.u_true[1])
    ax.scatter(
        [x],
        [y],
        marker="X",
        s=42,
        c=C_TRUTH,
        linewidths=0.6,
        edgecolors=C_PAPER,
        zorder=6,
        label=None if annotate else "true parameters",
    )
    if annotate:
        ax.annotate(
            "true $u$",
            (x, y),
            textcoords="offset points",
            xytext=annotate,
            fontsize=8.5,
            color=C_TRUTH,
            zorder=6,
        )


def _prior_state(n_particles, seed=0):
    """An initial state drawn from the problem's prior."""
    return eki.EKIState.from_prior(
        jax.random.key(seed), PROBLEM.prior, n_particles=n_particles
    )


def _run(state, update_rule=TUTORIALS_2_TO_4_UPDATE, **kwargs):
    """``eki.run`` on the tutorials' problem."""
    return eki.run(
        state,
        PROBLEM.forward,
        PROBLEM.y,
        PROBLEM.noise_cov,
        update_rule=update_rule,
        **kwargs,
    )


def _ladder(n_particles, schedule, update_rule, seed=0):
    """Run a ladder, returning ``(levels, clouds, records)``.

    Each cloud is paired with the level of the evaluation it came from, read
    off ``Evaluation.beta`` so the pairing cannot drift. The last pair is the
    terminal ensemble, the state the generator yields last, at the level the
    run reached, which no evaluation covers: the final record's ``beta`` is
    the level *entering* the last step. ``records`` is the run's history.
    """
    state = _prior_state(n_particles, seed)
    levels, clouds, records = [], [], []
    for current, record, evaluation in eki.iterate(  # noqa: B007 -- read after
        state,
        PROBLEM.forward,
        PROBLEM.y,
        PROBLEM.noise_cov,
        update_rule=update_rule,
        schedule=schedule,
    ):
        levels.append(float(evaluation.beta))
        clouds.append(np.asarray(evaluation.ensemble["u"]))
        records.append(record)
    levels.append(float(current.beta))
    clouds.append(np.asarray(current.ensemble["u"]))
    return levels, clouds, records


def _sd(ensemble):
    """Per-coordinate standard deviation, with the package's divisor."""
    return np.asarray(ensemble).std(axis=0, ddof=1)


# ---------------------------------------------------------------------------
# tutorial 1
# ---------------------------------------------------------------------------


#: The prediction panels' limits, shared so that "before" and "after" compare.
_PREDICTION_YLIM = (-0.6, 2.4)


def _draw_data(ax, label="observations"):
    """The observation and its error bars, on a prediction-space axes."""
    noise_sd = float(np.sqrt(np.asarray(PROBLEM.noise_cov.diag())[0]))
    ax.errorbar(
        np.asarray(PROBLEM.times),
        np.asarray(PROBLEM.y),
        yerr=noise_sd,
        fmt="o",
        ms=3.4,
        mfc=C_PAPER,
        mew=1.1,
        color=C_DATA,
        ecolor=C_DATA,
        elinewidth=1.1,
        capsize=2.0,
        zorder=5,
        label=label,
    )


def _draw_curves(ax, predictions, color, alpha=0.3):
    """One line per particle, in prediction space. Returns the clipped count."""
    times = np.asarray(PROBLEM.times)
    predictions = np.asarray(predictions)
    for row in predictions:
        ax.plot(times, row, color=color, alpha=alpha, linewidth=0.7, zorder=2)
    ax.set_ylim(*_PREDICTION_YLIM)
    ax.set_xlim(0.0, 3.15)
    ax.set_xlabel("time $t$")
    ax.set_ylabel("prediction")
    lo, hi = _PREDICTION_YLIM
    return int(((predictions < lo) | (predictions > hi)).any(axis=1).sum())


def _arrow_between(fig, left, right, label):
    """Draw ``label`` over an arrow centered in the gap between two axes.

    The positions are read after a draw, so an axes whose aspect ratio was
    fixed — which shrinks it inside its gridspec cell — is still measured
    where it actually ends up rather than where it was allocated.
    """
    from matplotlib.patches import FancyArrowPatch

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()
    # The tight bounding box, so that the arrow clears the axis labels rather
    # than only the axes rectangle -- the right panel's y-label sits outside
    # its own box, and a gap measured without it runs the arrow through.
    gap_lo = left.get_tightbbox(renderer).transformed(inverse).x1
    gap_hi = right.get_tightbbox(renderer).transformed(inverse).x0
    center = 0.5 * (gap_lo + gap_hi)
    half = 0.36 * (gap_hi - gap_lo)
    y = 0.5 * (right.get_position().y0 + right.get_position().y1)
    fig.add_artist(
        FancyArrowPatch(
            (center - half, y),
            (center + half, y),
            transform=fig.transFigure,
            arrowstyle="simple,head_width=5.5,head_length=7,tail_width=1.4",
            facecolor=C_ARROW,
            edgecolor="none",
            mutation_scale=1.0,
        )
    )
    fig.text(
        center,
        y + 0.045,
        label,
        ha="center",
        va="bottom",
        fontsize=11.5,
        color=C_TEXT,
    )


def _figure_legend(fig, handles, ncol, y=-0.14):
    """One legend for the whole figure, below every panel."""
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        columnspacing=1.5,
        handlelength=1.5,
        handletextpad=0.5,
    )


def _handle(kind, color, label, **kwargs):
    """A legend handle drawn to match how the figure draws the thing."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if kind == "patch":
        return Patch(facecolor=color, edgecolor="none", alpha=0.25, label=label)
    if kind == "line":
        return Line2D([], [], color=color, lw=1.4, label=label, **kwargs)
    if kind == "dashed":
        return Line2D([], [], color=color, lw=1.1, ls="--", label=label)
    if kind == "marker+line":
        return Line2D(
            [],
            [],
            color=color,
            lw=1.4,
            marker=kwargs.pop("marker", "o"),
            markerfacecolor=color,
            markeredgecolor=C_PAPER,
            markeredgewidth=0.4,
            markersize=4.0,
            label=label,
        )
    if kind == "observation":
        # Drawn as a line in one panel and as markers in another, so the
        # handle carries both rather than misrepresenting either.
        return Line2D(
            [],
            [],
            color=color,
            lw=1.1,
            marker="o",
            markerfacecolor=C_PAPER,
            markeredgecolor=color,
            markeredgewidth=1.1,
            markersize=5.0,
            label=label,
        )
    marker = kwargs.pop("marker", "o")
    return Line2D(
        [],
        [],
        marker=marker,
        linestyle="none",
        markerfacecolor=kwargs.pop("mfc", color),
        markeredgecolor=kwargs.pop("mec", C_PAPER),
        markeredgewidth=kwargs.pop("mew", 0.6),
        markersize=kwargs.pop("ms", 5.0),
        label=label,
        **kwargs,
    )


def _two_space_figure(
    *,
    ensemble,
    predictions,
    density_beta,
    box,
    titles,
    cloud_color,
    density_color,
    legend,
    ncol,
    equal_aspect=False,
):
    """A parameter-space panel, the forward model as an arrow, prediction space.

    Tutorial 1 draws this layout twice — before conditioning and after — and
    the pair only makes its point if the two are otherwise identical, so the
    layout is built once here rather than assembled twice.

    Returns the figure and the number of prediction curves that leave the
    right-hand panel.
    """
    fig = plt.figure(figsize=(7.0, 2.95))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 0.26, 1.15), wspace=0.34)
    left, middle, right = (fig.add_subplot(cell) for cell in grid)

    _draw_contours(left, density_beta, box, color=density_color)
    _cloud(left, ensemble, cloud_color)
    _mark_truth(left)
    if equal_aspect:
        left.set_aspect("equal")
        left.set_anchor("N")  # so a square panel's title still lines up
    left.set_xlabel("amplitude $a$")
    left.set_ylabel(r"decay rate $\lambda$")
    left.set_title(titles[0])

    middle.axis("off")

    clipped = _draw_curves(right, predictions, cloud_color, alpha=0.3)
    _draw_data(right, label=None)
    right.set_title(titles[1])

    _arrow_between(fig, left, right, r"$\mathcal{G}$")
    _figure_legend(fig, legend, ncol=ncol)
    return fig, clipped


def prior_predictive():
    """The prior, and the prior predictive distribution it induces.

    Two panels with the forward model as an arrow between them: the prior over
    the two parameters with an ensemble drawn from it, and the predictions
    those particles produce, against the observation.
    """
    _style()
    n_particles = 64
    state = _prior_state(n_particles)
    ensemble = np.asarray(state.ensemble["u"])
    predictions = np.asarray(PROBLEM.forward(state.ensemble["u"]))

    prior_mean = np.asarray(PROBLEM.prior.mean("u"))
    prior_sd = np.sqrt(np.asarray(PROBLEM.prior.cov("u").diag()))
    fig, clipped = _two_space_figure(
        ensemble=ensemble,
        predictions=predictions,
        density_beta=0.0,
        box=_box_around(prior_mean, prior_sd, 3.1),
        titles=("Prior distribution", "Prior predictive distribution"),
        cloud_color=C_PRIOR,
        density_color=C_PRIOR_LIGHT,
        legend=[
            _handle("patch", C_PRIOR_LIGHT, "prior density"),
            _handle("marker", C_PRIOR, "particles"),
            _handle("marker", C_TRUTH, "true $u$", marker="X", ms=6.0),
            _handle("line", C_PRIOR, "predictions", alpha=0.55),
            _handle("observation", C_DATA, "observation $y$"),
        ],
        ncol=5,
        equal_aspect=True,  # the prior's contours are circles; show them so
    )

    return fig, {
        "n_particles": n_particles,
        "prior_curves_leaving_panel": clipped,
        "prior_negative_rates": int((ensemble[:, 1] <= 0.0).sum()),
        "prior_predictive_sd": predictions.std(axis=0, ddof=1),
        "prediction_ylim": np.asarray(_PREDICTION_YLIM),
        "noise_sd": float(np.sqrt(np.asarray(PROBLEM.noise_cov.diag())[0])),
    }


def one_step():
    """After one approximate conditioning step, in the same two spaces.

    The direct counterpart of :func:`prior_predictive`: the same layout, the
    same axes in prediction space, with the ensemble the step produced in
    place of the prior's. The contours are the *true* posterior, so how well
    the particles track it is the thing the figure shows.
    """
    _style()
    n_particles = 64
    state = _prior_state(n_particles)
    updated = _run(
        state,
        TUTORIAL_1_UPDATE,
        schedule=eki.FixedSchedule.constant(1.0, n_steps=1),
    )
    ensemble = np.asarray(updated.ensemble["u"])
    predictions = np.asarray(PROBLEM.forward(updated.ensemble["u"]))

    # The panel is sized to hold the ensemble, not the posterior. Sizing it to
    # the posterior would put most of the particles outside it: after one step
    # the ensemble is about twenty times wider in the rate than the target.
    posterior_mean, posterior_sd = _tempered_moments(1.0)
    fig, clipped = _two_space_figure(
        ensemble=ensemble,
        predictions=predictions,
        density_beta=1.0,
        box=_box_around(ensemble.mean(axis=0), _sd(ensemble), 2.9),
        titles=("Posterior distribution", "Posterior predictive distribution"),
        cloud_color=C_ENSEMBLE,
        density_color=C_TARGET,
        legend=[
            _handle("patch", C_TARGET, "true posterior density"),
            _handle("marker", C_ENSEMBLE, "particles"),
            _handle("marker", C_TRUTH, "true $u$", marker="X", ms=6.0),
            _handle("line", C_ENSEMBLE, "predictions", alpha=0.55),
            _handle("observation", C_DATA, "observation $y$"),
        ],
        ncol=5,
    )

    return fig, {
        "n_particles": n_particles,
        "one_step_mean": np.asarray(updated.mean("u")),
        "one_step_sd": _sd(ensemble),
        "one_step_predictive_sd": predictions.std(axis=0, ddof=1),
        "one_step_curves_leaving_panel": clipped,
        "posterior_mean": posterior_mean,
        "posterior_sd": posterior_sd,
    }


#: The tempering levels tutorial 1's closing figure draws, prior to posterior.
BRIDGE_LEVELS = (0.0, 0.001, 0.01, 0.1, 1.0)


def _blend(start, end, fraction):
    """A color ``fraction`` of the way from ``start`` to ``end``."""
    from matplotlib.colors import to_hex, to_rgb

    a, b = np.asarray(to_rgb(start)), np.asarray(to_rgb(end))
    return to_hex(a + (b - a) * fraction)


def tempering_bridge():
    """The exact tempered distributions bridging the prior to the posterior.

    One panel per level, contours only — no ensemble, which is
    :doc:`tutorials/04-tempering-schedules`' subject. Each panel is scaled to
    its own level, because the distribution narrows by a factor of about
    thirty along the way and fixed limits would reduce the last panels to a
    dot. The dashed rectangle in each panel is the next panel's extent, so the
    zoom is visible rather than something to infer from the tick labels.
    """
    _style()
    levels = BRIDGE_LEVELS
    fig, axes = plt.subplots(
        1, len(levels), figsize=(7.4, 1.72), layout="constrained"
    )

    moments = [_tempered_moments(beta) for beta in levels]
    boxes = [_square_box(mean, sd, 3.4) for mean, sd in moments]
    colors = [
        _blend(C_PRIOR, C_ENSEMBLE, index / (len(levels) - 1))
        for index in range(len(levels))
    ]

    for index, (ax, beta) in enumerate(zip(axes, levels, strict=True)):
        _draw_contours(ax, beta, boxes[index], color=colors[index])
        if index + 1 < len(levels):  # where the next panel zooms to
            lo_a, hi_a, lo_r, hi_r = boxes[index + 1]
            ax.add_patch(
                plt.Rectangle(
                    (lo_a, lo_r),
                    hi_a - lo_a,
                    hi_r - lo_r,
                    fill=False,
                    edgecolor=C_AXIS,
                    linewidth=0.8,
                    linestyle=(0, (2.5, 1.8)),
                    zorder=4,
                )
            )
        ax.set_title(rf"$\beta = {beta:g}$", loc="center")
        ax.tick_params(labelsize=7.0)
        ax.locator_params(nbins=4)
        # Equal aspect at every level, so that the shapes the caption talks
        # about -- a curved ridge early, a tilted ellipse late -- are the
        # shapes the distributions have rather than artifacts of the panel.
        ax.set_aspect("equal")

    # constrained layout, so that the shared axis labels are placed against
    # the panels rather than floating below them.
    fig.supxlabel("amplitude $a$", fontsize=9.0)
    fig.supylabel(r"decay rate $\lambda$", fontsize=9.0)
    return fig, {
        "levels": np.asarray(levels),
        "sd": np.asarray([sd for _, sd in moments]),
        "mean": np.asarray([mean for mean, _ in moments]),
    }


def bridge_tracked():
    """The bridge again, with the ensemble the run actually produced on it.

    One panel per level of the default adaptive ladder, exact contours in the
    reference gray with that level's particles over them. The question the
    figure answers is how well the ensemble tracks the distribution it is
    meant to represent, so the panels are scaled to hold both.
    """
    _style()
    n_particles = 64
    levels, clouds, _ = _ladder(
        n_particles, eki.AdaptiveESSSchedule(), TUTORIAL_1_UPDATE
    )
    fig, moments, outside = _tracked_panels(
        levels, clouds, n_cols=5, figsize=(7.4, 3.5)
    )
    return fig, {
        "n_particles": n_particles,
        "levels": np.asarray(levels),
        "cloud_sd": np.asarray([c.std(axis=0, ddof=1) for c in clouds]),
        "exact_sd": np.asarray([sd for _, sd in moments]),
        "cloud_mean": np.asarray([c.mean(axis=0) for c in clouds]),
        "exact_mean": np.asarray([mean for mean, _ in moments]),
        "particles_outside_panel": np.asarray(outside),
    }


def _tracked_panels(levels, clouds, *, n_cols, figsize):
    """One panel per level: exact contours, with that level's particles over them.

    Each panel is scaled to hold both the exact distribution and the cloud,
    which at a level the ensemble tracks badly is the wider of the two, and
    carries a dashed rectangle marking the extent of the panel after it. The
    first spare cell carries the legend.

    Returns
    -------
    tuple
        ``(fig, moments, outside)``: the figure, the exact ``(mean, sd)`` at
        each level, and the number of particles each panel leaves outside its
        box.
    """
    moments = [_tempered_moments(beta) for beta in levels]
    boxes = [
        _square_box(mean, np.maximum(sd, cloud.std(axis=0, ddof=1)), 3.5)
        for (mean, sd), cloud in zip(moments, clouds, strict=True)
    ]
    # Several rows, because seven panels in one are too small to read. The
    # spare cell carries the legend, which is otherwise one more thing
    # competing for the width.
    n_rows = -(-(len(levels) + 1) // n_cols)  # +1, for the legend's own cell
    fig, grid = plt.subplots(n_rows, n_cols, figsize=figsize, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.09, hspace=0.02)
    axes = grid.ravel()
    outside = []
    for index, (ax, beta, cloud) in enumerate(
        zip(axes, levels, clouds, strict=False)
    ):
        box = boxes[index]
        _draw_contours(ax, beta, box, color=C_TARGET)
        _cloud(ax, cloud, C_ENSEMBLE, size=5.0, alpha=0.9)
        if index + 1 < len(levels):
            lo_a, hi_a, lo_r, hi_r = boxes[index + 1]
            ax.add_patch(
                plt.Rectangle(
                    (lo_a, lo_r),
                    hi_a - lo_a,
                    hi_r - lo_r,
                    fill=False,
                    edgecolor=C_AXIS,
                    linewidth=0.8,
                    linestyle=(0, (2.5, 1.8)),
                    zorder=5,
                )
            )
        ax.set_title(rf"$\beta = {beta:.2g}$", loc="center", fontsize=8.5)
        ax.tick_params(labelsize=6.0)
        ax.locator_params(nbins=3)
        ax.set_aspect("equal")
        outside.append(
            int(
                (
                    (cloud[:, 0] < box[0])
                    | (cloud[:, 0] > box[1])
                    | (cloud[:, 1] < box[2])
                    | (cloud[:, 1] > box[3])
                ).sum()
            )
        )

    # The legend goes in the first spare cell and any others are blanked. A
    # ladder long enough to need another row would otherwise draw one legend
    # per leftover cell.
    for index, spare in enumerate(axes[len(levels) :]):
        spare.axis("off")
        if index == 0:
            spare.legend(
                handles=[
                    _handle("patch", C_TARGET, "exact distribution"),
                    _handle("marker", C_ENSEMBLE, "particles"),
                ],
                loc="center",
                handlelength=1.5,
            )

    fig.supxlabel("amplitude $a$", fontsize=9.0)
    fig.supylabel(r"decay rate $\lambda$", fontsize=9.0)
    return fig, moments, outside


def answer():
    """The finished run, in the same two spaces as the two figures before it.

    The counterpart of :func:`one_step`, panel for panel and axis for axis, so
    that the whole ladder can be compared against the single step directly.
    """
    _style()
    n_particles = 64
    state = _prior_state(n_particles)
    result = _run(state, TUTORIAL_1_UPDATE, schedule=eki.AdaptiveESSSchedule())
    ensemble = np.asarray(result.ensemble["u"])
    predictions = np.asarray(PROBLEM.forward(result.ensemble["u"]))

    posterior_mean, posterior_sd = _tempered_moments(1.0)
    fig, clipped = _two_space_figure(
        ensemble=ensemble,
        predictions=predictions,
        density_beta=1.0,
        box=_box_around(ensemble.mean(axis=0), _sd(ensemble), 2.9),
        titles=("Posterior distribution", "Posterior predictive distribution"),
        cloud_color=C_ENSEMBLE,
        density_color=C_TARGET,
        legend=[
            _handle("patch", C_TARGET, "true posterior density"),
            _handle("marker", C_ENSEMBLE, "particles"),
            _handle("marker", C_TRUTH, "true $u$", marker="X", ms=6.0),
            _handle("line", C_ENSEMBLE, "predictions", alpha=0.55),
            _handle("observation", C_DATA, "observation $y$"),
        ],
        ncol=5,
    )

    return fig, {
        "n_particles": n_particles,
        "status": result.status,
        "n_evaluations": result.n_evaluations,
        "mean": np.asarray(result.mean("u")),
        "sd": _sd(ensemble),
        "predictive_sd": predictions.std(axis=0, ddof=1),
        "curves_leaving_panel": clipped,
        "posterior_mean": posterior_mean,
        "posterior_sd": posterior_sd,
    }


# ---------------------------------------------------------------------------
# tutorial 2
# ---------------------------------------------------------------------------


def trajectories():
    """Misfit, effective sample size and spread, for two ladders."""
    _style()
    n_particles = 64
    state = _prior_state(n_particles)
    runs = {}
    for label, schedule in (
        ("adaptive", eki.AdaptiveESSSchedule()),
        ("three equal steps", eki.FixedSchedule.uniform(3)),
    ):
        runs[label] = _run(state, schedule=schedule)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    styles = {
        "adaptive": dict(color=C_ENSEMBLE, marker="o"),
        "three equal steps": dict(color=C_ALT, marker="s"),
    }
    panels = [
        ("misfit_mean", "Mean misfit", True),
        ("ess", "Effective sample size", False),
        ("spread", "Ensemble spread", True),
    ]
    for ax, (field, title, log) in zip(axes, panels, strict=True):
        for label, result in runs.items():
            history = result.stacked
            ax.plot(
                np.asarray(history.step),
                np.asarray(getattr(history, field)),
                markersize=3.4,
                markeredgecolor=C_PAPER,
                markeredgewidth=0.4,
                **styles[label],
            )
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_title(title)

    # The two reference levels are labeled where they are drawn: one legend
    # covering both would have to give them the same entry, and they mean
    # different things in different panels.
    _reference_line(axes[0], PROBLEM.data_dim / 2, "$N/2$")
    _reference_line(axes[1], n_particles / 2, "$J/2$")

    fig.tight_layout()
    _figure_legend(
        fig,
        [
            _handle("marker+line", C_ENSEMBLE, "adaptive ladder", marker="o"),
            _handle("marker+line", C_ALT, "three equal steps", marker="s"),
            _handle("dashed", C_TARGET, "reference level"),
        ],
        ncol=3,
        y=-0.08,
    )
    return fig, {
        "n_particles": n_particles,
        "noise_floor": PROBLEM.data_dim / 2,
        **{
            f"{label.replace(' ', '_')}_{field}": np.asarray(
                getattr(result.stacked, field)
            )
            for label, result in runs.items()
            for field in ("beta", "ess", "misfit_mean", "spread")
        },
        **{
            f"{label.replace(' ', '_')}_sd": _sd(result.ensemble["u"])
            for label, result in runs.items()
        },
    }


# ---------------------------------------------------------------------------
# tutorial 3
# ---------------------------------------------------------------------------


def two_forms():
    """The two destinations: a posterior ensemble, and a collapsing fit."""
    _style()
    n_particles = 64
    state = _prior_state(n_particles)
    sampled = _run(state, schedule=eki.AdaptiveESSSchedule())
    stopped = _run(
        state,
        schedule=eki.FixedSchedule.constant(1.0, n_steps=200),
        stop=eki.DiscrepancyStop(tau=1.0),
    )
    unstopped = _run(state, schedule=eki.FixedSchedule.constant(1.0, n_steps=30))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))

    ax = axes[0]
    post_mean, post_sd = _tempered_moments(1.0)
    box = _box_around(post_mean, post_sd, 4.5)
    # The target density is drawn in the reference gray rather than in either
    # form's own color: it is what both are being measured against.
    _draw_contours(ax, 1.0, box, color=C_TARGET)
    _cloud(ax, sampled.ensemble["u"], C_ENSEMBLE)
    _cloud(ax, stopped.ensemble["u"], C_ALT)
    _cloud(ax, unstopped.ensemble["u"], C_COLLAPSED, size=5.0)
    _mark_truth(ax)
    ax.set_xlabel("amplitude $a$")
    ax.set_ylabel(r"decay rate $\lambda$")
    ax.set_title("Where each form ends up")
    _label_below(
        ax,
        ncol=2,
        handles=[
            _handle("line", C_TARGET, "the posterior"),
            _handle("marker", C_TRUTH, "true $u$", marker="X", ms=6.0),
            _handle("marker", C_ENSEMBLE, "sampling form"),
            _handle("marker", C_ALT, "optimization, stopped"),
            _handle("marker", C_COLLAPSED, r"the same, at $\beta = 30$", ms=4.0),
        ],
    )

    ax = axes[1]
    history = unstopped.stacked
    ax.plot(
        np.asarray(history.beta),
        np.asarray(history.spread),
        color=C_ALT,
        marker="s",
        markersize=3.0,
        markeredgecolor=C_PAPER,
        markeredgewidth=0.4,
        label="optimization form",
    )
    ax.axhline(
        float(np.mean(_sd(sampled.ensemble["u"]))),
        color=C_ENSEMBLE,
        lw=1.2,
        ls="--",
        label=r"sampling form, at $\beta = 1$",
    )
    ax.axvline(
        float(stopped.beta),
        color=C_DATA,
        lw=1.1,
        ls=":",
        label=f"stopping rule fires, $\\beta = {float(stopped.beta):.0f}$",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"tempering level $\beta$")
    ax.set_ylabel("ensemble spread")
    ax.set_title("The fit keeps collapsing")
    _label_below(ax, ncol=1)

    fig.tight_layout()
    return fig, {
        "n_particles": n_particles,
        "sampled_mean": np.asarray(sampled.mean("u")),
        "sampled_sd": _sd(sampled.ensemble["u"]),
        "stopped_status": stopped.status,
        "stopped_beta": float(stopped.beta),
        "stopped_evaluations": stopped.n_evaluations,
        "stopped_mean": np.asarray(stopped.mean("u")),
        "stopped_sd": _sd(stopped.ensemble["u"]),
        "unstopped_beta": float(unstopped.beta),
        "unstopped_sd": _sd(unstopped.ensemble["u"]),
        "unstopped_misfit": np.asarray(unstopped.stacked.misfit_mean),
        "reference_mean": post_mean,
        "reference_sd": post_sd,
    }


# ---------------------------------------------------------------------------
# tutorial 4
# ---------------------------------------------------------------------------

#: The number of steps of every tutorial 4 ladder but the single step: what
#: the adaptive ladder chooses on its own, so that the four compare at the
#: same cost in forward evaluations.
LADDER_STEPS = 6

#: The growth factor of tutorial 4's geometric ladder, from one increment to
#: the next.
GEOMETRIC_RATIO = 3.0

#: The uniform ladders of tutorial 4's refinement figure, by number of steps.
REFINEMENT_STEPS = (1, 2, 3, 6, 12, 24, 48, 96)


def geometric_increments(n_steps=LADDER_STEPS, ratio=GEOMETRIC_RATIO):
    """Increments growing by ``ratio`` per step and summing to one.

    Computed by the expression tutorial 4 prints, so the page's ladder and
    the figures' are the same floats.
    """
    growth = ratio ** jnp.arange(n_steps)
    return tuple(float(d) for d in growth / growth.sum())


#: Tutorial 4's ladders, in the page's order.
TUTORIAL_4_LADDERS = {
    "one-step": lambda: eki.FixedSchedule.constant(1.0, n_steps=1),
    "uniform": lambda: eki.FixedSchedule.uniform(LADDER_STEPS),
    "geometric": lambda: eki.FixedSchedule(geometric_increments()),
    "adaptive": lambda: eki.AdaptiveESSSchedule(),
    "misfit": lambda: eki.AdaptiveMisfitSchedule(),
}


def ladder_one_step():
    """A single unit step: the prior's particles, and where one update puts them."""
    return _tutorial_4_ladder("one-step", figsize=(7.4, 2.25))


def ladder_uniform():
    """Six equal steps, tracked against the tempered distribution at each level."""
    return _tutorial_4_ladder("uniform", figsize=(7.4, 4.1))


def ladder_geometric():
    """Six steps growing threefold, tracked against the tempered distribution."""
    return _tutorial_4_ladder("geometric", figsize=(7.4, 4.1))


def ladder_adaptive():
    """The ESS-adaptive ladder, tracked against the tempered distribution."""
    return _tutorial_4_ladder("adaptive", figsize=(7.4, 4.1))


def refinement():
    """The error at the posterior against the number of forward evaluations.

    Uniform ladders of every length in ``REFINEMENT_STEPS`` as a line, and the
    page's six-step ladders as points at their own cost, the two adaptive ones
    nudged apart. Two panels: the error
    in the mean, and the spread in excess of the target's in the decay rate,
    which is the worse-matched parameter in every run drawn.
    """
    _style()
    n_particles = 64
    state = _prior_state(n_particles)
    reference_mean, reference_sd = _tempered_moments(1.0)

    def measure(schedule):
        result = _run(state, TUTORIALS_2_TO_4_UPDATE, schedule=schedule)
        ensemble = np.asarray(result.ensemble["u"])
        error = float(np.abs(ensemble.mean(axis=0) - reference_mean).max())
        return result.n_evaluations, error, _sd(ensemble) / reference_sd

    uniform = [measure(eki.FixedSchedule.uniform(n)) for n in REFINEMENT_STEPS]
    others = {
        name: measure(TUTORIAL_4_LADDERS[name]())
        for name in ("geometric", "adaptive", "misfit")
    }
    styles = {
        "geometric": dict(color=C_ALT, marker="D", label="geometric, ratio 3"),
        "adaptive": dict(color=C_ENSEMBLE, marker="o", label="adaptive (ESS)"),
        "misfit": dict(
            color=C_ENSEMBLE, marker="^", label="adaptive (misfit)", mfc=C_PAPER
        ),
    }

    # The two adaptive points nearly coincide, so they are drawn a little to
    # either side of their true cost; the caption says so. The data returned
    # are the true costs.
    nudge = {"geometric": 1.0, "adaptive": 0.93, "misfit": 1.07}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    panels = [
        (lambda m: m[1], "Error in the mean"),
        (lambda m: m[2][1] - 1.0, "Excess spread in the decay rate"),
    ]
    for ax, (value, title) in zip(axes, panels, strict=True):
        ax.plot(
            [m[0] for m in uniform],
            [value(m) for m in uniform],
            color=C_ALT,
            marker="s",
            markersize=3.4,
            markeredgecolor=C_PAPER,
            markeredgewidth=0.4,
        )
        for name, measured in others.items():
            style = dict(styles[name])
            style.pop("label")
            ax.plot(
                [measured[0] * nudge[name]],
                [value(measured)],
                linestyle="none",
                markersize=5.5,
                markeredgewidth=1.0,
                markerfacecolor=style.pop("mfc", style["color"]),
                markeredgecolor=style["color"],
                zorder=4,
                **style,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([1, 3, 6, 12, 24, 48, 96])
        ax.set_xticklabels(["1", "3", "6", "12", "24", "48", "96"])
        ax.minorticks_off()
        ax.set_xlabel("forward evaluations")
        ax.set_title(title)

    fig.tight_layout()
    _figure_legend(
        fig,
        [
            _handle("marker+line", C_ALT, "uniform ladders", marker="s"),
            *[
                _handle(
                    "marker",
                    styles[name]["color"],
                    styles[name]["label"],
                    marker=styles[name]["marker"],
                    mfc=styles[name].get("mfc", styles[name]["color"]),
                    mec=styles[name]["color"],
                    mew=1.0,
                )
                for name in others
            ],
        ],
        ncol=4,
        y=-0.10,
    )
    return fig, {
        "n_particles": n_particles,
        "uniform_steps": np.asarray(REFINEMENT_STEPS),
        "uniform_evaluations": np.asarray([m[0] for m in uniform]),
        "uniform_mean_error": np.asarray([m[1] for m in uniform]),
        "uniform_sd_ratio": np.asarray([m[2] for m in uniform]),
        **{f"{name}_evaluations": m[0] for name, m in others.items()},
        **{f"{name}_mean_error": m[1] for name, m in others.items()},
        **{f"{name}_sd_ratio": m[2] for name, m in others.items()},
    }


def _tutorial_4_ladder(name, *, figsize):
    """One of tutorial 4's ladders, as tracked panels, with what the run recorded."""
    _style()
    n_particles = 64
    levels, clouds, records = _ladder(
        n_particles, TUTORIAL_4_LADDERS[name](), TUTORIALS_2_TO_4_UPDATE
    )
    fig, moments, outside = _tracked_panels(levels, clouds, n_cols=4, figsize=figsize)
    return fig, {
        "n_particles": n_particles,
        "levels": np.asarray(levels),
        "ess": np.asarray([float(record.ess) for record in records]),
        "cloud_sd": np.asarray([c.std(axis=0, ddof=1) for c in clouds]),
        "exact_sd": np.asarray([sd for _, sd in moments]),
        "cloud_mean": np.asarray([c.mean(axis=0) for c in clouds]),
        "exact_mean": np.asarray([mean for mean, _ in moments]),
        "particles_outside_panel": np.asarray(outside),
    }


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------

#: Every figure, by the file name it is written under.
FIGURES: dict[str, Callable[[], tuple[plt.Figure, dict]]] = {
    "01-prior-predictive": prior_predictive,
    "01-one-step": one_step,
    "01-tempering-bridge": tempering_bridge,
    "01-bridge-tracked": bridge_tracked,
    "01-answer": answer,
    "02-trajectories": trajectories,
    "03-two-forms": two_forms,
    "04-one-step": ladder_one_step,
    "04-uniform": ladder_uniform,
    "04-geometric": ladder_geometric,
    "04-adaptive": ladder_adaptive,
    "04-refinement": refinement,
}


def build(output_dir=OUTPUT_DIR, *, only=None) -> dict[str, dict]:
    """Write the figures as PNG files, and return what each one plotted.

    Each figure is written twice, as ``<name>.png`` in the light theme and
    ``<name>-dark.png`` in the dark one, and the returned data are the light
    figure's. The two draw the same numbers; only the palette differs.

    Parameters
    ----------
    output_dir
        The directory to write into. Created if it does not exist.
    only
        A figure name, or an iterable of them. Defaults to all of them.

    Returns
    -------
    dict
        The data dictionary of each figure built, by name.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if only is None:
        names = list(FIGURES)
    elif isinstance(only, str):
        names = [only]
    else:
        names = list(only)

    plotted = {}
    for name in names:
        for theme, suffix in (("dark", "-dark"), ("light", "")):
            _use_theme(theme)
            figure, data = FIGURES[name]()
            figure.savefig(output_dir / f"{name}{suffix}.png")
            plt.close(figure)
        plotted[name] = data
    return plotted


def _newest_source_time() -> float:
    """The most recent modification time of this module and the package."""
    package = Path(enskit.__file__).parent
    sources = [Path(__file__), *package.rglob("*.py")]
    return max(path.stat().st_mtime for path in sources)


def is_current(output_dir=OUTPUT_DIR) -> bool:
    """Whether every figure exists and postdates every source file."""
    output_dir = Path(output_dir)
    paths = [
        output_dir / f"{name}{suffix}.png" for name in FIGURES for suffix in ("", "-dark")
    ]
    if not all(path.exists() for path in paths):
        return False
    return min(path.stat().st_mtime for path in paths) > _newest_source_time()


def add_dark_variants(document) -> None:
    """Pair every generated figure in ``document`` with its dark variant.

    Each image whose path is under ``_generated/figures/`` gets Furo's
    ``only-light`` class, and a copy pointing at ``<name>-dark.png``, with the
    ``only-dark`` class, is inserted after it. A page writes one ``figure``
    directive per figure, and the theme the reader chose decides which image
    is shown.
    """
    from docutils import nodes

    for image in list(document.findall(nodes.image)):
        uri = image["uri"]
        generated = "_generated/figures/" in uri and uri.endswith(".png")
        if not generated or uri.endswith("-dark.png"):
            continue
        dark = image.deepcopy()
        dark["uri"] = uri.removesuffix(".png") + "-dark.png"
        image["classes"].append("only-light")
        dark["classes"].append("only-dark")
        image.parent.insert(image.parent.index(image) + 1, dark)


def setup(app):
    """Register the build-time hooks. Called by Sphinx; see the module notes."""
    from sphinx.transforms import SphinxTransform

    class DarkVariants(SphinxTransform):
        default_priority = 500

        def apply(self, **kwargs):
            # only HTML honors Furo's theme classes; elsewhere both would show
            if self.app.builder.format == "html":
                add_dark_variants(self.document)

    def generate(_app):
        force = os.environ.get("ENSKIT_DOCS_FIGURES") == "force"
        if not force and is_current():
            return
        build()

    app.connect("builder-inited", generate)
    app.add_transform(DarkVariants)
    return {"version": enskit.__version__, "parallel_read_safe": True}


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_DIR
    for name, data in build(target).items():
        print(f"{name}: {target / (name + '.png')}")
        for key, value in data.items():
            print(f"    {key} = {value}")
