r"""The conformance checks for the algorithm layer's policies."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..algorithms.eki import PREDICTION, Evaluation
from ..distribution import Ensemble

__all__ = ["check_schedule", "check_inflation", "check_relaxation", "check_stopping_rule"]

#: A keyword no policy knows, passed to check that unknown context is ignored.
_UNKNOWN_CONTEXT = "conformance_check_context"


def check_schedule(schedule, evaluation: Evaluation | None = None) -> None:
    r"""Check a :class:`~enskit.algorithms.eki.Schedule` against its protocol.

    Checks, in order:

    1. ``n_steps`` is ``None`` or a positive ``int``, and ``beta_target``
       ``None`` or a positive real number, each the same on a second read;
    2. ``next_increment`` returns ``None`` or a finite, strictly positive
       scalar;
    3. it is pure: a second call on the same evaluation returns the same
       value, bit for bit;
    4. it respects its own ``beta_target``: on an evaluation a quarter of the
       budget short of it, the increment is at most what remains;
    5. a ``nan`` misfit is never turned into a step: on an evaluation with a
       ``nan`` whitened residual, the result is ``None``, not finite, or the
       increment of the clean evaluation (a schedule that does not read the
       misfits).

    Parameters
    ----------
    schedule : Schedule
        The schedule to check.
    evaluation : Evaluation, optional
        The evaluation to call it on; a small fixture, ``J = 6`` particles of
        a block ``"u"`` of dimension 3 and data of dimension 4 at
        :math:`\beta = 0.25`, by default.

    Raises
    ------
    AssertionError
        Naming the first obligation that fails.
    """
    evaluation = _evaluation() if evaluation is None else evaluation
    name = repr(schedule)

    n_steps = getattr(schedule, "n_steps", "missing")
    assert n_steps is None or (type(n_steps) is int and n_steps >= 1), (
        f"{name}.n_steps: must be None or a positive int, got {n_steps!r}"
    )
    assert n_steps == schedule.n_steps, f"{name}.n_steps: changed between reads"
    beta_target = getattr(schedule, "beta_target", "missing")
    assert beta_target is None or (
        isinstance(beta_target, (int, float))
        and not isinstance(beta_target, bool)
        and beta_target > 0.0
    ), f"{name}.beta_target: must be None or a positive real, got {beta_target!r}"
    assert beta_target == schedule.beta_target, (
        f"{name}.beta_target: changed between reads"
    )

    first = schedule.next_increment(evaluation)
    second = schedule.next_increment(evaluation)
    if first is None:
        assert second is None, f"{name}.next_increment: not pure across two calls"
    else:
        value = _increment(name, first)
        _identical(
            value,
            jnp.asarray(second),
            f"{name}.next_increment: not pure across two calls on one evaluation",
        )

    if beta_target is not None and (n_steps is None or evaluation.step < n_steps):
        level = 0.75 * beta_target
        near = _with(evaluation, beta=level)
        got = schedule.next_increment(near)
        if got is not None:
            got = float(_increment(name, got))
            remaining = beta_target - level
            assert got <= remaining * (1.0 + 1e-12), (
                f"{name}.next_increment: returned {got} at beta {level}, past its "
                f"beta_target {beta_target}"
            )

    residuals = np.array(evaluation.whitened_residuals)
    residuals[0, 0] = np.nan
    poisoned = _with(evaluation, whitened_residuals=jnp.asarray(residuals))
    got = schedule.next_increment(poisoned)
    if got is not None and bool(jnp.all(jnp.isfinite(jnp.asarray(got)))):
        assert first is not None and bool(jnp.asarray(got) == jnp.asarray(first)), (
            f"{name}.next_increment: turned a nan misfit into the finite step "
            f"{got}; it must return nan (which the driver refuses) or None"
        )


def check_inflation(inflation, key=None, ensemble: Ensemble | None = None) -> None:
    r"""Check an :class:`~enskit.algorithms.Inflation` against its protocol.

    Calls ``inflation(key, ensemble=ensemble, step=0, beta=0.25)`` and checks
    that

    1. the result is an unweighted :class:`~enskit.distribution.Ensemble`
       with the same blocks in the same order, the same dimensions, the same
       particle count and the same dtype;
    2. a second call with the same key gives the same particles, bit for bit;
    3. a call with an extra, unknown context keyword gives them too.

    Parameters
    ----------
    inflation : Inflation
        The inflation to check.
    key : jax.random key, optional
        A typed key; ``jax.random.key(0)`` by default.
    ensemble : Ensemble, optional
        The particles to inflate; ``J = 6`` particles of blocks ``"u"`` and
        ``"v"``, of dimensions 3 and 2, by default.

    Raises
    ------
    AssertionError
        Naming the first obligation that fails.
    """
    key = jax.random.key(0) if key is None else key
    ensemble = _ensemble() if ensemble is None else ensemble
    name = repr(inflation)
    context = {"step": 0, "beta": jnp.asarray(0.25)}
    got = inflation(key, ensemble=ensemble, **context)
    _same_structure(name, got, ensemble)
    _identical_ensembles(
        got,
        inflation(key, ensemble=ensemble, **context),
        f"{name}: not deterministic given its key",
    )
    _identical_ensembles(
        got,
        inflation(key, ensemble=ensemble, **context, **{_UNKNOWN_CONTEXT: 1}),
        f"{name}: does not ignore a context keyword it does not use",
    )


def check_relaxation(
    relaxation, prior: Ensemble | None = None, posterior: Ensemble | None = None
) -> None:
    r"""Check an :class:`~enskit.algorithms.Relaxation` against its protocol.

    Calls ``relaxation(prior=prior, posterior=posterior, step=0, beta=0.25)``
    and checks that

    1. the result is an unweighted :class:`~enskit.distribution.Ensemble`
       with the blocks, dimensions, particle count and dtype of ``posterior``;
    2. a second call gives the same particles, bit for bit;
    3. a call with an extra, unknown context keyword gives them too.

    Parameters
    ----------
    relaxation : Relaxation
        The relaxation to check.
    prior, posterior : Ensemble, optional
        An update's input and output; by default two sets of ``J = 6``
        particles, the prior over blocks ``"u"``, ``"v"`` and a given block,
        the posterior over ``"u"`` and ``"v"`` with a smaller spread.

    Raises
    ------
    AssertionError
        Naming the first obligation that fails.
    """
    if prior is None or posterior is None:
        prior, posterior = _update_pair()
    name = repr(relaxation)
    context = {"step": 0, "beta": jnp.asarray(0.25)}
    got = relaxation(prior=prior, posterior=posterior, **context)
    _same_structure(name, got, posterior)
    _identical_ensembles(
        got,
        relaxation(prior=prior, posterior=posterior, **context),
        f"{name}: not deterministic",
    )
    _identical_ensembles(
        got,
        relaxation(
            prior=prior, posterior=posterior, **context, **{_UNKNOWN_CONTEXT: 1}
        ),
        f"{name}: does not ignore a context keyword it does not use",
    )


def check_stopping_rule(stop, evaluation: Evaluation | None = None) -> None:
    r"""Check a :class:`~enskit.algorithms.eki.StoppingRule` against its protocol.

    Checks that it returns a Python ``bool`` (not a 0-d array, whose truth
    value hides a traced one) and that it is pure: a second call on the same
    evaluation returns the same answer.

    Parameters
    ----------
    stop : StoppingRule
        The stopping rule to check.
    evaluation : Evaluation, optional
        The evaluation to call it on; the fixture of :func:`check_schedule` by
        default.

    Raises
    ------
    AssertionError
        Naming the first obligation that fails.
    """
    evaluation = _evaluation() if evaluation is None else evaluation
    name = repr(stop)
    first = stop(evaluation)
    assert type(first) is bool, (
        f"{name}: must return a Python bool, got {type(first).__name__}"
    )
    assert first == stop(evaluation), f"{name}: not pure across two calls"


# ---------------------------------------------------------------------------
# private
# ---------------------------------------------------------------------------


def _evaluation() -> Evaluation:
    """A small evaluation: the residuals have spread, so a criterion can measure it.

    The arrays are independent draws, not a consistent step: a fixture for
    shapes and purity.
    """
    rng = np.random.default_rng(0)
    ensemble = Ensemble(
        {
            "u": jnp.asarray(rng.normal(size=(6, 3))),
            PREDICTION: jnp.asarray(rng.normal(size=(6, 4))),
        }
    )
    return Evaluation(
        step=0,
        beta=0.25,
        ensemble=ensemble,
        whitened_residuals=jnp.asarray(rng.normal(size=(6, 4))),
        n_valid=6,
    )


def _ensemble() -> Ensemble:
    rng = np.random.default_rng(2)
    return Ensemble(
        u=jnp.asarray(rng.normal(size=(6, 3))), v=jnp.asarray(rng.normal(size=(6, 2)))
    )


def _update_pair() -> tuple[Ensemble, Ensemble]:
    rng = np.random.default_rng(3)
    prior = Ensemble(
        u=jnp.asarray(rng.normal(size=(6, 3))),
        v=jnp.asarray(rng.normal(size=(6, 2))),
        g=jnp.asarray(rng.normal(size=(6, 4))),
    )
    posterior = Ensemble(
        u=jnp.asarray(0.3 * rng.normal(size=(6, 3)) + 1.0),
        v=jnp.asarray(0.3 * rng.normal(size=(6, 2)) - 1.0),
    )
    return prior, posterior


def _with(evaluation: Evaluation, **changes) -> Evaluation:
    fields = {
        "step": evaluation.step,
        "beta": evaluation.beta,
        "ensemble": evaluation.ensemble,
        "whitened_residuals": evaluation.whitened_residuals,
        "n_valid": evaluation.n_valid,
    }
    fields.update(changes)
    return Evaluation(**fields)


def _increment(name: str, value):
    value = jnp.asarray(value)
    assert value.ndim == 0, (
        f"{name}.next_increment: must return a scalar, got shape {value.shape}"
    )
    assert bool(jnp.isfinite(value) & (value > 0.0)), (
        f"{name}.next_increment: must return a finite, strictly positive value or "
        f"None, got {value}"
    )
    return value


def _same_structure(name: str, got, like: Ensemble) -> None:
    assert isinstance(got, Ensemble), (
        f"{name}: must return an Ensemble, got {type(got).__name__}"
    )
    assert not got.is_weighted, f"{name}: returned a weighted ensemble"
    assert got.names == like.names and got.dims == like.dims, (
        f"{name}: returned blocks {got.dims}, expected {like.dims} in that order"
    )
    assert got.n_particles == like.n_particles, (
        f"{name}: returned {got.n_particles} particles, expected {like.n_particles}"
    )
    for n in got.names:
        assert got[n].dtype == like[n].dtype, (
            f"{name}: returned block {n!r} of dtype {got[n].dtype}, expected "
            f"{like[n].dtype}"
        )


def _identical(got, want, what: str) -> None:
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape, f"{what}: shape {got.shape} != {want.shape}"
    assert np.array_equal(got, want), f"{what}: not bit-identical"


def _identical_ensembles(got: Ensemble, want: Ensemble, what: str) -> None:
    assert isinstance(want, Ensemble) and want.names == got.names, what
    for n in got.names:
        _identical(got[n], want[n], f"{what} (block {n!r})")
