r"""The forecast, the analysis, and the cycle over a sequence of observations."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import Array

from ... import kalman, maps
from ...distribution import Ensemble, Gaussian
from ...linalg import PSDLinOp, value_check
from .. import _common as c
from ._values import PREDICTION, EnKFError, FilterResult

__all__ = ["forecast", "analysis", "filter"]


def forecast(
    ensemble: Ensemble,
    transition,
    *,
    state: str,
    inputs: str | Sequence[str] | None = None,
    transition_noise: PSDLinOp | None = None,
    key=None,
) -> Ensemble:
    r"""Push the state block through the transition, then add transition noise.

    .. math::

        x_j \mapsto M(z_j) + \eta_j, \qquad
        \eta_j \sim \mathcal N(0, Q) \ \text{independently},

    with :math:`z_j` particle :math:`j`'s input blocks and :math:`Q` =
    ``transition_noise`` (no :math:`\eta_j` without it). It is

    .. code-block:: python

        ens = maps.pushforward(ensemble, transition, inputs=inputs,
                               output=state, key=key_transition)
        ens = maps.pushforward(ens, maps.AdditiveNoise(transition_noise),
                               inputs=state, output=state, key=key_noise)

    with ``key_transition, key_noise = jax.random.split(key)``, and the second
    line only when ``transition_noise`` is given. Every other block is carried
    through unchanged, and the ensemble keeps its weights.

    Parameters
    ----------
    ensemble : Ensemble
        The particles.
    transition : callable or StructuredMap
        :math:`M`, a simulator in the sense of :func:`enskit.maps.pushforward`:
        one ``(J, d_b)`` array per input block in, ``(J, d)`` out, with
        :math:`d` the state block's dimension. A simulator declaring
        ``needs_key`` receives ``key_transition``.
    state : str
        Keyword-only. The block the transition writes.
    inputs : str or sequence of str, optional
        Keyword-only. The blocks the transition reads, in order;
        ``(state,)`` by default.
    transition_noise : PSDLinOp, optional
        Keyword-only. :math:`Q`, of the state block's side, supporting
        ``factor``; drawn per particle.
    key : jax.random key, optional
        Keyword-only. Required with ``transition_noise`` or a transition that
        declares ``needs_key``; split once into ``(transition, noise)``
        whatever is configured.

    Returns
    -------
    Ensemble
        Over the same blocks, in the same order.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    KeyError
        If ``state`` or an input is not a block.
    ValueError
        If ``transition_noise`` is given without a key or has the wrong
        side, an argument is a vmapped family, or the transition's output is
        not of the state block's dimension; and the checks of
        :func:`enskit.maps.pushforward`.

    Notes
    -----
    No value is checked, so the function runs under :func:`jax.jit` when the
    transition is traceable. A transition that returns a non-finite row is
    caught by :func:`filter`, not here.
    """
    where = "enkf.forecast"
    _check_ensemble(where, ensemble)
    state = _check_block(where, ensemble, "state", state)
    inputs = _check_inputs(where, ensemble, "inputs", inputs, default=(state,))
    _check_transition_noise(where, ensemble, state, transition_noise)
    _check_optional_key(where, key)
    if transition_noise is not None and key is None:
        raise ValueError(
            f"{where}: transition_noise is drawn per particle, so a key is required"
        )
    return _forecast(where, ensemble, transition, state, inputs, transition_noise, key)


def analysis(
    ensemble: Ensemble,
    observation,
    *,
    observe,
    noise_cov: PSDLinOp,
    update_rule,
    inputs: str | Sequence[str],
    approximation=None,
    key=None,
) -> tuple[Ensemble, Array]:
    r"""One analysis: predict the observation, update every block, report the evidence.

    For the observation model :math:`y = H(z) + e`, :math:`e \sim \mathcal
    N(0, R)`, with :math:`z` the input blocks, the analysis pushes the
    particles through :math:`H` to the block :data:`PREDICTION`, builds the
    joint Gaussian approximation :math:`\hat p` of every block and the
    prediction with :math:`R` added to the prediction, and conditions it on
    :math:`y` with ``update_rule``:

    .. code-block:: python

        predicted = maps.pushforward(ensemble, observe, inputs=inputs,
                                     output=PREDICTION)
        analyzed = kalman.update(predicted, {PREDICTION: observation},
                                 noise={PREDICTION: noise_cov},
                                 update_rule=update_rule,
                                 approximation=approximation, key=key)

    Every block of the ensemble is a target, the input blocks included, so
    parameters and earlier states carried as blocks are updated too. The
    log evidence is the density of the observation under the approximation's
    prediction marginal,

    .. math::

        \log \hat p(y) = \log \mathcal N\big(y;\ \bar h,\ \hat C_{hh} + R\big),

    with :math:`\bar h` and :math:`\hat C_{hh}` the predictions' sample mean
    and covariance under the default approximation, computed from the same
    approximation the update conditions.

    Parameters
    ----------
    ensemble : Ensemble
        Unweighted and finite, without a block named :data:`PREDICTION`.
    observation : Array
        :math:`y`, ``(N,)``.
    observe : callable or StructuredMap
        :math:`H`, a simulator from the input blocks to ``(J, N)``; a
        :class:`enskit.maps.Linear` for a linear observation model.
        Keyword-only.
    noise_cov : PSDLinOp
        :math:`R`, of side :math:`N`, supporting ``whiten``. Keyword-only.
    update_rule : UpdateRule
        Keyword-only and required: :class:`~enskit.kalman.SymmetricSquareRoot`,
        :class:`~enskit.kalman.Matheron`, a
        :class:`~enskit.kalman.LocalizedUpdateRule`, or any object with a
        ``build`` method.
    inputs : str or sequence of str
        Keyword-only and required. The blocks ``observe`` reads, in order.
    approximation : callable, optional
        Keyword-only. ``(ensemble, noise) -> Gaussian``, the joint Gaussian
        the update conditions; :func:`enskit.kalman.gaussian_approximation`
        by default. It receives the ensemble with the prediction block, and
        must cover every block.
    key : jax.random key, optional
        Keyword-only. Passed whole to the update; required by
        :class:`~enskit.kalman.Matheron`.

    Returns
    -------
    ensemble : Ensemble
        The analysis particles, over the ensemble's blocks in its order.
    log_evidence : Array
        0-d, of the ensemble's dtype. :math:`\log \hat p(y)`, the one-step
        log evidence.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    KeyError
        If an input is not a block.
    ValueError
        If the ensemble uses the name :data:`PREDICTION`, ``observation`` is
        not ``(N,)``, ``observe`` does not return ``N`` values per particle,
        the approximation leaves out a block, or an argument is a vmapped
        family; and the checks of :func:`enskit.maps.pushforward` and
        :func:`enskit.kalman.update`. In debug mode, also if ``observation``
        is not finite.

    Notes
    -----
    The prediction is never sampled with noise: :math:`R` enters the
    approximation as a covariance, which the update rule then uses
    (:class:`~enskit.kalman.Matheron` draws its own perturbations). The
    function runs under :func:`jax.jit` when ``observe`` is traceable.
    """
    where = "enkf.analysis"
    _check_ensemble(where, ensemble)
    _check_reserved(where, ensemble)
    if inputs is None:
        raise TypeError(
            f"{where}: inputs is required; name the blocks observe reads, such as "
            f"inputs='x'"
        )
    inputs = _check_inputs(where, ensemble, "inputs", inputs, default=None)
    data_dim = _check_noise_cov(where, noise_cov)
    observation = _check_observation(where, observation, data_dim)
    _check_update_rule(where, update_rule)
    _check_optional_callable(where, "approximation", approximation)
    _check_optional_key(where, key)
    return _analysis(
        where, ensemble, observation, observe, noise_cov, update_rule, inputs,
        approximation, data_dim, key,
    )


def filter(  # noqa: A001 - the module's verb, as in the design
    ensemble: Ensemble,
    observations,
    *,
    transition,
    observe,
    noise_cov: PSDLinOp,
    update_rule,
    state: str | None = None,
    transition_inputs: str | Sequence[str] | None = None,
    observe_inputs: str | Sequence[str] | None = None,
    transition_noise: PSDLinOp | None = None,
    inflation=None,
    relaxation=None,
    approximation=None,
    keep_ensembles: bool = False,
    start_time: int = 0,
    key=None,
) -> FilterResult:
    r"""Cycle forecast, inflation, analysis and relaxation over the observations.

    For each row :math:`y` of ``observations``, at time :math:`t` (counted
    from ``start_time``):

    1. split the key into ``(next, forecast, inflate, analysis)``;
    2. forecast: :func:`forecast` through ``transition``, with
       ``transition_noise``;
    3. inflate: ``inflation(key_inflate, ensemble=..., time=t)``;
    4. analyze: :func:`analysis` of :math:`y`, through ``observe``;
    5. relax: ``relaxation(prior=..., posterior=..., time=t)``, with
       ``prior`` the particles the analysis started from;
    6. record every block's analysis mean and the log evidence.

    The ensemble's particles are the analysis particles at the time before
    the first observation, :math:`t = -1`; the first operation is a
    forecast.

    Parameters
    ----------
    ensemble : Ensemble
        Unweighted and finite, without a block named :data:`PREDICTION`.
    observations : Array
        ``(T, N)``, one row per time, finite; :math:`T \ge 1`.
    transition : callable or StructuredMap
        Keyword-only. The transition :math:`M`; see :func:`forecast`. Called
        once per time with every particle; never traced.
    observe : callable or StructuredMap
        Keyword-only. The observation model :math:`H`; see :func:`analysis`.
    noise_cov : PSDLinOp
        Keyword-only. :math:`R`, the same at every time.
    update_rule : UpdateRule
        Keyword-only and required.
    state : str, optional
        Keyword-only. The block the transition writes; the ensemble's first
        block by default.
    transition_inputs, observe_inputs : str or sequence of str, optional
        Keyword-only. The blocks ``transition`` and ``observe`` read;
        ``(state,)`` by default.
    transition_noise : PSDLinOp, optional
        Keyword-only. :math:`Q`, drawn per particle at each forecast.
    inflation : Inflation, optional
        Keyword-only. Applied after each forecast; see
        :class:`enskit.algorithms.Inflation`.
    relaxation : Relaxation, optional
        Keyword-only. Applied after each analysis; see
        :class:`enskit.algorithms.Relaxation`.
    approximation : callable, optional
        Keyword-only. ``(ensemble, noise) -> Gaussian``, the joint Gaussian
        each analysis conditions; see :func:`analysis`.
    keep_ensembles : bool
        Keyword-only. Keep every analysis ensemble in the result.
    start_time : int
        Keyword-only. The time of the first row of ``observations``, 0 by
        default: the policies receive ``time=start_time + i`` for row
        :math:`i`, so a filter resumed from an :class:`EnKFError` with
        ``start_time=exc.time`` gives a time-dependent policy the times the
        uninterrupted filter would have.
    key : jax.random key, optional
        Keyword-only. The filter's random stream, split four ways at every
        time whatever is configured, so toggling a policy never shifts
        another's draws. Required when anything draws: transition noise, a
        stochastic update rule, additive inflation.

    Returns
    -------
    FilterResult

    Raises
    ------
    EnKFError
        If a particle is not finite after the forecast, the inflation, the
        observation model, the update or the relaxation. It carries the time,
        the key and the result so far, so the filter can be resumed::

            enkf.filter(exc.result.ensemble,
                        observations[exc.time - start_time:],
                        start_time=exc.time, key=exc.key, ...)
    TypeError
        If an argument has the wrong type, including a policy's output.
    KeyError
        If ``state`` or an input is not a block.
    ValueError
        If ``observations`` is not ``(T, N)`` with :math:`T \ge 1`, or not
        finite; if a policy's output does not match its input; and the
        checks of :func:`forecast` and :func:`analysis`.

    Warns
    -----
    UserWarning
        From :func:`enskit.maps.pushforward`, at each call whose simulator
        returned a narrower floating dtype than the particles'.

    Notes
    -----
    Inflation and relaxation compound when both are used: the relaxation's
    ``prior`` is the inflated forecast, so relaxing the spread fully restores
    the *inflated* spread.

    The loop is ordinary Python, so the transition and the observation model
    may be host-side simulators. A time-varying observation model, noise or
    transition, or an ensemble edited between times, is a loop over
    :func:`forecast` and :func:`analysis`.
    """
    where = "enkf.filter"
    _check_ensemble(where, ensemble)
    _check_reserved(where, ensemble)
    state = ensemble.names[0] if state is None else state
    state = _check_block(where, ensemble, "state", state)
    transition_inputs = _check_inputs(
        where, ensemble, "transition_inputs", transition_inputs, default=(state,)
    )
    observe_inputs = _check_inputs(
        where, ensemble, "observe_inputs", observe_inputs, default=(state,)
    )
    _check_transition_noise(where, ensemble, state, transition_noise)
    if transition_noise is not None and key is None:
        raise ValueError(
            f"{where}: transition_noise is drawn per particle, so a key is required"
        )
    _check_simulator(where, "transition", transition)
    _check_simulator(where, "observe", observe)
    data_dim = _check_noise_cov(where, noise_cov)
    observations = _check_observations(where, observations, data_dim)
    _check_update_rule(where, update_rule)
    for name, value in (
        ("inflation", inflation),
        ("relaxation", relaxation),
        ("approximation", approximation),
    ):
        _check_optional_callable(where, name, value)
    if not isinstance(keep_ensembles, bool):
        raise TypeError(
            f"{where}: keep_ensembles must be a bool, got {type(keep_ensembles).__name__}"
        )
    if type(start_time) is not int or start_time < 0:
        raise TypeError(
            f"{where}: start_time must be an int, at least 0, got {start_time!r}"
        )
    _check_optional_key(where, key)
    if ensemble.is_weighted:
        raise ValueError(
            f"{where}: the ensemble is weighted; an analysis takes unweighted "
            f"particles. Resample first, with enskit.distribution.resample."
        )
    if not _all_finite(ensemble):
        raise ValueError(f"{where}: the initial ensemble has a non-finite particle")

    means: dict[str, list[Array]] = {name: [] for name in ensemble.names}
    log_evidence: list[Array] = []
    kept: list[Ensemble] | None = [] if keep_ensembles else None
    for row in range(observations.shape[0]):
        t = start_time + row
        key_t = key
        if key is None:
            key_forecast = key_inflate = key_analysis = None
        else:
            key, key_forecast, key_inflate, key_analysis = jax.random.split(key, 4)
        try:
            ensemble, evidence = _cycle(
                where, t, ensemble, observations[row], transition, observe, noise_cov,
                update_rule, state, transition_inputs, observe_inputs,
                transition_noise, inflation, relaxation, approximation, data_dim,
                key_forecast, key_inflate, key_analysis,
            )
        except _NonFiniteError as failure:
            raise EnKFError(
                f"{where}: {failure} at time {t}. A non-finite particle would make "
                f"every later analysis nan, so the filter stops here; the error "
                f"carries the result so far.",
                time=t,
                key=key_t,
                result=_result(ensemble, means, log_evidence, kept),
            ) from None
        for name in ensemble.names:
            means[name].append(ensemble.mean(name))
        log_evidence.append(evidence)
        if kept is not None:
            kept.append(ensemble)
    return _result(ensemble, means, log_evidence, kept)


# ---------------------------------------------------------------------------
# private: the array work
# ---------------------------------------------------------------------------


class _NonFiniteError(Exception):
    """A stage of one time returned a non-finite particle; the message names it."""


def _cycle(
    where, t, ensemble, observation, transition, observe, noise_cov, update_rule,
    state, transition_inputs, observe_inputs, transition_noise, inflation,
    relaxation, approximation, data_dim, key_forecast, key_inflate, key_analysis,
):
    """One time of :func:`filter`: forecast, inflate, analyze, relax.

    Returns the analysis ensemble and the log evidence; raises
    :class:`_NonFiniteError` naming the stage whose output stopped being finite.
    """
    forecasted = _forecast(
        where, ensemble, transition, state, transition_inputs, transition_noise,
        key_forecast,
    )
    if not _all_finite(forecasted):
        raise _NonFiniteError(
            f"the transition {_describe(transition)} returned a non-finite particle"
        )
    background = forecasted
    if inflation is not None:
        background = inflation(key_inflate, ensemble=forecasted, time=t)
        c.check_policy_output(
            where, f"the inflation {inflation!r}", background, forecasted
        )
        if not _all_finite(background):
            raise _NonFiniteError(
                f"the inflation {inflation!r} returned a non-finite particle"
            )
    analyzed, evidence = _analysis(
        where, background, observation, observe, noise_cov, update_rule,
        observe_inputs, approximation, data_dim, key_analysis, check_finite=True,
    )
    if not _all_finite(analyzed):
        raise _NonFiniteError(
            f"the update {update_rule!r} returned a non-finite particle"
        )
    if relaxation is not None:
        relaxed = relaxation(prior=background, posterior=analyzed, time=t)
        c.check_policy_output(where, f"the relaxation {relaxation!r}", relaxed, analyzed)
        if not _all_finite(relaxed):
            raise _NonFiniteError(
                f"the relaxation {relaxation!r} returned a non-finite particle"
            )
        analyzed = relaxed
    return analyzed, evidence


def _forecast(where, ensemble, transition, state, inputs, transition_noise, key):
    """:func:`forecast`, with its arguments checked."""
    if key is None:
        key_transition = key_noise = None
    else:
        key_transition, key_noise = jax.random.split(key)
    forecasted = maps.pushforward(
        ensemble, transition, inputs=inputs, output=state, key=key_transition
    )
    if forecasted.dims[state] != ensemble.dims[state]:
        raise ValueError(
            f"{where}: the transition {_describe(transition)} returned "
            f"{forecasted.dims[state]} values per particle for block {state!r}, "
            f"which has dimension {ensemble.dims[state]}"
        )
    if transition_noise is not None:
        forecasted = maps.pushforward(
            forecasted,
            maps.AdditiveNoise(transition_noise),
            inputs=state,
            output=state,
            key=key_noise,
        )
    return forecasted


def _analysis(
    where, ensemble, observation, observe, noise_cov, update_rule, inputs,
    approximation, data_dim, key, *, check_finite=False,
):
    """:func:`analysis`, with its arguments checked.

    With ``check_finite``, as :func:`filter` calls it, a non-finite prediction
    raises :class:`_NonFiniteError` naming the observation model, before the
    update could be blamed for it.
    """
    predicted = maps.pushforward(ensemble, observe, inputs=inputs, output=PREDICTION)
    got = predicted.dims[PREDICTION]
    if got != data_dim:
        raise ValueError(
            f"{where}: the observation model {_describe(observe)} returned {got} "
            f"values per particle, but the noise covariance has side {data_dim}"
        )
    if check_finite and not _all_finite(predicted.marginal(PREDICTION)):
        raise _NonFiniteError(
            f"the observation model {_describe(observe)} returned a non-finite "
            f"prediction"
        )
    build = kalman.gaussian_approximation if approximation is None else approximation
    captured: list[Gaussian] = []

    def capture(particles, noise):
        approx = build(particles, noise)
        if isinstance(approx, Gaussian) and set(approx.names) != set(particles.names):
            raise ValueError(
                f"{where}: the approximation {_describe(approximation)} has blocks "
                f"{approx.names}; it must have exactly the blocks of the ensemble "
                f"and the prediction, {particles.names}"
            )
        captured.append(approx)
        return approx

    analyzed = kalman.update(
        predicted,
        {PREDICTION: observation},
        update_rule=update_rule,
        noise={PREDICTION: noise_cov},
        approximation=capture,
        key=key,
    )
    if analyzed.names != ensemble.names:
        analyzed = analyzed.marginal(*ensemble.names)
    if len(captured) != 1:
        raise RuntimeError(
            f"{where}: kalman.update called the approximation {len(captured)} times; "
            f"the log evidence needs the one approximation the update conditioned"
        )
    evidence = captured[0].marginal(PREDICTION).log_density({PREDICTION: observation})
    # The log evidence has the ensemble's dtype, whatever the observation's or
    # the noise covariance's, as every other output does.
    dtype = ensemble[ensemble.names[0]].dtype
    return analyzed, evidence.astype(dtype)


def _result(ensemble, means, log_evidence, kept) -> FilterResult:
    """A :class:`FilterResult` from the lists the loop accumulates."""
    dtype = ensemble[ensemble.names[0]].dtype
    return FilterResult(
        ensemble=ensemble,
        means={
            name: jnp.stack(rows) if rows else jnp.zeros((0, ensemble.dims[name]), dtype)
            for name, rows in means.items()
        },
        log_evidence=jnp.stack(log_evidence) if log_evidence else jnp.zeros((0,), dtype),
        ensembles=None if kept is None else tuple(kept),
    )


@jax.jit
def _finite(blocks) -> Array:
    """Whether every entry of every block is finite."""
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(b)) for b in blocks]))


def _all_finite(ensemble: Ensemble) -> bool:
    return bool(_finite(tuple(ensemble[n] for n in ensemble.names)))


def _describe(f) -> str:
    """A short name for a callable, for messages; never raises."""
    try:
        name = getattr(f, "__qualname__", None)
        return repr(f) if name is None else name
    except Exception:
        return type(f).__name__


# ---------------------------------------------------------------------------
# private: call-time validation
# ---------------------------------------------------------------------------


def _check_ensemble(where: str, ensemble) -> None:
    if not isinstance(ensemble, Ensemble):
        raise TypeError(
            f"{where}: ensemble must be an enskit.distribution.Ensemble, got "
            f"{type(ensemble).__name__}"
        )
    c.guard(ensemble, where)


def _check_reserved(where: str, ensemble: Ensemble) -> None:
    if PREDICTION in ensemble.names:
        raise ValueError(
            f"{where}: the ensemble has a block named {PREDICTION!r}, the name the "
            f"analysis predicts the observation into; rename it"
        )


def _check_block(where: str, ensemble: Ensemble, label: str, name) -> str:
    if not isinstance(name, str):
        raise TypeError(f"{where}: {label} must be a str, got {type(name).__name__}")
    if name not in ensemble.names:
        raise KeyError(
            f"{where}: {label} {name!r} is not a block; the blocks are {ensemble.names}"
        )
    return name


def _check_inputs(where: str, ensemble: Ensemble, label: str, inputs, *, default):
    """Block names as a tuple, each a block, none repeated."""
    if inputs is None:
        return default
    if isinstance(inputs, str):
        inputs = (inputs,)
    if not isinstance(inputs, Sequence):
        raise TypeError(
            f"{where}: {label} must be a str or a sequence of str, got "
            f"{type(inputs).__name__}"
        )
    inputs = tuple(inputs)
    if not inputs:
        raise ValueError(f"{where}: {label} must name at least one block")
    for name in inputs:
        _check_block(where, ensemble, f"{label} entry", name)
    if len(set(inputs)) != len(inputs):
        raise ValueError(f"{where}: a block is repeated in {label} {inputs}")
    return inputs


def _check_transition_noise(where, ensemble, state, transition_noise) -> None:
    if transition_noise is None:
        return
    if not isinstance(transition_noise, PSDLinOp):
        raise TypeError(
            f"{where}: transition_noise must be an enskit.linalg.PSDLinOp or None, "
            f"got {type(transition_noise).__name__}"
        )
    if transition_noise.batch_shape != ():
        raise ValueError(
            f"{where}: {transition_noise!r} is a vmapped family; a filter binds one "
            f"transition noise covariance"
        )
    if transition_noise.shape[0] != ensemble.dims[state]:
        raise ValueError(
            f"{where}: {transition_noise!r} has side {transition_noise.shape[0]}, but "
            f"block {state!r} has dimension {ensemble.dims[state]}"
        )


def _check_simulator(where: str, name: str, f) -> None:
    if not isinstance(f, maps.StructuredMap) and not callable(f):
        raise TypeError(
            f"{where}: {name} must be callable or an enskit.maps.StructuredMap, got "
            f"{type(f).__name__}"
        )


def _check_noise_cov(where: str, noise_cov) -> int:
    if not isinstance(noise_cov, PSDLinOp):
        raise TypeError(
            f"{where}: noise_cov must be an enskit.linalg.PSDLinOp, got "
            f"{type(noise_cov).__name__}"
        )
    if noise_cov.batch_shape != ():
        raise ValueError(
            f"{where}: {noise_cov!r} is a vmapped family; an analysis binds one noise "
            f"covariance"
        )
    return noise_cov.shape[0]


def _check_observation(where: str, observation, data_dim: int) -> Array:
    observation = jnp.asarray(observation)
    if observation.shape != (data_dim,):
        raise ValueError(
            f"{where}: expected an observation of shape ({data_dim},) to match the "
            f"noise covariance, got shape {observation.shape}"
        )
    value_check(
        observation,
        lambda y: bool(jnp.all(jnp.isfinite(y))),
        f"{where}: the observation must be finite",
    )
    return observation


def _check_observations(where: str, observations, data_dim: int) -> Array:
    observations = jnp.asarray(observations)
    if observations.ndim != 2 or observations.shape[1] != data_dim:
        raise ValueError(
            f"{where}: expected observations of shape (T, {data_dim}), one row per "
            f"time, to match the noise covariance; got shape {observations.shape}"
        )
    if observations.shape[0] < 1:
        raise ValueError(f"{where}: observations must have at least one row")
    if not bool(jnp.all(jnp.isfinite(observations))):
        rows = [int(i) for i in jnp.flatnonzero(~jnp.all(jnp.isfinite(observations), 1))]
        raise ValueError(
            f"{where}: the observations must be finite; rows {rows} are not. A "
            f"missing value is a hand loop over forecast and analysis with an "
            f"observation model for the values present."
        )
    return observations


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


def _check_optional_key(where: str, key):
    if key is not None:
        c.check_key(where, key)
    return key
