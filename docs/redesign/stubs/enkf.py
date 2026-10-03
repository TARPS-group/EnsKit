r"""Ensemble Kalman filtering: a forecast through a transition, an analysis of data.

For the state-space model

.. math::

    x_t = M(x_{t-1}) + \eta_t, \quad \eta_t \sim \mathcal N(0, Q), \qquad
    y_t = H(x_t) + e_t, \quad e_t \sim \mathcal N(0, R),

(the transition noise :math:`\eta_t` optional), this module cycles
forecast and analysis over a sequence of observations. It owns the vocabulary
of time, forecasts, analyses and observations; the layers below have none.

===================== ==========================================================
object                is
===================== ==========================================================
:func:`forecast`      push the state block through the transition
:func:`analysis`      one Kalman update on an observation, and its log evidence
:func:`filter`        the cycle over a sequence of observations
:class:`FilterResult` final ensemble, analysis means, log evidence per time
===================== ==========================================================

The ensemble may carry blocks beyond the state: parameters the transition
reads, or earlier states for smoothing. Every non-given block is updated at
each analysis, so parameter estimation and lag smoothing need no extra
machinery (Example 13).

References
----------
.. [1] Evensen, G. (1994). Sequential data assimilation with a nonlinear
   quasi-geostrophic model using Monte Carlo methods to forecast error
   statistics. Journal of Geophysical Research: Oceans, 99(C5), 10143–10162.
.. [2] Burgers, G., van Leeuwen, P. J. & Evensen, G. (1998). Analysis scheme
   in the ensemble Kalman filter. Monthly Weather Review, 126(6), 1719–1724.
.. [3] Houtekamer, P. L. & Mitchell, H. L. (1998). Data assimilation using an
   ensemble Kalman filter technique. Monthly Weather Review, 126(3), 796–811.
"""

PREDICTION: str = "prediction"


def forecast(key, ensemble: Ensemble, transition, *, state: str,
             inputs: Sequence[str] | None = None, transition_noise: PSDLinOp | None = None
             ) -> Ensemble:
    """Replace block ``state`` by ``transition(*inputs)``, then add transition noise.

    Parameters
    ----------
    key : jax.random key
        Used only when ``transition_noise`` is given (taken first, as every
        function in this layer takes it, so that drivers split keys the same
        way whatever is configured).
    transition : callable
        A simulator: ``(n_particles, d)`` arrays in, ``(n_particles, d_state)`` out.
    state : str
        Keyword-only. The block the transition writes.
    inputs : sequence of str, optional
        Keyword-only. The blocks it reads; ``(state,)`` by default.
    transition_noise : PSDLinOp, optional
        Keyword-only. Sampled per particle; needs ``factor``.
    """


def analysis(key, ensemble: Ensemble, observation: Array, *, observe, noise_cov: PSDLinOp,
             update_rule, inputs: Sequence[str], approximation=None) -> tuple[Ensemble, Array]:
    """One analysis: predict the observation, update every block, report the evidence.

    Pushes ``inputs`` through ``observe`` to the block :data:`PREDICTION`,
    builds the joint with ``joint`` (default :func:`enskit.kalman.gaussian_approximation`, which
    adds ``noise_cov`` as a covariance), and applies ``update_rule``.

    Returns
    -------
    ensemble : Ensemble
        The analysis particles over every block except :data:`PREDICTION`.
    log_evidence : Array
        0-d. The density of ``observation`` under the joint's prediction
        marginal: the one-step predictive log likelihood, whose sum over time
        is the objective for tuning ``R``, transition noise or inflation.
    """


class FilterResult:
    """What :func:`filter` returns. A frozen dataclass.

    Attributes
    ----------
    ensemble : Ensemble
        The analysis particles at the last time.
    means : dict[str, Array]
        Block name to ``(T, d)`` analysis means.
    log_evidence : Array
        ``(T,)`` one-step predictive log likelihoods.
    ensembles : tuple[Ensemble, ...] or None
        Every analysis ensemble, when ``keep_ensembles=True``.
    """


def filter(key, ensemble: Ensemble, observations, *, transition, observe, noise_cov,
           update_rule, state: str | None = None, transition_inputs=None, observe_inputs=None,
           transition_noise=None, inflation=None, relaxation=None, approximation=None,
           keep_ensembles: bool = False) -> FilterResult:
    """Cycle forecast, inflation, analysis and relaxation over ``observations``.

    Parameters
    ----------
    key : jax.random key
        Split into ``(next, forecast, inflate, analysis)`` at every time,
        whatever is configured, so toggling a policy never shifts another's
        random stream.
    ensemble : Ensemble
        The particles before the first forecast.
    observations : Array
        ``(T, N)``, one row per time.
    transition, observe : callable
        Simulators. ``observe`` may be a :class:`enskit.maps.Linear`.
    noise_cov : PSDLinOp
    update_rule : UpdateRule
        Keyword-only and required.
    state : str, optional
        The block the transition writes; the ensemble's first block by default.
    inflation : Inflation, optional
        Called as ``inflation(key, ensemble=..., time=t)`` after each forecast.
    relaxation : Relaxation, optional
        Called as ``relaxation(prior=..., posterior=..., time=t)`` after each
        analysis, with the forecast particles as ``prior``.
    approximation : callable, optional
        ``(ensemble, noise) -> Gaussian``, the joint Gaussian approximation
        every analysis conditions; see :func:`enskit.kalman.update`.
    """
