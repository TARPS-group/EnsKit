r"""Ensemble Kalman filtering: a forecast through a transition, an analysis of data.

For the state-space model

.. math::

    x_t = M(x_{t-1}) + \eta_t, \quad \eta_t \sim \mathcal N(0, Q), \qquad
    y_t = H(x_t) + e_t, \quad e_t \sim \mathcal N(0, R), \qquad
    t = 1, \dots, T,

with the transition noise :math:`\eta_t` optional, the ensemble Kalman
filter carries particles :math:`x_j` approximating the filtering
distribution :math:`p(x_t \mid y_{1:t})` from one time to the next: a
*forecast* moves every particle through :math:`M`, and an *analysis*
conditions the forecast particles on :math:`y_t` with one
:func:`enskit.kalman.update`.

Using it::

    from enskit import kalman, toy
    from enskit.algorithms import MultiplicativeInflation, enkf

    problem = toy.lorenz96()
    ensemble = problem.initial.sample(jax.random.key(0), n_particles=40)
    result = enkf.filter(ensemble, problem.observations,
                         transition=problem.transition, observe=problem.observe,
                         noise_cov=problem.noise_cov,
                         update_rule=kalman.Matheron(),
                         inflation=MultiplicativeInflation(1.05),
                         key=jax.random.key(1))
    result.means["x"]             # (T, 40) analysis means
    result.log_evidence.sum()     # the log evidence of the run's observations

===================== ==========================================================
object                is
===================== ==========================================================
:func:`forecast`      push the state block through the transition, and add
                      transition noise
:func:`analysis`      one Kalman update on an observation, and its log evidence
:func:`filter`        the cycle over a sequence of observations
:class:`FilterResult` the final ensemble, the analysis means, the log evidence
                      per time
:class:`EnKFError`    raised when a particle stops being finite
:data:`PREDICTION`    the name of the block an analysis predicts into
===================== ==========================================================

The update rule is any :class:`enskit.kalman.UpdateRule` and is required.
Inflation and relaxation are the shared policies of
:mod:`enskit.algorithms`.

Conventions shared by everything in the module:

- **Blocks beyond the state are carried and updated.** The ensemble may hold
  parameters the transition reads, or earlier states. Every block is a
  target of each analysis, so state augmentation and lag smoothing need no
  machinery of their own.
- **The observation is predicted into the block** :data:`PREDICTION`, a name
  the ensemble may not use; the noise covariance is added to the
  approximation, never sampled.
- **Randomness enters through a keyword-only, typed** ``key``, needed only
  when something draws. :func:`filter` splits it into
  ``(next, forecast, inflate, analysis)`` at every time, whatever is
  configured; :func:`forecast` splits its own into ``(transition, noise)``.
- **The loop is ordinary Python**, so the transition and the observation
  model may be host-side simulators.

Notes
-----
The behavior of this module is specified by the "Ensemble Kalman filter
contract" page of the documentation, which is normative. Exactness is claimed
for the linear-Gaussian case only: with linear :math:`M` and :math:`H`, no
transition noise, particles whose sample moments equal the initial
distribution's, more particles than state dimensions,
:class:`~enskit.kalman.SymmetricSquareRoot` and no inflation or relaxation,
the analysis means, covariances and log evidences are the Kalman filter's to
round-off.

References
----------
Evensen, G. (1994). Sequential data assimilation with a nonlinear
quasi-geostrophic model using Monte Carlo methods to forecast error
statistics. *Journal of Geophysical Research: Oceans*, 99(C5), 10143–10162.

Burgers, G., van Leeuwen, P. J. & Evensen, G. (1998). Analysis scheme in the
ensemble Kalman filter. *Monthly Weather Review*, 126(6), 1719–1724.

Houtekamer, P. L. & Mitchell, H. L. (1998). Data assimilation using an
ensemble Kalman filter technique. *Monthly Weather Review*, 126(3), 796–811.
"""

from ._filter import analysis, filter, forecast
from ._values import PREDICTION, EnKFError, FilterResult

__all__ = [
    "forecast",
    "analysis",
    "filter",
    "FilterResult",
    "EnKFError",
    "PREDICTION",
]

# The modules above are private, so the public names report the package they
# are imported from, in tracebacks, ``type()`` and pickles.
for _name in __all__:
    if not isinstance(globals()[_name], str):
        globals()[_name].__module__ = __name__
del _name
