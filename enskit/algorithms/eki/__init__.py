r"""Ensemble Kalman Inversion: a ladder of tempered targets, and the run along it.

For a prior :math:`\pi_0` over the parameters :math:`u`, a forward model
:math:`G`, data :math:`y` and a noise covariance :math:`R` with whitener
:math:`W` (:math:`W^\top W = R^{-1}`), the targets are

.. math::

    \pi_\beta(u) \propto \pi_0(u)\, e^{-\beta\,\Phi(G(u))}, \qquad
    \Phi(g) = \tfrac12 \lVert W (y - g) \rVert^2, \qquad \beta \ge 0 ,

the prior at :math:`\beta = 0` and the posterior at :math:`\beta = 1`. Since

.. math::

    \frac{\pi_{\beta+\delta}(u)}{\pi_\beta(u)} \propto e^{-\delta\Phi(G(u))}
    \propto \mathcal N\big(y;\, G(u),\, R/\delta\big),

moving from :math:`\beta` to :math:`\beta + \delta` is conditioning on
:math:`y` with noise :math:`R/\delta`, so a step is one
:func:`enskit.kalman.update`. Everything specific to inversion lives here: the
level, the schedule, the stopping rule, failed particles, and the history.

Two forms of EKI come from the same driver, as two schedules::

    from enskit import kalman
    from enskit.algorithms import eki

    state = eki.EKIState.from_prior(key, prior, n_particles=64)

    # Sampling: an adaptive ladder to beta = 1.
    sampled = eki.run(state, forward, y, noise_cov,
                      update_rule=kalman.Matheron(),
                      schedule=eki.AdaptiveESSSchedule())

    # Optimization: unit steps until the discrepancy principle fires.
    fit = eki.run(state, forward, y, noise_cov,
                  update_rule=kalman.SymmetricSquareRoot(),
                  schedule=eki.FixedSchedule.constant(1.0, n_steps=200),
                  stop=eki.DiscrepancyStop())

====================================== ========================================
object                                 is
====================================== ========================================
:class:`EKIState`                      the state carried from step to step
:class:`Evaluation`                    one evaluation of the forward model and
                                       its summaries
:class:`HistoryRecord`                 the record of one step
:class:`EKIResult`                     the final state, the history, and why
                                       the run ended
:class:`Schedule`,                     the policy protocols
:class:`StoppingRule`
:class:`FixedSchedule`,                schedules
:class:`AdaptiveESSSchedule`,
:class:`AdaptiveMisfitSchedule`
:class:`DiscrepancyStop`               a stopping rule
:func:`run`, :func:`iterate`           the driver, as a function and as a
                                       generator
:func:`evaluate`, :func:`assimilate`,  one step, as its two phases and their
:func:`advance`                        composition
:func:`misfits`,                       array-level helpers
:func:`effective_sample_size`,
:func:`repair_failed_particles`
:data:`PREDICTION`,                    the prediction block's name and the
:data:`SCHEDULE_EXHAUSTED`,            three statuses
:data:`STOPPING_RULE`,
:data:`INTERRUPTED`
:class:`EKIError`                      raised when a run cannot continue
====================================== ========================================

The update rule is any :class:`enskit.kalman.UpdateRule` and is required.
Inflation and relaxation are the shared policies of
:mod:`enskit.algorithms`.

Conventions shared by everything in the module:

- **The state's ensemble holds the parameter blocks.** The forward model
  receives them, one positional array per block, in block order (or the
  blocks named by ``inputs``), and its output becomes the block
  :data:`PREDICTION`, a name the state's ensemble may not use.
- **Steps take increments.** The state carries the level :math:`\beta`; a
  step takes :math:`\delta` and conditions with noise :math:`R/\delta`,
  never :math:`R/\beta`.
- **The misfit carries the factor** :math:`\tfrac12` and is measured against
  the base noise covariance.
- **Randomness enters through the state's typed key**, split once per step
  into ``(next, inflate, evaluate, update)`` whatever the policies, so turning
  inflation on never shifts the update's draws.

Notes
-----
The behavior of this module is specified by the "Ensemble Kalman Inversion
contract" page of the documentation, which is normative. Exactness is claimed
for the linear-Gaussian case only: with an affine forward model, a Gaussian
prior, particles whose sample moments equal the prior's,
:class:`~enskit.kalman.SymmetricSquareRoot`, no inflation or relaxation, no
failed particles, and increments summing exactly to 1, a run reproduces the
posterior mean and covariance to round-off. For a nonlinear forward model the
result is an approximation.

References
----------
Iglesias, M. A., Law, K. J. H. & Stuart, A. M. (2013). Ensemble Kalman
methods for inverse problems. *Inverse Problems*, 29(4), 045001.
"""

from ._driver import advance, assimilate, evaluate, iterate, run
from ._helpers import effective_sample_size, misfits, repair_failed_particles
from ._schedules import (
    AdaptiveESSSchedule,
    AdaptiveMisfitSchedule,
    DiscrepancyStop,
    FixedSchedule,
    Schedule,
    StoppingRule,
)
from ._values import (
    INTERRUPTED,
    PREDICTION,
    SCHEDULE_EXHAUSTED,
    STOPPING_RULE,
    EKIError,
    EKIResult,
    EKIState,
    Evaluation,
    HistoryRecord,
)

__all__ = [
    "EKIState",
    "Evaluation",
    "HistoryRecord",
    "EKIResult",
    "Schedule",
    "StoppingRule",
    "FixedSchedule",
    "AdaptiveESSSchedule",
    "AdaptiveMisfitSchedule",
    "DiscrepancyStop",
    "run",
    "iterate",
    "evaluate",
    "assimilate",
    "advance",
    "misfits",
    "effective_sample_size",
    "repair_failed_particles",
    "PREDICTION",
    "SCHEDULE_EXHAUSTED",
    "STOPPING_RULE",
    "INTERRUPTED",
    "EKIError",
]

# The modules above are private, so the public names report the package they
# are imported from, in tracebacks, ``type()`` and pickles.
for _name in __all__:
    if not isinstance(globals()[_name], str):
        globals()[_name].__module__ = __name__
del _name
