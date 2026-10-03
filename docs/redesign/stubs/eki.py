r"""Ensemble Kalman Inversion: a ladder of tempered targets, and the run along it.

For a prior :math:`\pi_0`, a forward model :math:`G`, data :math:`y` and a
noise covariance :math:`R` with whitener :math:`W`, the targets are

.. math::

    \pi_\beta(u) \propto \pi_0(u)\, e^{-\beta\,\Phi(G(u))}, \qquad
    \Phi(g) = \tfrac12 \lVert W (y - g) \rVert^2, \qquad \beta \ge 0 .

Since :math:`\pi_{\beta+\delta}/\pi_\beta \propto \mathcal N(y;\, G(u),\, R/\delta)`,
moving from :math:`\beta` to :math:`\beta + \delta` is conditioning on
:math:`y` with noise :math:`R/\delta`, so a step is one
:func:`enskit.kalman.update`. Everything specific to inversion lives
here: the level, the schedule, the stopping rule, failed particles, history.

==================================== ==========================================
object                               is
==================================== ==========================================
:class:`EKIState`                    the loop-carried state
:class:`Evaluation`                  one forward evaluation and its summaries
:class:`HistoryRecord`               one row of the run's history
:class:`EKIResult`                   final state, history, and why the run ended
:class:`Schedule`, :class:`StoppingRule`
                                     the policy protocols (inflation and
                                     relaxation are shared with every driver;
                                     see :mod:`enskit.algorithms`)
:class:`FixedSchedule`,              schedules
:class:`AdaptiveESSSchedule`,
:class:`AdaptiveMisfitSchedule`
:class:`DiscrepancyStop`             stopping rule
:func:`run`, :func:`iterate`         the driver, as a function and a generator
:func:`evaluate`, :func:`assimilate`, one step, as its two phases and
:func:`advance`                      their composition
:func:`misfits`,                     array-level helpers
:func:`effective_sample_size`,
:func:`repair_failed_members`
:class:`EKIError`                    raised when a run cannot continue
==================================== ==========================================

The update rule is any :class:`enskit.kalman.UpdateRule` and is required:
``update_rule=SymmetricSquareRoot()``, ``Matheron()``, ``LocalizedUpdateRule(...)`` or your own.

Conventions: the state's ensemble holds the parameter blocks; the forward
model receives them positionally in block order (or the subset named by
``inputs``) and its output becomes the block :data:`PREDICTION`, a name the
state's ensemble may not use. Steps take increments: the per-step noise is
``R / delta``, never ``R / beta``. The misfit carries the factor 1/2 and is
always measured against the base ``R``.

References
----------
.. [1] Iglesias, M. A., Law, K. J. H. & Stuart, A. M. (2013). Ensemble Kalman
   methods for inverse problems. Inverse Problems, 29(4), 045001.
"""

PREDICTION: str = "prediction"
SCHEDULE_EXHAUSTED: str = "schedule_exhausted"
STOPPING_RULE: str = "stopping_rule"
INTERRUPTED: str = "interrupted"


class EKIState:
    """The state carried from step to step.

    Parameters
    ----------
    ensemble : Ensemble
        Unweighted, ``n_particles >= 2``; its blocks are the parameters.
    beta : float or Array
        Keyword-only. The tempering level reached; 0 at the prior.
    step : int
        Keyword-only. Steps completed; static (a Python int).
    key : jax.random key
        Keyword-only. Typed key, split once per step into
        ``(next, inflate, update)`` whatever the policies, so toggling
        inflation never shifts the update's random stream.
    """

    @classmethod
    def from_prior(cls, key, prior: Gaussian, n_particles: int) -> EKIState:
        """Draw the initial ensemble from ``prior``; ``beta = 0``, ``step = 0``.

        Splits ``key`` into ``(sample, state)``.
        """

    def restart(self) -> EKIState:
        """The same particles and key with ``beta = 0`` and ``step = 0``.

        For chaining a fresh ladder onto a finished run; without it a budgeted
        schedule is already exhausted and the run does nothing.
        """


class Evaluation:
    """What one forward evaluation produced, and the summaries policies read.

    Attributes
    ----------
    step : int
        Static.
    beta : Array
    ensemble : Ensemble
        The evaluated particles (after inflation and repair) with the
        :data:`PREDICTION` block appended.
    whitened_residuals : Array
        ``(n_particles, N)``, ``W (y - g_j)`` against the base noise covariance,
        with ``N`` the data dimension.
    n_valid : Array
        Particles whose evaluation succeeded.

    Derived properties: ``misfits`` ``(n_particles,)``, ``center_misfit`` (the
    misfit of the mean prediction), ``n_particles``.
    """


class HistoryRecord:
    """The record of one step of a run: what the step started from, what it did, and the
    ensemble's summaries at its evaluation. Every field is a 0-d array, so the
    records of a run stack into arrays (``EKIResult.stacked``).

    Fields: ``step``, ``n_valid``, ``beta``, ``increment``, ``beta_next``,
    ``misfit_mean``, ``misfit_min``, ``misfit_max``, ``center_misfit``,
    ``spread``, ``ess`` (at the increment taken). A terminal record, written
    when a run ends on a stop or an exhausted schedule, has ``increment == 0``.
    """


class EKIResult:
    """The outcome of :func:`run`. A frozen dataclass, not a pytree.

    Attributes
    ----------
    state : EKIState
    history : tuple[HistoryRecord, ...]
    status : str
        :data:`SCHEDULE_EXHAUSTED` or :data:`STOPPING_RULE`.
    last_evaluation : Evaluation

    Properties: ``ensemble``, ``beta``, ``mean`` (per block), ``n_evaluations``,
    ``n_completed_steps``, ``min_n_valid``, ``stacked`` (the history as one
    record of arrays).
    """


class Schedule(Protocol):
    """How large the next increment is.

    Attributes
    ----------
    n_steps : int | None
        Exhaustion by count; ``None`` for unbounded.
    beta_target : float | None
        Exhaustion by level; ``None`` for unbounded.
    """

    n_steps: int | None
    beta_target: float | None

    def next_increment(self, evaluation: Evaluation) -> Array | None:
        """The next increment, finite and strictly positive, or ``None`` to stop.

        Pure: no trial updates and no trial evaluations.
        """


class FixedSchedule:
    """A fixed tuple of increments.

    Parameters
    ----------
    increments : tuple[float, ...]
        Static Python floats. Increments summing to exactly 1 (binary-exact
        values such as 1/16) reach the posterior; ES-MDA is this schedule
        with ``increments = (1/alpha_1, ..., 1/alpha_T)``.

    Constructors ``FixedSchedule.constant(increment, n_steps)`` and
    ``FixedSchedule.uniform(n_steps)``.

    References
    ----------
    .. [1] Emerick, A. A. & Reynolds, A. C. (2013). Ensemble smoother with
       multiple data assimilation. Computers & Geosciences, 55, 3–15.
    """


class AdaptiveESSSchedule:
    r"""Choose the increment that keeps the importance-weight ESS at a fraction of ``J``.

    .. math::

        \delta^* = \sup\{\delta : \mathrm{ESS}(\delta) \ge f J\}, \qquad
        \mathrm{ESS}(\delta) = \frac{\big(\sum_j e^{-\delta \Phi_j}\big)^2}{\sum_j e^{-2\delta \Phi_j}},

    with :math:`f` = ``ess_fraction``, found by bisection in log space, then
    clamped to
    ``[min_increment, max_increment]`` and to the remaining budget.

    Parameters
    ----------
    ess_fraction : float
        In ``(0, 1 - 1e-6]``; default 0.5.
    beta_target : float or None
        Default 1.0 (the posterior). ``None`` for an unbounded ladder.
    min_increment, max_increment : float
        Defaults ``1e-3`` and ``1.0``. The floor beats the criterion; the budget
        beats the floor.
    n_bisect : int
        Default 50.

    References
    ----------
    .. [1] Jasra, A., Stephens, D. A., Doucet, A. & Tsagaris, T. (2011).
       Inference for Lévy-driven stochastic volatility models via adaptive
       sequential Monte Carlo. Scandinavian Journal of Statistics, 38(1),
       1–22.
    """


class AdaptiveMisfitSchedule:
    r"""Choose the increment from the misfit's mean and spread (data-misfit controller).

    .. math::

        \delta^* = \max\Big(\frac{\theta}{\overline{\Phi}},\ \sqrt{\frac{\theta}{\operatorname{var}(\Phi)}}\Big),

    with :math:`\theta` = ``divergence_budget`` (default :math:`N/2`), then
    clamped as in
    :class:`AdaptiveESSSchedule`. Takes larger steps than the ESS rule; prefer
    it when the fit, not the ensemble, is the deliverable.

    References
    ----------
    .. [1] Iglesias, M. & Yang, Y. (2021). Adaptive regularisation for
       ensemble Kalman inversion. Inverse Problems, 37(2), 025008.
    """


class StoppingRule(Protocol):
    def __call__(self, evaluation: Evaluation) -> bool:
        """Whether to stop before taking the next step. Pure."""


class DiscrepancyStop:
    r"""Stop by the discrepancy principle.

    .. math::

        2\,\Phi(\bar g) \le \tau^2 N,

    with :math:`\bar g` the mean prediction and :math:`N` the data dimension.

    Parameters
    ----------
    tau : float
        Default 1.0. :math:`\tau^2 = 1 + k\sqrt{2/N}` sets the threshold
        :math:`k` standard deviations above the misfit's expectation.

    References
    ----------
    .. [1] Iglesias, M. A. (2016). A regularizing iterative ensemble Kalman
       method for PDE-constrained inverse problems. Inverse Problems, 32(2),
       025002.
    """


def evaluate(state: EKIState, forward, y, noise_cov, *, inflation=None, inputs=None,
             on_failure: str = "raise") -> Evaluation:
    """Phase 1 of a step: inflate, evaluate the forward model once, repair, summarize.

    ``inflation`` (an :class:`enskit.algorithms.Inflation`) is called as
    ``inflation(key_inflate, ensemble=..., step=..., beta=...)``.

    ``on_failure`` is ``"raise"`` (an :class:`EKIError` naming the failed
    particles) or ``"repair"`` (failed particles are moved to the valid particles'
    center, bit-exact when nothing failed). Fewer than two valid particles
    always raises.
    """


def assimilate(state: EKIState, evaluation: Evaluation, increment, y, noise_cov, *,
               update_rule, approximation=None) -> EKIState:
    """Phase 2 of a step: one Kalman update with noise ``noise_cov / increment``.

    Equivalent to ``kalman.update(evaluation.ensemble, {PREDICTION: y},
    noise={PREDICTION: noise_cov / increment}, update_rule=rule, approximation=joint,
    key=k_update)``,
    plus validation of the increment (finite, strictly positive), of
    provenance (the evaluation belongs to this state's step and level) and of
    the result (finite, same shape and dtype).
    """


def advance(state, forward, y, noise_cov, increment, *, update_rule, inflation=None,
            inputs=None, on_failure="raise", approximation=None) -> EKIState:
    """``assimilate(state, evaluate(state, ...), increment, ...)``."""


def iterate(state, forward, y, noise_cov, *, update_rule, schedule, stop=None, inflation=None,
            relaxation=None, inputs=None, on_failure="raise", approximation=None, max_steps=1000):
    """The driver as a generator: yields ``(state, HistoryRecord)`` after each step.

    Lets a caller inspect, plot, checkpoint or interrupt between steps. The
    order before each step is normative: exhaustion check, ``max_steps``
    bound, evaluate, stopping rule, increment.
    """


def run(state, forward, y, noise_cov, *, update_rule, schedule, stop=None, inflation=None,
        relaxation=None, inputs=None, on_failure="raise", approximation=None,
        max_steps=1000) -> EKIResult:
    """Run EKI until the schedule is exhausted or the stopping rule fires.

    Parameters
    ----------
    state : EKIState
    forward : callable
        The simulator; see the simulator contract in :mod:`enskit.maps`. Called
        once per step with every particle, never traced.
    y : Array
        ``(N,)``, finite.
    noise_cov : PSDLinOp
        Used only through ``whiten``.
    update_rule : UpdateRule
        Keyword-only and required.
    schedule : Schedule
        Keyword-only.
    stop : StoppingRule, optional
    inflation, relaxation : optional
        Shared policies from :mod:`enskit.algorithms`; inflation runs before
        each evaluation, relaxation after each update.
    approximation : callable, optional
        ``(ensemble, noise) -> Gaussian``, the joint Gaussian approximation each
        update conditions; see :func:`enskit.kalman.update`.
    inputs : sequence of str, optional
        Parameter blocks passed to ``forward``; all, in order, by default.
    on_failure : {"raise", "repair"}
    max_steps : int
        A safety bound counted over this call's steps; exceeding it raises
        :class:`EKIError`.

    Raises
    ------
    EKIError
        Carrying ``.state`` (the last good state) and ``.history``, so
        ``run(exc.state, ...)`` resumes exactly.
    """


def misfits(y, predictions, noise_cov) -> Array:
    r""":math:`\Phi(g) = \tfrac12 \lVert W(y - g) \rVert^2` per row; ``(..., N) -> (...)``."""


def effective_sample_size(misfits, increment) -> Array:
    """ESS of the tempering weights ``exp(-increment * misfits)``, in log space."""


def repair_failed_members(*, ensemble: Ensemble, valid: Array) -> Ensemble:
    """Move invalid particles to the valid particles' center, every block alike."""


class EKIError(RuntimeError):
    """A run could not continue. ``.state`` and ``.history`` resume it."""
