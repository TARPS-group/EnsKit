# Ensemble Kalman Inversion contract

This page specifies `enskit.algorithms.eki`, the Ensemble Kalman Inversion
driver, and the inflation and relaxation policies of `enskit.algorithms` that
every driver shares: the objects, the contract of every function, and the step
all of them serve. It is normative: an implementation that violates a rule
here is defective even if its tests pass. It is written for contributors
implementing or reviewing the layer, and for users who want a more precise
account of what a run computes than {doc}`user-guide/running-an-inversion`
gives.

*Must* and *never* state requirements, *should* states a strong default that a
documented reason may override, and *may* states a permission. The driver is
built on {doc}`maps-contract` (the forward model is evaluated by
`pushforward`) and {doc}`kalman-contract` (each step is one `update`), and
refers to those pages and to {doc}`distribution-contract` rather than restating
them. {doc}`design` records why the load-bearing decisions were made.

:::{admonition} Status: revised and implemented in PR 7
:class: note

This page specified the old `enskit.eki` module, built directly on the
retired `enskit.gauss`. PR 7 of the redesign plan revised it for the new
layers and implemented it as `enskit.algorithms.eki`, deleting the old module.
The numerical rules carry over unchanged (per-step noise $R/\Delta\beta$, the
exhaustion check before the bound, the log-space ESS, the guarded divisions,
the bit-exact skip of a repair when nothing failed); what changed is listed in
{ref}`eki-changes`. Where this page departs from the design's stubs
(`docs/redesign/stubs/eki.py` and `algorithms_policies.py`), it says so in
{ref}`eki-departures`, and this page wins.
:::

(eki-scope)=
## Scope

The layer turns one ensemble Kalman update into a *run*: an initial ensemble,
a ladder of tempered targets, an update per step, and a record of what
happened. It provides

- **the step**, as two public phases and a driver that owns the loop, the
  tempering level, the random stream, and the handling of failed particles;
- **tempering schedules**, fixed and adaptive, which decide how far each step
  moves, and **stopping rules**;
- **per-step diagnostics** and the run's history;
- **inflation and relaxation policies**, in `enskit.algorithms`, shared with
  every driver.

Two well-known forms of EKI come from the same driver, and neither is
privileged ({ref}`eki-axes`):

- the **sampling form** draws from the prior and runs a ladder of tempered
  targets to $\beta = 1$, giving an approximate posterior ensemble
  (Iglesias, Law & Stuart, 2013; Emerick & Reynolds, 2013);
- the **optimization form** runs unit steps without a budget until the data
  are fit, giving a collapsing ensemble whose center approximates a
  regularized least-squares solution (Iglesias, 2016).

**This layer conditions nothing itself.** Each step evaluates the forward
model with {func}`enskit.maps.pushforward` and conditions with
{func}`enskit.kalman.update`; the update rule, the Gaussian approximation and
every whitening, decomposition and draw belong to those layers. What this
layer computes is elementwise: whitened residuals' norms, log-space weight
ratios, the repair's masked means, and the clamps of the schedules.

Outside the layer: the forward model (any simulator, {ref}`maps-simulators`),
parameter transformations and constraints, and localization (an update rule,
PR 9). {ref}`eki-excluded` lists what else is left out and why.

(eki-notation)=
## Notation and conventions

| symbol | meaning |
| ------ | ------- |
| $u$ | the parameters: every block of the state's ensemble, $P$ coordinates in all |
| $g$ | the forward model's output, the block `PREDICTION`, of dimension $N$ |
| $G$ | the forward model, $u \mapsto g$ |
| $y$ | the data, a `(N,)` array |
| $R$, $W$ | the noise covariance, a `PSDLinOp`, and a whitener of it, $W^\top W = R^{-1}$ |
| $J$ | the number of particles |
| $\beta$ | the tempering level, $\beta \ge 0$ |
| $\Delta\beta_t$ | the increment of step $t$, strictly positive |
| $\Phi_j$ | particle $j$'s misfit (below) |

(eki-terminology)=
### Terminology

Five words name the parts of a run, and each means exactly one thing in this
page, the docstrings and the code.

| term | means | counted by |
| ---- | ----- | ---------- |
| **run** | one call of `run` or `iterate`, or a chain of them over one state | — |
| **step** | one update: it takes an increment, advances $\beta$ and `EKIState.step` | `EKIResult.n_completed_steps`; `Schedule.n_steps`; bounded by `max_steps` |
| **evaluation** | one call of the forward model, with every particle | `EKIResult.n_evaluations`, one per `HistoryRecord` |
| **phase** | `evaluate` or `assimilate`, the two public halves of a step | — |
| **operation** | one of the numbered actions inside a step ({ref}`eki-step`) | — |

**Steps and evaluations differ by one exactly when a run ends on a decision
that needed an evaluation**: a stopping rule that fired, or a schedule whose
`next_increment` returned `None`. A ladder that ends declaratively
(`Schedule.n_steps` reached, or `beta_target` reached) spends no evaluation on
the decision, because that check reads only `step` and $\beta$. So

$$
n_{\text{evaluations}} = n_{\text{steps}} + \begin{cases}
1 & \text{ended on a terminal evaluation,}\\
0 & \text{ended declaratively,}
\end{cases}
$$

and a run's cost in particle evaluations is $J\,n_{\text{evaluations}}$.
`max_steps` is checked before the forward model is called, so it bounds the
evaluations of a call by the same number. *Rung*, and *iteration* as a
countable noun, are retired.

Conventions, each normative:

- **The misfit carries the factor $\tfrac12$** and is measured against the
  base noise covariance:

  $$
  \Phi(g) = \tfrac12\lVert W(y - g)\rVert^2 = \tfrac12 (y-g)^\top R^{-1}(y-g).
  $$

  With it $e^{-\beta\Phi}$ is the tempered likelihood and $2\overline\Phi
  \approx N$ the benchmark of a well-specified fit. It is whitener-invariant.
- **The state carries a level, and steps take increments.** A step takes
  $\Delta\beta$ and conditions with noise $R/\Delta\beta$, never $R/\beta$.
- **The state's ensemble holds the parameter blocks**, and the forward
  model's output becomes the block `PREDICTION` (the string `"prediction"`),
  a name the state may not use.
- **Keys are typed keys**, consumed whole ({ref}`eki-prng`).
- **The loop is ordinary Python**, never `lax.scan`: the forward model may be
  a subprocess or a scheduler submission, and is never traced.

(eki-algorithm)=
## The algorithm

### The tempered family

For a prior $\pi_0$, the targets are

$$
\pi_\beta(u) \propto \pi_0(u)\, e^{-\beta\Phi(G(u))}, \qquad \beta \ge 0,
$$

the prior at $\beta = 0$ and the posterior at $\beta = 1$, concentrating on
the minimizers of $\Phi \circ G$ as $\beta \to \infty$. For any $\delta > 0$,

$$
\frac{\pi_{\beta+\delta}(u)}{\pi_\beta(u)} \propto e^{-\delta\Phi(G(u))}
\propto \mathcal N\big(y;\, G(u),\, R/\delta\big),
$$

so **moving one increment up the ladder is conditioning on $y$ with noise
$R/\delta$**. $R/\delta$ is a {class}`~enskit.linalg.PSDScaled`, whose
whitener is $\sqrt\delta\,W$, so nothing is refactorized.

### One step

Step $t$ carries the particles from $\beta_t$ to $\beta_{t+1} = \beta_t +
\Delta\beta_t$:

1. evaluate: `pushforward(particles, forward, inputs=..., output=PREDICTION)`,
   one call of the forward model with every particle, appending
   $g_j = G(u_j)$;
2. update: `kalman.update(evaluated, {PREDICTION: y}, noise={PREDICTION: R /
   delta}, update_rule=..., approximation=..., key=...)`, which by default
   fits the moment-matching Gaussian to the pairs $(u_j, g_j)$, adds the noise
   covariance to the prediction block, and conditions it on $y$
   ({ref}`kalman-update`).

Two approximations are inherent: the joint law of $(u, G(u))$ under
$\pi_{\beta_t}$ is replaced by a Gaussian, and its moments by $J$-particle
estimates. Both are exact when $G$ is affine and the particles' sample moments
equal $\pi_{\beta_t}$'s. A configured run may add others, each opt-in and
named where it arises: the stochastic rule's Monte Carlo noise
({ref}`eki-updates`), inflation and relaxation ({ref}`eki-inflation`), and
repairing failed particles ({ref}`eki-failures`).

### Telescoping

Per-step precisions add. For an affine $G$ and a Gaussian prior, each step
contributes $\Delta\beta_t\, G^\top R^{-1} G$ to the precision, so after $T$
steps it is $\Lambda_0 + (\sum_t \Delta\beta_t)\, G^\top R^{-1} G$: **the
ladder composes to one-shot conditioning at $\beta_T$**, and at $\beta_T = 1$
to the posterior. This is the layer's central correctness property and its
first conformance obligation.

:::{warning}
Using $R/\beta_t$ instead of $R/\Delta\beta_t$ is the layer's signature
silent failure. On a uniform $T$-step ladder it accumulates $\sum_t t/T =
(T+1)/2$ times the data precision instead of one, a factor of 3 at $T = 5$,
and produces a plausible posterior. Obligation 2 pins it.
:::

The identity holds exactly for the sample moments under
`SymmetricSquareRoot`, in expectation under `Matheron`, and not at all with
inflation or relaxation, which change the target on purpose.

(eki-subspace)=
### The subspace property

Both shipped rules move each particle by a combination of the particles' own
anomalies, so by induction **every particle of a run lies in the affine
subspace spanned by the initial particles**, of dimension at most $J - 1$,
however many steps are run. $J$ bounds what a run can represent, not only how
accurately it estimates moments. Multiplicative inflation and both relaxations
stay in the subspace; additive inflation is the only shipped mechanism that
leaves it; localization (PR 9) escapes it by giving each neighborhood its own
combination.

(eki-honesty)=
### What the layer does not promise

- **Exactness is claimed for the linear-Gaussian case only**: an affine $G$,
  a Gaussian prior, particles whose sample moments equal the prior's,
  `SymmetricSquareRoot`, no inflation or relaxation, no failed particles, and
  increments summing *exactly* to 1. Then the run reproduces the posterior
  mean and covariance to round-off. `FixedSchedule.uniform(T)` sums to 1 only
  to round-off, so it meets the claim only to round-off.
- **For a nonlinear $G$ the result is an approximation** with no consistency
  guarantee. It depends on the schedule, $J$ and the rule; two ladders give
  two answers and neither bounds the other.
- **Particles are not independent posterior draws.** The sample gain couples
  them; uncertainty is carried by the spread.
- **The optimization form collapses the ensemble on purpose**; its terminal
  spread measures convergence, not uncertainty.
- **Nothing in the loop re-consults the prior.** It enters once, through the
  initial particles, which is why a prior covariance with no cheap inverse is
  usable.
- **A run's cost is not knowable in advance under an adaptive schedule.** A
  caller who must budget evaluations uses a `FixedSchedule` or drives the
  phases by hand. Exceeding `max_steps` raises, carrying the state.

(eki-axes)=
## The choices a run is made of

A variant of EKI is a choice on independent axes, and the driver is the same
in every case.

| axis | argument | question it answers |
| ---- | -------- | ------------------- |
| how far, and when to stop | `schedule`, `stop` | what is $\Delta\beta_t$, and when does the run end? |
| how the particles move | `update_rule`, `approximation` | given the increment, what are the new particles? |
| how spread is maintained | `inflation`, `relaxation` | what happens before each evaluation, and after each update? |

The two forms are two schedules, not two drivers and not a flag:

```python
from enskit import kalman
from enskit.algorithms import eki

state = eki.EKIState.from_prior(key, prior, n_particles=64)

# Sampling form: an adaptive ladder to beta = 1.
sampled = eki.run(state, forward, y, noise_cov,
                  update_rule=kalman.Matheron(),
                  schedule=eki.AdaptiveESSSchedule())
particles = sampled.ensemble["u"]      # (64, P) approximate posterior particles
center = sampled.mean("u")             # (P,) their mean

# Optimization form: unit steps, no budget, stop on the discrepancy principle.
fit = eki.run(state, forward, y, noise_cov,
              update_rule=kalman.SymmetricSquareRoot(),
              schedule=eki.FixedSchedule.constant(1.0, n_steps=200),
              stop=eki.DiscrepancyStop(tau=1.0), max_steps=200)
fit.stop_fired                         # False means the ladder ran out first
```

Each axis is a protocol, not a base class. The axes compose without
restriction, and the driver never inspects one to decide another. That is not
a claim that every combination is meaningful:

| combination | reading |
| ----------- | ------- |
| a budgeted schedule, no inflation or relaxation | the sampling form; every claim of {ref}`eki-honesty` applies |
| an unbounded schedule and a stopping rule | the optimization form; the terminal spread is not an uncertainty |
| a budgeted schedule with inflation or relaxation | a deliberately widened variant, not a ladder for the target family |
| `DiscrepancyStop` on a budgeted ladder | constructible and usually a mistake: it can end a sampling run at $\beta = 0.4$; `stop_fired` and `budget_complete` make it visible |

(eki-objects)=
## The objects

| object | kind | role |
| ------ | ---- | ---- |
| `EKIState` | pytree | the loop-carried state: particles, level, step, key |
| `Evaluation` | pytree | one evaluation: the evaluated particles with `PREDICTION`, and the whitened residuals |
| `HistoryRecord` | pytree | the record of one step, every field a 0-d array |
| `EKIResult` | frozen dataclass | the final state, the history, and why the run ended |
| `Schedule`, `StoppingRule` | protocols | the increment and the ladder's length; whether to stop |
| `FixedSchedule`, `AdaptiveESSSchedule`, `AdaptiveMisfitSchedule`, `DiscrepancyStop` | policies | the shipped ones |
| `evaluate`, `assimilate`, `advance` | functions | one step, as its two phases and their composition |
| `run`, `iterate` | functions | the driver, as a function and a generator |
| `misfits`, `effective_sample_size`, `repair_failed_particles` | functions | the array-level pieces |
| `Inflation`, `Relaxation` and four policies | in `enskit.algorithms` | shared with every driver ({ref}`eki-inflation`) |

Rules governing the set:

1. **The three value classes are unbatched frozen pytrees** with explicitly
   declared data and static fields, as in the layers below: unflattening
   bypasses the constructor, objects compare by identity, a pytree rebuilt
   with stacked leaves is a vmapped family that refuses its methods
   ({ref}`eki-jax`). `EKIResult` is a report, never an argument to traced
   code, so it is a plain frozen dataclass.
2. **The protocols are structural.** Anything with the right members is a
   schedule, a stopping rule, an inflation or a relaxation; a plain function
   serves where the protocol is a single call. Update rules are
   {class}`enskit.kalman.UpdateRule`s.
3. **Policies are pure and stateless**: a function of their arguments and
   their own fields, holding no state across steps. This is what makes a run
   resumable from an `EKIState` alone, and why a schedule receives the step
   index rather than counting calls. `enskit.testing` checks it
   ({ref}`eki-testing`).

(eki-state)=
## `EKIState`

`EKIState(ensemble, *, key, beta=0.0, step=0)`: everything a run needs to
continue.

- `ensemble`: an unweighted {class}`~enskit.distribution.Ensemble` whose
  blocks are the parameters, not a vmapped family, with no block named
  `PREDICTION` (`TypeError` for another type; `ValueError` otherwise). An
  update takes unweighted particles, so a weighted ensemble is refused with
  the remedy (`resample`). In debug mode every particle must be finite.
- `key`: keyword-only and **required**, a typed key of shape `()`
  (`TypeError` for a raw `uint32` key or any other value, `ValueError` for
  another shape). It is the run's only source of randomness.
- `beta`: keyword-only, a real scalar stored as a 0-d floating array
  (`ValueError` for another shape, `TypeError` for a `bool`); in debug mode
  finite and not negative.
- `step`: keyword-only, a non-negative Python `int`, static (`TypeError`,
  `ValueError`). Position-dependent schedules index with it.

Fields in declaration order `ensemble`, `beta`, `key` (data) and `step`
(static). Derived: `n_particles`, `dims`, `mean(name)`, `batch_shape`.

**`EKIState.from_prior(key, prior, n_particles)`** draws the initial particles
from a {class}`~enskit.distribution.Gaussian` over the parameter blocks,
pinned as

```python
key_sample, key_state = jax.random.split(key)
EKIState(prior.sample(key_sample, n_particles), key=key_state)
```

so the particles are `Gaussian.sample`'s draw and the state's stream is
independent of it. A covariance that cannot `factor` raises
`UnsupportedOpError` from the operator layer, unmodified. A warm start (a
previous run's particles, a design) is direct construction; nothing requires
the initial particles to come from a prior.

**`restart()`** returns the same particles and key at `step = 0`, `beta =
0.0`.

**Resumption.** `run(state, ...)` on a state returned by an earlier run, or
carried by an `EKIError`, continues it, and the rest of the run is
bit-identical to an uninterrupted one. This is the sole mechanism for
checkpointing, and why policies may not hold state.

:::{warning}
**`step` and `beta` carry across runs.** That is what lets an interrupted
ladder resume at its next increment. It also makes chaining a *new* ladder
onto a finished state a no-op: a `FixedSchedule` finds `step >= n_steps`, a
budgeted schedule finds `beta` at its target, and the run returns at once
with an empty history and the particles unchanged. A new ladder needs
`state.restart()`. The driver logs a run that made no evaluation at
`WARNING`.
:::

(eki-step)=
## The step, in two phases

| function | does | returns |
| -------- | ---- | ------- |
| `evaluate(state, forward, y, noise_cov, *, inflation=None, inputs=None, on_failure="raise")` | inflate, evaluate, find failures, repair or raise, whiten | an `Evaluation` |
| `assimilate(state, evaluation, increment, y, noise_cov, *, update_rule, approximation=None, relaxation=None)` | validate, update, relax, check finiteness, build the state | the next `EKIState` |
| `advance(state, forward, y, noise_cov, increment, *, update_rule, inflation=None, relaxation=None, inputs=None, on_failure="raise", approximation=None)` | `assimilate(state, evaluate(state, ...), increment, ...)` | the next `EKIState` |

The phases are public because the evaluation is the resource a run is
organized around: one `Evaluation` serves any number of trial increments, and
a schedule or stopping rule can be consulted outside the driver.

**The order of operations is normative.** Operations 1–5 are `evaluate`;
0 and 6–9 are `assimilate`.

1. **Split the key**, always into four: `key_next, key_inflate, key_evaluate,
   key_update = jax.random.split(state.key, 4)`. Fixed arity means turning
   inflation on, or using a simulator that needs a key, never shifts the
   update's draws. Both phases split independently from `state.key`; the
   split is deterministic, so they agree, and no key is put on the
   `Evaluation` every schedule and stopping rule receives.
2. **Inflate**: `inflation(key_inflate, ensemble=state.ensemble,
   step=state.step, beta=state.beta)`, checked to return an unweighted
   ensemble with the state's blocks, order, dimensions, count and dtype; or
   `state.ensemble` unchanged when `inflation is None`. Inflating here, on the
   particles about to be evaluated, keeps predictions matched to the
   particles they update and keeps the returned ensemble uninflated.
3. **Evaluate**: `maps.pushforward(particles, forward, inputs=inputs,
   output=PREDICTION, key=key_evaluate)`, the only call of the forward model
   ({ref}`eki-failures`). Its output must have $N$ columns (`ValueError`).
4. **Find failures, then raise or repair.** A particle is *failed* when its
   prediction has a non-finite entry; $J_v$ counts the others. If $J_v = J$
   the particles pass through untouched, bit for bit; if $J_v < 2$ raise
   `EKIError`; if `on_failure == "raise"` raise `EKIError` naming the failed
   particles; otherwise `repair_failed_particles` on every block.
5. **Whiten and summarize**: `whitened_residuals = noise_cov.whiten(y -
   g)`, after repair, and build the `Evaluation`.
0. **Validate**, in `assimilate`, before anything else: the increment, a
   real scalar, finite and **strictly positive** (`ValueError`; a zero
   increment raises nothing downstream but returns the particles unchanged
   while $\beta$ never advances, so an adaptive ladder would spin to
   `max_steps`); that the evaluation is of this state's step and level, and
   particle count (`ValueError`); the update rule has `build` (`TypeError`);
   the approximation and relaxation are callable.
6. **Update**: `kalman.update(evaluation.ensemble, {PREDICTION: y},
   update_rule=update_rule, noise={PREDICTION: noise_cov / increment},
   approximation=approximation, key=key_update)`, the increment converted to
   the particles' dtype. Its result is the parameter blocks, checked by
   `update` ({ref}`kalman-update`) and put in the state's block order.
7. **Relax**: `relaxation(prior=evaluation.ensemble, posterior=...,
   step=state.step, beta=state.beta)`, checked like an inflation's result; or
   the update's result when `relaxation is None`.
8. **Finiteness**: a non-finite particle raises `EKIError` naming the step,
   the level and the rule. Silent `nan` through a long run is the worst
   outcome available to this layer, and the check is one reduction.
9. **Build the state**: `EKIState(posterior, key=key_next, beta=state.beta +
   increment, step=state.step + 1)`.

The provenance check of operation 0 compares position only, so it catches an
evaluation the state has moved past but **not** an evaluation of another
problem at the same position (every fresh run sits at `step = 0`, `beta = 0`).
Making it exact would put key material on the evaluation; keeping two runs
apart is the caller's, which is why `run` and `iterate` bind the problem for a
whole run.

One step reads three device values (the valid count, the increment, the
updated particles' finiteness), and a run through the driver adds the budget
check's read of `beta` and a stopping rule's own. The reads cannot be
coalesced: the increment decides whether to update, and the finiteness check
reads what the update produced.

A record is built from an evaluation and an increment by
**`HistoryRecord.from_evaluation(evaluation, increment=None)`**, as the driver
builds every record, so a hand-written loop keeps the driver's history.

(eki-step-backtracking)=
### Backtracking, as the two phases make it

States are immutable, so assimilating again from the same state at a smaller
increment is "reject and retry", and a rejected trial costs no evaluation of
the current state:

```python
rule = kalman.SymmetricSquareRoot()
s, delta = state, 1.0
current = eki.evaluate(s, forward, y, noise_cov)
while not done(current):
    trial = eki.assimilate(s, current, delta, y, noise_cov, update_rule=rule)
    probe = eki.evaluate(trial, forward, y, noise_cov)
    if probe.center_misfit < current.center_misfit:
        s, current, delta = trial, probe, delta * 1.5     # accept, lengthen
    else:
        delta = delta / 2                                 # reject, reuse current
```

The accepted branch reuses `probe` as the next evaluation, so the cost is one
evaluation per step plus one per rejection.

### What the driver adds

`run` and `iterate` wrap the phases with the decisions they refuse to make,
in this order, before each step:

1. **Exhaustion.** If the schedule's attributes say the ladder is finished
   ({ref}`eki-schedules`), end the run with `SCHEDULE_EXHAUSTED`. Checked
   before evaluating, so a $T$-step ladder costs exactly $T$ evaluations.
2. **The bound.** If this call has taken `max_steps` steps, raise `EKIError`
   naming `max_steps`, the schedule, and whether a stopping rule was given.
   `max_steps` counts the steps of *this call*, not `state.step`, so a resumed
   run gets the bound its caller asked for. **The order of 1 and 2 is
   normative**: a ladder that exhausts at step $T$ completes under
   `max_steps == T`.
3. **Evaluate** (operations 1–5), exactly once per step.
4. **Stop.** If `stop(evaluation)` is true, end the run with
   `STOPPING_RULE`, recording a terminal record and leaving the state
   unchanged.
5. **Increment.** `schedule.next_increment(evaluation)`; `None` ends the run
   with `SCHEDULE_EXHAUSTED` and a terminal record; any other value is
   validated as in operation 0 and assimilated (operations 6–9).

(eki-updates)=
## Update rules

The update rule is any {class}`enskit.kalman.UpdateRule` and is **required**:
`kalman.SymmetricSquareRoot()`, `kalman.Matheron()`, a localized rule
(PR 9) or your own, written against {ref}`kalman-update-rule`. There is no
default, for the reason the Kalman layer gives: neither shipped rule is right
everywhere.

| rule | character |
| ---- | --------- |
| `SymmetricSquareRoot()` | deterministic: the particles' sample mean and covariance are the fitted joint's conditional ones exactly, per step, so telescoping holds exactly; it keeps the particles' arrangement, rotated and scaled, and on a nonlinear problem can leave a few particles carrying most of the spread (Bishop et al., 2001; Hunt et al., 2007) |
| `Matheron()` | stochastic: each particle moves by $K(y - g_j - e_j)$ with its own noise draw, so the moments are right in expectation, with Monte Carlo error of order $KRK^\top/J$; it redraws the spread for every particle (Burgers et al., 1998; Wilson et al., 2021) |

The rule sees the evaluated particles, the approximation and the given name
`PREDICTION`, and nothing of the run: no increment, step or level. The
increment reaches it only through the noise covariance $R/\Delta\beta$ the
approximation carries. A behavior that varies along the ladder belongs in a
relaxation, which receives `step` and `beta` ({ref}`eki-inflation`), or in a
hand-written loop.

**`approximation`**, a callable `(ensemble, noise) -> Gaussian`, is passed to
`kalman.update` unchanged: by default
{func}`~enskit.kalman.gaussian_approximation`, which projects the evaluated
particles and adds $R/\Delta\beta$ to the prediction block *as a covariance*,
the lower-variance estimate of the same joint. A modified approximation (a
shrinkage estimator, a hybrid covariance) is a plain `Gaussian`; `Matheron`
takes its general path on one, and `SymmetricSquareRoot` refuses it
({ref}`kalman-alignment`).

(eki-schedules)=
## Schedules

A `Schedule` is one method and two attributes.

| member | kind | contract |
| ------ | ---- | -------- |
| `n_steps` | `int \| None` | the ladder's length in steps, or `None` |
| `beta_target` | `float \| None` | the level the ladder ends at, or `None` for an unbounded ladder |
| `next_increment(evaluation)` | method | a finite, strictly positive scalar, or `None` for "finished after all" |

**The driver decides exhaustion**, before each evaluation, from the attributes
alone:

```python
(schedule.n_steps is not None and step >= schedule.n_steps) or (
 schedule.beta_target is not None and beta >= beta_target * (1 - 1e-12))
```

The tolerance is relative, so a small budget is not swallowed whole. The
attributes are read, never called, and constant for the life of the schedule.
A schedule with both `None` is unbounded and must be ended by a stopping rule.
A schedule lacking any of the three members is refused with `TypeError`.

**`next_increment` may end the ladder** by returning `None`, on evidence only
the evaluation carries; it costs the evaluation that produced it, which is
recorded as the terminal record. Neither shipped schedule returns `None`.
`next_increment` is **pure** and performs no trial update and no trial
evaluation: the shipped criteria read only the misfits and the level, so
adaptivity costs $O(J)$ per bisection step and never an evaluation.

### `FixedSchedule(increments)`

A ladder given in advance: a non-empty tuple of finite, strictly positive
real numbers, stored as Python floats (static). `n_steps = len(increments)`,
`beta_target = None`, and `next_increment` returns `increments[step]`, indexed
by the cumulative step, so an interrupted ladder resumes correctly. Its repr
summarizes, `FixedSchedule(n_steps=200, total=200.0)`. The ensemble smoother
with multiple data assimilation is this schedule with the reciprocals of its
inflation factors (Emerick & Reynolds, 2013).

| constructor | increments | form |
| ----------- | ---------- | ---- |
| `FixedSchedule.uniform(n_steps)` | `(1/T,) * T` | a uniform ladder to $\beta = 1$, to round-off |
| `FixedSchedule.constant(increment, n_steps)` | `(c,) * T` | the optimization form when $cT > 1$; one Kalman update at `constant(1.0, 1)` |

(eki-adaptive)=
### Shared semantics of the adaptive schedules

Both adaptive schedules have `n_steps = None` and three fields:

| field | default | meaning |
| ----- | ------- | ------- |
| `beta_target` | `1.0` | the budget, or `None` for an unbounded ladder; positive |
| `min_increment` | `1e-3` | a floor guaranteeing progress; positive |
| `max_increment` | `1.0` | a ceiling; finite, at least the floor |

Each computes a criterion $\delta^\star$ and returns

$$
\delta = \min\big(\max(\delta^\star, \delta_{\min}),\ \delta_{\max},\
\beta_{\text{target}} - \beta\big),
$$

the budget term present only when `beta_target` is not `None`. **The order is
normative**, and both inversions are silent bugs: the floor beats the
criterion, so a step is always taken; the budget beats the floor, so the
ladder never passes its target. A budgeted run therefore takes at most
$\lceil(\beta_{\text{target}} - \beta)/\delta_{\min}\rceil$ further steps, and
**the driver checks that bound against `max_steps` at entry**, raising
`ValueError` before any evaluation (the shipped defaults meet it exactly:
$1/10^{-3} = 1000$). When every misfit is equal no increment changes the
weights, and each schedule takes the largest allowed step.

### `AdaptiveESSSchedule`

Chooses the increment that keeps the tempering weights' effective sample size
at a fraction of $J$ (Jasra et al., 2011), as a step-size rule only: no weights
are carried and nothing is resampled. For an increment $\delta$,

$$
w_j = e^{-\delta\Phi_j}, \qquad
\mathrm{ESS}(\delta) = \frac{(\sum_j w_j)^2}{\sum_j w_j^2}, \qquad
\delta^\star = \sup\{\delta : \mathrm{ESS}(\delta) \ge f J\},
$$

with $f$ = `ess_fraction` in $(0, 1 - 10^{-6}]$ (default 0.5; bounded away
from 1 because $\mathrm{ESS}(0)$ evaluates to `exp(log J)`) and `n_bisect`
bisections (default 50, at least 1). Requirements:

- **Bisect a bracket and return its safe end.** The bracket is $[0,
  \min(\delta_{\max}, \beta_{\text{target}} - \beta)]$; the loop keeps
  $\mathrm{ESS}(\text{lo}) \ge fJ > \mathrm{ESS}(\text{hi})$ and returns `lo`.
  The case where the top already meets the target is folded in branchlessly
  (`lo` starts at the top), so a degenerate ensemble takes the largest step
  exactly. The loop is a `lax.fori_loop`. $\mathrm{ESS}$ has zero derivative
  at $\delta = 0$, which is why the method is bisection.
- **Compute the ESS in log space**, $\exp(2\,\mathrm{lse}(-\delta\Phi) -
  \mathrm{lse}(-2\delta\Phi))$; the direct form underflows to `0/0` at the
  misfits of an early step.
- **Propagate `nan`.** Every comparison against `nan` is false, so a
  bisection would otherwise return its lower bracket and the floor would make
  a poisoned ensemble look like an ordinary small step.

### `AdaptiveMisfitSchedule`

The data misfit controller of Iglesias and Yang (2021), with one extra field,
`divergence_budget` $\theta$, default `None` meaning $N/2$, at which the
schedule has no tuning parameter. With $\overline\Phi$ and $\sigma^2_\Phi$ the
mean and sample variance (divisor $J - 1$) of the misfits,

$$
\delta^\star = \max\Big(\frac{\theta}{\overline\Phi},\ \sqrt{\frac{\theta}{\sigma^2_\Phi}}\Big).
$$

**The `max` is not a `min`.** The Jeffreys divergence between consecutive
targets is approximated by the smaller of two expressions valid in different
regimes, and bounding it by $\theta$ admits every $\delta$ below the larger of
the two thresholds; a `min` would impose the invalid regime's bound too and
stall. The mean bound is the larger when the misfits' coefficient of variation
exceeds $1/\sqrt\theta$, the common case. Both divisions are **guarded**:
$+\infty$ at a zero denominator (a collapsed ensemble, or a perfect fit), so
the clamps give the largest step; `nan` at a `nan` one, so a poisoned misfit
reaches the increment check and raises rather than taking the largest step.
The inner `where` keeps $\theta/0$ from being formed, whose derivative is
`nan`.

At the increment this schedule chooses, the ESS is near its floor of 1: it
takes far longer steps than `AdaptiveESSSchedule` at $f = 1/2$. The two
control different things. Prefer the ESS schedule when the posterior ensemble
is the deliverable, and this one when the fit is, or evaluations are scarce.

(eki-stopping)=
## Stopping rules

A `StoppingRule` is a callable `stop(evaluation) -> bool`, returning a Python
`bool`, pure, consulted after each evaluation and before the increment. A rule
that fires ends the run with `STOPPING_RULE` and costs the evaluation whose
update is then discarded, recorded as the terminal record.

**`DiscrepancyStop(tau=1.0)`** stops when the mean prediction fits to the
noise level (Iglesias, 2016):

$$
2\,\Phi(\bar g) \le \tau^2 N,
$$

with $\bar g$ the mean prediction, so $\Phi(\bar g)$ is the evaluation's
`center_misfit`. At the true parameters $2\Phi$ is a $\chi^2_N$ variate, so
$\tau^2 = 1 + k\sqrt{2/N}$ sets the threshold $k$ standard deviations above
its mean. `tau` is finite and positive (`ValueError`). The residual is that of
the mean *prediction*, not of the prediction at the mean parameters, which
would cost another evaluation.

(eki-inflation)=
## Inflation and relaxation

`enskit.algorithms` defines two protocols that every driver uses, and four
policies that wrap the functions of {ref}`kalman-inflation` in them.

| protocol | called | must return |
| -------- | ------ | ----------- |
| `Inflation` | `inflation(key, *, ensemble, **context)`, on the particles just before they are used (EKI: before each evaluation) | an unweighted `Ensemble` with the same blocks in order, dimensions, count and dtype |
| `Relaxation` | `relaxation(*, prior, posterior, **context)`, just after each update, `prior` its input and `posterior` its result | the same, relative to `posterior` |

A driver passes the context it has as keywords (EKI: `step` and `beta`), and a
policy must accept and ignore what it does not use. A policy consumes its key
whole, is deterministic given its arguments, and holds no state. The driver
checks each result's structure statically and raises (`ValueError`, or
`TypeError` for a type or dtype) naming the policy: an inflation that demoted
the dtype would demote every later step with nothing else raising.

| policy | wraps | effect |
| ------ | ----- | ------ |
| `MultiplicativeInflation(anomaly_scale, *, names=None)` | `inflate_multiplicative` | $x_j \mapsto \bar x + \lambda(x_j - \bar x)$; covariance $\times\lambda^2$ (Anderson & Anderson, 1999) |
| `AdditiveInflation(covs=None, /, **block_covs)` | `inflate_additive` | adds centered draws from $Q_b$ to block $b$ (Hamill & Whitaker, 2005) |
| `RelaxToPriorSpread(alpha, *, names=None)` | `relax_to_prior_spread` | relaxes each coordinate's spread toward the prior's (Whitaker & Hamill, 2012) |
| `RelaxToPriorPerturbations(alpha, *, names=None)` | `relax_to_prior_perturbations` | blends posterior and prior anomalies (Zhang, Snyder & Sun, 2004) |

Each is a pytree: the scalar or the covariances are data, so a traced value
flows through, and `names` is static. Construction checks that a scalar is a
real scalar (`ValueError` for an array, which would scale each coordinate
differently; `TypeError` for a `bool` or a non-number), that names are
distinct strings, and that each covariance is a `PSDLinOp` and not a family;
the values (a positive `anomaly_scale`, an `alpha` in $[0, 1]$, a block's
existence, a covariance's side and `factor`) are the Kalman functions' checks
at the call. A vmapped family of policies refuses to be called. The
numerics are the Kalman layer's and are tested there; the anomaly scale is
named for what it multiplies, so an intended variance inflation of 1.2 is
`anomaly_scale=math.sqrt(1.2)`.

In EKI, inflation runs on the state's particles before the evaluation
(operation 2), so a run with inflation never evaluates the initial particles
as given, and never returns inflated ones. Relaxation runs on the update's
result, with `prior` the evaluated particles (inflated and repaired, with the
prediction block), so its default `names` are the parameter blocks. **Both
change the target on purpose**: a run that uses them is not a tempering ladder
for $\pi_\beta$, and the sampling form should use neither.

The word "inflation" in much of the EKI literature names something else,
multiplying the *noise* covariance by $\alpha$, which here is the increment
$\Delta\beta = 1/\alpha$ and the schedule's business.

(eki-failures)=
## Forward models and failed particles

**The forward model is a simulator**, specified by the simulator contract of
{doc}`maps-contract` ({ref}`maps-simulators`), which governs what it
receives, what it may return and what it must be; this section states only
what the driver adds.

- **It is called once per step, through `pushforward`**, with one positional
  `(J, d_b)` array per input block: the parameter blocks named by `inputs`, in
  that order, or every block in block order. Each array is a concrete
  `jax.Array` of the particles' dtype, never a tracer, and the inflated
  particles when an inflation is given. Its output becomes the block
  `PREDICTION` and must have $N$ columns. A `StructuredMap` such as
  {class}`~enskit.maps.Linear` is accepted too.
- **A simulator declaring `needs_key`** is called as `f(key, *inputs)` with
  the step's `key_evaluate` ({ref}`eki-prng`). A stochastic simulator makes
  the run EKI for $\tilde G(u) = G(u) + \eta$: the gain is damped by the
  evaluation noise, the adaptive schedules read a noisier $\Phi$ and shorten
  their steps, and telescoping is a statement about $\tilde G$.
- **The dtype rule is the simulator contract's**: a narrower floating return
  is promoted with a `UserWarning`, a wider one, or an integer, boolean or
  complex one, raises `ValueError` naming the simulator. **`run` and
  `iterate` issue the promotion warning once per run**, at the caller's line;
  every other warning is passed through, shown once per location per run.
  `evaluate` warns at each call, having no run to be once per. Promotion does
  not recover digits the simulator never computed, which is why the caller is
  told.
- **A float32 run is not yet supported.** Scaling an operator by a scalar
  promotes it to float64 (#68), so the tempered noise $R/\Delta\beta$ of a
  float32 run is float64, and the first update raises `TypeError` from the
  Kalman layer's dtype check, with either rule. A strict expected-failure
  test (obligation 29) turns into a failure when #68 is fixed, so this
  sentence is removed with it.
- **Row independence** (row $j$ of the output depends only on row $j$ of the
  inputs) is required and invisible from inside a run; `enskit.testing.
  check_simulator` checks it from outside.

**Failure is signaled by a non-finite prediction.** A wrapper around a model
that can crash, time out or lose a worker must catch that itself and return a
non-finite row; an exception that escapes it propagates out of the driver.
Validity is read from the prediction block only: a non-finite parameter,
which only a defective inflation can produce, reaches the update, which
refuses it in debug mode and otherwise returns `nan`, raised at operation 8. Finite nonsense (a solver returning zeros, a fill value of
$-9999$) is invisible: it is "valid", and a fill value's enormous misfit
reads to an adaptive schedule as disagreement, so the ladder crawls at its
floor.

`on_failure` must be one of two strings; anything else raises `ValueError`
rather than defaulting:

| value | behavior |
| ----- | -------- |
| `"raise"` (default) | raise `EKIError` naming the step, the level and the failed particles |
| `"repair"` | move the failed particles to the valid particles' center, in every block |

Either way $J_v < 2$ raises: one particle has no anomalies. With
`"repair"`, failures are reported three ways, since a repaired run otherwise
returns a normal-looking result: every record's `n_valid`, a `WARNING` log
record per step with a failure, and one `UserWarning` per run from `run`,
together with `EKIResult.min_n_valid`.

### `repair_failed_particles(*, ensemble, valid)`

With $m_j \in \{0, 1\}$ the validity, $J_v = \sum_j m_j$, and $\hat x$ a
block's mean over the valid particles,

$$
x_j \mapsto m_j x_j + (1 - m_j)\,\hat x
$$

in **every block**, the prediction block included: valid particles are left
exactly where they are and failed ones are moved to the center. Repairing one
block and not another would corrupt the cross-covariance silently. Then,
exactly: the mean over all $J$ particles is $\hat x$; the sample covariance
and cross-covariance (divisor $J - 1$) are the valid particles' (divisor $J_v
- 1$) times $c = (J_v - 1)/(J - 1)$; and a repaired particle has zero anomaly
in every block, so it rejoins the ensemble.

**The damping is the intended trade.** With both blocks scaled by $c$ the
gain is the one at increment $c\,\Delta\beta$: a failed particle costs a
slightly shorter step, and narrows the spread by $c$ for later steps.
Rescaling the valid anomalies by $\sqrt{(J-1)/(J_v-1)}$ would make the moments
exact instead, but it moves every valid particle outward by a data-dependent
factor ($\sqrt{99/89} \approx 1.055$ at $J = 100$ with 10% failing), a silent
inflation under the name "repair"; the function does not do that.

`ensemble` is an unweighted `Ensemble` (`TypeError`, `ValueError`; a weighted
ensemble drops a failed particle by giving it weight zero instead), `valid` a
`(J,)` boolean array (`ValueError`) with at least two `True` entries, checked
when it can be read. Both are keyword-only.

**A step with no failure skips the repair**, in Python on the read count, so
that adding failure handling changes nothing about a run in which nothing
fails: the formula is the identity there but not bit-exactly.

**Misfits are computed after the repair.** A repaired particle contributes
$\Phi(\hat g)$, which sits below the valid particles' mean misfit by half
their whitened spread ({ref}`eki-diagnostics`), so the summaries are slightly
optimistic; `n_valid` is on the evaluation for a criterion that wants it.

(eki-diagnostics)=
## Diagnostics

An `Evaluation` exists for one step and carries arrays; a `HistoryRecord` is
kept for the run and carries scalars. The driver builds every record from an
evaluation and the chosen increment, so a record field that disagreed with its
evaluation would be a defect (obligation 21).

### `Evaluation`

`Evaluation(*, step, beta, ensemble, whitened_residuals, n_valid)`, every
argument keyword-only.

| field | kind | meaning |
| ----- | ---- | ------- |
| `step` | static `int` | the step's index |
| `beta` | 0-d | the level entering the step |
| `ensemble` | `Ensemble` | the evaluated particles, after inflation and repair, with `PREDICTION` appended; unweighted, with at least one parameter block |
| `whitened_residuals` | `(J, N)` | row $j$ is $b_j = W(y - g_j)$, after repair |
| `n_valid` | 0-d integer | the number of valid particles, in $[2, J]$ (checked in debug mode) |

`n_valid` is data, not static, so that a `jit`-ed policy does not retrace once
per distinct count. Derived, on access:

| property | value |
| -------- | ----- |
| `misfits` | `(J,)`, $\Phi_j = \tfrac12\lVert b_j\rVert^2$ |
| `center_misfit` | 0-d, $\Phi(\bar g) = \tfrac12\lVert\bar b\rVert^2$ |
| `rms_parameter_spread` | 0-d, $\big(\sum_b\lVert A_b\rVert_F^2/((J-1)P)\big)^{1/2}$ over the parameter blocks |
| `n_particles`, `data_dim` | `int` |

**`center_misfit` is not the mean of `misfits`**:

$$
\overline{\Phi_j} = \Phi(\bar g) + \tfrac{J-1}{2J}\operatorname{tr}\big(W\widehat C_{gg}W^\top\big).
$$

A discrepancy principle asks about the center, a tempering criterion about
the particles. **The whitened residuals determine** the misfits, the center
misfit and the whitened prediction anomalies, $b_j - \bar b = -W(g_j - \bar
g)$, so a criterion that needs the anomaly structure (a Langevin step size,
the singular values of the whitened factor) has it. They are a diagnostic,
never a substitute for the update's own whitening, which centers before it
whitens: centering already-whitened residuals cancels a common $W\bar g$ and
loses accuracy as the ensemble collapses. The driver therefore whitens the
residuals once per step for the diagnostics and the update whitens the factor
again; for a dense whitener that doubles a step's dominant cost, the price of
keeping the conditioning sealed in the layers below.

`rms_parameter_spread` depends on the parameters' units, and is dominated by
the largest-magnitude coordinate when they differ. A third quantity, the
misfit at the mean parameters $\Phi(G(\bar u))$, is absent: it would cost an
evaluation.

### `HistoryRecord`

Eleven fields, **every one a 0-d array**, keyword-only: `step`, `n_valid`,
`beta`, `increment`, `beta_next`, `misfit_mean`, `misfit_min`, `misfit_max`,
`center_misfit`, `spread`, `ess`. No field is static: two records with
different static fields would be different pytree types and the history would
not stack.

`HistoryRecord.from_evaluation(evaluation, increment)` builds the record of a
step: `beta_next = beta + increment`, the three misfit summaries,
`center_misfit`, `spread` the evaluation's `rms_parameter_spread`, and `ess`
the tempering weights' ESS **at the increment taken**, for every schedule.
With `increment=None` it builds a **terminal record**: `increment` exactly 0,
`beta_next == beta`, and `ess` exactly $J$ (written, not computed: `exp(log
J)` is not $J$). A terminal record appears at most once, last, when a stopping
rule fired or a schedule returned `None`; a zero increment in a record means
"evaluated, then stopped". The per-particle misfits and anything else of size
$J$ are absent, so a long history costs nothing; `iterate` yields the
evaluation for a caller who wants them.

(eki-driver)=
## The driver

`run` and `iterate` take the same arguments and wrap one private driver that
knows why it stopped. Neither is implemented over the other: two of the three
ways a run ends emit an identical terminal record, so a consumer of the
yielded stream cannot tell `STOPPING_RULE` from `SCHEDULE_EXHAUSTED`.

```text
run(state, forward, y, noise_cov, *, update_rule, schedule, stop=None,
    inflation=None, relaxation=None, inputs=None, on_failure="raise",
    approximation=None, max_steps=1000)
```

### `iterate(...)`

A generator yielding `(state, record, evaluation)` after every evaluation,
including a terminal one, and returning the status as its `StopIteration`
value. It is the extension point for observing or interrupting a run:
checkpointing, logging, a wall-clock budget, an early `break`. Its validation
runs at the first `next`, since a generator's body does not run before. The
evaluation is yielded because the record holds scalars only. A caller who
ends the loop has what a result needs:

```python
records = []
for state, record, evaluation in eki.iterate(state, forward, y, noise_cov,
                                             update_rule=rule, schedule=sched):
    records.append(record)
    if len(records) >= 2:
        break
result = eki.EKIResult(state=state, history=tuple(records),
                       status=eki.INTERRUPTED, last_evaluation=evaluation)
```

### `run(...)` and `EKIResult`

`run` returns an `EKIResult`, a plain frozen dataclass with four keyword-only
fields: `state`, `history` (a tuple of records, one per evaluation), `status`
and `last_evaluation` (the final `Evaluation`, or `None` when the run made
none). Properties: `ensemble`, `beta`, `n_evaluations`, `n_completed_steps`
(the records less a terminal one), `min_n_valid` (or `None`), `stacked` (the
history as one record of `(T,)` arrays, `(0,)` on an empty history, where the
one-liner `jax.tree.map(lambda *xs: jnp.stack(xs), *history)` raises),
`stop_fired` and `budget_complete`; and the method `mean(name)`.

`status` is one of three strings, exported as constants so a comparison
cannot be misspelled:

| status | meaning |
| ------ | ------- |
| `SCHEDULE_EXHAUSTED` | the ladder finished, by its attributes or by `next_increment` returning `None` |
| `STOPPING_RULE` | the stopping rule fired; the last record is terminal |
| `INTERRUPTED` | the caller ended the run; never produced by `run` |

**Two booleans, and no `converged`.** `stop_fired` answers the optimization
form's *did it fit?*, and `budget_complete` the sampling form's *did the
ladder finish?*; a single `converged` would answer one of them wrongly. There
is no `max_steps` status, because exceeding the bound raises.

**The returned ensemble has never been evaluated** on a schedule-exhausted
run: the last update produced it, so `last_evaluation` holds the particles
before that update. On a stopping-rule run the state is unchanged after the
last evaluation, so the two agree. `last_evaluation` is what makes the final
misfit and the predictive available without another evaluation.

Moments beyond the mean go through the distribution layer:
`result.ensemble.project()` is the moment-matching Gaussian of the final
particles, of rank at most $J - 1$. It is the fit to the terminal ensemble,
not a posterior, and conditioning it would be a further update.

**Progress** is reported on the standard library logger
`enskit.algorithms.eki`: one `INFO` record per step (step, level, increment,
mean misfit) and a `WARNING` on a step with a failure and on a run that made
no evaluation. No handler is installed.

**The arguments.** `y` a `(N,)` finite array, checked once per run (a
non-finite datum otherwise surfaces as unrelated errors); `noise_cov` a
`PSDLinOp` of side $N$, not a family, supporting `whiten` (checked before any
evaluation, `UnsupportedOpError` unmodified); `update_rule` with `build`;
`schedule` with its three members; `stop`, `inflation`, `relaxation` and
`approximation` callable or `None`; `inputs` a `str` or sequence of distinct
parameter block names (`KeyError` for an unknown one); `max_steps` a positive
`int`. All are validated before the first evaluation.

(eki-prng)=
## Randomness

- **The state owns the stream.** `EKIState.key` is the only source of
  randomness, and a run is determined by its initial state and its policies.
- **The split is pinned.** `from_prior` splits once into `(sample, state)`.
  Each step splits into exactly four, `(next, inflate, evaluate, update)`,
  in that order, whether or not the inflation, the simulator and the rule
  consume theirs. `key_inflate` goes whole to the inflation, `key_evaluate` to
  `pushforward` (and so to a simulator that needs a key), and `key_update` to
  `kalman.update`, whose rules split it as {ref}`kalman-prng` pins.
- Identical arrays across EnsKit releases for a fixed JAX version and PRNG
  configuration; the tests snapshot a short `Matheron` run and the state's key
  after three steps.

(eki-validation)=
## Validation and errors

The four tiers of {ref}`contract-validation` apply as in the layers below:
static conditions always, values only in debug mode on concrete arrays.

| tier | checks |
| ---- | ------ |
| construction | `EKIState`: the ensemble's type, weights, family, reserved name; `step`; `beta`'s shape; the key. `Evaluation`: the ensemble's blocks, the residuals' shape, `n_valid`'s type. The schedules' and the stop's fields; the policies' scalars, names and covariances |
| call | the problem's shapes and `y`'s finiteness, `noise_cov`'s type, family and `whiten`; `max_steps`; `on_failure`; `inputs`; the update rule, schedule and policies' types; every policy's and the simulator's output, every step; the increment, every step |
| value (debug) | the state's particles finite and `beta` finite and not negative; `n_valid` in $[2, J]$; the Kalman and distribution layers' own checks |

**`EKIError(RuntimeError)`** is raised when a run cannot continue: `max_steps`
reached, fewer than two valid particles, a failure under `on_failure="raise"`,
or a non-finite updated particle. Each message names the step, the level and
the condition. **It carries the run**: `state`, the last good state, and
`history`, the records before the failure, set on every raise path, so a
caught error is a checkpoint:

```python
try:
    result = eki.run(state, forward, y, noise_cov, update_rule=rule,
                     schedule=sched, max_steps=3)
except eki.EKIError as exc:
    checkpoint(exc.state)        # eki.run(exc.state, ...) continues exactly
    diagnose(exc.history)
```

Everything else follows the layers below: `ValueError`, `TypeError` and
`KeyError` for validation, and `UnsupportedOpError` propagated unmodified. The
layer never falls back to dense algebra.

(eki-jax)=
## JAX integration

- **The forward model is never traced**, and the driver loop is ordinary
  Python. Every array computation is eager or a small jitted helper; the
  update runs eagerly, so the Kalman and distribution layers' debug checks run
  on concrete arrays.
- **Compilations do not grow with the number of steps.** A thirty-step run
  compiles nothing a three-step run of the same problem has not, counted by
  JAX's own compilation events. The increment reaches the update as a 0-d
  array inside `PSDScaled`, and no object with a static field that changes per
  step crosses a `jit` boundary.
- **Families.** The three value classes report a `batch_shape` from their
  leaves; when it is not empty the object is a vmapped family, whose methods
  and array properties refuse with the family message, and whose repr is
  `vmapped(...)`. A run cannot be vmapped: a family of runs is a Python loop.
- **Debugging is `jax.disable_jit()`**; there is no `jit=` argument.

(eki-variants)=
## Expressing variants

Not normative; the evidence that the choices of {ref}`eki-axes` cover the
common variants.

| variant | expressed as |
| ------- | ------------ |
| tempered posterior sampling, ES-MDA (Emerick & Reynolds, 2013) | `FixedSchedule` or `AdaptiveESSSchedule()` with either rule |
| one Kalman update | `FixedSchedule.constant(1.0, 1)` |
| EKI as iterative regularization (Iglesias, 2016) | `FixedSchedule.constant(1.0, n)` with `DiscrepancyStop()` |
| adaptive regularization (Iglesias & Yang, 2021) | `AdaptiveMisfitSchedule` with `DiscrepancyStop()` |
| inflated or relaxed variants | any of the above with `inflation=` or `relaxation=` |
| localized EKI | `update_rule=` a localized rule (PR 9) |
| a hybrid or shrinkage covariance | `approximation=` and `Matheron` |
| Tikhonov-regularized EKI | an augmented problem, below |
| changing the data between steps, backtracking | a loop over `evaluate` and `assimilate` |

*Tikhonov regularization needs no code.* Appending the parameters to the
predictions and the prior mean to the data adds $\tfrac12\lVert
C_0^{-1/2}(u - m_0)\rVert^2$ to the misfit:

```python
def forward_aug(u):
    return jnp.concatenate([forward(u), u], axis=-1)

y_aug = jnp.concatenate([y, prior_mean])
noise_aug = block_diag(noise_cov, prior_cov)
```

:::{warning}
**The augmentation is for the optimization form.** Started from a prior
ensemble and run to $\beta = 1$, the prior enters twice, once through the
particles and once through the appended block, and the result is
over-concentrated by exactly one extra copy of the prior precision, with
nothing raised. For a posterior the initial ensemble must be diffuse.
:::

*A Langevin-type sampler* (the ensemble Kalman sampler of Garbuno-Iñigo et
al., 2020) needs the increment as a step size and the prior's precision, which
an update rule does not see. It is a loop over `evaluate` with its own move,
using the evaluation's whitened residuals; it needs `solve` of the prior
covariance where a run needs only `factor`.

**What does not fit**: mean-field and unscented schemes (their particles do
not move freely between steps), multilevel schemes (several ensembles and
models), and a driver that changes the data (expressible by hand, as above).

(eki-consumers)=
## How the layers around this one connect

Not normative. The driver calls `maps.pushforward` once and `kalman.update`
once per step; the shared policies call the four Kalman functions; nothing
else of the layers below is used. The EnKF driver (PR 8) reuses the policies
and their protocols with `time` as its context, and calls the same `update`
after its forecast. Localization (PR 9) arrives as an update rule, with no
change here. `enskit.testing` holds the policy checks; `enskit.toy` holds
problems for the tests and documentation, and nothing here imports either.

(eki-repr)=
## `repr`

Type name and static sizes, never array contents, and never raising:
`EKIState(n_particles=64, blocks={'u': 2}, step=3)`,
`Evaluation(step=3, n_particles=64)`, `HistoryRecord(step=3)`,
`EKIResult(status='schedule_exhausted', n_evaluations=17, beta=1)`. Policies
print their fields: `AdaptiveESSSchedule(beta_target=1.0, min_increment=0.001,
max_increment=1.0, ess_fraction=0.5, n_bisect=50)`,
`DiscrepancyStop(tau=1.0)`, `MultiplicativeInflation(anomaly_scale=1.05)`,
`RelaxToPriorSpread(alpha=0.5, names=('u',))`,
`AdditiveInflation(u=PSDDiagonal(3, 3))`; `FixedSchedule` summarizes. A family
takes the `vmapped(...)` form.

(eki-surface)=
## Public surface

`enskit.algorithms.eki` exports exactly: `EKIState`, `Evaluation`,
`HistoryRecord`, `EKIResult`; `Schedule`, `StoppingRule`; `FixedSchedule`,
`AdaptiveESSSchedule`, `AdaptiveMisfitSchedule`, `DiscrepancyStop`; `run`,
`iterate`, `evaluate`, `assimilate`, `advance`; `misfits`,
`effective_sample_size`, `repair_failed_particles`; `PREDICTION`,
`SCHEDULE_EXHAUSTED`, `STOPPING_RULE`, `INTERRUPTED`; and `EKIError`.
`enskit.algorithms` exports `eki`, `Inflation`, `Relaxation`,
`MultiplicativeInflation`, `AdditiveInflation`, `RelaxToPriorSpread` and
`RelaxToPriorPerturbations`. The modules are private.

| helper | signature | returns |
| ------ | --------- | ------- |
| `misfits(y, predictions, noise_cov)` | `(N,), (..., N), PSDLinOp -> (...)` | $\tfrac12\lVert W(y - g)\rVert^2$ per row |
| `effective_sample_size(misfits, increment)` | `(J,), scalar -> 0-d` | the ESS of $e^{-\delta\Phi}$, in log space |
| `repair_failed_particles(*, ensemble, valid)` | `Ensemble, (J,) bool -> Ensemble` | the repair of {ref}`eki-failures` |

`misfits` lets the convention be applied outside a run, and agrees with
`Evaluation.misfits`; the normalized log-likelihood is the log density of the
Gaussian with mean $y$ and covariance $R$, and differs from $-\Phi$ by
$\tfrac12(\log\det R + N\log 2\pi)$. `eki.effective_sample_size` is of the
weights an increment would give; `enskit.distribution.effective_sample_size`
is of an ensemble's own weights.

(eki-testing)=
### Checks in `enskit.testing`

| function | checks |
| -------- | ------ |
| `check_schedule(schedule, evaluation=None)` | the two attributes' types, unchanged by a second read; `next_increment` returns `None` or a finite, strictly positive scalar; purity, two calls bit-identical; the budget respected from three quarters of it; a `nan` misfit never turned into a finite step (unless the schedule ignores the misfits) |
| `check_inflation(inflation, key=None, ensemble=None)` | the result's structure and dtype; determinism given the key; an unknown context keyword ignored |
| `check_relaxation(relaxation, prior=None, posterior=None)` | the same, relative to `posterior` |
| `check_stopping_rule(stop, evaluation=None)` | a Python `bool`; purity |

Each builds a small fixture by default and raises `AssertionError` naming the
failed obligation. Purity is why they exist: a policy holding state breaks
resumption silently, and calling it twice on one argument catches that in a
user's code. Update rules are checked by `check_update_rule`
({ref}`kalman-testing`) and simulators by `check_simulator`
({ref}`maps-check-simulator`).

(eki-conformance)=
## Conformance

Obligations on `tests/test_eki.py`, `tests/test_algorithms.py`,
`tests/test_testing.py` and `tests/test_conformance.py`. The dense reference is
hand-written in NumPy, and exactness is checked against closed forms.

1. **Telescoping.** An affine model, a Gaussian prior, exact-moment particles,
   `SymmetricSquareRoot` and ladders summing exactly to 1 reproduce the
   posterior's mean and covariance to a few thousand $\varepsilon$ times their
   scale, for several ladders.
2. **That tolerance catches $R/\beta$**: the mis-scaled ladder, written with
   `kalman.update`, lands on the posterior at level $(T+1)/2$ and misses the
   right one by many orders above test 1's tolerance.
3. **`Matheron` composes**: elementwise against a dense perturbed-observation
   reference with the pinned draws, and in expectation over 400 keys within
   three standard errors of the $KRK^\top/J$ scale.
4. **Schedules**: `FixedSchedule` takes its increments and completes under
   `max_steps == T`; `None` ends with a terminal record; both adaptive
   schedules reach their budget without passing it, keep the clamp
   precedence in all three regimes, take the largest step on a degenerate
   ensemble, run unbounded, take exactly four steps under a 0.3 ceiling and
   ten (not eleven) of 0.1; the bisection returns its safe end against a
   hand-written one; the misfit schedule takes the larger bound in each regime
   and guards both divisions; the entry budget check raises before any
   evaluation, measured on the remaining budget.
5. **The ESS** against its definition, monotone, $J/(1 + \mathrm{cv}^2)$, and
   finite at misfits of $10^4$ where the direct form is `nan`.
6. **The two criteria differ**: the misfit step drives the ESS below 4 and is
   at least five times the ESS step.
7. **Stopping**: `DiscrepancyStop` fires exactly at its threshold, and a fired
   stop ends the run at step 0 with a terminal record and the state unchanged.
8. **`EKIError`** on every raise path carries `state` and `history`, and
   resuming from it reproduces the uninterrupted run.
9. **Failures**: the repair's three identities as equalities, on two blocks;
   a no-failure step returns the particles untouched; `"raise"` is the default
   and names the particles; $J_v < 2$ raises under both; an unknown
   `on_failure` raises; a repaired run's statistics are finite.
10. **Inflation**: each policy equals its Kalman function bit for bit;
    multiplicative inflation stays in the span and additive inflation leaves
    it.
11. **Misfits** against a dense quadratic form at batch ranks 0 to 2,
    whitener-invariant, with the half; against the Gaussian log density for
    four noise structures; the center-misfit gap exact; recovery from the
    whitened residuals.
12. **Reproducibility and resumption**: bit-identical repeats; a run stopped
    after four steps and resumed matches; `iterate` and `run` agree.
13. **The optimization form** converges monotonically to the least-squares
    fit restricted to the initial subspace, distinguished from the
    unrestricted one.
14. **JAX**: compilations constant in the number of steps; the value classes
    round-trip with sentinel leaves; families report and refuse; `n_valid` is
    data.
15. **The history stacks**, `step` and `n_valid` included, and an empty one
    gives `(0,)` fields.
16. **The phases compose**: `advance` equals `assimilate` of `evaluate`; one
    evaluation serves two increments with no further call; the increment is
    checked before any work; `iterate` yields what a hand-written loop with
    `HistoryRecord.from_evaluation` yields; the provenance check catches a
    stale evaluation and, as documented, not a foreign one.
17. **Degeneracy**: $J = 2$, $N = 1$, $P = 1$ with both rules; a collapsed
    ensemble does not move; an all-failing model and a non-finite update
    raise.
18. **Validation, repr and snapshots**: every tier-2 and tier-3 rule; the
    reprs; the pinned prior draw and a short `Matheron` run snapshotted.
19. **The result**: `stop_fired` and `budget_complete` on four fixtures, the
    third stopping on position on a problem it has not fit; `min_n_valid` the
    worst step; `last_evaluation` the final call, off by one exactly on a
    schedule-exhausted run.
20. **Placement**: inflation before the evaluation and relaxation after the
    update see the true `step` and `beta`, the first input is the inflated
    one, a `beta`-dependent relaxation resumes exactly, the relaxations equal
    their Kalman functions of the evaluated particles and the update's
    result, and the reported spread is that of the evaluated particles.
21. **Every record field** against a recomputation from the model's recorded
    inputs.
22. **`rms_parameter_spread`** exact against closed forms, distinguishing a
    divisor of $J$ and a missing $1/\sqrt P$, and the same over two blocks.
23. **A finished ladder is a no-op**, for a fixed and an adaptive schedule,
    with zero calls, a warning in the log, and `restart` giving the full
    ladder.
24. **The update receives both repaired blocks**, the parameters and
    `PREDICTION`, equal to `repair_failed_particles` of the inflated particles
    and raw predictions.
25. **The axes compose**: every schedule with both rules, three inflations,
    two relaxations and with and without a stop terminates, stacks and stays
    finite, with the forward-call count matching.
26. **This page's recipes run**: the two forms, the pinned draw and
    `restart`, backtracking at its stated cost, `stacked` and `project`, the
    `EKIError` and `INTERRUPTED` patterns, the Tikhonov augmentation with its
    double counting asserted exactly; the external-executable wrapper of
    {doc}`user-guide/writing-a-forward-model` against real subprocesses; and
    the blocks of {doc}`user-guide/running-an-inversion` and of the landing
    page.
27. **The forward model receives** concrete, read-only `jax.Array`s of the
    particles' dtype, the inflated ones when inflating.
28. **A jax array, a NumPy array and a nested list** give bit-identical runs.
29. **The promotion warning** is issued once per run at the caller's line,
    other warnings pass through, `evaluate` warns per call, and an integer
    return raises. A float32 run staying float32 is a strict expected failure
    until #68 is fixed.
30. **Counts**: `n_evaluations` and `n_completed_steps` on all four exits,
    exact in both directions, and `max_steps` an exact cap on calls.
31. **Several blocks**: a problem split into two blocks gives the
    single-block run, the order of `inputs` is the order of the arguments, and
    a block the model does not read is still updated.
32. **The `approximation` hook** is called once per step with the tempered
    noise and, by default, changes nothing; a plain `Gaussian` takes
    `Matheron`'s general path and `SymmetricSquareRoot` refuses it.
33. **The four-way split**: a simulator declaring `needs_key` receives the
    third key of each step, and the state carries the first.
34. **The policies and the checks**: the four policies validate, round-trip,
    print and keep float32; the shipped schedules, inflations, relaxations and
    stop pass their checks; each check fails a mutant of each obligation.

Targeted regression tests guard the layer's silent failures, under the
package's do-not-delete rule: the inflation convention; the repair not
rescaling the survivors; misfits after the repair; a schedule counting its
calls; the update's stream unchanged by inflation; a fill value stalling a
ladder; a systematically failing particle visible only in `n_valid`; a failing
step logged; a float32 rule refused and a float32 model promoted; a collapsed
ensemble's spread exactly zero at any magnitude; a vmapped policy refusing; a
`nan` misfit never becoming the floor step.

(eki-ported)=
### Ported regression tests

The old `tests/test_eki.py` became the new one. Every obligation test kept its
number and its claim, ported to the new API, except as listed; its regression
tests are below.

| old test | becomes |
| -------- | ------- |
| `test_3_...` (`PathwiseUpdate` against a dense reference) | the same for `Matheron`, with its pinned draws |
| `test_10_multiplicative_inflation_scales_the_anomalies_not_the_covariance`, `test_10_additive_inflation_matches_its_pinned_elementwise_definition` | `test_10_the_inflation_policies_are_the_kalman_functions`; the formulas are tested in `tests/test_kalman.py` |
| `test_16_the_two_phases_compose...` (records from `assimilate`) | the same, with `assimilate` returning the state; records via `HistoryRecord.from_evaluation` in `test_16_iterate_yields_...` |
| `test_20_inflation_and_the_update_see_the_true_ladder...` | `test_20_inflation_and_relaxation_see_the_true_ladder...`: an update rule no longer sees `step` or `beta`; a relaxation does |
| `test_26_the_additive_inflation_definition_and_the_stacked_one_liner_run` | `test_26_the_stacked_and_moments_blocks_run`; the additive draw is pinned in `tests/test_kalman.py` |
| `test_26_the_external_executable_wrapper_of_the_guide_runs` | `test_26_the_external_executable_wrapper_runs` |
| `test_28_...` | kept at the driver; the container rule itself is `tests/test_maps.py`'s test 28 |
| `test_29_promotion_only_ever_widens` | `tests/test_maps.py`, ported in PR 5 (a wider return now raises) |
| `test_31` (`check_forward_model`) | `tests/test_maps.py`, `check_simulator`, ported in PR 5 |
| `test_regression_inflation_scales_the_anomalies_not_the_covariance` | same name; also `tests/test_kalman.py` |
| `test_regression_the_repair_does_not_rescale_the_surviving_members` | `..._surviving_particles` |
| `test_regression_misfits_are_computed_after_the_repair` | same name |
| `test_regression_a_schedule_that_counts_its_own_calls_is_caught` | same name, with `enskit.testing.check_schedule` |
| `test_regression_the_key_split_is_three_way_in_a_pinned_order` | `test_regression_turning_inflation_on_does_not_shift_the_update_stream` and obligation 33 (now four-way) |
| `test_regression_a_fill_value_model_stalls_an_adaptive_ladder_silently` | same name |
| `test_regression_a_systematically_failing_member_is_visible_only_in_n_valid` | `..._failing_particle_...` |
| `test_regression_a_failing_step_logs_at_warning_level` | same name, logger `enskit.algorithms.eki` |
| `test_regression_a_float32_update_cannot_quietly_demote_a_run` | same name: `kalman.update`'s result check refuses the rule (`TypeError`) |
| `test_regression_the_anomalies_are_formed_stably` | `test_regression_a_collapsed_ensemble_has_exactly_zero_spread_at_any_magnitude`; the stable anomalies are `Ensemble.anomalies`, tested in `tests/test_distribution.py` |
| `test_regression_a_vmapped_inflation_refuses_rather_than_broadcasting` | `test_regression_a_vmapped_policy_refuses_rather_than_broadcasting`; also `tests/test_kalman.py` |
| `test_regression_a_nan_misfit_does_not_become_the_floor_step` | same name |
| `test_20_the_reported_spread_is_of_the_ensemble_that_was_evaluated`, `test_8_every_eki_error_path_...`, `test_4_budget_tol_...`, `test_16_the_provenance_check_...`, `test_14_n_valid_is_data_...`, `test_4_the_entry_budget_check_...` | same names |

The old `tests/test_conformance.py` ran `check_update` on the two old rules;
`check_update_rule` runs on the Kalman rules in `tests/test_testing.py`.

(eki-departures)=
## Departures from the design

The stubs in `docs/redesign/stubs/eki.py` and `algorithms_policies.py` are the
design; where this page differs, it governs.

1. **The key splits four ways**, `(next, inflate, evaluate, update)`, not
   three: a simulator that declares `needs_key` (a declaration PR 5 made
   available to any callable) needs a key of its own, and taking it from
   another stream would consume a key twice.
2. **`iterate` yields `(state, record, evaluation)`**, not `(state, record)`:
   the record holds scalars only, and every recipe that observes a run needs
   the evaluation.
3. **`assimilate` and `advance` take `relaxation=`**, so that `advance` is
   exactly one step of the driver.
4. **`HistoryRecord.from_evaluation` is public**, since `assimilate` returns
   only the state, as the stub has it: a hand-written loop needs the driver's
   records.
5. **`EKIResult` keeps `stop_fired` and `budget_complete`**, and `mean(name)`
   is a method, as on `Ensemble`.
6. **`EKIState`'s key is required**, and `beta` and `step` default to 0.
7. **`rms_parameter_spread` is a property** of the evaluation, over every
   parameter block, and the evaluation has `data_dim`.
8. **`check_relaxation` is separate from `check_inflation`**: the two
   protocols are called differently. `check_schedule` checks the attributes
   too.
9. **`repair_failed_members` is `repair_failed_particles`**, for the
   package's one word per concept.
10. **The promotion warning is deduplicated by the driver**, once per run, as
   the old contract promised, where `pushforward` warns once per call.

(eki-changes)=
## Changes from the previous contract

1. **The layer is built on the new layers**: `enskit.algorithms.eki`,
   evaluating with `maps.pushforward` and updating with `kalman.update`. It
   computes no conditioning of its own.
2. **The state holds an `Ensemble`** of named parameter blocks; the forward
   model receives one array per input block (`inputs=`); its output is the
   block `PREDICTION`; the evaluation carries one ensemble with both, not two
   arrays.
3. **The update rule is a `kalman.UpdateRule` and is required.** The old
   default `TransformUpdate` and `PathwiseUpdate` became
   `kalman.SymmetricSquareRoot` and `kalman.Matheron`; the old update protocol
   (key, ensemble, predictions, increment, step, beta) is gone, so a rule no
   longer sees the run's position. The old `leaves_span` declaration and
   `check_update` went with it.
4. **`approximation=`** carries a modified joint Gaussian into every update.
5. **`on_failure` defaults to `"raise"`**, not `"repair"`: repairing changes
   the estimator, and it is now opted into.
6. **Relaxation joins inflation** (RTPS and RTPP), applied after each update;
   the old contract excluded it. Both kinds of policy live in
   `enskit.algorithms`, take and return `Ensemble`s, and receive the context
   as keywords; `anomaly_factor` is `anomaly_scale`, and `AdditiveInflation`
   takes a covariance per block. The `changes_mean` declaration is gone.
7. **The forward-model contract moved to the maps layer** as the simulator
   contract: a return wider than the particles' dtype now raises (#19) rather
   than being kept; `check_forward_model` became `check_simulator`.
8. **The policy checks moved to `enskit.testing`**, and
   `synthetic_evaluation` was dropped: the checks build their own fixture.
9. **`assimilate` returns the state** and takes the increment positionally;
   records come from `HistoryRecord.from_evaluation`.
10. **`Evaluation`** has no `predictions` or `rms_parameter_spread` field (the
    first is a block, the second a property), and is keyword-only;
    `n_members`, `u_dim` and `v_dim` are `n_particles`, `dims` and `data_dim`.
11. **The driver runs the update eagerly** rather than in its own jitted
    functions, and the compilation bound is checked with JAX's compilation
    events across the whole run.
12. **The logger** is `enskit.algorithms.eki`.

(eki-references)=
## References

- Anderson, J. L. & Anderson, S. L. (1999). A Monte Carlo implementation of
  the nonlinear filtering problem to produce ensemble assimilations and
  forecasts. *Monthly Weather Review*, 127(12), 2741–2758.
- Bishop, C. H., Etherton, B. J. & Majumdar, S. J. (2001). Adaptive sampling
  with the ensemble transform Kalman filter. Part I: Theoretical aspects.
  *Monthly Weather Review*, 129(3), 420–436.
- Burgers, G., van Leeuwen, P. J. & Evensen, G. (1998). Analysis scheme in the
  ensemble Kalman filter. *Monthly Weather Review*, 126(6), 1719–1724.
- Emerick, A. A. & Reynolds, A. C. (2013). Ensemble smoother with multiple
  data assimilation. *Computers & Geosciences*, 55, 3–15.
- Garbuno-Iñigo, A., Hoffmann, F., Li, W. & Stuart, A. M. (2020).
  Interacting Langevin diffusions: gradient structure and ensemble Kalman
  sampler. *SIAM Journal on Applied Dynamical Systems*, 19(1), 412–441.
- Hamill, T. M. & Whitaker, J. S. (2005). Accounting for the error due to
  unresolved scales in ensemble data assimilation: a comparison of different
  approaches. *Monthly Weather Review*, 133(11), 3132–3147.
- Hunt, B. R., Kostelich, E. J. & Szunyogh, I. (2007). Efficient data
  assimilation for spatiotemporal chaos: a local ensemble transform Kalman
  filter. *Physica D*, 230(1–2), 112–126.
- Iglesias, M. A. (2016). A regularizing iterative ensemble Kalman method for
  PDE-constrained inverse problems. *Inverse Problems*, 32(2), 025002.
- Iglesias, M. A., Law, K. J. H. & Stuart, A. M. (2013). Ensemble Kalman
  methods for inverse problems. *Inverse Problems*, 29(4), 045001.
- Iglesias, M. & Yang, Y. (2021). Adaptive regularisation for ensemble
  Kalman inversion. *Inverse Problems*, 37(2), 025008.
- Jasra, A., Stephens, D. A., Doucet, A. & Tsagaris, T. (2011). Inference for
  Lévy-driven stochastic volatility models via adaptive sequential Monte
  Carlo. *Scandinavian Journal of Statistics*, 38(1), 1–22.
- Whitaker, J. S. & Hamill, T. M. (2012). Evaluating methods to account for
  system errors in ensemble data assimilation. *Monthly Weather Review*,
  140(9), 3078–3089.
- Wilson, J. T., Borovitskiy, V., Terenin, A., Mostowsky, P. & Deisenroth,
  M. P. (2021). Pathwise conditioning of Gaussian processes. *Journal of
  Machine Learning Research*, 22(105), 1–47.
- Zhang, F., Snyder, C. & Sun, J. (2004). Impacts of initial estimate and
  observation availability on convective-scale data assimilation with an
  ensemble Kalman filter. *Monthly Weather Review*, 132(5), 1238–1253.

(eki-excluded)=
## Deliberately excluded

**A `lax.scan` driver.** The forward model may not be traceable, and the
per-step decisions (adaptive increments, termination, failures) are Python.

**A default update rule.** Neither shipped rule is right everywhere
({ref}`eki-updates`).

**A `Problem` container** bundling `(forward, y, noise_cov)`: three arguments,
no behavior, and every call site would have to accept both forms.

**A `callback` argument.** `iterate` is the extension point.

**A `jit=` flag.** `jax.disable_jit()` is global, as a debugging switch
should be.

**Trial evaluations as a policy interface.** A schedule that could spend
evaluations would make every schedule's cost unpredictable from its type.
Backtracking and line searches are written against the two phases.

**Changing the data between steps** in the driver, which binds the problem
for a run; a loop over the phases does it.

**Importance weights and resampling.** The ESS is a step-size heuristic and a
diagnostic only.

**Parameter transformations and constraints**, which the caller composes
into the forward model and the prior.

**Mean-field, unscented and multilevel variants**, which do not carry one
freely moving ensemble.

**The moment-exact repair**, a silent inflation ({ref}`eki-failures`).

**Stopping on parameter stagnation**, which needs history and so a stateful
rule; write it as an `iterate` loop.

**Composing inflations, and adaptive inflation.** Two inflations compose in a
three-line callable; an inflation that adapts to the misfits would need the
evaluation.

**The misfit at the mean parameters**, and **the singular values in the
history**: each costs an evaluation or a second decomposition per step, and
each is recoverable in an `iterate` loop.

**Checkpointing to disk.** A state is a pytree; serializing it is the
caller's choice, and resuming from it is exact.

**The validity mask on the evaluation.** Only `n_valid` is carried; a
localized rule that wants the mask can say what shape it needs when it
arrives.

**An adaptive ensemble size.** $J$ is fixed for a run, since every shape
depends on it; failed particles are repaired or raised, not dropped.
