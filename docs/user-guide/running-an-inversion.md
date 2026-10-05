# Running an inversion

`enskit.algorithms.eki` runs Ensemble Kalman Inversion (EKI; Iglesias, Law &
Stuart, 2013): it starts from particles drawn from a prior, evaluates the
forward model on all of them, moves them with one ensemble Kalman update, and
repeats, keeping a record of each step. Each update is one call of
{doc}`enskit.kalman <updates>`, and each evaluation one pushforward through
the forward model, as {doc}`maps` describes.

This page is about *when and why* to reach for each piece. The
{doc}`../eki-contract` specifies exactly *what* each one does, and is the
place to look for precise shapes, error behavior, and the mathematics of the
two adaptive schedules.

## The shortest complete run

```python
import enskit  # enables float64; import this before creating arrays
import jax
from enskit import kalman, toy
from enskit.algorithms import eki

problem = toy.exponential_decay()            # two parameters, twelve data
forward, y, noise_cov = problem.forward, problem.y, problem.noise_cov

state = eki.EKIState.from_prior(jax.random.key(0), problem.prior, n_particles=64)
result = eki.run(state, forward, y, noise_cov,
                 update_rule=kalman.Matheron(),
                 schedule=eki.AdaptiveESSSchedule())

result.ensemble["u"]   # (64, 2) particles approximating the posterior
result.mean("u")       # (2,) their mean, near problem.u_true = (2.0, 1.5)
result.beta            # 1.0: the ladder finished
```

The parameters are a named block, here `"u"`: `problem.prior` is a
`Gaussian` over it, and the state carries an `Ensemble` of it. `noise_cov` is
any `PSDLinOp`, used only through `whiten`, so a diagonal or block-structured
noise covariance stays cheap.

`forward` is a *simulator*: any callable that takes the `(J, P)` array of all
the particles and returns a `(J, N)` array of predictions. It is called **once
per step with every particle**, never one particle at a time, and it may be
`jit`-ed, fan out over processes, or block on a job scheduler. EnsKit ships no
forward models for real use; {doc}`toy-models` has three small problems for
trying the library. A forward model that can fail also owes the run a failure
signal, described under [Failed particles](#failed-particles) below, and
{doc}`writing-a-forward-model` states the whole obligation and works through
wrapping an external executable.

## Choices, and the same driver

A variant of EKI is a choice on a few independent axes, and `run` is the same
function in every case:

| axis | argument | question it answers |
| --- | --- | --- |
| how the particles move | `update_rule=` (required) | given the increment, what are the new particles? |
| how far, and when to stop | `schedule=`, `stop=` | what is the next increment, and when does the run end? |
| how spread is maintained | `inflation=`, `relaxation=` | what happens to the particles before each evaluation, and after each update? |
| what a failure does | `on_failure=` | raise, or repair the failed particles? |

The axes compose without restriction, and the driver never inspects one to
decide another. That does not make every combination *meaningful*; see
[Two traps](#two-traps) below.

## The tempering ladder, in one identity

With $\pi_0$ the prior, $G$ the forward model and $W$ a whitener of the noise
covariance $R$ ($W^\top W = R^{-1}$), the run's targets are

$$
\pi_\beta(u) \;\propto\; \pi_0(u)\, e^{-\beta \Phi(G(u))},
\qquad \Phi(g) = \tfrac12 \lVert W(y - g)\rVert^2 ,
$$

the prior at $\beta = 0$ and the posterior at $\beta = 1$. Moving one
increment $\delta$ up that ladder is an identity, not an approximation:

$$
\frac{\pi_{\beta+\delta}(u)}{\pi_\beta(u)} \;\propto\; e^{-\delta\Phi(G(u))}
\;\propto\; \mathcal N\bigl(y;\, G(u),\, R/\delta\bigr),
$$

so a step conditions on the same data $y$ with the noise covariance divided
by the *increment*. The state carries the level $\beta$; each step takes
$\delta$.

Two consequences are worth carrying around. Per-step precisions **add**, so a
ladder whose increments sum to 1 composes to one-shot conditioning at
$\beta = 1$; for a linear forward model and a Gaussian prior that is exact,
which is the one property the driver proves about itself. And the divisor is
the increment, never the accumulated level: dividing by $\beta$ produces a
plausible-looking posterior that is wrong by a factor growing with the
ladder's length.

## The two forms are two schedules

The sampling form and the optimization form of EKI are not two drivers and
not a flag. They differ in whether the ladder has a budget.

```python
# Sampling: a ladder to beta = 1, with adaptive increments.
sampled = eki.run(state, forward, y, noise_cov,
                  update_rule=kalman.Matheron(),
                  schedule=eki.AdaptiveESSSchedule())

# Optimization: unit steps, no budget, stop once the data are fit.
fit = eki.run(state, forward, y, noise_cov,
              update_rule=kalman.SymmetricSquareRoot(),
              schedule=eki.FixedSchedule.constant(1.0, n_steps=200),
              stop=eki.DiscrepancyStop(tau=1.0))
fit.stop_fired         # True: the data were fit before the 200 steps ran out
```

`sampled.ensemble` approximates the posterior, under the caveats in
[What a run does not promise](#what-a-run-does-not-promise); with a fixed
ladder it is the ensemble smoother with multiple data assimilation (ES-MDA;
Emerick & Reynolds, 2013). `fit.ensemble` is a *collapsing* ensemble around a
regularized fit, which is not a posterior at all. The optimization form is the
iterative ensemble Kalman method of Iglesias, Law & Stuart (2013): each unit
step conditions on the full data again, and stopping by the discrepancy
principle (Iglesias, 2016) is what regularizes the fit. That is why neither
result is named `posterior`.

## The misfit and the likelihood

$\Phi$ is the negative log-likelihood of the Gaussian noise model, less a
constant that does not depend on the prediction:

$$
\log \mathcal{N}(y \mid g, R) \;=\; -\Phi(g)
\;-\; \tfrac12\bigl(\log\det R + N \log 2\pi\bigr).
$$

Most uses never need the constant. A sampler run as a baseline against EKI
targets the same posterior as the run, with the same $R$ and the same factor
of $\tfrac12$, through `eki.misfits`:

```python
import jax.numpy as jnp

def log_target(u):
    g = forward(u[None])[0]                  # one point, as a batch of one
    log_p = problem.prior.log_density(u=u) - eki.misfits(y, g, noise_cov)
    return jnp.where(jnp.all(jnp.isfinite(g)), log_p, -jnp.inf)

log_target(problem.u_true)
```

The last line is the sampler's version of a failed particle. Neither function
turns a non-finite prediction into $-\infty$ on its own; the value is usually
`nan`, which stops most samplers instead of rejecting the proposal.
`eki.misfits` takes predictions with any leading batch axes, so the `(J, N)`
predictions of an ensemble give `(J,)` misfits. Where the normalized value
matters, it is `Gaussian.independent(y=(y, noise_cov)).log_density(y=g)`,
which additionally needs `logdet` of the noise covariance.

## Choosing the update rule

`update_rule` is required: the two shipped rules answer different questions,
and the choice belongs to you. {doc}`updates` describes both in full.

- **`kalman.SymmetricSquareRoot()`** is deterministic. For a linear forward
  model, a Gaussian prior and particles whose sample moments equal the
  prior's, a ladder summing to 1 reproduces the posterior mean and
  covariance *exactly*. It needs no key, so two runs agree, and a difference
  between them is a bug rather than a seed. Use it when moments matter most,
  the problem is close to linear, or you want the optimization form's fit.
- **`kalman.Matheron()`** is the perturbed-observation update most of the
  EKI literature is written in. Its moments are right only in expectation
  over the key, but because it redraws a perturbation for every particle at
  every step, it keeps the shape of a nonlinear posterior better: the
  square-root update can leave a few particles carrying most of the spread,
  and this one does not. Use it when the posterior ensemble is the
  deliverable and the forward model is nonlinear.

Any object with a `build` method is an update rule;
{doc}`updates` shows how to write one, and `run` accepts it as it accepts the
shipped two.

## Choosing a schedule

| schedule | ladder | reach for it when |
| --- | --- | --- |
| `FixedSchedule.uniform(T)` | `T` steps of `1/T`, to $\beta = 1$ | you must know the evaluation budget in advance |
| `FixedSchedule((d1, d2, ...))` | the increments you give | an ES-MDA ladder, or one summing to exactly 1 |
| `FixedSchedule.constant(c, T)` | `T` steps of `c` | the optimization form, or `constant(1.0, 1)` for a single Kalman update |
| `AdaptiveESSSchedule()` | adaptive, to $\beta = 1$ | the posterior ensemble is the deliverable |
| `AdaptiveMisfitSchedule()` | adaptive, to $\beta = 1$ | the *fit* is the deliverable, or evaluations are scarce |

A fixed ladder whose increments are the reciprocals $1/\alpha_t$ of ES-MDA's
inflation factors is ES-MDA (Emerick & Reynolds, 2013). Increments that sum
to exactly 1 in binary floating point, such as powers of two, end exactly at
the posterior; `FixedSchedule.uniform(T)` sums to 1 only to round-off.

```python
halving = eki.FixedSchedule((0.5, 0.25, 0.25))     # sums to exactly 1
by_misfit = eki.run(state, forward, y, noise_cov,
                    update_rule=kalman.Matheron(),
                    schedule=eki.AdaptiveMisfitSchedule())
by_misfit.stacked.increment                          # the ladder it chose
```

The two adaptive schedules answer the same question, how far the target can
move before this ensemble stops describing it, and measure it differently.
`AdaptiveESSSchedule` keeps the effective sample size of the tempering weights
$e^{-\delta \Phi_j}$ above a fraction of $J$, so each intermediate target stays
representable by the particles that must describe it; it is the adaptive
tempering of sequential Monte Carlo (Jasra et al., 2011), used only to choose
the step, with no weights carried and nothing resampled.
`AdaptiveMisfitSchedule` is the data misfit controller of Iglesias & Yang
(2021): it chooses the increment from the mean and spread of the particles'
misfits, calibrated to the data dimension $N$ rather than to the ensemble
size. It tends to take longer steps and lets the effective sample size fall
further, which is its logic, not a defect; its `divergence_budget` defaults
to $N/2$, so at the default it has no tuning parameter at all.

Neither adaptive schedule evaluates the forward model to choose an increment.
Both read only the misfits the driver already computed, so adaptivity costs
nothing in the resource that matters.

:::{note}
A budgeted adaptive schedule cannot overshoot its budget: the increment is
clamped to what remains last, after the floor `min_increment` and the ceiling
`max_increment`. At the defaults, `beta_target=1.0` and `min_increment=1e-3`,
the worst case is 1000 steps, which is exactly `run`'s default `max_steps`.
Lowering the floor or raising the budget breaks that relation, and `run`
checks the arithmetic on entry and raises `ValueError` before the first
evaluation rather than a thousand evaluations later.
:::

## Stopping

A stopping rule is consulted after each evaluation, before the increment is
chosen, and ends the run when it returns `True`. `DiscrepancyStop(tau)` is
the discrepancy principle (Iglesias, 2016): it fires once the mean prediction
$\bar g$ fits the data to the noise level,

$$
2\,\Phi(\bar g) \;\le\; \tau^2 N .
$$

At the true parameters $2\Phi$ is a $\chi^2_N$ variate, of mean $N$ and
standard deviation $\sqrt{2N}$, so `tau=1.0` stops at the noise level, and
$\tau^2 = 1 + k\sqrt{2/N}$ stops $k$ standard deviations above it, leaving a
margin for a noise covariance that is only approximately known. A stopping
rule is the regularization of the optimization form: without one, unit steps
go on fitting the noise.

A stopping rule is any callable taking an `Evaluation` and returning a Python
`bool`; [Writing your own policy](#writing-your-own-policy) shows one.

## Failed particles

A particle has *failed* when its prediction row contains any non-finite
entry. That is the whole failure signal, and it puts one obligation on the
forward model: **a model that may crash, time out or lose a worker must catch
that itself and return a non-finite row** for the particles affected. An
exception that escapes the callable stops the run.

By default a failed particle raises an `EKIError` naming the step and the
particles. A problem whose forward model is undefined on part of the prior
fails at the first step:

```python
fragile = toy.restricted_decay()           # the decay model, undefined at negative rates
start = eki.EKIState.from_prior(jax.random.key(0), fragile.prior, n_particles=64)

try:
    eki.run(start, fragile.forward, fragile.y, fragile.noise_cov,
            update_rule=kalman.Matheron(), schedule=eki.AdaptiveESSSchedule())
except eki.EKIError as exc:
    failure = exc              # "13 of 64 particles' predictions were not finite ..."
```

`on_failure="repair"` is the opt-in alternative: the failed particles are
moved to the center of the valid ones, in every block, and the valid ones are
left exactly where they were. The particle count never changes, so no shape
downstream becomes dynamic.

```python
repaired = eki.run(failure.state, fragile.forward, fragile.y, fragile.noise_cov,
                   update_rule=kalman.Matheron(), schedule=eki.AdaptiveESSSchedule(),
                   on_failure="repair")
repaired.stacked.n_valid       # valid particles at each step; 51 of 64 at the first
```

Repair has a price. With $J_v$ of the $J$ particles valid, the repaired
particles have zero anomaly, so the sample covariances the update uses are
the valid particles' multiplied by $c = (J_v - 1)/(J - 1)$, which gives the
gain of a shorter increment $c\,\delta$: each step with a failure moves the
particles less far than its increment says, while `beta` still advances by
the full increment. The bias is in the safe direction and bounded by the
failure fraction, but a run with many failures has not reached the level it
reports. Put the other way, the failed particles' spread is discarded before
the update rather than carried through it, so the ensemble is a little
narrower after every step that repairs. Raise rather than repair when you
would rather learn that a model fails than absorb it.

Repairs are reported three ways, because any one is easy to miss: every
record carries `n_valid`, the driver logs a warning at each step that repairs,
and `run` issues one `UserWarning` per run in which any particle was
repaired. Fewer than two valid particles raises under either setting, since a
single particle has no anomalies.

What the signal *cannot* see is finite nonsense: a solver returning zeros,
its initial condition, or a sentinel such as `-9999`. A fill value is the
dangerous case. Its enormous misfit reads to an adaptive schedule as genuine
disagreement among the particles, so the schedule shrinks the increment and
the run stalls on a broken particle instead of flagging it. Map such values to
non-finite rows in your own wrapper, where the information exists;
{doc}`writing-a-forward-model` does this for an external executable.

## Inflation and relaxation

A finite ensemble underestimates its own spread, and an iterated update
shrinks it further. Four policies in `enskit.algorithms` counteract that,
two of each kind:

| policy | applied | does |
| --- | --- | --- |
| `MultiplicativeInflation(anomaly_scale)` | before each evaluation | scales the anomalies by $\lambda$, so the covariance by $\lambda^2$ (Anderson & Anderson, 1999) |
| `AdditiveInflation(u=Q)` | before each evaluation | adds centered draws from $\mathcal N(0, Q)$, in directions the particles may not span (Hamill & Whitaker, 2005) |
| `RelaxToPriorSpread(alpha)` | after each update | RTPS: moves each coordinate's spread a fraction $\alpha$ of the way back to its spread before the update (Whitaker & Hamill, 2012) |
| `RelaxToPriorPerturbations(alpha)` | after each update | RTPP: blends each particle's anomaly after the update with its anomaly before, weight $\alpha$ on the latter (Zhang, Snyder & Sun, 2004) |

```python
import math
from enskit.algorithms import (
    AdditiveInflation,
    MultiplicativeInflation,
    RelaxToPriorPerturbations,
    RelaxToPriorSpread,
)
from enskit.linalg import PSDDiagonal

widen = MultiplicativeInflation(anomaly_scale=math.sqrt(1.2))   # variance x 1.2
jitter = AdditiveInflation(u=PSDDiagonal(jnp.full(2, 1e-4)))
rtpp = RelaxToPriorPerturbations(alpha=0.5)

relaxed = eki.run(state, forward, y, noise_cov,
                  update_rule=kalman.SymmetricSquareRoot(),
                  schedule=eki.FixedSchedule.constant(1.0, n_steps=200),
                  stop=eki.DiscrepancyStop(),
                  relaxation=RelaxToPriorSpread(alpha=0.5))
relaxed.stacked.spread         # shrinks more slowly than fit.stacked.spread
```

**Inflation** acts on the particles about to be evaluated, so the predictions
always match the particles they update and the ensemble a run returns is
never an inflated one; the price is that the initial particles are inflated
before they are first evaluated. The argument is `anomaly_scale`, not
`factor`, because it multiplies the *anomalies*: the literature uses both
conventions with the same symbols, and a caller passing an intended variance
inflation of 1.2 would silently get 1.44.

**Relaxation** acts on an update's result, with the particles the update
started from (after inflation and repair) as its reference, particle $j$ of
one corresponding to particle $j$ of the other. Its correction is in
proportion to what the update did: a coordinate the data did not inform keeps
its spread under RTPS untouched, where multiplicative inflation would widen
it every step. That makes relaxation the gentler tool when only some
coordinates collapse, and inflation the one for spread that is too small
from the start.

Each policy takes `names=` (or, for `AdditiveInflation`, one covariance per
block) to act on some parameter blocks only.

:::{warning}
**Inflation and relaxation leave the ladder, by design.** A run that uses
either is no longer a tempering ladder for $\pi_\beta$: it is a deliberately
widened variant, and the exactness property above no longer holds.
Sampling-form runs should leave both off, which is the default.

"Inflation" is also an overloaded word. Here it means *ensemble* inflation.
In much of the ensemble Kalman literature the same word names inflating the
*noise covariance* by a factor $\alpha$, which in EnsKit is the increment
$\delta = 1/\alpha$ and is the schedule's business. Read any external
formula's definition before transcribing it.
:::

## Several parameter blocks

The parameters may be several named blocks. The forward model then receives
one `(J, d_b)` array per block, positionally, in block order, and every block
is updated:

```python
from enskit.distribution import Gaussian

def decay(amplitude, rate):                      # (J, 1), (J, 1) -> (J, 12)
    return amplitude * jnp.exp(-rate * problem.times)

prior2 = Gaussian.independent(amplitude=(jnp.ones(1), PSDDiagonal(jnp.ones(1))),
                              rate=(jnp.ones(1), PSDDiagonal(jnp.ones(1))))
two = eki.EKIState.from_prior(jax.random.key(0), prior2, n_particles=64)
split = eki.run(two, decay, y, noise_cov, update_rule=kalman.Matheron(),
                schedule=eki.FixedSchedule.uniform(4))
split.mean("amplitude"), split.mean("rate")
```

Named blocks pay off when the parameters have different meanings, units or
priors: a policy can act on some of them (`names="rate"`), and results read
by name rather than by column.

`inputs=` names the blocks the forward model receives, in the order it takes
them: `inputs=("rate", "amplitude")` for a model with that signature, or
`inputs="rate"` for one that reads only the rate. A block that is not passed
is still updated, through its sample correlation with the predictions; if it
is independent of the inputs under the prior, that correlation is sampling
noise, and the block is moved by noise alone. The name `eki.PREDICTION`
(`"prediction"`) is reserved for the forward model's output and may not name
a parameter block.

## A different Gaussian approximation

Each update conditions a Gaussian approximation of the joint of parameters
and predictions; by default `kalman.gaussian_approximation`, the particles'
mean and covariance with the tempered noise $R/\delta$ added to the
predictions. `approximation=` replaces it with any function
`(ensemble, noise) -> Gaussian`, called once per step with the evaluated
particles and `noise = {eki.PREDICTION: noise_cov / increment}`. That is the
extension point for a hybrid or shrinkage covariance, or for localization.
{doc}`updates` shows a hybrid approximation and explains which update rules
accept which approximations, and the {doc}`../kalman-contract` states the
hook's obligations.

## Reading the result

```python
result.status              # eki.SCHEDULE_EXHAUSTED or eki.STOPPING_RULE
result.budget_complete     # did the ladder finish?
result.stop_fired          # did the stopping rule fire?
result.n_evaluations       # forward model calls: one per record
result.n_completed_steps   # updates made
result.min_n_valid         # the worst step's count of valid particles
result.stacked.beta        # the history, every field a (T,) array
result.last_evaluation     # the final evaluation of the forward model
```

`n_completed_steps` equals `n_evaluations`, or one less: a run ended by a
stopping rule, or by a schedule returning `None`, needed an evaluation to
reach that decision and then made no update. Your cost in forward model calls
is `n_evaluations`, and in particle evaluations `n_evaluations * J`.

`stacked` is the whole history as one record of `(T,)` arrays, which is what
you plot: `plt.plot(result.stacked.beta, result.stacked.misfit_mean)`. It
works on an empty history too, returning `(0,)` fields rather than raising.

There are deliberately **two** termination booleans and no single
`converged`. A name like that would have to pick one of two questions and
then answer the other wrongly: defined as "a stopping rule fired", it is
`False` on a completed $\beta = 1$ ladder and `True` on an early exit at
$\beta = 0.4$.

**`last_evaluation` is not the returned ensemble.** On a run that ended with
an exhausted schedule, the last update produced the returned particles and
the run then ended, so they have never been evaluated:
`last_evaluation.ensemble` holds the particles *before* that update, with
their predictions as the block `eki.PREDICTION`. It is still the cheapest
answer to the two questions asked first, *what is my final misfit* and *what
do the predictions look like*:

```python
final = result.last_evaluation
final.ensemble[eki.PREDICTION]  # (64, 12) predictions, before the last update
final.center_misfit             # the misfit of the mean prediction
```

Moments beyond the mean are one line through the distribution layer:

```python
moments = result.ensemble.project()            # a Gaussian fitted to the particles
moments.cov("u").diag()                         # (2,) per-coordinate variances
moments.sample(jax.random.key(1), 1000)["u"]    # (1000, 2) draws from the fit
```

That is a *fit to the final particles*, not a further conditioning step and
not a posterior.

(two-traps)=
## Two traps

**A stopping rule on a budgeted ladder.** A stopping rule fires on the
misfits however much budget remains, so pairing `DiscrepancyStop` with
`AdaptiveESSSchedule()` can end a sampling run at $\beta = 0.4$, whose
ensemble is neither a posterior nor a fit. Nothing raises; `stop_fired=True`
with `budget_complete=False` is what says so.

**Chaining a new ladder onto a finished state.** `state.step` and
`state.beta` carry across runs, which is exactly what makes resumption work,
so handing a *finished* state to a fresh schedule finds the ladder already
exhausted and returns at once, with an empty history and the particles
unchanged. Nothing raises, because a finished ladder legitimately returns;
the driver logs a warning. Use `restart()`:

```python
phase2 = result.state.restart()      # step = 0, beta = 0.0, same particles and key
```

## The two phases, and driving the loop yourself

A step is two phases. `eki.evaluate` evaluates the forward model once and
moves nothing; `eki.assimilate` moves the particles by a given increment,
using an evaluation you already have, and calls no forward model. Phase 2 is
exactly one Kalman update with noise $R/\delta$:

```python
rule = kalman.SymmetricSquareRoot()

evaluation = eki.evaluate(state, forward, y, noise_cov)          # phase 1
moved = eki.assimilate(state, evaluation, 0.25, y, noise_cov,
                       update_rule=rule)                         # phase 2
by_hand = kalman.update(evaluation.ensemble, {eki.PREDICTION: y},
                        noise={eki.PREDICTION: noise_cov / 0.25},
                        update_rule=rule)
bool(jnp.all(moved.ensemble["u"] == by_hand["u"]))               # True
```

`eki.advance(state, forward, y, noise_cov, increment, update_rule=...)` is
the two phases in one call.

`eki.iterate` is `run` as a generator, yielding `(state, record, evaluation)`
after every evaluation. It is the extension point for anything that needs to
*observe* or *interrupt* a run: per-step checkpointing, custom logging, a
wall-clock budget, an early `break`. A loop you end yourself has what a
result needs:

```python
records = []
for current, record, evaluation in eki.iterate(state, forward, y, noise_cov,
                                               update_rule=rule,
                                               schedule=eki.FixedSchedule.uniform(8)):
    records.append(record)
    if len(records) == 3:             # or a wall-clock budget, a signal, ...
        break

interrupted = eki.EKIResult(state=current, history=tuple(records),
                            status=eki.INTERRUPTED, last_evaluation=evaluation)
```

Anything that needs to *revisit* a step, such as backtracking, damping or
trial increments, uses the two phases directly, because one evaluation serves
any number of trial increments:

```python
s, delta = state, 1.0
current = eki.evaluate(s, forward, y, noise_cov)
for _ in range(10):                                       # or: until the data are fit
    trial = eki.assimilate(s, current, delta, y, noise_cov, update_rule=rule)
    probe = eki.evaluate(trial, forward, y, noise_cov)
    if probe.center_misfit < current.center_misfit:
        s, current, delta = trial, probe, delta * 1.5     # accept, lengthen
    else:
        delta = delta / 2                                 # reject, reuse current
```

The accepted branch reuses `probe` as the next step's evaluation, so this
costs one evaluation per step plus one per rejection, which is what
backtracking costs in any implementation. A loop written against `advance`
alone would evaluate the current state again on every trial.

The same pair is how you change the *data* between steps, for example a
subsampled or randomized data vector, which `run` and `iterate` fix for a
whole run.

## Checkpointing, resumption, and errors

`run` on a state returned by an earlier run **continues** it, and the rest of
the run is bit-identical to an uninterrupted one. That is the whole
checkpointing mechanism, and it is why a policy may hold no state across
steps: a schedule that counted its own calls could not be resumed.
`EKIState` is a pytree of arrays and one small static, so serializing it is
your choice of format.

Every `EKIError` carries `state`, the last good state, and `history`, the
records before the failure, so a run that raised resumes from where it
stopped. Exceeding `max_steps` raises, too:

```python
schedule = eki.FixedSchedule.uniform(8)
try:
    eki.run(state, forward, y, noise_cov, update_rule=rule, schedule=schedule,
            max_steps=3)
except eki.EKIError as exc:
    checkpoint = exc.state           # after three steps; exc.history has their records

resumed = eki.run(checkpoint, forward, y, noise_cov, update_rule=rule,
                  schedule=schedule)
whole = eki.run(state, forward, y, noise_cov, update_rule=rule, schedule=schedule)
bool(jnp.all(resumed.ensemble["u"] == whole.ensemble["u"]))     # True
```

The failure example above resumed the same way, from `failure.state`, with
`on_failure="repair"`: the state's key is unchanged, so the resumed step
draws exactly what the failed one did.

There is no `"max_steps"` status, because exceeding `max_steps` **raises**: a
sampling run that silently returned particles at $\beta = 0.7$, labeled as a
posterior, is the failure the two termination booleans exist to expose.
`max_steps` bounds the steps of *this call*, not `state.step`, so a resumed
run gets the allowance you asked for.

## Progress reporting

The driver logs one record per step at `INFO` on the logger named
`"enskit.algorithms.eki"`, carrying the step, the level, the increment and
the mean misfit, and one at `WARNING` at each step that repairs a failed
particle. No handler is installed and no configuration is read, so you see
the progress records only once you ask for them, for example with
`logging.basicConfig(level=logging.INFO)`. Timings, profiles and progress
bars are yours to add around an `iterate` loop.

(writing-your-own-policy)=
## Writing your own policy

Each axis is a **protocol**, not a base class: an implementation is anything
with the right methods and attributes.

| policy | is | checked by |
| --- | --- | --- |
| schedule | `n_steps`, `beta_target`, and `next_increment(evaluation)` returning a positive increment or `None` | `enskit.testing.check_schedule` |
| stopping rule | a callable of an `Evaluation` returning a Python `bool` | `enskit.testing.check_stopping_rule` |
| inflation | `inflation(key, *, ensemble, **context) -> Ensemble` | `enskit.testing.check_inflation` |
| relaxation | `relaxation(*, prior, posterior, **context) -> Ensemble` | `enskit.testing.check_relaxation` |
| update rule | an object with `build`; see {doc}`updates` | `enskit.testing.check_update_rule` |

Every policy must be **pure**: no state across steps and no counters, which is
what keeps a run resumable. Context arrives as keywords (`step=` and `beta=`
for EKI), and a policy ignores what it does not use. Here are a geometric
ladder and a stopping rule on the particles' spread:

```python
from dataclasses import dataclass
from enskit.testing import check_schedule, check_stopping_rule

@dataclass(frozen=True)
class GeometricSchedule:
    """Increments first, first * ratio, first * ratio**2, ..., to beta = 1."""

    first: float = 0.01
    ratio: float = 2.0
    n_steps = None                 # not bounded by a count
    beta_target = 1.0              # the level the ladder ends at

    def next_increment(self, evaluation):
        remaining = self.beta_target - float(evaluation.beta)
        return min(self.first * self.ratio**evaluation.step, remaining)

def spread_collapsed(evaluation):
    return bool(evaluation.rms_parameter_spread < 1e-3)

check_schedule(GeometricSchedule())
check_stopping_rule(spread_collapsed)
geometric = eki.run(state, forward, y, noise_cov, update_rule=kalman.Matheron(),
                    schedule=GeometricSchedule())
```

The schedule reads `evaluation.step`, which counts across resumed runs, so
it resumes correctly; and it clamps to the remaining budget itself, since the
driver ends a ladder at `beta_target` but does not shorten the last step. The
checks run a policy against a small synthetic `Evaluation`, so testing one
never means running a forward model.

(what-a-run-does-not-promise)=
## What a run does not promise

- **Exactness is claimed for the linear-Gaussian case only.** A linear (or
  affine) forward model, a Gaussian prior, particles whose sample moments
  equal the prior's, `SymmetricSquareRoot`, no inflation or relaxation, no
  failed particles, and increments summing *exactly* to 1. Every clause is
  needed, and the last is easy to miss: `FixedSchedule.uniform(T)` sums to 1
  only to round-off.
- **For a nonlinear model the output is an approximation with no consistency
  guarantee.** It depends on the schedule, the ensemble size and the update
  rule; two ladders give two answers and neither bounds the other. A finer
  ladder makes each step's Gaussian approximation more accurate but
  accumulates more sampling error and collapses the ensemble further, so
  refinement is not monotone improvement.
- **Particles are not independent posterior draws.** The empirical gain
  couples them, and that coupling is the sampling error $J$ controls.
  Uncertainty lives in the ensemble's spread, not in any one particle.
- **$J$ bounds what a run can represent.** Without inflation or relaxation,
  every update, under either shipped rule, keeps the particles in the affine
  subspace spanned by the initial ones, of dimension at most $J - 1$, however
  many steps you run. Directions absent from the initial particles are
  unreachable. `AdditiveInflation` is the shipped policy that adds
  directions; `RelaxToPriorSpread`, which rescales each coordinate
  separately, moves the subspace but does not enlarge it.
- **The optimization form deliberately collapses the ensemble.** Its final
  spread measures numerical convergence, not posterior uncertainty.

## References

- Anderson, J. L. & Anderson, S. L. (1999). A Monte Carlo implementation of
  the nonlinear filtering problem to produce ensemble assimilations and
  forecasts. *Monthly Weather Review*, 127(12), 2741–2758.
- Emerick, A. A. & Reynolds, A. C. (2013). Ensemble smoother with multiple
  data assimilation. *Computers & Geosciences*, 55, 3–15.
- Hamill, T. M. & Whitaker, J. S. (2005). Accounting for the error due to
  unresolved scales in ensemble data assimilation: a comparison of different
  approaches. *Monthly Weather Review*, 133(11), 3132–3147.
- Iglesias, M. A. (2016). A regularizing iterative ensemble Kalman method for
  PDE-constrained inverse problems. *Inverse Problems*, 32(2), 025002.
- Iglesias, M. A., Law, K. J. H. & Stuart, A. M. (2013). Ensemble Kalman
  methods for inverse problems. *Inverse Problems*, 29(4), 045001.
- Iglesias, M. & Yang, Y. (2021). Adaptive regularisation for ensemble Kalman
  inversion. *Inverse Problems*, 37(2), 025008.
- Jasra, A., Stephens, D. A., Doucet, A. & Tsagaris, T. (2011). Inference for
  Lévy-driven stochastic volatility models via adaptive sequential Monte
  Carlo. *Scandinavian Journal of Statistics*, 38(1), 1–22.
- Whitaker, J. S. & Hamill, T. M. (2012). Evaluating methods to account for
  system errors in ensemble data assimilation. *Monthly Weather Review*,
  140(9), 3078–3089.
- Zhang, F., Snyder, C. & Sun, J. (2004). Impacts of initial estimate and
  observation availability on convective-scale data assimilation with an
  ensemble Kalman filter. *Monthly Weather Review*, 132(5), 1238–1253.
