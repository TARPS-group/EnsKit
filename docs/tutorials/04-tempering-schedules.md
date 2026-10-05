# 4. Tempering schedules

A run reaches the posterior through a sequence of tempering levels,

$$0 = \beta_0 < \beta_1 < \cdots < \beta_T = 1 ,$$

and {doc}`01-first-inversion` explained why: each step conditions on the
observations with their noise inflated to $\Sigma / \Delta\beta_t$, so each
step's Gaussian approximation only has to hold over a short distance. That
page let an adaptive schedule choose the levels. This one is about the choice
itself: how gradually to assimilate the observations, and how to tell from a
finished run whether the choice was gradual enough.

The page builds up a sequence of ladders on the same problem, each one
answering a defect in the one before it. The problem, the initial ensemble and the update
rule are the ones {doc}`02-reading-a-run` and {doc}`03-sampling-or-optimizing`
used, `kalman.SymmetricSquareRoot`:

```python
import enskit
import jax
import jax.numpy as jnp
from enskit import kalman, toy
from enskit.algorithms import eki

problem = toy.exponential_decay()
state = eki.EKIState.from_prior(jax.random.key(0), problem.prior,
                                n_particles=64)

def run_ladder(schedule):
    return eki.run(state, problem.forward, problem.y, problem.noise_cov,
                   update_rule=kalman.SymmetricSquareRoot(),
                   schedule=schedule)
```

Every ladder is measured against the target distribution at $\beta = 1$,
whose mean is `[1.9769, 1.4719]` and whose standard deviations are
`[0.0366, 0.0317]`. As in {doc}`03-sampling-or-optimizing`, these come from
quadrature on a grid, which this two-parameter problem is small enough to
allow. The *error in the mean* is the larger of the two parameters' absolute
errors, and the *spread relative to the target* is each parameter's ensemble
standard deviation divided by the target's.

## One step

The coarsest ladder is a single step of $\Delta\beta = 1$: one Kalman update,
the one {doc}`01-first-inversion` illustrated, here under the square-root
rule.

```python
one = run_ladder(eki.FixedSchedule.constant(1.0, n_steps=1))

one.mean("u")                              # [1.9883  1.5709]
one.ensemble["u"].std(axis=0, ddof=1)      # [0.0584  0.6448]
one.stacked.ess                            # [1.]
```

`FixedSchedule` is a ladder given in advance as a tuple of increments, and
`constant(1.0, n_steps=1)` is the tuple `(1.0,)`. Under the square-root rule
the ensemble's mean and spread are exactly those of the conditioned Gaussian
{doc}`01-first-inversion` computed by hand. The ensemble lands near the
right place, 0.099 from the target's mean, and is far too wide: 1.60 times the
target's spread in the amplitude and 20.3 times in the decay rate.

```{figure} ../_generated/figures/04-one-step.png
:alt: Two panels. Left, the prior's contours with 64 particles drawn from it. Right, the particles after one unit step, spread along a vertical line in the decay rate, around a target distribution so small it is barely visible.
:width: 100%

The ensemble at each level of the one-step ladder, over contours of the
exact distribution at that level. Each panel is scaled to hold both, and the
dashed rectangle marks the extent of the next panel.
```

One moment-matched Gaussian has to describe the whole distance from the prior
to the posterior, and on this curved problem it cannot. The run's own record
says so: `ess`, the effective sample size of the tempering weights at the
increment taken ({doc}`02-reading-a-run`), is 1.0, meaning that a single one
of the 64 particles carried essentially all the weight of the step.

## Equal steps

The obvious refinement is to split the distance into equal parts.
`FixedSchedule.uniform(6)` takes six steps of $1/6$, the same number the
adaptive ladder below chooses on its own, so the ladders on this page compare
at equal cost.

```python
uniform = run_ladder(eki.FixedSchedule.uniform(6))

uniform.stacked.beta    # [0.      0.1667  0.3333  0.5     0.6667  0.8333]
uniform.stacked.ess     # [ 1.      4.3995 44.1291 56.8427 59.0309 60.0377]
uniform.ensemble["u"].std(axis=0, ddof=1)   # [0.0403  0.0376]
```

Six steps do much better than one. The error in the mean falls to 0.0034, and
the spread to 1.10 and 1.19 times the target's. But the record shows where
the remaining error comes from: the first step's `ess` is 1.0 again, and the
second's 4.4, while the last three, at 56.8 to 60.0 of 64, reweight hardly at
all.

```{figure} ../_generated/figures/04-uniform.png
:alt: Seven panels, one per level of a uniform six-step ladder, each showing contours of the exact tempered distribution with that level's particles over them. At beta equal to 0.17 the particles are strung along a vertical line far wider than the target; over the later levels they close in on it but stay wider in the decay rate.
:width: 100%

The uniform ladder, at each of its levels. Panels are scaled independently,
and each dashed rectangle marks the extent of the following panel.
```

The figure shows why. The target moves almost all of the way in the first
step. Between $\beta = 0$ and $\beta = 1/6$ its mean goes from `[1, 1]` to
within 0.003 of the posterior's, and its standard deviations fall from 1 to
`[0.089, 0.077]`. The other five steps, five sixths of the ladder, narrow it by
a further factor of 2.4 and barely move it. So the equal steps are not equal
in effect: the first does nearly all of the work, and most of the others are
spent where the target has nearly stopped moving. The
ensemble at $\beta = 1/6$ is 8.7 times too wide in the decay rate, and the
later steps never fully recover from that.

## Steps that start small and grow

If the target moves most at the start, the steps should be smallest there.
`FixedSchedule` takes any tuple of positive increments, so a ladder whose
increments grow geometrically, normalized to sum to one, is a direct
construction:

```python
ratio = 3.0
growth = ratio ** jnp.arange(6)
increments = tuple(float(d) for d in growth / growth.sum())
# (0.0027, 0.0082, 0.0247, 0.0742, 0.2225, 0.6676)

geometric = run_ladder(eki.FixedSchedule(increments))

geometric.stacked.beta  # [0.      0.0027  0.011   0.0357  0.1099  0.3324]
geometric.stacked.ess   # [11.6003 15.2073 33.5608 43.5112 45.3461 46.2407]
geometric.ensemble["u"].std(axis=0, ddof=1)   # [0.0373  0.0333]
```

The same six forward evaluations, spent where the target is moving. The error
in the mean halves, to 0.0017, and the spread comes within 2% and 5% of the
target's. No step has an `ess` below 11.6.

```{figure} ../_generated/figures/04-geometric.png
:alt: Seven panels, one per level of a geometric six-step ladder, each showing contours of the exact tempered distribution with that level's particles over them. The particles follow the contours far more closely than under equal steps, the curved early levels included.
:width: 100%

The geometric ladder, at each of its levels, drawn as before.
```

The ratio of 3 is a choice made in advance, and it matters less than the
shape: ratios of 2, 4 and 5 also beat the uniform ladder on this problem, in
both the mean and the spread. At a ratio of 8 the last increment is 0.875,
and the error in the mean is worse than the uniform ladder's.

## Steps chosen as the run goes

A geometric ladder needs you to know in advance that the target moves most at
the start, and roughly how fast it settles. An adaptive schedule chooses each
increment from the ensemble it actually has instead. `AdaptiveESSSchedule`
takes the largest increment $\delta$ at which the tempering weights
$w_j = e^{-\delta \Phi_j}$, with $\Phi_j$ the misfit of particle $j$, keep an
effective sample size of half the ensemble:

$$\delta^\star = \sup\bigl\{\delta \ge 0 :
  \mathrm{ESS}(\delta) \ge J/2\bigr\}, \qquad
  \mathrm{ESS}(\delta) = \frac{\bigl(\sum_j w_j\bigr)^2}{\sum_j w_j^2} .$$

```python
adaptive = run_ladder(eki.AdaptiveESSSchedule())

adaptive.n_evaluations      # 6
adaptive.stacked.beta       # [0.      0.001   0.0029  0.0112  0.0571  0.3152]
adaptive.stacked.increment  # [0.001   0.0019  0.0083  0.0459  0.2581  0.6848]
adaptive.stacked.ess        # [24.5511 32.     32.     32.     32.     44.567 ]
adaptive.ensemble["u"].std(axis=0, ddof=1)   # [0.0374  0.0338]
```

It chose six steps, and a ladder of nearly the same shape as the geometric
one, without being told either. The error in the mean is 0.0025, and the
spread 2% and 7% wider than the target's. The `ess` holds at 32, half of 64,
except at the first step, where the schedule wanted less than its minimum
increment of 0.001 and had to take 0.001 anyway, and at the last, which was
cut short by the remaining budget.

```{figure} ../_generated/figures/04-adaptive.png
:alt: Seven panels, one per level of the adaptive ladder, each showing contours of the exact tempered distribution with that level's particles over them. The particles follow the contours at every level.
:width: 100%

The adaptive ladder, at each of its levels, drawn as before.
```

On this problem the adaptive ladder is not better than the geometric one: a
fixed ladder of the right shape did as well, and slightly better. What the
adaptive ladder buys is not having to choose. On a problem of
your own you will not know how quickly the target settles, and it adapts to
the problem it is given.

`AdaptiveMisfitSchedule` asks the same question on a different scale. Writing
$\overline\Phi$ and $\sigma^2_\Phi$ for the mean and variance of the
particles' misfits, it takes

$$\delta^\star = \max\Bigl(\frac{N/2}{\overline\Phi},\
  \sqrt{\frac{N/2}{\sigma^2_\Phi}}\Bigr),$$

which measures each step against the number of observations $N$ rather than
the number of particles.

```python
misfit = run_ladder(eki.AdaptiveMisfitSchedule())

misfit.n_evaluations      # 6
misfit.stacked.beta       # [0.      0.001   0.002   0.013   0.0799  0.3914]
misfit.ensemble["u"].std(axis=0, ddof=1)   # [0.0375  0.034 ]
```

With twelve observations it chose nearly the same ladder as the ESS schedule,
and its answer differs from that one only in the fourth decimal. On other
problems the two part ways, and the misfit schedule usually takes the longer
steps: on the same model observed at 1000 times instead of 12, it takes 4
steps where the ESS schedule takes 6. Longer steps suit a run whose purpose
is the fit rather than the ensemble ({doc}`03-sampling-or-optimizing`).

## The ladders side by side

| ladder | forward evaluations | error in the mean | spread, relative to the target | smallest `ess` |
| --- | --- | --- | --- | --- |
| one step | 1 | 0.0989 | `[1.60, 20.31]` | 1.0 |
| six equal steps | 6 | 0.0034 | `[1.10, 1.19]` | 1.0 |
| six steps growing threefold | 6 | 0.0017 | `[1.02, 1.05]` | 11.6 |
| adaptive, ESS | 6 | 0.0025 | `[1.02, 1.07]` | 24.6 |
| adaptive, misfit | 6 | 0.0025 | `[1.03, 1.07]` | 18.8 |

These numbers come from one initial ensemble. On six of them, drawn from keys
0 to 5, the geometric ladder beat the uniform one in both the mean and the
spread every time.
The adaptive ladders choose their own length, so they did not always cost six
evaluations: the misfit schedule beat the uniform ladder every time, once
taking seven steps, and the ESS schedule on five of the six, losing on the one
key where it took five.

This also settles the question {doc}`02-reading-a-run` left open. Its three
equal steps end 0.0078 from the target's mean, with a spread 1.22 and 1.55
times the target's, against 0.0025 and `[1.02, 1.07]` for the adaptive run.
The diagnostics that page read off the history were right.

## Why the early steps are so small

The adaptive ladder's increments grow from 0.001 to 0.68, by a factor of almost
700. That is the shape of the problem, not a defect of the schedule.

A step reweights the target by $e^{-\Delta\beta\, \Phi(u)}$, so what a step
does depends on the increment *times* the misfits it multiplies. At the prior
the misfits are enormous: their median is 2900, because the prior's particles
predict curves far from the observations, and their mean is
$5.9 \times 10^5$, dominated by two particles whose curves are wildly off.
Entering the last step the mean is 8.1. So an increment of 0.001 at the start
moves the target further than an increment of 0.68 at the end, and the
figures show it: between $\beta = 0$ and $\beta = 0.001$ the target's mean
moves by 0.56 in the amplitude, about 0.9 of its new standard deviation,
while across the whole last step, from $\beta = 0.3152$ to 1, it moves by
0.001, a fortieth of one.

The early increments are where the observations are most informative relative
to the prior, and a ladder has to take them slowly there.

## Refining a ladder

A longer ladder of the same shape costs more forward evaluations, and each
step's Gaussian approximation covers a shorter distance. From this initial
ensemble, with this update rule, that bought a better answer at every length
tried:

```{figure} ../_generated/figures/04-refinement.png
:alt: Two panels against the number of forward evaluations, both on log scales. Left, the error in the mean; right, the decay rate's spread in excess of the target's. A line of uniform ladders from 1 to 96 steps falls steadily in both. Points for the geometric and the two adaptive ladders, all at six evaluations, sit below the uniform line.
:width: 100%

Uniform ladders of 1 to 96 steps, and the three six-step ladders above, all
from the same initial ensemble; the two adaptive points, which nearly
coincide, are drawn slightly apart. Right, the decay rate's standard
deviation divided by the target's, minus one; the decay rate is the
worse-matched of the two parameters in every run drawn.
```

The uniform line falls steadily, from an error in the mean of 0.099 at one
step to 0.0001 at 96. A better shape buys what a longer ladder does, for less:
the six-step geometric ladder matches a uniform ladder of twelve steps in the
mean (0.0017 against 0.0018) and of 24 in the spread (both 5% wide in the
decay rate), which would cost two to four times as much.

From other initial ensembles refinement did not always help. On keys 4 and 5,
two equal steps ended further from the target's mean than one step did (0.17
against 0.11 on key 4); from three steps on, a longer uniform ladder was
better on all six keys. That refinement kept helping here is a measurement on
this problem, not a property of the method. For a nonlinear model nothing guarantees that a finer ladder is
closer to the target ({doc}`../eki-contract`), and whether it is depends on the
update rule as well as the ladder; {doc}`05-transform-or-pathwise` compares the
two rules.

## What a ladder costs

Every step is one evaluation of the forward model on the whole ensemble: the
six-step ladders here cost $6 \times 64 = 384$ particle evaluations each, and a
ladder twice as long costs twice that. For a forward model that is expensive
to run, that is nearly the whole cost of a run.

Neither adaptive schedule evaluates the forward model to choose an increment.
Both work from the misfits of the evaluation the step needs anyway, and
`eki.effective_sample_size` ({doc}`02-reading-a-run`) is all the ESS schedule
computes in its search. So adaptivity is free in the resource that matters:
choosing the adaptive ladder's six increments cost no evaluations at all.

## The increment floor

An adaptive schedule never takes less than `min_increment`, 0.001 by default,
except for a last step that finishes the ladder, so a run to $\beta = 1$ takes
at most 1000 steps. That is exactly `run`'s
default `max_steps`. Lowering the floor, or raising the schedule's
`beta_target`, can break that relation, and `run` checks it before the first
evaluation and raises `ValueError` rather than failing a thousand evaluations
later; see the note in {doc}`../user-guide/running-an-inversion`.

## Drawing a ladder yourself

The figures on this page pair each ensemble with its own level, and
`eki.iterate`, the generator form of `run`, gives both:

```python
levels, clouds = [], []
for current, record, evaluation in eki.iterate(
        state, problem.forward, problem.y, problem.noise_cov,
        update_rule=kalman.SymmetricSquareRoot(),
        schedule=eki.FixedSchedule.uniform(6)):
    levels.append(evaluation.beta)             # the level this ensemble is at
    clouds.append(evaluation.ensemble["u"])
levels.append(current.beta)                    # the state after the last step
clouds.append(current.ensemble["u"])

len(levels)    # 7
```

An evaluation carries the level its ensemble is at, so `evaluation.beta` is the
right pairing; `record.beta` is the same number, and `record.beta_next` is the
level the step moved *to*. The ensemble at the final level is not part of any
evaluation, since the run ends without evaluating it, so it comes from the
state the loop yields last.

For a problem with two parameters, the contours come from the unnormalized
tempered density on a grid, which is all a contour plot needs. Here is the
last panel, on the box {doc}`01-first-inversion` used for its reference:

```python
import matplotlib.pyplot as plt

n = 160
amp = jnp.linspace(1.70, 2.25, n)
rate = jnp.linspace(1.25, 1.70, n)
A, R = jnp.meshgrid(amp, rate, indexing="ij")
grid = jnp.stack([A.ravel(), R.ravel()], axis=-1)                       # (n*n, 2)

log_prior = problem.prior.log_density(u=grid)                           # (n*n,)
phi = eki.misfits(problem.y, problem.forward(grid), problem.noise_cov)  # (n*n,)
log_pi = log_prior - levels[-1] * phi   # the tempered density, up to a constant
density = jnp.exp(log_pi - log_pi.max()).reshape(n, n)

plt.contour(amp, rate, density.T)
plt.scatter(clouds[-1][:, 0], clouds[-1][:, 1])
```

Neither `log_prior` nor `phi` depends on the level, so a grid serves every
level whose mass it holds, and only the multiplier on `phi` changes. Few
levels share a grid, though: the target's spread falls by a factor of about
thirty from the prior to the posterior, so limits that suit one level make
another a dot, and each panel above has its own.

## Which ladder to use

Use `AdaptiveESSSchedule()` unless you have a reason not to. It costs no extra
forward evaluations, it needs no knowledge of how quickly your target settles,
and its defaults reach $\beta = 1$ in a bounded number of steps.

Then read `result.stacked.ess`. A step whose `ess` is near 1 conditioned on a
handful of particles, and was too long. A step whose `ess` is near
`n_particles` reweighted the particles only slightly, so a run with many such
steps could have taken fewer, longer ones. Either says the ladder is spending
its steps in the wrong place.

## Next

Every ladder on this page used the same update rule. The other one, and what
the choice between them costs, is {doc}`05-transform-or-pathwise`.
