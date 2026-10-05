# Toy problems

`enskit.toy` ships three small calibration problems, each bundling a forward
model with a prior, an observation error covariance, a synthetic observation
and the parameters that generated it, `u_true`, and two state-space problems
for filtering ([below](#state-space-problems)). They exist so that the package's own
tests, this documentation and the tutorials all work the same problems, and so
that trying EnsKit needs no data and no model of your own.

:::{important}
**These are not production models, and they are not an interface.** EnsKit
ships no forward models for real use and defines no base class, protocol or
registry for one — a forward model is any callable from a `(J, P)` array of
parameters to `(J, N)` predictions, and {doc}`writing-a-forward-model` is the
whole obligation. What these problems exemplify is that callable and the setup
around it. Nothing in the library imports this module.
:::

## The three calibration problems

| factory | model | for |
| --- | --- | --- |
| `linear_gaussian(parameter_dim=…, data_dim=…)` | $v = Gu$, at any pair of dimensions | a problem whose posterior is known exactly, and — at $P \gg J$ — the ensemble-size limit |
| `exponential_decay()` | $v_i = u_0 e^{-u_1 t_i}$, two parameters | a mildly nonlinear problem, where the tempering ladder earns its cost |
| `restricted_decay()` | the same, defined only for rates above a floor | a model that fails, reproducibly |

Each factory takes a `seed`, and every array it builds is a deterministic
function of its arguments — so a run over a toy problem gives the same numbers
in a docs build, in a test, and on your machine.

Every problem's parameters are the one block `"u"`: `problem.prior` is an
`enskit.distribution.Gaussian` over it, and a run's particles carry it.

(state-space-problems)=
## The two state-space problems

| factory | model | for |
| --- | --- | --- |
| `lorenz96(state_dim=40, n_times=300, obs_every=2)` | the Lorenz-96 system on a ring of sites (Lorenz, 1996), one fourth-order Runge–Kutta step per time, every `obs_every`-th site observed | a chaotic system an ensemble filter can track, and that needs localization at small ensembles |
| `linear_state_space(state_dim=3, data_dim=2, n_times=20)` | $x_t = A x_{t-1} + w_t$, $y_t = H x_t + e_t$, all Gaussian | a filter whose answer is known exactly, from `exact_filter()` |

Both carry what `enkf.filter` takes under the same names, `initial`,
`transition`, `observe`, `noise_cov` and `observations`, and the states that
generated the observations as `truth`, row $t$ with observation row $t$. The
state is the block `"x"`. `Lorenz96` adds `coords` and `obs_coords`, the site
of each state coordinate and of each observed value, for
`kalman.DomainLocalization`; `toy.lorenz96_step` is its transition as a
function, with a forcing that may differ per particle.
`linear_state_space(transition_noise_std=0.0)` has no transition noise, and
then an ensemble filter with the square-root update reproduces
`exact_filter()` to round-off. {doc}`filtering` runs both.

## A complete run, against a known answer

The linear problem is the one to reach for first, because its posterior is
available in closed form and the run can be compared against it rather than
against a tolerance:

```python
import jax
from enskit import kalman, toy
from enskit.algorithms import eki

problem = toy.linear_gaussian(parameter_dim=4, data_dim=8)

state = eki.EKIState.from_prior(jax.random.key(0), problem.prior, n_particles=32)
result = eki.run(state, problem.forward, problem.y, problem.noise_cov,
                 update_rule=kalman.SymmetricSquareRoot(),
                 schedule=eki.AdaptiveESSSchedule())

exact = problem.posterior()               # the closed form, as a Gaussian
fitted = result.ensemble.project()        # the run's particles, as a Gaussian

result.mean("u")   # [-1.3258  1.3698  0.6372  0.133 ]
exact.mean("u")    # [-1.3303  1.3749  0.6281  0.1409]
problem.u_true     # [-1.4009  1.4321  0.6248  0.2005]
```

The run's mean is within `0.01` of the exact posterior mean, and the two sit
about the same distance from `u_true` — the remaining gap is what eight noisy
observations can support, not error in the run. The spreads agree to within
`0.0004`:

```python
fitted.cov("u").diag() ** 0.5    # [0.0686  0.1273  0.1168  0.0876]
exact.cov("u").diag() ** 0.5     # [0.0687  0.1276  0.1165  0.0876]
```

The update rule is the deterministic square root, so the particles are read
out of the conditioned Gaussian rather than drawn; {doc}`updates` says when
to choose it and when `kalman.Matheron()` is the better choice.

Three details of that call are worth naming, because they are the ones a
reader will carry to their own problem:

- **`forward`, `y` and `noise_cov` are three arguments**, and a problem is not
  callable. That is the real signature of a run; the container is a
  convenience for setting one up, and the EKI contract deliberately excludes
  one being accepted in their place.
- **`posterior()` is not new mathematics.** It is three operations of the
  layers below — {doc}`maps` pushes the prior through the linear map, and
  {doc}`distributions` adds the noise and conditions:

  ```python
  from enskit import maps

  beta = 1.0
  joint = problem.prior.pipe(maps.pushforward, maps.Linear(problem.G),
                             inputs="u", output="g")
  posterior = joint.add_noise(g=problem.noise_cov / beta).condition(g=problem.y)
  ```

  Copy them to reach the rest of that object: `joint` is the joint Gaussian
  of the parameters and the noise-free predictions, and `joint.marginal("g")`
  is the prior predictive distribution of the predictions.
- **`posterior(beta=…)` is the tempered target** a run passes through on the
  way to `beta=1.0`, which is what makes an intermediate step checkable too —
  and it is named as `EKIState.beta` is, so `problem.posterior(result.beta)`
  reads directly.

## A model that fails

`restricted_decay()` is the decay model restricted to positive rates, so a
particle whose rate is not positive gets a wholly non-finite prediction row —
which is how a forward model signals a failed particle. A run raises on one by
default; `on_failure="repair"` moves each failed particle to the center of
the valid ones for that step and carries on. The prior puts about 16% of its
mass below the floor, so the first step loses thirteen of the sixty-four
particles, the second one more, and no later step loses any:

```python
problem = toy.restricted_decay()
state = eki.EKIState.from_prior(jax.random.key(0), problem.prior, n_particles=64)
result = eki.run(state, problem.forward, problem.y, problem.noise_cov,
                 update_rule=kalman.SymmetricSquareRoot(),
                 schedule=eki.AdaptiveESSSchedule(),
                 on_failure="repair")

result.min_n_valid            # 51
result.stacked.n_valid        # [51 63 64 64 64]
result.mean("u")              # [1.9796  1.4739]  true values [2. 1.5]
```

Which particles fail is a deterministic function of the ensemble, so this is
reproducible; the *fraction* is a property of the problem rather than an
injected rate, and `rate_floor=` moves it. Raise the floor toward the prior
mean to fail more particles and see where repair stops being adequate; it must
stay below the true rate of 1.5, or the observation would have been generated
where the model does not evaluate.

The failure here is signaled with `jnp.where`, which is the cheap version.
The realistic one is a wrapper that catches its own subprocess failures and
returns non-finite rows: {doc}`writing-a-forward-model` works one through.

## When the parameters outnumber the ensemble

`linear_gaussian` takes its dimensions, so the high-dimensional case is the
same problem at a different size — and the closed form is what makes its
lesson visible:

```python
problem = toy.linear_gaussian(parameter_dim=2000, data_dim=40)
state = eki.EKIState.from_prior(jax.random.key(0), problem.prior, n_particles=40)
result = eki.run(state, problem.forward, problem.y, problem.noise_cov,
                 update_rule=kalman.SymmetricSquareRoot(),
                 schedule=eki.AdaptiveESSSchedule())

result.ensemble["u"].std(axis=0, ddof=1).mean()       # 0.3022
(problem.posterior().cov("u").diag() ** 0.5).mean()   # 0.9900
```

The ensemble reports an average posterior standard deviation about a third of
the exact posterior's. Forty observations cannot constrain two thousand
parameters, and the exact posterior says so; the run can move its particles
only within the 39 directions its initial ensemble spans, fits the
observations there, and reports the spread along those directions as if it
were the whole answer. Nothing raises, and no field of `eki.HistoryRecord`
flags it.

`posterior()` builds a `(P, k)` factor for a prior covariance factor of width
`k`, and a `(k, k)` transform on the way, so it raises above a budget of 20
million elements rather than allocating. At a full-rank prior in 2000
dimensions the returned factor is 32 MB and the peak is nearer 180 MB; above
about four thousand dimensions the closed form is simply unavailable, while
the run is not.

## What these models are, and are not, an example of

Every model here is pure JAX, so each is `jit`-able, `vmap`-pable and
differentiable. **That is a convenience of these particular models** — chosen
to keep a test suite that uses them heavily cheap — **and not a requirement on
yours.** A run's driver loop is ordinary Python precisely so that a
subprocess, a job-scheduler submission or a legacy binary is a legal forward
model, and none of those is traceable.

What they *are* an example of is the batched convention and row independence:

- **A forward model is called once per step with the whole ensemble.** The
  linear model applies its operator to the trailing axis; both decay models
  are `jax.vmap` of a function of one particle, which is the wrapper a
  per-particle model needs.
- **Row `j` of the return depends only on row `j` of the argument.** That is
  the one requirement nothing inside a run detects, and both idioms above make
  it structural rather than a claim. From outside a run it *is* detectable:
  `enskit.testing.check_simulator` permutes the particles and re-evaluates a
  subset of them, which between them catch an order-dependent coupling and a
  symmetric one. Neither is sufficient alone.

```python
from enskit.testing import check_simulator

check_simulator(my_forward, 12, 40)   # 12 parameters in, 40 predictions out
```

It calls the model five times, so check a cheap configuration of an expensive
one. Pass `stochastic=True` for a model that is legitimately not
deterministic; the simulator contract of {doc}`maps` permits one.

## Modifying a problem

Every field is a plain public value, so a variant is a direct construction
rather than a new factory. Giving the linear problem correlated observation
error, for instance, changes nothing else about the run:

```python
import dataclasses
import numpy as np
import jax.numpy as jnp
from enskit.linalg import DensePSD

rng = np.random.default_rng(1)
M = rng.normal(size=(8, 8))
R = jnp.asarray(M @ M.T / 8 + 0.01 * np.eye(8))

problem = toy.linear_gaussian(parameter_dim=4, data_dim=8)
correlated = dataclasses.replace(problem, noise_cov=DensePSD(R))

correlated.posterior().mean("u")    # [-1.4093  0.8248  0.4646  0.0692]
```

Replacing `noise_cov` is the safe case. Replacing `G` or `prior` leaves `y`
as the old map generated it, so `u_true` is no longer the parameters behind the
observation and `posterior()` answers a problem nobody posed — build a fresh
one from the factory instead.

The closed form follows the new covariance, since `posterior()` reads the
field.

## References

- Lorenz, E. N. (1996). Predictability: a problem partly solved. In
  *Proceedings of the ECMWF Seminar on Predictability*, vol. 1, 1–18.
  Reading, UK.
