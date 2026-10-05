# Forecast and update, separately

Every algorithm EnsKit ships alternates two operations. A **pushforward**
({doc}`maps`) runs a simulator on every particle and adds or replaces a block
with its outputs; an **update** ({doc}`updates`) conditions the particles on a
value of one of their blocks. The drivers fix how the two alternate. This page
is about writing the alternation yourself, which is how to run an algorithm
the drivers do not.

## One cycle

A state-space model advances a state $x$ and observes it with noise,

$$
x_t = A x_{t-1} + \eta_t, \quad \eta_t \sim \mathcal N(0, Q), \qquad
y_t = H x_t + e_t, \quad e_t \sim \mathcal N(0, R).
$$

One cycle of the ensemble Kalman filter is a forecast, two pushforwards that
move each particle through the transition and add a draw of its noise, and an
update, a pushforward that predicts each particle's observation followed by an
update that conditions on the observed value:

$$
x_j \mapsto A x_j + \eta_j, \qquad
g_j = H x_j, \qquad
x_j \mapsto x_j + K(y_t - g_j - e_j), \quad K = \hat C_{xg}(\hat C_{gg} + R)^{-1},
$$

with $\hat C$ the sample covariances and $e_j \sim \mathcal N(0, R)$ for
{class}`~enskit.kalman.Matheron`, the rule used here:

```python
import enskit  # enables float64; import this before creating arrays
import jax
import jax.numpy as jnp
from enskit import kalman, maps, toy
from enskit.algorithms import enkf

problem = toy.linear_state_space()       # A, Q, H, R, the initial state, 20 observations
rule = kalman.Matheron()

def forecast(ens, key):
    k_transition, k_noise = jax.random.split(key)
    ens = maps.pushforward(ens, problem.transition, inputs="x", output="x",
                           key=k_transition)
    return maps.pushforward(ens, maps.AdditiveNoise(problem.transition_noise),
                            inputs="x", output="x", key=k_noise)

def update(ens, y, key):
    ens = maps.pushforward(ens, problem.observe, inputs="x", output="g")
    return kalman.update(ens, g=y, noise={"g": problem.noise_cov},
                         update_rule=rule, key=key)     # returns the block x alone

ens = problem.initial.sample(jax.random.key(0), n_particles=1000)
key = jax.random.key(1)
for y in problem.observations:
    key, k_forecast, k_update = jax.random.split(key, 3)
    ens = update(forecast(ens, k_forecast), y, k_update)
```

The update conditions on `g` and returns the other blocks, so the predicted
observation does not accumulate from one cycle to the next. Nothing in the
loop is specific to filtering: neither layer knows that `x` is a state or that
`y` was observed.

## The same cycle on a Gaussian

{func}`~enskit.maps.pushforward` accepts a {class}`~enskit.distribution.Gaussian`
as well as an {class}`~enskit.distribution.Ensemble` when the map is linear
or is additive noise, and a Gaussian's conditioning is exact. The same loop
with the two operations on a Gaussian is the Kalman filter itself:

```python
belief = problem.initial
for y in problem.observations:
    belief = belief.pipe(maps.pushforward, problem.transition, inputs="x", output="x")
    belief = belief.pipe(maps.pushforward, maps.AdditiveNoise(problem.transition_noise),
                         inputs="x", output="x")
    belief = belief.pipe(maps.pushforward, problem.observe, inputs="x", output="g")
    belief = belief.add_noise(g=problem.noise_cov).condition(g=y).compress()

gap = jnp.abs(ens.mean("x") - belief.mean("x")).max()
spread = jnp.sqrt(belief.cov("x").diag()).min()
float(gap / spread) < 0.15                              # True
```

On the Gaussian, the observation's noise is added exactly with `add_noise`
rather than drawn, and `compress` keeps the latent width at the state's
dimension. The ensemble's mean after twenty cycles is within a small fraction
of a posterior standard deviation of the exact one, and
{meth}`toy.LinearStateSpace.exact_filter <enskit.toy.LinearStateSpace.exact_filter>`
is this loop.

## The drivers' halves are these calls

Each driver's halves are public, and each is one of the two operations, or
one of each:

| driver | its pushforward half | its update half |
| --- | --- | --- |
| {mod}`~enskit.algorithms.enkf` | `forecast`: the transition, then its noise | `analysis`: the observation model, then `kalman.update` |
| {mod}`~enskit.algorithms.eki` | `evaluate`: the forward model | `assimilate`: `kalman.update` with noise $R/\delta$ |

They agree with the hand-written cycle bit for bit, given the same keys:

```python
start = problem.initial.sample(jax.random.key(0), n_particles=50)
k_forecast, k_update = jax.random.split(jax.random.key(5))
y = problem.observations[0]

mine = update(forecast(start, k_forecast), y, k_update)
theirs, log_evidence = enkf.analysis(
    enkf.forecast(start, problem.transition, state="x",
                  transition_noise=problem.transition_noise, key=k_forecast),
    y, observe=problem.observe, noise_cov=problem.noise_cov,
    update_rule=rule, inputs="x", key=k_update,
)
bool(jnp.all(mine["x"] == theirs["x"]))                 # True
```

`analysis` also returns the one-step log evidence, which the hand-written
`update` does not compute. {ref}`writing-the-loop` in {doc}`filtering`, and
*The two phases, and driving the loop yourself* in
{doc}`running-an-inversion`, write each driver's loop with its own halves.

## When to write the loop yourself

- **A block the driver does not carry.** Every block of an ensemble is
  updated by each analysis, through its sample correlation with the
  prediction. A parameter the transition reads is estimated along with the
  state, and a copy of the state taken before each forecast is smoothed
  (example 13 in the {doc}`../examples/index`).
- **Data or noise that change between steps**: a subsampled data vector, a
  noise covariance re-estimated from the data, or a step that is retried at a
  shorter increment (example 4).
- **An algorithm that is neither EKI nor a filter**: a sweep that updates
  each block conditionally on the others (example 9), or a Gaussian
  approximation of your own passed to the update (example 11).

## What a hand-written loop takes on

The drivers do four things a loop must then do itself.

- **Keys.** A key is consumed whole. Split it at every cycle, and once more
  for each random call inside it, as above; reusing a key reuses the noise.
- **Failed particles.** A driver checks that every particle is finite after
  each stage and raises or repairs. An update given a non-finite particle
  returns non-finite particles, and raises only inside
  {func}`~enskit.linalg.debug_checks`; {doc}`updates` shows how to drop
  them first.
- **Inflation and relaxation** are the functions of {mod}`enskit.kalman`
  (`inflate_multiplicative`, `relax_to_prior_spread` and the rest), called
  where the algorithm needs them.
- **A record of the run.** Keep whatever you need as the loop runs: the
  drivers' result types are built from their own records, not from yours.
