# Ensemble Kalman filter contract

This page specifies `enskit.algorithms.enkf`, the ensemble Kalman filter: the
forecast, the analysis, the filter that cycles them over a sequence of
observations, and what each promises. It is normative: an implementation that
violates a rule here is defective even if its tests pass. It is written for
contributors implementing or reviewing the module, and for users who want a
more precise account of what a filter computes than {doc}`user-guide/filtering`
gives.

*Must* and *never* state requirements, *should* states a strong default that a
documented reason may override, and *may* states a permission. The module is
built on {doc}`maps-contract` (the transition and the observation model are
evaluated by `pushforward`) and {doc}`kalman-contract` (each analysis is one
`update`), and shares the inflation and relaxation policies of
{doc}`eki-contract`, whose section {ref}`eki-inflation` is their normative
home. It refers to those pages rather than restating them.

:::{admonition} Status: written and implemented in PR 8
:class: note

This page was written for PR 8 of the redesign plan, from the design's stub
(`docs/redesign/stubs/enkf.py`), and implemented with it. Where it departs
from the stub, it says so in {ref}`enkf-departures`, and this page wins.
:::

(enkf-scope)=
## Scope

The module turns one ensemble Kalman update into a *filter*: particles carried
from one observation time to the next, each time moved through a transition
and then conditioned on that time's observation. It provides

- **the forecast**, a pushforward of the state block through the transition,
  with optional transition noise;
- **the analysis**, one Kalman update on an observation, which also reports
  the observation's log evidence under the forecast's Gaussian approximation;
- **the filter**, the cycle of forecast, inflation, analysis and relaxation
  over a `(T, N)` array of observations, owning the loop and the random
  stream;
- **two toy problems** in `enskit.toy`, the Lorenz-96 system and a linear
  state-space model whose exact filter is available, for the tests and the
  documentation ({ref}`enkf-toy`).

**This module conditions nothing itself.** The transition and the
observation model are evaluated with {func}`enskit.maps.pushforward`, and each
analysis is {func}`enskit.kalman.update`; the update rule, the Gaussian
approximation and every whitening, decomposition and draw belong to those
layers. What this module computes is the loop, the key split, the analysis
means and the finiteness checks.

Outside the module: the transition and the observation model (any simulator,
{ref}`maps-simulators`), localization (an update rule of the Kalman layer,
`kalman.LocalizedUpdateRule`), and every smoother, particle filter and
variational method. {ref}`enkf-excluded` lists what else is left out and why.

(enkf-notation)=
## Notation and conventions

| symbol | meaning |
| ------ | ------- |
| $x_t$ | the state at time $t$, the block `state`, of dimension $d$ |
| $M$ | the transition, $x_{t-1} \mapsto x_t$, possibly reading other blocks |
| $Q$ | the transition noise covariance, optional |
| $H$ | the observation model, from the blocks `observe_inputs` to $\mathbb R^N$ |
| $y_t$ | the observation at time $t$, row $t$ of `observations` |
| $R$ | the observation noise covariance, a `PSDLinOp` of side $N$ |
| $J$ | the number of particles |
| $T$ | the number of observation times |

The state-space model is

$$
x_t = M(x_{t-1}) + \eta_t, \quad \eta_t \sim \mathcal N(0, Q), \qquad
y_t = H(x_t) + e_t, \quad e_t \sim \mathcal N(0, R), \qquad t = 1, \dots, T,
$$

and the filter's particles approximate the filtering distribution
$p(x_t \mid y_{1:t})$ after each analysis.

(enkf-terminology)=
### Terminology

| term | means |
| ---- | ----- |
| **time** | one row of `observations`; the filter's loop index $t = 0, \dots, T - 1$, passed to the policies as `time` |
| **forecast** | the particles moved through the transition: an approximation of $p(x_t \mid y_{1:t-1})$ |
| **background** | the forecast after inflation: the particles the analysis starts from |
| **analysis** | the background conditioned on $y_t$: an approximation of $p(x_t \mid y_{1:t})$ |
| **log evidence** | $\log \hat p(y_t \mid y_{1:t-1})$, the density of $y_t$ under the background's Gaussian approximation |

Code and documentation index times from 0, as rows of `observations`; the
mathematics indexes them from 1, as above. Row $t$ of every per-time output
belongs to row $t$ of `observations`.

Conventions, each normative:

- **The ensemble holds the state and any other blocks.** Parameters the
  transition reads, earlier states and anything else are carried as blocks,
  and every block is a target of each analysis
  ({ref}`enkf-blocks`).
- **The observation is predicted into the block `PREDICTION`** (the string
  `"prediction"`), a name the ensemble may not use. Its known noise $R$ is
  added to the approximation as a covariance, never sampled.
- **Keys are typed keys, keyword-only and optional**, needed only when
  something draws, and consumed whole ({ref}`enkf-prng`).
- **The loop is ordinary Python**, never `lax.scan`: the transition may be a
  subprocess or a scheduler submission, and is never traced.

(enkf-algorithm)=
## The algorithm

(enkf-one-time)=
### One time of the filter

At each time $t$, `filter` performs, in order:

1. **split** the key into `(next, forecast, inflate, analysis)`;
2. **forecast**: `forecast(ensemble, transition, state=state,
   inputs=transition_inputs, transition_noise=transition_noise,
   key=key_forecast)`;
3. **check** that every forecast particle is finite;
4. **inflate**: `inflation(key_inflate, ensemble=forecast, time=t)`, checked
   against the forecast's structure and for finiteness; without an inflation
   the background is the forecast;
5. **analyze**: `analysis(background, observations[t], observe=observe,
   noise_cov=noise_cov, update_rule=update_rule, inputs=observe_inputs,
   approximation=approximation, key=key_analysis)`, checking that every
   prediction and every analysis particle is finite;
6. **relax**: `relaxation(prior=background, posterior=analysis, time=t)`,
   checked against the analysis's structure and for finiteness;
7. **record** every block's analysis mean and the log evidence, and the
   analysis ensemble when `keep_ensembles=True`.

The ensemble passed to `filter` holds the particles before the first
forecast, the analysis at the time before the first observation. A caller
who wants the first observation assimilated without a forecast before it
calls `analysis` first.

(enkf-evidence)=
### The log evidence

The analysis's Gaussian approximation $\hat p$ of the background and its
prediction gives the observation a density,

$$
\log \hat p(y_t \mid y_{1:t-1}) =
\log \mathcal N\big(y_t;\ \bar h_t,\ \hat C_{hh,t} + R\big),
$$

with $\bar h_t$ and $\hat C_{hh,t}$ the sample mean and covariance of the
predictions $H(x_j)$ of the background particles under the default
approximation, or the prediction marginal of whatever `approximation=`
returns. It is computed from **the same approximation object** the update
conditions, by `marginal(PREDICTION).log_density`, so it never disagrees with
the update about $\hat p$. It does not depend on the update rule. Its sum over
the times estimates the log evidence of the observations,
$\log p(y_{1:T})$, the objective for comparing models or tuning $R$, $Q$ or an
inflation (Carrassi et al., 2017). It is exact in the case of
{ref}`enkf-exactness` and an approximation otherwise.

(enkf-exactness)=
### Exactness in the linear-Gaussian case

With a linear transition ($M(x) = Ax$, a `maps.Linear`) and a linear
observation model ($H(x) = Hx$), **no transition noise**, an initial ensemble
whose sample mean and covariance (divisor $J - 1$) equal the initial
distribution's, **at least $d + 1$ particles**, `kalman.SymmetricSquareRoot`,
no inflation, no relaxation and the default approximation, the filter's
analysis means, analysis sample covariances and log evidences equal the
Kalman filter's (Kalman, 1960) to round-off, at every time:

$$
\begin{aligned}
m_t^- &= A m_{t-1}, & P_t^- &= A P_{t-1} A^\top, \\
S_t &= H P_t^- H^\top + R, & K_t &= P_t^- H^\top S_t^{-1}, \\
m_t &= m_t^- + K_t (y_t - H m_t^-), & P_t &= P_t^- - K_t S_t K_t^\top, \\
&& \log \hat p(y_t \mid y_{1:t-1}) &= \log \mathcal N(y_t;\ H m_t^-,\ S_t).
\end{aligned}
$$

The reason is that the sample mean and covariance of particles pushed through
a linear map are the pushed mean and covariance exactly, and the symmetric
square-root update replaces the particles by ones whose sample moments are
the conditional's exactly ({ref}`kalman-square-root`). With fewer than $d + 1$
particles the sample covariance has rank at most $J - 1 < d$, and the
filter is exact only for an initial covariance of that rank.
`toy.linear_state_space(transition_noise_std=0.0)` and its `exact_filter()`
are the reference.

(enkf-not-promised)=
### What the module does not promise

- **Nothing beyond the linear-Gaussian case** above. Transition noise is
  drawn per particle, so even a linear model with $Q \ne 0$ is exact only in
  expectation over the draws; the stochastic update is exact in expectation
  only; for a nonlinear $M$ or $H$ the filter is an approximation whose error
  no output reports.
- **No guard against filter divergence.** A filter whose spread collapses
  tracks its own mean rather than the truth, with nothing raised; inflation,
  relaxation and localization are the remedies, and choosing them is the
  caller's.
- **No missing values**: every row of `observations` must be finite.

(enkf-forecast)=
## `forecast(ensemble, transition, *, state, inputs=None, transition_noise=None, key=None)`

$$
x_j \mapsto M(z_j) + \eta_j, \qquad \eta_j \sim \mathcal N(0, Q)
\ \text{independently},
$$

with $z_j$ particle $j$'s input blocks. It is exactly, and bit for bit,

```python
key_transition, key_noise = jax.random.split(key)        # when key is given
ens = maps.pushforward(ensemble, transition, inputs=inputs, output=state,
                       key=key_transition)
ens = maps.pushforward(ens, maps.AdditiveNoise(transition_noise),
                       inputs=state, output=state, key=key_noise)  # when Q is given
```

- `inputs` defaults to `(state,)`. The transition must return the state
  block's dimension; another dimension raises `ValueError`.
- Every other block is carried through unchanged, in its position, and the
  ensemble keeps its weights: a forecast is not an analysis, and a weighted
  ensemble may be forecast.
- `transition_noise` is a `PSDLinOp` of the state block's side, supporting
  `factor`, and needs a key (`ValueError` without one). A transition that
  declares `needs_key` receives `key_transition`, and `pushforward` raises
  if it is missing.
- **No value is checked**, so `forecast` runs under `jax.jit` when the
  transition is traceable. A non-finite row is the filter's to catch.

(enkf-analysis)=
## `analysis(ensemble, observation, *, observe, noise_cov, update_rule, inputs, approximation=None, key=None)`

Returns `(analysis_ensemble, log_evidence)`. It is

```python
predicted = maps.pushforward(ensemble, observe, inputs=inputs, output=PREDICTION)
analysis_ensemble = kalman.update(predicted, {PREDICTION: observation},
                                  noise={PREDICTION: noise_cov},
                                  update_rule=update_rule,
                                  approximation=approximation, key=key)
```

with the result reordered to the ensemble's block order, and the log evidence
of {ref}`enkf-evidence` computed from the approximation the update built.

- **`inputs` is required.** The ensemble may hold blocks the observation
  model must not read, so there is no default; `filter` defaults it to
  `(state,)`, where the state is known.
- **Every block of the ensemble is a target.** The analysis returns the
  ensemble's blocks in its order, without the prediction.
- `observe` must return $N$ values per particle, $N$ the side of
  `noise_cov`, and `observation` must be `(N,)`; both raise `ValueError`
  otherwise, naming the sizes.
- `approximation` is the hook of {func}`enskit.kalman.update`,
  `(ensemble, noise) -> Gaussian`, called **once** per analysis with the
  ensemble *including* the prediction block. Its result must cover every
  block of the ensemble and the prediction; one that leaves out a block
  raises `ValueError` naming it.
- `noise_cov` is used only through `whiten` and `log_density`; `update`'s
  rules decide what else they need.
- The ensemble must be unweighted and finite, as `kalman.update` requires;
  in debug mode a non-finite observation raises.
- **No value is checked beyond the layers' debug checks**, so `analysis`
  runs under `jax.jit` when `observe` is traceable.

(enkf-filter)=
## `filter(ensemble, observations, *, transition, observe, noise_cov, update_rule, ...)`

The full signature is

```python
enkf.filter(ensemble, observations, *, transition, observe, noise_cov,
            update_rule, state=None, transition_inputs=None, observe_inputs=None,
            transition_noise=None, inflation=None, relaxation=None,
            approximation=None, keep_ensembles=False, start_time=0, key=None)
```

and it performs {ref}`enkf-one-time` at every time.

- `state` defaults to the ensemble's first block; `transition_inputs` and
  `observe_inputs` default to `(state,)`.
- **Times count from `start_time`**, an `int` at least 0, by default 0: row
  $i$ of `observations` is time `start_time + i`, which is what the policies
  receive and what `EnKFError.time` reports. It exists so that a resumed
  filter gives a time-dependent policy the times the uninterrupted one
  would have.
- `observations` is `(T, N)` with $T \ge 1$, and **must be finite**, checked
  eagerly before the first forecast, naming the rows that are not.
- `noise_cov`, `observe` and `transition` are the same at every time. A
  time-varying model, noise or observation set is a loop over `forecast` and
  `analysis` ({ref}`enkf-variants`).
- The initial ensemble must be unweighted and finite, without the block
  `PREDICTION`, checked before the first forecast.
- **Every argument is checked before the transition is first called**: the
  blocks, inputs, the noise covariances' types and sides, transition noise
  without a key, the observations, `transition` and `observe` callable or
  structured maps, the update rule's `build`, the policies and the
  approximation callable, `keep_ensembles`, `start_time` and the key.
- **The filter is a function of its arguments**: the same arguments give
  identical arrays.

### `FilterResult`

A frozen dataclass, not a pytree, with keyword-only fields:

| field | is |
| ----- | -- |
| `ensemble` | the analysis particles at the last time: the ensemble to continue from |
| `means` | a `dict` from every block of `ensemble`, in its order, to a `(T, d_b)` array of analysis means |
| `log_evidence` | `(T,)`, the one-step log evidences |
| `ensembles` | a tuple of the $T$ analysis ensembles with `keep_ensembles=True`, else `None` |

`n_times` is $T$ and `total_log_evidence` the sum of `log_evidence`.
Construction checks the types and that the fields agree on $T$ and on the
blocks. Forecast means are not recorded: a forecast mean is a hand loop's, or
`keep_ensembles` and a transition away.

### `EnKFError`

**`EnKFError(RuntimeError)`** is raised when a particle stops being finite:
after the forecast, the inflation, the observation model (a prediction), the
update or the relaxation. The message names the stage, its callable and the
time. **It carries the filter**:

| attribute | is |
| --------- | -- |
| `time` | the time being assimilated when it failed, counted from `start_time` |
| `key` | the filter's key at the start of that time, before its split |
| `result` | a `FilterResult` of the times before `time`, whose `ensemble` is the last good analysis (or the initial ensemble) |

so a caught error resumes with exactly the draws and the times the
uninterrupted filter would have used:

```python
try:
    result = enkf.filter(ens, ys, key=key, transition=..., ...)
except enkf.EnKFError as exc:
    rest = enkf.filter(exc.result.ensemble, ys[exc.time:], key=exc.key,
                       start_time=exc.time, transition=fixed_transition, ...)
```

Without `start_time`, the draws would agree and the times would restart at
0, so a policy that reads `time` would compute something else.

The finiteness checks run on every time, eagerly, since a single non-finite
particle makes every later analysis `nan`. They cost one reduction per stage.

(enkf-blocks)=
## Blocks beyond the state

Every block of the ensemble is a target of each analysis, so blocks other
than the state are updated by their sample correlation with the prediction:

- **parameters**, a block the transition reads (`transition_inputs=("x",
  "theta")`) and never writes, are estimated jointly with the state: state
  augmentation (Anderson, 2001; Evensen, 2009);
- **an earlier state**, a copy of the state made before the forecast, is
  smoothed by the later observation: the lag-1 ensemble Kalman smoother
  (Evensen & van Leeuwen, 2000);
- **a block nothing reads** is updated the same way, which is what an
  analysis means and usually what is wanted.

A block that must not be updated is a hand loop: drop it before `analysis`
and assign it back after.

(enkf-policies)=
## Inflation and relaxation

The protocols and the four policies are those of `enskit.algorithms`,
specified in {ref}`eki-inflation`; this section says only where the filter
calls them.

| policy | called as | at |
| ------ | --------- | -- |
| inflation | `inflation(key_inflate, ensemble=forecast, time=t)` | after each forecast, before the analysis |
| relaxation | `relaxation(prior=background, posterior=analysis, time=t)` | after each analysis |

The context is `time`, the loop index $t$, a Python `int`. Each output is
checked statically against the ensemble it replaces (`ValueError`, or
`TypeError` for a type or dtype, naming the policy) and for finiteness
(`EnKFError`).

**The relaxation's `prior` is the background**, the inflated forecast, which
is the particles the analysis started from and what the cited papers call the
prior or background ensemble (Zhang, Snyder & Sun, 2004; Whitaker & Hamill,
2012). Used together, inflation and relaxation therefore compound: with
`MultiplicativeInflation(lam)` and `RelaxToPriorSpread(1.0)`, each analysis's
spread is restored to $\lambda$ times the forecast's, so the analysis keeps
none of the update's reduction in spread. The two are usually alternatives;
the drivers do not refuse both, since a small inflation with a partial
relaxation is a legitimate tuning. This settles #71 for both drivers, and the
EKI contract says the same.

(enkf-rules)=
## Update rules and approximations

The update rule is any `kalman.UpdateRule` and is required; there is no
default.

- **`SymmetricSquareRoot`** is the deterministic ensemble transform filter
  (Bishop, Etherton & Majumdar, 2001; Hunt, Kostelich & Szunyogh, 2007). It
  needs the default, aligned approximation.
- **`Matheron`** is the perturbed-observation filter (Burgers, van Leeuwen &
  Evensen, 1998; Houtekamer & Mitchell, 1998), and needs a key.
- **`LocalizedUpdateRule`** goes in as `update_rule` with no change here. It
  needs row-local noise, a `PSDDiagonal` or `Identity` `noise_cov`, and
  coordinates for the state's sites and for the predicted values.
- **A hybrid approximation** (Hamill & Snyder, 2000), the sample covariance
  blended with a static one, goes in as `approximation=`. It is a plain
  `Gaussian`, so only `Matheron` accepts it, and its static part must be
  carried into the prediction block: build it on the state block and push it
  through a `maps.Linear` observation model, which absorbs the independent
  term ({ref}`maps-linear`). Left as the state block's independent term
  without that, `Matheron` would add a draw of it to every particle, an
  additive inflation, and the prediction would not see it.

(enkf-prng)=
## Randomness

- **The key is keyword-only and optional.** A filter that draws nothing (no
  transition noise, a deterministic rule and deterministic policies) runs
  without one; one that draws raises from the layer that needed it
  (`kalman.update` for `Matheron`, `kalman.inflate_additive` for additive
  inflation, `forecast` for transition noise).
- **The split is pinned.** At every time the filter splits its key into
  exactly four, `(next, forecast, inflate, analysis)`, in that order, whether
  or not anything consumes them, so turning a policy on or off never shifts
  another's draws. `forecast` splits its key into `(transition, noise)`
  likewise. `key_inflate` goes whole to the inflation and `key_analysis`
  whole to `kalman.update`, whose rules split it as {ref}`kalman-prng` pins.
- Identical arrays across EnsKit releases for a fixed JAX version and PRNG
  configuration; the tests snapshot a short `Matheron` filter.

(enkf-validation)=
## Validation and errors

The four tiers of {ref}`contract-validation` apply.

| tier | checks |
| ---- | ------ |
| call | the ensemble's type and family; the reserved name; `state` and every input a block, none repeated; the noise covariances' types, families and sides; `observation`'s or `observations`' shape; `observations` finite and the initial ensemble unweighted and finite (`filter`); the update rule's `build`; the callables; `keep_ensembles`; the key's type |
| every time (`filter`) | each stage's output finite, the predictions included; each policy's output's structure and dtype; the transition's and the observation model's output dimensions |
| value (debug) | a non-finite `observation` (`analysis`); the Kalman and distribution layers' own checks |

`EnKFError` is raised when a particle stops being finite during `filter`.
Everything else follows the layers below: `ValueError`, `TypeError` and
`KeyError` for validation, and `UnsupportedOpError` propagated unmodified. The
module never falls back to dense algebra.

(enkf-jax)=
## JAX integration

- **The transition and the observation model are never traced by `filter`**,
  whose loop is ordinary Python; every array computation is eager or a small
  jitted helper. `forecast` and `analysis` check no values, so a caller may
  `jax.jit` them, or a loop of them, when the simulators are traceable.
- **Nothing compiles per time.** A few operations depend on the length of
  `observations` (its finiteness check, the indexing of its rows, the stacking
  of the means) and compile once for each new length; nothing else compiles
  after the first time, with inflation and relaxation on. Counted by JAX's own
  compilation events: going from 30 to 60 times costs exactly the
  compilations that going from 3 to 30 did.
- **Float32 runs stay float32**: the filter never scales the noise
  covariance, so #68 does not arise, and every output has the ensemble's
  dtype. The log evidence is cast to it, since float64 observations or a
  float64 noise covariance would otherwise give a float64 density next to
  float32 particles.
- **A filter cannot be vmapped**: a family of filters is a Python loop.
  Vmapped families of ensembles or covariances are refused with the family
  message.

(enkf-variants)=
## Expressing variants

Not normative; the evidence that the three functions cover the common
variants.

| variant | expressed as |
| ------- | ------------ |
| stochastic EnKF (Burgers et al., 1998; Houtekamer & Mitchell, 1998) | `update_rule=kalman.Matheron()` |
| ETKF (Bishop et al., 2001) | `update_rule=kalman.SymmetricSquareRoot()` |
| LETKF, localized stochastic EnKF (Hunt et al., 2007) | `update_rule=kalman.LocalizedUpdateRule(...)` |
| hybrid EnKF (Hamill & Snyder, 2000) | `approximation=` and `Matheron` |
| inflated or relaxed filters | `inflation=` or `relaxation=` |
| state augmentation, lag-1 smoothing | extra blocks ({ref}`enkf-blocks`) |
| time-varying $H$, $R$ or $M$; missing values; editing the ensemble between times | a loop over `forecast` and `analysis` |
| tuning $R$, $Q$ or an inflation | maximize `result.total_log_evidence` over it |

A hand loop, here with a time-varying observation model and noise, has the
filter's shape (with its own key split):

```python
for t, y in enumerate(observations):
    key, k_fc, k_an = jax.random.split(key, 3)
    ens = enkf.forecast(ens, transition, state="x", key=k_fc)
    ens = kalman.inflate_multiplicative(ens, 1.05)
    ens, log_evidence = enkf.analysis(ens, y, observe=observe_at[t],
                                      noise_cov=noise_at[t],
                                      update_rule=kalman.Matheron(),
                                      inputs="x", key=k_an)
```

**What does not fit**: smoothers beyond a fixed lag (the ensemble Kalman
smoother over a window, iterative smoothers), particle filters (weights
across times), and filters whose analysis needs more than one Kalman update
per time.

(enkf-toy)=
## Toy problems

Two state-space problems join `enskit.toy`, for the tests and the
documentation. Their docstrings specify them; this section records what the
tests rely on.

| factory | class | model |
| ------- | ----- | ----- |
| `lorenz96(*, state_dim=40, n_times=300, obs_every=2, noise_std=1.0, initial_std=1.0, forcing=8.0, dt=0.05, seed=0)` | `Lorenz96` | Lorenz-96 on a ring of `state_dim` sites (Lorenz, 1996), one fourth-order Runge–Kutta step per time, no transition noise; every `obs_every`-th site observed |
| `linear_state_space(*, state_dim=3, data_dim=2, n_times=20, transition_noise_std=0.3, noise_std=0.5, initial_std=1.0, seed=0)` | `LinearStateSpace` | $A = 0.95\,U$ for a random orthogonal $U$, Gaussian $H$, $Q$ and $R$ multiples of the identity |

- Both have the fields a filter takes, under the same names: `initial` (a
  `Gaussian` over the block `"x"`), `transition`, `observe`, `noise_cov`,
  `observations` (`(T, N)`), and `truth` (`(T, d)`, row $t$ the state
  observation $t$ was generated from). `LinearStateSpace` adds
  `transition_noise`, `None` when `transition_noise_std=0`.
- `Lorenz96.transition` is a method from `(J, d)` to `(J, d)`, and
  `lorenz96_step(x, forcing, dt)` is the public step, over the trailing axis,
  with `forcing` broadcasting so a `(J, 1)` block gives each particle its
  own. `Lorenz96.coords` (`(d, 1)`) and `obs_coords` (`(N, 1)`) are site
  indices for `DomainLocalization`; the ring's periodic distance is the
  caller's.
- `Lorenz96.initial` is centered on the truth before the first step, with
  standard deviation `initial_std`; the truth was spun up for 1000 steps from
  near $x = F$.
- `LinearStateSpace.exact_filter()` returns the $T$ filtering distributions
  as `Gaussian`s over `"x"` and the `(T,)` log evidences, computed with the
  distribution and maps layers' exact operations; it agrees with the dense
  Kalman recursion to about $10^{-15}$.
- The keys are split from `jax.random.fold_in(jax.random.key(seed),
  0x746F79)`, so no key a caller splits from `jax.random.key(seed)` into
  fewer than `0x746F79` (about 7.6 million) keys reproduces a problem's draws
  (#73's hazard, avoided here from the start).
  Folding in a *small* integer would not do: with JAX's partitionable
  threefry, the default, `fold_in(k, i)` is `split(k, n)[i]` for every
  $n > i$, so a first version keyed the Lorenz-96 observation noise exactly as
  `initial.sample(jax.random.key(0), ...)` keyed its draw.

(enkf-surface)=
## Public surface

`enskit.algorithms.enkf` exports exactly: `forecast`, `analysis`, `filter`;
`FilterResult`; `EnKFError`; and `PREDICTION`. `enskit.algorithms` exports
`enkf` beside `eki` and the shared policies. The modules are private.
`enskit.toy` gains `lorenz96`, `lorenz96_step`, `Lorenz96`,
`linear_state_space` and `LinearStateSpace`.

`repr`s give the type and static sizes, never array contents, and never
raise: `FilterResult(n_times=300, blocks=('x',))`,
`Lorenz96(state_dim=40, data_dim=20, n_times=300)`,
`LinearStateSpace(state_dim=3, data_dim=2, n_times=20)`.

(enkf-conformance)=
## Conformance

`tests/test_enkf.py` checks each obligation below, by number.

1. **`forecast` is its two pushforwards**, bit for bit, with the pinned
   `(transition, noise)` split; other blocks pass through unchanged and
   weights are kept.
2. **`forecast` refuses** transition noise without a key, a noise of the
   wrong side, an output of the wrong dimension, and unknown blocks.
3. **`analysis` is `kalman.update` on the predicted ensemble**, bit for bit,
   for both shipped rules and a localized one, with the ensemble's block
   order restored.
4. **The log evidence** equals the dense $\log\mathcal N(y; \bar h,
   \hat C_{hh} + R)$, is the same for both rules, and comes from one call of
   the approximation, which also builds the update.
5. **`analysis` refuses** a missing `inputs`, the reserved name, a wrong
   observation shape, a wrong prediction dimension and an approximation that
   leaves out a block.
6. **`filter` is the loop of {ref}`enkf-one-time`**: a hand loop of
   `forecast`, the policies and `analysis` with the pinned split gives
   identical arrays.
7. **Exactness**: on `linear_state_space(transition_noise_std=0.0)` with an
   exact-moment initial ensemble, `SymmetricSquareRoot` and $J \in \{d + 1,
   2d + 2\}$, the means, the last covariance and the log evidences equal
   `exact_filter()` to $10^{-12}$.
8. **The split is pinned**: an identity inflation, which consumes no
   randomness, leaves a `Matheron` filter's output bit-identical; a snapshot
   pins a short filter's arrays.
9. **The policies are called as specified**: the inflation with its key and
   `time`, the relaxation with `prior` the background and `time`, both
   counted from `start_time`; an inflation or a relaxation output of another
   dtype raises `TypeError` naming it.
10. **A non-finite particle raises `EnKFError`** at each of the five stages,
    naming it, and a resumed filter from the error's `result`, `time` and
    `key`, with `start_time=exc.time`, reproduces the uninterrupted one bit
    for bit, with transition noise, additive inflation and a policy that
    reads `time`.
11. **`filter` refuses** non-finite observations (naming the rows), a wrong
    observation width, an empty sequence, a weighted or non-finite initial
    ensemble, the reserved name, a non-callable `observe`, transition noise
    without a key and a bad `start_time`, before calling the transition.
12. **Compilations**: a 60-time filter after a 30-time one compiles exactly
    as many operations as the 30-time one after a 3-time one, and at most 8.
13. **Float32** filters stay float32 with both rules, inflation, relaxation
    and transition noise, and the log evidence stays float32 with float64
    observations and noise.
14. **Blocks beyond the state**: a parameter read by the transition is
    estimated (the scale of a linear transition, to within 0.05 of the truth
    from a prior two standard deviations away, with a third of the prior's
    spread), and `means` has every block; the lagged copy is obligation 16's
    Example 13.
15. **`keep_ensembles`** keeps $T$ ensembles, the last the result's.
16. **The design's examples run unchanged** in substance: Example 2
    (Lorenz-96, $J = 40$, `Matheron`) tracks the truth; Example 12's
    localized rules beat the global one at $J = 10$; Example 11's hybrid
    beats the plain filter at $J = 10$, and `SymmetricSquareRoot` refuses it;
    Example 13 recovers the forcing and its smoother beats its filter. The
    examples' settings are the design's, except that Example 13 inflates by
    1.05 rather than 1.02: on the new toy problem 1.02 lost the truth at two
    of the seeds 0 to 3 (errors 2.8 and 2.3), and 1.05 tracked it at all
    four (0.31 to 0.39).
17. **The toy problems**: reproducible from their seed; the truth follows
    `lorenz96_step`; the observations are the observed truth plus noise of
    the stated size; `exact_filter` matches the dense recursion; the classes
    validate their fields.
18. **The user guide's page** runs, and its stated numbers are what its
    blocks compute.

(enkf-departures)=
## Departures from the design

The stub `docs/redesign/stubs/enkf.py` and the toy stubs were refined as
follows.

1. **The key is keyword-only and optional** in all three functions, where the
   stub took it first and positionally. `CLAUDE.md`'s rule is that a
   sometimes-random function takes `key=`, as `kalman.update` and
   `maps.pushforward` do; `forecast` draws only with transition noise and
   `analysis` only with a stochastic rule. Settled with the maintainer.
2. **`forecast` splits its key into `(transition, noise)`**, so a transition
   that declares `needs_key` gets a key of its own; the stub used one key for
   the noise only.
3. **`EnKFError`** is new: the stub said nothing about non-finite particles.
   It carries the time, the key and the result so far, like `EKIError`.
4. **`FilterResult` gained `n_times` and `total_log_evidence`**, and
   `ensembles` defaults to `None`.
5. **`analysis` checks that the approximation covers every block**, and
   calls it once for both the update and the log evidence. The stub's
   docstring named the hook `joint`; it is `approximation`, as in the Kalman
   layer.
6. **The toy problems are two classes**, `Lorenz96` and `LinearStateSpace`,
   one per problem as the module already had, rather than one
   `StateSpaceProblem`; `LinearStateSpace` gained `exact_filter()`, the
   reference for the exactness obligation. Settled with the maintainer.
7. **The toy factories' arguments follow the module's existing names**:
   `state_dim`, `data_dim`, `n_times` and `noise_std`, for the stub's `dim`,
   `obs_dim`, `n_steps` and `noise_sd`; all are keyword-only. The step
   function is `lorenz96_step`, not `l96_step`. Settled with the maintainer.
8. **#71 is settled by keeping the background as the relaxation's prior**
   ({ref}`enkf-policies`). Settled with the maintainer.
9. **`filter` takes `start_time`**, which the stub did not have. Without
   it, a filter resumed from an `EnKFError` restarts `time` at 0, so a
   policy that reads `time` computes something else; the adversarial review
   found this (an inflation of scale $1 + 0.01\,t$, failing at time 5 of 8,
   received times 0 to 2 where the uninterrupted filter gave 5 to 7). EKI's
   `step` likewise counts across resumed runs.
10. **`filter` checks the predictions for finiteness**, so a non-finite
    observation model is named rather than the update rule, and **the log
    evidence is cast to the ensemble's dtype**. Both from the review.

(enkf-references)=
## References

- Anderson, J. L. (2001). An ensemble adjustment Kalman filter for data
  assimilation. *Monthly Weather Review*, 129(12), 2884–2903.
- Bishop, C. H., Etherton, B. J. & Majumdar, S. J. (2001). Adaptive sampling
  with the ensemble transform Kalman filter. Part I: Theoretical aspects.
  *Monthly Weather Review*, 129(3), 420–436.
- Burgers, G., van Leeuwen, P. J. & Evensen, G. (1998). Analysis scheme in the
  ensemble Kalman filter. *Monthly Weather Review*, 126(6), 1719–1724.
- Carrassi, A., Bocquet, M., Hannart, A. & Ghil, M. (2017). Estimating model
  evidence using data assimilation. *Quarterly Journal of the Royal
  Meteorological Society*, 143(703), 866–880.
- Evensen, G. (1994). Sequential data assimilation with a nonlinear
  quasi-geostrophic model using Monte Carlo methods to forecast error
  statistics. *Journal of Geophysical Research: Oceans*, 99(C5), 10143–10162.
- Evensen, G. (2009). The ensemble Kalman filter for combined state and
  parameter estimation. *IEEE Control Systems Magazine*, 29(3), 83–104.
- Evensen, G. & van Leeuwen, P. J. (2000). An ensemble Kalman smoother for
  nonlinear dynamics. *Monthly Weather Review*, 128(6), 1852–1867.
- Hamill, T. M. & Snyder, C. (2000). A hybrid ensemble Kalman filter–3D
  variational analysis scheme. *Monthly Weather Review*, 128(8), 2905–2919.
- Houtekamer, P. L. & Mitchell, H. L. (1998). Data assimilation using an
  ensemble Kalman filter technique. *Monthly Weather Review*, 126(3), 796–811.
- Hunt, B. R., Kostelich, E. J. & Szunyogh, I. (2007). Efficient data
  assimilation for spatiotemporal chaos: a local ensemble transform Kalman
  filter. *Physica D*, 230(1–2), 112–126.
- Kalman, R. E. (1960). A new approach to linear filtering and prediction
  problems. *Journal of Basic Engineering*, 82(1), 35–45.
- Lorenz, E. N. (1996). Predictability: a problem partly solved. In
  *Proceedings of the ECMWF Seminar on Predictability*, vol. 1, 1–18.
  Reading, UK.
- Whitaker, J. S. & Hamill, T. M. (2012). Evaluating methods to account for
  system errors in ensemble data assimilation. *Monthly Weather Review*,
  140(9), 3078–3089.
- Zhang, F., Snyder, C. & Sun, J. (2004). Impacts of initial estimate and
  observation availability on convective-scale data assimilation with an
  ensemble Kalman filter. *Monthly Weather Review*, 132(5), 1238–1253.

(enkf-excluded)=
## Deliberately excluded

| left out | why |
| -------- | --- |
| missing values in `observations` | which values are present changes $H$ and $R$; a hand loop says so explicitly |
| time-varying $H$, $R$, $M$ or $Q$ arguments | a sequence-of-callables API would duplicate the hand loop it would be implemented as |
| forecast means and spreads in the result | `keep_ensembles=True` or a hand loop; recording them always would double the per-time cost of the means |
| an `on_failure="repair"` as in EKI | a failed transition in a filter usually means the state left the model's domain, and moving it to the center hides that; resume from `EnKFError` instead |
| `lax.scan` over times | the transition may be a host-side simulator; a traceable one can be scanned by hand over `forecast` and `analysis` |
| smoothers over a window, iterative smoothers, particle filters | each needs machinery beyond one update per time ({ref}`enkf-variants`) |
| adaptive inflation | an adaptive scheme carries its estimate from one time to the next, which a policy may not do, since policies hold no state; it is a hand loop |
