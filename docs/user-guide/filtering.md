# Filtering a dynamical system

`enskit.algorithms.enkf` is the ensemble Kalman filter (EnKF): it estimates
the state of a system observed sequentially in time, from a model it can only
run forward (Evensen, 1994). This page shows when to use the one-call
`enkf.filter`, how to read what it returns, and when to write the loop
yourself with `enkf.forecast` and `enkf.analysis`. {doc}`../enkf-contract`
specifies all three exactly.

## When to use it

The filter fits a state-space model

$$
x_t = M(x_{t-1}) + \eta_t, \quad \eta_t \sim \mathcal N(0, Q), \qquad
y_t = H(x_t) + e_t, \quad e_t \sim \mathcal N(0, R),
$$

where the transition $M$ is a simulator you can run but not differentiate or
linearize, and the state is too large for a particle filter. It carries $J$
particles from one time to the next: a **forecast** moves every particle
through $M$, and an **analysis** conditions the forecast on $y_t$ with one
ensemble Kalman update ({doc}`updates`). With a linear model and Gaussian
noise it is the Kalman filter; otherwise it is an approximation that works
well when the forecast distribution is roughly Gaussian.

Use **`enkf.filter`** when the transition, the observation model and the
noise are the same at every time and the analysis means and the log evidence
are what you need. Write the loop yourself when any of them changes between
times, when values are missing, or when you edit the ensemble between times
(see [Writing the loop](#writing-the-loop)).

## A filter in one call

The toy Lorenz-96 problem is a chaotic system on a ring of 40 sites, with
every other site observed at each of 300 times with noise of standard
deviation 1 (Lorenz, 1996). Its `initial` distribution is centered on the
true starting state.

```python
import jax
import jax.numpy as jnp
from enskit import kalman, toy
from enskit.algorithms import MultiplicativeInflation, enkf

problem = toy.lorenz96(state_dim=40, n_times=300, obs_every=2, noise_std=1.0)
ensemble = problem.initial.sample(jax.random.key(0), n_particles=40)

result = enkf.filter(
    ensemble, problem.observations,
    transition=problem.transition,      # (J, 40) -> (J, 40)
    observe=problem.observe,            # maps.Linear: every other site
    noise_cov=problem.noise_cov,        # R = I
    update_rule=kalman.Matheron(),
    inflation=MultiplicativeInflation(1.05),
    key=jax.random.key(1),
)
```

Each time, the filter forecasts, multiplies the forecast anomalies by 1.05
(Anderson & Anderson, 1999), and applies the perturbed-observation update
(Burgers, van Leeuwen & Evensen, 1998). `result.means["x"]` holds the `(300,
40)` analysis means, one row per observation. The error against the truth,
averaged over the 40 sites and over the times after the first 50, is well
below the observation noise, although only half the sites are observed:

```python
def rmse(means, truth):
    return jnp.sqrt(jnp.mean((means - truth) ** 2, axis=1))

error = rmse(result.means["x"], problem.truth)[50:].mean()   # 0.308
```

The update rule is required and has no default. `kalman.Matheron()` draws
perturbations and needs the key; `kalman.SymmetricSquareRoot()`, the ensemble
transform filter (Bishop, Etherton & Majumdar, 2001), is deterministic, and a
filter with it and no random policy needs no key at all.

## The log evidence

At each time the analysis also reports the density of the observation under
the forecast's Gaussian approximation,

$$
\log \hat p(y_t \mid y_{1:t-1}) =
\log \mathcal N\big(y_t;\ \bar h_t,\ \hat C_{hh,t} + R\big),
$$

with $\bar h_t$ and $\hat C_{hh,t}$ the sample mean and covariance of the
predicted observations. `result.log_evidence` holds the `(300,)` values and
`result.total_log_evidence` their sum, an estimate of $\log p(y_{1:T})$
(Carrassi et al., 2017). On its own the number means little. Its use is
comparing settings on the same observations **without knowing the truth**.
Here it chooses the inflation factor:

```python
def run(anomaly_scale):
    return enkf.filter(
        ensemble, problem.observations,
        transition=problem.transition, observe=problem.observe,
        noise_cov=problem.noise_cov, update_rule=kalman.SymmetricSquareRoot(),
        inflation=MultiplicativeInflation(anomaly_scale),
    )

scales = (1.0, 1.05, 1.1, 1.2)
runs = [run(s) for s in scales]
evidence = [float(r.total_log_evidence) for r in runs]
errors = [float(rmse(r.means["x"], problem.truth)[50:].mean()) for r in runs]
```

| `anomaly_scale` | 1.0 | 1.05 | 1.1 | 1.2 |
| --------------- | --- | ---- | --- | --- |
| total log evidence | −8769.5 | −8808.3 | −9016.5 | −9520.5 |
| error | 0.290 | 0.316 | 0.406 | 0.614 |

The evidence ranks the four the same way as the error, which needs the truth.
With 40 particles and the deterministic update, this filter does best
without inflation. For a smaller ensemble or another update rule, the same
comparison chooses the factor.

## Checking against the exact filter

With a linear transition and observation model and no transition noise, the
filter is exact: started from particles whose sample mean and covariance are
the initial distribution's, with at least one more particle than the state
has dimensions and the deterministic update, it reproduces the Kalman filter
(Kalman, 1960) to round-off. `toy.linear_state_space` is such a problem, and
its `exact_filter()` computes the answer with the exact operations of
{doc}`distributions`:

```python
from enskit.distribution import exact_moment_ensemble

linear = toy.linear_state_space(state_dim=3, data_dim=2, n_times=20,
                                transition_noise_std=0.0)
particles = exact_moment_ensemble(jax.random.key(0), linear.initial, n_particles=4)
filtered = enkf.filter(
    particles, linear.observations,
    transition=linear.transition, observe=linear.observe,
    noise_cov=linear.noise_cov, update_rule=kalman.SymmetricSquareRoot(),
)

exact, exact_log_evidence = linear.exact_filter()
exact_means = jnp.stack([g.mean("x") for g in exact])
mean_gap = jnp.max(jnp.abs(filtered.means["x"] - exact_means))            # 4e-16
evidence_gap = jnp.max(jnp.abs(filtered.log_evidence - exact_log_evidence))  # 1e-15
```

This is the check to run when a filter of your own misbehaves: if it is exact
here, the trouble is in the model or the ensemble size, not in the update.

## Small ensembles

With fewer particles than state dimensions the sample covariance has rank at
most $J - 1$, so every correction lies in the span of the forecast anomalies,
and spurious correlations between distant sites do the rest. With 10
particles the global filter above loses the truth. Domain localization
({doc}`localization`) is an update rule, so it goes in as `update_rule` and
nothing else changes:

```python
def periodic(point, points):          # distance on the ring of 40 sites
    d = jnp.abs(points[:, 0] - point[0])
    return jnp.minimum(d, 40.0 - d)

localization = kalman.DomainLocalization(
    target_coords={"x": problem.coords},     # (40, 1): the sites
    given_coords=problem.obs_coords,         # (20, 1): the observed sites
    radius=8.0, max_neighbors=10, distance=periodic,
)

def small(update_rule):
    particles = problem.initial.sample(jax.random.key(0), n_particles=10)
    run = enkf.filter(
        particles, problem.observations,
        transition=problem.transition, observe=problem.observe,
        noise_cov=problem.noise_cov, update_rule=update_rule,
        inflation=MultiplicativeInflation(1.05),
    )
    return rmse(run.means["x"], problem.truth)[50:].mean()

error_global = small(kalman.SymmetricSquareRoot())                    # 3.613
error_local = small(kalman.LocalizedUpdateRule(
    kalman.SymmetricSquareRoot(), localization))                      # 0.356
```

The localized filter is the local ensemble transform Kalman filter (Hunt,
Kostelich & Szunyogh, 2007). A static covariance blended with the sample one,
a hybrid filter (Hamill & Snyder, 2000), is the other common remedy; it goes in
as `approximation=` with `kalman.Matheron()`, and {doc}`../enkf-contract`
says how it must be built.

(writing-the-loop)=
## Writing the loop

`enkf.filter` is a loop over two functions, and writing it out is how a filter
does something the one call does not. Every block of the ensemble is updated
by each analysis, through its sample correlation with the predicted
observations, so a parameter the transition reads is estimated along with the
state (Anderson, 2001; Evensen, 2009). Here the Lorenz-96 forcing $F$, which
generated the data at 8, starts unknown with a prior $\mathcal N(6, 1)$:

```python
from enskit.distribution import Ensemble

short = toy.lorenz96(n_times=100)
k_state, k_forcing, key = jax.random.split(jax.random.key(2), 3)
ens = Ensemble(
    x=short.initial.sample(k_state, n_particles=40)["x"],
    forcing=6.0 + jax.random.normal(k_forcing, (40, 1)),
)

for y in short.observations:
    key, k_analysis = jax.random.split(key)
    ens = enkf.forecast(ens, toy.lorenz96_step, state="x",
                        inputs=("x", "forcing"))      # reads two blocks
    ens = kalman.inflate_multiplicative(ens, 1.05)
    ens, _ = enkf.analysis(ens, y, observe=short.observe,
                           noise_cov=short.noise_cov,
                           update_rule=kalman.Matheron(),
                           inputs="x", key=k_analysis)

forcing = ens.mean("forcing")[0]      # 8.025
```

`toy.lorenz96_step(x, forcing)` broadcasts the `(40, 1)` forcing block against
the `(40, 40)` states, so each particle runs with its own forcing. The
analysis's `inputs` names the blocks the observation model reads, and is
required, since an ensemble usually holds blocks it must not read. A copy of
the state assigned before each forecast, `ens.assign(x_prev=ens["x"])`, is
updated the same way, which is the lag-1 ensemble Kalman smoother (Evensen &
van Leeuwen, 2000).

## Inflation, relaxation and failures

The filter calls an inflation after each forecast and a relaxation after each
analysis, with the shared policies of `enskit.algorithms` and `time=t` as
context. The relaxation's prior is the *inflated* forecast, the particles the
analysis started from, so the two compound: `RelaxToPriorSpread(1.0)` after
`MultiplicativeInflation(1.05)` restores the inflated spread and undoes the
analysis's reduction entirely. Use one or the other, or both only with a small
inflation and a partial relaxation.

If any stage returns a particle that is not finite, the filter raises
`enkf.EnKFError`, which carries the time, the key and the result so far, so
that `enkf.filter(exc.result.ensemble, observations[exc.time:],
key=exc.key, start_time=exc.time, ...)` resumes with the draws and the times
the uninterrupted filter would have used.

## References

- Anderson, J. L. (2001). An ensemble adjustment Kalman filter for data
  assimilation. *Monthly Weather Review*, 129(12), 2884–2903.
- Anderson, J. L. & Anderson, S. L. (1999). A Monte Carlo implementation of
  the nonlinear filtering problem to produce ensemble assimilations and
  forecasts. *Monthly Weather Review*, 127(12), 2741–2758.
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
- Hunt, B. R., Kostelich, E. J. & Szunyogh, I. (2007). Efficient data
  assimilation for spatiotemporal chaos: a local ensemble transform Kalman
  filter. *Physica D*, 230(1–2), 112–126.
- Kalman, R. E. (1960). A new approach to linear filtering and prediction
  problems. *Journal of Basic Engineering*, 82(1), 35–45.
- Lorenz, E. N. (1996). Predictability: a problem partly solved. In
  *Proceedings of the ECMWF Seminar on Predictability*, vol. 1, 1–18.
  Reading, UK.
