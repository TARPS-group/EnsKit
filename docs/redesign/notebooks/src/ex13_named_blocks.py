# %% [markdown]
# # Named blocks: parameters and smoothing
#
# Filter a Lorenz-96 state while estimating its unknown forcing and a lag-1
# smoothed state, by carrying both as extra blocks of the ensemble.
#
# ## When to use this
#
# An ensemble update moves every block of the ensemble that is not
# conditioned on, using its sample correlation with the predicted data. A
# static parameter appended to the state is therefore estimated by the same
# update as the state, which is the state augmentation approach to joint
# state-parameter estimation [@anderson2001; @evensen2009]. A copy of the
# previous state, appended in the same way, is updated by the new data,
# which gives the lag-1 ensemble Kalman smoother [@evensen2000]. Writing the
# loop by hand with `enkf.forecast` and `enkf.analysis` is the right level
# when the transition reads blocks other than the state, or when the
# ensemble is edited between steps.
#
# ## Setup
#
# The state $x \in \mathbb R^{40}$ follows the Lorenz-96 model on a ring of
# 40 sites [@lorenz1996],
#
# $$
# \frac{dx_i}{dt} = (x_{i+1} - x_{i-2})\, x_{i-1} - x_i + F,
# $$
#
# advanced between data times by one fourth-order Runge-Kutta step of length
# $\Delta t = 0.05$, written $x_t = M(x_{t-1}, F)$. The data were generated
# with $F = 8$; at each of 200 times
#
# $$
# y_t = H x_t + e_t, \qquad e_t \sim \mathcal N(0, R), \quad R = I_{20},
# $$
#
# with $H$ selecting the 20 even-numbered sites. The forcing is unknown, with
# prior $F \sim \mathcal N(6, 1)$. The $J = 40$ particles carry three blocks,
# $(x_j, F_j, x_{\mathrm{prev}, j})$, with $x_j$ initially drawn from
# $\mathcal N(x_0, I_{40})$ around the true initial state $x_0$.

# %%
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, toy
from enskit.distribution import Ensemble
from enskit.algorithms import enkf

problem = toy.lorenz96(dim=40, n_steps=200, obs_every=2, noise_sd=1.0)   # true forcing 8

key = jax.random.key(0)
k1, k2, key = jax.random.split(key, 3)
ens = Ensemble(
    x=problem.initial.sample(k1, 40)["x"],
    forcing=6.0 + jax.random.normal(k2, (40, 1)),        # unknown parameter, prior N(6, 1)
)

# %% [markdown]
# ## A transition that reads two blocks
#
# Each particle is advanced with its own forcing value, so the transition
# reads `x` and `forcing` and writes `x`. The `inputs` argument of
# `enkf.forecast` names the blocks it reads.

# %%
def transition(x, forcing):                              # reads two blocks, writes one
    return toy.l96_step(x, forcing)

# %% [markdown]
# ## The loop
#
# At each time the current state is copied into `x_prev`, the state is
# advanced, and the anomalies of every block are multiplied by 1.02
# [@anderson1999]. The analysis then predicts $g_j = H x_j$ and applies the
# stochastic update [@burgers1998] to every block $z \in \{x, F,
# x_{\mathrm{prev}}\}$,
#
# $$
# z_j \leftarrow z_j + C_{zg}\,(C_{gg} + R)^{-1}\,(y_t - g_j - e_j), \qquad
# e_j \sim \mathcal N(0, R),
# $$
#
# with $C_{zg}$ and $C_{gg}$ sample covariances over the particles. After
# assimilating $y_t$, the mean of `x` estimates $x_t$, the mean of `x_prev`
# estimates $x_{t-1}$ given data up to time $t$, and the mean of `forcing`
# estimates $F$.

# %%
filtered, smoothed, forcing = [], [], []
for y in problem.observations:
    key, k_fc, k_an = jax.random.split(key, 3)
    ens = ens.assign(x_prev=ens["x"])                    # carry the previous state along
    ens = enkf.forecast(k_fc, ens, transition, state="x", inputs=("x", "forcing"))
    ens = kalman.inflate_multiplicative(ens, 1.02)
    ens, _ = enkf.analysis(k_an, ens, y, observe=problem.observe,
                           noise_cov=problem.noise_cov, update_rule=kalman.Matheron(),
                           inputs=("x",))
    # one update moved every non-given block: x (filter), x_prev (smoother), forcing
    filtered.append(ens.mean("x"))
    smoothed.append(ens.mean("x_prev"))
    forcing.append(ens.mean("forcing")[0])

# %% [markdown]
# ## Scoring
#
# Filter and smoother are compared on the same states $x_t$, $t = 50, \dots,
# 198$, by the root-mean-square error over the 40 sites, averaged over time.

# %%
filtered, smoothed = jnp.stack(filtered), jnp.stack(smoothed)
rmse = lambda m, t: jnp.sqrt(jnp.mean((m - t) ** 2, axis=1))[50:].mean()  # noqa: E731
print(f"filter RMSE at t-1 {rmse(filtered[:-1], problem.truth[:-1]):.3f}   "
      f"lag-1 smoother RMSE {rmse(smoothed[1:], problem.truth[:-1]):.3f}")
print(f"forcing estimate {forcing[-1]:.3f} (truth 8)")

# %% [markdown]
# ## What to notice
#
# The forcing estimate moves from the prior mean of 6 to 8.038, close to the
# value of 8 that generated the data. With one more time of data, the lag-1
# smoother has a lower error than the filter for the same states, 0.335
# against 0.362. Both results come from one call to `enkf.analysis` per
# time, which moves every block it does not condition on.

# %%
# checks
assert rmse(smoothed[1:], problem.truth[:-1]) < rmse(filtered[:-1], problem.truth[:-1])
assert abs(forcing[-1] - 8.0) < 0.3
