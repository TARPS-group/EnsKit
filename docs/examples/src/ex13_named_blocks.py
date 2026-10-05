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
# when the ensemble is edited between times, as the copy is here.
#
# ## Setup
#
# The state $x \in \mathbb R^{40}$ lives on a ring of sites $i = 0, \dots, 39$
# (indices modulo 40) and follows the Lorenz-96 equations [@lorenz1996]
#
# $$
# \frac{dx_i}{dt} = (x_{i+1} - x_{i-2})\, x_{i-1} - x_i + F .
# $$
#
# One time step is one fourth-order Runge-Kutta step of length
# $\Delta t = 0.05$, written $x_t = M(x_{t-1}, F)$, with no model noise. The
# data were generated with $F = 8$: the true trajectory starts at
# $8 + 0.01\, z$ for a standard normal $z \in \mathbb R^{40}$, is run for 1000
# steps onto the attractor to give $x_0^\dagger$, and then for $T = 200$
# further steps, $x_t^\dagger = M(x_{t-1}^\dagger, 8)$. After each step the
# 20 even-numbered sites are observed,
#
# $$
# y_t = H x_t^\dagger + e_t, \qquad e_t \sim \mathcal N(0, R), \quad
# R = I_{20}, \qquad t = 1, \dots, T,
# $$
#
# with $H \in \mathbb R^{20 \times 40}$ selecting sites $0, 2, \dots, 38$.
# The forcing is treated as unknown, with prior $F \sim \mathcal N(6, 1)$.
# Each of the $J = 40$ particles carries its own state $x_j$, drawn from
# $\mathcal N(x_0^\dagger, I_{40})$, and its own forcing $F_j$, drawn from
# the prior; the loop adds a third block, the previous state
# $x_{\mathrm{prev}, j}$.

# %%
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, toy
from enskit.algorithms import enkf
from enskit.distribution import Ensemble

problem = toy.lorenz96(state_dim=40, n_times=200, obs_every=2, noise_std=1.0)
print(problem, "  true forcing", problem.forcing)

key = jax.random.key(0)
k1, k2, key = jax.random.split(key, 3)
ens = Ensemble(
    x=problem.initial.sample(k1, n_particles=40)["x"],
    forcing=6.0 + jax.random.normal(k2, (40, 1)),    # unknown parameter, prior N(6, 1)
)

# %% [markdown]
# ## A transition that reads two blocks
#
# Each particle is advanced with its own forcing value, so the transition
# reads `x` and `forcing` and writes `x`. The `inputs` argument of
# `enkf.forecast` names the blocks it reads, in the order the transition
# takes them, and the forecast leaves every other block unchanged.
# `toy.lorenz96_step` broadcasts a `(J, 1)` forcing against the `(J, 40)`
# states.

# %%
def transition(x, forcing):                              # (J, 40) and (J, 1) -> (J, 40)
    return toy.lorenz96_step(x, forcing, dt=problem.dt)

# %% [markdown]
# ## The loop
#
# At each time the current state is copied into `x_prev`, the state is
# advanced, and the anomalies of every block are multiplied by 1.05
# [@anderson1999], so that the forcing's spread does not collapse either. The
# analysis then predicts $g_j = H x_j$ and applies the perturbed-observation
# update [@burgers1998; @houtekamer1998] to every block
# $z \in \{x, F, x_{\mathrm{prev}}\}$,
#
# $$
# z_j \mapsto z_j + C_{zg}\,(C_{gg} + R)^{-1}\,(y_t - g_j - e_j), \qquad
# e_j \sim \mathcal N(0, R),
# $$
#
# with $C_{zg}$ and $C_{gg}$ sample covariances over the inflated particles.
# After assimilating $y_t$, the mean of `x` estimates $x_t^\dagger$, the mean
# of `x_prev` estimates $x_{t-1}^\dagger$ given the data up to time $t$, and
# the mean of `forcing` estimates $F$.

# %%
filtered, smoothed = [], []
for y in problem.observations:
    key, k_analysis = jax.random.split(key)
    ens = ens.assign(x_prev=ens["x"])                    # carry the previous state along
    ens = enkf.forecast(ens, transition, state="x", inputs=("x", "forcing"))
    ens = kalman.inflate_multiplicative(ens, 1.05)
    ens, _ = enkf.analysis(ens, y, observe=problem.observe,
                           noise_cov=problem.noise_cov, update_rule=kalman.Matheron(),
                           inputs=("x",), key=k_analysis)
    # one update moved every block it did not condition on: x, forcing and x_prev
    filtered.append(ens.mean("x"))
    smoothed.append(ens.mean("x_prev"))
forcing = float(ens.mean("forcing")[0])

# %% [markdown]
# ## Scoring
#
# Filter and smoother are compared on the same states $x_t^\dagger$,
# $t = 51, \dots, 199$, by the root-mean-square error over the 40 sites,
# averaged over time. The filter's estimate of $x_t^\dagger$ is the mean of
# `x` after assimilating $y_t$; the smoother's is the mean of `x_prev` after
# assimilating $y_{t+1}$.

# %%
def rmse(means, truth):
    return float(jnp.sqrt(jnp.mean((means - truth) ** 2, axis=1))[50:].mean())


filtered, smoothed = jnp.stack(filtered), jnp.stack(smoothed)
filter_error = rmse(filtered[:-1], problem.truth[:-1])
smoother_error = rmse(smoothed[1:], problem.truth[:-1])
print(f"filter RMSE {filter_error:.2f}   lag-1 smoother RMSE {smoother_error:.2f}")
print(f"forcing estimate {forcing:.2f} (truth 8)")

# %% [markdown]
# ## What to notice
#
# The forcing estimate moves from the prior mean of 6 to within 0.3 of the
# value 8 that generated the data. With one more time of data, the lag-1
# smoother has a lower error than the filter for the same states; both are
# about 0.3, well below the observation noise standard deviation of 1. Both
# results come from one call to `enkf.analysis` per time, which moves every
# block it does not condition on.
#
# The printed numbers are those of the machine that built this page. A
# chaotic system amplifies rounding: the spin-up that produces the true
# trajectory turns a difference in the last digit of one arithmetic operation
# into a different trajectory, so another platform filters a different truth
# at the same seed and prints different digits. The checks below test bands.

# %%
# checks
assert abs(forcing - 8.0) < 0.3
assert smoother_error < filter_error < 0.6
