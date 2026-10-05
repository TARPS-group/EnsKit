# %% [markdown]
# # Ensemble Kalman filter
#
# Track the state of a chaotic 40-variable system from noisy observations of
# half of its coordinates, with one call to the filtering driver.
#
# ## When to use this
#
# The ensemble Kalman filter (EnKF) estimates the state of a dynamical system
# that is observed sequentially in time [@evensen1994]. It needs only the
# ability to run the model forward, and it represents uncertainty with a
# small ensemble even when the state is large. `enkf.filter` is the right
# level of the API when the model, the observations and the update rule are
# fixed in advance and the analysis means are all you need from the run.
#
# ## Setup
#
# The state $x \in \mathbb R^{40}$ follows the Lorenz-96 equations
# [@lorenz1996], with cyclic indices and forcing $F = 8$:
#
# $$
# \frac{dx_i}{dt} = (x_{i+1} - x_{i-2})\, x_{i-1} - x_i + F,
# \qquad i = 0, \dots, 39 .
# $$
#
# One time step is one fourth-order Runge-Kutta step of size
# $\Delta t = 0.05$, written $x_{t} = M(x_{t-1})$, with no model noise. After
# a spin-up onto the attractor, the true state $x_0$ is run for $T = 300$
# steps and every other coordinate is observed at every step:
#
# $$
# y_t = H x_t + \varepsilon_t, \qquad
# \varepsilon_t \sim \mathcal N(0, R), \quad R = I_{20},
# $$
#
# where $H \in \mathbb R^{20 \times 40}$ selects the sites $i = 0, 2, \dots, 38$.
# The filter starts from $J = 40$ particles drawn from
# $\mathcal N(x_0, I_{40})$.

# %%
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, toy
from enskit.algorithms import MultiplicativeInflation, enkf

problem = toy.lorenz96(state_dim=40, n_times=300, obs_every=2, noise_std=1.0)

# %% [markdown]
# ## Filtering
#
# Each cycle moves every particle through $M$, then scales the anomalies
# about the ensemble mean, $x_j \mapsto \bar x + \lambda (x_j - \bar x)$ with
# $\lambda = 1.05$, to offset the spread the small ensemble loses
# [@anderson1999]. The analysis is Matheron's rule, the perturbed-observation
# update [@burgers1998; @houtekamer1998]:
#
# $$
# x_j \mapsto x_j + K (y_t - H x_j - e_j), \qquad
# K = P H^\top (H P H^\top + R)^{-1}, \quad e_j \sim \mathcal N(0, R),
# $$
#
# with $P$ the sample covariance of the inflated forecast. A smaller factor,
# 1.02 or 1.03, lets the filter lose the truth at some seeds; 1.05 tracks it
# at every seed tried.

# %%
ensemble = problem.initial.sample(jax.random.key(0), n_particles=40)   # Ensemble over "x"
result = enkf.filter(
    ensemble, problem.observations,
    transition=problem.transition,        # (J, 40) -> (J, 40), a plain JAX function
    observe=problem.observe,              # maps.Linear: every other coordinate
    noise_cov=problem.noise_cov,          # PSDDiagonal
    update_rule=kalman.Matheron(),
    inflation=MultiplicativeInflation(1.05),
    key=jax.random.key(1),
)

# %% [markdown]
# ## Error and evidence
#
# The error at step $t$ is the root mean square difference between the
# analysis mean and the truth over the 40 coordinates, averaged here over
# times $t = 51, \dots, 300$ (rows 50 to 299). The driver also returns, at
# each step, the log density of $y_t$ under the forecast's Gaussian approximation,
# $\log \mathcal N(y_t;\ H \bar x_t,\ H P_t H^\top + R)$. Their sum over the
# run estimates the log model evidence [@carrassi2017].

# %%
rmse = jnp.sqrt(jnp.mean((result.means["x"] - problem.truth) ** 2, axis=1))
print(f"time-averaged RMSE after spin-up: {float(rmse[50:].mean()):.2f}")
print(f"total log evidence: {float(result.total_log_evidence):.1f}")

# %% [markdown]
# ## What to notice
#
# The time-averaged error after spin-up is about 0.3, under a third of the
# observation noise standard deviation of 1.0, although only half of the
# coordinates are observed. The total log evidence has no meaning on its own;
# it becomes useful when compared with the same quantity for a different
# model or noise level on the same observations.
#
# The printed numbers are those of the machine that built this page. A
# chaotic system amplifies rounding: the spin-up that produces the true
# trajectory turns a difference in the last digit of one arithmetic operation
# into a different trajectory, so another platform filters a different truth
# at the same seed and prints different digits. The checks below test a band.

# %%
# checks
assert rmse[50:].mean() < 0.6   # well below the observation noise sd of 1.0
