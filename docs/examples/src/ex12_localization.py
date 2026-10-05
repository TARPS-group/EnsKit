# %% [markdown]
# # Domain localization
#
# Track a 40-dimensional Lorenz-96 state with 10 particles, comparing a global
# ensemble transform update with two domain-localized ones.
#
# ## When to use this
#
# Localization is needed when the ensemble is much smaller than the state and
# the state is spatially extended, so that each coordinate is informed mainly
# by nearby data. Domain localization runs a separate analysis for each state
# coordinate using only the data near it, with the noise of each datum
# inflated by a taper of its distance, as in the local ensemble transform
# Kalman filter (LETKF) [@ott2004; @hunt2007]. Covariance localization, which
# tapers the sample covariance itself, is the main alternative
# [@houtekamer2001; @hamill2001]; the two are closely related [@sakov2011].
# `kalman.LocalizedUpdateRule` provides domain localization around either
# shipped update rule.
#
# ## Setup
#
# The state $x \in \mathbb R^{40}$ lives on a ring of sites $i = 0, \dots, 39$
# (indices modulo 40) and follows the Lorenz-96 equations [@lorenz1996]
#
# $$
# \frac{dx_i}{dt} = (x_{i+1} - x_{i-2})\, x_{i-1} - x_i + F, \qquad F = 8 .
# $$
#
# One time step is one fourth-order Runge-Kutta step of length
# $\Delta t = 0.05$, written $x_t = M(x_{t-1})$, with no model noise. The true
# trajectory starts at $F + 0.01\, z$ for a standard normal
# $z \in \mathbb R^{40}$, is run for 1000 steps onto the attractor to give
# $x_0^\dagger$, and then for $T = 300$ further steps,
# $x_t^\dagger = M(x_{t-1}^\dagger)$. After each step the 20 even-numbered
# sites are observed,
#
# $$
# y_t = H x_t^\dagger + e_t, \qquad e_t \sim \mathcal N(0, R), \quad
# R = I_{20}, \qquad t = 1, \dots, T,
# $$
#
# with $H \in \mathbb R^{20 \times 40}$ selecting sites $0, 2, \dots, 38$, so
# datum $p$ lives at site $s_p = 2p$ and has noise variance $r_p = 1$. The
# $J = 10$ initial particles are drawn from $\mathcal N(x_0^\dagger, I_{40})$.
# Before each analysis the anomalies of the forecast are scaled,
# $x_j \mapsto \bar x + \lambda (x_j - \bar x)$ with $\lambda = 1.05$
# [@anderson1999]. The score is the root-mean-square error of the analysis
# mean $\bar x_t$,
#
# $$
# \mathrm{RMSE}_t = \Big(\tfrac{1}{40} \lVert \bar x_t - x_t^\dagger \rVert^2\Big)^{1/2},
# $$
#
# averaged over the last 250 times, $t = 51, \dots, 300$ (rows 50 to 299 of
# the filter's output).

# %%
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, toy
from enskit.algorithms import MultiplicativeInflation, enkf

problem = toy.lorenz96(state_dim=40, n_times=300, obs_every=2, noise_std=1.0)
print(problem)

# %% [markdown]
# ## The localization
#
# State site $i$ and datum $p$ are at periodic distance
# $d_{ip} = \min(|i - s_p|,\ 40 - |i - s_p|)$. The analysis of $x_i$ uses the
# $K = 10$ data nearest to $i$, each with weight
#
# $$
# \rho_{ip} = \mathrm{GC}(d_{ip} / L), \qquad L = 8,
# $$
#
# where $\mathrm{GC}$ is the Gaspari-Cohn taper [@gaspari1999], equal to 1 at
# zero distance and to 0 for $d_{ip} \ge L$. Datum $p$ enters with noise
# variance $r_p / \rho_{ip}$ in place of $r_p$, so data at distance 8 or more
# have no influence. Every datum closer than 8 to a site is among its 10
# nearest, so here $K$ truncates nothing.

# %%
def periodic(point, points):              # distance on the ring of 40 sites
    d = jnp.abs(points[:, 0] - point[0])
    return jnp.minimum(d, 40.0 - d)


localization = kalman.DomainLocalization(
    target_coords={"x": problem.coords},  # (40, 1): where each state coordinate lives
    given_coords=problem.obs_coords,      # (20, 1): where each predicted datum lives
    radius=8.0, max_neighbors=10, distance=periodic,
)

# %% [markdown]
# ## Three filters
#
# Each run calls `enkf.filter` with the same particles, data, inflation and
# keys, and differs only in the update rule. Let $g_j = H x_j$ be the
# predicted data of particle $j$ of the inflated forecast, $\bar g$ their
# mean, $f_i \in \mathbb R^J$ the scaled anomalies
# $(x_{ji} - \bar x_i)/\sqrt{J-1}$ of site $i$, and, for the data
# $p_1, \dots, p_K$ nearest to $i$,
#
# $$
# (S_i)_{jk} = \sqrt{\rho_{ip_k} / r_{p_k}}\
#   \frac{g_{jp_k} - \bar g_{p_k}}{\sqrt{J-1}}, \qquad
# (b_i)_k = \sqrt{\rho_{ip_k} / r_{p_k}}\ (y_{t,p_k} - \bar g_{p_k}),
# \qquad A_i = I_J + S_i S_i^\top .
# $$
#
# The localized symmetric square-root update [@bishop2001; @wang2004; @hunt2007]
# sets
#
# $$
# x_{ji}^a = \bar x_i + f_i^\top A_i^{-1} S_i b_i
#   + \sqrt{J-1}\,\big(A_i^{-1/2} f_i\big)_j .
# $$
#
# The stochastic variant wraps `kalman.Matheron` instead [@burgers1998]: each
# particle uses its own residual, perturbed by one standard normal draw
# $\varepsilon \in \mathbb R^{J \times 20}$ shared by all local analyses,
#
# $$
# x_{ji}^a = x_{ji} + f_i^\top A_i^{-1} S_i b_{ij}, \qquad
# (b_{ij})_k = \sqrt{\rho_{ip_k} / r_{p_k}}\ (y_{t,p_k} - g_{jp_k})
#   - \varepsilon_{jp_k} .
# $$
#
# In the local problem's whitened coordinates $\varepsilon_{jp_k}$ is a draw
# of noise of variance $r_{p_k} / \rho_{ip_k}$.

# %%
def run(update_rule):
    ensemble = problem.initial.sample(jax.random.key(0), n_particles=10)
    result = enkf.filter(
        ensemble, problem.observations,
        transition=problem.transition, observe=problem.observe,
        noise_cov=problem.noise_cov, update_rule=update_rule,
        inflation=MultiplicativeInflation(1.05), key=jax.random.key(1),
    )
    rmse = jnp.sqrt(jnp.mean((result.means["x"] - problem.truth) ** 2, axis=1))
    return float(rmse[50:].mean())

# %% [markdown]
# The global update is the same square-root formula with all 20 data and
# $\rho \equiv 1$. Its sample covariance has rank at most $J - 1 = 9 < 40$,
# so every correction lies in the 9-dimensional span of the forecast
# anomalies. Each local analysis has its own weights, so the localized
# corrections taken together are not confined to that span.

# %%
rmse_global = run(kalman.SymmetricSquareRoot())
rmse_local = run(kalman.LocalizedUpdateRule(kalman.SymmetricSquareRoot(), localization))
rmse_local_stochastic = run(kalman.LocalizedUpdateRule(kalman.Matheron(), localization))
print(f"J = 10   global ETKF {rmse_global:.1f}   local ETKF {rmse_local:.1f}   "
      f"local stochastic EnKF {rmse_local_stochastic:.1f}")

# %% [markdown]
# ## What to notice
#
# The global update has an average error of about 3.6, well above the
# observation noise standard deviation of 1, so with 10 particles it does not
# track the truth. Both localized filters do, with errors of about 0.4 and
# under half the global one. The localization object is the same in both;
# only the wrapped rule changes.
#
# The printed numbers are those of the machine that built this page. A
# chaotic system amplifies rounding: the spin-up that produces the true
# trajectory turns a difference in the last digit of one arithmetic operation
# into a different trajectory, so another platform filters a different truth
# at the same seed and prints different digits. The checks below test bands.

# %%
# checks
assert 2.0 < rmse_global < 6.0                  # the global filter loses the truth
for local in (rmse_local, rmse_local_stochastic):
    assert local < 0.7 and local < 0.5 * rmse_global
