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
# tapers the sample covariance itself, is the main alternative [@houtekamer2001; @hamill2001]; the two are
# closely related [@sakov2011]. `kalman.LocalizedUpdateRule` provides domain
# localization around either shipped update rule.
#
# ## Setup
#
# The state $x \in \mathbb R^{40}$ lives on a ring of sites $i = 0, \dots, 39$
# (indices modulo 40) and follows the Lorenz-96 model [@lorenz1996]
#
# $$
# \frac{dx_i}{dt} = (x_{i+1} - x_{i-2})\, x_{i-1} - x_i + F, \qquad F = 8,
# $$
#
# advanced between data times by one fourth-order Runge-Kutta step of length
# $\Delta t = 0.05$, with no model noise. At each of 300 times the data are
# the 20 even-numbered sites plus noise,
#
# $$
# y_t = H x_t + e_t, \qquad e_t \sim \mathcal N(0, I_{20}),
# $$
#
# with $H$ selecting sites $0, 2, \dots, 38$. The $J = 10$ initial particles
# are drawn from $\mathcal N(x_0, I_{40})$, with $x_0$ the true initial state.
# Before each analysis the anomalies are multiplied by 1.05 [@anderson1999].
# The score is the root-mean-square error of the analysis mean,
#
# $$
# \mathrm{RMSE}_t = \Big(\tfrac{1}{40} \lVert \bar x_t - x_t \rVert^2\Big)^{1/2},
# $$
#
# averaged over $t = 50, \dots, 299$.

# %%
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, toy
from enskit.algorithms import MultiplicativeInflation, enkf

problem = toy.lorenz96(dim=40, n_steps=300, obs_every=2, noise_sd=1.0)

# %% [markdown]
# ## The localization
#
# State site $i$ and datum $p$, located at site $s_p$, are at periodic distance
# $d_{ip} = \min(|i - s_p|,\ 40 - |i - s_p|)$. The analysis of $x_i$ uses the
# 10 data nearest to $i$, each with weight
#
# $$
# \rho_{ip} = \mathrm{GC}(d_{ip} / L), \qquad L = 8,
# $$
#
# where $\mathrm{GC}$ is the Gaspari-Cohn taper [@gaspari1999], equal to 1 at
# zero distance and to 0 for $d_{ip} \ge L$. Datum $p$ enters with noise
# variance $r_p / \rho_{ip}$ in place of $r_p = 1$, so data at distance 8 or
# more have no influence.

# %%
def periodic(a, b):                       # distance on the ring of 40 sites
    d = jnp.abs(b[:, 0] - a[0])
    return jnp.minimum(d, 40.0 - d)


localization = kalman.DomainLocalization(
    target_coords={"x": problem.coords},  # (40, 1): where each state coordinate lives
    given_coords=problem.obs_coords,      # (20, 1): where each predicted datum lives
    radius=8.0, max_neighbors=10, distance=periodic,
)

# %% [markdown]
# ## Three filters
#
# Each run calls `enkf.filter` with the same particles, data and inflation,
# and differs only in the update rule. Let $g_j = H x_j$ be the predicted
# data of particle $j$, $f_i \in \mathbb R^J$ the scaled anomalies
# $(x_{ji} - \bar x_i)/\sqrt{J-1}$ of site $i$, and, over the 10 data $p$
# nearest to $i$,
#
# $$
# (S_i)_{jp} = \sqrt{\rho_{ip} / r_p}\ \frac{g_{jp} - \bar g_p}{\sqrt{J-1}}, \qquad
# (b_i)_p = \sqrt{\rho_{ip} / r_p}\ (y_p - \bar g_p).
# $$
#
# The localized symmetric square-root update [@bishop2001; @wang2004; @hunt2007] sets
#
# $$
# \bar x_i^a = \bar x_i + f_i^\top (I + S_i S_i^\top)^{-1} S_i b_i, \qquad
# x_{ji}^a = \bar x_i^a + \sqrt{J-1}\,\big[f_i^\top (I + S_i S_i^\top)^{-1/2}\big]_j .
# $$
#
# The stochastic variant wraps `kalman.Matheron` instead: each particle uses
# its own residual, perturbed by one noise draw shared by all local
# analyses [@burgers1998].

# %%
def run(update_rule):
    ens = problem.initial.sample(jax.random.key(0), n_particles=10)
    result = enkf.filter(jax.random.key(1), ens, problem.observations,
                         transition=problem.transition, observe=problem.observe,
                         noise_cov=problem.noise_cov, update_rule=update_rule,
                         inflation=MultiplicativeInflation(1.05))
    return jnp.sqrt(jnp.mean((result.means["x"] - problem.truth) ** 2, axis=1))[50:].mean()

# %% [markdown]
# The global update uses the same formulas with all 20 data and $\rho \equiv 1$.
# Its sample covariance has rank at most $J - 1 = 9 < 40$, so every
# correction lies in the 9-dimensional span of the forecast anomalies. Each
# local analysis has its own weights, so the localized corrections are not
# confined to that span.

# %%
rmse_global = run(kalman.SymmetricSquareRoot())
rmse_local = run(kalman.LocalizedUpdateRule(kalman.SymmetricSquareRoot(), localization))
rmse_local_pw = run(kalman.LocalizedUpdateRule(kalman.Matheron(), localization))
print(f"J = 10   global ETKF {rmse_global:.3f}   local ETKF {rmse_local:.3f}   "
      f"local stochastic EnKF {rmse_local_pw:.3f}")

# %% [markdown]
# ## What to notice
#
# The global update has an average error of 4.667, well above the
# observation noise standard deviation of 1, so it does not track the truth
# with 10 particles. Both localized filters do, with errors of 0.388 for the
# square-root form and 0.411 for the stochastic form. The localization
# object is the same in both; only the wrapped rule changes.

# %%
# checks
assert rmse_local < 0.5 and rmse_local < 0.5 * rmse_global
