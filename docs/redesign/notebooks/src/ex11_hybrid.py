# %% [markdown]
# # A hybrid ensemble Kalman filter
#
# Filter the 40-variable Lorenz-96 system with ten particles, once with the
# sample covariance alone and once with a blend of the sample covariance and a
# static covariance, passed to the filter through its `approximation=` argument.
#
# ## When to use this
#
# With fewer particles than state dimensions, the sample covariance is rank
# deficient and its small correlations between distant coordinates are mostly
# sampling noise. If a static covariance is available, for example a
# climatological one, blending it with the sample covariance fills in the
# missing directions [@hamill2000]. Consider a hybrid when the ensemble is
# small, the observation model is linear, and such a static covariance exists.
#
# ## Setup
#
# The state $x \in \mathbb R^{40}$ follows the Lorenz-96 equations with
# forcing $F = 8$ on a periodic ring,
#
# $$
# \frac{dx_k}{dt} = (x_{k+1} - x_{k-2})\, x_{k-1} - x_k + F,
# $$
#
# advanced by one fourth-order Runge-Kutta step of length $0.05$ per time step,
# with no transition noise. Every second coordinate is observed at each of
# 300 steps, $y_t = H x_t + e_t$ with $H \in \mathbb R^{20 \times 40}$ a
# selection matrix and $e_t \sim \mathcal N(0, I_{20})$. The initial
# ensemble of $J = 10$ particles is drawn from $\mathcal N(x_0^\dagger, I_{40})$.
# At each analysis the hybrid filter replaces the sample covariance
# $\hat C$ of the forecast ensemble by
#
# $$
# C = \alpha\, \hat C + (1 - \alpha)\, B, \qquad
# B_{ij} = 0.3\, e^{-d_{ij}^2/2} + 10^{-6} \delta_{ij}, \qquad \alpha = 0.2,
# $$
#
# where $d_{ij} = \min(|i - j|,\ 40 - |i - j|)$ is the distance on the ring, and
# moves each particle by $x_j \leftarrow x_j + K (y_t - H x_j - e_j)$ with
# $K = C H^\top (H C H^\top + R)^{-1}$ and a fresh draw $e_j \sim \mathcal N(0, R)$.
# The score is the root-mean-square error of the analysis mean, averaged over
# steps 51 to 300.

# %%
import math

import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, maps, toy
from enskit.distribution import Gaussian
from enskit.algorithms import MultiplicativeInflation, enkf
from enskit.linalg import DensePSD

problem = toy.lorenz96(dim=40, n_steps=300, obs_every=2, noise_sd=1.0)
d = jnp.abs(jnp.arange(40)[:, None] - jnp.arange(40)[None, :])
d = jnp.minimum(d, 40 - d)
B = DensePSD(0.3 * jnp.exp(-0.5 * d**2) + 1e-6 * jnp.eye(40))

# %% [markdown]
# ## The hybrid approximation
#
# The filter conditions a joint Gaussian of the state and the predicted data,
# built by a function passed as `approximation=`. The default,
# `kalman.gaussian_approximation`, moment-matches the particles. The function
# below builds a `Gaussian` over `"x"` with the ensemble mean, the factor
# $\sqrt{\alpha}\, X$, where $X$ holds the anomalies scaled so that
# $\hat C = X X^\top$, and the independent term $(1 - \alpha) B$, then pushes
# it through $H$. Because $H$ is linear, $C_{xy} = C H^\top$ and
# $C_{yy} = H C H^\top + R$ hold exactly for any $C$, including $B$, which is
# not represented by any particles. For a nonlinear observation model these
# blocks would have to be estimated from particles, and $B$ could not enter.

# %%
def hybrid_approximation(ensemble, noise, alpha=0.2):
    """The joint Gaussian approximation, with alpha * sample cov + (1 - alpha) * B on "x"."""
    ((given, R),) = noise.items()
    fit = ensemble.marginal("x").project()
    hybrid = Gaussian({"x": fit.mean("x")},
                           factors={"x": fit.factor("x") * math.sqrt(alpha)},
                           block_covs={"x": B * (1.0 - alpha)})
    return (hybrid
            .pipe(maps.pushforward, problem.observe, inputs="x", output=given)
            .add_noise({given: R}))

# %% [markdown]
# ## Two runs of one filter
#
# Both runs use the same particles, keys, stochastic `Matheron` update and
# multiplicative inflation of the anomalies by $1.05$ before each analysis. They
# differ only in the `approximation` argument.

# %%
def run(**options):
    ens = problem.initial.sample(jax.random.key(0), n_particles=10)
    result = enkf.filter(jax.random.key(1), ens, problem.observations,
                         transition=problem.transition, observe=problem.observe,
                         noise_cov=problem.noise_cov, update_rule=kalman.Matheron(),
                         inflation=MultiplicativeInflation(1.05), **options)
    return jnp.sqrt(jnp.mean((result.means["x"] - problem.truth) ** 2, axis=1))[50:].mean()


rmse_plain = run()
rmse_hybrid = run(approximation=hybrid_approximation)
print(f"J = 10   plain EnKF RMSE {rmse_plain:.3f}   hybrid EnKF RMSE {rmse_hybrid:.3f}")

# %% [markdown]
# ## What to notice
#
# With ten particles for 40 coordinates the plain filter has an RMSE of 4.783,
# larger than the observation noise standard deviation of 1. The hybrid filter
# reaches 0.669. Localization, shown in Example 12, does better on this system
# with the same ten particles, with RMSEs of 0.411 for the localized stochastic
# update and 0.388 for the localized square-root update. The two methods are not
# exclusive, since localization acts on the sample part of the covariance and
# a static part can be blended in alongside it.

# %%
# checks
assert rmse_hybrid < 0.5 * rmse_plain
