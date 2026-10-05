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
# $x_t^\dagger = M(x_{t-1}^\dagger)$. After each step every second site is
# observed,
#
# $$
# y_t = H x_t^\dagger + e_t, \qquad e_t \sim \mathcal N(0, R), \quad
# R = I_{20}, \qquad t = 1, \dots, T,
# $$
#
# with $H \in \mathbb R^{20 \times 40}$ selecting sites $0, 2, \dots, 38$. The
# initial ensemble of $J = 10$ particles is drawn from
# $\mathcal N(x_0^\dagger, I_{40})$. Each forecast is inflated,
# $x_j \mapsto \bar x + \lambda (x_j - \bar x)$ with $\lambda = 1.05$
# [@anderson1999], and $\hat C$ denotes the sample covariance of the
# inflated forecast. The hybrid filter replaces $\hat C$ at each analysis by
#
# $$
# C = \alpha\, \hat C + (1 - \alpha)\, B, \qquad
# B_{ik} = 0.3\, e^{-d_{ik}^2/2} + 10^{-6} \delta_{ik}, \qquad \alpha = 0.2,
# $$
#
# where $d_{ik} = \min(|i - k|,\ 40 - |i - k|)$ is the distance on the ring,
# and moves each particle by the perturbed-observation update
# [@burgers1998; @houtekamer1998]
#
# $$
# x_j \mapsto x_j + K (y_t - H x_j - e_j), \qquad
# K = C H^\top (H C H^\top + R)^{-1}, \quad e_j \sim \mathcal N(0, R),
# $$
#
# with a fresh draw $e_j$ for every particle. The plain filter is the same
# update with $C = \hat C$. The score is the root-mean-square error of the
# analysis mean $\bar x_t$,
# $\big(\tfrac{1}{40} \lVert \bar x_t - x_t^\dagger \rVert^2\big)^{1/2}$,
# averaged over the last 250 times, $t = 51, \dots, 300$ (rows 50 to 299 of
# the filter's output).

# %%
import math

import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, maps, toy
from enskit.algorithms import MultiplicativeInflation, enkf
from enskit.distribution import Gaussian
from enskit.linalg import DensePSD

problem = toy.lorenz96(state_dim=40, n_times=300, obs_every=2, noise_std=1.0)
d = jnp.abs(jnp.arange(40)[:, None] - jnp.arange(40)[None, :])
d = jnp.minimum(d, 40 - d)
B = DensePSD(0.3 * jnp.exp(-0.5 * d**2) + 1e-6 * jnp.eye(40))
print(problem)

# %% [markdown]
# ## The hybrid approximation
#
# The filter conditions a joint Gaussian of the state and the predicted data,
# built by a function passed as `approximation=`. The default,
# `kalman.gaussian_approximation`, moment-matches the particles. The function
# below builds a `Gaussian` over `"x"` with the ensemble mean, the factor
# $\sqrt{\alpha}\, X$, where $X$ holds the anomalies scaled so that
# $\hat C = X X^\top$, and the independent term $(1 - \alpha) B$, so that its
# covariance is $C$. Pushing it through the linear map $H$ absorbs $B$ into
# the shared factor, so the predicted data are correlated with it:
# $C_{xy} = C H^\top$ and, once the noise is added, $C_{yy} = H C H^\top + R$,
# exactly, although $B$ is not represented by any particles. For a nonlinear
# observation model these blocks would have to be estimated from particles,
# and $B$ could not enter.

# %%
def hybrid_approximation(ensemble, noise, alpha=0.2):
    """The joint Gaussian, with alpha * sample cov + (1 - alpha) * B on "x"."""
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
# Both runs use the same particles, keys, `Matheron` update and inflation.
# They differ only in the `approximation` argument. The hybrid is a plain
# `Gaussian`, not one aligned with the particles, so `Matheron` takes its
# general path on it; `kalman.SymmetricSquareRoot` needs an aligned
# approximation and refuses this one.

# %%
def run(**options):
    ensemble = problem.initial.sample(jax.random.key(0), n_particles=10)
    result = enkf.filter(
        ensemble, problem.observations,
        transition=problem.transition, observe=problem.observe,
        noise_cov=problem.noise_cov, update_rule=kalman.Matheron(),
        inflation=MultiplicativeInflation(1.05), key=jax.random.key(1), **options,
    )
    rmse = jnp.sqrt(jnp.mean((result.means["x"] - problem.truth) ** 2, axis=1))
    return float(rmse[50:].mean())


rmse_plain = run()
rmse_hybrid = run(approximation=hybrid_approximation)
print(f"J = 10   plain EnKF RMSE {rmse_plain:.1f}   hybrid EnKF RMSE {rmse_hybrid:.1f}")

# %% [markdown]
# ## What to notice
#
# With ten particles for 40 coordinates the plain filter has an average error
# of about 4.9, far above the observation noise standard deviation of 1: it
# does not track the truth. The hybrid filter reaches about 0.9, under half
# the plain filter's error. Localization, in Example 12, does better on this
# system with the same ten particles. The two methods are not exclusive,
# since localization acts on the sample part of the covariance and a static
# part can be blended in alongside it.
#
# The printed numbers are those of the machine that built this page. A
# chaotic system amplifies rounding: the spin-up that produces the true
# trajectory turns a difference in the last digit of one arithmetic operation
# into a different trajectory, so another platform filters a different truth
# at the same seed and prints different digits. The checks below test bands.

# %%
# checks
assert 2.0 < rmse_plain < 6.0                   # the plain filter loses the truth
assert rmse_hybrid < 1.5 and rmse_hybrid < 0.5 * rmse_plain
