# %% [markdown]
# # Writing an update rule
#
# Implement the deterministic ensemble Kalman filter (DEnKF) as a new update
# rule, and use it unchanged in a single update, a filter and an EKI run.
#
# ## When to use this
#
# Write an update rule when the drivers fit the problem but the way particles
# are moved does not. A rule is a class with one method, `build`, so it can be
# passed wherever `kalman.Matheron` or `kalman.SymmetricSquareRoot` is
# accepted. The DEnKF [@sakov2008] is a short example: like ensemble
# square-root filters [@tippett2003] it uses no perturbed data, and it
# replaces their matrix square root with a half-gain correction of the
# anomalies.
#
# ## Setup
#
# Conditioning a jointly Gaussian pair $(u, g)$ on $g + e = y^*$, with
# $e \sim \mathcal N(0, R)$, is represented by the map
#
# $$
# (u, g, e) \mapsto u + K\,(y^* - g - e), \qquad K = C_{ug}\,(C_{gg} + R)^{-1},
# $$
#
# where $C_{ug}$ and $C_{gg}$ are covariances of the ensemble's Gaussian
# approximation. The DEnKF applies this map with $e = 0$ to particles
# whose $g$ block has been pulled halfway to its mean, $g_j \to \bar g +
# \tfrac12 (g_j - \bar g)$, so particle $j$ moves to
#
# $$
# u_j' = u_j + K\big(y^* - \bar g - \tfrac12 (g_j - \bar g)\big).
# $$
#
# The mean receives the full Kalman correction, $\bar u' = \bar u + K(y^* -
# \bar g)$, and the anomalies half of it, $u_j' - \bar u' = (u_j - \bar u) -
# \tfrac12 K (g_j - \bar g)$. For a linear model $g = A u$ with prior
# covariance $P$ the resulting covariance is
#
# $$
# \big(I - \tfrac12 K A\big) P \big(I - \tfrac12 K A\big)^\top
# = (I - K A) P + \tfrac14 K A P A^\top K^\top,
# $$
#
# the Kalman posterior covariance plus a positive semidefinite term.

# %%
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, maps, toy
from enskit.distribution import exact_moment_ensemble
from enskit.algorithms import MultiplicativeInflation, eki, enkf

# %% [markdown]
# ## The rule
#
# `build` receives the particles, their Gaussian approximation and the names
# of the given blocks, but not their values. It builds the conditional map
# and the halfway particles once, and returns a function that takes the
# values $y^*$ at call time. Calling the map without a key adds no noise.

# %%
class DEnKF:
    """Update rule: full gain on the mean, half gain on the anomalies.

    Particle j moves by K (y* - g_bar - (g_j - g_bar) / 2): the approximation's
    Matheron map, applied with no noise to particles whose given blocks are
    pulled halfway to their mean.
    """

    def build(self, particles, approximation, given):
        cmap = approximation.conditional_map(given)            # pointwise, value-free
        halfway = particles.assign(
            {c: particles.mean(c) + 0.5 * particles.anomalies(c) for c in given})

        def update(values=None, /, *, key=None, **block_values):  # the value arrives here
            return cmap(halfway, values, **block_values).marginal(*cmap.targets)

        return update

# %% [markdown]
# ## One update
#
# The test problem is linear: $u \in \mathbb R^4$ with prior $\mathcal N(0,
# C)$, $g = A u \in \mathbb R^6$ with fixed random $A$, and $R = 0.3^2 I_6$.
# The 32 particles have sample mean and covariance exactly equal to the
# prior's, so `kalman.SymmetricSquareRoot` reproduces the Kalman posterior
# and serves as the reference.

# %%
# It plugs into the one-call update, and therefore into every driver.
problem = toy.linear_gaussian(parameter_dim=4, data_dim=6)
ens = exact_moment_ensemble(jax.random.key(0), problem.prior, n_particles=32)
ens = maps.pushforward(ens, problem.forward, inputs="u", output="g")
post = kalman.update(ens, g=problem.y, noise={"g": problem.noise_cov}, update_rule=DEnKF())
exact = kalman.update(ens, g=problem.y, noise={"g": problem.noise_cov},
                      update_rule=kalman.SymmetricSquareRoot())

# %% [markdown]
# ## A filter and an EKI run
#
# The same rule drives `enkf.filter` on a 40-dimensional Lorenz-96 state
# [@lorenz1996] observed at its 20 even-numbered sites with unit noise
# variance, using 40 particles and anomaly inflation 1.02 [@anderson1999]. It
# also drives `eki.run` on a two-parameter exponential decay
# $G(u)_i = a\, e^{-r t_i}$, $u = (a, r)$, at ten times $t_i$ in $[0.1, 2]$
# with $R = 0.05^2 I_{10}$, using 64 particles and an adaptive schedule to
# $\beta = 1$; a step from $\beta$ to $\beta + \delta$ conditions with noise
# covariance $R / \delta$.

# %%
l96 = toy.lorenz96(dim=40, n_steps=300)
res = enkf.filter(jax.random.key(1), l96.initial.sample(jax.random.key(2), 40),
                  l96.observations, transition=l96.transition, observe=l96.observe,
                  noise_cov=l96.noise_cov, update_rule=DEnKF(),
                  inflation=MultiplicativeInflation(1.02))
rmse = jnp.sqrt(jnp.mean((res.means["x"] - l96.truth) ** 2, axis=1))[50:].mean()

decay = toy.exponential_decay()
fit = eki.run(eki.EKIState.from_prior(jax.random.key(3), decay.prior, 64), decay.forward,
              decay.y, decay.noise_cov, update_rule=DEnKF(), schedule=eki.AdaptiveESSSchedule())
print(f"DEnKF: L96 RMSE {rmse:.3f};  EKI mean {fit.ensemble.mean('u')}")

# %% [markdown]
# ## What to notice
#
# The filter's root-mean-square error, averaged over times 50 to 299, is
# 0.294, below the observation noise standard deviation of 1. The EKI run
# ends with mean $(1.490, 1.164)$, near the parameters $(1.5, 1.2)$ that
# generated the data. The checks confirm the linear case: the DEnKF mean
# equals the Kalman mean, and its variances are at least the Kalman ones.

# %%
# checks: the mean update is the exact Kalman mean; the spread is larger
assert jnp.allclose(post.mean("u"), exact.mean("u"), atol=1e-12)
assert jnp.all(jnp.diag(post.cov("u").to_dense()) >= jnp.diag(exact.cov("u").to_dense()))
assert rmse < 0.6
