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
# approximation [@wilson2021]. The DEnKF applies this map with $e = 0$ to
# particles whose $g$ block has been pulled halfway to its mean,
# $g_j \mapsto \bar g + \tfrac12 (g_j - \bar g)$, so particle $j$ moves to
#
# $$
# u_j' = u_j + K\big(y^* - \bar g - \tfrac12 (g_j - \bar g)\big).
# $$
#
# The mean receives the full Kalman correction, $\bar u' = \bar u + K(y^* -
# \bar g)$, and the anomalies half of it, $u_j' - \bar u' = (u_j - \bar u) -
# \tfrac12 K (g_j - \bar g)$. For a linear model $g = A u$ whose particles
# have sample covariance $P$ the anomalies are $(I - \tfrac12 K A)(u_j - \bar
# u)$, so the updated sample covariance is
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
from enskit.algorithms import MultiplicativeInflation, eki, enkf
from enskit.distribution import exact_moment_ensemble

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
# ## Checking the rule
#
# `enskit.testing.check_update_rule` runs a rule through the obligations every
# update rule must meet, on a linear-Gaussian problem of its own: the blocks,
# shapes and dtype it returns and the errors it raises; that it is
# reproducible from its key; that one build serves any value; that its mean is
# the exact conditional mean and, unless told otherwise, its covariance the
# exact conditional covariance; and that it agrees under `jax.jit` and
# `jax.vmap`. The DEnKF's covariance is larger by design, so the covariance
# check is turned off. Run it before trusting a rule of your own.

# %%
from enskit.testing import check_update_rule

check_update_rule(DEnKF(), exact_covariance=False)      # raises if an obligation fails

# %% [markdown]
# ## One update
#
# The test problem is linear: $u \in \mathbb R^4$ with prior
# $\mathcal N(0, I_4)$, $g = A u \in \mathbb R^6$ with the entries of $A$
# drawn once from $\mathcal N(0, 1/4)$ and then fixed, and
# $R = 0.1^2 I_6$. The 32 particles have sample mean and covariance exactly
# equal to the prior's, so $P = I_4$, and `kalman.SymmetricSquareRoot`
# reproduces the Kalman posterior and serves as the reference.

# %%
# It plugs into the one-call update, and therefore into every driver.
problem = toy.linear_gaussian(parameter_dim=4, data_dim=6)
print(problem)
ens = exact_moment_ensemble(jax.random.key(0), problem.prior, n_particles=32)
ens = maps.pushforward(ens, problem.forward, inputs="u", output="g")
post = kalman.update(ens, g=problem.y, noise={"g": problem.noise_cov},
                     update_rule=DEnKF())
exact = kalman.update(ens, g=problem.y, noise={"g": problem.noise_cov},
                      update_rule=kalman.SymmetricSquareRoot())

# the closed forms of the Setup, with P = I
A, R = problem.G.to_dense(), problem.noise_cov.to_dense()
K = A.T @ jnp.linalg.inv(A @ A.T + R)
P_kalman = jnp.eye(4) - K @ A
P_denkf = P_kalman + 0.25 * K @ A @ A.T @ K.T

# %% [markdown]
# ## A filter and an EKI run
#
# The same rule drives `enkf.filter` on the 40-dimensional Lorenz-96 state of
# Example 2 [@lorenz1996]: one fourth-order Runge-Kutta step of length 0.05
# per time, forcing 8, $T = 300$ times, the 20 even-numbered sites observed
# with noise covariance $I_{20}$, and 40 initial particles drawn from
# $\mathcal N(x_0^\dagger, I_{40})$ around the true initial state. The
# anomalies are inflated by 1.02 before each analysis [@anderson1999], less
# than Example 2's 1.05, because the DEnKF's halved anomaly update already
# leaves the ensemble wider than a square-root update would. The rule draws
# nothing, so the filter needs no key.
#
# It also drives `eki.run` on a two-parameter exponential decay
# $G(u)_i = u_0\, e^{-u_1 t_i}$ at the twelve times $t_i = i/4$,
# $i = 1, \dots, 12$, with prior $\mathcal N\big((1, 1), I_2\big)$, data
# generated from $u = (2, 1.5)$ with noise covariance $R = 0.02^2 I_{12}$, 64
# particles, and an adaptive schedule to $\beta = 1$
# [@iglesias2013; @jasra2011]. A step from $\beta$ to $\beta + \delta$
# conditions on the data with noise covariance $R / \delta$.

# %%
l96 = toy.lorenz96(state_dim=40, n_times=300, obs_every=2, noise_std=1.0)
res = enkf.filter(l96.initial.sample(jax.random.key(0), n_particles=40),
                  l96.observations, transition=l96.transition, observe=l96.observe,
                  noise_cov=l96.noise_cov, update_rule=DEnKF(),
                  inflation=MultiplicativeInflation(1.02))
rmse = float(jnp.sqrt(jnp.mean((res.means["x"] - l96.truth) ** 2, axis=1))[50:].mean())

decay = toy.exponential_decay()
fit = eki.run(eki.EKIState.from_prior(jax.random.key(3), decay.prior, n_particles=64),
              decay.forward, decay.y, decay.noise_cov,
              update_rule=DEnKF(), schedule=eki.AdaptiveESSSchedule())
a, r = fit.mean("u")
print(f"DEnKF: Lorenz-96 RMSE {rmse:.2f};  EKI mean ({a:.2f}, {r:.2f})")

# %% [markdown]
# ## What to notice
#
# The filter's root-mean-square error, averaged over times $t = 51, \dots,
# 300$, is about 0.3, well below the observation noise standard deviation of
# 1. The EKI run ends with a mean within 0.1 of the parameters $(2, 1.5)$ that
# generated the data. The checks confirm the linear case: the DEnKF mean
# equals the Kalman mean, its covariance is the closed form of the Setup, and
# so its variances are at least the Kalman ones.
#
# The Lorenz-96 number is that of the machine that built this page. A
# chaotic system amplifies rounding: the spin-up that produces the true
# trajectory turns a difference in the last digit of one arithmetic operation
# into a different trajectory, so another platform filters a different truth
# at the same seed and prints different digits. The check below tests a band.

# %%
# checks: the mean update is the exact Kalman mean; the spread is larger
assert jnp.allclose(exact.mean("u"), problem.posterior().mean("u"), atol=1e-12)
assert jnp.allclose(exact.cov("u").to_dense(), P_kalman, atol=1e-12)
assert jnp.allclose(post.mean("u"), exact.mean("u"), atol=1e-12)
assert jnp.allclose(post.cov("u").to_dense(), P_denkf, atol=1e-12)
assert jnp.all(jnp.diag(post.cov("u").to_dense()) >= jnp.diag(exact.cov("u").to_dense()))
assert rmse < 0.6
assert jnp.all(jnp.abs(fit.mean("u") - decay.u_true) < 0.1)
