# %% [markdown]
# # One update, one call
#
# Apply a single ensemble Kalman update to a linear-Gaussian problem with
# `kalman.update`, and compare the result with the exact posterior.
#
# ## When to use this
#
# `kalman.update` performs one analysis on an ensemble that already carries
# both the quantity to be updated and its prediction of the data. It is the
# right tool when you control the surrounding loop yourself, or when there is
# only one batch of data to assimilate. The update rule is an argument. The
# symmetric square-root rule is deterministic [@bishop2001; @tippett2003],
# and Matheron's rule perturbs each particle's predicted data with a fresh
# noise draw [@burgers1998].
#
# ## Setup
#
# The parameter $u \in \mathbb R^4$ has prior $\mathcal N(0, C)$ with
# $C = B B^\top / 4 + \tfrac12 I_4$, where $B$ is a fixed $4 \times 4$ matrix
# of standard normal entries. The data are
#
# $$
# y = A u^\dagger + \varepsilon, \qquad
# \varepsilon \sim \mathcal N(0, R), \quad R = 0.3^2 I_6,
# $$
#
# with $A \in \mathbb R^{6 \times 4}$ a fixed matrix of entries drawn from
# $\mathcal N(0, 1/4)$ and $u^\dagger$ a draw from the prior. The posterior is
# Gaussian, with
#
# $$
# m_{\mathrm{post}} = K y, \qquad
# C_{\mathrm{post}} = C - K A C, \qquad
# K = C A^\top (A C A^\top + R)^{-1} .
# $$

# %%
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, maps, toy
from enskit.distribution import exact_moment_ensemble

problem = toy.linear_gaussian(parameter_dim=4, data_dim=6)

# %% [markdown]
# ## The ensemble
#
# `exact_moment_ensemble` draws $J = 32$ particles $u_j$ whose sample mean is
# exactly $0$ and whose sample covariance, normalized by $J - 1$, is exactly
# $C$ [@pham2001]. Pushing them through the forward model adds the block
# $g_j = A u_j$.

# %%
ens = exact_moment_ensemble(jax.random.key(0), problem.prior, n_particles=32)
ens = maps.pushforward(ens, problem.forward, inputs="u", output="g")   # blocks: u, g

# %% [markdown]
# ## Two updates
#
# Each call conditions on $g = y$, with $g$ observed through additive noise
# of covariance $R$. Both rules use the ensemble's sample moments in place of
# $C$ in the formulas above. The square-root rule moves the ensemble mean by
# $K(y - \bar g)$ and transforms the anomalies so that their sample covariance
# equals $C - KAC$. Matheron's rule moves each particle by
# $K(y - g_j - e_j)$ with $e_j \sim \mathcal N(0, R)$, so it needs a key.

# %%
# Condition on g, observed through additive noise with covariance noise_cov.
sqrt_post = kalman.update(ens, g=problem.y, noise={"g": problem.noise_cov},
                          update_rule=kalman.SymmetricSquareRoot())
pert_post = kalman.update(ens, g=problem.y, noise={"g": problem.noise_cov},
                          update_rule=kalman.Matheron(), key=jax.random.key(1))
print(sqrt_post)          # Ensemble(n_particles=32, blocks={'u': 4}, weighted=False)

# %% [markdown]
# ## The exact answer
#
# The same operations applied to the prior itself, rather than to an
# ensemble, give the exact Gaussian posterior.

# %%
# The exact answer, from the same building blocks applied to the prior itself.
exact = (problem.prior
         .pipe(maps.pushforward, problem.forward, inputs="u", output="g")
         .add_noise(g=problem.noise_cov)
         .condition(g=problem.y))

# %% [markdown]
# ## What to notice
#
# The update returns an ensemble over `u` alone; the conditioned block `g` is
# dropped. Because the particles reproduce the prior's mean and covariance
# exactly and the model is linear, the square-root update matches the exact
# posterior mean and covariance to $10^{-12}$. Matheron's update matches the
# exact posterior only in expectation over the noise draws, so its mean is
# checked to a tolerance of 0.3.

# %%
# checks: the square-root update of an exact-moment ensemble is exact
assert jnp.allclose(sqrt_post.mean("u"), exact.mean("u"), atol=1e-12)
assert jnp.allclose(sqrt_post.cov("u").to_dense(), exact.cov("u").to_dense(), atol=1e-12)
assert jnp.allclose(pert_post.mean("u"), exact.mean("u"), atol=0.3)
print("square-root update matches the exact posterior")
