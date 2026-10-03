# %% [markdown]
# # Two estimators of the predicted-data covariance
#
# Estimate the covariance of the noisy predicted data in two ways, by sampling
# the noise and by adding its known covariance, and compare their errors over
# many ensembles.
#
# ## When to use this
#
# An ensemble Kalman update needs the covariance of the predicted data
# including the noise. When the noise covariance is known, it can either be
# sampled, as in the perturbed-observation update, or added exactly to the
# sample covariance of the noise-free predictions. Sampling it adds error that
# a small ensemble does not average away [@whitaker2002]. This example
# measures that error, and shows how the order of the pushforward and the
# Gaussian fit selects the estimator.
#
# ## Setup
#
# The problem is the linear-Gaussian one of the previous examples with
# unit noise: $u \sim \mathcal N(0, C)$ in $\mathbb R^4$, $g = A u$ in
# $\mathbb R^6$, and $y = g + e$ with $e \sim \mathcal N(0, R)$, $R = I_6$.
# The target is
#
# $$
# C_{yy} = A C A^\top + R .
# $$
#
# From $J = 20$ prior samples $u_j$, with $g_j = A u_j$ and noise draws $e_j$,
# the two estimators are
#
# $$
# \begin{aligned}
# \hat C^{\mathrm{s}} &= \frac{1}{J-1} \sum_{j=1}^{J} (y_j - \bar y)(y_j - \bar y)^\top,
# \qquad y_j = g_j + e_j, \\
# \hat C^{\mathrm{f}} &= \frac{1}{J-1} \sum_{j=1}^{J} (g_j - \bar g)(g_j - \bar g)^\top + R .
# \end{aligned}
# $$
#
# Both are unbiased. Writing $\hat C_{ge}$ and $\hat C_{ee}$ for the sample
# cross-covariance of $g$ and $e$ and the sample covariance of $e$,
#
# $$
# \hat C^{\mathrm{s}} = \hat C^{\mathrm{f}} + \Delta, \qquad
# \Delta = \hat C_{ge} + \hat C_{ge}^\top + (\hat C_{ee} - R) .
# $$
#
# Given the $g_j$, $\Delta$ has mean zero, so it is uncorrelated with
# $\hat C^{\mathrm{f}} - C_{yy}$ and
#
# $$
# \mathbb E \lVert \hat C^{\mathrm{s}} - C_{yy} \rVert_F^2
# = \mathbb E \lVert \hat C^{\mathrm{f}} - C_{yy} \rVert_F^2
# + \mathbb E \lVert \Delta \rVert_F^2 .
# $$
#
# Adding $R$ exactly therefore lowers the mean squared error by
# $\mathbb E \lVert \Delta \rVert_F^2$, the error that sampling the noise
# introduces.

# %%
import jax
import jax.numpy as jnp

import enskit
from enskit import maps, toy

problem = toy.linear_gaussian(parameter_dim=4, data_dim=6, noise_sd=1.0)
A = problem.forward.op.to_dense()
true_cyy = A @ problem.prior.cov("u").to_dense() @ A.T + problem.noise_cov.to_dense()
noise = maps.AdditiveNoise(problem.noise_cov)

# %% [markdown]
# ## Two orders of operations
#
# Both estimators come from the same pushforward of $g$ through the noise.
# Applied to the ensemble, with a key, it draws $e_j$ for each particle, and
# `project` then fits $\hat C^{\mathrm{s}}$. Applied to the fitted Gaussian,
# it adds $R$ to the covariance of $g$ exactly and gives $\hat C^{\mathrm{f}}$.
# The function returns the squared Frobenius error of each.

# %%
def both_estimates(key):
    k1, k2 = jax.random.split(key)
    ens = problem.prior.sample(k1, n_particles=20)
    ens = maps.pushforward(ens, problem.forward, inputs="u", output="g")
    # sample the noise, then fit:   C_yy ~ sample cov of (g + e)
    sampled = maps.pushforward(ens, noise, inputs="g", output="y", key=k2).project()
    # fit, then add the noise:      C_yy ~ sample cov of g + R, exactly
    structured = maps.pushforward(ens.project(), noise, inputs="g", output="y")
    err = lambda g: jnp.sum((g.cov("y").to_dense() - true_cyy) ** 2)  # noqa: E731
    return err(sampled), err(structured)


# %% [markdown]
# ## The comparison
#
# The errors are averaged over 500 independent ensembles.

# %%
errs = jnp.array([both_estimates(k) for k in jax.random.split(jax.random.key(0), 500)])
print("mean squared Frobenius error of C_yy  (sample-then-fit, fit-then-add):",
      errs.mean(axis=0))

# %% [markdown]
# ## What to notice
#
# The mean squared error is 13.97 when the noise is sampled and 6.03 when
# $R$ is added exactly. The difference, about 7.9, estimates
# $\mathbb E \lVert \Delta \rVert_F^2$, so with 20 particles and unit noise
# more than half of the first estimator's error comes from sampling the
# noise.

# %%
# checks
assert errs[:, 1].mean() < errs[:, 0].mean()
