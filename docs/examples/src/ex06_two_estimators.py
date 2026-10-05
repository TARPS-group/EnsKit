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
# unit noise: $u \sim \mathcal N(0, C)$ in $\mathbb R^4$ with $C = I_4$,
# $g = G u$ in $\mathbb R^6$ with $G$ a fixed matrix of entries drawn from
# $\mathcal N(0, 1/4)$, and $y = g + e$ with $e \sim \mathcal N(0, R)$,
# $R = I_6$. The target is
#
# $$
# C_{yy} = S + R, \qquad S = G C G^\top .
# $$
#
# From $J = 20$ prior samples $u_j$, with $g_j = G u_j$ and noise draws $e_j$,
# the two estimators are
#
# $$
# \begin{aligned}
# \hat C^{\mathrm{s}} &= \frac{1}{J-1} \sum_{j=1}^{J}
#   (y_j - \bar y)(y_j - \bar y)^\top,
# \qquad y_j = g_j + e_j, \\
# \hat C^{\mathrm{f}} &= \frac{1}{J-1} \sum_{j=1}^{J}
#   (g_j - \bar g)(g_j - \bar g)^\top + R .
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
# introduces. Both errors have closed forms here. The sample covariance
# $\hat\Sigma$ of $J$ independent draws from $\mathcal N(0, \Sigma)$ has
# $\mathbb E \lVert \hat\Sigma - \Sigma \rVert_F^2
# = \big(\operatorname{tr}(\Sigma^2) + \operatorname{tr}(\Sigma)^2\big)/(J-1)$,
# and $\hat C^{\mathrm{s}}$ is that estimator for $\Sigma = C_{yy}$, while
# $\hat C^{\mathrm{f}} - C_{yy}$ is that error for $\Sigma = S$.

# %%
import jax
import jax.numpy as jnp

import enskit
from enskit import maps, toy

problem = toy.linear_gaussian(parameter_dim=4, data_dim=6, noise_std=1.0)
noise = maps.AdditiveNoise(problem.noise_cov)
exact = (problem.prior
         .pipe(maps.pushforward, maps.Linear(problem.G), inputs="u", output="g")
         .pipe(maps.pushforward, noise, inputs="g", output="y"))
S = exact.cov("g").to_dense()              # G C G^T
true_cyy = exact.cov("y").to_dense()       # G C G^T + R

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
# The errors are averaged over 500 independent ensembles, one per key, with
# `jax.vmap` running the function over all 500 keys at once. The closed forms
# above give the expected values, and the standard error of each average
# says how far from them the averages may fall.

# %%
n_ensembles, J = 500, 20
keys = jax.random.split(jax.random.key(0), n_ensembles)
errs = jnp.stack(jax.vmap(both_estimates)(keys), axis=1)       # (500, 2)
mean_err = errs.mean(axis=0)
std_err = errs.std(axis=0, ddof=1) / jnp.sqrt(n_ensembles)


def expected_error(sigma):
    return (jnp.trace(sigma @ sigma) + jnp.trace(sigma) ** 2) / (J - 1)


expected = jnp.array([expected_error(true_cyy), expected_error(S)])
names = ("sample, then fit", "fit, then add R")
for name, m, se, e in zip(names, mean_err, std_err, expected, strict=True):
    print(f"{name:17s} mean squared error {float(m):5.2f} +- {float(se):.2f}, "
          f"closed form {float(e):5.2f}")
print(f"share of the sampled estimator's error from sampling the noise: "
      f"{float(1 - expected[1] / expected[0]):.2f}")

# %% [markdown]
# ## What to notice
#
# The mean squared error is 10.40 when the noise is sampled and 3.27 when
# $R$ is added exactly, against closed forms of 10.30 and 3.25, each within
# one standard error. The difference of the closed forms, 7.05, is
# $\mathbb E \lVert \Delta \rVert_F^2$, so with 20 particles and unit noise
# more than two thirds of the first estimator's error comes from sampling the
# noise.

# %%
# checks
assert [f"{float(m):.2f}" for m in mean_err] == ["10.40", "3.27"]
assert [f"{float(e):.2f}" for e in expected] == ["10.30", "3.25"]
assert jnp.all(jnp.abs(mean_err - expected) < std_err)
assert f"{float(expected[0] - expected[1]):.2f}" == "7.05"
assert expected[0] - expected[1] > 2 / 3 * expected[0]
