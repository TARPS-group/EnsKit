# %% [markdown]
# # Gibbs sampling with an unknown noise covariance
#
# Estimate a parameter and an unknown $3 \times 3$ noise covariance by
# alternating an ensemble Kalman update for the parameter with a conjugate
# draw of the covariance written in plain JAX.
#
# ## When to use this
#
# Some models have a block for which an ensemble Kalman update is a good
# conditional sampler and another block with a closed-form full conditional.
# An unknown noise covariance with an inverse-Wishart prior is a common
# example, since its full conditional is again inverse-Wishart
# [@gelman2013]. Alternating the two in a Gibbs sweep needs no support from
# the package beyond one `kalman.update` call; the other step is ordinary user
# code. Component-wise iterative EKI for static models with an unknown
# measurement error covariance has the same alternating structure
# [@botha2023].
#
# ## Setup
#
# A parameter $u \in \mathbb R^2$ is observed through a linear map
# $G \in \mathbb R^{3 \times 2}$ in $n = 40$ replicates with a common unknown
# noise covariance $R$:
#
# $$
# y_r = G u + e_r, \quad e_r \sim \mathcal N(0, R), \quad r = 1, \dots, n,
# \qquad
# u \sim \mathcal N(0, C_0), \qquad
# R \sim \mathcal{IW}(\nu_0, \Psi_0),
# $$
#
# with $\nu_0 = 6$ and $\Psi_0 = 0.8\, I_3$, so that the prior mean of $R$ is
# $\Psi_0 / (\nu_0 - 3 - 1) = 0.4\, I_3$. $G$, $C_0$ and the true parameter
# $u^\dagger$ come from `toy.linear_gaussian(parameter_dim=2, data_dim=3)`:
# $C_0 = I_2$, $G$ has independent $\mathcal N(0, 1/2)$ entries, and
# $u^\dagger$ is a draw from the prior. The toy's own observation and noise
# covariance are not used; the replicates are generated with
#
# $$
# R^\dagger = \begin{pmatrix}
#   0.5 & 0.2 & 0 \\ 0.2 & 0.4 & 0.1 \\ 0 & 0.1 & 0.3
# \end{pmatrix}.
# $$
#
# The two full conditionals are
#
# $$
# \begin{aligned}
# u \mid R, y_{1:n} &\sim \text{the posterior of } u \text{ given }
#   \bar y = G u + \bar e, \ \ \bar e \sim \mathcal N(0, R/n), \\
# R \mid u, y_{1:n} &\sim \mathcal{IW}\Big(\nu_0 + n,\ \Psi_0 + \sum_{r=1}^{n}
#   (y_r - G u)(y_r - G u)^\top\Big),
# \end{aligned}
# $$
#
# where $\bar y$ is the mean of the replicates.

# %%
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, maps, toy
from enskit.linalg import DensePSD

problem = toy.linear_gaussian(parameter_dim=2, data_dim=3)
G = problem.forward
R_true = jnp.array([[0.5, 0.2, 0.0], [0.2, 0.4, 0.1], [0.0, 0.1, 0.3]])
n_rep = 40
key = jax.random.key(0)
key, sub = jax.random.split(key)
ys = G(problem.u_true[None])[0] + jax.random.multivariate_normal(
    sub, jnp.zeros(3), R_true, (n_rep,))
y_bar = ys.mean(axis=0)
nu0, Psi0 = 6.0, 0.4 * 2.0 * jnp.eye(3)

# %% [markdown]
# ## The covariance step
#
# The draw of $R$ uses the fact that if the rows of $Z \in \mathbb R^{\nu \times 3}$
# are independent $\mathcal N(0, \Psi^{-1})$, then $Z^\top Z$ is Wishart with
# scale $\Psi^{-1}$ and $(Z^\top Z)^{-1} \sim \mathcal{IW}(\nu, \Psi)$. This is
# user code and touches nothing in the package.

# %%
def draw_noise_cov(key, u):
    """R | u, ys ~ IW(nu0 + n_rep, Psi0 + sum_r (y_r - G u)(y_r - G u)^T)."""
    resid = ys - G(u[None])[0]
    nu, Psi = nu0 + n_rep, Psi0 + resid.T @ resid
    L = jnp.linalg.cholesky(jnp.linalg.inv(Psi))
    Z = jax.random.normal(key, (int(nu), 3)) @ L.T
    return jnp.linalg.inv(Z.T @ Z)

# %% [markdown]
# ## The sweep
#
# The chain starts from $R = 0.4\, I_3$, the prior mean. Each sweep draws 128
# fresh particles from the prior of $u$, pushes them through $G$, and applies
# one stochastic (`Matheron`) update toward $\bar y$ with noise covariance
# $R/n$ [@burgers1998; @houtekamer1998]. One particle, picked at random, is taken as the
# draw of $u$, and $R$ is then redrawn given that $u$. A particle of a
# stochastic update with a finite ensemble is only an approximate draw from
# $u \mid R, y_{1:n}$, so this is an approximate Gibbs sampler. Because the
# model is linear and Gaussian, an exact draw is also available by conditioning
# the Gaussian prior pushed through $G$ and sampling the result, as in
# Example 3.

# %%
R = 0.4 * jnp.eye(3)
cov_draws, u_draws = [], []
for _ in range(400):
    key, k_ens, k_upd, k_pick, k_cov = jax.random.split(key, 5)
    ens = problem.prior.sample(k_ens, n_particles=128)
    ens = maps.pushforward(ens, G, inputs="u", output="g")
    post = kalman.update(ens, g=y_bar, noise={"g": DensePSD(R / n_rep)},
                         update_rule=kalman.Matheron(), key=k_upd)
    u = post["u"][jax.random.randint(k_pick, (), 0, post.n_particles)]
    R = draw_noise_cov(k_cov, u)
    cov_draws.append(R)
    u_draws.append(u)

R_mean = jnp.mean(jnp.stack(cov_draws[50:]), axis=0)
u_mean = jnp.mean(jnp.stack(u_draws[50:]), axis=0)
print("posterior mean of R:\n", jnp.round(R_mean, 3), "\ntruth:\n", R_true)
print(f"posterior mean of u: ({float(u_mean[0]):.3f}, {float(u_mean[1]):.3f}), "
      f"truth ({float(problem.u_true[0]):.3f}, {float(problem.u_true[1]):.3f})")

# %% [markdown]
# ## What to notice
#
# After discarding the first 50 of 400 sweeps, the estimated posterior mean of
# $R$ has diagonal $(0.428, 0.378, 0.228)$ against the true $(0.5, 0.4, 0.3)$,
# and it recovers the positive covariances $0.234$ and $0.079$ where the truth
# has $0.2$ and $0.1$. The prior mean $0.4\, I_3$ has no off-diagonal terms, so
# these come from the 40 replicates. The posterior mean of $u$ is
# $(-1.347, 1.486)$, near $u^\dagger = (-1.401, 1.432)$.

# %%
# checks
def close(x, stated):
    return jnp.allclose(jnp.round(x, 3), jnp.asarray(stated), rtol=0, atol=1e-12)


assert close(jnp.diag(R_mean), [0.428, 0.378, 0.228])
assert close(jnp.array([R_mean[0, 1], R_mean[1, 2]]), [0.234, 0.079])
assert close(u_mean, [-1.347, 1.486])
assert close(problem.u_true, [-1.401, 1.432])
assert jnp.allclose(R_mean, R_true, atol=0.1)
