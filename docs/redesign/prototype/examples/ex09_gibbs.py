"""Example 9 - Gibbs-style alternation with code outside the package.

Unknown noise covariance with an inverse-Wishart prior. The package supplies
"an ensemble update conditional on a fixed noise covariance"; the conjugate
noise-covariance draw is ordinary JAX written by the user.
"""
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, maps, toy
from enskit.linalg import DensePSD

problem = toy.linear_gaussian(parameter_dim=2, data_dim=3)
G = problem.forward
R_true = jnp.array([[0.5, 0.2, 0.0], [0.2, 0.4, 0.1], [0.0, 0.1, 0.3]])
n_rep = 40                                             # replicated observations
key = jax.random.key(0)
key, sub = jax.random.split(key)
ys = G(problem.u_true[None])[0] + jax.random.multivariate_normal(
    sub, jnp.zeros(3), R_true, (n_rep,))
y_bar = ys.mean(axis=0)
nu0, Psi0 = 6.0, 0.4 * 2.0 * jnp.eye(3)                # prior mean of R is 0.4 I


def draw_noise_cov(key, u):
    """R | u, ys ~ IW(nu0 + n_rep, Psi0 + sum_r (y_r - G u)(y_r - G u)^T). User code."""
    resid = ys - G(u[None])[0]
    nu, Psi = nu0 + n_rep, Psi0 + resid.T @ resid
    L = jnp.linalg.cholesky(jnp.linalg.inv(Psi))
    Z = jax.random.normal(key, (int(nu), 3)) @ L.T     # rows ~ N(0, Psi^{-1})
    return jnp.linalg.inv(Z.T @ Z)                     # inverse of a Wishart draw


R = 0.4 * jnp.eye(3)
draws_R, draws_u = [], []
for sweep in range(400):
    key, k_ens, k_upd, k_pick, k_R = jax.random.split(key, 5)
    # u | R, ys: the replicates enter as their mean, with noise R / n_rep
    ens = problem.prior.sample(k_ens, n_particles=128)
    ens = maps.pushforward(ens, G, inputs="u", output="g")
    post = kalman.update(ens, g=y_bar, noise={"g": DensePSD(R / n_rep)},
                         update_rule=kalman.Matheron(), key=k_upd)
    u = post["u"][jax.random.randint(k_pick, (), 0, post.n_particles)]
    # R | u, ys: outside the package
    R = draw_noise_cov(k_R, u)
    draws_R.append(R)
    draws_u.append(u)

R_mean = jnp.mean(jnp.stack(draws_R[50:]), axis=0)
print("posterior mean of R:\n", R_mean, "\ntruth:\n", R_true)
print("posterior mean of u:", jnp.mean(jnp.stack(draws_u[50:]), axis=0), "truth", problem.u_true)

# ---- check
assert jnp.allclose(R_mean, R_true, atol=0.2)
