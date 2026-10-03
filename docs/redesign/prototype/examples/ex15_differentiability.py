"""Example 15 - gradients through degenerate spectra, and through a whole run."""
import jax
import jax.numpy as jnp

import enskit
import numpy as np

from enskit import kalman, maps, toy
from enskit.distribution import Ensemble
from enskit.algorithms import eki
from enskit.linalg import PSDDiagonal

problem = toy.linear_gaussian(parameter_dim=4, data_dim=6)

# (a) Three predicted coordinates are identical across particles (a fixed boundary
#     value, say), so the whitened factor has exactly-zero singular values.
def forward_with_fixed(u):
    return jnp.concatenate([problem.forward(u), jnp.ones((u.shape[0], 3))], axis=1)

y = jnp.concatenate([problem.y, jnp.ones(3)])
ens = problem.prior.sample(jax.random.key(0), n_particles=30)
ens = maps.pushforward(ens, forward_with_fixed, inputs="u", output="g")


def posterior_spread(log_sd):
    noise = PSDDiagonal(jnp.full(9, 0.09)) * jnp.exp(2 * log_sd)
    post = kalman.update(ens, g=y, noise={"g": noise}, update_rule=kalman.SymmetricSquareRoot())
    return jnp.sum(post.cov("u").to_dense())


def naive_posterior_spread(log_sd):          # the same update via a plain SVD
    Fu = ens.project().factor("u").to_dense()
    S = (ens.project().factor("g").to_dense() / (0.3 * jnp.exp(log_sd))).T
    U, s, _ = jnp.linalg.svd(S, full_matrices=False)
    T = jnp.eye(S.shape[0]) + (U * (1 / jnp.sqrt(1 + s**2) - 1)) @ U.T
    return jnp.sum((Fu @ T) @ (Fu @ T).T)


eps = 1e-6
fd = (posterior_spread(eps) - posterior_spread(-eps)) / (2 * eps)
print("plain-SVD gradient:", jax.grad(naive_posterior_spread)(0.0))
print("enskit gradient:    ", jax.grad(posterior_spread)(0.0), "  finite difference:", fd)

# (b) A traceable forward model: differentiate a whole 4-step run end to end.
decay = toy.exponential_decay()
start = decay.prior.sample(jax.random.key(1), 32)


def final_misfit(log_noise_scale):
    noise_cov = decay.noise_cov * jnp.exp(2 * log_noise_scale)
    state = eki.EKIState(start, key=jax.random.key(2))
    for dbeta in (0.125, 0.125, 0.25, 0.5):
        state = eki.advance(state, decay.forward, decay.y, noise_cov, dbeta,
                            update_rule=kalman.Matheron())
    g = decay.forward(state.ensemble.mean("u")[None])[0]
    return jnp.sum((g - decay.y) ** 2)


g_ad = jax.grad(final_misfit)(0.0)
g_fd = (final_misfit(1e-5) - final_misfit(-1e-5)) / 2e-5
print("end-to-end gradient:", g_ad, " finite difference:", g_fd)

# (c) A host-side simulator. BlackBox lets it run under jit and vmap, and its
#     outputs are constants: only the noise model is differentiated.
def numpy_decay(u):                                  # plain NumPy, not traceable
    return u[:, :1] * np.exp(-u[:, 1:2] * np.linspace(0.1, 2.0, 10))


simulator = maps.BlackBox(numpy_decay, output_dim=10)


@jax.jit
def ensemble_evidence(log_sd, u):
    ens = maps.pushforward(Ensemble(u=u), simulator, inputs="u", output="g")
    noise = decay.noise_cov * jnp.exp(2 * log_sd)
    return kalman.gaussian_approximation(ens, {"g": noise}).log_density(g=decay.y)


u = start["u"]
print("d/d log_sd:", jax.grad(ensemble_evidence)(0.0, u))
print("d/du is zero through the simulator:", bool(jnp.all(jax.grad(ensemble_evidence, 1)(0.0, u) == 0)))
two = jax.vmap(ensemble_evidence, in_axes=(None, 0))(0.0, jnp.stack([u, 1.1 * u]))
print("vmapped over two ensembles:", two)

# ---- checks
assert jnp.isnan(jax.grad(naive_posterior_spread)(0.0))
assert jnp.allclose(naive_posterior_spread(0.0), posterior_spread(0.0))
assert jnp.allclose(jax.grad(posterior_spread)(0.0), fd, rtol=1e-6)
assert jnp.allclose(g_ad, g_fd, rtol=1e-4)
assert jnp.allclose(two[0], ensemble_evidence(0.0, u))
assert jnp.allclose(ensemble_evidence(0.0, u),
                    kalman.gaussian_approximation(maps.pushforward(Ensemble(u=u), decay.forward,
                                                  inputs="u", output="g"),
                                 {"g": decay.noise_cov}).log_density(g=decay.y))
