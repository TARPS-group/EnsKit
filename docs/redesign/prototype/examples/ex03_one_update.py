"""Example 3 - one ensemble Kalman update, one call."""
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, maps, toy
from enskit.distribution import exact_moment_ensemble

problem = toy.linear_gaussian(parameter_dim=4, data_dim=6)

ens = exact_moment_ensemble(jax.random.key(0), problem.prior, n_particles=32)
ens = maps.pushforward(ens, problem.forward, inputs="u", output="g")   # blocks: u, g

# Condition on g, observed through additive noise with covariance noise_cov.
sqrt_post = kalman.update(ens, g=problem.y, noise={"g": problem.noise_cov},
                          update_rule=kalman.SymmetricSquareRoot())
pert_post = kalman.update(ens, g=problem.y, noise={"g": problem.noise_cov},
                          update_rule=kalman.Matheron(), key=jax.random.key(1))
print(sqrt_post)          # Ensemble(n_particles=32, blocks={'u': 4}, weighted=False)

# The exact answer, from the same building blocks applied to the prior itself.
exact = (problem.prior
         .pipe(maps.pushforward, problem.forward, inputs="u", output="g")
         .add_noise(g=problem.noise_cov)
         .condition(g=problem.y))

# ---- checks: the square-root update of an exact-moment ensemble is exact
assert jnp.allclose(sqrt_post.mean("u"), exact.mean("u"), atol=1e-12)
assert jnp.allclose(sqrt_post.cov("u").to_dense(), exact.cov("u").to_dense(), atol=1e-12)
assert jnp.allclose(pert_post.mean("u"), exact.mean("u"), atol=0.3)
print("square-root update matches the exact posterior")
