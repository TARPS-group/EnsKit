"""Example 5 - the two updates as compositions of probabilistic operations."""
import jax
import jax.numpy as jnp

import enskit
from enskit import maps, toy

problem = toy.linear_gaussian(parameter_dim=4, data_dim=6)

# (a) Exact Gaussian algebra. Matheron's rule: joint samples, transported by the
#     conditional map, are exact samples of the conditional.
joint = (problem.prior
         .pipe(maps.pushforward, problem.forward, inputs="u", output="g")
         .pipe(maps.pushforward, maps.AdditiveNoise(problem.noise_cov), inputs="g", output="y"))
posterior = joint.condition(y=problem.y)                    # Gaussian over u, g
transport = joint.conditional_map("y")                           # x -> x + K (y* - y)
samples = transport(joint.sample(jax.random.key(0), 200_000), y=problem.y)

# (b) The same two routes, applied to an ensemble's moment-matched Gaussian.
ens = problem.prior.sample(jax.random.key(1), n_particles=32)
ens = maps.pushforward(ens, problem.forward, inputs="u", output="g")
ens_joint = (ens.project()                                       # member-aligned
             .pipe(maps.pushforward, maps.AdditiveNoise(problem.noise_cov),
                   inputs="g", output="y"))

# square root: realize after condition, built once as a map and called with the value
sqrt_map = ens_joint.square_root_map("y")
sqrt_particles = sqrt_map(y=problem.y)
assert jnp.allclose(sqrt_particles["u"],
                    ens_joint.condition(y=problem.y).realize_particles()["u"])

# Matheron: transport after realize (the map draws the noise, whitened)
particles = ens_joint.realize_particles(exclude_block_covs=("y",))   # y_j = g_j here
pert_particles = ens_joint.conditional_map("y")(particles, y=problem.y,
                                                key=jax.random.key(2))

# ---- checks
assert jnp.allclose(samples.mean("u"), posterior.mean("u"), atol=1e-2)
assert jnp.allclose(samples.cov("u").to_dense(), posterior.cov("u").to_dense(), atol=1e-2)
fit = ens_joint.condition(y=problem.y)
assert jnp.allclose(sqrt_particles.mean("u"), fit.mean("u"), atol=1e-12)
assert jnp.allclose(sqrt_particles.cov("u").to_dense(), fit.cov("u").to_dense(), atol=1e-12)
assert pert_particles.names == ("u", "g")
print("Matheron samples match the exact conditional; realize . condition is exact on the fit")
