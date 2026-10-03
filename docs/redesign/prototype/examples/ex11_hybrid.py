"""Example 11 - a hybrid EnKF: exploiting a linear map and a static covariance."""
import math

import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, maps, toy
from enskit.distribution import Gaussian
from enskit.algorithms import MultiplicativeInflation, enkf
from enskit.linalg import DensePSD

problem = toy.lorenz96(dim=40, n_steps=300, obs_every=2, noise_sd=1.0)
d = jnp.abs(jnp.arange(40)[:, None] - jnp.arange(40)[None, :])
d = jnp.minimum(d, 40 - d)
B = DensePSD(0.3 * jnp.exp(-0.5 * d**2) + 1e-6 * jnp.eye(40))   # a static covariance


def hybrid_approximation(ensemble, noise, alpha=0.2):
    """The joint Gaussian approximation, with alpha * sample cov + (1 - alpha) * B on "x"."""
    ((given, R),) = noise.items()             # the block conditioned on, and its noise
    fit = ensemble.marginal("x").project()
    hybrid = Gaussian({"x": fit.mean("x")},
                           factors={"x": fit.factor("x") * math.sqrt(alpha)},
                           block_covs={"x": B * (1.0 - alpha)})
    # Linear pushforward is exact: C_xy = C H^T and C_yy = H C H^T + R, with B
    # carried as a structured factor, never as a sampled estimate.
    return (hybrid
            .pipe(maps.pushforward, problem.observe, inputs="x", output=given)
            .add_noise({given: R}))


def run(**options):
    ens = problem.initial.sample(jax.random.key(0), n_particles=10)
    result = enkf.filter(jax.random.key(1), ens, problem.observations,
                         transition=problem.transition, observe=problem.observe,
                         noise_cov=problem.noise_cov, update_rule=kalman.Matheron(),
                         inflation=MultiplicativeInflation(1.05), **options)
    return jnp.sqrt(jnp.mean((result.means["x"] - problem.truth) ** 2, axis=1))[50:].mean()


rmse_plain = run()                          # the default: kalman.gaussian_approximation
rmse_hybrid = run(approximation=hybrid_approximation)   # same filter, different joint
print(f"J = 10   plain EnKF RMSE {rmse_plain:.3f}   hybrid EnKF RMSE {rmse_hybrid:.3f}")

# ---- check
assert rmse_hybrid < 0.5 * rmse_plain
