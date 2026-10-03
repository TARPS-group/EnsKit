"""Example 12 - a localized ETKF with fewer particles than state dimensions."""
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, toy
from enskit.algorithms import MultiplicativeInflation, enkf

problem = toy.lorenz96(dim=40, n_steps=300, obs_every=2, noise_sd=1.0)


def periodic(a, b):                       # distance on the ring of 40 sites
    d = jnp.abs(b[:, 0] - a[0])
    return jnp.minimum(d, 40.0 - d)


localization = kalman.DomainLocalization(
    target_coords={"x": problem.coords},  # (40, 1): where each state coordinate lives
    given_coords=problem.obs_coords,      # (20, 1): where each predicted datum lives
    radius=8.0, max_neighbors=10, distance=periodic,
)


def run(update_rule):
    ens = problem.initial.sample(jax.random.key(0), n_particles=10)
    result = enkf.filter(jax.random.key(1), ens, problem.observations,
                         transition=problem.transition, observe=problem.observe,
                         noise_cov=problem.noise_cov, update_rule=update_rule,
                         inflation=MultiplicativeInflation(1.05))
    return jnp.sqrt(jnp.mean((result.means["x"] - problem.truth) ** 2, axis=1))[50:].mean()


rmse_global = run(kalman.SymmetricSquareRoot())
rmse_local = run(kalman.LocalizedUpdateRule(kalman.SymmetricSquareRoot(), localization))
rmse_local_pw = run(kalman.LocalizedUpdateRule(kalman.Matheron(), localization))
print(f"J = 10   global ETKF {rmse_global:.3f}   local ETKF {rmse_local:.3f}   "
      f"local stochastic EnKF {rmse_local_pw:.3f}")

# ---- check
assert rmse_local < 0.5 and rmse_local < 0.5 * rmse_global
