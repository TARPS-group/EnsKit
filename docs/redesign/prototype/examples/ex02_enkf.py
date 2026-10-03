"""Example 2 - EnKF on Lorenz-96, the high-level way."""
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, toy
from enskit.algorithms import MultiplicativeInflation, enkf

problem = toy.lorenz96(dim=40, n_steps=300, obs_every=2, noise_sd=1.0)

ensemble = problem.initial.sample(jax.random.key(0), n_particles=40)   # Ensemble over "x"
result = enkf.filter(
    jax.random.key(1), ensemble, problem.observations,
    transition=problem.transition,        # (J, 40) -> (J, 40), a plain JAX function
    observe=problem.observe,              # maps.Linear: every other coordinate
    noise_cov=problem.noise_cov,          # PSDDiagonal
    update_rule=kalman.Matheron(),
    inflation=MultiplicativeInflation(1.05),
)

rmse = jnp.sqrt(jnp.mean((result.means["x"] - problem.truth) ** 2, axis=1))
print("time-averaged RMSE after spin-up:", rmse[50:].mean())
print("total log evidence:", result.log_evidence.sum())

# ---- checks
assert rmse[50:].mean() < 0.6   # well below the observation noise sd of 1.0
