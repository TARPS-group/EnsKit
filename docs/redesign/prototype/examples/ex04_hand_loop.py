"""Example 4 - EKI written by hand: forecast and update as separate calls."""
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, maps, toy
from enskit.algorithms import eki

problem = toy.exponential_decay()
increments = (1 / 16, 1 / 16, 1 / 8, 1 / 4, 1 / 2)      # binary-exact: sums to exactly 1

key = jax.random.key(0)
key, sub = jax.random.split(key)
ens = problem.prior.sample(sub, n_particles=64)            # Ensemble over "u"

for step, dbeta in enumerate(increments):
    # forecast: one evaluation of the forward model for the whole ensemble
    ens = maps.pushforward(ens, problem.forward, inputs="u", output="g")
    misfit = 0.5 * jnp.sum(problem.noise_cov.whiten(problem.y - ens["g"]) ** 2, axis=1)
    print(f"step {step}: increment {dbeta:.4f}, mean misfit {misfit.mean():8.2f}, "
          f"spread {jnp.std(ens['u'], axis=0)}")
    # update: condition on g with the tempered noise covariance R / dbeta
    ens = kalman.update(ens, g=problem.y, noise={"g": problem.noise_cov / dbeta},
                        update_rule=kalman.SymmetricSquareRoot())   # returns an Ensemble over "u" only

print("hand-written loop:", ens.mean("u"))

# ---- check: identical to the driver with the same fixed schedule
state = eki.EKIState(problem.prior.sample(jax.random.split(jax.random.key(0))[1], 64),
                     key=jax.random.key(1))
result = eki.run(state, problem.forward, problem.y, problem.noise_cov,
                 update_rule=kalman.SymmetricSquareRoot(), schedule=eki.FixedSchedule(increments))
assert jnp.allclose(result.ensemble["u"], ens["u"], atol=1e-12)
print("driver agrees:", result.ensemble.mean("u"))
