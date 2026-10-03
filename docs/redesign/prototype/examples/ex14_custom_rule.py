"""Example 14 - writing a new update method: the deterministic EnKF (DEnKF)."""
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, maps, toy
from enskit.distribution import exact_moment_ensemble
from enskit.algorithms import MultiplicativeInflation, eki, enkf


class DEnKF:
    """Update rule: full gain on the mean, half gain on the anomalies.

    Particle j moves by K (y* - g_bar - (g_j - g_bar) / 2): the approximation's
    Matheron map, applied with no noise to particles whose given blocks are
    pulled halfway to their mean.
    """

    def build(self, particles, approximation, given):
        cmap = approximation.conditional_map(given)            # pointwise, value-free
        halfway = particles.assign(
            {c: particles.mean(c) + 0.5 * particles.anomalies(c) for c in given})

        def update(values=None, /, *, key=None, **block_values):  # the value arrives here
            return cmap(halfway, values, **block_values).marginal(*cmap.targets)

        return update


# It plugs into the one-call update, and therefore into every driver.
problem = toy.linear_gaussian(parameter_dim=4, data_dim=6)
ens = exact_moment_ensemble(jax.random.key(0), problem.prior, n_particles=32)
ens = maps.pushforward(ens, problem.forward, inputs="u", output="g")
post = kalman.update(ens, g=problem.y, noise={"g": problem.noise_cov}, update_rule=DEnKF())
exact = kalman.update(ens, g=problem.y, noise={"g": problem.noise_cov},
                      update_rule=kalman.SymmetricSquareRoot())

l96 = toy.lorenz96(dim=40, n_steps=300)
res = enkf.filter(jax.random.key(1), l96.initial.sample(jax.random.key(2), 40),
                  l96.observations, transition=l96.transition, observe=l96.observe,
                  noise_cov=l96.noise_cov, update_rule=DEnKF(),
                  inflation=MultiplicativeInflation(1.02))
rmse = jnp.sqrt(jnp.mean((res.means["x"] - l96.truth) ** 2, axis=1))[50:].mean()

decay = toy.exponential_decay()
fit = eki.run(eki.EKIState.from_prior(jax.random.key(3), decay.prior, 64), decay.forward,
              decay.y, decay.noise_cov, update_rule=DEnKF(), schedule=eki.AdaptiveESSSchedule())
print(f"DEnKF: L96 RMSE {rmse:.3f};  EKI mean {fit.ensemble.mean('u')}")

# ---- checks: the mean update is the exact Kalman mean; the spread is larger
assert jnp.allclose(post.mean("u"), exact.mean("u"), atol=1e-12)
assert jnp.all(jnp.diag(post.cov("u").to_dense()) >= jnp.diag(exact.cov("u").to_dense()))
assert rmse < 0.6
