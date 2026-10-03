"""Example 13 - named blocks: joint state-parameter estimation and lag-1 smoothing."""
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, toy
from enskit.distribution import Ensemble
from enskit.algorithms import enkf

problem = toy.lorenz96(dim=40, n_steps=200, obs_every=2, noise_sd=1.0)   # true forcing 8

key = jax.random.key(0)
k1, k2, key = jax.random.split(key, 3)
ens = Ensemble(
    x=problem.initial.sample(k1, 40)["x"],
    forcing=6.0 + jax.random.normal(k2, (40, 1)),        # unknown parameter, prior N(6, 1)
)


def transition(x, forcing):                              # reads two blocks, writes one
    return toy.l96_step(x, forcing)


filtered, smoothed, forcing = [], [], []
for y in problem.observations:
    key, k_fc, k_an = jax.random.split(key, 3)
    ens = ens.assign(x_prev=ens["x"])                    # carry the previous state along
    ens = enkf.forecast(k_fc, ens, transition, state="x", inputs=("x", "forcing"))
    ens = kalman.inflate_multiplicative(ens, 1.02)
    ens, _ = enkf.analysis(k_an, ens, y, observe=problem.observe,
                           noise_cov=problem.noise_cov, update_rule=kalman.Matheron(),
                           inputs=("x",))
    # one update moved every non-given block: x (filter), x_prev (smoother), forcing
    filtered.append(ens.mean("x"))
    smoothed.append(ens.mean("x_prev"))
    forcing.append(ens.mean("forcing")[0])

filtered, smoothed = jnp.stack(filtered), jnp.stack(smoothed)
rmse = lambda m, t: jnp.sqrt(jnp.mean((m - t) ** 2, axis=1))[50:].mean()  # noqa: E731
print(f"filter RMSE at t-1 {rmse(filtered[:-1], problem.truth[:-1]):.3f}   "
      f"lag-1 smoother RMSE {rmse(smoothed[1:], problem.truth[:-1]):.3f}")
print(f"forcing estimate {forcing[-1]:.3f} (truth 8)")

# ---- checks
assert rmse(smoothed[1:], problem.truth[:-1]) < rmse(filtered[:-1], problem.truth[:-1])
assert abs(forcing[-1] - 8.0) < 0.3
