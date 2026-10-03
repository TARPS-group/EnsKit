"""Example 6 - two estimators of the same joint: the order of operations matters."""
import jax
import jax.numpy as jnp

import enskit
from enskit import maps, toy

problem = toy.linear_gaussian(parameter_dim=4, data_dim=6, noise_sd=1.0)
A = problem.forward.op.to_dense()
true_cyy = A @ problem.prior.cov("u").to_dense() @ A.T + problem.noise_cov.to_dense()
noise = maps.AdditiveNoise(problem.noise_cov)


def both_estimates(key):
    k1, k2 = jax.random.split(key)
    ens = problem.prior.sample(k1, n_particles=20)
    ens = maps.pushforward(ens, problem.forward, inputs="u", output="g")
    # sample the noise, then fit:   C_yy ~ sample cov of (g + e)
    sampled = maps.pushforward(ens, noise, inputs="g", output="y", key=k2).project()
    # fit, then add the noise:      C_yy ~ sample cov of g + R, exactly
    structured = maps.pushforward(ens.project(), noise, inputs="g", output="y")
    err = lambda g: jnp.sum((g.cov("y").to_dense() - true_cyy) ** 2)  # noqa: E731
    return err(sampled), err(structured)


errs = jnp.array([both_estimates(k) for k in jax.random.split(jax.random.key(0), 500)])
print("mean squared Frobenius error of C_yy  (sample-then-fit, fit-then-add):",
      errs.mean(axis=0))

# ---- check
assert errs[:, 1].mean() < errs[:, 0].mean()
