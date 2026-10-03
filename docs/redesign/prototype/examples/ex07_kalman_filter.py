"""Example 7 - an exact Kalman filter from the Gaussian algebra alone."""
import jax.numpy as jnp

import enskit
from enskit import maps, toy

problem = toy.linear_state_space(dim=3, obs_dim=2, n_steps=20)

belief = problem.initial                                   # Gaussian over "x"
log_likelihood = 0.0
means = []
for y in problem.observations:
    belief = (belief
              .pipe(maps.pushforward, problem.transition, inputs="x", output="x")
              .pipe(maps.pushforward, maps.AdditiveNoise(problem.transition_noise),
                    inputs="x", output="x"))
    joint = (belief
             .pipe(maps.pushforward, problem.observe, inputs="x", output="y")
             .add_noise(y=problem.noise_cov))
    log_likelihood += joint.log_density(y=y)                # density of the y marginal
    belief = joint.condition(y=y).compress()                # keep the latent width bounded
    means.append(belief.mean("x"))

# ---- check against a hand-written dense Kalman filter
A = problem.transition.op.to_dense(); H = problem.observe.op.to_dense()
Q = problem.transition_noise.to_dense(); R = problem.noise_cov.to_dense()
m, P, ll = jnp.zeros(3), jnp.eye(3), 0.0
import jax.scipy.stats as st
for t, y in enumerate(problem.observations):
    m, P = A @ m, A @ P @ A.T + Q
    S = H @ P @ H.T + R
    ll += st.multivariate_normal.logpdf(y, H @ m, S)
    K = P @ H.T @ jnp.linalg.inv(S)
    m, P = m + K @ (y - H @ m), P - K @ S @ K.T
    assert jnp.allclose(means[t], m, atol=1e-10)
assert jnp.allclose(belief.cov("x").to_dense(), P, atol=1e-10)
assert jnp.allclose(log_likelihood, ll, atol=1e-9)
print("matches the dense Kalman filter; log likelihood", log_likelihood)
print("latent width after 20 steps:", belief.latent_dim)
