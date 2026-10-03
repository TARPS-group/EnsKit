"""Example 8 - tuning prior and noise hyperparameters with jax.grad."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import enskit
from enskit import kalman, maps, toy
from enskit.distribution import Gaussian
from enskit.linalg import Dense, DensePSD, PSDDiagonal

# (a) Exact: a Gaussian-process prior on a grid, observed at 16 points.
grid = jnp.linspace(0.0, 1.0, 50)
observe = maps.Linear(Dense(jnp.eye(50)[2::3]))                # 16 grid points


def prior_cov(log_lengthscale):
    d2 = (grid[:, None] - grid[None, :]) ** 2
    K = jnp.exp(-0.5 * d2 / jnp.exp(2 * log_lengthscale))
    return DensePSD(K + 1e-8 * jnp.eye(50))                     # Cholesky runs here


def log_evidence(theta, y):
    log_lengthscale, log_noise_sd = theta
    prior = Gaussian.independent(f=(jnp.zeros(50), prior_cov(log_lengthscale)))
    joint = (prior
             .pipe(maps.pushforward, observe, inputs="f", output="y")
             .add_noise(y=PSDDiagonal(jnp.full(16, jnp.exp(2 * log_noise_sd)))))
    return joint.log_density(y=y)


truth = jnp.log(jnp.array([0.15, 0.1]))
k1, k2 = jax.random.split(jax.random.key(0))
L_true = jnp.linalg.cholesky(prior_cov(truth[0]).to_dense())
f_true = jax.random.normal(k1, (8, 50)) @ L_true.T                # 8 independent fields
y = observe(f_true) + 0.1 * jax.random.normal(k2, (8, 16))       # batch of 8 data vectors

# log_density follows the batch contract: a (8, 16) operand gives 8 log densities
objective = jax.jit(jax.value_and_grad(lambda th: -jnp.sum(log_evidence(th, y))))
fit = minimize(lambda th: tuple(np.asarray(v) for v in objective(th)),
               np.log([0.5, 0.5]), jac=True, method="L-BFGS-B")
print("(a) lengthscale, noise sd:", jnp.exp(fit.x), " (truth 0.15, 0.10)")

# (b) Ensemble: the simulator is a black box, so its outputs are constants and
#     only the noise model is differentiated. The ensemble evidence is the
#     density of y under the moment-matched joint.
problem = toy.exponential_decay(noise_sd=0.05)
ens = problem.prior.sample(jax.random.key(1), n_particles=256)
ens = maps.pushforward(ens, problem.forward, inputs="u", output="g")   # evaluated once


def ensemble_log_evidence(log_noise_sd):
    noise = PSDDiagonal(jnp.full(problem.y.shape[0], jnp.exp(2 * log_noise_sd)))
    return kalman.gaussian_approximation(ens, noise={"g": noise}).log_density(g=problem.y)


grad = jax.grad(ensemble_log_evidence)
print("(b) d log Z / d log sd at sd = 0.05:", grad(jnp.log(0.05)))
neg_b = jax.jit(jax.value_and_grad(lambda t: -ensemble_log_evidence(t[0])))
fit_b = minimize(lambda t: tuple(np.asarray(v) for v in neg_b(t)), np.log([0.5]),
                 jac=True, method="L-BFGS-B")
print("(b) evidence-maximizing noise sd:", jnp.exp(fit_b.x[0]))

# ---- checks
assert fit.success and jnp.allclose(jnp.exp(fit.x), jnp.array([0.15, 0.1]), rtol=0.25)
eps = 1e-6
fd = (ensemble_log_evidence(jnp.log(0.05) + eps) - ensemble_log_evidence(jnp.log(0.05) - eps)) / (2 * eps)
assert jnp.allclose(grad(jnp.log(0.05)), fd, rtol=1e-6)
