# %% [markdown]
# # Hyperparameters by gradient
#
# Fit a prior lengthscale and a noise level by maximizing the log evidence with
# `jax.value_and_grad` and SciPy's L-BFGS-B, first exactly for a Gaussian
# process prior and then approximately from an ensemble.
#
# ## When to use this
#
# Every operation on a `Gaussian` is differentiable, so a log evidence built
# from them can be maximized by a gradient-based optimizer. For a Gaussian
# prior and a linear observation model the evidence is exact, and maximizing
# it is a common way to choose Gaussian process hyperparameters
# [@rasmussen2006]. When the model is a nonlinear simulator, the evidence of
# the moment-matched Gaussian built from one ensemble of simulator outputs is
# a usable approximation [@carrassi2017], and parameters of the noise model can
# be tuned against it without differentiating the simulator.
#
# ## Setup
#
# **(a) Exact evidence.** A field $f \in \mathbb R^{50}$ on a grid $s_1, \dots,
# s_{50}$ equally spaced on $[0, 1]$ has a zero-mean Gaussian process prior with
# squared-exponential kernel, and is observed at the 16 grid points
# $s_3, s_6, \dots, s_{48}$ through the selection matrix
# $H \in \mathbb R^{16 \times 50}$:
#
# $$
# f \sim \mathcal N(0, K_\ell), \quad
# (K_\ell)_{ij} = \exp\!\Big(-\frac{(s_i - s_j)^2}{2 \ell^2}\Big) + 10^{-8} \delta_{ij},
# \qquad
# y = H f + e, \quad e \sim \mathcal N(0, \sigma^2 I_{16}).
# $$
#
# Eight independent fields are drawn with $\ell = 0.15$ and observed with
# $\sigma = 0.1$, giving data $y^{(1)}, \dots, y^{(8)}$. The objective is the
# summed log evidence over $\theta = (\log \ell, \log \sigma)$,
#
# $$
# \mathcal L(\theta) = \sum_{i=1}^{8} \log \mathcal N\big(y^{(i)};\ 0,\
# H K_\ell H^\top + \sigma^2 I_{16}\big).
# $$
#
# **(b) Ensemble evidence.** For the decay model $G(u)_k = a\, e^{-r t_k}$ of
# Example 1, with ten data $y$ generated at noise level $0.05$, draw $J = 256$
# particles $u_j$ from the prior and evaluate $g_j = G(u_j)$ once. With $\bar g$
# and $C_{gg}$ the sample mean and covariance of the $g_j$, the ensemble
# evidence as a function of $\tau = \log \sigma$ is
#
# $$
# \widehat{\mathcal L}(\tau) = \log \mathcal N\big(y;\ \bar g,\
# C_{gg} + \sigma^2 I_{10}\big).
# $$

# %%
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import enskit
from enskit import kalman, maps, toy
from enskit.distribution import Gaussian
from enskit.linalg import Dense, DensePSD, PSDDiagonal

# %% [markdown]
# ## (a) The exact evidence
#
# `log_evidence` builds the joint Gaussian of $(f, y)$ for given $\theta$ and
# returns the density of its $y$ marginal. The data form an $(8, 16)$ array,
# and `log_density` returns one value per row, which the objective sums.

# %%
grid = jnp.linspace(0.0, 1.0, 50)
observe = maps.Linear(Dense(jnp.eye(50)[2::3]))


def prior_cov(log_lengthscale):
    d2 = (grid[:, None] - grid[None, :]) ** 2
    K = jnp.exp(-0.5 * d2 / jnp.exp(2 * log_lengthscale))
    return DensePSD(K + 1e-8 * jnp.eye(50))


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
f_true = jax.random.normal(k1, (8, 50)) @ L_true.T
y = observe(f_true) + 0.1 * jax.random.normal(k2, (8, 16))

# %% [markdown]
# `jax.value_and_grad` returns $-\mathcal L(\theta)$ and its gradient in one
# call, and `jax.jit` compiles both. SciPy's `minimize` with `jac=True` expects
# exactly that pair, as NumPy arrays, and runs L-BFGS-B from
# $\ell = \sigma = 0.5$.

# %%
objective = jax.jit(jax.value_and_grad(lambda th: -jnp.sum(log_evidence(th, y))))
fit = minimize(lambda th: tuple(np.asarray(v) for v in objective(th)),
               np.log([0.5, 0.5]), jac=True, method="L-BFGS-B")
print("(a) lengthscale, noise sd:", jnp.exp(fit.x), " (truth 0.15, 0.10)")

# %% [markdown]
# ## (b) The ensemble evidence
#
# The simulator is treated as a black box. It is evaluated once, outside the
# objective, so its outputs enter $\widehat{\mathcal L}$ as constants and only
# the noise model is differentiated. `kalman.gaussian_approximation` builds the
# moment-matched Gaussian of the particles and adds the noise
# $\sigma^2 I_{10}$ to the block `"g"`.

# %%
problem = toy.exponential_decay(noise_sd=0.05)
ens = problem.prior.sample(jax.random.key(1), n_particles=256)
ens = maps.pushforward(ens, problem.forward, inputs="u", output="g")


def ensemble_log_evidence(log_noise_sd):
    noise = PSDDiagonal(jnp.full(problem.y.shape[0], jnp.exp(2 * log_noise_sd)))
    return kalman.gaussian_approximation(ens, noise={"g": noise}).log_density(g=problem.y)


grad = jax.grad(ensemble_log_evidence)
print("(b) d log Z / d log sd at sd = 0.05:", grad(jnp.log(0.05)))
neg_b = jax.jit(jax.value_and_grad(lambda t: -ensemble_log_evidence(t[0])))
fit_b = minimize(lambda t: tuple(np.asarray(v) for v in neg_b(t)), np.log([0.5]),
                 jac=True, method="L-BFGS-B")
print("(b) evidence-maximizing noise sd:", jnp.exp(fit_b.x[0]))

# %% [markdown]
# ## What to notice
#
# In (a) the optimizer recovers $\ell = 0.153$ and $\sigma = 0.085$ from eight
# fields of 16 points each, against true values of 0.15 and 0.10. In (b) the
# derivative of $\widehat{\mathcal L}$ at the true noise level is $-1.39$, so
# the ensemble evidence increases toward smaller $\sigma$, and its maximizer is
# $\sigma = 0.044$. The checks confirm the gradient in (b) against a central
# finite difference.

# %%
# checks
assert fit.success and jnp.allclose(jnp.exp(fit.x), jnp.array([0.15, 0.1]), rtol=0.25)
eps = 1e-6
fd = (ensemble_log_evidence(jnp.log(0.05) + eps) - ensemble_log_evidence(jnp.log(0.05) - eps)) / (2 * eps)
assert jnp.allclose(grad(jnp.log(0.05)), fd, rtol=1e-6)
