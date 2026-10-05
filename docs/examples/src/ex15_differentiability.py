# %% [markdown]
# # Differentiating through updates
#
# Take derivatives with respect to a noise scale through one update with a
# degenerate spectrum, through a whole EKI run, and through an ensemble
# likelihood whose simulator is not traceable.
#
# ## When to use this
#
# Derivatives through ensemble Kalman updates let noise levels and other
# hyperparameters be tuned by gradient methods, and model components be
# trained through a filter or an inversion [@chen2022]. They require the
# forward model to be written in JAX; a simulator that is not can still run
# inside a traced function, with derivatives taken through the rest.
#
# ## Setup
#
# All three parts differentiate with respect to $s$, the log of a scale on
# the noise covariance, at $s = 0$.
#
# 1. The linear model of `toy.linear_gaussian(parameter_dim=4, data_dim=6,
#    noise_std=0.3)`: $g = A u$ with $u \in \mathbb R^4$, prior
#    $\mathcal N(0, I_4)$, $A \in \mathbb R^{6 \times 4}$ with independent
#    $\mathcal N(0, 1/4)$ entries, and data $y = A u^\dagger + e$ with
#    $e \sim \mathcal N(0, 0.3^2 I_6)$. It is extended by three coordinates
#    fixed at 1, so $\tilde g(u) = (A u, 1, 1, 1)$ and
#    $\tilde y = (y, 1, 1, 1)$. The noise is $R(s) = 0.3^2 e^{2s} I_9$ and
#    there are 30 particles. The output is $\phi(s) = \mathbf 1^\top
#    \hat C_u^a(s)\, \mathbf 1$, the sum of the entries of the sample posterior
#    covariance of $u$ after a symmetric square-root update
#    [@tippett2003].
# 2. The decay model of `toy.exponential_decay`, $G(u)_i = a\, e^{-r t_i}$ with
#    $u = (a, r)$, at the twelve times $t_i = i/4$, $i = 1, \dots, 12$, with
#    prior $\mathcal N\big((1, 1),\ I_2\big)$, data generated from
#    $u^\dagger = (2, 1.5)$, $R(s) = 0.02^2 e^{2s} I_{12}$ and 32 particles.
#    The output is the misfit $m(s) = \lVert G(\bar u_4) - y \rVert^2$ of the
#    mean $\bar u_4$ after four EKI steps.
# 3. The same decay model, written in NumPy, and the ensemble log-likelihood
#
# $$
# \ell(s, U) = \log \mathcal N\big(y;\ \bar g(U),\ \hat C_{gg}(U) + R(s)\big),
# $$
#
# where $\bar g$ and $\hat C_{gg}$ are the sample mean and covariance of the
# simulator outputs for the particles $U \in \mathbb R^{32 \times 2}$.

# %%
import jax
import jax.numpy as jnp
import numpy as np

import enskit
from enskit import kalman, maps, toy
from enskit.algorithms import eki
from enskit.distribution import Ensemble
from enskit.linalg import PSDDiagonal

# %% [markdown]
# ## Degenerate spectra
#
# The update works with the whitened factor $S = (R^{-1/2} F_g)^\top \in
# \mathbb R^{30 \times 9}$, where $F_g$ holds the predicted anomalies divided by
# $\sqrt{J-1}$, and the posterior sample covariance is
# $\hat C_u^a = F_u T T^\top F_u^\top$ with $T = (I + S S^\top)^{-1/2}$. The
# three fixed coordinates give $S$ three zero columns, so three of its
# singular values are exactly zero.

# %%
problem = toy.linear_gaussian(parameter_dim=4, data_dim=6, noise_std=0.3)

# (a) Three predicted coordinates are identical across particles (a fixed boundary
#     value, say), so the whitened factor has exactly-zero singular values.
def forward_with_fixed(u):
    return jnp.concatenate([problem.forward(u), jnp.ones((u.shape[0], 3))], axis=1)

y = jnp.concatenate([problem.y, jnp.ones(3)])
ens = problem.prior.sample(jax.random.key(0), n_particles=30)
ens = maps.pushforward(ens, forward_with_fixed, inputs="u", output="g")

# %% [markdown]
# Both functions below compute $\phi$. The second forms $T$ from a plain
# SVD, whose derivative contains $1 / (\sigma_i^2 - \sigma_j^2)$, infinite
# for repeated singular values. The first uses `IdentityPlusGram`, whose
# custom derivatives, with $M = (I + S S^\top)^{-1}$, $w = M S b$ and
# $dA = dS\, S^\top + S\, dS^\top$, are
#
# $$
# dw = M (dS\, b + S\, db) - M\, dA\, w, \qquad
# dT = U \big(\Gamma \circ (U^\top dA\, U)\big) U^\top, \quad
# \Gamma_{ij} = \frac{-1}{s_i s_j (s_i + s_j)}.
# $$
#
# Here $S S^\top = U \operatorname{diag}(\lambda) U^\top$ with $U$ orthogonal,
# and $s_i = \sqrt{1 + \lambda_i}$. $\Gamma$ holds the divided differences
# of $f(\lambda) = (1 + \lambda)^{-1/2}$ in the Daleckii-Krein formula
# [@higham2008], and stays finite when eigenvalues coincide.

# %%
def posterior_spread(log_sd):
    noise = PSDDiagonal(jnp.full(9, 0.09)) * jnp.exp(2 * log_sd)
    post = kalman.update(ens, g=y, noise={"g": noise},
                         update_rule=kalman.SymmetricSquareRoot())
    return jnp.sum(post.cov("u").to_dense())


def naive_posterior_spread(log_sd):          # the same update via a plain SVD
    Fu = ens.project().factor("u").to_dense()
    S = (ens.project().factor("g").to_dense() / (0.3 * jnp.exp(log_sd))).T
    U, s, _ = jnp.linalg.svd(S, full_matrices=False)
    T = jnp.eye(S.shape[0]) + (U * (1 / jnp.sqrt(1 + s**2) - 1)) @ U.T
    return jnp.sum((Fu @ T) @ (Fu @ T).T)


eps = 1e-6
fd = (posterior_spread(eps) - posterior_spread(-eps)) / (2 * eps)
grad_naive = jax.grad(naive_posterior_spread)(0.0)
grad_spread = jax.grad(posterior_spread)(0.0)
print(f"plain-SVD gradient: {float(grad_naive)}")
print(f"enskit gradient:    {float(grad_spread):.4f}  finite difference: {float(fd):.4f}")

# %% [markdown]
# ## A whole run
#
# Starting from fixed particles and a fixed key, four steps with increments
# $\delta = 1/8, 1/8, 1/4, 1/2$ take $\beta$ from 0 to 1, each conditioning
# with noise covariance $R(s)/\delta$ and the stochastic update
# [@burgers1998]. The noise draws are fixed by the key, so $m(s)$ is a
# deterministic function of $s$ and `jax.grad` differentiates it end to end.

# %%
# (b) A traceable forward model: differentiate a whole 4-step run end to end.
decay = toy.exponential_decay()
start = decay.prior.sample(jax.random.key(1), n_particles=32)


def final_misfit(log_noise_scale):
    noise_cov = decay.noise_cov * jnp.exp(2 * log_noise_scale)
    state = eki.EKIState(start, key=jax.random.key(2))
    for dbeta in (0.125, 0.125, 0.25, 0.5):
        state = eki.advance(state, decay.forward, decay.y, noise_cov, dbeta,
                            update_rule=kalman.Matheron())
    g = decay.forward(state.ensemble.mean("u")[None])[0]
    return jnp.sum((g - decay.y) ** 2)


g_ad = jax.grad(final_misfit)(0.0)
g_fd = (final_misfit(1e-5) - final_misfit(-1e-5)) / 2e-5
print(f"end-to-end gradient: {float(g_ad):.4g}   finite difference: {float(g_fd):.4g}")

# %% [markdown]
# ## A simulator outside JAX
#
# `maps.BlackBox` calls the NumPy function from inside traced code and
# declares its derivative to be zero. Under `jax.jit` and `jax.vmap` it
# runs as usual; $\partial \ell / \partial s$ comes from the noise model
# alone. The computed $\partial \ell / \partial U$ is zero because of that
# declaration, although $\ell$ does depend on $U$.

# %%
# (c) A host-side simulator. BlackBox lets it run under jit and vmap, and its
#     outputs are constants: only the noise model is differentiated.
times = np.asarray(decay.times)


def numpy_decay(u):                                  # plain NumPy, not traceable
    return u[:, :1] * np.exp(-u[:, 1:2] * times)


simulator = maps.BlackBox(numpy_decay, output_dim=times.size)


@jax.jit
def ensemble_evidence(log_sd, u):
    ens = maps.pushforward(Ensemble(u=u), simulator, inputs="u", output="g")
    noise = decay.noise_cov * jnp.exp(2 * log_sd)
    return kalman.gaussian_approximation(ens, {"g": noise}).log_density(g=decay.y)


u = start["u"]
grad_s = jax.grad(ensemble_evidence)(0.0, u)
grad_u = jax.grad(ensemble_evidence, 1)(0.0, u)
two = jax.vmap(ensemble_evidence, in_axes=(None, 0))(0.0, jnp.stack([u, 1.1 * u]))
print(f"d/d log_sd: {float(grad_s):.3f}")
print("d/du is zero through the simulator:", bool(jnp.all(grad_u == 0)))
print(f"vmapped over two ensembles: {float(two[0]):.3f}, {float(two[1]):.3f}")

# %% [markdown]
# ## What to notice
#
# With three exactly zero singular values the plain-SVD derivative is `nan`,
# while the custom rules give 0.604, matching the central finite difference
# to a relative error below $10^{-9}$. The derivative of the four-step run,
# $-9.364 \times 10^{-6}$, agrees with its finite difference to a relative
# error below $10^{-6}$. The simulator run under `jax.jit` and `jax.vmap`
# gives a finite $\partial \ell / \partial s = 0.467$ and an exactly zero
# derivative in $U$.

# %%
# checks
def rel_err(a, b):
    return float(jnp.abs(a - b) / jnp.abs(b))


assert jnp.isnan(grad_naive)
assert jnp.allclose(naive_posterior_spread(0.0), posterior_spread(0.0), rtol=1e-12)
assert round(float(grad_spread), 3) == 0.604
assert rel_err(grad_spread, fd) < 1e-9
assert f"{float(g_ad):.4g}" == "-9.364e-06"
assert rel_err(g_ad, g_fd) < 1e-6
assert jnp.isfinite(grad_s) and round(float(grad_s), 3) == 0.467
assert jnp.all(grad_u == 0)
assert jnp.allclose(two[0], ensemble_evidence(0.0, u), rtol=1e-12)
traced = maps.pushforward(Ensemble(u=u), decay.forward, inputs="u", output="g")
assert jnp.allclose(ensemble_evidence(0.0, u),
                    kalman.gaussian_approximation(traced, {"g": decay.noise_cov})
                    .log_density(g=decay.y), rtol=1e-12)
