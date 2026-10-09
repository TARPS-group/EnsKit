# %% [markdown]
# # Statistical linearization of a simulator
#
# Fit the affine map that best predicts a simulator's outputs from its inputs
# over an ensemble, compare it with the simulator's average and pointwise
# Jacobians, read its residuals, and recover an ensemble update as the exact
# linear-Gaussian update of that map.
#
# ## When to use this
#
# An ensemble Kalman method never differentiates the simulator. It relies on
# a linear model fitted to the particles: the statistical linear regression
# of the simulator with respect to the ensemble [@lefebvre2002], known in
# the iterative ensemble smoother literature as the *average sensitivity*
# [@chen2013]. Computing that model explicitly is useful when you want a
# derivative-free sensitivity of outputs to inputs, a linear surrogate to
# push a Gaussian through exactly, or a measure of how far the simulator is
# from affine over the ensemble's spread.
#
# ## Setup
#
# The simulator is the decay model of `toy.exponential_decay`,
#
# $$
# G(u)_i = a\, e^{-r t_i}, \qquad u = (a, r), \qquad
# t_i = i/4, \quad i = 1, \dots, 12,
# $$
#
# and the inputs are $u \sim \mathcal N(m, s^2 I_2)$ with $m = (1, 1)$, at two
# spreads, $s = 0.1$ and $s = 0.4$. From $J$ samples $u_j$ and outputs
# $g_j = G(u_j)$, the statistical linearization is the least-squares fit
#
# $$
# A = \hat C_{gu}\hat C_{uu}^{-1}, \qquad b = \bar g - A\bar u, \qquad
# \Omega = \frac{1}{J-1}\sum_{j=1}^{J} e_j e_j^\top, \qquad
# e_j = g_j - Au_j - b .
# $$
#
# For Gaussian inputs, Stein's lemma [@stein1981] gives
# $C_{gu} = \mathbb E[DG(u)]\, C_{uu}$, so the population fit is the average
# Jacobian $\mathbb E[DG(u)]$. Here it has a closed form: with
# $\mathbb E[e^{-rt}] = e^{-t + s^2t^2/2}$ for $r \sim \mathcal N(1, s^2)$,
#
# $$
# \mathbb E[DG(u)]_{i\cdot} = e^{-t_i + s^2t_i^2/2}\,\big(1,\ -t_i\big),
# $$
#
# which at $s = 0$ is the Jacobian $DG(m)$ at the mean.

# %%
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, maps, toy
from enskit.distribution import Gaussian
from enskit.linalg import PSDDiagonal

problem = toy.exponential_decay()
t = problem.times
m = jnp.array([1.0, 1.0])


def average_jacobian(s):
    E = jnp.exp(-t + s**2 * t**2 / 2)
    return jnp.stack([E, -t * E], axis=1)        # (12, 2)


def relative(A, B):
    return float(jnp.linalg.norm(A - B) / jnp.linalg.norm(B))


def linearize(s, J, key):
    """Sample the inputs, run the simulator once, and fit."""
    inputs = Gaussian.independent(u=(m, PSDDiagonal(jnp.full(2, s**2))))
    ens = inputs.sample(key, J)
    ens = maps.pushforward(ens, problem.forward, inputs="u", output="g")
    return ens, maps.statistical_linearization(ens, inputs="u", output="g")


# %% [markdown]
# ## The fit is the average Jacobian, not the pointwise one
#
# At $J = 2000$ particles, `fit.map.op` is $A$. It is compared with the
# closed-form average Jacobian at the same spread, and with the Jacobian at
# the mean.

# %%
J = 2000
fits = {}
for s in (0.1, 0.4):
    ens, fit = linearize(s, J, jax.random.key(0))
    fits[s] = (ens, fit)
    A = fit.map.op.to_dense()
    print(f"s = {s}: A is {relative(A, average_jacobian(s)):.3f} from E[DG] and "
          f"{relative(A, average_jacobian(0.0)):.3f} from DG(m); "
          f"E[DG] is {relative(average_jacobian(s), average_jacobian(0.0)):.3f} "
          f"from DG(m)")

# %% [markdown]
# At $s = 0.1$ every candidate agrees to about one percent: over a narrow
# ensemble the simulator is nearly affine. At $s = 0.4$ the average Jacobian
# is 22% from the Jacobian at the mean, and the fit follows the average, at
# 3.5%, a sampling error, rather than the pointwise one. As an ensemble
# collapses, its linearization tends to the Jacobian at its mean.
#
# ## The residuals measure the nonlinearity, when there are enough particles
#
# The residuals $e_j$ are what the fit leaves over. Their root mean square is
# set against the observation noise, whose standard deviation is 0.02.

# %%
for s, (_, fit) in fits.items():
    rms = float(jnp.sqrt(jnp.mean(fit.residuals**2)))
    print(f"s = {s}: residual rms {rms:.4f}, {rms / 0.02:.2f} noise standard deviations")

ens3, fit3 = linearize(0.4, 3, jax.random.key(0))
print(f"J = 3: largest residual {float(jnp.abs(fit3.residuals).max()):.1e}")

# %% [markdown]
# At $s = 0.4$ the residuals are about seven times the noise, so the update
# would treat the simulator's nonlinearity as the larger source of error. At
# $s = 0.1$ they are about a fifth of it. With $J = 3 = d_u + 1$ particles, the fit
# passes through all three and the residuals vanish to round-off whatever
# the simulator is: in-sample residuals carry information only when
# $J - 1$ exceeds the number of inputs.
#
# ## The update is the linear-Gaussian update of the fit
#
# A deterministic square-root update of $u$ on the observation $y$, with
# noise $R$ [@tippett2003], gives the same posterior mean as the exact Kalman
# update of the Gaussian $\mathcal N(\bar u, \hat C_{uu})$ through the linear
# model $g = Au + b + e$, $e \sim \mathcal N(0, R + \Omega)$ (a stochastic
# update agrees with it only in expectation over its noise draws):
#
# $$
# \bar u^a = \bar u + K\big(y - A\bar u - b\big), \qquad
# K = \hat C_{uu}A^\top\big(A\hat C_{uu}A^\top + R + \Omega\big)^{-1}.
# $$
#
# The residual covariance enters as extra noise. Dropping it gives a
# different, overconfident update.

# %%
ens, fit = fits[0.4]
R = problem.noise_cov.to_dense()
post = kalman.update(ens, g=problem.y, noise={"g": problem.noise_cov},
                     update_rule=kalman.SymmetricSquareRoot())
A, b, Omega = fit.map.op.to_dense(), fit.map.shift, fit.residual_cov.to_dense()
u_bar, C = ens.mean("u"), ens.cov("u").to_dense()


def kalman_mean(noise):
    K = C @ A.T @ jnp.linalg.inv(A @ C @ A.T + noise)
    return u_bar + K @ (problem.y - A @ u_bar - b)


with_omega, without_omega = kalman_mean(R + Omega), kalman_mean(R)
print("ensemble update mean        ", post.mean("u"))
print("linear model, noise R + Omega", with_omega)
print("linear model, noise R        ", without_omega)
print(f"difference with Omega {float(jnp.abs(post.mean('u') - with_omega).max()):.1e}, "
      f"without {float(jnp.abs(post.mean('u') - without_omega).max()):.1e}")

# %% [markdown]
# ## What to notice
#
# - The statistical linearization is a property of the simulator *and* the
#   input distribution. At $s = 0.4$ it is 3.5% from the average Jacobian
#   and 26% from the Jacobian at the mean; at $s = 0.1$ the two Jacobians
#   are 1.1% apart.
# - The residuals are informative only with more particles than inputs plus
#   one: at $J = 2000$ and $s = 0.4$ their root mean square is 7.1 noise
#   standard deviations, and at $J = 3$ they are zero to round-off.
# - The square-root update's mean equals the linear-Gaussian update with noise
#   $R + \Omega$ to round-off, and differs from the one with noise $R$ alone
#   by 0.06 in the amplitude.

# %%
# checks
A1 = fits[0.1][1].map.op.to_dense()
A4 = fits[0.4][1].map.op.to_dense()
assert f"{relative(A4, average_jacobian(0.4)):.3f}" == "0.035"
assert f"{relative(A4, average_jacobian(0.0)):.2f}" == "0.26"
assert f"{relative(average_jacobian(0.4), average_jacobian(0.0)):.2f}" == "0.22"
assert f"{relative(average_jacobian(0.1), average_jacobian(0.0)):.3f}" == "0.011"
assert relative(A1, average_jacobian(0.1)) < 0.02
assert relative(A1, average_jacobian(0.0)) < 0.02
rms4 = float(jnp.sqrt(jnp.mean(fits[0.4][1].residuals ** 2)))
rms1 = float(jnp.sqrt(jnp.mean(fits[0.1][1].residuals ** 2)))
assert f"{rms4 / 0.02:.1f}" == "7.1" and 0.15 < rms1 / 0.02 < 0.25
assert float(jnp.abs(fit3.residuals).max()) < 1e-12
assert float(jnp.abs(post.mean("u") - with_omega).max()) < 1e-10
assert f"{float(jnp.abs(post.mean('u') - without_omega).max()):.2f}" == "0.06"
assert int(jnp.argmax(jnp.abs(post.mean('u') - without_omega))) == 0
