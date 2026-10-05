# %% [markdown]
# # Ensemble Kalman inversion
#
# Calibrate the two parameters of a decay model from noisy data, running the
# same driver first in its sampling form and then in its optimization form.
#
# ## When to use this
#
# Ensemble Kalman inversion (EKI) estimates the parameters of a forward model
# using only evaluations of the model, never its derivatives, and evaluates
# the whole ensemble at once. That suits expensive or black-box simulators.
# Run to tempering level $\beta = 1$ it approximates the Bayesian posterior;
# in the limit of many particles the approximation is exact for a linear
# model with a Gaussian prior, and biased otherwise
# [@iglesias2013; @ernst2015]. Repeated with unit steps and stopped by the
# discrepancy principle it becomes a derivative-free, regularized optimizer
# [@iglesias2016].
#
# ## Setup
#
# The forward model maps parameters $u = (u_0, u_1)$, an amplitude and a
# decay rate, to an exponential decay observed at the $N = 12$ times
# $t_i = i/4$, $i = 1, \dots, 12$, evenly spaced on $(0, 3]$:
#
# $$
# G(u)_i = u_0\, e^{-u_1 t_i}, \qquad
# y = G(u^\dagger) + e, \quad e \sim \mathcal N(0, R), \quad R = 0.02^2 I_{12},
# $$
#
# with $u^\dagger = (2, 1.5)$ and prior
# $\pi_0 = \mathcal N\big((1, 1),\ I_2\big)$.
# EKI moves an ensemble through the tempered targets
#
# $$
# \pi_\beta(u) \propto \pi_0(u)\, e^{-\beta \Phi(u)}, \qquad
# \Phi(u) = \tfrac12 \lVert R^{-1/2}(y - G(u)) \rVert^2 ,
# $$
#
# and a step from $\beta$ to $\beta + \delta$ is one ensemble Kalman update
# with noise covariance $R/\delta$.

# %%
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, toy
from enskit.algorithms import eki

problem = toy.exponential_decay()   # prior over "u", forward model, noise_cov, y
print("times:", problem.times)
print("u_true:", problem.u_true)

# %% [markdown]
# ## Sampling form
#
# The schedule picks each increment $\delta$ so that the importance weights
# $w_j \propto e^{-\delta \Phi(u_j)}$ keep an effective sample size of half the
# ensemble [@jasra2011], but never below a floor of $10^{-3}$, and stops at
# $\beta = 1$. The update rule is
# Matheron's: each particle moves by
#
# $$
# u_j \mapsto u_j + K(y - g_j - e_j), \qquad
# K = \hat C_{ug} \big(\hat C_{gg} + R/\delta\big)^{-1},
# \quad e_j \sim \mathcal N(0, R/\delta),
# $$
#
# with $g_j = G(u_j)$, $\hat C_{ug}$ and $\hat C_{gg}$ the particles' sample
# covariances, and a fresh noise draw $e_j$ for each particle [@burgers1998].

# %%
state = eki.EKIState.from_prior(jax.random.key(0), problem.prior, n_particles=64)
result = eki.run(
    state, problem.forward, problem.y, problem.noise_cov,
    update_rule=kalman.Matheron(),
    schedule=eki.AdaptiveESSSchedule(ess_fraction=0.5),
)
print(f"{result.status}: {result.n_completed_steps} steps, "
      f"{result.n_evaluations} evaluations, beta = {float(result.beta):.3f}")
print("posterior mean:", result.mean("u"))

# %% [markdown]
# ## Optimization form
#
# Unit increments, repeated, drive the ensemble toward minimizers of
# $\Phi$. The run stops by the discrepancy principle, at the first evaluation
# where
#
# $$
# \lVert R^{-1/2}(y - \bar g) \rVert^2 \le \tau^2 N, \qquad \tau = 1,
# $$
#
# with $\bar g$ the mean of the particles' predictions $g_j$ [@iglesias2016].
# The update rule is the deterministic symmetric square root
# [@bishop2001; @tippett2003].

# %%
fit = eki.run(
    state, problem.forward, problem.y, problem.noise_cov,
    update_rule=kalman.SymmetricSquareRoot(),
    schedule=eki.FixedSchedule.constant(1.0, n_steps=50),
    stop=eki.DiscrepancyStop(tau=1.0),
)
print(f"{fit.status}: {fit.n_completed_steps} steps, {fit.n_evaluations} evaluations")
print("ensemble mean:", fit.mean("u"))

# %% [markdown]
# ## What to notice
#
# The two forms differ only in the schedule and the stopping rule, not in the
# driver. The sampling run takes seven steps to reach $\beta = 1$, one
# evaluation each, and its ensemble approximates the posterior. Its first
# step is the floor, since at the prior the misfits are so large that even
# that step takes the effective sample size below half the ensemble. The
# optimization run stops after three evaluations: two steps, then the
# evaluation at which the fit is as good as the noise allows, whose update
# is not taken. Both ensemble means lie within 0.03 of $u^\dagger$ in each
# coordinate.

# %%
# checks
assert result.status == eki.SCHEDULE_EXHAUSTED
assert abs(float(result.beta) - 1.0) < 1e-12
assert result.n_completed_steps == 7 and result.n_evaluations == 7
assert float(result.stacked.increment[0]) == 1e-3
assert float(result.stacked.ess[0]) < 32
assert fit.status == eki.STOPPING_RULE
assert fit.n_completed_steps == 2 and fit.n_evaluations == 3
assert jnp.all(jnp.abs(result.mean("u") - problem.u_true) < 0.03)
assert jnp.all(jnp.abs(fit.mean("u") - problem.u_true) < 0.03)
