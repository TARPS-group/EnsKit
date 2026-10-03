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
# the approximation is exact for a linear model with a Gaussian prior and
# biased otherwise [@iglesias2013; @ernst2015]. Iterated with unit steps and
# stopped by the discrepancy principle it becomes a derivative-free,
# regularized optimizer [@iglesias2016].
#
# ## Setup
#
# The forward model maps parameters $u = (a, r)$ to an exponential decay
# observed at ten times $t_i$ equally spaced on $[0.1, 2]$:
#
# $$
# G(u)_i = a\, e^{-r t_i}, \qquad
# y = G(u^\dagger) + e, \quad e \sim \mathcal N(0, R), \quad R = 0.05^2 I_{10},
# $$
#
# with $u^\dagger = (1.5, 1.2)$ and prior
# $u \sim \mathcal N\big((1, 0.5),\ \operatorname{diag}(0.5, 0.5)\big)$.
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

# %% [markdown]
# ## Sampling form
#
# The schedule picks each increment $\delta$ so that the importance weights
# $w_j \propto e^{-\delta \Phi(u_j)}$ keep an effective sample size of half the
# ensemble [@jasra2011], and stops at $\beta = 1$. The update rule is
# Matheron's: each particle moves by $K(y - g_j - e_j)$ with a fresh noise draw
# $e_j$ [@burgers1998].

# %%
state = eki.EKIState.from_prior(jax.random.key(0), problem.prior, n_particles=64)
result = eki.run(
    state, problem.forward, problem.y, problem.noise_cov,
    update_rule=kalman.Matheron(),
    schedule=eki.AdaptiveESSSchedule(ess_fraction=0.5),
)
print(result.status, result.n_evaluations, "evaluations")
print("posterior mean", result.ensemble.mean("u"), "truth", problem.u_true)

# %% [markdown]
# ## Optimization form
#
# Unit increments, repeated, drive the ensemble toward minimizers of
# $\Phi$. The run stops by the discrepancy principle, at the first step where
# $2\Phi(\bar g) \le \tau^2 N$, with $\bar g$ the mean prediction and $N = 10$
# the data dimension [@iglesias2016]. The deterministic square-root rule is
# used here [@bishop2001].

# %%
fit = eki.run(
    state, problem.forward, problem.y, problem.noise_cov,
    update_rule=kalman.SymmetricSquareRoot(),
    schedule=eki.FixedSchedule.constant(1.0, n_steps=50),
    stop=eki.DiscrepancyStop(tau=1.0),
)
print(fit.status, "after", fit.n_evaluations, "evaluations:", fit.ensemble.mean("u"))

# %% [markdown]
# ## What to notice
#
# The two forms differ only in the schedule and the stopping rule, not in the
# driver. The sampling run spends its six evaluations reaching $\beta = 1$,
# and its ensemble approximates the posterior. The optimization run stops as
# soon as the fit is as good as the noise allows, with an ensemble mean close
# to the parameters that generated the data.

# %%
# checks
assert result.status == "schedule_exhausted"
assert abs(float(result.state.beta) - 1.0) < 1e-12
assert fit.status == "stopping_rule"
assert jnp.allclose(fit.ensemble.mean("u"), problem.u_true, atol=0.2)
