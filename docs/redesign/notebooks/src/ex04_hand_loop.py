# %% [markdown]
# # Ensemble Kalman inversion written by hand
#
# Write the EKI loop as explicit calls, one evaluation of the forward model
# and one update per step, and check that it reproduces the driver.
#
# ## When to use this
#
# The driver `eki.run` covers the common schedules and stopping rules. A
# hand-written loop is the right tool when a step needs something the driver
# does not offer, such as logging intermediate ensembles, changing the data
# between steps, or inserting a custom operation between the evaluation and
# the update. Each step is one ensemble Kalman update with a scaled noise
# covariance [@iglesias2013]. The ensemble smoother with multiple data
# assimilation (ES-MDA) has the same structure with a fixed schedule
# [@emerick2013].
#
# ## Setup
#
# The problem is the decay model of the first example. Parameters
# $u = (a, r)$ map to $G(u)_i = a\, e^{-r t_i}$ at ten times $t_i$ equally
# spaced on $[0.1, 2]$, the data are $y = G(u^\dagger) + e$ with
# $e \sim \mathcal N(0, R)$, $R = 0.05^2 I_{10}$ and $u^\dagger = (1.5, 1.2)$,
# and the prior is $\mathcal N\big((1, 0.5),\ \operatorname{diag}(0.5, 0.5)\big)$.
# The loop moves $J = 64$ particles through the tempered targets
#
# $$
# \pi_{\beta}(u) \propto \pi_0(u)\, e^{-\beta \Phi(u)}, \qquad
# \Phi(u) = \tfrac12 \lVert R^{-1/2}(y - G(u)) \rVert^2 ,
# $$
#
# at levels $\beta_0 = 0$ and $\beta_{k+1} = \beta_k + \delta_k$, with the
# increments
#
# $$
# (\delta_0, \dots, \delta_4) = \big(\tfrac1{16}, \tfrac1{16}, \tfrac18, \tfrac14, \tfrac12\big),
# \qquad \textstyle\sum_k \delta_k = 1 .
# $$
#
# The increments are powers of two, so their floating-point sum is exactly 1
# and the last step ends at $\beta = 1$.

# %%
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, maps, toy
from enskit.algorithms import eki

problem = toy.exponential_decay()
increments = (1 / 16, 1 / 16, 1 / 8, 1 / 4, 1 / 2)      # binary-exact: sums to exactly 1

# %% [markdown]
# ## The initial ensemble
#
# The particles are drawn from the prior, with the key split the same way
# the driver's state is built below.

# %%
key = jax.random.key(0)
key, sub = jax.random.split(key)
ens = problem.prior.sample(sub, n_particles=64)            # Ensemble over "u"

# %% [markdown]
# ## The loop
#
# Since $e^{-\delta \Phi(u)}$ is, up to a constant, a Gaussian likelihood
# for $y$ with covariance $R/\delta$, the step from $\beta$ to $\beta + \delta$
# conditions on $g = y$ with noise covariance $R/\delta$. Each step evaluates
# $g_j = G(u_j)$ for all particles at once, prints the mean of $\Phi(u_j)$ and
# the standard deviation of each parameter across the particles, and then
# applies the symmetric square-root update.

# %%
for step, dbeta in enumerate(increments):
    # forecast: one evaluation of the forward model for the whole ensemble
    ens = maps.pushforward(ens, problem.forward, inputs="u", output="g")
    misfit = 0.5 * jnp.sum(problem.noise_cov.whiten(problem.y - ens["g"]) ** 2, axis=1)
    print(f"step {step}: increment {dbeta:.4f}, mean misfit {misfit.mean():8.2f}, "
          f"spread {jnp.std(ens['u'], axis=0)}")
    # update: condition on g with the tempered noise covariance R / dbeta
    ens = kalman.update(ens, g=problem.y, noise={"g": problem.noise_cov / dbeta},
                        update_rule=kalman.SymmetricSquareRoot())   # returns an Ensemble over "u" only

print("hand-written loop:", ens.mean("u"))

# %% [markdown]
# ## What to notice
#
# The mean misfit falls from 1826 at the prior to 5.1 before the last step,
# and the spread of $(a, r)$ shrinks from about $(0.67, 0.63)$ to
# $(0.071, 0.093)$. The final mean is $(1.480, 1.155)$, close to
# $u^\dagger = (1.5, 1.2)$. The check below runs `eki.run` with
# `eki.FixedSchedule(increments)` on the same initial particles and finds the
# same final ensemble to within $10^{-12}$.

# %%
# checks: identical to the driver with the same fixed schedule
state = eki.EKIState(problem.prior.sample(jax.random.split(jax.random.key(0))[1], 64),
                     key=jax.random.key(1))
result = eki.run(state, problem.forward, problem.y, problem.noise_cov,
                 update_rule=kalman.SymmetricSquareRoot(), schedule=eki.FixedSchedule(increments))
assert jnp.allclose(result.ensemble["u"], ens["u"], atol=1e-12)
print("driver agrees:", result.ensemble.mean("u"))
