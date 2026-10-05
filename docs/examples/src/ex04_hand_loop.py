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
# $u = (u_0, u_1)$ map to $G(u)_i = u_0\, e^{-u_1 t_i}$ at the $N = 12$ times
# $t_i = i/4$, $i = 1, \dots, 12$, the data are $y = G(u^\dagger) + e$ with
# $e \sim \mathcal N(0, R)$, $R = 0.02^2 I_{12}$ and $u^\dagger = (2, 1.5)$,
# and the prior is $\pi_0 = \mathcal N\big((1, 1),\ I_2\big)$.
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
# (\delta_0, \dots, \delta_4)
#   = \big(\tfrac1{16}, \tfrac1{16}, \tfrac18, \tfrac14, \tfrac12\big),
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
# The particles are drawn from the prior with the key split as
# `eki.EKIState.from_prior` splits it: the first half draws the particles,
# and the second becomes the state's key, which the deterministic update
# below never uses.

# %%
key_sample, key_state = jax.random.split(jax.random.key(0))
ens = problem.prior.sample(key_sample, n_particles=64)      # Ensemble over "u"

# %% [markdown]
# ## The loop
#
# Since $e^{-\delta \Phi(u)}$ is, up to a constant, a Gaussian likelihood
# for $y$ with covariance $R/\delta$, the step from $\beta$ to $\beta + \delta$
# conditions on $g = y$ with noise covariance $R/\delta$. Each step evaluates
# $g_j = G(u_j)$ for all particles at once, prints the mean of $\Phi(u_j)$ and
# the standard deviation of each parameter across the particles, and then
# applies the symmetric square-root update [@bishop2001; @tippett2003].

# %%
mean_misfits, spreads = [], []
for step, dbeta in enumerate(increments):
    # evaluate: one call of the forward model, with every particle
    ens = maps.pushforward(ens, problem.forward, inputs="u", output="g")
    misfit = 0.5 * jnp.sum(problem.noise_cov.whiten(problem.y - ens["g"]) ** 2, axis=1)
    spread = jnp.std(ens["u"], axis=0)
    mean_misfits.append(misfit.mean())
    spreads.append(spread)
    print(f"step {step}: increment {dbeta:.4f}, mean misfit {float(misfit.mean()):9.2f}, "
          f"spread ({float(spread[0]):.3f}, {float(spread[1]):.3f})")
    # update: condition on g with the tempered noise covariance R / dbeta;
    # the result is an Ensemble over "u" only
    ens = kalman.update(ens, g=problem.y, noise={"g": problem.noise_cov / dbeta},
                        update_rule=kalman.SymmetricSquareRoot())

final_mean = ens.mean("u")
final_spread = jnp.std(ens["u"], axis=0)
print(f"hand-written loop: mean ({float(final_mean[0]):.3f}, "
      f"{float(final_mean[1]):.3f}), "
      f"spread ({float(final_spread[0]):.3f}, {float(final_spread[1]):.3f})")

# %% [markdown]
# ## What to notice
#
# The mean misfit falls from about 593,000 at the prior to 7.15 before the
# last step, near $N/2 = 6$, the mean of $\Phi$ at the parameters that
# generated the data. The spread of $(u_0, u_1)$ shrinks from about
# $(1.01, 1.01)$ at the prior to $(0.055, 0.051)$ before the last step and
# $(0.038, 0.034)$ after it. The final mean is $(1.979, 1.474)$, close to
# $u^\dagger = (2, 1.5)$. The check
# below runs `eki.run` with `eki.FixedSchedule(increments)` from
# `eki.EKIState.from_prior` with the same key, and finds the same final
# particles to within $10^{-12}$ and the same mean misfit at every step.

# %%
# checks: the numbers stated above
assert round(float(mean_misfits[0]), -3) == 593_000
assert f"{float(mean_misfits[-1]):.2f}" == "7.15"
assert [f"{float(s):.2f}" for s in spreads[0]] == ["1.01", "1.01"]
assert [f"{float(s):.3f}" for s in spreads[-1]] == ["0.055", "0.051"]
assert [f"{float(s):.3f}" for s in final_spread] == ["0.038", "0.034"]
assert [f"{float(m):.3f}" for m in final_mean] == ["1.979", "1.474"]
# checks: identical to the driver with the same fixed schedule
state = eki.EKIState.from_prior(jax.random.key(0), problem.prior, n_particles=64)
result = eki.run(state, problem.forward, problem.y, problem.noise_cov,
                 update_rule=kalman.SymmetricSquareRoot(),
                 schedule=eki.FixedSchedule(increments))
assert result.n_evaluations == len(increments)
assert float(result.beta) == 1.0
assert jnp.allclose(result.ensemble["u"], ens["u"], rtol=0, atol=1e-12)
assert jnp.allclose(result.stacked.misfit_mean, jnp.array(mean_misfits), rtol=1e-12)
print("driver agrees:", result.mean("u"))
