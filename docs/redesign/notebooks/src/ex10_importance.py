# %% [markdown]
# # EKI as an importance-sampling proposal
#
# Run EKI on the decay model of Example 1, fit a widened Gaussian to its
# ensemble, and use that Gaussian as the proposal for importance sampling of
# the exact posterior.
#
# ## When to use this
#
# For a nonlinear forward model, the EKI ensemble at $\beta = 1$ is a biased
# approximation of the posterior, and the bias does not vanish as the ensemble
# grows [@ernst2015]. When it is close to the posterior it can serve as an
# importance-sampling proposal, and importance weights then remove the bias
# at the cost of more forward-model evaluations. This suits problems with
# a few parameters, where a few thousand evaluations are affordable and
# importance sampling from the prior would waste most of them.
#
# ## Setup
#
# The model, data and prior are those of Example 1: $u = (a, r)$, ten data
# $y = G(u^\dagger) + e$ with $G(u)_i = a\, e^{-r t_i}$, $R = 0.05^2 I_{10}$, and
# prior density $\pi_0$. The target is the posterior
#
# $$
# \pi(u) \propto \pi_0(u)\, e^{-\Phi(u)}, \qquad
# \Phi(u) = \tfrac12 \lVert R^{-1/2}(y - G(u)) \rVert^2 .
# $$
#
# With $\hat m$ and $\hat C$ the mean and sample covariance of the 64-particle
# EKI ensemble, the proposal is $q = \mathcal N(\hat m,\ 1.5^2\, \hat C)$. For
# $M = 4000$ draws $u_i \sim q$, the self-normalized estimator of a posterior
# expectation and its effective sample size [@kong1994] are
#
# $$
# \mathbb E_\pi[f] \approx \sum_{i=1}^{M} \bar w_i\, f(u_i), \qquad
# \bar w_i = \frac{w_i}{\sum_k w_k}, \quad
# w_i = \frac{\pi_0(u_i)\, e^{-\Phi(u_i)}}{q(u_i)}, \qquad
# \mathrm{ESS} = \frac{1}{\sum_i \bar w_i^2}.
# $$

# %%
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, maps, toy
from enskit.distribution import effective_sample_size, resample, reweight
from enskit.algorithms import eki

problem = toy.exponential_decay()

# %% [markdown]
# ## The EKI run
#
# This is the sampling form of Example 1: an adaptive schedule to $\beta = 1$
# and the stochastic `Matheron` update.

# %%
state = eki.EKIState.from_prior(jax.random.key(0), problem.prior, n_particles=64)
result = eki.run(state, problem.forward, problem.y, problem.noise_cov,
                 update_rule=kalman.Matheron(), schedule=eki.AdaptiveESSSchedule())

# %% [markdown]
# ## The proposal and the weights
#
# `inflate_multiplicative` scales the anomalies by 1.5, and `project` returns
# the moment-matched Gaussian, which is $q$. A proposal narrower than the
# target gives importance weights with heavy tails, so the proposal is
# widened. Defensive mixtures address the same risk by adding a wide component
# to the proposal [@hesterberg1995]. Each draw costs one forward-model
# evaluation, and `reweight` attaches $\log w_i$ to the ensemble.

# %%
proposal = kalman.inflate_multiplicative(result.ensemble, 1.5).project()
draws = proposal.sample(jax.random.key(1), n_particles=4000)
draws = maps.pushforward(draws, problem.forward, inputs="u", output="g")


def log_posterior(ens):
    r = problem.noise_cov.whiten(problem.y - ens["g"])
    return problem.prior.log_density(u=ens["u"]) - 0.5 * jnp.sum(r * r, axis=1)


weighted = reweight(draws, log_posterior(draws) - proposal.log_density(u=draws["u"]))
print("ESS:", effective_sample_size(weighted), "of", weighted.n_particles)
print("EKI mean:", result.ensemble.mean("u"), " IS mean:", weighted.mean("u"))

# %% [markdown]
# Systematic resampling turns the weighted ensemble back into 64 equally
# weighted particles, the form a later ensemble Kalman update expects.

# %%
refined = resample(jax.random.key(2), weighted, n_particles=64).drop("g")

# %% [markdown]
# ## A reference solution
#
# For comparison, importance sampling from the prior with 400,000 draws gives
# a reference posterior mean.

# %%
ref = problem.prior.sample(jax.random.key(3), n_particles=400_000)
ref = maps.pushforward(ref, problem.forward, inputs="u", output="g")
ref = reweight(ref, log_posterior(ref) - problem.prior.log_density(u=ref["u"]))
print("reference mean:", ref.mean("u"))
err_is = jnp.abs(weighted.mean("u") - ref.mean("u")).max()
err_eki = jnp.abs(result.ensemble.mean("u") - ref.mean("u")).max()
print("error vs reference: EKI", err_eki, " IS", err_is)

# %% [markdown]
# ## What to notice
#
# The widened proposal keeps an effective sample size of 2651 of 4000 draws.
# Measured as the largest coordinate error of the posterior mean against the
# reference, EKI is off by 0.0093 and importance sampling by 0.0020. The
# reference is itself a Monte Carlo estimate, so each number combines the
# error of the method with the error of the reference.

# %%
# checks
assert err_is < err_eki
assert refined.names == ("u",) and not refined.is_weighted
