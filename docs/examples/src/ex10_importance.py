# %% [markdown]
# # EKI as an importance-sampling proposal
#
# Run EKI on a two-parameter decay model, fit a widened Gaussian to its
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
# The model, data and prior are those of `toy.exponential_decay`, as in
# Example 1: $u = (a, r)$, twelve data at the times $t_i = i/4$,
# $i = 1, \dots, 12$, and
#
# $$
# G(u)_i = a\, e^{-r t_i}, \qquad
# y = G(u^\dagger) + e, \quad e \sim \mathcal N(0, R), \quad R = 0.02^2 I_{12},
# \qquad \pi_0 = \mathcal N\big((1, 1),\ I_2\big),
# $$
#
# with $u^\dagger = (2, 1.5)$. The target is the posterior
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
from enskit.algorithms import eki
from enskit.distribution import Ensemble, effective_sample_size, resample, reweight

problem = toy.exponential_decay()

# %% [markdown]
# ## The EKI run
#
# This is the sampling form of Example 1 [@iglesias2013]: 64 particles from
# the prior, an adaptive schedule to $\beta = 1$ that keeps the effective
# sample size of each step's tempering weights at half the ensemble
# [@jasra2011], and the stochastic `Matheron` update [@burgers1998].

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
ess = effective_sample_size(weighted)
print(f"ESS: {float(ess):.0f} of {weighted.n_particles}")

# %% [markdown]
# Systematic resampling [@kitagawa1996] turns the weighted ensemble back into
# 64 equally weighted particles, the form a later ensemble Kalman update
# expects.

# %%
refined = resample(jax.random.key(2), weighted, n_particles=64).drop("g")

# %% [markdown]
# ## A reference solution
#
# With two parameters the posterior mean can be computed by quadrature. On a
# grid $u_{kl} = (a_k, r_l)$ of $401 \times 401$ points spanning
# $[1.7, 2.25] \times [1.2, 1.75]$, at least seven posterior standard
# deviations on each side of the mean in each coordinate,
#
# $$
# \mathbb E_\pi[u] \approx \sum_{k, l} p_{kl}\, u_{kl}, \qquad
# p_{kl} = \frac{\pi_0(u_{kl})\, e^{-\Phi(u_{kl})}}
#               {\sum_{k', l'} \pi_0(u_{k'l'})\, e^{-\Phi(u_{k'l'})}} .
# $$
#
# The same function `log_posterior` gives the weights, applied to an
# `Ensemble` of the grid points.

# %%
a, r = jnp.linspace(1.7, 2.25, 401), jnp.linspace(1.2, 1.75, 401)
grid = Ensemble(u=jnp.stack(jnp.meshgrid(a, r, indexing="ij"), axis=-1).reshape(-1, 2))
grid = maps.pushforward(grid, problem.forward, inputs="u", output="g")
log_p = log_posterior(grid)
p = jnp.exp(log_p - jax.scipy.special.logsumexp(log_p))
reference = p @ grid["u"]

err_eki = jnp.abs(result.ensemble.mean("u") - reference).max()
err_is = jnp.abs(weighted.mean("u") - reference).max()
print(f"posterior mean: ({float(reference[0]):.4f}, {float(reference[1]):.4f})")
print(f"largest error of the mean: EKI {float(err_eki):.4f}, IS {float(err_is):.4f}")

# %% [markdown]
# ## What to notice
#
# The widened proposal keeps an effective sample size of about 2640 of 4000
# draws. Measured as the largest coordinate error of the posterior mean
# against the quadrature, EKI is off by about 0.0024 and importance sampling
# by about 0.0004. The posterior standard deviations are near 0.03, so
# importance sampling's error is within the Monte Carlo error expected from
# that effective sample size. EKI's error, several times larger, combines the
# bias of the method with the Monte Carlo error of 64 particles; importance
# sampling removes the first and, with more draws, shrinks the second. The
# grid carries a negligible share of the posterior mass at its
# edges, so the quadrature error is far below either.

# %%
# checks
assert round(float(ess), -1) == 2640
assert 0.0022 < err_eki < 0.0027
assert 0.0003 < err_is < 0.00045

post_sd = jnp.sqrt(p @ (grid["u"] - reference) ** 2)
assert jnp.all((post_sd > 0.025) & (post_sd < 0.04))        # "near 0.03"
assert jnp.all(jnp.abs(weighted.mean("u") - reference) < 2 * post_sd / jnp.sqrt(ess))
assert err_eki > 4 * err_is                                 # "several times larger"
lo, hi = jnp.array([1.7, 1.2]), jnp.array([2.25, 1.75])
assert jnp.all(jnp.minimum(reference - lo, hi - reference) > 7 * post_sd)
edges = p.reshape(401, 401)
assert edges[jnp.array([0, -1])].sum() + edges[:, jnp.array([0, -1])].sum() < 1e-10
assert refined.names == ("u",) and not refined.is_weighted and refined.n_particles == 64
