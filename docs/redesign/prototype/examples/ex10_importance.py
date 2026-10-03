"""Example 10 - EKI, then an importance-sampling correction."""
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, maps, toy
from enskit.distribution import effective_sample_size, resample, reweight
from enskit.algorithms import eki

problem = toy.exponential_decay()

state = eki.EKIState.from_prior(jax.random.key(0), problem.prior, n_particles=64)
result = eki.run(state, problem.forward, problem.y, problem.noise_cov,
                 update_rule=kalman.Matheron(), schedule=eki.AdaptiveESSSchedule())

# Proposal: the moment-matched Gaussian of the EKI ensemble, widened for safety.
proposal = kalman.inflate_multiplicative(result.ensemble, 1.5).project()       # Gaussian over "u"
draws = proposal.sample(jax.random.key(1), n_particles=4000)
draws = maps.pushforward(draws, problem.forward, inputs="u", output="g")   # 4000 evaluations


def log_posterior(ens):  # unnormalized: prior times likelihood
    r = problem.noise_cov.whiten(problem.y - ens["g"])
    return problem.prior.log_density(u=ens["u"]) - 0.5 * jnp.sum(r * r, axis=1)


weighted = reweight(draws, log_posterior(draws) - proposal.log_density(u=draws["u"]))
print("ESS:", effective_sample_size(weighted), "of", weighted.n_particles)
print("EKI mean:", result.ensemble.mean("u"), " IS mean:", weighted.mean("u"))
refined = resample(jax.random.key(2), weighted, n_particles=64).drop("g")

# ---- check against brute-force importance sampling from the prior
ref = problem.prior.sample(jax.random.key(3), n_particles=400_000)
ref = maps.pushforward(ref, problem.forward, inputs="u", output="g")
ref = reweight(ref, log_posterior(ref) - problem.prior.log_density(u=ref["u"]))
print("reference mean:", ref.mean("u"))
err_is = jnp.abs(weighted.mean("u") - ref.mean("u")).max()
err_eki = jnp.abs(result.ensemble.mean("u") - ref.mean("u")).max()
print("error vs reference: EKI", err_eki, " IS", err_is)
assert err_is < err_eki
assert refined.names == ("u",) and not refined.is_weighted
