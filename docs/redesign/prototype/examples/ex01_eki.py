"""Example 1 - EKI, the high-level way."""
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, toy
from enskit.algorithms import eki

problem = toy.exponential_decay()   # prior over "u", forward model, noise_cov, y

# Sampling form: adaptive ladder to level 1, stochastic (pathwise) update.
state = eki.EKIState.from_prior(jax.random.key(0), problem.prior, n_particles=64)
result = eki.run(
    state, problem.forward, problem.y, problem.noise_cov,
    update_rule=kalman.Matheron(),
    schedule=eki.AdaptiveESSSchedule(ess_fraction=0.5),
)
print(result.status, result.n_evaluations, "evaluations")
print("posterior mean", result.ensemble.mean("u"), "truth", problem.u_true)

# Optimization form: unit steps, stop on the discrepancy principle.
fit = eki.run(
    state, problem.forward, problem.y, problem.noise_cov,
    update_rule=kalman.SymmetricSquareRoot(),
    schedule=eki.FixedSchedule.constant(1.0, n_steps=50),
    stop=eki.DiscrepancyStop(tau=1.0),
)
print(fit.status, "after", fit.n_evaluations, "evaluations:", fit.ensemble.mean("u"))

# ---- checks
assert result.status == "schedule_exhausted"
assert abs(float(result.state.beta) - 1.0) < 1e-12
assert fit.status == "stopping_rule"
assert jnp.allclose(fit.ensemble.mean("u"), problem.u_true, atol=0.2)
