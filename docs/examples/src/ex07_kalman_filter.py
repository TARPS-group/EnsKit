# %% [markdown]
# # An exact Kalman filter
#
# Filter a three-dimensional linear state-space model with the operations of
# `Gaussian` alone, and compute the log likelihood of the data along the way.
#
# ## When to use this
#
# When the transition and the observation model are both linear and every
# noise term is Gaussian, the filtering distributions are Gaussian and the
# Kalman recursion computes them exactly [@kalman1960]. No ensemble is needed.
# The operations used here (`pushforward` through a `Linear` map, `add_noise`,
# `condition` and `log_density`) are the same ones an ensemble method uses on
# its moment-matched approximation, so this level of the API also serves to
# build exact reference solutions for ensemble filters.
#
# ## Setup
#
# The state $x_t \in \mathbb R^3$ evolves linearly and is observed in two
# linear combinations at each of $T = 20$ times:
#
# $$
# x_0 \sim \mathcal N(0, I_3), \qquad
# x_t = A x_{t-1} + w_t, \quad w_t \sim \mathcal N(0, Q), \qquad
# y_t = H x_t + e_t, \quad e_t \sim \mathcal N(0, R),
# $$
#
# for $t = 1, \dots, 20$, with $Q = 0.3^2 I_3$, $R = 0.5^2 I_2$,
# $A = 0.95\, U$ for a random orthogonal $U$, and $H \in \mathbb R^{2 \times 3}$
# with independent $\mathcal N(0, 1/3)$ entries. These are the defaults of
# `toy.linear_state_space`, which draws $A$, $H$, a true trajectory and the
# observations $y_1, \dots, y_{20}$ from a fixed seed. Write $m_t, P_t$ for the
# mean and covariance of $x_t \mid y_{1:t}$, with $m_0 = 0$ and $P_0 = I_3$. One
# step of the recursion is
#
# $$
# \begin{aligned}
# m_t^- &= A m_{t-1}, & P_t^- &= A P_{t-1} A^\top + Q, \\
# S_t &= H P_t^- H^\top + R, & K_t &= P_t^- H^\top S_t^{-1}, \\
# m_t &= m_t^- + K_t (y_t - H m_t^-), & P_t &= P_t^- - K_t S_t K_t^\top .
# \end{aligned}
# $$
#
# The log likelihood is the sum of the one-step predictive densities,
#
# $$
# \log p(y_{1:T}) = \sum_{t=1}^{T} \log \mathcal N\big(y_t;\ H m_t^-,\ S_t\big).
# $$

# %%
import jax.numpy as jnp
import jax.scipy.stats as st

import enskit
from enskit import maps, toy

problem = toy.linear_state_space()   # state_dim=3, data_dim=2, n_times=20
print(problem)

# %% [markdown]
# ## The filter
#
# Each pass of the loop forms the predictive $x_t \mid y_{1:t-1}$, adds the
# block $y_t$ to build the joint Gaussian of $(x_t, y_t)$, reads off the
# density of the $y$ marginal at the observed value, and conditions on it.
#
# A `Gaussian` stores its covariance as $F F^\top$ plus independent per-block
# terms. Pushing $x$ through a linear map moves the independent term of $x$
# into the shared factor $F$. The initial covariance contributes three columns,
# and each absorbed transition-noise term $Q$ adds three more, so without
# intervention the latent width would reach $3 + 3T = 63$ after 20 steps.
# `compress` re-factors $F$ to a width of at most the total dimension of the
# remaining blocks, here 3.

# %%
belief = problem.initial
log_likelihood = 0.0
filtered = []
for y in problem.observations:
    belief = (belief
              .pipe(maps.pushforward, problem.transition, inputs="x", output="x")
              .pipe(maps.pushforward, maps.AdditiveNoise(problem.transition_noise),
                    inputs="x", output="x"))
    joint = (belief
             .pipe(maps.pushforward, problem.observe, inputs="x", output="y")
             .add_noise(y=problem.noise_cov))
    log_likelihood += joint.log_density(y=y)
    belief = joint.condition(y=y).compress()
    filtered.append(belief)

print(f"log likelihood: {float(log_likelihood):.2f}")
print(f"latent width after {problem.n_times} steps: {belief.latent_dim}")

# %% [markdown]
# ## What to notice
#
# The loop never forms a Kalman gain. The predictive density of $y_t$ comes from
# `log_density` on the joint, and the update comes from `condition`; together
# they give a log likelihood of $-44.21$ over the 20 observations. After
# `compress` the latent width stays at 3, the state dimension, instead of
# growing with the number of steps.
#
# `toy.LinearStateSpace.exact_filter` runs this same loop and returns the
# filtering distributions and the one-step log evidences, so a test of an
# ensemble filter on this problem can compare against it in one line. The
# checks below compare the loop with it, then run the dense recursion from the
# setup, and confirm that the means, the covariances and the log likelihood
# agree to within $10^{-9}$.

# %%
# checks
assert round(float(log_likelihood), 2) == -44.21
assert belief.latent_dim == 3

# without compress, each step widens the factor by 3: checked over the first 8 steps
uncompressed = problem.initial
for t, y in enumerate(problem.observations[:8], start=1):
    uncompressed = (uncompressed
                    .pipe(maps.pushforward, problem.transition, inputs="x", output="x")
                    .pipe(maps.pushforward, maps.AdditiveNoise(problem.transition_noise),
                          inputs="x", output="x")
                    .pipe(maps.pushforward, problem.observe, inputs="x", output="y")
                    .add_noise(y=problem.noise_cov)
                    .condition(y=y))
    assert uncompressed.latent_dim == 3 + 3 * t
assert 3 + 3 * problem.n_times == 63

reference, log_evidence = problem.exact_filter()
assert jnp.allclose(log_likelihood, jnp.sum(log_evidence), rtol=0, atol=1e-9)
for ours, theirs in zip(filtered, reference, strict=True):
    assert jnp.allclose(ours.mean("x"), theirs.mean("x"), rtol=0, atol=1e-9)
    assert jnp.allclose(ours.cov("x").to_dense(), theirs.cov("x").to_dense(),
                        rtol=0, atol=1e-9)

A, H = problem.transition.op.to_dense(), problem.observe.op.to_dense()
Q, R = problem.transition_noise.to_dense(), problem.noise_cov.to_dense()
m, P, ll = jnp.zeros(3), jnp.eye(3), 0.0
for t, y in enumerate(problem.observations):
    m, P = A @ m, A @ P @ A.T + Q
    S = H @ P @ H.T + R
    ll += st.multivariate_normal.logpdf(y, H @ m, S)
    K = P @ H.T @ jnp.linalg.inv(S)
    m, P = m + K @ (y - H @ m), P - K @ S @ K.T
    assert jnp.allclose(filtered[t].mean("x"), m, rtol=0, atol=1e-9)
    assert jnp.allclose(filtered[t].cov("x").to_dense(), P, rtol=0, atol=1e-9)
assert jnp.allclose(log_likelihood, ll, rtol=0, atol=1e-9)
