# %% [markdown]
# # The two updates as probabilistic operations
#
# Build Matheron's update and the square-root update from Gaussian
# operations, first on an exact Gaussian and then on the Gaussian fitted to
# an ensemble.
#
# ## When to use this
#
# This level of the API is for reading, checking or extending an update rule.
# Each rule is a composition of a few operations on Gaussians and ensembles:
# pushforward, conditioning, a conditional map, and realizing particles from
# a fitted Gaussian. Matheron's rule turns joint samples into conditional
# samples by a linear transport [@journel1978; @wilson2021], and the
# square-root update replaces the particles with ones whose sample moments
# equal those of the conditioned fit [@bishop2001; @tippett2003].
#
# ## Setup
#
# The problem is the linear-Gaussian one of the previous example:
# $u \in \mathbb R^4$ with prior $\mathcal N(0, C)$, $g = A u$ with
# $A \in \mathbb R^{6 \times 4}$, and $y = g + e$ with
# $e \sim \mathcal N(0, R)$, $R = 0.3^2 I_6$, observed at the value $y^*$.
# Write $x = (u, g)$ for the blocks to be updated. For any jointly Gaussian
# $(x, y)$, Matheron's rule states that
#
# $$
# x + C_{xy} C_{yy}^{-1} (y^* - y) \sim p(x \mid y = y^*)
# \quad \text{when} \quad (x, y) \sim p(x, y),
# $$
#
# so the conditional is reached by moving joint samples with the gain
# $K = C_{xy} C_{yy}^{-1}$, here with $C_{xy} = (C A^\top,\ A C A^\top)$ and
# $C_{yy} = A C A^\top + R$.

# %%
import jax
import jax.numpy as jnp

import enskit
from enskit import maps, toy

problem = toy.linear_gaussian(parameter_dim=4, data_dim=6)

# %% [markdown]
# ## Exact Gaussian algebra
#
# The joint over $(u, g, y)$ is built by two pushforwards of the prior.
# `condition` gives the exact posterior over $(u, g)$, and
# `conditional_map("y")` gives the transport $x \mapsto x + K(y^* - y)$. The
# map is built from the block name alone; the value $y^*$ is passed when it is
# called on 200,000 joint samples.

# %%
joint = (problem.prior
         .pipe(maps.pushforward, problem.forward, inputs="u", output="g")
         .pipe(maps.pushforward, maps.AdditiveNoise(problem.noise_cov), inputs="g", output="y"))
posterior = joint.condition(y=problem.y)                    # Gaussian over u, g
transport = joint.conditional_map("y")                           # x -> x + K (y* - y)
samples = transport(joint.sample(jax.random.key(0), 200_000), y=problem.y)

# %% [markdown]
# ## The fitted Gaussian
#
# An ensemble of $J = 32$ prior samples, with $g_j = A u_j$, is replaced by its
# moment-matched Gaussian with `project`. That Gaussian is member-aligned: its
# covariance factor has one column per particle, $(x_j - \bar x)/\sqrt{J-1}$.
# Pushing $g$ through the noise adds $y$ with covariance $\hat C_{gg} + R$,
# where hats denote sample moments.

# %%
ens = problem.prior.sample(jax.random.key(1), n_particles=32)
ens = maps.pushforward(ens, problem.forward, inputs="u", output="g")
ens_joint = (ens.project()                                       # member-aligned
             .pipe(maps.pushforward, maps.AdditiveNoise(problem.noise_cov),
                   inputs="g", output="y"))

# %% [markdown]
# ## Square root: condition, then realize
#
# Conditioning the fit on $y = y^*$ shifts its mean by $\hat K(y^* - \bar g)$
# and multiplies its factor on the right by $T = (I_J + S S^\top)^{-1/2}$,
# where row $j$ of $S$ is $R^{-1/2}(g_j - \bar g)/\sqrt{J-1}$. Realizing reads
# one particle off each column:
#
# $$
# x_j^{a} = \bar x + \hat K (y^* - \bar g) + \sum_{i=1}^{J} (x_i - \bar x)\, T_{ij} .
# $$
#
# `square_root_map("y")` packages these two operations as a map that is built
# from the block name and called with the value.

# %%
# square root: realize after condition, built once as a map and called with the value
sqrt_map = ens_joint.square_root_map("y")
sqrt_particles = sqrt_map(y=problem.y)
assert jnp.allclose(sqrt_particles["u"],
                    ens_joint.condition(y=problem.y).realize_particles()["u"])

# %% [markdown]
# ## Matheron: realize, then transport
#
# Realizing the fit without the noise term on $y$ returns the original
# particles, with $y_j = g_j$. The conditional map then moves each one by
# $\hat K(y^* - g_j - e_j)$, drawing $e_j \sim \mathcal N(0, R)$ as a standard
# normal in whitened coordinates.

# %%
# Matheron: transport after realize (the map draws the noise, whitened)
particles = ens_joint.realize_particles(exclude_block_covs=("y",))   # y_j = g_j here
pert_particles = ens_joint.conditional_map("y")(particles, y=problem.y,
                                                key=jax.random.key(2))

# %% [markdown]
# ## What to notice
#
# The checks below confirm three things. The 200,000 transported samples
# match the exact posterior mean and covariance of $u$ to within $10^{-2}$.
# The square-root particles reproduce the conditioned fit's mean and
# covariance to within $10^{-12}$, because realizing preserves the moments of
# a member-aligned Gaussian. The Matheron particles carry the blocks `u` and
# `g`, with the conditioned block `y` dropped.

# %%
# checks
assert jnp.allclose(samples.mean("u"), posterior.mean("u"), atol=1e-2)
assert jnp.allclose(samples.cov("u").to_dense(), posterior.cov("u").to_dense(), atol=1e-2)
fit = ens_joint.condition(y=problem.y)
assert jnp.allclose(sqrt_particles.mean("u"), fit.mean("u"), atol=1e-12)
assert jnp.allclose(sqrt_particles.cov("u").to_dense(), fit.cov("u").to_dense(), atol=1e-12)
assert pert_particles.names == ("u", "g")
print("Matheron samples match the exact conditional; realize . condition is exact on the fit")
