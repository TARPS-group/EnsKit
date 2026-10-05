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
# $u \in \mathbb R^4$ with prior $\mathcal N(0, C)$, $C = I_4$; $g = G u$
# with $G \in \mathbb R^{6 \times 4}$ a fixed matrix of entries drawn from
# $\mathcal N(0, 1/4)$; and $y = g + e$ with $e \sim \mathcal N(0, R)$,
# $R = 0.1^2 I_6$, observed at the value $y^*$.
# Write $x = (u, g)$ for the blocks to be updated. For any jointly Gaussian
# $(x, y)$, Matheron's rule states that
#
# $$
# x + C_{xy} C_{yy}^{-1} (y^* - y) \sim p(x \mid y = y^*)
# \quad \text{when} \quad (x, y) \sim p(x, y),
# $$
#
# so the conditional is reached by moving joint samples with the gain
# $K = C_{xy} C_{yy}^{-1}$, here with $C_{xy} = (C G^\top,\ G C G^\top)$ and
# $C_{yy} = G C G^\top + R$.

# %%
import jax
import jax.numpy as jnp

import enskit
from enskit import kalman, maps, toy

problem = toy.linear_gaussian(parameter_dim=4, data_dim=6)   # noise_std=0.1, prior_std=1
noise = maps.AdditiveNoise(problem.noise_cov)                # g -> g + e, e ~ N(0, R)

# %% [markdown]
# ## Exact Gaussian algebra
#
# The joint over $(u, g, y)$ is built by two pushforwards of the prior, the
# first through the linear map `maps.Linear(G)`, since a `Gaussian` can be
# pushed only through a map that keeps it Gaussian. `condition` gives the
# exact posterior over $(u, g)$, and `conditional_map("y")` gives the
# transport $x \mapsto x + K(y^* - y)$. The map is built from the block name
# alone; the value $y^*$ is passed when it is called on 200,000 joint
# samples.

# %%
joint = (problem.prior
         .pipe(maps.pushforward, maps.Linear(problem.G), inputs="u", output="g")
         .pipe(maps.pushforward, noise, inputs="g", output="y"))
posterior = joint.condition(y=problem.y)                         # Gaussian over u, g
transport = joint.conditional_map("y")                           # x -> x + K (y* - y)
samples = transport(joint.sample(jax.random.key(0), 200_000), y=problem.y)

post_cov = posterior.cov("u").to_dense()
mc_mean_err = jnp.abs(samples.mean("u") - posterior.mean("u")).max()
mc_cov_err = (jnp.abs(samples.cov("u").to_dense() - post_cov).max()
              / jnp.abs(post_cov).max())
print(f"largest error in the mean of u:       {float(mc_mean_err):.4f}")
print(f"largest error in the covariance of u: {100 * float(mc_cov_err):.2f}%"
      " of its largest entry")

# %% [markdown]
# ## The fitted Gaussian
#
# An ensemble of $J = 32$ prior samples, with $g_j = G u_j$, is replaced by its
# moment-matching Gaussian with `project`. That Gaussian is an
# `EnsembleGaussian`, aligned with the particles: its covariance factor has
# one column per particle, $(x_j - \bar x)/\sqrt{J-1}$. Pushing $g$ through
# the noise adds $y$ with covariance $\hat C_{gg} + R$, where hats denote
# sample moments.

# %%
ens = problem.prior.sample(jax.random.key(1), n_particles=32)
ens = maps.pushforward(ens, problem.forward, inputs="u", output="g")
ens_joint = ens.project().pipe(maps.pushforward, noise, inputs="g", output="y")
print(ens_joint)

# %% [markdown]
# ## Square root: condition, then realize
#
# Conditioning the fit on $y = y^*$ shifts its mean by $\hat K(y^* - \bar g)$
# and multiplies its factor on the right by $T = (I_J + S S^\top)^{-1/2}$,
# where row $j$ of $S$ is $R^{-1/2}(g_j - \bar g)/\sqrt{J-1}$ and
# $\hat K = \hat C_{xg}(\hat C_{gg} + R)^{-1}$. Realizing reads one particle
# off each column:
#
# $$
# x_j^{a} = \bar x + \hat K (y^* - \bar g) + \sum_{i=1}^{J} (x_i - \bar x)\, T_{ij} .
# $$
#
# `square_root_map("y")` packages these two operations as a map that is built
# from the block name and called with the value.

# %%
sqrt_map = ens_joint.square_root_map("y")
sqrt_particles = sqrt_map(y=problem.y)
realized = ens_joint.condition(y=problem.y).realize_particles()
print(sqrt_particles)

# %% [markdown]
# ## Matheron: realize, then transport
#
# Realizing the fit without the noise term on $y$ returns the original
# particles, with $y_j = g_j$. The conditional map then moves each one by
# $\hat K(y^* - g_j - e_j)$, drawing $e_j \sim \mathcal N(0, R)$ as a standard
# normal in whitened coordinates.

# %%
particles = ens_joint.realize_particles(exclude_block_covs=("y",))   # y_j = g_j here
pert_particles = ens_joint.conditional_map("y")(particles, y=problem.y,
                                                key=jax.random.key(2))
print(pert_particles)

# %% [markdown]
# ## The same updates, through `kalman.update`
#
# `kalman.update` builds the same fit from the ensemble, conditions on
# $g = y^*$ with noise covariance $R$, and returns the parameter block. Its
# square-root rule is the square-root map above. Its Matheron rule splits its
# key as `k_targets, k_noise = jax.random.split(key)` and draws the noise with
# `k_noise`, so passing that half to the conditional map gives the same
# particles.

# %%
key = jax.random.key(3)
via_update = {
    "square root": kalman.update(ens, g=problem.y, noise={"g": problem.noise_cov},
                                 update_rule=kalman.SymmetricSquareRoot()),
    "Matheron": kalman.update(ens, g=problem.y, noise={"g": problem.noise_cov},
                              update_rule=kalman.Matheron(), key=key),
}
via_maps = {
    "square root": sqrt_particles,
    "Matheron": ens_joint.conditional_map("y")(particles, y=problem.y,
                                               key=jax.random.split(key)[1]),
}
for name in via_update:
    gap = jnp.abs(via_update[name]["u"] - via_maps[name]["u"]).max()
    print(f"{name}: largest difference {float(gap):.1e}")

# %% [markdown]
# ## What to notice
#
# The 200,000 transported samples match the exact posterior mean of $u$ to
# within 0.005 and its covariance to within 1% of the covariance's largest
# entry, a Monte Carlo error. The square-root particles equal the realized
# conditional fit's, and reproduce its mean and covariance to within
# $10^{-12}$, because realizing preserves the moments of a Gaussian aligned
# with its particles. The Matheron particles carry the blocks `u` and `g`,
# with the conditioned block `y` dropped. Both compositions give the
# particles of `kalman.update` to within $10^{-12}$.

# %%
# checks
assert mc_mean_err < 0.005
assert mc_cov_err < 0.01
fit = ens_joint.condition(y=problem.y)
assert jnp.allclose(sqrt_particles["u"], realized["u"], rtol=0, atol=1e-12)
assert jnp.allclose(sqrt_particles.mean("u"), fit.mean("u"), rtol=0, atol=1e-12)
assert jnp.allclose(sqrt_particles.cov("u").to_dense(), fit.cov("u").to_dense(),
                    rtol=0, atol=1e-12)
assert pert_particles.names == ("u", "g")
for name in via_update:
    assert jnp.allclose(via_update[name]["u"], via_maps[name]["u"], rtol=0, atol=1e-12)
