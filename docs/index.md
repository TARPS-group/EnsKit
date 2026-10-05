# EnsKit

Building blocks for ensemble Kalman methods, and Ensemble Kalman Inversion and
the ensemble Kalman filter built from them.

:::{admonition} Pre-alpha
:class: warning

Every layer is implemented and tested, and the interfaces may still change
before the first release.
:::

## What problem does this solve?

You have a model you can run but not differentiate: a legacy simulator, a
coupled code, or a binary invoked as a subprocess. You have noisy
observations of what it predicts, and you want to estimate the quantities it
depends on, with an honest account of how uncertain the estimate is. Two
versions of the problem are common.

**Calibration.** A forward model $\mathcal{G}$ maps parameters to
predictions, and the data are those predictions corrupted by noise,

$$y = \mathcal{G}(\theta) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \Sigma).$$

Ensemble Kalman Inversion (EKI) moves an ensemble of parameter vectors toward
the posterior distribution of $\theta$ using only forward evaluations, with no
gradients, no adjoint, and no access to the model's internals.

**Filtering.** A dynamical system advances a state, $x_t = M(x_{t-1})$, and
is observed at each time, $y_t = H(x_t) + \varepsilon_t$. The ensemble Kalman
filter (EnKF) tracks the distribution of the current state from the
observations so far, with an ensemble far smaller than the state.

Both are built from the same few operations, and EnsKit provides those
operations as well as the two algorithms, so a method that is neither is
assembled from the same pieces.

## What EnsKit provides

::::{grid} 2
:gutter: 3

:::{grid-item-card} Structured operators
`enskit.linalg` represents covariance matrices by how they act on vectors, so
that structure (block, diagonal, triangular, low-rank and Kronecker) is
exploited rather than materialized.
:::

:::{grid-item-card} Distributions
`enskit.distribution` provides ensembles of particles and Gaussians over named
blocks: sampling, marginals, conditioning, and the Gaussian fitted to an
ensemble.
:::

:::{grid-item-card} Maps
`enskit.maps` pushes a distribution through a simulator, any callable from a
batch of inputs to a batch of outputs, or exactly through a linear map or
additive noise.
:::

:::{grid-item-card} Ensemble Kalman updates
`enskit.kalman` moves particles toward a conditional distribution, with the
square-root and the perturbed-observation (Matheron) update rules, and
inflation and relaxation around them.
:::

:::{grid-item-card} Algorithms
`enskit.algorithms.eki` runs Ensemble Kalman Inversion: tempering schedules,
stopping rules, failure handling and the driver loop, in both the
approximate-sampling and the optimization form. `enskit.algorithms.enkf` runs
the ensemble Kalman filter over a sequence of observations.
:::

:::{grid-item-card} Checks and toy problems
`enskit.testing` checks a simulator, update rule or policy of your own
against its contract, and `enskit.toy` holds small problems for trying the
library.
:::
::::

## What EnsKit is not

EnsKit does not implement production forward models, priors, or Gaussian
process kernels. It ships five toy problems, in `enskit.toy`, for its own
tests and this documentation.
A forward model or a transition is any callable from a batch of inputs to a
batch of outputs, and a prior covariance is any operator meeting the
covariance interface.
Building those belongs to the caller, which keeps EnsKit independent of the
domain being calibrated. {doc}`user-guide/writing-a-forward-model` states
everything a forward model must satisfy, and works through wrapping an
external executable.

## Quick example

Calibrating a two-parameter decay model against three noisy observations:

```python
import enskit                      # enables float64; import before creating arrays
import jax, jax.numpy as jnp
from enskit import kalman
from enskit.algorithms import eki
from enskit.distribution import Gaussian
from enskit.linalg import PSDDiagonal

# The forward model: any callable from (J, 2) parameters to (J, 3) predictions.
times = jnp.array([0.5, 1.0, 2.0])
def forward(u):
    return u[:, :1] * jnp.exp(-u[:, 1:2] * times)

y = jnp.array([1.75, 1.38, 0.82])                             # observations
noise = PSDDiagonal(jnp.full(3, 0.01))                        # their error covariance
prior = Gaussian.independent(u=(jnp.zeros(2), PSDDiagonal(jnp.array([4.0, 1.0]))))

state = eki.EKIState.from_prior(jax.random.key(0), prior, n_particles=64)
result = eki.run(state, forward, y, noise,
                 update_rule=kalman.Matheron(),
                 schedule=eki.AdaptiveESSSchedule())

result.mean("u")       # posterior mean estimate
result.stacked.ess     # effective sample size at each step of the ladder
```

{doc}`tutorials/01-first-inversion` builds this up from the beginning and explains
every choice in it.

## How this documentation is organized

::::{grid} 2
:gutter: 3

:::{grid-item-card} Tutorials
Read in order, each building on the last. Start here if you are new — the
first one runs an inversion in twenty lines and assumes nothing.
+++
{doc}`tutorials/index`
:::

:::{grid-item-card} User guide
Answers "when and why" for a specific choice, organized by level of
abstraction, from running an algorithm down to operators. Read out of order,
as needed.
+++
{doc}`user-guide/index`
:::

:::{grid-item-card} Examples
Fifteen worked examples, each a problem stated precisely and solved end to
end, from one call of a driver to a custom update rule.
+++
{doc}`examples/index`
:::

:::{grid-item-card} Reference
The normative contracts specifying each layer's behavior exactly, the design
notes, and the API.
+++
{doc}`api/index`
:::
::::

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

installation
```

```{toctree}
:maxdepth: 2
:caption: Tutorials
:hidden:

tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: User guide
:hidden:

user-guide/index
```

```{toctree}
:maxdepth: 2
:caption: Examples
:hidden:

examples/index
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

linop-contract
distribution-contract
maps-contract
kalman-contract
eki-contract
enkf-contract
joint-factor
design
redesign/index
api/index
```
