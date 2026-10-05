# EnsKit

Building blocks for ensemble Kalman methods, and Ensemble Kalman Inversion
built from them.

:::{admonition} Pre-alpha
:class: warning

The operator, distribution, map and update layers, and Ensemble Kalman
Inversion on top of them, are implemented and tested: you can run an
inversion today. The ensemble Kalman filter and localization are planned and
not yet built.
:::

## What problem does this solve?

You have a forward model $\mathcal{G}$ that maps parameters to predictions, and
observations $y$ of those predictions corrupted by noise:

$$y = \mathcal{G}(\theta) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \Sigma).$$

You want to estimate $\theta$ and quantify how uncertain that estimate is. The
difficulty is that $\mathcal{G}$ is expensive, and often you cannot
differentiate it: it may be a legacy simulator, a coupled code, or a binary
invoked as a subprocess.

Ensemble Kalman Inversion (EKI) handles exactly this case. It moves an
ensemble of parameter vectors toward the posterior using only forward
evaluations, requiring no gradients, no adjoint, and no access to the model's
internals.

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
approximate-sampling and the optimization form.
:::

:::{grid-item-card} Checks and toy problems
`enskit.testing` checks a simulator, update rule or policy of your own
against its contract, and `enskit.toy` holds small problems for trying the
library.
:::
::::

## What EnsKit is not

EnsKit does not implement production forward models, priors, or Gaussian
process kernels. It ships three toy problems, in `enskit.toy`, for its own
tests and this documentation.
The forward model is any callable from parameters to predicted observations,
and a prior covariance is any operator meeting the covariance interface.
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
Answers "when and why" for a specific choice, once you know the shape of the
problem. Read out of order, as needed.
+++
{doc}`user-guide/running-an-inversion`
:::

:::{grid-item-card} Examples
Runnable notebooks working through a problem end to end, with plots and
diagnostics.
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

user-guide/quickstart
user-guide/operators
user-guide/distributions
user-guide/maps
user-guide/updates
user-guide/running-an-inversion
user-guide/writing-a-forward-model
user-guide/toy-models
user-guide/writing-an-operator
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
joint-factor
design
redesign/index
api/index
```
