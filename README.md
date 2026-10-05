# EnsKit

Building blocks for ensemble Kalman methods, and Ensemble Kalman Inversion
built from them.

**Status: pre-alpha.** The operator, distribution, map and update layers, and
Ensemble Kalman Inversion on top of them, are implemented, tested and
documented: you can run an inversion today, with domain localization
(`enskit.kalman.LocalizedUpdateRule`) as its update rule where the parameter
dimension far exceeds the ensemble size. The ensemble Kalman filter is planned
and not yet built.

## What it is

Ensemble Kalman Inversion (EKI) estimates the parameters of an expensive,
possibly non-differentiable forward model from noisy observations. It needs
only forward evaluations: no gradients, no adjoint, no access to the model's
internals. That makes it well suited to legacy simulators, coupled codes, and
anything that runs as a subprocess.

EnsKit aims to be a small, robust, efficient toolkit for ensemble Kalman
methods, in layers that build on each other:

- **`enskit.linalg`**: structured linear operators, so covariance structure is
  exploited rather than materialized as dense arrays.
- **`enskit.distribution`**: ensembles of particles and Gaussians over named
  blocks, with sampling, marginals and conditioning.
- **`enskit.maps`**: pushing a distribution through a simulator, or exactly
  through a linear map or additive noise.
- **`enskit.kalman`**: one ensemble Kalman update, with the square-root and the
  perturbed-observation (Matheron) rules, domain localization, and inflation
  and relaxation.
- **`enskit.algorithms.eki`**: Ensemble Kalman Inversion itself: tempering
  schedules, stopping rules, failure handling and the driver loop, in both the
  approximate-sampling and the optimization form.

## What it is not

EnsKit does not implement production forward models, priors, or Gaussian
process kernels. It ships three toy problems, in `enskit.toy`, for its own
tests and its documentation.
The forward model is any callable mapping parameters to predicted observations,
and a prior covariance is any operator satisfying the covariance interface.
Building those is the caller's job, which keeps EnsKit independent of the
domain being calibrated. What a forward model must satisfy is stated in one
place, the simulator contract, and the "Forward model requirements" page of
the user guide works through it.

## Installation

```bash
uv sync
```

For development, including docs and test tooling:

```bash
uv sync --group dev
```

## Quick example

Calibrating a two-parameter decay model against three noisy observations:

```python
import enskit                     # enables float64; import before creating arrays
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

The same driver runs the optimization form. It is a different schedule, not a
different function or a flag:

```python
fit = eki.run(state, forward, y, noise,
              update_rule=kalman.SymmetricSquareRoot(),
              schedule=eki.FixedSchedule.constant(1.0, n_steps=200),
              stop=eki.DiscrepancyStop(tau=1.0))
```

Covariances are structured operators, so a mix of independent and correlated
observation error costs the sum over blocks rather than the cube of the total
size:

```python
from enskit.linalg import DensePSD, block_diag

noise = block_diag(
    PSDDiagonal(jnp.array([0.5, 0.5, 2.0])),   # independent errors
    DensePSD(jnp.eye(2) + 0.3),                # correlated block
)

noise.shape          # (5, 5)
noise.logdet()       # summed over blocks, never forms a 5x5 matrix
noise.whiten(jnp.ones(5))   # applied block by block
```

## Documentation

<https://tarps-group.github.io/pyEKI/>

Start with the tutorials, which build up from a first inversion; the user guide
answers "when and why" for each choice; the contracts specify behavior
normatively.

## License

MIT
