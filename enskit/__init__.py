"""EnsKit — building blocks for ensemble Kalman methods.

Ensemble Kalman methods estimate the inputs of a model that can be run but not
differentiated, from noisy observations of its outputs, by moving an ensemble
of particles with linear updates computed from the ensemble's own statistics.
EnsKit provides the building blocks, and two algorithms built from them:
Ensemble Kalman Inversion (EKI), which calibrates the parameters of a forward
model, and the ensemble Kalman filter (EnKF), which tracks the state of a
dynamical system observed sequentially in time.

- :mod:`enskit.linalg` — structured linear operators, so covariance structure is
  exploited rather than materialized as dense arrays.
- :mod:`enskit.distribution` — distributions over named blocks: ensembles,
  Gaussians, and the conditional maps between them.
- :mod:`enskit.maps` — pushing distributions through maps and simulators.
- :mod:`enskit.kalman` — ensemble Kalman updates, localization, inflation and
  relaxation.
- :mod:`enskit.algorithms` — the algorithms themselves:
  :mod:`~enskit.algorithms.eki` and :mod:`~enskit.algorithms.enkf`, and the
  inflation and relaxation policies they share.
- :mod:`enskit.testing` — conformance checks for a simulator, update rule or
  policy of your own.
- :mod:`enskit.toy` — small problems for this package's tests and
  documentation, not for production use.

A forward model or a transition is any callable from a batch of inputs to a
batch of outputs, so EnsKit is independent of the model it is applied to.

Importing this package enables JAX float64. JAX defaults to float32, which is
not accurate enough for the conditioning arithmetic: ensemble anomalies are
formed by subtraction, and the resulting cancellation loses several digits.
Import EnsKit before creating any array, since arrays made beforehand stay
float32 and are not promoted afterward.

Notes
-----
The float64 setting applies to the current process only. Worker processes
created by a process pool do not inherit it; set the environment variable
``JAX_ENABLE_X64=1`` when forward-model evaluations run in separate processes.
"""

import jax

jax.config.update("jax_enable_x64", True)

__version__ = "0.2.0.dev0"

__all__ = ["__version__"]
