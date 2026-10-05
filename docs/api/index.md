# API reference

```{eval-rst}
.. currentmodule:: enskit
```

## enskit.linalg

Structured linear operators. See {doc}`../user-guide/operators` for the
catalog with costs, {doc}`../user-guide/writing-an-operator` for adding a
new structure, and {doc}`../linop-contract` for the full behavioral
contract.

### Base classes

```{eval-rst}
.. autoclass:: enskit.linalg.LinOp
   :members:

.. autoclass:: enskit.linalg.SquareLinOp
   :members:

.. autoclass:: enskit.linalg.PSDLinOp
   :members:
```

### Operators defined by their own arrays

```{eval-rst}
.. autoclass:: enskit.linalg.Identity
.. autoclass:: enskit.linalg.Zero
.. autoclass:: enskit.linalg.PSDDiagonal
.. autoclass:: enskit.linalg.Dense
.. autoclass:: enskit.linalg.DenseSquare
.. autoclass:: enskit.linalg.Triangular
.. autoclass:: enskit.linalg.DensePSD
.. autoclass:: enskit.linalg.PSDLowRank
```

### Operators built from other operators

```{eval-rst}
.. autoclass:: enskit.linalg.Transposed
.. autoclass:: enskit.linalg.Scaled
.. autoclass:: enskit.linalg.SquareScaled
.. autoclass:: enskit.linalg.PSDScaled
.. autoclass:: enskit.linalg.Product
.. autoclass:: enskit.linalg.HStack
.. autoclass:: enskit.linalg.BlockDiag
.. autoclass:: enskit.linalg.PSDBlockDiag
.. autoclass:: enskit.linalg.PSDDiagCongruence
.. autoclass:: enskit.linalg.LowRankUpdate
```

### Kronecker products

```{eval-rst}
.. automodule:: enskit.linalg.kronecker
   :no-members:

.. autoclass:: enskit.linalg.Kronecker
.. autoclass:: enskit.linalg.SquareKronecker
.. autoclass:: enskit.linalg.PSDKronecker
```

### The conditioning core

```{eval-rst}
.. automodule:: enskit.linalg.gram
   :no-members:

.. autoclass:: enskit.linalg.IdentityPlusGram
   :members: solve_factor, inverse_sqrt

.. autoclass:: enskit.linalg.IdentityPlusGramInverseSqrt
```

### Factory functions

```{eval-rst}
.. autofunction:: enskit.linalg.block_diag
.. autofunction:: enskit.linalg.product
.. autofunction:: enskit.linalg.hstack
.. autofunction:: enskit.linalg.diag_congruence
.. autofunction:: enskit.linalg.kron
```

### Helpers for defining operators

```{eval-rst}
.. autofunction:: enskit.linalg.linop
.. autofunction:: enskit.linalg.static_field
.. autofunction:: enskit.linalg.dense_matvec
.. autofunction:: enskit.linalg.tri_solve
.. autofunction:: enskit.linalg.densify
.. autofunction:: enskit.linalg.dense_fallback
.. autofunction:: enskit.linalg.set_debug_checks
.. autofunction:: enskit.linalg.debug_checks
.. autofunction:: enskit.linalg.value_check
.. autoexception:: enskit.linalg.UnsupportedOpError
```

### Conformance testing

```{eval-rst}
.. automodule:: enskit.linalg.testing
   :members: check_operator, check_core, check_transpose, check_solve,
             check_factor, check_whiten, check_scalars,
             check_dense_independence, check_capabilities,
             check_operand_validation, check_pytree, check_repr,
             check_arithmetic, check_family
```

## enskit.distribution

Distributions over named blocks, conditioning and conditional maps. See
{doc}`../user-guide/distributions` for when to use each piece, and
{doc}`../distribution-contract` for the full behavioral contract.

```{eval-rst}
.. automodule:: enskit.distribution
   :no-members:
```

### Distributions

```{eval-rst}
.. autoclass:: enskit.distribution.Ensemble
   :members: names, dims, n_particles, is_weighted, log_weights, weights,
             all_finite, batch_shape, __getitem__, marginal, drop, assign,
             rename, pipe, mean, anomalies, cov, project

.. autoclass:: enskit.distribution.Gaussian
   :members: independent, dims, latent_dim, batch_shape, mean, factor,
             block_cov, cov, marginal, drop, rename, pipe, add_noise, absorb,
             compress, condition, conditional_map, log_density, sample

.. autoclass:: enskit.distribution.EnsembleGaussian
   :members: realize_particles, square_root_map
```

### Conditional maps

```{eval-rst}
.. autoclass:: enskit.distribution.ConditionalMap
   :members: __call__

.. autoclass:: enskit.distribution.MatheronMap
   :members: __call__, coefficients, particle_coefficients, batch_shape

.. autoclass:: enskit.distribution.SquareRootMap
   :members: __call__, batch_shape
```

### Weights, and the exact-moment fixture

```{eval-rst}
.. autofunction:: enskit.distribution.reweight
.. autofunction:: enskit.distribution.effective_sample_size
.. autofunction:: enskit.distribution.resample
.. autofunction:: enskit.distribution.exact_moment_ensemble
```

## enskit.maps

Pushing distributions through maps, and the simulator contract. See
{doc}`../user-guide/maps` for when to use each route, and
{doc}`../maps-contract` for the full behavioral contract.

```{eval-rst}
.. automodule:: enskit.maps
   :no-members:

.. autofunction:: enskit.maps.pushforward

.. autoclass:: enskit.maps.StructuredMap
   :members: push_ensemble, push_gaussian

.. autoclass:: enskit.maps.Linear
   :members: op, output_dim, __call__, push_ensemble, push_gaussian

.. autoclass:: enskit.maps.AdditiveNoise
   :members: dim, push_ensemble, push_gaussian

.. autoclass:: enskit.maps.BlackBox
   :members: __call__
```

## enskit.kalman

Ensemble Kalman updates, localization, inflation and relaxation. See
{doc}`../user-guide/updates` for when to use each rule,
{doc}`../user-guide/localization` for localization, and
{doc}`../kalman-contract` for the full behavioral contract.

```{eval-rst}
.. automodule:: enskit.kalman
   :no-members:
```

### The update

```{eval-rst}
.. autofunction:: enskit.kalman.update
.. autofunction:: enskit.kalman.gaussian_approximation

.. autoclass:: enskit.kalman.UpdateRule
   :members: build

.. autoclass:: enskit.kalman.ParticleUpdate
   :members: __call__
```

### Update rules

```{eval-rst}
.. autoclass:: enskit.kalman.SymmetricSquareRoot
   :members: build

.. autoclass:: enskit.kalman.Matheron
   :members: build
```

### Localization

```{eval-rst}
.. autoclass:: enskit.kalman.LocalizedUpdateRule
   :members: build

.. autoclass:: enskit.kalman.DomainLocalization

.. autofunction:: enskit.kalman.gaspari_cohn
```

### Inflation and relaxation

```{eval-rst}
.. autofunction:: enskit.kalman.inflate_multiplicative
.. autofunction:: enskit.kalman.inflate_additive
.. autofunction:: enskit.kalman.relax_to_prior_spread
.. autofunction:: enskit.kalman.relax_to_prior_perturbations
```

## enskit.testing

Conformance checks for code written against EnsKit's interfaces: a
simulator, an update rule, a conditional map, and the policies of a run.

```{eval-rst}
.. automodule:: enskit.testing
   :no-members:

.. autofunction:: enskit.testing.check_simulator
.. autofunction:: enskit.testing.check_update_rule
.. autofunction:: enskit.testing.check_conditional_map
.. autofunction:: enskit.testing.check_schedule
.. autofunction:: enskit.testing.check_inflation
.. autofunction:: enskit.testing.check_relaxation
.. autofunction:: enskit.testing.check_stopping_rule
```

## enskit.algorithms

The algorithms, and the inflation and relaxation policies every driver
shares. See {doc}`../user-guide/running-an-inversion` and
{doc}`../user-guide/filtering` for when to use each piece, and
{doc}`../eki-contract` and {doc}`../enkf-contract` for the full behavioral
contracts.

```{eval-rst}
.. automodule:: enskit.algorithms
   :no-members:
```

### Inflation and relaxation policies

```{eval-rst}
.. autoclass:: enskit.algorithms.Inflation
   :members: __call__

.. autoclass:: enskit.algorithms.Relaxation
   :members: __call__

.. autoclass:: enskit.algorithms.MultiplicativeInflation
   :members: __call__, batch_shape
.. autoclass:: enskit.algorithms.AdditiveInflation
   :members: __call__, batch_shape
.. autoclass:: enskit.algorithms.RelaxToPriorSpread
   :members: __call__, batch_shape
.. autoclass:: enskit.algorithms.RelaxToPriorPerturbations
   :members: __call__, batch_shape
```

## enskit.algorithms.eki

```{eval-rst}
.. automodule:: enskit.algorithms.eki
   :no-members:
```

### Value classes

```{eval-rst}
.. autoclass:: enskit.algorithms.eki.EKIState
   :members: from_prior, restart, n_particles, dims, mean, batch_shape

.. autoclass:: enskit.algorithms.eki.Evaluation
   :members: misfits, center_misfit, rms_parameter_spread, n_particles,
             data_dim, batch_shape

.. autoclass:: enskit.algorithms.eki.HistoryRecord
   :members: from_evaluation, batch_shape

.. autoclass:: enskit.algorithms.eki.EKIResult
   :members: ensemble, beta, mean, n_evaluations, n_completed_steps,
             min_n_valid, stacked, stop_fired, budget_complete
```

### Schedules and stopping rules

```{eval-rst}
.. autoclass:: enskit.algorithms.eki.Schedule
   :members: next_increment

.. autoclass:: enskit.algorithms.eki.StoppingRule
   :members: __call__

.. autoclass:: enskit.algorithms.eki.FixedSchedule
   :members: uniform, constant, n_steps, beta_target, next_increment
.. autoclass:: enskit.algorithms.eki.AdaptiveESSSchedule
   :members: n_steps, next_increment
.. autoclass:: enskit.algorithms.eki.AdaptiveMisfitSchedule
   :members: n_steps, next_increment
.. autoclass:: enskit.algorithms.eki.DiscrepancyStop
   :members: __call__
```

### The driver, and one step

```{eval-rst}
.. autofunction:: enskit.algorithms.eki.run
.. autofunction:: enskit.algorithms.eki.iterate
.. autofunction:: enskit.algorithms.eki.evaluate
.. autofunction:: enskit.algorithms.eki.assimilate
.. autofunction:: enskit.algorithms.eki.advance
```

### Helpers, constants, and the exception

```{eval-rst}
.. autofunction:: enskit.algorithms.eki.misfits
.. autofunction:: enskit.algorithms.eki.effective_sample_size
.. autofunction:: enskit.algorithms.eki.repair_failed_particles

.. autodata:: enskit.algorithms.eki.PREDICTION
.. autodata:: enskit.algorithms.eki.SCHEDULE_EXHAUSTED
.. autodata:: enskit.algorithms.eki.STOPPING_RULE
.. autodata:: enskit.algorithms.eki.INTERRUPTED

.. autoexception:: enskit.algorithms.eki.EKIError
```

## enskit.algorithms.enkf

```{eval-rst}
.. automodule:: enskit.algorithms.enkf
   :no-members:
```

### The filter, and its two halves

```{eval-rst}
.. autofunction:: enskit.algorithms.enkf.filter
.. autofunction:: enskit.algorithms.enkf.forecast
.. autofunction:: enskit.algorithms.enkf.analysis
```

### The result, the exception, and the prediction block

```{eval-rst}
.. autoclass:: enskit.algorithms.enkf.FilterResult
   :members: n_times, total_log_evidence

.. autoexception:: enskit.algorithms.enkf.EnKFError

.. autodata:: enskit.algorithms.enkf.PREDICTION
```

## enskit.toy

Toy problems for tests and documentation — not for production use. See
{doc}`../user-guide/toy-models` for what they are for and what they do and do
not exemplify about the forward-model interface.

```{eval-rst}
.. automodule:: enskit.toy
   :no-members:
```

### The problems

```{eval-rst}
.. autoclass:: enskit.toy.LinearGaussian
   :members: parameter_dim, data_dim, forward, posterior

.. autoclass:: enskit.toy.ExponentialDecay
   :members: parameter_dim, data_dim, forward

.. autoclass:: enskit.toy.RestrictedDecay
   :members: parameter_dim, data_dim, forward

.. autoclass:: enskit.toy.Lorenz96
   :members: state_dim, data_dim, n_times, coords, transition

.. autoclass:: enskit.toy.LinearStateSpace
   :members: state_dim, data_dim, n_times, exact_filter
```

### Factories

```{eval-rst}
.. autofunction:: enskit.toy.linear_gaussian
.. autofunction:: enskit.toy.exponential_decay
.. autofunction:: enskit.toy.restricted_decay
.. autofunction:: enskit.toy.lorenz96
.. autofunction:: enskit.toy.linear_state_space
.. autofunction:: enskit.toy.lorenz96_step
```
