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

## enskit.gauss

Joint Gaussian distributions and conditioning. See
{doc}`../user-guide/conditioning` for when to use each piece, and
{doc}`../gaussian-contract` for the full behavioral contract.

```{eval-rst}
.. automodule:: enskit.gauss
   :no-members:
```

### Distributions

```{eval-rst}
.. autoclass:: enskit.gauss.Gaussian
   :members: from_samples, dim, batch_shape, sample, log_density

.. autoclass:: enskit.gauss.GaussianJoint
   :members: from_linear_map, from_samples, from_factors, u_dim, v_dim,
             latent_dim, batch_shape, u_marginal, v_marginal, condition,
             pathwise

.. autoclass:: enskit.gauss.EmpiricalJoint
   :members: n_samples, u_dim, v_dim, batch_shape, u_mean, v_mean,
             u_anomalies, v_anomalies, to_gaussian_joint, transform_update,
             pathwise_update
```

### Conditioning primitives

```{eval-rst}
.. autofunction:: enskit.gauss.gain_weights
.. autofunction:: enskit.gauss.sqrt_transform
```

## enskit.eki

Ensemble Kalman Inversion: the ladder, the policies that shape it, and the
run. See {doc}`../user-guide/running-an-inversion` for when to use each piece,
and {doc}`../eki-contract` for the full behavioral contract.

```{eval-rst}
.. automodule:: enskit.eki
   :no-members:
```

### Value classes

```{eval-rst}
.. autoclass:: enskit.eki.EKIState
   :members: from_prior, restart, n_members, u_dim, mean, batch_shape

.. autoclass:: enskit.eki.Evaluation
   :members: misfits, center_misfit, n_members, u_dim, v_dim, batch_shape

.. autoclass:: enskit.eki.HistoryRecord
   :members: batch_shape

.. autoclass:: enskit.eki.EKIResult
   :members: ensemble, beta, mean, n_evaluations, n_completed_steps,
             min_n_valid, stacked, stop_fired, budget_complete
```

### The three axes, as protocols

```{eval-rst}
.. autoclass:: enskit.eki.EnsembleUpdate
   :members: __call__

.. autoclass:: enskit.eki.Schedule
   :members: next_increment

.. autoclass:: enskit.eki.StoppingRule
   :members: __call__

.. autoclass:: enskit.eki.Inflation
   :members: __call__
```

### Update rules

```{eval-rst}
.. autoclass:: enskit.eki.TransformUpdate
   :members: __call__
.. autoclass:: enskit.eki.PathwiseUpdate
   :members: __call__
```

### Schedules and stopping rules

```{eval-rst}
.. autoclass:: enskit.eki.FixedSchedule
   :members: uniform, constant, n_steps, beta_target, next_increment
.. autoclass:: enskit.eki.AdaptiveESSSchedule
   :members: n_steps, next_increment
.. autoclass:: enskit.eki.AdaptiveMisfitSchedule
   :members: n_steps, next_increment
.. autoclass:: enskit.eki.DiscrepancyStop
   :members: __call__
```

### Inflation

```{eval-rst}
.. autoclass:: enskit.eki.MultiplicativeInflation
   :members: __call__, batch_shape
.. autoclass:: enskit.eki.AdditiveInflation
   :members: __call__
```

### The driver, and one step

```{eval-rst}
.. autofunction:: enskit.eki.run
.. autofunction:: enskit.eki.iterate
.. autofunction:: enskit.eki.evaluate
.. autofunction:: enskit.eki.assimilate
.. autofunction:: enskit.eki.advance
```

### Helpers, status constants, and the exception

```{eval-rst}
.. autofunction:: enskit.eki.misfits
.. autofunction:: enskit.eki.effective_sample_size
.. autofunction:: enskit.eki.repair_failed_members

.. autodata:: enskit.eki.SCHEDULE_EXHAUSTED
.. autodata:: enskit.eki.STOPPING_RULE
.. autodata:: enskit.eki.INTERRUPTED

.. autoexception:: enskit.eki.EKIError
```

### Conformance testing

```{eval-rst}
.. automodule:: enskit.eki.testing
   :members: check_schedule, check_update, check_inflation,
             check_stopping_rule, check_forward_model, synthetic_evaluation
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
   :members: u_dim, v_dim, forward, posterior

.. autoclass:: enskit.toy.ExponentialDecay
   :members: u_dim, v_dim, forward

.. autoclass:: enskit.toy.RestrictedDecay
   :members: u_dim, v_dim, forward
```

### Factories

```{eval-rst}
.. autofunction:: enskit.toy.linear_gaussian
.. autofunction:: enskit.toy.exponential_decay
.. autofunction:: enskit.toy.restricted_decay
```
