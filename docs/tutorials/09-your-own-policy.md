# 9. Writing your own policy

:::{admonition} Stub
:class: note
Not yet written. The scope below is settled; the prose is not.
:::

## Goal

The reader can write a schedule, an update rule, or an inflation of their own,
and validate it against the conformance harness before trusting it.

## Prerequisites

Tutorials 1 to 8.

## What this page covers

- Where the extension seams are, and why they are where they are:
  `enskit.linalg` is extended by writing an operator, `enskit.kalman` by
  writing an update rule, and `enskit.algorithms` by writing a policy. One
  paragraph.
- The protocols — `kalman.UpdateRule`, `eki.Schedule`, `eki.StoppingRule`,
  `algorithms.Inflation` and `algorithms.Relaxation` — and what the driver
  calls on each and when. An update rule is a factory: `build(particles,
  approximation, given)` returns the update, which receives the given values
  when called.
- The two phases of a step, `evaluate` and `assimilate`, and `advance` as their
  composition. A policy author needs this to know what is available when.
- A worked schedule: something small and genuinely useful, for example a
  schedule that caps the increment by a target misfit reduction. Write it,
  then run it.
- A worked update rule as well: example 14 of the {doc}`../examples/index`
  writes the deterministic EnKF of Sakov & Oke (2008) as a rule.
- **Validate it**: `check_schedule`, `check_update_rule`,
  `check_stopping_rule`, `check_inflation` and `check_relaxation` from
  `enskit.testing`, each of which builds its own inputs when none are given.
  Make clear this is the step that catches the errors that produce wrong
  numbers rather than exceptions.
- The `**_` seam in the protocol signatures, and that it is what makes future
  fields non-breaking — so a policy should accept it.
- The JAX rules a policy author will otherwise learn the hard way:
  - **Never pass an `EKIState` or an `Evaluation` whole across a `jit`
    boundary.** A static field on an object crossing the boundary retraces
    every step; a Python float passed as an argument does not.
  - Every comparison against `nan` is `False`, so a bisection on such a
    comparison silently returns its lower bracket — and a floor then makes that
    look like an ordinary step. A single-`where` guard is not enough; a `nan`
    misfit must not be allowed to reach the `inf` branch.
  - `jnp.mean(x, axis=-2)` without `keepdims` right-aligns the subtraction
    against the batch axis, so an operand whose leading axis equals $J$
    broadcasts and returns wrong anomalies without raising.
  - Return JAX scalars, not Python floats: converting fails on a tracer under
    `jit`.
- A closing pointer: {doc}`../user-guide/writing-an-operator` for the other
  extension seam, and the {doc}`../eki-contract` for the normative obligations
  a policy must meet.

## Deliberately not covered

- writing an operator
- the conformance harness's own internals

## API exercised

`enskit.algorithms.eki`: `Schedule`, `StoppingRule`, `evaluate`,
`assimilate`, `advance`, `misfits`, `effective_sample_size`.
`enskit.algorithms`: `Inflation`, `Relaxation`. `enskit.kalman`:
`UpdateRule`. `enskit.testing`: `check_schedule`, `check_update_rule`,
`check_stopping_rule`, `check_inflation`, `check_relaxation`.

## Notes for the writer

The user guide's "Writing your own policy" section is the reference version.
The value this page adds over it is a single worked example carried all the way
through validation, and the JAX traps stated as things that will happen to the
reader rather than as rules.
