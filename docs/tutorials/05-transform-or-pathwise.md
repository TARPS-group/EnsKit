# 5. Square root or Matheron

:::{admonition} Stub
:class: note
Not yet written. The scope below is settled; the prose is not.
:::

## Goal

The reader can choose between the two ways EnsKit moves an ensemble through a
step, and knows what each one costs.

## Prerequisites

Tutorials 1 to 4.

## What this page covers

Both rules answer the same question — the ensemble and its predictions are in
hand, and the increment is chosen, so where do the particles go? They differ in
how they produce the spread.

- **`kalman.Matheron`**, the classical perturbed-observation form. Each
  particle is conditioned on the observation plus its own draw from the
  observation error, so the spread comes from those draws. It consumes the
  run's key, so two runs from different keys differ.
- **`kalman.SymmetricSquareRoot`**, the deterministic square-root form. The
  ensemble's anomalies are transformed so that their empirical covariance
  equals the conditioned one directly, with nothing drawn.
- **What the choice costs.** `SymmetricSquareRoot` is deterministic, so a
  difference between two runs is a bug rather than a seed. The exactness
  property holds *exactly* under it and only *in expectation* under the
  stochastic form. `Matheron` is the form most of the literature is written
  in, and the one whose map moves any sample of the joint, not only the
  particles it was built from.
- **Same problem, both rules, measured.** Not just the two answers, but the
  variability of each across keys: the honest comparison is one run of
  `SymmetricSquareRoot` against the spread of many `Matheron` runs, because a
  single stochastic run tells the reader nothing about whether the difference
  they see is the rule or the seed.
- **Where each is preferable**, stated equally plainly. The stochastic form's
  errors are unbiased across keys rather than systematic, and averaging over
  keys is available to it.
- Both rules are a few lines over `enskit.distribution`: the square-root
  update is *realize after condition* and Matheron's is *transport after
  realize* ({doc}`../user-guide/distributions`, and example 5 of the
  {doc}`../examples/index`). Link; do not reproduce.

## Deliberately not covered

- the derivation of either update → {doc}`../kalman-contract` and
  {doc}`../distribution-contract`
- why `square_root_map` exists only on an `EnsembleGaussian` →
  {doc}`../distribution-contract`
- writing an update rule of your own → {doc}`09-your-own-policy`

## API exercised

`kalman.SymmetricSquareRoot`, `kalman.Matheron`, `EKIState.key`,
`distribution.EnsembleGaussian`.

## Notes for the writer

Measured before PR 7 (re-measure), on `toy.exponential_decay()` at 64
particles, `AdaptiveESSSchedule`: the stochastic run took 8 forward
evaluations against the square-root run's 6, and
the two posterior means differ in the third digit. That is one key, and one
key is not a comparison — the page's own point above is that this needs
replication across keys before anything is claimed from it.

**One finding this page should carry, because it is what moved
{doc}`01-first-inversion` onto `Matheron`.** On the same problem, over eight
keys, the square-root rule reproduces the target's covariance but not its
shape:
the ensemble's least-varying principal direction comes out with a kurtosis
between 6 and 34, against 3 for a Gaussian, and at 64 particles two particles can
hold 72% of that direction's variance. `Matheron` gives 2.3 to 3.6.

It is the nonlinearity rather than sampling error, and the two checks that
establish that are worth repeating on the page. Ensemble size does not help —
the kurtosis is 19 at 64 particles and 186 at 4096. And on
`toy.linear_gaussian` the same rule leaves the ensemble Gaussian
(kurtosis 3.2) *identically* at 1, 6 and 20 steps, so the rule is not
intrinsically spiky: the square-root update is a linear recombination of the
anomalies the ensemble already has, and a curved forward model gives it
anomalies whose off-ridge mass sits in a few particles.

This is one problem at one size, which is exactly the caveat below. Do not
promote it to a ranking of the two rules.

**Refinement depends on the rule, and tutorial 4 left this to this page.**
Measured while writing {doc}`04-tempering-schedules`, on the same problem at
64 particles, uniform ladders of 1 to 400 steps. Under `SymmetricSquareRoot`
a longer ladder was better from three steps on, on every key from 0 to 5
(key 0: an error in the mean of 0.0034 at 6 steps, 0.0018 at 12, 0.0009 at
24); only the step from 1 to 2 worsened the mean, on keys 4 and 5. Tutorial 4
shows both. Under `Matheron` it is not: key 1 gives
0.0014 at 6 steps, 0.0054 at 12, 0.0075 at 24 and 0.0089 at 48, and the
spread wanders on either side of the target's rather than approaching it.
Each step draws fresh perturbations, and a longer ladder accumulates more of
them. This is where tutorial 3's "two ladders give two answers, and neither
bounds the other" can be shown with refinement at every length; under the
square-root rule on this problem it shows only at the shortest ladders. Re-measure, and replicate across keys, before
writing.

The temptation to resist is ranking the two rules. A run takes its rule
explicitly, with no default; this page should leave a reader able to defend
either choice on their own problem.
