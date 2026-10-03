"""Probability distributions over named vector blocks.

This module provides the two distributions every ensemble Kalman method is
built from, and the exact operations on them.

=============================== ===============================================
object                          represents
=============================== ===============================================
:class:`Ensemble`               an empirical distribution: ``n_particles`` points,
                                optionally weighted, over named blocks
:class:`Gaussian`               a joint Gaussian over named blocks, held as a
                                shared factor plus an independent term per block
:class:`EnsembleGaussian`       a Gaussian whose latent coordinates are an
                                ensemble's particles
:class:`ConditionalMap`         protocol: a pointwise map carrying samples of a
                                joint to samples of a conditional
:class:`MatheronMap`            the exact conditional of a :class:`Gaussian`,
                                as the affine map ``x -> x + K (y* - y)``
:class:`SquareRootMap`          the square-root update of an
                                :class:`EnsembleGaussian`'s particle set
:func:`reweight`                multiply an ensemble's weights by likelihood
                                factors, in log space
:func:`effective_sample_size`   ``1 / sum(w**2)`` for normalized weights
:func:`resample`                draw an unweighted ensemble from a weighted one
:func:`exact_moment_ensemble`   particles whose sample moments equal a
                                Gaussian's exactly
=============================== ===============================================

Conventions shared by everything in the module:

- **Blocks are named vectors.** A block is a ``str`` name and a dimension
  ``d``. Block values are ``(d,)`` arrays; an ensemble stores block ``b`` as a
  ``(n_particles, d)`` array, one particle per row. Names are ordered: the order
  in which blocks were given is the order every method reports them in.
- **The particles axis is a batch axis** in the sense of :mod:`enskit.linalg`,
  so ``op.matvec(ensemble["x"])`` applies an operator to every particle.
- **Distributions are unbatched.** A family of distributions is expressed with
  :func:`jax.vmap`; a pytree rebuilt with stacked leaves is a vmapped family
  that refuses every method until applied under :func:`jax.vmap`.
- **Block values may be passed as keywords.** Every argument that is a set of
  block values (or covariances) accepts a positional-only mapping, keywords,
  or both: ``g.condition(y=y_obs)`` is ``g.condition({"y": y_obs})``. The
  mapping form is always available, and is required for a name that is not a
  Python identifier or that collides with one of the function's own
  parameters.
- **Samples and particles.** A *sample* is a draw from any distribution; a
  *particle* is an element of an :class:`Ensemble`. An ensemble's particles
  are treated as samples wherever a function asks for samples.
- **Anomalies are raw deviations from the mean**; empirical covariances use the
  divisor ``n_particles - 1`` (unweighted) or ``1 - sum(w**2)`` (weighted).
- **Randomness enters through an explicit typed key**
  (:func:`jax.random.key`), consumed whole. Nothing stores or advances a key.

Notes
-----
The conditioning arithmetic is specified by the "Distribution contract" page
of the documentation, which is normative. Every conditioning path goes
through one whitened singular value decomposition,
:class:`enskit.linalg.IdentityPlusGram`; see :meth:`Gaussian.condition`.
"""
