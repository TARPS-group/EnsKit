r"""Ensemble Kalman updates: approximate conditioning of an ensemble's particles.

An ensemble Kalman update takes the particles of an ensemble over *target*
and *given* blocks, and a value :math:`y^*` for the given blocks, and returns
particles approximating the conditional distribution of the targets at
:math:`y^*`. Every update has the same three stages:

1. **Approximate.** Fit a joint Gaussian to the particles,
   :func:`gaussian_approximation` by default.
2. **Build.** An :class:`UpdateRule` builds a :class:`ParticleUpdate` from the
   particles, the approximation and the *names* of the given blocks. Nothing
   here depends on the given values.
3. **Call.** The particle update is called with the given *values* and
   returns the updated particles.

:func:`update` runs all three in one call.

==================================== ===========================================
object                               is
==================================== ===========================================
:func:`update`                       one ensemble Kalman update, one call
:func:`gaussian_approximation`       the default joint Gaussian approximation
:class:`UpdateRule`                  protocol: builds a particle update
:class:`ParticleUpdate`              protocol: given values in, updated
                                     particles out
:class:`SymmetricSquareRoot`         the deterministic rule (ETKF form)
:class:`Matheron`                    the stochastic rule (Matheron's rule)
:func:`inflate_multiplicative`       scale the anomalies
:func:`inflate_additive`             add centered Gaussian draws
:func:`relax_to_prior_spread`        RTPS: relax the spread toward the prior's
:func:`relax_to_prior_perturbations` RTPP: blend posterior and prior anomalies
==================================== ===========================================

Conventions shared by everything in the module:

- The *given* blocks are the ones conditioned on; every other block of the
  approximation is a *target* and is updated. The layer knows nothing about
  where the given values came from.
- Updates take **unweighted, finite** particles. A weighted ensemble is
  refused (resample first, with :func:`enskit.distribution.resample`), and a
  failed particle must be repaired or dropped before an update.
- Block values and covariances follow the distribution layer's rules: a
  positional mapping, keywords, or both.
- Randomness enters through typed keys, consumed whole, and every draw is
  pinned.

Notes
-----
The behavior of this module is specified by the "Kalman contract" page of the
documentation, which is normative. Two kinds of map sit underneath. A
:class:`~enskit.distribution.MatheronMap` is *pointwise*: it moves any samples
of the joint, one at a time. A :class:`~enskit.distribution.SquareRootMap`
moves *the particle set it was built from*, as a whole. :class:`Matheron`
applies a pointwise map to the particles; :class:`SymmetricSquareRoot` takes
the square-root map itself.
"""

from ._inflation import (
    inflate_additive,
    inflate_multiplicative,
    relax_to_prior_perturbations,
    relax_to_prior_spread,
)
from ._rules import Matheron, SymmetricSquareRoot
from ._update import ParticleUpdate, UpdateRule, gaussian_approximation, update

__all__ = [
    "update",
    "gaussian_approximation",
    "UpdateRule",
    "ParticleUpdate",
    "SymmetricSquareRoot",
    "Matheron",
    "inflate_multiplicative",
    "inflate_additive",
    "relax_to_prior_spread",
    "relax_to_prior_perturbations",
]

# The modules above are private, so the public names report the package they
# are imported from, in tracebacks, ``type()`` and pickles.
for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
