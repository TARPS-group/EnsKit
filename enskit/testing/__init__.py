"""Conformance checks for code you write against EnsKit's interfaces.

Each check calls your code the way the layers will and raises
``AssertionError`` naming the obligation it violates, and returns ``None``
when every obligation holds. They are for your own test suite: none of them
is called by the library, and nothing in the layers imports this module.

================================ ==============================================
function                         checks
================================ ==============================================
:func:`check_simulator`          a simulator against the simulator contract of
                                 :mod:`enskit.maps`: containers, shapes and
                                 dtypes, determinism, and row independence
:func:`check_update_rule`        a :class:`~enskit.kalman.UpdateRule`: result
                                 structure, dtype, value-free builds, keys,
                                 exactness on a linear-Gaussian problem,
                                 ``jit`` and ``vmap``
:func:`check_conditional_map`    a :class:`~enskit.distribution.ConditionalMap`:
                                 values, result structure, pointwise action,
                                 keys, agreement with
                                 :meth:`~enskit.distribution.Gaussian.condition`
================================ ==============================================

An operator is checked with :func:`enskit.linalg.testing.check_operator`,
which lives beside the operator layer because that layer's own tests run it.

Notes
-----
Row independence is the reason the simulator check exists. A simulator whose
output for one particle depends on the others returns the right shapes and
finite numbers, so nothing inside a pushforward can see the defect; calling it
on a permutation and on a subset of the particles, from outside, can. The
obligations of the other two are those of the "Kalman contract" and
"Distribution contract" pages of the documentation.
"""

from ._conditional_map import check_conditional_map
from ._simulator import check_simulator
from ._update_rule import check_update_rule

__all__ = ["check_simulator", "check_update_rule", "check_conditional_map"]

for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
