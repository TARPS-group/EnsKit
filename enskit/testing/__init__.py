"""Conformance checks for code you write against EnsKit's interfaces.

Each check calls your code the way the layers will and raises
``AssertionError`` naming the obligation it violates. They are for your own
test suite: none of them is called by the library, and nothing in the
layers imports this module.

========================= =====================================================
function                  checks
========================= =====================================================
:func:`check_simulator`   a simulator against the simulator contract of
                          :mod:`enskit.maps`: containers, shapes and dtypes,
                          determinism, and row independence
========================= =====================================================

An operator is checked with :func:`enskit.linalg.testing.check_operator`,
which lives beside the operator layer because that layer's own tests run it.

Notes
-----
Row independence is the reason this module exists. A simulator whose output
for one particle depends on the others returns the right shapes and finite
numbers, so nothing inside a pushforward can see the defect; calling it on a
permutation and on a subset of the particles, from outside, can.
"""
from ._simulator import check_simulator

__all__ = ["check_simulator"]

for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
