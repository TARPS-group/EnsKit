"""Structured linear operators.

Operators represent matrices implicitly, by how they act on vectors, so that
known structure is exploited instead of storing or factorizing dense arrays.
A block-diagonal covariance, for example, is solved block by block at a cost
that is the sum over blocks rather than cubic in the total size.

This is a lean layer aimed at what ensemble Kalman methods need — applying
operators and their transposes, solving against them, and taking square roots
to sample and whiten — rather than a general-purpose linear algebra library.
Its behavior is specified by the "Linear operator contract" page of the
documentation.

- :mod:`~enskit.linalg.base` defines the class hierarchy, the array-shape
  convention, and how to add a new operator.
- :mod:`~enskit.linalg.elementary` holds operators defined by their own
  arrays.
- :mod:`~enskit.linalg.composite` holds operators built from other operators,
  and the factory functions that construct them.
- :mod:`~enskit.linalg.gram` holds :class:`IdentityPlusGram`, the operator
  :math:`I + S S^\top` computed from one SVD of :math:`S`, with derivative
  rules that stay finite at degenerate spectra.
- :mod:`~enskit.linalg.testing` holds conformance checks for new operator
  types.

Import :mod:`enskit` before creating any array, so that float64 is enabled
first.
"""
from .base import (
    LinOp,
    PSDLinOp,
    SquareLinOp,
    UnsupportedOpError,
    debug_checks,
    dense_fallback,
    dense_matvec,
    densify,
    linop,
    set_debug_checks,
    static_field,
    tri_solve,
    value_check,
)
from .composite import (
    BlockDiag,
    HStack,
    LowRankUpdate,
    Product,
    PSDBlockDiag,
    PSDDiagCongruence,
    PSDScaled,
    Scaled,
    SquareScaled,
    Transposed,
    block_diag,
    diag_congruence,
    hstack,
    product,
)
from .elementary import (
    Dense,
    DensePSD,
    DenseSquare,
    Identity,
    PSDDiagonal,
    PSDLowRank,
    Triangular,
    Zero,
)
from .gram import IdentityPlusGram, IdentityPlusGramInverseSqrt

__all__ = [
    # hierarchy and machinery
    "LinOp",
    "SquareLinOp",
    "PSDLinOp",
    "UnsupportedOpError",
    "densify",
    "dense_fallback",
    "linop",
    "static_field",
    "dense_matvec",
    "tri_solve",
    "set_debug_checks",
    "debug_checks",
    "value_check",
    # elementary operators
    "Identity",
    "Zero",
    "PSDDiagonal",
    "Dense",
    "DenseSquare",
    "Triangular",
    "DensePSD",
    "PSDLowRank",
    # composites
    "Transposed",
    "Scaled",
    "SquareScaled",
    "PSDScaled",
    "Product",
    "HStack",
    "BlockDiag",
    "PSDBlockDiag",
    "PSDDiagCongruence",
    "LowRankUpdate",
    # I + S S^T
    "IdentityPlusGram",
    "IdentityPlusGramInverseSqrt",
    # factories
    "block_diag",
    "product",
    "hstack",
    "diag_congruence",
]
