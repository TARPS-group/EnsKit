r"""Kronecker products of operators.

===========================  ===============================================
name                         represents
===========================  ===============================================
:class:`Kronecker`           :math:`A \otimes B` for any two operators
:class:`SquareKronecker`     :math:`A \otimes B` for two square operators
:class:`PSDKronecker`        :math:`A \otimes B` for two PSD operators
:func:`kron`                 the factory, choosing the class
===========================  ===============================================

For :math:`A` of shape :math:`(n_A, k_A)` and :math:`B` of shape
:math:`(n_B, k_B)`, the Kronecker product has shape
:math:`(n_A n_B,\ k_A k_B)` and entries

.. math::

    (A \otimes B)_{\,i n_B + p,\ j k_B + q} = A_{ij}\, B_{pq},
    \qquad 0 \le i < n_A,\ 0 \le j < k_A,\ 0 \le p < n_B,\ 0 \le q < k_B,

so block :math:`(i, j)` of the product is :math:`A_{ij} B`. The first
operand's index is the slow one, the ordering ``numpy.kron`` uses. A vector
the product is applied to is laid out the same way: its trailing axis,
reshaped row-major to :math:`X \in \mathbb{R}^{k_A \times k_B}`, gives

.. math::

    (A \otimes B)\, \operatorname{vec}(X) = \operatorname{vec}(A X B^\top),

where :math:`\operatorname{vec}` stacks the rows of a matrix. Every
operation that takes a vector applies :math:`B` along the last axis of the
reshaped vector and :math:`A` along the one before it, and ``diag``,
``logdet`` and ``factor`` combine the operands' own results, so only
``to_dense`` forms an array of side :math:`n_A n_B`.

Build Kronecker products with :func:`kron`, which picks the most capable
class for the operands, as ``c * op`` does for the scaled operators.
Constructing a class directly is allowed but never upgrades to a more
capable one.

Notes
-----
Reversing the order of the operands is not a loud error. :math:`B \otimes A`
is a different matrix from :math:`A \otimes B`, but it is positive definite
whenever :math:`A \otimes B` is, and it always has the same shape,
:math:`(n_A n_B,\ k_A k_B)`, so a reversed ordering gives a valid operator
with the wrong meaning at any sizes.

References
----------
Van Loan, C. F. (2000). The ubiquitous Kronecker product. *Journal of
Computational and Applied Mathematics*, 123(1–2), 85–100.
"""
from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array

from .base import (
    LinOp,
    PSDLinOp,
    SquareLinOp,
    _broadcast_batch,
    _check_not_family,
    _construct_unchecked,
    linop,
)

__all__ = ["Kronecker", "SquareKronecker", "PSDKronecker", "kron"]


@linop
class Kronecker(LinOp):
    r"""The Kronecker product :math:`A \otimes B` of two operators.

    For :math:`A` of shape :math:`(n_A, k_A)`, :math:`B` of shape
    :math:`(n_B, k_B)` and a vector :math:`x` whose row-major reshape to
    :math:`(k_A, k_B)` is :math:`X`,

    .. math::

        (A \otimes B)\, x = \operatorname{vec}(A X B^\top), \qquad
        (A \otimes B)^\top = A^\top \otimes B^\top,

    with :math:`\operatorname{vec}` stacking rows. Either operand may be
    rectangular, and the product provides application and transposition
    only. This is what :meth:`PSDKronecker.factor` returns when an operand's
    factor is not a :class:`~enskit.linalg.SquareLinOp`.

    Parameters
    ----------
    A
        The operand whose index is the slow one, of shape ``(n_A, k_A)``.
    B
        The operand whose index is the fast one, of shape ``(n_B, k_B)``.

    Raises
    ------
    TypeError
        If either operand is not a :class:`~enskit.linalg.LinOp`.
    ValueError
        If either operand is a vmapped family.
    """

    A: LinOp
    B: LinOp

    def __post_init__(self) -> None:
        name = type(self).__name__
        for field_name, op in (("A", self.A), ("B", self.B)):
            if not isinstance(op, LinOp):
                raise TypeError(
                    f"{name}.{field_name} must be a LinOp, got {type(op).__name__}"
                )
            _check_not_family(name, op)

    @property
    def shape(self) -> tuple[int, int]:
        (n_a, k_a), (n_b, k_b) = self.A.shape, self.B.shape
        return (n_a * n_b, k_a * k_b)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return _broadcast_batch(
            type(self).__name__, self.A.batch_shape, self.B.batch_shape
        )

    @property
    def T(self) -> Kronecker:  # noqa: N802 - mirrors the NumPy attribute
        r"""The transpose, :math:`A^\top \otimes B^\top`.

        Of the class :func:`kron` would choose for the operands' transposes,
        so the transpose of a :class:`SquareKronecker` of operands whose own
        transposes can solve also solves. Works on a vmapped family too.

        Notes
        -----
        Because the class is chosen from the transposes, ``T`` may be more
        capable than the operator: the transpose of a directly constructed
        ``Kronecker`` of two PSD operators is a :class:`PSDKronecker`. It is
        built without running the constructor, which would reject a
        family's children.
        """
        A, B = self.A.T, self.B.T
        return _construct_unchecked(_kronecker_class(A, B), A=A, B=B)

    def _matvec(self, x: Array) -> Array:
        return _kron_apply(x, self.A.matmat, self.B.matvec, self._in_core)

    def _rmatvec(self, x: Array) -> Array:
        return _kron_apply(x, self.A.rmatmat, self.B.rmatvec, self._out_core)

    def _to_dense(self) -> Array:
        return jnp.kron(self.A.to_dense(), self.B.to_dense())

    @property
    def _in_core(self) -> tuple[int, int]:
        """``(k_A, k_B)``, the shape an operand of ``matvec`` is reshaped to."""
        return (self.A.shape[1], self.B.shape[1])

    @property
    def _out_core(self) -> tuple[int, int]:
        """``(n_A, n_B)``, the shape an operand of ``rmatvec`` is reshaped to."""
        return (self.A.shape[0], self.B.shape[0])


@linop
class SquareKronecker(Kronecker, SquareLinOp):
    r"""The Kronecker product of two square operators; adds the square
    operations.

    With :math:`n_A` and :math:`n_B` the operands' sides,

    .. math::

        (A \otimes B)^{-1} &= A^{-1} \otimes B^{-1}, \\
        \log\lvert\det(A \otimes B)\rvert &= n_B \log\lvert\det A\rvert
            + n_A \log\lvert\det B\rvert, \\
        \operatorname{diag}(A \otimes B) &= \operatorname{diag}(A) \otimes
            \operatorname{diag}(B),

    each computed from the operands' own operations, so an operation is
    supported exactly when both operands support it.

    Parameters
    ----------
    A
        The square operand whose index is the slow one, of side ``n_A``.
    B
        The square operand whose index is the fast one, of side ``n_B``.

    Raises
    ------
    TypeError
        If either operand is not a :class:`~enskit.linalg.SquareLinOp`.
    ValueError
        If either operand is a vmapped family.

    Notes
    -----
    In the log-determinant each operand's term is multiplied by the side of
    the *other* operand. The two coefficients coincide when the operands
    have the same side, so a test with equal sides cannot tell the correct
    pairing from the swapped one.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        for field_name, op in (("A", self.A), ("B", self.B)):
            if not isinstance(op, SquareLinOp):
                raise TypeError(
                    f"SquareKronecker.{field_name} must be a SquareLinOp, got "
                    f"{type(op).__name__}; use kron() to choose the class"
                )

    def supports(self, name: str) -> bool:
        return (
            super().supports(name) and self.A.supports(name) and self.B.supports(name)
        )

    def _solve(self, b: Array) -> Array:
        return _kron_apply(b, self.A.solve_mat, self.B.solve, self._in_core)

    def _logdet(self) -> Array:
        return self.B.dim * self.A.logdet() + self.A.dim * self.B.logdet()

    def _diag(self) -> Array:
        d_a, d_b = self.A.diag(), self.B.diag()
        outer = d_a[..., :, None] * d_b[..., None, :]
        return outer.reshape(*outer.shape[:-2], -1)


@linop
class PSDKronecker(SquareKronecker, PSDLinOp):
    r"""The Kronecker product of two PSD operators; adds the PSD operations.

    With :math:`L_A L_A^\top = A`, :math:`L_B L_B^\top = B`, and whiteners
    :math:`W_A A W_A^\top = I` and :math:`W_B B W_B^\top = I`,

    .. math::

        (L_A \otimes L_B)(L_A \otimes L_B)^\top &= A \otimes B, \\
        (W_A \otimes W_B)(A \otimes B)(W_A \otimes W_B)^\top &= I,

    so ``factor`` returns ``kron(A.factor(), B.factor())`` and ``whiten``
    applies :math:`W_A \otimes W_B`, one operand at a time. The factor's
    class is the one :func:`kron` picks for the operands' factors: a
    :class:`SquareKronecker` (or :class:`PSDKronecker`) when both are
    :class:`~enskit.linalg.SquareLinOp`, which then solves when they do — so
    the Kronecker product of two :class:`~enskit.linalg.Triangular` Cholesky
    factors keeps its triangular solve — and a plain :class:`Kronecker`
    otherwise, as when a factor is rectangular or a square
    :class:`~enskit.linalg.Dense`. The square operations are those of
    :class:`SquareKronecker`.

    Parameters
    ----------
    A
        The PSD operand whose index is the slow one, of side ``n_A``.
    B
        The PSD operand whose index is the fast one, of side ``n_B``.

    Raises
    ------
    TypeError
        If either operand is not a :class:`~enskit.linalg.PSDLinOp`.
    ValueError
        If either operand is a vmapped family.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        for field_name, op in (("A", self.A), ("B", self.B)):
            if not isinstance(op, PSDLinOp):
                raise TypeError(
                    f"PSDKronecker.{field_name} must be a PSDLinOp, got "
                    f"{type(op).__name__}; use kron() to choose the class"
                )

    @property
    def T(self) -> PSDKronecker:  # noqa: N802 - mirrors the NumPy attribute
        """The transpose, which is the operator itself, being self-adjoint."""
        return self

    def _rmatvec(self, x: Array) -> Array:
        return self._matvec(x)

    def _factor(self) -> LinOp:
        return kron(self.A.factor(), self.B.factor())

    def _whiten(self, x: Array) -> Array:
        return _kron_apply(x, self.A.whiten_mat, self.B.whiten, self._in_core)


def kron(A: LinOp, B: LinOp) -> Kronecker:
    r"""Build the Kronecker product :math:`A \otimes B`, choosing the class.

    Returns a :class:`PSDKronecker` when both operands are
    :class:`~enskit.linalg.PSDLinOp`, a :class:`SquareKronecker` when both
    are :class:`~enskit.linalg.SquareLinOp`, and a :class:`Kronecker`
    otherwise. The first operand's index is the slow one, as in
    ``numpy.kron``:

    .. math::

        (A \otimes B)_{\,i n_B + p,\ j k_B + q} = A_{ij}\, B_{pq},

    for :math:`A` of shape :math:`(n_A, k_A)` and :math:`B` of shape
    :math:`(n_B, k_B)`.

    Parameters
    ----------
    A
        The operand whose index is the slow one.
    B
        The operand whose index is the fast one.

    Raises
    ------
    TypeError
        If either operand is not a :class:`~enskit.linalg.LinOp`.
    ValueError
        If either operand is a vmapped family.
    """
    for op in (A, B):
        if not isinstance(op, LinOp):
            raise TypeError(
                f"kron() arguments must be operators, got {type(op).__name__}"
            )
    return _kronecker_class(A, B)(A, B)


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------


def _kronecker_class(A: LinOp, B: LinOp) -> type[Kronecker]:
    """The most capable Kronecker class for two operands' levels."""
    if isinstance(A, PSDLinOp) and isinstance(B, PSDLinOp):
        return PSDKronecker
    if isinstance(A, SquareLinOp) and isinstance(B, SquareLinOp):
        return SquareKronecker
    return Kronecker


def _kron_apply(
    x: Array,
    apply_a: Callable[[Array], Array],
    apply_b: Callable[[Array], Array],
    core: tuple[int, int],
) -> Array:
    """Apply a Kronecker product of two maps to the trailing axis of ``x``.

    Reshapes the trailing axis to ``core``, applies ``apply_b`` — a vector
    method, contracting the last axis — and then ``apply_a`` — a matrix
    method, contracting the axis before it — and flattens the result. The
    result's core shape is read from what the maps return rather than
    passed in, because it differs from ``core`` whenever ``A`` or ``B`` is
    rectangular: for ``matvec`` the reshaped vector is ``(k_A, k_B)``, the
    intermediate ``(k_A, n_B)`` and the result ``(n_A, n_B)``.
    """
    X = x.reshape(*x.shape[:-1], *core)
    Z = apply_a(apply_b(X))
    return Z.reshape(*Z.shape[:-2], Z.shape[-2] * Z.shape[-1])
