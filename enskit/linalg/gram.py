r"""The operator :math:`I + S S^\top`, and its inverse square root.

=======================================  ======================================
class                                    represents
=======================================  ======================================
:class:`IdentityPlusGram`                :math:`A = I_k + S S^\top` for a
                                         ``(k, N)`` array :math:`S`
:class:`IdentityPlusGramInverseSqrt`     :math:`A^{-1/2}`, as returned by
                                         :meth:`IdentityPlusGram.inverse_sqrt`
=======================================  ======================================

Both are built from one thin singular value decomposition of :math:`S`,

.. math::

    S = U \Sigma V^\top, \qquad U \in \mathbb R^{k \times r},\
    \Sigma = \operatorname{diag}(\sigma_1, \dots, \sigma_r),\
    V \in \mathbb R^{N \times r}, \qquad r = \min(k, N),

computed at construction and stored. Every operation is a function of the
singular values applied in the basis :math:`U` and completed by the identity
on its orthogonal complement, so neither :math:`S S^\top` nor
:math:`S^\top S` is formed: forming either squares the condition number and
rounds away every :math:`\sigma_i < \sqrt{\varepsilon}\,\sigma_{\max}`.

Derivatives
-----------
The operations that read the decomposition — ``solve``,
:meth:`~IdentityPlusGram.solve_factor`, ``logdet``, ``whiten`` and the
inverse square root — carry custom derivative rules written in terms of
:math:`S` alone. Their first derivatives are finite and correct at every
:math:`S`, including at exactly repeated and exactly zero singular values,
where the derivative of a plain SVD is ``nan``. Such spectra are routine for
this operator: a column of :math:`S` that is zero, or two columns that are
orthogonal with equal norms. Derivatives with respect to an operator passed
as a pytree flow through its field ``S``; the stored decomposition receives
no derivative of its own.

Second derivatives are correct wherever the singular values are distinct
and nonzero. At exactly repeated or zero singular values only the first
derivative is guaranteed finite: second derivatives of ``logdet`` are finite
there, but :func:`jax.hessian` of the other operations may return ``nan``.

References
----------
Higham, N. J. (2008). *Functions of Matrices: Theory and Computation*. SIAM.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .base import (
    LinOp,
    PSDLinOp,
    _broadcast_batch,
    _check_core_rank,
    _check_real,
    _check_vec,
    _construct_unchecked,
    dense_matvec,
    linop,
)
from .elementary import Dense, Identity

__all__ = ["IdentityPlusGram", "IdentityPlusGramInverseSqrt"]


@linop
class IdentityPlusGram(PSDLinOp):
    r"""The operator :math:`A = I_k + S S^\top` for a ``(k, N)`` array :math:`S`.

    .. math::

        A = I_k + S S^\top, \qquad S = U \Sigma V^\top \ \text{(thin SVD)},

    so that :math:`A = I_k + U \Sigma^2 U^\top` and every eigenvalue of
    :math:`A` is at least 1. The thin SVD is computed once, at construction.

    ==================  ===================================================
    operation           computed as
    ==================  ===================================================
    ``matvec``          :math:`x + S(S^\top x)`
    ``solve``           :math:`b + U\big((I + \Sigma^2)^{-1} - I\big)U^\top b`
    ``logdet``          :math:`\sum_i \log(1 + \sigma_i^2)`
    ``whiten``          :math:`A^{-1/2} x`, the symmetric whitener
    ``factor``          :math:`[\,I_k \ \ S\,]`, of shape ``(k, k + N)``
    ``diag``            :math:`1 + \sum_j S_{ij}^2`
    ``solve_factor``    :math:`A^{-1} S b`
    ``inverse_sqrt``    :math:`A^{-1/2}`, as an operator
    ==================  ===================================================

    Parameters
    ----------
    S
        The ``(k, N)`` array, exactly 2-D, both sizes at least 1. No relation
        between ``k`` and ``N`` is required.

    Raises
    ------
    ValueError
        If ``S`` is not exactly 2-D with positive sizes.
    TypeError
        If ``S`` is complex or boolean.

    Notes
    -----
    ``factor`` is exact but wide; a caller sampling from :math:`A` usually
    has a cheaper route through whatever produced :math:`S`.

    Finiteness of ``S`` is not checked, even in debug mode: a non-finite
    ``S`` gives non-finite results, which the caller that built ``S`` is
    better placed to diagnose, since it knows the likely cause.

    The fields are ``S`` and its decomposition ``U``, ``sigma`` and ``Vt``,
    of shapes ``(k, r)``, ``(r,)`` and ``(r, N)``. Replacing ``S`` with
    :func:`dataclasses.replace` pairs it with the old decomposition; build
    a new ``IdentityPlusGram(S)`` instead.

    References
    ----------
    Higham, N. J. (2008). *Functions of Matrices: Theory and Computation*.
    SIAM.
    """

    S: Array
    U: Array
    sigma: Array
    Vt: Array

    def __init__(self, S) -> None:
        S = jnp.asarray(S)
        _check_core_rank("IdentityPlusGram", "S", S, 2)
        _check_real("IdentityPlusGram", "S", S)
        if not jnp.issubdtype(S.dtype, jnp.floating):
            S = S.astype(jnp.result_type(float))
        U, sigma, Vt = jnp.linalg.svd(S, full_matrices=False)
        object.__setattr__(self, "S", S)
        object.__setattr__(self, "U", U)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "Vt", Vt)

    @property
    def shape(self) -> tuple[int, int]:
        k = self.S.shape[-2]
        return (k, k)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return _broadcast_batch(
            "IdentityPlusGram",
            self.S.shape[:-2],
            self.U.shape[:-2],
            self.sigma.shape[:-1],
            self.Vt.shape[:-2],
        )

    def solve_factor(self, b) -> Array:
        r"""Apply :math:`A^{-1} S` to a batch of vectors.

        .. math::

            w = (I + S S^\top)^{-1} S\, b
              = U \operatorname{diag}\!\Big(\frac{\sigma_i}{1+\sigma_i^2}\Big)
                V^\top b
              = S\,(I + S^\top S)^{-1} b .

        The multipliers :math:`\sigma_i/(1+\sigma_i^2)` are at most 1/2, so
        :math:`\lVert w \rVert \le \lVert b \rVert / 2` however large or
        collapsed :math:`S` is.

        Parameters
        ----------
        b
            Array of shape ``(..., N)`` — vectors along the trailing axis,
            any number of leading batch axes.

        Returns
        -------
        Array
            Shape ``(..., k)``, the batch axes of ``b`` preserved.

        Raises
        ------
        ValueError
            If ``b`` has no axes or its trailing axis is not ``N``, or if this
            operator is a vmapped family.

        Notes
        -----
        The derivative rule, with :math:`M = A^{-1}`, is

        .. math::

            dw = M S\,(db - dS^\top w) + M\, dS\,(b - S^\top w),

        whose two terms are this method and ``solve`` applied to other
        vectors: rational in :math:`S`, with no division by differences of
        singular values.
        """
        self._check_not_vmap_family("solve_factor")
        b = _check_vec(self, "solve_factor", b, self.S.shape[-1])
        return _solve_factor(self.S, self.U, self.sigma, self.Vt, b)

    def inverse_sqrt(self) -> IdentityPlusGramInverseSqrt:
        r"""Return the symmetric inverse square root, :math:`A^{-1/2}`.

        .. math::

            A^{-1/2} = I_k + U\big((I + \Sigma^2)^{-1/2} - I\big) U^\top,

        exact at every rank: the identity term supplies the orthogonal
        complement of :math:`U`'s columns, which a thin decomposition does
        not store. It reuses this operator's decomposition, so it computes
        no new SVD, and nothing of size ``(k, k)`` is formed until
        ``to_dense`` is called.

        Returns
        -------
        IdentityPlusGramInverseSqrt
            The operator :math:`A^{-1/2}`, of side ``k``.

        Raises
        ------
        ValueError
            If this operator is a vmapped family.
        """
        self._check_not_vmap_family("inverse_sqrt")
        return _construct_unchecked(
            IdentityPlusGramInverseSqrt, S=self.S, U=self.U, sigma=self.sigma
        )

    def _matvec(self, x: Array) -> Array:
        return x + dense_matvec(self.S, dense_matvec(self.S.swapaxes(-1, -2), x))

    def _matmat(self, X: Array) -> Array:
        return X + self.S @ (self.S.swapaxes(-1, -2) @ X)

    def _solve(self, b: Array) -> Array:
        return _resolvent(self.S, self.U, self.sigma, self.Vt, b)

    def _logdet(self) -> Array:
        return _logdet(self.S, self.U, self.sigma, self.Vt)

    def _diag(self) -> Array:
        return 1.0 + jnp.sum(self.S * self.S, axis=-1)

    def _factor(self) -> LinOp:
        from .composite import HStack

        return HStack((Identity(self.dim), Dense(self.S)))

    def _whiten(self, x: Array) -> Array:
        return _inverse_sqrt_apply(self.S, self.U, self.sigma, x)

    def _to_dense(self) -> Array:
        eye = jnp.eye(self.dim, dtype=self.S.dtype)
        return eye + self.S @ self.S.swapaxes(-1, -2)


@linop
class IdentityPlusGramInverseSqrt(PSDLinOp):
    r"""The symmetric inverse square root :math:`(I_k + S S^\top)^{-1/2}`.

    .. math::

        T = (I_k + S S^\top)^{-1/2}
          = I_k + U\big((I + \Sigma^2)^{-1/2} - I\big) U^\top,
        \qquad S = U \Sigma V^\top \ \text{(thin SVD)}.

    What :meth:`IdentityPlusGram.inverse_sqrt` returns, sharing that
    operator's decomposition. :math:`T` is symmetric positive definite with
    eigenvalues in :math:`(0, 1]`. It provides ``matvec`` and ``to_dense``;
    applying it costs :math:`O(kr)` per vector.

    Parameters
    ----------
    S
        The ``(k, N)`` array, exactly 2-D, both sizes at least 1. Constructing
        directly computes the thin SVD; prefer
        :meth:`IdentityPlusGram.inverse_sqrt` when that operator exists.

    Raises
    ------
    ValueError
        If ``S`` is not exactly 2-D with positive sizes.
    TypeError
        If ``S`` is complex or boolean.

    Notes
    -----
    The derivative rule is the Daleckii–Krein formula for
    :math:`f(\lambda) = (1 + \lambda)^{-1/2}` (Higham, 2008), written in the thin
    basis. With :math:`P = I - U U^\top`, :math:`s_i = (1+\sigma_i^2)^{1/2}`
    and :math:`dA = dS\, S^\top + S\, dS^\top`,

    .. math::

        dT = U \big(G \circ (U^\top dA\, U)\big) U^\top
           + U \operatorname{diag}(g)\, U^\top dA\, P
           + P\, dA\, U \operatorname{diag}(g)\, U^\top,

    .. math::

        G_{ij} = \frac{-1}{s_i s_j (s_i + s_j)}, \qquad
        g_i = \frac{-1}{s_i (s_i + 1)} .

    :math:`G_{ij}` is the divided difference of :math:`f` between the
    eigenvalues :math:`\sigma_i^2` and :math:`\sigma_j^2`, and :math:`g_i`
    the one between :math:`\sigma_i^2` and the eigenvalue 0 of the
    complement, so neither divides by a difference of singular values. The
    complement–complement block vanishes because :math:`P S = 0`.

    References
    ----------
    Higham, N. J. (2008). *Functions of Matrices: Theory and Computation*.
    SIAM.
    """

    S: Array
    U: Array
    sigma: Array

    def __init__(self, S) -> None:
        gram = IdentityPlusGram(S)
        object.__setattr__(self, "S", gram.S)
        object.__setattr__(self, "U", gram.U)
        object.__setattr__(self, "sigma", gram.sigma)

    @property
    def shape(self) -> tuple[int, int]:
        k = self.S.shape[-2]
        return (k, k)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return _broadcast_batch(
            "IdentityPlusGramInverseSqrt",
            self.S.shape[:-2],
            self.U.shape[:-2],
            self.sigma.shape[:-1],
        )

    def _matvec(self, x: Array) -> Array:
        return _inverse_sqrt_apply(self.S, self.U, self.sigma, x)

    def _to_dense(self) -> Array:
        return _inverse_sqrt_dense(self.S, self.U, self.sigma)


# ---------------------------------------------------------------------------
# private: the kernels and their derivative rules
# ---------------------------------------------------------------------------
#
# Each kernel takes S and its stored decomposition as separate arguments and
# computes its value from the decomposition. Each rule reads only the tangent
# of S (and of the vector operand), never those of U, sigma and Vt: it treats
# the decomposition as the function of S it is. The decomposition is not
# wrapped in stop_gradient, deliberately: a second derivative differentiates
# a rule's own arithmetic, which reads U and sigma, and must then see their
# dependence on S. Through the SVD's derivative that is exact at distinct
# nonzero singular values and nan elsewhere; with stop_gradient it would be
# finite and wrong.


def _transpose(M: Array) -> Array:
    return M.swapaxes(-1, -2)


@jax.custom_jvp
def _resolvent(S: Array, U: Array, sigma: Array, Vt: Array, b: Array) -> Array:
    """``(I + S S^T)^{-1} b`` for ``b`` of shape ``(..., k)``."""
    modifier = -(sigma**2) / (1.0 + sigma**2)
    return b + dense_matvec(U, modifier * dense_matvec(_transpose(U), b))


@_resolvent.defjvp
def _resolvent_jvp(primals, tangents):
    # d(A^{-1} b) = A^{-1}(db - dS S^T y) - A^{-1} S (dS^T y), y = A^{-1} b
    S, U, sigma, Vt, b = primals
    dS, _, _, _, db = tangents
    y = _resolvent(S, U, sigma, Vt, b)
    St_y = dense_matvec(_transpose(S), y)
    dy = _resolvent(S, U, sigma, Vt, db - dense_matvec(dS, St_y)) - _solve_factor(
        S, U, sigma, Vt, dense_matvec(_transpose(dS), y)
    )
    return y, dy


@jax.custom_jvp
def _solve_factor(S: Array, U: Array, sigma: Array, Vt: Array, b: Array) -> Array:
    """``(I + S S^T)^{-1} S b`` for ``b`` of shape ``(..., N)``."""
    return dense_matvec(U, (sigma / (1.0 + sigma**2)) * dense_matvec(Vt, b))


@_solve_factor.defjvp
def _solve_factor_jvp(primals, tangents):
    # d(A^{-1} S b) = A^{-1} S (db - dS^T w) + A^{-1} dS (b - S^T w), w = A^{-1} S b.
    # Grouping by A^{-1} S keeps the large factor S out of a subtraction:
    # b - S^T w = (I + S^T S)^{-1} b is bounded by |b|.
    S, U, sigma, Vt, b = primals
    dS, _, _, _, db = tangents
    w = _solve_factor(S, U, sigma, Vt, b)
    residual = b - dense_matvec(_transpose(S), w)
    dw = _solve_factor(
        S, U, sigma, Vt, db - dense_matvec(_transpose(dS), w)
    ) + _resolvent(S, U, sigma, Vt, dense_matvec(dS, residual))
    return w, dw


@jax.custom_jvp
def _logdet(S: Array, U: Array, sigma: Array, Vt: Array) -> Array:
    """``log det(I + S S^T)``, a 0-d array."""
    return jnp.sum(jnp.log1p(sigma**2), axis=-1)


@_logdet.defjvp
def _logdet_jvp(primals, tangents):
    # d log det A = tr(A^{-1} dA) = 2 <A^{-1} S, dS>
    S, U, sigma, Vt = primals
    dS = tangents[0]
    identity = jnp.eye(S.shape[-1], dtype=S.dtype)
    gain = _transpose(_solve_factor(S, U, sigma, Vt, identity))  # A^{-1} S, (k, N)
    return _logdet(S, U, sigma, Vt), 2.0 * jnp.sum(gain * dS)


def _inverse_sqrt_modifier(sigma: Array) -> Array:
    # Written as the difference rather than an equivalent rational form so
    # that it is exactly 0.0 once sigma**2 is below the resolution of 1.0.
    return 1.0 / jnp.sqrt(1.0 + sigma**2) - 1.0


def _divided_differences(sigma: Array) -> tuple[Array, Array]:
    """``G`` and ``g`` of the Daleckii-Krein rule, from the singular values."""
    s = jnp.sqrt(1.0 + sigma**2)
    G = -1.0 / (s[:, None] * s[None, :] * (s[:, None] + s[None, :]))
    g = -1.0 / (s * (s + 1.0))
    return G, g


@jax.custom_jvp
def _inverse_sqrt_apply(S: Array, U: Array, sigma: Array, x: Array) -> Array:
    """``(I + S S^T)^{-1/2} x`` for ``x`` of shape ``(..., k)``."""
    modifier = _inverse_sqrt_modifier(sigma)
    return x + dense_matvec(U, modifier * dense_matvec(_transpose(U), x))


@_inverse_sqrt_apply.defjvp
def _inverse_sqrt_apply_jvp(primals, tangents):
    S, U, sigma, x = primals
    dS, _, _, dx = tangents
    y = _inverse_sqrt_apply(S, U, sigma, x)
    G, g = _divided_differences(sigma)
    Ut, St, dSt = _transpose(U), _transpose(S), _transpose(dS)
    St_U = St @ U  # (N, r), equal to V Sigma
    half = (Ut @ dS) @ St_U  # U^T dS S^T U, (r, r)
    H = half + _transpose(half)  # U^T dA U

    def apply_dA(v):  # noqa: N802 - (dS S^T + S dS^T) v, along the trailing axis
        return dense_matvec(dS, dense_matvec(St, v)) + dense_matvec(
            S, dense_matvec(dSt, v)
        )

    def project_out(v):  # P v = v - U U^T v
        return v - dense_matvec(U, dense_matvec(Ut, v))

    Ut_x = dense_matvec(Ut, x)
    inner = dense_matvec(U, dense_matvec(G * H, Ut_x))
    left = dense_matvec(U, g * dense_matvec(Ut, apply_dA(project_out(x))))
    right = project_out(apply_dA(dense_matvec(U, g * Ut_x)))
    dy = _inverse_sqrt_apply(S, U, sigma, dx) + inner + left + right
    return y, dy


@jax.custom_jvp
def _inverse_sqrt_dense(S: Array, U: Array, sigma: Array) -> Array:
    """``(I + S S^T)^{-1/2}`` as a dense ``(k, k)`` array."""
    modifier = _inverse_sqrt_modifier(sigma)
    eye = jnp.eye(S.shape[-2], dtype=U.dtype)
    # (k, r) @ (r, k): both operands are exactly 2-D, so this is the plain
    # matrix product, not a batch of vectors.
    return eye + (U * modifier) @ _transpose(U)


@_inverse_sqrt_dense.defjvp
def _inverse_sqrt_dense_jvp(primals, tangents):
    S, U, sigma = primals
    dS = tangents[0]
    T = _inverse_sqrt_dense(S, U, sigma)
    G, g = _divided_differences(sigma)
    Ut = _transpose(U)
    St_U = _transpose(S) @ U
    half = (Ut @ dS) @ St_U
    H = half + _transpose(half)
    Ut_dA = (Ut @ dS) @ _transpose(S) + _transpose(St_U) @ _transpose(dS)  # (r, k)
    Ut_dA_P = Ut_dA - (Ut_dA @ U) @ Ut
    side = (U * g) @ Ut_dA_P
    dT = U @ (G * H) @ Ut + side + _transpose(side)
    return T, dT
