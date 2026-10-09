r""":class:`Regression` and the computation behind :meth:`Gaussian.regression`."""
from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array

from ..linalg import (
    Dense,
    IdentityPlusGram,
    LinOp,
    PSDLinOp,
    Zero,
    product,
    static_field,
    tri_solve,
    value_check,
)
from . import _common as c
from ._common import distribution_class

__all__ = ["Regression"]


@distribution_class
class Regression:
    r"""The regression of one block on others in a :class:`Gaussian`.

    The conditional distribution of the target block :math:`y` given the
    blocks :math:`x_1, \dots, x_m`, written as an affine function of the
    given values plus an independent residual:

    .. math::

        y \mid x \sim \mathcal N\Big(\sum_{b=1}^{m} A_b x_b + c,\ \Omega\Big),

    with :math:`A_b` = ``coefficients[b]`` of shape ``(d_y, d_b)``, the
    intercept :math:`c` = ``intercept`` of shape ``(d_y,)``, and the
    residual covariance :math:`\Omega` = ``residual_cov``. Returned by
    :meth:`Gaussian.regression`; not constructed directly.

    Its static attributes are ``target``, the target block's name, and
    ``given``, the given blocks' names in block order.
    """

    target: str = static_field()
    given: tuple[str, ...] = static_field()
    _coefficients: tuple[LinOp, ...]
    intercept: Array
    residual_cov: PSDLinOp

    @property
    def coefficients(self) -> dict[str, LinOp]:
        """The operators :math:`A_b`, keyed by given block, in block order."""
        return dict(zip(self.given, self._coefficients, strict=True))

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The family's batch shape, ``()`` for a regression computed directly."""
        shapes = [tuple(self.intercept.shape[:-1]), self.residual_cov.batch_shape]
        shapes += [A.batch_shape for A in self._coefficients]
        return c.broadcast_batch("Regression", *shapes)

    def __repr__(self) -> str:
        """Type name and static sizes, never array contents; never raises."""
        return c.safe_repr(
            lambda: (
                f"Regression(target={self.target!r}, given={self.given}, "
                f"target_dim={self.intercept.shape[-1]})"
            ),
            "Regression",
            lambda: self.batch_shape,
        )


def regression(g, target, given, min_norm: bool) -> Regression:
    """The body of :meth:`Gaussian.regression`; ``g`` is the Gaussian."""
    from ._gaussian import EnsembleGaussian, Gaussian

    where = f"{g!r}.regression"
    c.guard(g, "regression")
    if not isinstance(target, str):
        raise TypeError(
            f"{where}: target must be a block name, got {type(target).__name__}"
        )
    c.check_known(where, g.names, (target,))
    given = c.given_names(where, g.names, given)
    if target in given:
        raise ValueError(f"{where}: block {target!r} is both the target and given")
    if not isinstance(min_norm, bool):
        raise TypeError(
            f"{where}: min_norm must be a bool, got {type(min_norm).__name__}"
        )
    noisy = g._given_kind(where, given)
    i_y = g.names.index(target)
    F_y, D_y, m_y = g._factors[i_y], g._block_covs[i_y], g._means[i_y]
    dims = [g.dims[n] for n in given]

    if noisy:
        for name in given:
            if g._factors[g.names.index(name)] is not None:
                c.require(g._term(name), "whiten", "solve")
        coefs, F_res = _noisy(where, g, given, F_y)
    else:
        N, k = sum(dims), g.latent_dim
        aligned = isinstance(g, EnsembleGaussian)
        rank = k - 1 if aligned else k
        if N <= rank:
            coefs, F_res = _least_squares(where, g, given, F_y)
        elif not min_norm:
            limit = "J - 1" if aligned else "the latent width k"
            raise ValueError(
                f"{where}: the given blocks have N = {N} coordinates, more than "
                f"{limit} = {rank}, so the least-squares coefficients are not "
                f"unique. Pass min_norm=True for the minimum-norm solution, or "
                f"give the blocks an independent term with add_noise(...) for a "
                f"ridge regression."
            )
        else:
            coefs, F_res = _min_norm(where, g, given, F_y, aligned)

    dtype = g._dtype
    if coefs is None:
        coefs = tuple(Zero(g.dims[target], d, dtype=dtype) for d in dims)
    intercept = m_y
    for A, name in zip(coefs, given, strict=True):
        intercept = intercept - A.matvec(g._means[g.names.index(name)])
    c.result_check(where, "intercept", intercept)
    residual = c.build(
        Gaussian,
        names=(target,),
        latent_dim=g.latent_dim,
        _means=(m_y,),
        _factors=(F_res,),
        _block_covs=(D_y,),
    )
    return c.build(
        Regression,
        target=target,
        given=given,
        _coefficients=tuple(coefs),
        intercept=intercept,
        residual_cov=residual.cov(target),
    )


# -- private ---------------------------------------------------------------------


def _split(g, given, make):
    """One coefficient operator per given block, from its column range."""
    out, start = [], 0
    for name in given:
        d = g.dims[name]
        out.append(make(slice(start, start + d)))
        start += d
    return tuple(out)


def _noisy(where, g, given, F_y):
    r"""Ridge: :math:`A = F_y (I + S S^\top)^{-1} F_c^\top D_c^{-1}`."""
    k = g.latent_dim
    if k == 0 or F_y is None:
        return None, F_y
    S, _ = g._whiten_given(where, given, None)
    gram = IdentityPlusGram(S)
    cols = []
    for name in given:
        i = g.names.index(name)
        F, D = g._factors[i], g._block_covs[i]
        if F is None:
            cols.append(jnp.zeros((g.dims[name], k), g._dtype))
        else:
            cols.append(D.solve_mat(F.to_dense()))
    Bt = jnp.concatenate(cols, axis=0).T  # (k, N): F_c^T D_c^{-1}
    C = gram.solve_mat(Bt)  # (k, N): A^{-1} F_c^T D_c^{-1}
    c.result_check(where, "regression coefficients", C)
    coefs = _split(g, given, lambda s: product(F_y, Dense(C[:, s])))
    return coefs, product(F_y, gram.inverse_sqrt())


def _given_rows(g, given) -> Array:
    return jnp.concatenate([g._factors[g.names.index(n)].to_dense() for n in given])


def _rank_check(where: str, R: Array, sizes, what: str) -> None:
    eps = float(jnp.finfo(R.dtype).eps)
    value_check(
        jnp.abs(jnp.diagonal(R)),
        lambda d: bool(jnp.min(d) > max(sizes) * eps * jnp.max(d)),
        f"{where}: {what} are rank deficient to working precision, so the "
        f"regression coefficients are not defined",
    )


def _least_squares(where, g, given, F_y):
    r"""Unique least squares: :math:`F_c^\top = QR`, :math:`A = F_y Q R^{-\top}`."""
    if F_y is None:
        return None, None
    Fc = _given_rows(g, given)  # (N, k)
    Q, R = jnp.linalg.qr(Fc.T)  # (k, N), (N, N)
    _rank_check(where, R, Fc.shape, "the given factor rows")
    Fy = F_y.to_dense()
    FQ = Fy @ Q  # (d_y, N)
    A = tri_solve(R, FQ, lower=False)  # A = F_y Q R^{-T}, row by row
    c.result_check(where, "regression coefficients", A)
    coefs = _split(g, given, lambda s: Dense(A[:, s]))
    return coefs, Dense(Fy - FQ @ Q.T)


def _min_norm(where, g, given, F_y, aligned: bool):
    r"""Minimum norm: :math:`G_c = F_c H = QR`, :math:`A = F_y H R^{-1} Q^\top`."""
    if F_y is None:
        return None, None
    Fc = _given_rows(g, given)  # (N, k)
    Fy = F_y.to_dense()
    if aligned:
        Gc, Gy = _centered_coordinates(Fc), _centered_coordinates(Fy)
    else:
        Gc, Gy = Fc, Fy
    Q, R = jnp.linalg.qr(Gc)  # (N, r), (r, r)
    _rank_check(where, R, Gc.shape, "the given factor rows' columns")
    L = tri_solve(R, Gy, lower=False, trans=1)  # L = G_y R^{-1}, row by row
    c.result_check(where, "regression coefficients", L)
    coefs = _split(g, given, lambda s: product(Dense(L), Dense(Q[s]).T))
    return coefs, Dense(Fy - L @ (Q.T @ Fc))


def _centered_coordinates(F: Array) -> Array:
    r"""``F H``, ``(n, J - 1)``: :math:`H` orthonormal, :math:`H^\top\mathbf 1 = 0`.

    :math:`H` is columns 2 to :math:`J` of the Householder reflector
    :math:`P = I - 2ww^\top`, :math:`w \propto e_1 - \mathbf 1/\sqrt J`, which
    maps :math:`e_1` to :math:`\mathbf 1/\sqrt J`; applied without forming it,
    in :math:`O(nJ)`.
    """
    J = F.shape[-1]
    w = -jnp.full((J,), 1.0 / math.sqrt(J), F.dtype)
    w = w.at[0].add(1.0)
    w = w / jnp.linalg.norm(w)
    return (F - 2.0 * jnp.outer(F @ w, w))[:, 1:]
