"""Prototype operator layer: the current pyeki.linalg, re-exported."""
from pyeki.linalg import *  # noqa: F401,F403
from pyeki.linalg import (  # noqa: F401
    Dense, DensePSD, Identity, LinOp, PSDDiagonal, PSDLinOp, PSDLowRank, PSDScaled,
    dense_matvec, hstack, block_diag,
)

import jax
import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# the whitened-SVD kernel, with gradients that survive degenerate spectra
# ---------------------------------------------------------------------------


def _svd(S):
    return jnp.linalg.svd(S, full_matrices=False)


@jax.custom_jvp
def _solve_factor(S: Array, B: Array) -> Array:
    """(I + S S^T)^{-1} S b for S (k, N) and b (..., N); returns (..., k)."""
    U, sig, Vt = _svd(S)
    coef = jnp.einsum("rn,...n->...r", Vt, B) * (sig / (1.0 + sig**2))
    return jnp.einsum("kr,...r->...k", U, coef)


def _resolvent(S):
    """M = (I + S S^T)^{-1}, from the SVD, with the identity completion."""
    U, sig, _ = _svd(S)
    k = S.shape[0]
    return jnp.eye(k, dtype=S.dtype) + (U * (1.0 / (1.0 + sig**2) - 1.0)) @ U.T


@_solve_factor.defjvp
def _gain_jvp(primals, tangents):
    S, B = primals
    dS, dB = tangents
    w = _solve_factor(S, B)
    M = _resolvent(S)
    t1 = jnp.einsum("kn,...n->...k", dS, B) + jnp.einsum("kn,...n->...k", S, dB)
    Stw = jnp.einsum("kn,...k->...n", S, w)
    dStw = jnp.einsum("kn,...k->...n", dS, w)
    t2 = jnp.einsum("kn,...n->...k", dS, Stw) + jnp.einsum("kn,...n->...k", S, dStw)
    return w, jnp.einsum("jk,...k->...j", M, t1 - t2)


@jax.custom_jvp
def _inverse_sqrt(S: Array) -> Array:
    """T = (I + S S^T)^{-1/2}, k x k, exact at every rank."""
    U, sig, _ = _svd(S)
    k = S.shape[0]
    return jnp.eye(k, dtype=S.dtype) + (U * (1.0 / jnp.sqrt(1.0 + sig**2) - 1.0)) @ U.T


@_inverse_sqrt.defjvp
def __inverse_sqrt_jvp(primals, tangents):
    # Daleckii-Krein in the full left singular basis. For f(l) = (1+l)^{-1/2}
    # the divided difference is -1 / (s_i s_j (s_i + s_j)), s = sqrt(1+l): no
    # cancellation and no special case for repeated eigenvalues.
    (S,), (dS,) = primals, tangents
    k = S.shape[0]
    U, sig, _ = jnp.linalg.svd(S, full_matrices=True)
    lam = jnp.zeros(k, dtype=S.dtype).at[: sig.shape[0]].set(sig**2)
    s = jnp.sqrt(1.0 + lam)
    T = (U / s) @ U.T
    dA = dS @ S.T + S @ dS.T
    G = -1.0 / (s[:, None] * s[None, :] * (s[:, None] + s[None, :]))
    return T, U @ (G * (U.T @ dA @ U)) @ U.T


@jax.custom_jvp
def _logdet(S: Array) -> Array:
    """log det(I + S S^T) = sum log(1 + sigma^2)."""
    sig = jnp.linalg.svd(S, compute_uv=False)
    return jnp.sum(jnp.log1p(sig**2))


@_logdet.defjvp
def _logdet_jvp(primals, tangents):
    (S,), (dS,) = primals, tangents
    U, sig, Vt = _svd(S)
    MS = (U * (sig / (1.0 + sig**2))) @ Vt  # (I + S S^T)^{-1} S
    return jnp.sum(jnp.log1p(sig**2)), 2.0 * jnp.sum(MS * dS)




class IdentityPlusGram:
    """I + S S^T for a (k, N) array S, through one thin SVD.

    Prototype: a plain object rather than a full LinOp; the three methods carry
    custom JVPs that stay finite at repeated and zero singular values.
    """

    def __init__(self, S):
        self.S = jnp.asarray(S)

    def solve_factor(self, b):
        """(I + S S^T)^{-1} S b, for b of shape (..., N)."""
        return _solve_factor(self.S, b)

    def inverse_sqrt(self):
        """(I + S S^T)^{-1/2} as an operator (prototype: dense)."""
        return Dense(_inverse_sqrt(self.S))

    def logdet(self):
        """log det(I + S S^T)."""
        return _logdet(self.S)
