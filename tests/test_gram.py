"""Targeted tests for ``IdentityPlusGram``, the conditioning core, and for
``LowRankUpdate``, which is built on it.

Conformance lives in ``test_conformance.py``. This file holds the exactness
checks against closed forms, the derivative tests at the degenerate spectra
where a plain SVD's derivative is ``nan``, and the SVD counts.

The degenerate spectra are the ones the design measured: exactly repeated
singular values, and exactly zero ones from zero-padded columns (what a
localization mask produces) or from an exactly collapsed ``S``. Each is
confirmed degenerate by a sentinel — the gradient of the same quantity
computed through a plain SVD is ``nan`` there — so no test passes because its
operand quietly stopped being degenerate.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import enskit  # noqa: F401  -- enables x64 before any array exists
from enskit.linalg import (
    Dense,
    DensePSD,
    Identity,
    IdentityPlusGram,
    IdentityPlusGramInverseSqrt,
    LowRankUpdate,
    PSDDiagonal,
    PSDLinOp,
    PSDLowRank,
    UnsupportedOpError,
    Zero,
    hstack,
    linop,
)

RNG = np.random.default_rng(0)


def _with_zero_columns(k: int, rank: int, n_zero: int) -> np.ndarray:
    """A ``(k, rank + n_zero)`` array whose last ``n_zero`` columns are zero."""
    return np.concatenate([RNG.normal(size=(k, rank)), np.zeros((k, n_zero))], axis=1)


#: S at which a plain SVD's derivative is nan, by spectrum.
DEGENERATE = {
    "collapsed": np.zeros((5, 3)),
    "repeated": 2.0 * np.eye(5, 3),
    "repeated-wide": 2.0 * np.eye(3, 5),
    "zero-padded-tall": _with_zero_columns(5, 2, 2),  # two zero singular values
    "zero-padded-wide": _with_zero_columns(3, 2, 3),  # one zero singular value
}
GENERIC = {
    "generic-tall": RNG.normal(size=(5, 3)),
    "generic-wide": RNG.normal(size=(3, 5)),
}
ALL = {**DEGENERATE, **GENERIC}


def _plain_svd_quantities(S):
    """The inverse square root and the gain matrix, through a plain SVD."""
    U, sigma, Vt = jnp.linalg.svd(S, full_matrices=False)
    T = jnp.eye(S.shape[0]) + (U * (1.0 / jnp.sqrt(1.0 + sigma**2) - 1.0)) @ U.T
    return jnp.sum(T) + jnp.sum((U * (sigma / (1.0 + sigma**2))) @ Vt)


@pytest.mark.parametrize("name", DEGENERATE)
def test_sentinel_the_degenerate_spectra_defeat_a_plain_svd(name):
    """Every degenerate operand below makes a plain SVD's gradient nan."""
    S = jnp.asarray(DEGENERATE[name])
    assert bool(jnp.isnan(jax.grad(_plain_svd_quantities)(S)).any())


# ---------------------------------------------------------------------------
# exactness against closed forms
# ---------------------------------------------------------------------------


def _dense_inverse_sqrt(S: np.ndarray) -> np.ndarray:
    """``(I + S S^T)^{-1/2}`` from a dense symmetric eigendecomposition."""
    A = np.eye(S.shape[0]) + S @ S.T
    lam, Q = np.linalg.eigh(A)
    return (Q / np.sqrt(lam)) @ Q.T


@pytest.mark.parametrize("name", ALL)
def test_every_operation_matches_its_closed_form(name):
    S = ALL[name]
    k, N = S.shape
    A = np.eye(k) + S @ S.T
    op = IdentityPlusGram(jnp.asarray(S))
    b_k = RNG.normal(size=(2, 3, k))
    b_N = RNG.normal(size=(2, 3, N))
    tol = dict(rtol=1e-12, atol=1e-12)

    np.testing.assert_allclose(
        op.solve_factor(b_N), np.linalg.solve(A, S @ b_N[..., None])[..., 0], **tol
    )
    np.testing.assert_allclose(
        op.solve(b_k), np.linalg.solve(A, b_k[..., None])[..., 0], **tol
    )
    np.testing.assert_allclose(op.logdet(), np.linalg.slogdet(A)[1], **tol)
    T = _dense_inverse_sqrt(S)
    np.testing.assert_allclose(op.inverse_sqrt().to_dense(), T, **tol)
    np.testing.assert_allclose(op.whiten(b_k), b_k @ T.T, **tol)


def test_inverse_sqrt_needs_the_identity_completion():
    """With ``k > N`` the thin basis spans only ``N`` of the ``k`` dimensions;
    the naive thin form ``U (I + Sigma^2)^{-1/2} U^T`` is then singular, and
    wrong without raising."""
    S = RNG.normal(size=(6, 2))
    T = np.asarray(IdentityPlusGram(jnp.asarray(S)).inverse_sqrt().to_dense())
    A = np.eye(6) + S @ S.T
    np.testing.assert_allclose(T @ A @ T, np.eye(6), atol=1e-12)
    U, sigma, _ = np.linalg.svd(S, full_matrices=False)
    naive = (U / np.sqrt(1.0 + sigma**2)) @ U.T
    assert np.linalg.matrix_rank(naive) == 2


def test_solve_factor_is_bounded_however_large_s_is():
    """The multipliers sigma / (1 + sigma^2) are at most 1/2, so the result
    is bounded by half the operand at every scale of S, with no
    regularization; forming S S^T at sigma = 1e9 would overflow nothing but
    round the identity away."""
    for scale in (1e-9, 1.0, 1e9):
        S = scale * RNG.normal(size=(4, 6))
        b = RNG.normal(size=(6,))
        w = IdentityPlusGram(jnp.asarray(S)).solve_factor(jnp.asarray(b))
        assert np.linalg.norm(w) <= 0.5 * np.linalg.norm(b) * (1 + 1e-12)


@pytest.mark.parametrize("scale", [1e4, 1e6, 1e8])
def test_regression_large_s_with_a_square_basis_keeps_full_accuracy(scale):
    """With k <= N the basis U is square and the complement is empty. Writing
    A^{-1} b as b + U((I + Sigma^2)^{-1} - I)U^T b then cancels b against
    U U^T b, leaving about eps |b| where the answer is |b| / sigma^2: a
    relative error of 41 at scale 1e8, though A's condition number is 6.
    The derivative of solve_factor went through the same kernel."""
    rng = np.random.default_rng(3)
    S = scale * rng.normal(size=(6, 10))
    A = np.eye(6) + S @ S.T
    b = rng.normal(size=(6,))
    op = IdentityPlusGram(jnp.asarray(S))

    def rel(got, want):
        return np.linalg.norm(np.asarray(got) - want) / np.linalg.norm(want)

    assert rel(op.solve(jnp.asarray(b)), np.linalg.solve(A, b)) < 1e-12
    lam, Q = np.linalg.eigh(A)
    assert rel(op.whiten(jnp.asarray(b)), (Q / np.sqrt(lam)) @ Q.T @ b) < 1e-12
    c, dS = rng.normal(size=(10,)), rng.normal(size=(6, 10))
    got = jax.jvp(
        lambda S: IdentityPlusGram(S).solve_factor(jnp.asarray(c)),
        (jnp.asarray(S),),
        (jnp.asarray(dS),),
    )[1]
    want = jax.jvp(
        lambda S: _dense_solve_factor(S, jnp.asarray(c)),
        (jnp.asarray(S),),
        (jnp.asarray(dS),),
    )[1]
    assert rel(got, np.asarray(want)) < 1e-9


def test_solve_factor_validates_its_operand():
    op = IdentityPlusGram(jnp.asarray(RNG.normal(size=(4, 6))))
    with pytest.raises(ValueError, match="solve_factor"):
        op.solve_factor(jnp.zeros(4))  # length k, not N
    with pytest.raises(ValueError, match="solve_factor"):
        op.solve_factor(jnp.asarray(0.0))


def test_constructor_rejects_bad_s():
    with pytest.raises(ValueError, match="rank 2"):
        IdentityPlusGram(jnp.zeros(4))
    with pytest.raises(ValueError, match="rank 2"):
        IdentityPlusGram(jnp.zeros((2, 3, 4)))
    with pytest.raises(ValueError, match="positive"):
        IdentityPlusGram(jnp.zeros((0, 3)))
    with pytest.raises(TypeError, match="real"):
        IdentityPlusGram(jnp.zeros((2, 3), dtype=jnp.complex128))


def test_inverse_sqrt_constructed_directly_matches_the_method():
    S = jnp.asarray(RNG.normal(size=(4, 3)))
    direct = IdentityPlusGramInverseSqrt(S)
    via = IdentityPlusGram(S).inverse_sqrt()
    np.testing.assert_allclose(direct.to_dense(), via.to_dense(), rtol=1e-14)
    assert type(via) is IdentityPlusGramInverseSqrt


# ---------------------------------------------------------------------------
# first derivatives, at degenerate spectra and generic ones
# ---------------------------------------------------------------------------


def _dense_solve_factor(S, b):
    """``(I + S S^T)^{-1} S b`` along the trailing axis of ``b``."""
    return _dense_solve(S, b @ S.T)


def _dense_solve(S, b):
    """``(I + S S^T)^{-1} b`` along the trailing axis of ``b``."""
    return jnp.linalg.solve(jnp.eye(S.shape[0]) + S @ S.T, b.T).T


def _dense_logdet(S):
    return jnp.linalg.slogdet(jnp.eye(S.shape[0]) + S @ S.T)[1]


def _central_difference(f, S: np.ndarray, dS: np.ndarray, h: float = 1e-5):
    return (f(S + h * dS) - f(S - h * dS)) / (2 * h)


@pytest.mark.parametrize("name", ALL)
def test_rational_operations_differentiate_like_their_dense_forms(name):
    """solve_factor, solve and logdet against the dense rational forms, whose
    derivatives JAX computes through LU and is finite at every S."""
    S = jnp.asarray(ALL[name])
    k, N = S.shape
    dS = jnp.asarray(RNG.normal(size=(k, N)))
    b_N, db_N = jnp.asarray(RNG.normal(size=(2, N))), jnp.asarray(RNG.normal(size=(2, N)))
    b_k, db_k = jnp.asarray(RNG.normal(size=(2, k))), jnp.asarray(RNG.normal(size=(2, k)))
    cases = (
        (
            "solve_factor",
            lambda S, b: IdentityPlusGram(S).solve_factor(b),
            _dense_solve_factor,
            (b_N, db_N),
        ),
        (
            "solve",
            lambda S, b: IdentityPlusGram(S).solve(b),
            _dense_solve,
            (b_k, db_k),
        ),
    )
    for what, ours, dense, (b, db) in cases:
        got = jax.jvp(ours, (S, b), (dS, db))[1]
        want = jax.jvp(dense, (S, b), (dS, db))[1]
        assert bool(jnp.all(jnp.isfinite(got))), what
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10, err_msg=what)
    got = jax.jvp(lambda S: IdentityPlusGram(S).logdet(), (S,), (dS,))[1]
    want = jax.jvp(_dense_logdet, (S,), (dS,))[1]
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10, err_msg="logdet")


@pytest.mark.parametrize("name", ALL)
def test_inverse_sqrt_differentiates_like_finite_differences(name):
    """The thin Daleckii-Krein rule, both as matvec and as to_dense, against a
    central difference of the dense inverse square root (the function is
    real-analytic in S, the spectrum of I + S S^T being bounded below by 1)."""
    S = ALL[name]
    k, N = S.shape
    dS = RNG.normal(size=(k, N))
    x = RNG.normal(size=(3, k))
    want = _central_difference(_dense_inverse_sqrt, S, dS)

    dense = jax.jvp(
        lambda S: IdentityPlusGram(S).inverse_sqrt().to_dense(),
        (jnp.asarray(S),),
        (jnp.asarray(dS),),
    )[1]
    applied = jax.jvp(
        lambda S: IdentityPlusGram(S).inverse_sqrt().matvec(jnp.asarray(x)),
        (jnp.asarray(S),),
        (jnp.asarray(dS),),
    )[1]
    whitened = jax.jvp(
        lambda S: IdentityPlusGram(S).whiten(jnp.asarray(x)),
        (jnp.asarray(S),),
        (jnp.asarray(dS),),
    )[1]
    np.testing.assert_allclose(dense, want, atol=1e-8)
    np.testing.assert_allclose(applied, x @ want.T, atol=1e-8)
    np.testing.assert_allclose(whitened, x @ want.T, atol=1e-8)


@pytest.mark.parametrize("name", ALL)
def test_reverse_mode_agrees_with_forward_mode(name):
    """``grad`` is the transpose of the same rules: <grad f, dS> equals the
    directional derivative, under jit too, for every operation at once."""
    S = jnp.asarray(ALL[name])
    k, N = S.shape
    dS = jnp.asarray(RNG.normal(size=(k, N)))
    b_N, b_k = jnp.asarray(RNG.normal(size=(N,))), jnp.asarray(RNG.normal(size=(k,)))
    weights = jnp.asarray(RNG.normal(size=(5, k)))

    def f(S):
        op = IdentityPlusGram(S)
        T = op.inverse_sqrt()
        return (
            weights[0] @ op.solve_factor(b_N)
            + weights[1] @ op.solve(b_k)
            + weights[2] @ op.whiten(b_k)
            + weights[3] @ T.matvec(b_k)
            + weights[4] @ T.to_dense() @ b_k
            + op.logdet()
        )

    directional = jax.jvp(f, (S,), (dS,))[1]
    for grad in (jax.grad(f), jax.jit(jax.grad(f))):
        g = grad(S)
        assert bool(jnp.all(jnp.isfinite(g)))
        np.testing.assert_allclose(jnp.sum(g * dS), directional, rtol=1e-10, atol=1e-10)


def test_derivatives_are_finite_under_vmap_over_a_degenerate_family():
    """A family mixing degenerate and generic members, as a vmapped local
    analysis produces: every member's gradient matches its own."""
    stack = jnp.stack([jnp.asarray(DEGENERATE["collapsed"]),
                       jnp.asarray(DEGENERATE["repeated"]),
                       jnp.asarray(DEGENERATE["zero-padded-tall"][:, :3]),
                       jnp.asarray(GENERIC["generic-tall"])])
    b = jnp.asarray(RNG.normal(size=(3,)))

    def f(S):
        op = IdentityPlusGram(S)
        return jnp.sum(op.solve_factor(b)) + jnp.sum(op.inverse_sqrt().to_dense())

    family = jax.vmap(jax.grad(f))(stack)
    assert bool(jnp.all(jnp.isfinite(family)))
    for i in range(stack.shape[0]):
        np.testing.assert_allclose(
            family[i], jax.grad(f)(stack[i]), rtol=1e-12, atol=1e-12
        )


# ---------------------------------------------------------------------------
# second derivatives: the documented limit
# ---------------------------------------------------------------------------


def test_second_derivatives_are_exact_at_generic_spectra():
    S = jnp.asarray(GENERIC["generic-tall"])
    b = jnp.asarray(RNG.normal(size=(3,)))
    x = jnp.asarray(RNG.normal(size=(5,)))
    for ours, dense in (
        (lambda S: jnp.sum(IdentityPlusGram(S).solve_factor(b) ** 2),
         lambda S: jnp.sum(_dense_solve_factor(S, b) ** 2)),
        (lambda S: IdentityPlusGram(S).logdet(), _dense_logdet),
        # |whiten(x)|^2 = x^T A^{-1} x, whatever the whitener
        (lambda S: jnp.sum(IdentityPlusGram(S).whiten(x) ** 2),
         lambda S: x @ _dense_solve(S, x)),
    ):
        np.testing.assert_allclose(
            jax.hessian(ours)(S), jax.hessian(dense)(S), rtol=1e-9, atol=1e-9
        )


def test_second_derivatives_at_a_degenerate_spectrum_are_finite_only_for_logdet():
    """The module docstring's limit, asserted so that a change is visible:
    at a collapsed S the Hessian of logdet is finite and exact, while that of
    solve_factor is nan — linearization inlines the rule's own arithmetic,
    which reads the decomposition through the SVD's derivative."""
    S = jnp.zeros((4, 3))
    b = jnp.asarray(RNG.normal(size=(3,)))
    np.testing.assert_allclose(
        jax.hessian(lambda S: IdentityPlusGram(S).logdet())(S),
        jax.hessian(_dense_logdet)(S),
        atol=1e-12,
    )
    hessian = jax.hessian(lambda S: jnp.sum(IdentityPlusGram(S).solve_factor(b) ** 2))(S)
    assert bool(jnp.isnan(hessian).any())


# ---------------------------------------------------------------------------
# counts
# ---------------------------------------------------------------------------


def _count_primitive(jaxpr, name: str) -> int:
    """Count ``name`` primitives in a jaxpr, recursing into sub-jaxprs."""
    total = 0
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == name:
            total += 1
        for param in eqn.params.values():
            candidates = param if isinstance(param, (tuple, list)) else [param]
            for candidate in candidates:
                inner = getattr(candidate, "jaxpr", candidate)
                if hasattr(inner, "eqns"):
                    total += _count_primitive(inner, name)
    return total


def test_one_svd_at_construction_and_none_per_operation_even_under_grad():
    """Every operation reuses the stored decomposition, and so does every
    derivative rule: a gradient through all of them computes one SVD, the
    construction's. (The design's prototype recomputed a full SVD inside the
    inverse square root's rule.)"""
    S = jnp.asarray(RNG.normal(size=(5, 3)))
    b_N, b_k = jnp.ones(3), jnp.ones(5)

    def everything(S):
        op = IdentityPlusGram(S)
        return (
            jnp.sum(op.solve_factor(b_N)) + jnp.sum(op.solve(b_k)) + op.logdet()
            + jnp.sum(op.whiten(b_k)) + jnp.sum(op.inverse_sqrt().to_dense())
        )

    assert _count_primitive(jax.make_jaxpr(everything)(S).jaxpr, "svd") == 1
    assert _count_primitive(jax.make_jaxpr(jax.grad(everything))(S).jaxpr, "svd") == 1
    op = IdentityPlusGram(S)
    ops_only = jax.make_jaxpr(
        lambda op: (op.solve_factor(b_N), op.solve(b_k), op.logdet(), op.whiten(b_k))
    )(op)
    assert _count_primitive(ops_only.jaxpr, "svd") == 0


def test_regression_logdet_gradient_allocates_nothing_of_size_n_by_n():
    """The logdet rule once applied solve_factor to an (N, N) identity: 512 MB
    of temporaries for a LowRankUpdate gradient at n = 8000."""
    k, N = 3, 50
    S = jnp.asarray(RNG.normal(size=(k, N)))
    jaxpr = jax.make_jaxpr(jax.grad(lambda S: IdentityPlusGram(S).logdet()))(S)
    for eqn in jaxpr.jaxpr.eqns:
        for var in eqn.outvars:
            assert tuple(var.aval.shape) != (N, N), eqn.primitive.name


def test_solve_factor_derivative_needs_no_gram_matrix():
    """The rules apply S and S^T to vectors; no (k, k) or (N, N) product of S
    with itself appears in a gradient, which would square the condition
    number exactly as forming the Gram matrix in the value would."""
    k, N = 5, 3
    S = jnp.asarray(RNG.normal(size=(k, N)))
    b = jnp.ones(N)
    jaxpr = jax.make_jaxpr(
        jax.grad(lambda S: jnp.sum(IdentityPlusGram(S).solve_factor(b)))
    )(S)
    for eqn in jaxpr.jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            shapes = [tuple(v.aval.shape) for v in eqn.invars]
            assert shapes not in ([(k, N), (k, N)], [(N, k), (N, k)]), shapes


# ---------------------------------------------------------------------------
# LowRankUpdate
# ---------------------------------------------------------------------------


def _low_rank_update_problem(F: np.ndarray):
    n = F.shape[0]
    d = RNG.uniform(0.5, 2.0, n)
    return jnp.asarray(d), jnp.asarray(F)


@pytest.mark.parametrize("n_zero", [0, 2])
def test_low_rank_update_differentiates_at_zero_padded_factors(n_zero):
    """A factor padded with zero columns — what growing a latent space with
    ``Zero`` blocks produces — gives S exact zero singular values. solve,
    logdet and the whitened norm differentiate finitely there, and match the
    dense forms; |W_C x|^2 = x^T C^{-1} x for every valid whitener."""
    F0 = RNG.normal(size=(6, 2))
    F = np.concatenate([F0, np.zeros((6, n_zero))], axis=1)
    d, F = _low_rank_update_problem(F)
    x = jnp.asarray(RNG.normal(size=(6,)))
    dF = jnp.asarray(RNG.normal(size=F.shape))

    def cov(F):
        return jnp.diag(d) + F @ F.T

    pairs = (
        (lambda F: LowRankUpdate(PSDDiagonal(d), Dense(F)).solve(x),
         lambda F: jnp.linalg.solve(cov(F), x)),
        (lambda F: LowRankUpdate(PSDDiagonal(d), Dense(F)).logdet(),
         lambda F: jnp.linalg.slogdet(cov(F))[1]),
        (lambda F: jnp.sum(LowRankUpdate(PSDDiagonal(d), Dense(F)).whiten(x) ** 2),
         lambda F: x @ jnp.linalg.solve(cov(F), x)),
    )
    for ours, dense in pairs:
        got = jax.jvp(ours, (F,), (dF,))[1]
        assert bool(jnp.all(jnp.isfinite(got)))
        want = jax.jvp(dense, (F,), (dF,))[1]
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)
        g = jax.grad(lambda F, f=ours: jnp.sum(f(F)))(F)
        assert bool(jnp.all(jnp.isfinite(g)))


def test_low_rank_update_computes_its_svd_once_at_construction():
    d, F = _low_rank_update_problem(RNG.normal(size=(6, 3)))
    op = LowRankUpdate(PSDDiagonal(d), Dense(F))
    x = jnp.ones(6)
    jaxpr = jax.make_jaxpr(lambda op: (op.solve(x), op.logdet(), op.whiten(x)))(op)
    assert _count_primitive(jaxpr.jaxpr, "svd") == 0
    build = jax.make_jaxpr(lambda F: LowRankUpdate(PSDDiagonal(d), Dense(F)).solve(x))(F)
    assert _count_primitive(build.jaxpr, "svd") == 1


@linop
class _WhitenOnly(PSDLinOp):
    """A PSD test operator that whitens but supports nothing else optional."""

    diagonal: jax.Array

    @property
    def shape(self):
        return (self.diagonal.shape[-1],) * 2

    @property
    def batch_shape(self):
        return tuple(self.diagonal.shape[:-1])

    def _matvec(self, x):
        return self.diagonal * x

    def _whiten(self, x):
        return x / jnp.sqrt(self.diagonal)

    def _to_dense(self):
        return jnp.diag(self.diagonal)


def test_low_rank_update_capabilities_follow_the_base():
    d, F = _low_rank_update_problem(RNG.normal(size=(4, 2)))
    full = LowRankUpdate(PSDDiagonal(d), Dense(F))
    assert full.capabilities() == PSDDiagonal(d).capabilities()
    bare = LowRankUpdate(_WhitenOnly(d), Dense(F))
    assert bare.capabilities() == frozenset({"whiten", "whiten_mat"})
    for name in ("solve", "logdet", "diag", "factor"):
        assert not bare.supports(name)
    with pytest.raises(UnsupportedOpError):
        bare.solve(jnp.ones(4))
    # and the whitener of the bare one is still valid
    W = np.asarray(bare.whiten(jnp.eye(4))).T
    dense = np.asarray(bare.to_dense())
    np.testing.assert_allclose(W @ dense @ W.T, np.eye(4), atol=1e-12)


def test_low_rank_update_rejects_bad_arguments():
    F = Dense(jnp.asarray(RNG.normal(size=(4, 2))))
    with pytest.raises(UnsupportedOpError, match="whiten"):
        LowRankUpdate(PSDLowRank(jnp.asarray(RNG.normal(size=(4, 2)))), F)
    with pytest.raises(ValueError, match="rows"):
        LowRankUpdate(Identity(5), F)
    with pytest.raises(TypeError, match="PSDLinOp"):
        LowRankUpdate(Dense(jnp.eye(4)), F)
    with pytest.raises(TypeError, match="LinOp"):
        LowRankUpdate(Identity(4), jnp.ones((4, 2)))


def test_low_rank_update_factor_is_the_base_factor_beside_f():
    M = RNG.normal(size=(4, 4))
    base = DensePSD(jnp.asarray(M @ M.T + 4 * np.eye(4)))
    F = Dense(jnp.asarray(RNG.normal(size=(4, 3))))
    L = LowRankUpdate(base, F).factor()
    assert L.shape == (4, 7)
    np.testing.assert_allclose(L.to_dense()[:, 4:], F.to_dense())


def test_low_rank_update_over_a_zero_padded_factor_operator():
    """The factor may be any operator; one padded with a ``Zero`` block is
    exactly the zero-column case above, built the way absorbing a block
    builds it."""
    d, F0 = _low_rank_update_problem(RNG.normal(size=(5, 2)))
    padded = LowRankUpdate(PSDDiagonal(d), hstack(Dense(F0), Zero(5, 3)))
    plain = LowRankUpdate(PSDDiagonal(d), Dense(F0))
    x = jnp.asarray(RNG.normal(size=(5,)))
    np.testing.assert_allclose(padded.solve(x), plain.solve(x), rtol=1e-12)
    np.testing.assert_allclose(padded.logdet(), plain.logdet(), rtol=1e-12)
    np.testing.assert_allclose(
        jnp.sum(padded.whiten(x) ** 2), jnp.sum(plain.whiten(x) ** 2), rtol=1e-12
    )
