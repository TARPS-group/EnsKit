"""Targeted tests for the Kronecker operators.

Conformance lives in ``test_conformance.py``. It compares each operation
with ``to_dense``, and ``to_dense`` is ``jnp.kron`` of the operands, so an
implementation that reversed the operands' order in every method at once
would agree with itself and pass, at any sizes, since ``B (x) A`` always has
the shape of ``A (x) B``. The tests here pin each method to ``numpy.kron`` of
the operands' dense forms instead. They use operands of equal side, where a
*partial* reversal (one method applying ``A`` along the fast axis) is also
shape-valid, as well as rectangular ones. They also hold
the closed forms (the log-determinant's pairing of each operand with the
other's side, the factor of two Cholesky factors), the class each
construction chooses, and a count that no operation forms an array of side
``n_A n_B``.

Each hazard has a sentinel showing that the operands distinguish the right
answer from the wrong one, so no test passes because its operands stopped
being informative.
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
    DenseSquare,
    Identity,
    Kronecker,
    PSDKronecker,
    PSDLowRank,
    SquareKronecker,
    Transposed,
    Triangular,
    UnsupportedOpError,
    kron,
)

RNG = np.random.default_rng(0)
_TOL = {"rtol": 1e-11, "atol": 1e-11}


def _psd(n: int) -> np.ndarray:
    M = RNG.normal(size=(n, n))
    return M @ M.T + n * np.eye(n)


def _square(n: int) -> np.ndarray:
    return RNG.normal(size=(n, n)) + n * np.eye(n)


def _apply(M: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.einsum("ij,...j->...i", M, x)


# ---------------------------------------------------------------------------
# orientation, pinned to numpy.kron
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sides", [(3, 3), (3, 4)], ids=["equal", "unequal"])
def test_sentinel_a_reversed_order_is_a_valid_covariance_of_the_same_shape(sides):
    """``B (x) A`` is PSD and the same shape as ``A (x) B`` but a different
    matrix, at equal and at unequal sides, so only a comparison of values can
    catch a reversed order."""
    A, B = _psd(sides[0]), _psd(sides[1])
    right, reversed_ = np.kron(A, B), np.kron(B, A)
    assert right.shape == reversed_.shape
    assert np.linalg.eigvalsh(reversed_).min() > 0
    assert np.abs(right - reversed_).max() > 1.0


@pytest.mark.parametrize("batch", [(), (4,), (2, 9)], ids=["rank0", "rank1", "rank2"])
@pytest.mark.parametrize("level", ["psd", "square"])
def test_regression_matvec_and_rmatvec_match_numpy_kron(batch, level):
    """Application is pinned to ``numpy.kron`` directly, at batch ranks 0, 1
    and 2, never to ``to_dense``: the two are separate code paths, and a
    reversed order in both would agree with itself. The square operands are
    not symmetric, so ``rmatvec`` is tested apart from ``matvec``."""
    make, wrap = (_psd, DensePSD) if level == "psd" else (_square, DenseSquare)
    A, B = make(3), make(3)
    op = kron(wrap(jnp.asarray(A)), wrap(jnp.asarray(B)))
    K = np.kron(A, B)
    x = RNG.normal(size=(*batch, 9))
    np.testing.assert_allclose(op.matvec(x), _apply(K, x), **_TOL)
    np.testing.assert_allclose(op.rmatvec(x), _apply(K.T, x), **_TOL)


@pytest.mark.parametrize("batch", [(), (4,), (2, 8)], ids=["rank0", "rank1", "rank2"])
def test_regression_rectangular_operands_reshape_operand_and_result_apart(batch):
    """With all four sizes distinct, the operand is ``(k_A, k_B)``, the
    intermediate ``(k_A, n_B)`` and the result ``(n_A, n_B)``: an
    implementation that reshapes operand and result alike fails here."""
    A, B = RNG.normal(size=(2, 3)), RNG.normal(size=(4, 5))
    op = kron(Dense(jnp.asarray(A)), Dense(jnp.asarray(B)))
    K = np.kron(A, B)
    assert op.shape == K.shape == (8, 15)
    x = RNG.normal(size=(*batch, 15))
    y = RNG.normal(size=(*batch, 8))
    np.testing.assert_allclose(op.matvec(x), _apply(K, x), **_TOL)
    np.testing.assert_allclose(op.rmatvec(y), _apply(K.T, y), **_TOL)
    np.testing.assert_allclose(op.T.matvec(y), _apply(K.T, y), **_TOL)


def test_regression_to_dense_matches_numpy_kron():
    """``to_dense`` is pinned to ``numpy.kron`` on its own, for the same
    reason ``matvec`` is: pinning the two to each other constrains nothing."""
    A, B = _psd(3), _psd(3)
    op = kron(DensePSD(jnp.asarray(A)), DensePSD(jnp.asarray(B)))
    np.testing.assert_allclose(op.to_dense(), np.kron(A, B), **_TOL)
    R, S = RNG.normal(size=(2, 3)), RNG.normal(size=(4, 5))
    rect = kron(Dense(jnp.asarray(R)), Dense(jnp.asarray(S)))
    np.testing.assert_allclose(rect.to_dense(), np.kron(R, S), **_TOL)


@pytest.mark.parametrize("batch", [(), (4,), (2, 9)], ids=["rank0", "rank1", "rank2"])
def test_regression_solve_whiten_and_diag_match_numpy_kron(batch):
    """``solve`` against the inverse of ``numpy.kron``, ``diag`` against its
    diagonal, and ``whiten`` against ``W_A (x) W_B``, the Kronecker product
    of the operands' own whiteners in the same order."""
    A, B = _psd(3), _psd(3)
    op_a, op_b = DensePSD(jnp.asarray(A)), DensePSD(jnp.asarray(B))
    op = kron(op_a, op_b)
    K = np.kron(A, B)
    W_a, W_b = (np.asarray(o.whiten(jnp.eye(3))).T for o in (op_a, op_b))
    W = np.kron(W_a, W_b)
    x = RNG.normal(size=(*batch, 9))
    np.testing.assert_allclose(op.solve(x), _apply(np.linalg.inv(K), x), **_TOL)
    np.testing.assert_allclose(op.whiten(x), _apply(W, x), **_TOL)
    np.testing.assert_allclose(op.diag(), np.diag(K), **_TOL)


# ---------------------------------------------------------------------------
# closed forms
# ---------------------------------------------------------------------------


def test_sentinel_the_swapped_log_determinant_pairing_differs():
    """With sides 3 and 5, pairing each operand's log-determinant with its
    own side instead of the other's gives a different number; with equal
    sides the two coincide, which is why the test below uses unequal ones."""
    A, B = _psd(3), _psd(5)
    right = 5 * np.linalg.slogdet(A)[1] + 3 * np.linalg.slogdet(B)[1]
    swapped = 3 * np.linalg.slogdet(A)[1] + 5 * np.linalg.slogdet(B)[1]
    np.testing.assert_allclose(right, np.linalg.slogdet(np.kron(A, B))[1], rtol=1e-12)
    assert abs(right - swapped) > 1.0


def test_regression_logdet_pairs_each_operand_with_the_other_side():
    """``log det(A (x) B) = n_B log det A + n_A log det B``, for PSD operands
    and for square ones with negative pivots, where ``log |det|`` applies."""
    A, B = _psd(3), _psd(5)
    psd = kron(DensePSD(jnp.asarray(A)), DensePSD(jnp.asarray(B)))
    np.testing.assert_allclose(
        psd.logdet(), np.linalg.slogdet(np.kron(A, B))[1], rtol=1e-12
    )
    C, D = _square(2), _square(4)
    C[0] *= -1.0
    square = kron(DenseSquare(jnp.asarray(C)), DenseSquare(jnp.asarray(D)))
    np.testing.assert_allclose(
        square.logdet(), np.linalg.slogdet(np.kron(C, D))[1], rtol=1e-12
    )


def test_factor_of_two_cholesky_factors_is_the_cholesky_factor_and_solves():
    """The Kronecker product of two lower-triangular Cholesky factors is lower
    triangular and is the Cholesky factor of the product, so the factor is a
    ``SquareKronecker`` that keeps the operands' triangular solve."""
    A, B = _psd(3), _psd(5)
    L = kron(DensePSD(jnp.asarray(A)), DensePSD(jnp.asarray(B))).factor()
    assert type(L) is SquareKronecker
    assert type(L.A) is Triangular and type(L.B) is Triangular
    assert L.supports("solve")
    Ld = np.asarray(L.to_dense())
    assert np.abs(np.triu(Ld, 1)).max() == 0.0
    np.testing.assert_allclose(
        Ld, np.linalg.cholesky(np.kron(A, B)), rtol=1e-12, atol=1e-12
    )
    b = RNG.normal(size=(2, 15))
    np.testing.assert_allclose(L.solve(b), _apply(np.linalg.inv(Ld), b), **_TOL)


def test_factor_with_a_rectangular_operand_factor_is_a_plain_kronecker():
    """A rectangular operand factor makes the product's factor rectangular,
    and the class says so: it has no ``solve`` attribute at all."""
    F = RNG.normal(size=(3, 2))
    op = kron(DensePSD(jnp.asarray(_psd(2))), PSDLowRank(jnp.asarray(F)))
    L = op.factor()
    assert type(L) is Kronecker
    assert L.shape == (6, 4)
    assert not hasattr(L, "solve")
    np.testing.assert_allclose(L.to_dense() @ L.to_dense().T, op.to_dense(), **_TOL)


# ---------------------------------------------------------------------------
# classes and capabilities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("A", "B", "cls"),
    [
        (Identity(2), DensePSD(jnp.asarray(_psd(3))), PSDKronecker),
        (
            DenseSquare(jnp.asarray(_square(2))),
            DensePSD(jnp.asarray(_psd(3))),
            SquareKronecker,
        ),
        (Triangular(jnp.eye(2)), Triangular(jnp.eye(3)), SquareKronecker),
        (Dense(jnp.ones((2, 2))), Identity(3), Kronecker),
        (Dense(jnp.ones((2, 3))), Dense(jnp.ones((3, 2))), Kronecker),
        (Transposed(DenseSquare(jnp.asarray(_square(2)))), Identity(2), Kronecker),
    ],
    ids=["psd", "square-psd", "triangular", "dense-square", "square-shape", "view"],
)
def test_kron_chooses_the_class_by_the_operands_levels(A, B, cls):
    """The level is decided by the operands' types, never by the product's
    shape: two rectangular operands can give a square shape, and that
    product is singular, so it is a plain ``Kronecker``."""
    assert type(kron(A, B)) is cls


def test_direct_construction_validates_and_never_upgrades():
    A, B = DensePSD(jnp.asarray(_psd(2))), DensePSD(jnp.asarray(_psd(3)))
    plain = Kronecker(A, B)
    assert type(plain) is Kronecker and not hasattr(plain, "solve")
    with pytest.raises(TypeError, match="SquareLinOp"):
        SquareKronecker(Dense(jnp.ones((2, 2))), B)
    with pytest.raises(TypeError, match="PSDLinOp"):
        PSDKronecker(DenseSquare(jnp.asarray(_square(2))), B)
    with pytest.raises(TypeError, match="operators"):
        kron(jnp.eye(2), B)
    family = jax.vmap(DensePSD)(jnp.stack([_psd(2), _psd(2)]))
    with pytest.raises(ValueError, match="vmapped family"):
        kron(family, B)


def test_transpose_of_a_square_kronecker_keeps_its_solve():
    """``(A (x) B)^T = A^T (x) B^T``, built from the operands' structured
    transposes, so a ``SquareKronecker`` of operators whose transposes solve
    stays one; the default ``Transposed`` view would lose ``solve``."""
    C, D = _square(2), _square(3)
    op = kron(DenseSquare(jnp.asarray(C)), DenseSquare(jnp.asarray(D)))
    t = op.T
    assert type(t) is SquareKronecker
    b = RNG.normal(size=(4, 6))
    np.testing.assert_allclose(
        t.solve(b), _apply(np.linalg.inv(np.kron(C, D).T), b), **_TOL
    )
    psd = kron(DensePSD(jnp.asarray(_psd(2))), Identity(2))
    assert psd.T is psd


def test_capabilities_intersect_over_the_operands():
    """An operand without a cheap ``solve``, ``whiten`` or ``logdet`` takes
    each away from the product, and calling it raises."""
    F = jnp.asarray(RNG.normal(size=(3, 2)))
    op = kron(DensePSD(jnp.asarray(_psd(2))), PSDLowRank(F))
    assert op.capabilities() == {"diag", "factor"}
    for name, call in [
        ("solve", lambda: op.solve(jnp.ones(6))),
        ("whiten", lambda: op.whiten(jnp.ones(6))),
        ("logdet", op.logdet),
    ]:
        with pytest.raises(UnsupportedOpError, match=name):
            call()


# ---------------------------------------------------------------------------
# counts
# ---------------------------------------------------------------------------


def _largest_intermediate(jaxpr) -> int:
    """The number of entries in the largest array a jaxpr computes,
    recursing into sub-jaxprs."""
    largest = 0
    for eqn in jaxpr.eqns:
        for var in eqn.outvars:
            largest = max(largest, int(np.prod(var.aval.shape)))
        for param in eqn.params.values():
            candidates = param if isinstance(param, (tuple, list)) else [param]
            for candidate in candidates:
                inner = getattr(candidate, "jaxpr", candidate)
                if hasattr(inner, "eqns"):
                    largest = max(largest, _largest_intermediate(inner))
    return largest


@pytest.mark.parametrize(
    "operation",
    ["matvec", "rmatvec", "solve", "whiten", "logdet", "diag", "factor"],
)
def test_no_operation_forms_an_array_of_the_products_side(operation):
    """Every operation works on the operands one at a time: with sides 6 and
    7 the product has side 42, and no operation computes an array with even
    a quarter of its 1764 entries. The operands' own arrays have 36 and 49."""
    op = kron(DensePSD(jnp.asarray(_psd(6))), DensePSD(jnp.asarray(_psd(7))))
    n = op.dim
    calls = {
        "matvec": lambda o, x: o.matvec(x),
        "rmatvec": lambda o, x: o.rmatvec(x),
        "solve": lambda o, x: o.solve(x),
        "whiten": lambda o, x: o.whiten(x),
        "logdet": lambda o, x: o.logdet(),
        "diag": lambda o, x: o.diag(),
        "factor": lambda o, x: o.factor().matvec(x),
    }
    jaxpr = jax.make_jaxpr(calls[operation])(op, jnp.ones(n))
    assert _largest_intermediate(jaxpr.jaxpr) < n * n // 4
    # sentinel: to_dense does form it, so the count can see such an array
    dense = jax.make_jaxpr(lambda o: o.to_dense())(op)
    assert _largest_intermediate(dense.jaxpr) == n * n
