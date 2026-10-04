"""Conformance tests: every extensible type, through its contract's suite.

Two layers ship a conformance harness, because two layers are open to
extension. One instance per operator class — plus variants whose
capabilities, structure depth, block count, size, or sign differ — runs
through :func:`enskit.linalg.testing.check_operator`, each paired with a
second instance of the same structure and different values to make up the
family it is applied as under ``vmap``; and every shipped EKI policy runs
through the check for its axis in :mod:`enskit.eki.testing`, which is the
harness a user's own schedule, update rule or inflation is meant to be run
through.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import enskit  # noqa: F401  -- enables x64 before any array exists
from enskit.eki import (
    AdaptiveESSSchedule,
    AdaptiveMisfitSchedule,
    AdditiveInflation,
    DiscrepancyStop,
    FixedSchedule,
    MultiplicativeInflation,
    PathwiseUpdate,
    TransformUpdate,
)
from enskit.eki.testing import (
    check_inflation,
    check_schedule,
    check_stopping_rule,
    check_update,
)
from enskit.linalg import (
    BlockDiag,
    Dense,
    DensePSD,
    DenseSquare,
    Identity,
    IdentityPlusGram,
    LinOp,
    LowRankUpdate,
    PSDDiagonal,
    PSDLowRank,
    Transposed,
    Triangular,
    Zero,
    block_diag,
    diag_congruence,
    hstack,
    kron,
    product,
)
from enskit.linalg.testing import check_operator


def _instances(rng: np.random.Generator) -> list[LinOp]:
    """The conformance instances, with their values drawn from ``rng``.

    Every call builds the same list of types and shapes, so two calls with
    different generators pair each instance with a second, different one of
    identical pytree structure.
    """

    def normal(*shape):
        return jnp.asarray(rng.normal(size=shape))

    def uniform(low, high, n):
        return jnp.asarray(rng.uniform(low, high, n))

    def psd(n: int) -> jnp.ndarray:
        M = rng.normal(size=(n, n))
        return jnp.asarray(M @ M.T + n * np.eye(n))

    def square(n: int) -> jnp.ndarray:
        return jnp.asarray(rng.normal(size=(n, n)) + n * np.eye(n))

    d = uniform(0.5, 3.0, 6)
    well_conditioned = square(5)
    lu_t = jax.scipy.linalg.lu_factor(well_conditioned.T)
    return [
        Identity(6),
        2.5 * Identity(6),  # the scaled identity, via arithmetic
        PSDDiagonal(d),
        Dense(normal(4, 6)),
        Dense(normal(4, 4)),  # square, so a wrong contraction is silent
        DenseSquare(well_conditioned),
        Triangular(jnp.linalg.cholesky(psd(5)), lower=True),
        Triangular(jnp.linalg.cholesky(psd(4)).T, lower=False),
        DensePSD(psd(5)),
        DensePSD(L=jnp.linalg.cholesky(psd(4))),
        # negative pivots, so a logdet that drops the absolute value fails;
        # the triangular factor of a PSD matrix never has one
        Triangular(
            jnp.asarray(
                np.tril(rng.normal(size=(4, 4)), -1) + np.diag([2.0, -3.0, 1.5, -1.0])
            ),
            lower=True,
        ),
        Triangular(
            jnp.asarray(
                np.triu(rng.normal(size=(3, 3)), 1) + np.diag([-2.0, 1.0, 1.5])
            ),
            lower=False,
        ),
        DenseSquare(well_conditioned.at[0].multiply(-1.0)),
        DenseSquare(well_conditioned, lu=lu_t[0], piv=lu_t[1], lu_of_transpose=True),
        # a low-rank PSD operator at each width: thin (singular), square,
        # and wide (generically nonsingular, yet still no solve/whiten)
        PSDLowRank(normal(5, 2)),
        PSDLowRank(normal(4, 4)),
        PSDLowRank(normal(3, 6)),
        # size one, where a batch axis of length n and k = n + 1 = 2 meet
        Identity(1),
        PSDDiagonal(uniform(0.5, 3.0, 1)),
        Dense(normal(1, 3)),
        Dense(normal(3, 1)),
        DenseSquare(-jnp.asarray([[rng.uniform(0.5, 2.0)]])),
        # composites
        product(PSDDiagonal(d), Dense(normal(6, 4))),
        hstack(Dense(normal(5, 2)), DensePSD(psd(5))),
        BlockDiag((Dense(normal(2, 3)), Dense(normal(3, 3)))),
        block_diag(PSDDiagonal(d), DensePSD(psd(3))),
        block_diag(Identity(2), 4.0 * Identity(3)),
        # composites over a block that disclaims solve/whiten/logdet: the
        # capability intersection must survive, and the block-diagonal
        # factor is rectangular because one block's factor is
        block_diag(PSDDiagonal(d), PSDLowRank(normal(5, 2))),
        diag_congruence(PSDLowRank(normal(4, 2)), uniform(0.5, 2, 4)),
        2.5 * PSDLowRank(normal(4, 2)),
        diag_congruence(DensePSD(psd(4)), uniform(0.5, 2, 4)),
        # three or more blocks or factors, so split-point accumulation and
        # the order of composition are exercised
        hstack(Dense(normal(4, 2)), Dense(normal(4, 3)), Dense(normal(4, 1))),
        BlockDiag((Dense(normal(2, 3)), Dense(normal(1, 2)), Dense(normal(3, 1)))),
        block_diag(PSDDiagonal(uniform(0.5, 3.0, 2)), DensePSD(psd(3)), Identity(1)),
        product(Dense(normal(3, 4)), DensePSD(psd(4)), Dense(normal(4, 2))),
        # a square product, and nesting
        product(PSDDiagonal(d), DensePSD(psd(6))),
        block_diag(
            diag_congruence(DensePSD(psd(3)), uniform(0.5, 2, 3)), Identity(2)
        ),
        2.0 * block_diag(PSDDiagonal(uniform(0.5, 3.0, 2)), DensePSD(psd(3))),
        diag_congruence(
            diag_congruence(DensePSD(psd(3)), uniform(0.5, 2, 3)), uniform(0.5, 2, 3)
        ),
        block_diag(2.0 * DensePSD(psd(3)), 3.0 * PSDDiagonal(uniform(0.5, 3.0, 2))),
        # direct view construction, over a symmetric and a non-symmetric
        # operator; only the second shows a matvec/rmatvec swap
        Transposed(DensePSD(psd(4))),
        Transposed(DenseSquare(square(4))),
        # arithmetic-built composites, both signs
        2.0 * DensePSD(psd(4)),
        3.0 * DenseSquare(well_conditioned),
        -2.0 * DenseSquare(well_conditioned),
        2.0 * DenseSquare(square(4)).T,
        1.5 * Dense(normal(3, 5)),
        Dense(normal(3, 5)).T,
        # I + S S^T at each shape of S, and at exactly repeated and exactly
        # zero singular values, where the stored SVD's basis is arbitrary
        IdentityPlusGram(normal(5, 3)),
        IdentityPlusGram(normal(3, 5)),
        IdentityPlusGram(normal(4, 4)),
        IdentityPlusGram(uniform(0.5, 3.0, 1)[0] * jnp.eye(5, 3)),
        IdentityPlusGram(jnp.concatenate([normal(5, 2), jnp.zeros((5, 2))], axis=1)),
        IdentityPlusGram(normal(1, 1)),
        IdentityPlusGram(normal(5, 3)).inverse_sqrt(),
        IdentityPlusGram(normal(3, 5)).inverse_sqrt(),
        IdentityPlusGram(
            jnp.concatenate([normal(5, 2), jnp.zeros((5, 2))], axis=1)
        ).inverse_sqrt(),
        # the zero operator, alone and as a block a composite skips
        Zero(3, 4),
        Zero(4, 4),
        Zero(1, 1),
        hstack(Dense(normal(5, 2)), Zero(5, 3)),
        hstack(Zero(4, 1), Zero(4, 2)),
        product(Dense(normal(4, 3)), Zero(3, 2)),
        # a PSD base plus a low-rank term, over bases of each kind, a wide
        # factor, and a zero-padded factor (exact zero singular values)
        LowRankUpdate(PSDDiagonal(d), Dense(normal(6, 2))),
        LowRankUpdate(Identity(5), Dense(normal(5, 5))),
        LowRankUpdate(DensePSD(psd(4)), Dense(normal(4, 7))),
        LowRankUpdate(
            block_diag(PSDDiagonal(d[:3]), DensePSD(psd(2))), Dense(normal(5, 2))
        ),
        LowRankUpdate(PSDDiagonal(d), hstack(Dense(normal(6, 2)), Zero(6, 2))),
        LowRankUpdate(
            2.0 * DensePSD(psd(3)), product(Dense(normal(3, 2)), Dense(normal(2, 2)))
        ),
        # Kronecker products at each level. Sides differ, so a logdet that
        # pairs each operand with its own side fails; the PSD factors are
        # triangular, so factor() is a SquareKronecker that solves.
        kron(DensePSD(psd(3)), DensePSD(psd(4))),
        kron(PSDDiagonal(uniform(0.5, 3.0, 2)), DensePSD(psd(3))),
        kron(Identity(2), 3.0 * Identity(3)),
        kron(DensePSD(psd(2)), Identity(1)),
        # a rectangular factor, so factor() is a plain Kronecker, and an
        # operand without solve, whiten or logdet, so the product has none
        kron(DensePSD(psd(2)), PSDLowRank(normal(3, 2))),
        kron(PSDLowRank(normal(2, 3)), PSDDiagonal(uniform(0.5, 3.0, 2))),
        # square, not PSD: negative pivots and a transpose that keeps solve
        kron(DenseSquare(well_conditioned.at[0].multiply(-1.0)), DenseSquare(square(2))),
        kron(
            Triangular(jnp.asarray(np.diag([2.0, -3.0, 1.5])), lower=True),
            Triangular(jnp.linalg.cholesky(psd(2)).T, lower=False),
        ),
        # rectangular, with all four sizes distinct, so reshaping the operand
        # and the result alike is shape-invalid; and one square operand
        kron(Dense(normal(2, 3)), Dense(normal(4, 5))),
        kron(Dense(normal(3, 2)), DensePSD(psd(2))),
        kron(Dense(normal(1, 3)), Dense(normal(2, 1))),
        kron(Zero(2, 3), Dense(normal(2, 2))),
        # nested, as composites' children, and as a LowRankUpdate base
        kron(PSDDiagonal(uniform(0.5, 3.0, 2)), kron(DensePSD(psd(2)), Identity(2))),
        kron(kron(Dense(normal(2, 1)), Dense(normal(1, 2))), Dense(normal(2, 3))),
        block_diag(
            kron(DensePSD(psd(2)), PSDDiagonal(uniform(0.5, 3.0, 2))), Identity(2)
        ),
        2.0 * kron(DensePSD(psd(2)), DensePSD(psd(3))),
        LowRankUpdate(kron(DensePSD(psd(2)), PSDDiagonal(d[:3])), Dense(normal(6, 2))),
    ]


_PAIRS = list(
    zip(
        _instances(np.random.default_rng(0)),
        _instances(np.random.default_rng(1)),
        strict=True,
    )
)


@pytest.mark.parametrize(
    ("op", "other"), _PAIRS, ids=[type(op).__name__ for op, _ in _PAIRS]
)
def test_conformance(op, other):
    check_operator(op, other=other)


def test_densepsd_factor_solves():
    """The factor of a ``DensePSD`` built from its matrix solves exactly
    against the Cholesky factor, in both orientations."""
    rng = np.random.default_rng(2)
    M = rng.normal(size=(4, 4))
    A = M @ M.T + 4 * np.eye(4)
    C = np.linalg.cholesky(A)
    L = DensePSD(jnp.asarray(A)).factor()
    b = jnp.asarray(rng.normal(size=(3, 4)))
    np.testing.assert_allclose(L.to_dense(), C, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        L.solve(b), np.linalg.solve(C, np.asarray(b).T).T, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        L.T.solve(b), np.linalg.solve(C.T, np.asarray(b).T).T, rtol=1e-12, atol=1e-12
    )


# ---------------------------------------------------------------------------
# every shipped EKI policy, through the harness the layer ships for user ones
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "schedule",
    [
        FixedSchedule.uniform(4),
        FixedSchedule.constant(1.0, 1),
        FixedSchedule((0.25, 0.5, 0.25)),
        AdaptiveESSSchedule(),
        AdaptiveESSSchedule(beta_target=None, ess_fraction=0.3, n_bisect=12),
        AdaptiveMisfitSchedule(),
        AdaptiveMisfitSchedule(beta_target=None, divergence_budget=3.0),
    ],
    ids=repr,
)
def test_schedule_conformance(schedule):
    check_schedule(schedule)


@pytest.mark.parametrize("update", [TransformUpdate(), PathwiseUpdate()], ids=repr)
def test_update_conformance(update):
    check_update(update)


@pytest.mark.parametrize(
    "inflation",
    [
        MultiplicativeInflation(1.02),
        MultiplicativeInflation(2.0),
        AdditiveInflation(DensePSD(jnp.eye(3) * 0.05)),
        AdditiveInflation(PSDDiagonal(jnp.full((3,), 0.02))),
    ],
    ids=repr,
)
def test_inflation_conformance(inflation):
    check_inflation(inflation)


@pytest.mark.parametrize(
    "stop", [DiscrepancyStop(), DiscrepancyStop(tau=2.0)], ids=repr
)
def test_stopping_rule_conformance(stop):
    check_stopping_rule(stop)
