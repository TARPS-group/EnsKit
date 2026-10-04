"""Conformance and regression tests for the distribution layer.

The file has three sections. The first works through the numbered conformance
obligations of the "Distribution contract"; the second holds the regression
tests ported from ``tests/test_gauss.py`` under the contract's table, each
keeping its old docstring's reasoning; the third holds the new regression
tests the contract lists. Regression tests document why a rule exists, and
deleting one as redundant loses that.

Rules for the reference throughout:

- **The dense reference is hand-written here.** Plain dense linear algebra
  over means, anomalies and materialized operators, never routed through
  ``enskit.distribution`` or ``IdentityPlusGram``, so every comparison is
  between independent paths. ``exact_moment_ensemble`` may build inputs; it
  is never the reference for an output.
- **Exactness tests compare against closed forms**, at a tolerance of a few
  machine epsilons times the quantity's natural scale.
- **Every test draws from its own deterministic stream**, reseeded from the
  test's id.
"""

from __future__ import annotations

import math
import zlib
from decimal import Decimal, getcontext
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import enskit  # noqa: F401  -- enables x64 before any array exists
from enskit.distribution import (
    ConditionalMap,
    Ensemble,
    EnsembleGaussian,
    Gaussian,
    MatheronMap,
    SquareRootMap,
    effective_sample_size,
    exact_moment_ensemble,
    resample,
    reweight,
)
from enskit.linalg import (
    Dense,
    DensePSD,
    HStack,
    LowRankUpdate,
    Product,
    PSDDiagonal,
    PSDLinOp,
    PSDLowRank,
    Triangular,
    UnsupportedOpError,
    Zero,
    debug_checks,
    dense_matvec,
    kron,
    linop,
    tri_solve,
)
from enskit.linalg.testing import check_operator

RNG = np.random.default_rng(0)
EPS = float(np.finfo(np.float64).eps)


@pytest.fixture(autouse=True)
def _reseed_rng(request):
    """Give every test its own deterministic stream, seeded from its id."""
    global RNG
    RNG = np.random.default_rng(zlib.crc32(request.node.nodeid.encode()))


# ---------------------------------------------------------------------------
# fixtures: test-local operators
# ---------------------------------------------------------------------------


def _psd(n: int) -> np.ndarray:
    """A well-conditioned dense PSD matrix, as a NumPy array."""
    M = RNG.normal(size=(n, n))
    return M @ M.T + n * np.eye(n)


@linop
class RotatedWhitenPSD(PSDLinOp):
    """A dense PSD operator whose whitener carries an arbitrary rotation.

    ``Q L^-1`` whitens exactly when ``Q`` is orthogonal and ``L L^T`` is the
    matrix. No shipped operator exercises the freedom the operator contract
    leaves in the choice of whitener, so this class gives the layer two
    genuinely different whiteners for the same matrix.
    """

    L: Array
    Q: Array

    @classmethod
    def from_matrix(cls, A, Q) -> RotatedWhitenPSD:
        return cls(jnp.linalg.cholesky(jnp.asarray(A)), jnp.asarray(Q))

    @property
    def shape(self) -> tuple[int, int]:
        n = self.L.shape[-1]
        return (n, n)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.L.shape[:-2])

    def _matvec(self, x: Array) -> Array:
        return dense_matvec(self.L, dense_matvec(self.L.swapaxes(-1, -2), x))

    def _to_dense(self) -> Array:
        return self.L @ self.L.swapaxes(-1, -2)

    def _factor(self):
        return Triangular(self.L, lower=True)

    def _whiten(self, x: Array) -> Array:
        return dense_matvec(self.Q, tri_solve(self.L, x, lower=True))

    def _solve(self, b: Array) -> Array:
        return tri_solve(self.L, tri_solve(self.L, b, lower=True), lower=True, trans=1)

    def _logdet(self) -> Array:
        d = jnp.diagonal(self.L, axis1=-2, axis2=-1)
        return 2.0 * jnp.sum(jnp.log(d), axis=-1)

    def _diag(self) -> Array:
        return jnp.sum(self.L * self.L, axis=-1)


@linop
class WhitenOnlyPSD(PSDLinOp):
    """A PSD operator that whitens, but has no ``factor`` and no ``logdet``."""

    L: Array

    @property
    def shape(self) -> tuple[int, int]:
        n = self.L.shape[-1]
        return (n, n)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.L.shape[:-2])

    def _matvec(self, x: Array) -> Array:
        return dense_matvec(self.L, dense_matvec(self.L.swapaxes(-1, -2), x))

    def _to_dense(self) -> Array:
        return self.L @ self.L.swapaxes(-1, -2)

    def _whiten(self, x: Array) -> Array:
        return tri_solve(self.L, x, lower=True)


@linop
class CountingWhitenPSD(PSDLinOp):
    """A whitening operator that records how many vectors it has whitened.

    The count is a plain list on the instance, invisible to the pytree
    machinery; only the instance :meth:`counting` builds counts. It supports
    ``factor`` and ``logdet`` too, so it can stand in for any term.
    """

    L: Array

    @property
    def shape(self) -> tuple[int, int]:
        n = self.L.shape[-1]
        return (n, n)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.L.shape[:-2])

    def _matvec(self, x: Array) -> Array:
        return dense_matvec(self.L, dense_matvec(self.L.swapaxes(-1, -2), x))

    def _to_dense(self) -> Array:
        return self.L @ self.L.swapaxes(-1, -2)

    def _factor(self):
        return Triangular(self.L, lower=True)

    def _logdet(self) -> Array:
        d = jnp.diagonal(self.L, axis1=-2, axis2=-1)
        return 2.0 * jnp.sum(jnp.log(d), axis=-1)

    def _whiten(self, x: Array) -> Array:
        log = self.__dict__.get("log")
        if log is not None:
            log.append(1 if x.ndim == 1 else int(np.prod(x.shape[:-1])))
        return tri_solve(self.L, x, lower=True)

    @classmethod
    def counting(cls, A) -> CountingWhitenPSD:
        op = cls(jnp.linalg.cholesky(jnp.asarray(A)))
        object.__setattr__(op, "log", [])
        return op

    @property
    def count(self) -> int:
        return sum(object.__getattribute__(self, "log"))

    def reset(self) -> None:
        object.__getattribute__(self, "log").clear()


def test_local_operators_satisfy_the_operator_contract():
    """The test-local operators are contract-valid, so the layer's failures
    cannot be blamed on the fixtures."""
    Q, _ = np.linalg.qr(RNG.normal(size=(4, 4)))
    check_operator(RotatedWhitenPSD.from_matrix(_psd(4), Q))
    check_operator(WhitenOnlyPSD(jnp.linalg.cholesky(jnp.asarray(_psd(4)))))
    check_operator(CountingWhitenPSD.counting(_psd(4)))


# ---------------------------------------------------------------------------
# the hand-written dense reference
#
# Plain dense linear algebra over the NumPy arrays a problem was built from.
# Nothing below calls into enskit.distribution.
# ---------------------------------------------------------------------------


class _Dense:
    """A joint Gaussian as one dense mean vector and covariance matrix."""

    def __init__(self, dims, means, rows, covs, k):
        self.names = list(dims)
        self.dims = dict(dims)
        self.offsets, offset = {}, 0
        for name in self.names:
            self.offsets[name] = offset
            offset += dims[name]
        self.mu = np.concatenate([means[n] for n in self.names])
        F = np.concatenate(
            [rows.get(n, np.zeros((dims[n], k))) for n in self.names], axis=0
        )
        self.C = F @ F.T
        for name, D in covs.items():
            s = self.idx(name)
            self.C[np.ix_(s, s)] += D

    def idx(self, *names):
        return np.concatenate(
            [np.arange(self.offsets[n], self.offsets[n] + self.dims[n]) for n in names]
        )

    def mean(self, *names):
        return self.mu[self.idx(*names)]

    def cov(self, a, b=None):
        b = a if b is None else b
        return self.C[np.ix_(self.idx(a), self.idx(b))]

    def condition(self, values):
        given = [n for n in self.names if n in values]
        targets = [n for n in self.names if n not in values]
        c, t = self.idx(*given), self.idx(*targets)
        y = np.concatenate([values[n] for n in given])
        K = self.C[np.ix_(t, c)] @ np.linalg.inv(self.C[np.ix_(c, c)])
        return (
            self.mu[t] + K @ (y - self.mu[c]),
            self.C[np.ix_(t, t)] - K @ self.C[np.ix_(c, t)],
        )

    def log_density(self, values):
        named = [n for n in self.names if n in values]
        c = self.idx(*named)
        v = np.concatenate([np.asarray(values[n]) for n in named], axis=-1)
        C = self.C[np.ix_(c, c)]
        r = v - self.mu[c]
        quad = np.einsum("...i,ij,...j->...", r, np.linalg.inv(C), r)
        return -0.5 * (quad + np.linalg.slogdet(C)[1] + len(c) * np.log(2 * np.pi))


def _problem(dims, k, *, terms=(), rowless=(), term_op=DensePSD):
    """A random Gaussian over ``dims`` and its dense reference.

    ``terms`` names the blocks with independent terms; ``rowless`` the
    blocks without factor rows.
    """
    means = {b: RNG.normal(size=d) for b, d in dims.items()}
    rows = {
        b: RNG.normal(size=(d, k)) for b, d in dims.items() if k > 0 and b not in rowless
    }
    covs = {b: _psd(dims[b]) for b in terms}
    g = Gaussian(
        {b: jnp.asarray(m) for b, m in means.items()},
        factors={b: jnp.asarray(F) for b, F in rows.items()},
        block_covs={b: term_op(jnp.asarray(D)) for b, D in covs.items()},
        latent_dim=k,
    )
    return g, _Dense(dims, means, rows, covs, k)


def _ensemble_problem(J, dims, *, noise=("g",)):
    """An ensemble, its projection with noise added, and the dense reference."""
    arrays = {b: RNG.normal(size=(J, d)) for b, d in dims.items()}
    ens = Ensemble({b: jnp.asarray(a) for b, a in arrays.items()})
    covs = {b: _psd(dims[b]) for b in noise}
    approx = ens.project().add_noise(
        {b: DensePSD(jnp.asarray(R)) for b, R in covs.items()}
    )
    means = {b: a.mean(axis=0) for b, a in arrays.items()}
    rows = {b: (a - a.mean(axis=0)).T / np.sqrt(J - 1) for b, a in arrays.items()}
    return ens, approx, _Dense(dims, means, rows, covs, J), arrays, covs


def _close(got, want, scale=None, factor=1e3):
    got, want = np.asarray(got), np.asarray(want)
    scale = max(1.0, np.abs(want).max()) if scale is None else scale
    np.testing.assert_allclose(got, want, rtol=0, atol=factor * EPS * scale)


def _cov_dense(g, name):
    return np.asarray(g.cov(name).to_dense())


def _recovered_whitener(D, n: int) -> np.ndarray:
    """The whitener a term applies, as a dense matrix: ``whiten(I)`` transposed."""
    return np.asarray(D.whiten(jnp.eye(n))).T


def _sample_moments(x: np.ndarray):
    """Sample mean and covariance (divisor J - 1) of a row-wise ensemble."""
    x = np.asarray(x)
    a = x - x.mean(axis=0)
    return x.mean(axis=0), a.T @ a / (x.shape[0] - 1)


def _stacked(obj, reps: int = 3):
    """A vmapped family, built the way a pytree reconstruction builds one."""
    return jax.tree_util.tree_map(lambda leaf: jnp.stack([leaf] * reps), obj)


def _count_svd(jaxpr) -> int:
    """Count ``svd`` primitives in a jaxpr, recursing into sub-jaxprs."""
    total = 0
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "svd":
            total += 1
        for param in eqn.params.values():
            candidates = param if isinstance(param, (tuple, list)) else [param]
            for candidate in candidates:
                inner = getattr(candidate, "jaxpr", candidate)
                if hasattr(inner, "eqns"):
                    total += _count_svd(inner)
    return total


def _linear_gaussian(P: int, N: int):
    """A linear-Gaussian joint (u, g = H u + e), and its NumPy parts."""
    m0 = RNG.normal(size=P)
    L0 = np.linalg.cholesky(_psd(P))
    H = RNG.normal(size=(N, P))
    R = _psd(N)
    joint = Gaussian(
        {"u": jnp.asarray(m0), "g": jnp.asarray(H @ m0)},
        factors={"u": jnp.asarray(L0), "g": jnp.asarray(H @ L0)},
        block_covs={"g": DensePSD(jnp.asarray(R))},
    )
    ref = _Dense(
        {"u": P, "g": N}, {"u": m0, "g": H @ m0}, {"u": L0, "g": H @ L0}, {"g": R}, P
    )
    return joint, ref, {"m0": m0, "L0": L0, "H": H, "R": R}


# ===========================================================================
# Section 1 -- the conformance obligations
# ===========================================================================

# --- 1. gain against dense ---------------------------------------------------

REGIMES = [
    pytest.param(5, {"x": 3, "g": 8}, id="N>k"),
    pytest.param(6, {"x": 3, "g": 6}, id="N=k"),
    pytest.param(9, {"x": 3, "g": 4}, id="N<k"),
]


@pytest.mark.parametrize(("k", "dims"), REGIMES)
def test_1_noisy_conditioning_matches_the_dense_gain(k, dims):
    g, ref = _problem(dims, k, terms=("g",))
    y = RNG.normal(size=dims["g"])
    post = g.condition(g=jnp.asarray(y))
    m_ref, C_ref = ref.condition({"g": y})
    _close(post.mean("x"), m_ref)
    _close(_cov_dense(post, "x"), C_ref)


@pytest.mark.parametrize("k", [3, 6, 12])
def test_1_several_given_blocks_and_blocks_without_rows(k):
    """Several given blocks, in an order different from how they were spelled,
    one of them without a factor row; a target without a row is independent
    of them and comes back unchanged."""
    dims = {"x": 3, "a": 2, "z": 2, "b": 3}
    g, ref = _problem(dims, k, terms=("a", "b", "z"), rowless=("b", "z"))
    values = {"b": RNG.normal(size=3), "a": RNG.normal(size=2)}
    post = g.condition({n: jnp.asarray(v) for n, v in values.items()})
    assert post.names == ("x", "z")
    m_ref, C_ref = ref.condition(values)
    _close(post.mean("x"), m_ref[:3])
    _close(_cov_dense(post, "x"), C_ref[:3, :3])
    assert post.mean("z") is g.mean("z") and post.factor("z") is None
    assert post.block_cov("z") is g.block_cov("z")


# --- 2. whitener invariance --------------------------------------------------


def test_2_results_do_not_depend_on_the_choice_of_whitener():
    J, dims = 7, {"x": 3, "g": 4}
    ens = Ensemble({b: jnp.asarray(RNG.normal(size=(J, d))) for b, d in dims.items()})
    R = _psd(4)
    Q, _ = np.linalg.qr(RNG.normal(size=(4, 4)))
    plain = ens.project().add_noise(g=DensePSD(jnp.asarray(R)))
    rotated = ens.project().add_noise(g=RotatedWhitenPSD.from_matrix(R, Q))
    W1 = _recovered_whitener(plain.block_cov("g"), 4)
    W2 = _recovered_whitener(rotated.block_cov("g"), 4)
    assert np.abs(W1 - W2).max() > 0.1, "the two whiteners must differ"

    y = jnp.asarray(RNG.normal(size=4))
    key = jax.random.key(3)
    a, b = plain.condition(g=y), rotated.condition(g=y)
    _close(a.mean("x"), b.mean("x"))
    _close(_cov_dense(a, "x"), _cov_dense(b, "x"))
    _close(plain.log_density(g=y), rotated.log_density(g=y))
    _close(plain.square_root_map("g")(g=y)["x"], rotated.square_root_map("g")(g=y)["x"])
    # keyless Matheron is invariant elementwise; with a key, the whitened draw
    # is mapped through different whiteners, so only the law is shared
    _close(
        plain.conditional_map("g")(ens, g=y)["x"],
        rotated.conditional_map("g")(ens, g=y)["x"],
    )
    assert plain.conditional_map("g").particle_coefficients(g=y, key=key).shape == (J, J)


# --- 3. the transform --------------------------------------------------------


def _with_spectrum(rows: int, cols: int, sigmas, seed: int = 11) -> np.ndarray:
    """A ``(rows, cols)`` matrix with exactly the prescribed singular values."""
    rng = np.random.default_rng(seed)
    Q1, _ = np.linalg.qr(rng.normal(size=(rows, rows)))
    Q2, _ = np.linalg.qr(rng.normal(size=(cols, cols)))
    D = np.zeros((rows, cols))
    for i, value in enumerate(sigmas):
        D[i, i] = value
    return Q1 @ D @ Q2.T


def _transform_of(Fc: np.ndarray):
    """Condition a Gaussian with whitened factor ``S = Fc^T`` (identity noise)
    and return the transform ``T`` the layer computed, as a dense array."""
    N, k = Fc.shape
    g = Gaussian(
        {"x": jnp.zeros(k), "c": jnp.zeros(N)},
        factors={"x": jnp.eye(k), "c": jnp.asarray(Fc)},
        block_covs={"c": PSDDiagonal(jnp.ones(N))},
    )
    row = g.condition(c=jnp.zeros(N)).factor("x")
    assert isinstance(row, Product)
    return np.asarray(row.ops[-1].to_dense())


def test_3_the_stably_formed_invariant_holds_at_large_sigma_max():
    """``T T^T + (T S)(T S)^T = I`` to ``eps * sigma_max``, on a spectrum
    mixing large singular values with ones of order 1."""
    k, N, sigma_max = 5, 7, 1e10
    S = _with_spectrum(k, N, [sigma_max, 1e5, 1.0, 1.0, 1.0])
    T = _transform_of(S.T)
    TS = T @ S
    assert np.abs(T @ T.T + TS @ TS.T - np.eye(k)).max() <= 128 * EPS * sigma_max


@pytest.mark.parametrize("J", [5, 40])
def test_3_the_transform_fixes_the_ones_vector_for_a_centered_factor(J):
    """``T 1 = 1`` to ``c1 J eps + c2 (eps sigma_max)^2``, at more than one J."""
    N = 3
    for scale in (1.0, 1e4):
        X = RNG.normal(size=(J, N)) * scale
        ens = Ensemble(x=jnp.asarray(RNG.normal(size=(J, 2))), c=jnp.asarray(X))
        approx = ens.project().add_noise(c=PSDDiagonal(jnp.ones(N)))
        row = approx.condition(c=jnp.zeros(N)).factor("x")
        T = np.asarray(row.ops[-1].to_dense())
        A = X - X.mean(axis=0)
        sigma_max = np.linalg.svd(A / np.sqrt(J - 1), compute_uv=False).max()
        bound = 16 * J * EPS + 16 * (EPS * sigma_max) ** 2
        assert np.abs(T @ np.ones(J) - 1.0).max() <= bound


# --- 4. exact values ---------------------------------------------------------


@pytest.mark.parametrize("k", [4, 9])
def test_4_exact_conditioning_and_density_match_the_dense_forms(k):
    dims = {"x": 3, "c": 4}
    g, ref = _problem(dims, k)
    y = RNG.normal(size=4)
    post = g.condition(c=jnp.asarray(y))
    m_ref, C_ref = ref.condition({"c": y})
    _close(post.mean("x"), m_ref)
    _close(_cov_dense(post, "x"), C_ref)
    assert isinstance(post.factor("x"), Dense)
    v = RNG.normal(size=(2, 4))
    _close(g.log_density(c=jnp.asarray(v)), ref.log_density({"c": v}))


def test_4_exact_size_conditions_raise():
    g, _ = _problem({"x": 2, "c": 5}, 4)
    with pytest.raises(ValueError, match="N <= k"):
        g.condition(c=jnp.zeros(5))
    with pytest.raises(ValueError, match="N <= k"):
        g.log_density(c=jnp.zeros(5))
    # J = N on an EnsembleGaussian passes N <= k but not N <= J - 1
    J = 4
    ens = Ensemble(
        x=jnp.asarray(RNG.normal(size=(J, 2))), c=jnp.asarray(RNG.normal(size=(J, J)))
    )
    with pytest.raises(ValueError, match="J - 1"):
        ens.project().condition(c=jnp.zeros(J))
    with pytest.raises(ValueError, match="J - 1"):
        ens.project().log_density(c=jnp.zeros(J))


def test_4_the_exact_rank_check_is_debug_only():
    F = RNG.normal(size=(3, 5))
    F[2] = F[0] + F[1]  # rank 2
    g = Gaussian(
        {"x": jnp.zeros(2), "c": jnp.zeros(3)},
        factors={"x": jnp.asarray(RNG.normal(size=(2, 5))), "c": jnp.asarray(F)},
    )
    g.condition(c=jnp.ones(3))  # no raise outside debug mode
    with debug_checks(True):
        with pytest.raises(ValueError, match="rank deficient"):
            g.condition(c=jnp.ones(3))
        with pytest.raises(ValueError, match="rank deficient"):
            g.log_density(c=jnp.ones(3))


# --- 5. square root ----------------------------------------------------------


def test_5_the_square_root_map_equals_condition_then_realize():
    J = 9
    ens, approx, *_ = _ensemble_problem(J, {"x": 3, "g": 4})
    y = jnp.asarray(RNG.normal(size=4))
    smap = approx.square_root_map("g")
    assert isinstance(smap, SquareRootMap) and smap.targets == ("x",)
    _close(smap(g=y)["x"], approx.condition(g=y).realize_particles()["x"])


def test_5_square_root_moments_equal_the_exact_conditional():
    """On particles with the joint's exact moments, the output's sample mean
    and covariance equal the exact conditional's."""
    P, N, J = 3, 4, 10
    joint, ref, parts = _linear_gaussian(P, N)
    prior = joint.marginal("u")
    u = exact_moment_ensemble(jax.random.key(0), prior, J)
    ens = u.assign(g=u["u"] @ jnp.asarray(parts["H"]).T)
    approx = ens.project().add_noise(g=DensePSD(jnp.asarray(parts["R"])))
    y = RNG.normal(size=N)
    m_ref, C_ref = ref.condition({"g": y})
    mean, cov = _sample_moments(approx.square_root_map("g")(g=jnp.asarray(y))["u"])
    _close(mean, m_ref, factor=1e4)
    _close(cov, C_ref, factor=1e4)


def test_5_one_svd_at_build_and_none_per_call():
    J = 6
    _, approx, *_ = _ensemble_problem(J, {"x": 2, "g": 3})
    build = jax.make_jaxpr(lambda a: a.square_root_map("g"))(approx)
    assert _count_svd(build.jaxpr) == 1
    smap = approx.square_root_map("g")
    call = jax.make_jaxpr(lambda m, y: m(g=y))(smap, jnp.zeros(3))
    assert _count_svd(call.jaxpr) == 0


# --- 6. Matheron -------------------------------------------------------------


def test_6_the_map_equals_the_dense_perturbed_update_elementwise():
    J, n = 8, 5
    _, approx, ref, _, covs = _ensemble_problem(J, {"x": 3, "g": 4})
    samples = Ensemble(
        x=jnp.asarray(RNG.normal(size=(n, 3))), g=jnp.asarray(RNG.normal(size=(n, 4)))
    )
    y = RNG.normal(size=4)
    key = jax.random.key(7)
    got = np.asarray(approx.conditional_map("g")(samples, g=jnp.asarray(y), key=key)["x"])

    eps = np.asarray(jax.random.normal(key, (n, 4)))
    W = _recovered_whitener(approx.block_cov("g"), 4)
    K = ref.cov("x", "g") @ np.linalg.inv(ref.cov("g"))
    perturbed = np.asarray(samples["g"]) + np.linalg.solve(W, eps.T).T
    want = np.asarray(samples["x"]) + (y - perturbed) @ K.T
    _close(got, want)
    # without a key the samples carry their noise, and nothing is drawn
    keyless = np.asarray(approx.conditional_map("g")(samples, g=jnp.asarray(y))["x"])
    _close(keyless, np.asarray(samples["x"]) + (y - np.asarray(samples["g"])) @ K.T)


def test_6_exact_moment_joint_samples_are_carried_to_the_conditional():
    P, N, J = 3, 4, 12
    joint, ref, _ = _linear_gaussian(P, N)
    samples = exact_moment_ensemble(jax.random.key(1), joint, J)  # s = P + N
    y = RNG.normal(size=N)
    m_ref, C_ref = ref.condition({"g": y})
    moved = joint.conditional_map("g")(samples, g=jnp.asarray(y))
    assert moved.names == ("u",)
    mean, cov = _sample_moments(moved["u"])
    _close(mean, m_ref, factor=1e4)
    _close(cov, C_ref, factor=1e4)


def test_6_particle_coefficients_agree_with_the_call_on_realized_particles():
    J = 7
    _, approx, *_ = _ensemble_problem(J, {"x": 3, "g": 4})
    cmap = approx.conditional_map("g")
    y, key = jnp.asarray(RNG.normal(size=4)), jax.random.key(2)
    fast = cmap.particle_coefficients(g=y, key=key)
    slow = cmap.coefficients(
        approx.realize_particles(exclude_block_covs=("g",)), g=y, key=key
    )
    _close(fast, slow, scale=np.abs(np.asarray(slow)).max())


# --- 7. project and realize --------------------------------------------------


def test_7_project_reproduces_every_moment_unweighted():
    J, dims = 6, {"x": 3, "y": 2}
    arrays = {b: RNG.normal(size=(J, d)) for b, d in dims.items()}
    ens = Ensemble({b: jnp.asarray(a) for b, a in arrays.items()})
    g = ens.project()
    assert isinstance(g, EnsembleGaussian) and g.n_particles == J == g.latent_dim
    A = {b: a - a.mean(axis=0) for b, a in arrays.items()}
    for b in dims:
        _close(g.mean(b), arrays[b].mean(axis=0))
        _close(_cov_dense(g, b), A[b].T @ A[b] / (J - 1))
        F = np.asarray(g.factor(b).to_dense())
        assert np.abs(F.sum(axis=1)).max() <= 10 * J * EPS * np.abs(F).max()
    _close(g.cov("x", "y").to_dense(), A["x"].T @ A["y"] / (J - 1))
    _close(ens.cov("x", "y").to_dense(), A["x"].T @ A["y"] / (J - 1))
    _close(ens.cov("x").to_dense(), A["x"].T @ A["x"] / (J - 1))
    for b in dims:
        _close(g.realize_particles()[b], arrays[b])


def test_7_project_reproduces_every_moment_weighted():
    J, dims = 6, {"x": 3, "y": 2}
    arrays = {b: RNG.normal(size=(J, d)) for b, d in dims.items()}
    lw = RNG.normal(size=J)
    ens = Ensemble(
        {b: jnp.asarray(a) for b, a in arrays.items()}, log_weights=jnp.asarray(lw)
    )
    g = ens.project()
    assert type(g) is Gaussian and g.latent_dim == J
    w = np.exp(lw - lw.max())
    w /= w.sum()
    m = {b: w @ a for b, a in arrays.items()}
    A = {b: a - m[b] for b, a in arrays.items()}
    div = 1 - np.sum(w**2)
    for b in dims:
        _close(g.mean(b), m[b])
        _close(_cov_dense(g, b), (w[:, None] * A[b]).T @ A[b] / div)
    _close(g.cov("x", "y").to_dense(), (w[:, None] * A["x"]).T @ A["y"] / div)
    _close(ens.cov("x", "y").to_dense(), (w[:, None] * A["x"]).T @ A["y"] / div)


def test_7_the_weighted_divisor_stays_accurate_near_one_dominant_weight():
    """At an ESS of ``1 + 1e-6`` the naive ``1 - sum(w**2)`` keeps about
    ten digits; the log-space form keeps them all."""
    J, L = 4, 15.2
    lw = np.array([0.0, -L, -L, -L])
    ens = Ensemble(x=jnp.asarray(RNG.normal(size=(J, 1))), log_weights=jnp.asarray(lw))
    getcontext().prec = 60
    e = [Decimal(float(v)).exp() for v in lw]
    total = sum(e)
    exact = float(1 - sum((x / total) ** 2 for x in e))
    ess = float(effective_sample_size(ens))
    assert 1 < ess < 1 + 1e-5
    got = float(ens._weight_pieces()[2])
    assert abs(got - exact) <= 8 * EPS * exact
    w = np.asarray(ens.weights)
    naive = 1 - np.sum(w**2)
    assert abs(naive - exact) > 1e3 * abs(got - exact)


def test_7_concentrated_weights_raise_in_debug_mode_only():
    lw = jnp.asarray([0.0, -1e4, -1e4])
    ens = Ensemble(x=jnp.asarray(RNG.normal(size=(3, 2))), log_weights=lw)
    assert not bool(jnp.all(jnp.isfinite(ens.project().factor("x").to_dense())))
    with debug_checks(True):
        with pytest.raises(ValueError, match="effective sample size"):
            ens.project()


# --- 8. exact moments --------------------------------------------------------


def test_8_exact_moments_jointly_with_and_without_terms():
    for terms in ((), ("y",)):
        g, ref = _problem({"x": 3, "y": 2}, 2, terms=terms)
        s = 2 + (2 if terms else 0)
        ens = exact_moment_ensemble(jax.random.key(0), g, s + 3)
        X = np.concatenate([np.asarray(ens["x"]), np.asarray(ens["y"])], axis=1)
        mean, cov = _sample_moments(X)
        _close(mean, ref.mu, factor=1e2)
        _close(cov, ref.C, factor=1e2)
        with pytest.raises(ValueError, match="exceed the number of sources"):
            exact_moment_ensemble(jax.random.key(0), g, s)


# --- 9. densities and sampling -----------------------------------------------


@pytest.mark.parametrize("batch", [(), (3,), (2, 3)])
def test_9_log_density_matches_the_dense_closed_form(batch):
    dims = {"x": 3, "g": 4, "h": 2}
    for k in (0, 5):
        g, ref = _problem(dims, k, terms=("x", "g", "h"))
        for named in (("g",), ("g", "h"), ("x", "g", "h")):
            values = {n: RNG.normal(size=(*batch, dims[n])) for n in named}
            got = g.log_density({n: jnp.asarray(v) for n, v in values.items()})
            assert got.shape == batch
            _close(
                got, ref.log_density(values), scale=np.abs(ref.log_density(values)).max()
            )


def test_9_sample_matches_its_pinned_definition():
    g, _ = _problem({"x": 3, "y": 2, "z": 2}, 4, terms=("y", "z"), rowless=("z",))
    key, n = jax.random.key(4), 5
    got = g.sample(key, n)
    keys = jax.random.split(key, 3)
    xi = jax.random.normal(keys[0], (n, 4))
    want = {
        "x": g.mean("x") + g.factor("x").matvec(xi),
        "y": g.mean("y")
        + g.factor("y").matvec(xi)
        + g.block_cov("y").factor().matvec(jax.random.normal(keys[1], (n, 2))),
        "z": g.mean("z")
        + g.block_cov("z").factor().matvec(jax.random.normal(keys[2], (n, 2))),
    }
    for b in want:
        np.testing.assert_array_equal(got[b], want[b])


def test_9_realize_particles_matches_its_pinned_definition():
    J = 5
    _, approx, *_ = _ensemble_problem(J, {"x": 3, "g": 2, "h": 2}, noise=("g", "h"))
    key = jax.random.key(8)
    got = approx.realize_particles(key=key, exclude_block_covs=("g",))
    keys = jax.random.split(key, 1)
    sqrt = math.sqrt(J - 1)
    for b in ("x", "g"):
        want = approx.mean(b) + sqrt * approx.factor(b).to_dense().T
        np.testing.assert_array_equal(got[b], want)
    want_h = (
        approx.mean("h")
        + sqrt * approx.factor("h").to_dense().T
        + approx.block_cov("h").factor().matvec(jax.random.normal(keys[0], (J, 2)))
    )
    np.testing.assert_array_equal(got["h"], want_h)


# --- 10. structure -----------------------------------------------------------


def _joint_cov(g, names):
    """The dense joint covariance of ``names``, assembled from ``cov``."""
    return np.block([[np.asarray(g.cov(a, b).to_dense()) for b in names] for a in names])


def test_10_marginal_drop_and_rename_keep_the_distribution_and_the_kind():
    J = 6
    _, approx, ref, *_ = _ensemble_problem(J, {"x": 3, "y": 2, "g": 2})
    for g in (approx.marginal("g", "x"), approx.drop("y")):
        assert isinstance(g, EnsembleGaussian) and g.latent_dim == J
    assert approx.marginal("g", "x").names == ("g", "x")
    assert approx.drop("y").names == ("x", "g")
    _close(
        _joint_cov(approx.marginal("g", "x"), ("g", "x")),
        ref.C[np.ix_(ref.idx("g", "x"), ref.idx("g", "x"))],
    )
    renamed = approx.rename({"x": "y", "y": "x"}, g="h")  # a swap, and a keyword
    assert renamed.names == ("y", "x", "h") and isinstance(renamed, EnsembleGaussian)
    _close(_cov_dense(renamed, "y"), ref.cov("x"))
    assert renamed.block_cov("h") is approx.block_cov("g")


def test_10_add_noise_keeps_the_kind_on_a_new_term_and_absorbs_an_old_one():
    J = 5
    _, approx, ref, _, covs = _ensemble_problem(J, {"x": 3, "g": 2})
    R2 = _psd(2)
    again = approx.add_noise(g=DensePSD(jnp.asarray(R2)))
    assert type(again) is Gaussian and again.latent_dim == J + 2
    _close(_cov_dense(again, "g"), ref.cov("g") + R2)
    _close(again.cov("x", "g").to_dense(), ref.cov("x", "g"))
    fresh = approx.add_noise(x=DensePSD(jnp.eye(3)))
    assert isinstance(fresh, EnsembleGaussian) and fresh.latent_dim == J
    _close(_cov_dense(fresh, "x"), ref.cov("x") + np.eye(3))


def test_10_absorb_pads_with_zero_and_materializes_nothing():
    g, ref = _problem({"x": 3, "y": 2, "z": 2}, 4, terms=("x", "y", "z"), rowless=("z",))
    out = g.absorb("z", "x")  # block order, not argument order
    assert type(out) is Gaussian and out.latent_dim == 4 + 3 + 2
    assert out.block_cov("x") is None and out.block_cov("z") is None
    assert out.block_cov("y") is g.block_cov("y")
    row_x, row_y, row_z = out.factor("x"), out.factor("y"), out.factor("z")
    assert isinstance(row_x, HStack)
    assert [type(op) for op in row_x.ops] == [Dense, Triangular, Zero]
    assert row_x.ops[0] is g.factor("x")
    assert [type(op) for op in row_y.ops] == [Dense, Zero]
    assert [type(op) for op in row_z.ops] == [Zero, Triangular]
    assert row_z.ops[0].shape == (2, 4 + 3)
    _close(_joint_cov(out, ("x", "y", "z")), ref.C)
    # a row that is a single operator is that operator itself
    alone = Gaussian.independent(u=(jnp.zeros(2), DensePSD(jnp.asarray(_psd(2))))).absorb(
        "u"
    )
    assert isinstance(alone.factor("u"), Triangular)


def test_10_compress_keeps_the_distribution_and_always_returns_a_plain_gaussian():
    J = 9
    _, approx, ref, *_ = _ensemble_problem(J, {"x": 2, "g": 3})
    wide = approx.absorb("g")  # k = 12 > D_F = 5
    small = wide.compress()
    assert type(small) is Gaussian and small.latent_dim == 5
    _close(_joint_cov(small, ("x", "g")), ref.C)
    narrow = approx.drop("g").compress(max_dim=10)  # k = 9 > D_F = 2, compressed
    assert narrow.latent_dim == 2
    kept = Ensemble(x=jnp.asarray(RNG.normal(size=(3, 5)))).project().compress()
    assert type(kept) is Gaussian and kept.latent_dim == 3  # k <= D_F: unchanged


def test_10_compress_raises_over_max_dim_before_allocating():
    """Stacking this row would allocate 10^5 x 10^5 doubles, 80 GB."""
    n = 10**5
    g = Gaussian({"x": jnp.zeros(n)}, factors={"x": Zero(n, n + 1)})
    with pytest.raises(ValueError, match="max_dim"):
        g.compress()
    # nothing to compress at k <= D_F, so nothing to refuse
    assert (
        Gaussian({"x": jnp.zeros(n)}, factors={"x": Zero(n, 7)}).compress().latent_dim
        == 7
    )
    with pytest.raises(TypeError, match="max_dim"):
        g.compress(max_dim=4.0)
    with pytest.raises(ValueError, match="max_dim"):
        g.compress(max_dim=0)


def test_10_cov_returns_the_documented_operator():
    g, ref = _problem({"a": 3, "b": 2, "c": 2}, 4, terms=("a", "c"), rowless=("c",))
    assert isinstance(g.cov("a"), LowRankUpdate)
    assert isinstance(g.cov("b"), PSDLowRank)
    assert g.cov("c") is g.block_cov("c")
    assert isinstance(g.cov("a", "b"), Product)
    assert isinstance(g.cov("a", "c"), Zero) and g.cov("a", "c").shape == (3, 2)
    for a in ("a", "b", "c"):
        _close(_cov_dense(g, a), ref.cov(a))
        _close(g.cov(a, a).to_dense(), ref.cov(a))
    _close(g.cov("a", "b").to_dense(), ref.cov("a", "b"))
    with pytest.raises(UnsupportedOpError, match="whiten"):
        Gaussian(
            {"a": jnp.zeros(2)},
            factors={"a": jnp.ones((2, 1))},
            block_covs={"a": PSDLowRank(jnp.ones((2, 2)))},
        ).cov("a")


# --- 11. weights -------------------------------------------------------------


def test_11_weights_and_ess_for_log_weights_spanning_hundreds_of_units():
    lw = np.array([500.0, 0.0, -300.0, 499.0])
    ens = Ensemble(x=jnp.asarray(RNG.normal(size=(4, 2))), log_weights=jnp.asarray(lw))
    getcontext().prec = 50
    e = [Decimal(float(v)).exp() for v in lw]
    w = [x / sum(e) for x in e]
    _close(ens.weights, [float(x) for x in w])
    _close(effective_sample_size(ens), float(1 / sum(x * x for x in w)))
    assert effective_sample_size(Ensemble(x=jnp.zeros((5, 1)))).shape == ()
    assert float(effective_sample_size(Ensemble(x=jnp.zeros((5, 1))))) == 5.0


def test_11_reweight_composes_and_marks_zero_weights():
    ens = Ensemble(x=jnp.asarray(RNG.normal(size=(4, 2))))
    a, b = RNG.normal(size=4), RNG.normal(size=4)
    twice = reweight(reweight(ens, jnp.asarray(a)), jnp.asarray(b))
    _close(twice.log_weights, a + b)
    zeroed = reweight(ens, jnp.asarray([0.0, -np.inf, 0.0, 0.0]))
    assert float(zeroed.weights[1]) == 0.0 and zeroed.is_weighted
    with pytest.raises(ValueError, match=r"\(4,\)"):
        reweight(ens, jnp.zeros(3))


def test_11_resampling_matches_its_pinned_definitions():
    lw = jnp.asarray(RNG.normal(size=5))
    ens = Ensemble(x=jnp.asarray(RNG.normal(size=(5, 2))), log_weights=lw)
    key = jax.random.key(9)
    out = resample(key, ens, 7)
    w = np.asarray(ens.weights)
    u = float(jax.random.uniform(key, (), jnp.float64))
    idx = np.clip(
        np.searchsorted(np.cumsum(w), (u + np.arange(7)) / 7, side="right"), 0, 4
    )
    np.testing.assert_array_equal(out["x"], np.asarray(ens["x"])[idx])
    assert not out.is_weighted and out.n_particles == 7
    multi = resample(key, ens, scheme="multinomial")
    idx = np.asarray(jax.random.categorical(key, lw, shape=(5,)))
    np.testing.assert_array_equal(multi["x"], np.asarray(ens["x"])[idx])
    flat = resample(key, Ensemble(x=ens["x"]), scheme="multinomial")
    idx = np.asarray(jax.random.categorical(key, jnp.zeros(5), shape=(5,)))
    np.testing.assert_array_equal(flat["x"], np.asarray(ens["x"])[idx])


# --- 12. degeneracy ----------------------------------------------------------


def test_12_zero_given_anomalies_leave_the_targets_unchanged():
    J = 6
    X = RNG.normal(size=(J, 3))
    ens = Ensemble(x=jnp.asarray(X), g=jnp.full((J, 2), 0.75))
    approx = ens.project().add_noise(g=DensePSD(jnp.asarray(_psd(2))))
    y = jnp.asarray(RNG.normal(size=2))
    post = approx.condition(g=y)
    np.testing.assert_array_equal(post.mean("x"), approx.mean("x"))
    _close(_cov_dense(post, "x"), _cov_dense(approx, "x"))
    _close(approx.square_root_map("g")(g=y)["x"], X)
    w = approx.conditional_map("g").particle_coefficients(g=y, key=jax.random.key(0))
    np.testing.assert_array_equal(w, jnp.zeros((J, J)))
    moved = ens["x"] + approx.factor("x").matvec(w)
    np.testing.assert_array_equal(moved, ens["x"])


def test_12_the_smallest_sizes_work_and_a_collapsed_ensemble_gives_no_nan():
    ens, approx, ref, arrays, covs = _ensemble_problem(2, {"x": 1, "g": 1})
    y = RNG.normal(size=1)
    m_ref, C_ref = ref.condition({"g": y})
    _close(approx.condition(g=jnp.asarray(y)).mean("x"), m_ref)
    collapsed = Ensemble(x=jnp.ones((4, 2)), g=jnp.ones((4, 3)))
    approx = collapsed.project().add_noise(g=PSDDiagonal(jnp.ones(3)))
    for out in (
        approx.condition(g=jnp.zeros(3)).factor("x").to_dense(),
        approx.square_root_map("g")(g=jnp.zeros(3))["x"],
        approx.conditional_map("g")(collapsed, g=jnp.zeros(3), key=jax.random.key(0))[
            "x"
        ],
        approx.log_density(g=jnp.zeros(3)),
    ):
        assert bool(jnp.all(jnp.isfinite(out)))


# --- 13. validation ----------------------------------------------------------


def test_13_ensemble_construction():
    x = jnp.zeros((4, 2))
    with pytest.raises(ValueError, match="at least one block"):
        Ensemble()
    with pytest.raises(ValueError, match="rank"):
        Ensemble(x=jnp.zeros(4))
    with pytest.raises(ValueError, match="no columns"):
        Ensemble(x=jnp.zeros((4, 0)))
    with pytest.raises(ValueError, match="at least 2 particles"):
        Ensemble(x=jnp.zeros((1, 2)))
    with pytest.raises(ValueError, match=r"\(4, d\)"):
        Ensemble(x=x, y=jnp.zeros((5, 2)))
    with pytest.raises(TypeError, match="floating"):
        Ensemble(x=jnp.zeros((4, 2), int))
    with pytest.raises(TypeError, match="dtype"):
        Ensemble(x=x, y=jnp.zeros((4, 2), jnp.float32))
    with pytest.raises(TypeError, match="twice"):
        Ensemble({"x": x}, x=x)
    with pytest.raises(ValueError, match=r"\(4,\)"):
        Ensemble(x=x, log_weights=jnp.zeros(3))
    with pytest.raises(TypeError, match="dtype"):
        Ensemble(x=x, log_weights=jnp.zeros(4, jnp.float32))
    assert Ensemble({"log_weights": x}).names == ("log_weights",)


def test_13_ensemble_calls():
    ens = Ensemble(x=jnp.zeros((4, 2)), y=jnp.zeros((4, 1)))
    for call in (
        lambda: ens["z"],
        lambda: ens.mean("z"),
        lambda: ens.marginal("z"),
        lambda: ens.cov("x", "z"),
        lambda: ens.rename(z="w"),
    ):
        with pytest.raises(KeyError, match="'z'"):
            call()
    with pytest.raises(ValueError, match="more than once"):
        ens.marginal("x", "x")
    with pytest.raises(ValueError, match="remain"):
        ens.drop("x", "y")
    with pytest.raises(ValueError, match="collides"):
        ens.rename(x="y")
    with pytest.raises(ValueError, match="not distinct"):
        ens.rename(x="a", y="a")
    with pytest.raises(ValueError, match=r"\(4, d\)"):
        ens.assign(z=jnp.zeros((3, 2)))
    assert ens.assign(y=jnp.ones((4, 3)), z=jnp.ones((4, 1))).dims == {
        "x": 2,
        "y": 3,
        "z": 1,
    }


def test_13_gaussian_construction():
    m = {"a": jnp.zeros(2), "b": jnp.zeros(3)}
    D = {"a": DensePSD(jnp.eye(2)), "b": DensePSD(jnp.eye(3))}
    with pytest.raises(ValueError, match="at least one block"):
        Gaussian({})
    with pytest.raises(ValueError, match="rank"):
        Gaussian({"a": jnp.zeros((2, 1))}, block_covs={"a": DensePSD(jnp.eye(2))})
    with pytest.raises(TypeError, match="floating"):
        Gaussian({"a": jnp.zeros(2, int)}, block_covs={"a": DensePSD(jnp.eye(2))})
    with pytest.raises(TypeError, match="dtype"):
        Gaussian({"a": jnp.zeros(2), "b": jnp.zeros(3, jnp.float32)}, block_covs=D)
    with pytest.raises(ValueError, match="neither"):
        Gaussian(m, block_covs={"a": D["a"]})
    with pytest.raises(ValueError, match="not a block"):
        Gaussian(m, block_covs={**D, "c": D["a"]})
    with pytest.raises(TypeError, match="PSDLinOp"):
        Gaussian(m, block_covs={"a": jnp.eye(2), "b": D["b"]})
    with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
        Gaussian(m, block_covs={"a": D["a"], "b": D["a"]})
    with pytest.raises(ValueError, match="rows"):
        Gaussian(m, factors={"a": jnp.ones((3, 2))}, block_covs=D)
    with pytest.raises(ValueError, match="latent width"):
        Gaussian(m, factors={"a": jnp.ones((2, 2)), "b": jnp.ones((3, 1))})
    with pytest.raises(ValueError, match="disagrees"):
        Gaussian(m, factors={"a": jnp.ones((2, 2))}, block_covs=D, latent_dim=3)
    with pytest.raises(TypeError, match="latent_dim"):
        Gaussian(m, block_covs=D, latent_dim=True)
    with pytest.raises(TypeError, match="LinOp or"):
        Gaussian(m, factors={"a": "row"}, block_covs=D)
    with pytest.raises(TypeError):
        Gaussian(m, D)  # factors and block_covs are keyword-only
    with pytest.raises(TypeError, match="pair"):
        Gaussian.independent(a=jnp.zeros(2))
    assert Gaussian(m, block_covs=D, latent_dim=7).latent_dim == 7
    family = _stacked(DensePSD(jnp.eye(2)))
    with pytest.raises(ValueError, match="vmapped family"):
        Gaussian({"a": jnp.zeros(2)}, block_covs={"a": family})


def test_13_ensemble_gaussian_construction():
    F = jnp.asarray(RNG.normal(size=(2, 4)))
    with pytest.raises(TypeError, match="n_particles"):
        EnsembleGaussian({"a": jnp.zeros(2)}, factors={"a": F}, n_particles=4.0)
    with pytest.raises(ValueError, match="at least 2"):
        EnsembleGaussian(
            {"a": jnp.zeros(2)}, block_covs={"a": DensePSD(jnp.eye(2))}, n_particles=1
        )
    with pytest.raises(ValueError, match="n_particles=5"):
        EnsembleGaussian({"a": jnp.zeros(2)}, factors={"a": F}, n_particles=5)
    EnsembleGaussian({"a": jnp.zeros(2)}, factors={"a": F}, n_particles=4)  # not debug
    with debug_checks(True):
        with pytest.raises(ValueError, match="not centered"):
            EnsembleGaussian({"a": jnp.zeros(2)}, factors={"a": F}, n_particles=4)
        centered = F - F.mean(axis=1, keepdims=True)
        EnsembleGaussian({"a": jnp.zeros(2)}, factors={"a": centered}, n_particles=4)


def test_13_gaussian_calls():
    g, _ = _problem({"x": 2, "g": 3, "c": 2}, 4, terms=("g",))
    noisy, _ = _problem({"x": 2, "g": 3}, 4, terms=("x", "g"))
    for call in (
        lambda: g.condition(z=jnp.zeros(1)),
        lambda: g.log_density(z=jnp.zeros(1)),
        lambda: g.conditional_map("z"),
        lambda: g.marginal("z"),
        lambda: g.mean("z"),
    ):
        with pytest.raises(KeyError, match="'z'"):
            call()
    with pytest.raises(TypeError, match="twice"):
        g.condition({"g": jnp.zeros(3)}, g=jnp.zeros(3))
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        g.condition(g=jnp.zeros((1, 3)))
    with pytest.raises(ValueError, match="mixed"):
        g.condition(g=jnp.zeros(3), c=jnp.zeros(2))
    with pytest.raises(ValueError, match="mixed"):
        g.log_density(g=jnp.zeros(3), c=jnp.zeros(2))
    with pytest.raises(ValueError, match="remain"):
        noisy.condition(g=jnp.zeros(3), x=jnp.zeros(2))
    with pytest.raises(ValueError, match="at least one block"):
        g.condition()
    with pytest.raises(ValueError, match="independent term"):
        g.conditional_map("c")
    with pytest.raises(ValueError, match="batch shape"):
        noisy.log_density(g=jnp.zeros((2, 3)), x=jnp.zeros((3, 2)))
    with pytest.raises(TypeError, match="Python int"):
        g.sample(jax.random.key(0), 3.0)
    with pytest.raises(ValueError, match="at least 2"):
        g.sample(jax.random.key(0), 1)
    with pytest.raises(TypeError, match="typed key"):
        g.sample(jax.random.PRNGKey(0), 3)
    with pytest.raises(ValueError, match="key is required"):
        g.sample(None, 3)
    with pytest.raises(ValueError, match="no independent term"):
        g.absorb("x")


def test_13_map_and_weight_calls():
    J = 5
    ens, approx, *_ = _ensemble_problem(J, {"x": 2, "g": 3, "h": 1}, noise=("g", "h"))
    cmap = approx.conditional_map("g")
    assert cmap.targets == ("x", "h")
    with pytest.raises(KeyError, match="not a given block"):
        cmap(ens, g=jnp.zeros(3), x=jnp.zeros(2))
    with pytest.raises(ValueError, match="missing"):
        cmap(ens)
    with pytest.raises(ValueError, match="must contain"):
        cmap(ens.drop("h"), g=jnp.zeros(3))
    with pytest.raises(TypeError, match="Ensemble"):
        cmap(ens["x"], g=jnp.zeros(3))
    with pytest.raises(ValueError, match="key is required"):
        cmap.particle_coefficients(g=jnp.zeros(3), key=None)
    plain = approx.absorb("h").conditional_map("g")
    with pytest.raises(ValueError, match="plain Gaussian"):
        plain.particle_coefficients(g=jnp.zeros(3), key=jax.random.key(0))
    smap = approx.square_root_map("g")
    with pytest.raises(ValueError, match="key is required"):
        smap(g=jnp.zeros(3))
    with pytest.raises(ValueError, match="key is required"):
        approx.realize_particles()
    with pytest.raises(ValueError, match="no independent term"):
        approx.realize_particles(key=jax.random.key(0), exclude_block_covs=("x",))
    with pytest.raises(ValueError, match="scheme"):
        resample(jax.random.key(0), ens, scheme="stratified")
    with pytest.raises(TypeError, match="Python int"):
        resample(jax.random.key(0), ens, np.int64(3))


# --- 14. capabilities --------------------------------------------------------


def test_14_missing_capabilities_raise_before_any_work():
    counting = CountingWhitenPSD.counting(_psd(2))
    no_whiten = PSDLowRank(jnp.asarray(RNG.normal(size=(3, 3))))
    g = Gaussian(
        {"x": jnp.zeros(2), "a": jnp.zeros(2), "b": jnp.zeros(3)},
        factors={"x": jnp.ones((2, 2)), "a": jnp.ones((2, 2)), "b": jnp.ones((3, 2))},
        block_covs={"a": counting, "b": no_whiten},
    )
    for call in (
        lambda: g.condition(a=jnp.zeros(2), b=jnp.zeros(3)),
        lambda: g.conditional_map(("a", "b")),
        lambda: g.log_density(a=jnp.zeros(2), b=jnp.zeros(3)),
    ):
        with pytest.raises(UnsupportedOpError, match="whiten"):
            call()
    assert counting.count == 0
    whiten_only = WhitenOnlyPSD(jnp.linalg.cholesky(jnp.asarray(_psd(2))))
    h = Gaussian(
        {"y": jnp.zeros(2)},
        factors={"y": jnp.ones((2, 1))},
        block_covs={"y": whiten_only},
    )
    with pytest.raises(UnsupportedOpError, match="logdet"):
        h.log_density(y=jnp.zeros(2))
    for call in (
        lambda: h.sample(jax.random.key(0), 3),
        lambda: h.absorb("y"),
        lambda: h.add_noise(y=DensePSD(jnp.eye(2))),
        lambda: exact_moment_ensemble(jax.random.key(0), h, 5),
    ):
        with pytest.raises(UnsupportedOpError, match="factor"):
            call()
    ens = Ensemble(
        x=jnp.asarray(RNG.normal(size=(4, 2))), g=jnp.asarray(RNG.normal(size=(4, 2)))
    )
    aligned = ens.project().add_noise(x=whiten_only, g=DensePSD(jnp.eye(2)))
    with pytest.raises(UnsupportedOpError, match="factor"):
        aligned.square_root_map("g")
    with pytest.raises(UnsupportedOpError, match="factor"):
        aligned.realize_particles(key=jax.random.key(0))


# --- 15. counts --------------------------------------------------------------


def test_15_whitened_vectors_per_operation():
    J, N, n = 6, 3, 4
    X = RNG.normal(size=(J, 2))
    G = RNG.normal(size=(J, N))
    noise = CountingWhitenPSD.counting(_psd(N))
    ens = Ensemble(x=jnp.asarray(X), g=jnp.asarray(G))
    approx = ens.project().add_noise(g=noise)
    y = jnp.asarray(RNG.normal(size=N))
    samples = Ensemble(
        x=jnp.asarray(RNG.normal(size=(n, 2))), g=jnp.asarray(RNG.normal(size=(n, N)))
    )

    def count(call):
        noise.reset()
        call()
        return noise.count

    assert count(lambda: approx.condition(g=y)) == J + 1
    assert count(lambda: approx.log_density(g=jnp.zeros((n, N)))) == J + n
    assert count(lambda: approx.cov("g")) == J
    assert count(lambda: approx.conditional_map("g")) == J
    cmap = approx.conditional_map("g")
    assert count(lambda: cmap(samples, g=y)) == n
    assert count(lambda: cmap.particle_coefficients(g=y, key=jax.random.key(0))) == 1
    assert count(lambda: approx.square_root_map("g")) == J
    smap = approx.square_root_map("g")
    assert count(lambda: smap(g=y)) == 1
    prior = Gaussian.independent(g=(jnp.zeros(N), noise))
    assert count(lambda: prior.log_density(g=jnp.zeros((n, N)))) == n
    # J + 1 through the particle route, 2J calling the map on the particles
    assert (
        count(
            lambda: approx.conditional_map("g").particle_coefficients(
                g=y, key=jax.random.key(0)
            )
        )
        == J + 1
    )
    assert (
        count(lambda: approx.conditional_map("g")(ens, g=y, key=jax.random.key(0)))
        == 2 * J
    )


def test_15_one_svd_per_build_or_call():
    J = 5
    ens, approx, *_ = _ensemble_problem(J, {"x": 2, "g": 3})
    y = jnp.zeros(3)
    for name, fn in (
        ("condition", lambda a: a.condition(g=y).mean("x")),
        ("condition-factor", lambda a: a.condition(g=y).factor("x").to_dense()),
        ("log_density", lambda a: a.log_density(g=y)),
        ("conditional_map", lambda a: a.conditional_map("g")(ens, g=y)["x"]),
        (
            "particle_coefficients",
            lambda a: a.conditional_map("g").particle_coefficients(
                g=y, key=jax.random.key(0)
            ),
        ),
        ("square_root_map", lambda a: a.square_root_map("g")(g=y)["x"]),
    ):
        count = _count_svd(jax.make_jaxpr(fn)(approx).jaxpr)
        assert count == 1, f"{name} computed {count} SVDs"


# --- 16. diagnostics ---------------------------------------------------------


def _singular_problem():
    """An aligned Gaussian whose given block's term has a zero variance."""
    J = 5
    ens = Ensemble(
        x=jnp.asarray(RNG.normal(size=(J, 2))), g=jnp.asarray(RNG.normal(size=(J, 3)))
    )
    singular = PSDDiagonal(jnp.asarray([1.0, 0.0, 2.0]))  # built with debug checks off
    return ens, ens.project().add_noise(g=singular)


def _conditioning_paths(ens, approx, y):
    cmap, smap = approx.conditional_map("g"), approx.square_root_map("g")
    key = jax.random.key(0)
    return {
        "condition": lambda: approx.condition(g=y),
        "log_density": lambda: approx.log_density(g=y),
        "conditional_map": lambda: approx.conditional_map("g"),
        "map call": lambda: cmap(ens, g=y),
        "coefficients": lambda: cmap.coefficients(ens, g=y),
        "particle_coefficients": lambda: cmap.particle_coefficients(g=y, key=key),
        "square_root_map": lambda: approx.square_root_map("g"),
        "square-root call": lambda: smap(g=y),
    }


def test_16_the_whitening_check_names_the_block_on_every_path():
    ens, approx = _singular_problem()
    y = jnp.ones(3)
    paths = _conditioning_paths(ens, approx, y)  # maps built outside debug mode
    with debug_checks(True):
        for name, call in paths.items():
            with pytest.raises(ValueError, match="given block 'g' is not finite") as info:
                call()
            assert "singular" in str(info.value), name
    # outside debug mode, nan without raising
    assert bool(jnp.isnan(approx.condition(g=y).mean("x")).all())
    assert bool(jnp.isnan(approx.square_root_map("g")(g=y)["x"]).all())


def test_16_the_result_check_fires_on_an_overflowing_target_row():
    J = 6
    G = RNG.normal(size=(J, 2))
    X = RNG.normal(size=(J, 2))
    X = (X - X.mean(axis=0)) * 1e306
    ens = Ensemble(x=jnp.asarray(X), g=jnp.asarray(G))
    approx = ens.project().add_noise(g=PSDDiagonal(jnp.ones(2)))
    y = jnp.asarray(G.mean(axis=0) + 1e3)
    cmap, smap = approx.conditional_map("g"), approx.square_root_map("g")
    with debug_checks(True):
        for call in (
            lambda: approx.condition(g=y),
            lambda: cmap(ens, g=y),
            lambda: smap(g=y),
            lambda: ens.project().condition(g=y),  # exact values
        ):
            with pytest.raises(ValueError, match="must be finite. The whitened inputs"):
                call()
        # log_density: whitened values of order 1e200, whose square overflows
        with pytest.raises(ValueError, match="log density must be finite"):
            approx.log_density(g=jnp.full(2, 1e200))


def test_16_neither_check_fires_under_a_trace():
    ens, approx = _singular_problem()
    y = jnp.ones(3)
    with debug_checks(True):
        out = jax.jit(lambda a, y: a.condition(g=y).mean("x"))(approx, y)
        assert bool(jnp.isnan(out).all())
        out = jax.vmap(lambda y: approx.condition(g=y).mean("x"))(jnp.stack([y, y]))
        assert bool(jnp.isnan(out).all())
        out = jax.jit(lambda: approx.square_root_map("g")(g=y)["x"])()  # closed over
        assert bool(jnp.isnan(out).all())
        jax.jit(lambda: approx.log_density(g=y))()
        jax.jit(lambda: approx.conditional_map("g")(ens, g=y)["x"])()


# --- 17. accuracy ------------------------------------------------------------


def _fr(A):
    return [[Fraction(float(x)) for x in row] for row in np.atleast_2d(A)]


def _fr_solve(M, B):
    """Solve ``M X = B`` exactly over ``Fraction`` by Gaussian elimination."""
    n = len(M)
    M, B = [r[:] for r in M], [r[:] for r in B]
    for col in range(n):
        p = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[p], B[col], B[p] = M[p], M[col], B[p], B[col]
        for r in range(col + 1, n):
            f = M[r][col] / M[col][col]
            M[r] = [a - f * b for a, b in zip(M[r], M[col], strict=True)]
            B[r] = [a - f * b for a, b in zip(B[r], B[col], strict=True)]
    X = [[Fraction(0)] * len(B[0]) for _ in range(n)]
    for r in reversed(range(n)):
        for j in range(len(B[0])):
            X[r][j] = (B[r][j] - sum(M[r][i] * X[i][j] for i in range(r + 1, n))) / M[r][
                r
            ]
    return X


def _fr_det(M):
    n = len(M)
    M = [r[:] for r in M]
    det = Fraction(1)
    for col in range(n):
        p = max(range(col, n), key=lambda r: abs(M[r][col]))
        if p != col:
            M[col], M[p] = M[p], M[col]
            det = -det
        det *= M[col][col]
        for r in range(col + 1, n):
            f = M[r][col] / M[col][col]
            M[r] = [a - f * b for a, b in zip(M[r], M[col], strict=True)]
    return det


def _exact_given(Fc: np.ndarray):
    """``F_c F_c^T + I`` exactly, for identity noise."""
    F = _fr(Fc)
    N, k = len(F), len(F[0])
    return F, [
        [
            sum(F[i][m] * F[j][m] for m in range(k)) + (1 if i == j else 0)
            for j in range(N)
        ]
        for i in range(N)
    ]


def _identity_noise_gaussian(Fc: np.ndarray, *, x_row=None):
    N, k = Fc.shape
    Fx = np.eye(k) if x_row is None else x_row
    return Gaussian(
        {"x": jnp.zeros(Fx.shape[0]), "c": jnp.zeros(N)},
        factors={"x": jnp.asarray(Fx), "c": jnp.asarray(Fc)},
        block_covs={"c": PSDDiagonal(jnp.ones(N))},
    )


def _axis_rows(N, k, sigma):
    """Given rows along the latent coordinate axes, singular values (sigma, 1, ...)."""
    Fc = np.zeros((N, k))
    Fc[0, 0] = sigma
    for i in range(1, N):
        Fc[i, i] = 1.0
    return Fc


def _rotated_rows(N, k, sigma, rng):
    """Given rows with singular values (sigma, 1, ...), in random directions."""
    Q1, _ = np.linalg.qr(rng.normal(size=(N, N)))
    Q2, _ = np.linalg.qr(rng.normal(size=(k, k)))
    return Q1 @ np.diag([sigma] + [1.0] * (N - 1)) @ Q2[:, :N].T


def _exact_conditional(Fc, y):
    """The exact conditional mean and covariance of ``x`` (row ``I_k``) given
    ``c = y`` under identity noise, over ``Fraction``."""
    N, k = Fc.shape
    F, M = _exact_given(Fc)
    z = _fr_solve(M, [[Fraction(float(v))] for v in y])
    mean = np.array([float(sum(F[i][m] * z[i][0] for i in range(N))) for m in range(k)])
    Z = _fr_solve(M, F)
    cov = np.array(
        [
            [
                float((1 if a == b else 0) - sum(F[i][a] * Z[i][b] for i in range(N)))
                for b in range(k)
            ]
            for a in range(k)
        ]
    )
    return mean, cov


def _exact_log_density(Fc, v):
    """The exact log density of ``c`` at ``v``, to 50 digits."""
    N = Fc.shape[0]
    _, M = _exact_given(Fc)
    q = _fr_solve(M, [[Fraction(float(x))] for x in v])
    quad = sum(Fraction(float(v[i])) * q[i][0] for i in range(N))
    det = _fr_det(M)
    getcontext().prec = 50
    to_dec = lambda f: Decimal(f.numerator) / Decimal(f.denominator)  # noqa: E731
    logdet = to_dec(det).ln()
    return float(-(to_dec(quad) + logdet + N * Decimal(2 * math.pi).ln()) / 2)


def _relative(got, want):
    return np.abs(np.asarray(got) - want).max() / np.abs(want).max()


def test_17_axis_aligned_rows_are_exact_to_round_off_at_any_sigma_max():
    """Where the decomposition of ``S`` is exact, as for given rows along
    latent coordinate axes, the mean is accurate to a few eps at every
    ``sigma_max``, and the covariance and log density are too when
    ``rho = k`` (for the density, ``N <= k``)."""
    rng = np.random.default_rng(3)
    N = 3
    y = rng.normal(size=N)
    for sigma in (1e5, 1e10, 1e15, 1e20):
        Fc = _axis_rows(N, 5, sigma)
        mean, _ = _exact_conditional(Fc, y)
        got = _identity_noise_gaussian(Fc).condition(c=jnp.asarray(y)).mean("x")
        assert _relative(got, mean) <= 16 * EPS, sigma
    for sigma in (1e8, 1e12):
        Fc = _axis_rows(N, N, sigma)
        g = _identity_noise_gaussian(Fc)
        _, cov = _exact_conditional(Fc, y)
        assert _relative(_cov_dense(g.condition(c=jnp.asarray(y)), "x"), cov) <= 16 * EPS
        for k in (N, 5):
            v = _axis_rows(N, k, sigma) @ rng.normal(size=k) + rng.normal(size=N)
            got = float(
                _identity_noise_gaussian(_axis_rows(N, k, sigma)).log_density(
                    c=jnp.asarray(v)
                )
            )
            exact = _exact_log_density(_axis_rows(N, k, sigma), v)
            assert abs(got - exact) <= 16 * EPS * abs(exact), (sigma, k)


def test_17_rotated_rows_lose_no_more_than_the_conditional_itself_moves():
    """In general the computation is backward stable, not exact: its error is
    that of the exact conditional for given rows perturbed by about eps
    relative to ``sigma_max``, and the conditional itself moves by about
    ``eps * sigma_max`` under such a perturbation. Measured here as a ratio
    against that sensitivity, at ``rho = k`` and ``rho < k``."""
    N = 3
    for k in (3, 5):
        for sigma in (1e5, 1e8, 1e12):
            rng = np.random.default_rng(int(np.log10(sigma)) + 10 * k)
            Fc = _rotated_rows(N, k, sigma, rng)
            y = rng.normal(size=N)
            mean, cov = _exact_conditional(Fc, y)
            post = _identity_noise_gaussian(Fc).condition(c=jnp.asarray(y))
            moved = 0.0
            for _ in range(3):
                nearby = Fc + rng.normal(size=Fc.shape) * EPS * sigma
                m2, _ = _exact_conditional(nearby, y)
                moved = max(moved, _relative(m2, mean))
            assert moved > 10 * EPS * sigma / 100, "the problem must be sensitive here"
            assert _relative(post.mean("x"), mean) <= 10 * moved, (k, sigma)
            assert _relative(_cov_dense(post, "x"), cov) <= 10 * moved, (k, sigma)


def test_17_the_density_route_beats_the_cancelling_form():
    """``||b||^2 - <S b, A^-1 S b>`` cancels catastrophically; the route
    through ``LowRankUpdate.whiten`` does not."""
    rng = np.random.default_rng(6)
    N, k, sigma = 1, 3, 1e8
    Fc = np.zeros((N, k))
    Fc[0, :] = [sigma, 0.0, 0.0]
    v = Fc @ rng.normal(size=k) + rng.normal(size=N)
    exact = _exact_log_density(Fc, v)
    got = float(_identity_noise_gaussian(Fc).log_density(c=jnp.asarray(v)))
    S, b = Fc.T, v  # identity noise: W = I
    A = np.eye(k) + S @ S.T
    quad = b @ b - (S @ b) @ np.linalg.solve(A, S @ b)
    cancelling = -0.5 * (quad + np.log(np.linalg.det(A)) + N * np.log(2 * np.pi))
    assert abs(got - exact) <= 16 * EPS * abs(exact)
    assert abs(cancelling - exact) > 1e6 * abs(got - exact) + 1e-3


def test_17_the_thin_basis_loss_is_pinned_until_issue_54():
    """When ``rho < k`` the conditional variance in a constrained direction
    carries a relative error of about ``eps * sigma_max``: pinned here at its
    current values, on the axis-aligned case and on one seeded random
    direction (three given coordinates, ``k = 5``). Issue #54 tracks the fix;
    when it lands these regressions change by design."""
    N, k = 3, 5

    def rel_error(Fc, v):
        F, M = _exact_given(Fc)
        Fv = [[sum(F[i][m] * Fraction(float(v[m])) for m in range(k))] for i in range(N)]
        z = _fr_solve(M, Fv)
        exact = float(
            sum(Fraction(float(x)) ** 2 for x in v)
            - sum(Fv[i][0] * z[i][0] for i in range(N))
        )
        row = np.asarray(
            _identity_noise_gaussian(Fc).condition(c=jnp.zeros(N)).factor("x").to_dense()
        )
        return (float(v @ row @ row.T @ v) - exact) / exact

    def axis_case(sigma):
        Fc = np.zeros((N, k))
        Fc[0, 0], Fc[1, 1], Fc[2, 2] = sigma, 1.0, 1.0
        return rel_error(Fc, np.eye(k)[0])

    def random_case(sigma):
        rng = np.random.default_rng(5)
        Q1, _ = np.linalg.qr(rng.normal(size=(N, N)))
        Q2, _ = np.linalg.qr(rng.normal(size=(k, k)))
        return rel_error(Q1 @ np.diag([sigma, 1.0, 1.0]) @ Q2[:, :N].T, Q2[:, 0])

    # measured on main at this PR: -1.2e-8, then exactly -1 (the variance is 0)
    assert 3e-9 < abs(axis_case(1e8)) < 5e-8
    assert axis_case(1e17) == -1.0
    # measured: 8.3e-3 at sigma_max = 1e13, against 5e-11 at 1e5
    assert 2e-3 < abs(random_case(1e13)) < 3e-2
    assert abs(random_case(1e5)) < 1e-9


# --- 18. derivatives ---------------------------------------------------------


def _fd_check(f, x, *, rtol=1e-6):
    """Forward and reverse derivatives of a scalar ``f`` against central
    differences along a random direction, eagerly and under ``jit``."""
    d = jnp.asarray(RNG.normal(size=x.shape))
    h = 1e-6
    fd = (f(x + h * d) - f(x - h * d)) / (2 * h)
    for g in (f, jax.jit(f)):
        _, fwd = jax.jvp(g, (x,), (d,))
        rev = jnp.vdot(jax.grad(g)(x), d)
        for value in (fwd, rev):
            assert bool(jnp.isfinite(value))
            assert abs(float(value) - float(fd)) <= rtol * max(1.0, abs(float(fd)))


def _degenerate_given_rows():
    """Given rows at the three degenerate spectra the contract names."""
    k = 4
    repeated = np.zeros((3, k))
    repeated[0, 0] = repeated[1, 1] = repeated[2, 2] = 2.0  # sigma = 2, 2, 2
    padded = np.concatenate([RNG.normal(size=(3, 2)), np.zeros((3, 2))], axis=1)
    collapsed = np.zeros((3, k))
    return {"repeated": repeated, "zero-padded": padded, "collapsed": collapsed}


@pytest.mark.parametrize("case", ["repeated", "zero-padded", "collapsed"])
def test_18_first_derivatives_are_finite_and_correct_at_degenerate_spectra(case):
    Fc = jnp.asarray(_degenerate_given_rows()[case])
    Fx = jnp.asarray(RNG.normal(size=(2, 4)))
    y = jnp.asarray(RNG.normal(size=3))
    weights = jnp.asarray(RNG.normal(size=(5, 2)))
    samples = Ensemble(
        x=jnp.asarray(RNG.normal(size=(5, 2))), c=jnp.asarray(RNG.normal(size=(5, 3)))
    )
    key = jax.random.key(1)

    def gaussian(F):
        return Gaussian(
            {"x": jnp.zeros(2), "c": jnp.zeros(3)},
            factors={"x": Fx, "c": F},
            block_covs={"c": PSDDiagonal(jnp.asarray([1.0, 2.0, 0.5]))},
        )

    def condition(F):
        post = gaussian(F).condition(c=y)
        return jnp.sum(post.mean("x") * weights[0]) + jnp.sum(
            post.factor("x").to_dense() * weights[:2, :1]
        )

    _fd_check(condition, Fc)
    _fd_check(lambda F: gaussian(F).log_density(c=y), Fc)
    _fd_check(
        lambda F: jnp.sum(
            gaussian(F).conditional_map("c")(samples, c=y, key=key)["x"] * weights
        ),
        Fc,
    )
    # under vmap, over two points of the same degenerate kind
    grads = jax.vmap(jax.grad(condition))(jnp.stack([Fc, 2 * Fc]))
    assert bool(jnp.all(jnp.isfinite(grads)))


def test_18_derivatives_through_a_collapsed_ensemble():
    """The square-root map and the particle route, differentiated with
    respect to particles at an exactly collapsed given block."""
    J = 5
    X = jnp.asarray(RNG.normal(size=(J, 2)))
    G = jnp.full((J, 3), 0.5)
    y = jnp.asarray(RNG.normal(size=3))
    weights = jnp.asarray(RNG.normal(size=(J, 2)))

    def approx(G):
        return Ensemble(x=X, g=G).project().add_noise(g=PSDDiagonal(jnp.ones(3)))

    _fd_check(lambda G: jnp.sum(approx(G).square_root_map("g")(g=y)["x"] * weights), G)
    _fd_check(
        lambda G: jnp.sum(
            approx(G)
            .conditional_map("g")
            .particle_coefficients(g=y, key=jax.random.key(0))[:, :2]
            * weights
        ),
        G,
    )
    _fd_check(lambda G: approx(G).log_density(g=y), G)


# --- 19. JAX -----------------------------------------------------------------


def _all_objects():
    J = 5
    ens, approx, *_ = _ensemble_problem(J, {"x": 2, "g": 3})
    return {
        "Ensemble": ens,
        "weighted Ensemble": reweight(ens, jnp.zeros(J)),
        "Gaussian": approx.absorb("g"),
        "EnsembleGaussian": approx,
        "MatheronMap": approx.conditional_map("g"),
        "SquareRootMap": approx.square_root_map("g"),
    }


def test_19_flatten_and_unflatten_preserve_type_and_behavior():
    for name, obj in _all_objects().items():
        leaves, treedef = jax.tree_util.tree_flatten(obj)
        back = jax.tree_util.tree_unflatten(treedef, leaves)
        assert type(back) is type(obj), name
        assert repr(back) == repr(obj), name
        # sentinel leaves, as jax.custom_vjp reconstructs with
        sentinel = jax.tree_util.tree_unflatten(treedef, [object()] * len(leaves))
        assert type(sentinel) is type(obj)
    g = _all_objects()["EnsembleGaussian"]
    assert g._block_covs[0] is None  # a None entry is structure, not a leaf
    back = jax.tree_util.tree_unflatten(*reversed(jax.tree_util.tree_flatten(g)))
    _close(
        back.condition(g=jnp.zeros(3)).mean("x"), g.condition(g=jnp.zeros(3)).mean("x")
    )


def test_19_conditioning_and_maps_run_under_jit():
    objs = _all_objects()
    approx, ens = objs["EnsembleGaussian"], objs["Ensemble"]
    y = jnp.asarray(RNG.normal(size=3))
    key = jax.random.key(0)
    for fn in (
        lambda a, y: a.condition(g=y).mean("x"),
        lambda a, y: a.log_density(g=y),
        lambda a, y: a.conditional_map("g")(ens, g=y, key=key)["x"],
        lambda a, y: a.conditional_map("g").particle_coefficients(g=y, key=key),
        lambda a, y: a.square_root_map("g")(g=y)["x"],
        lambda a, y: a.realize_particles(key=key)["g"],
    ):
        _close(jax.jit(fn)(approx, y), fn(approx, y), factor=1e4)
    m = objs["MatheronMap"]
    _close(jax.jit(lambda m, y: m(ens, g=y)["x"])(m, y), m(ens, g=y)["x"], factor=1e4)


def test_19_a_vmapped_family_is_legible_and_inert():
    for name, obj in _all_objects().items():
        family = _stacked(obj)
        assert family.batch_shape == (3,), name
        assert repr(family) == f"vmapped({obj!r}, batch=(3,))"
        assert (getattr(family, "names", None) or family.given) == (
            getattr(obj, "names", None) or obj.given
        )
    family = _stacked(_all_objects()["EnsembleGaussian"])
    assert (
        family.dims == {"x": 2, "g": 3} and family.n_particles == 5 == family.latent_dim
    )
    for call in (
        lambda: family.condition(g=jnp.zeros(3)),
        lambda: family.mean("x"),
        lambda: family.cov("x"),
        lambda: family.sample(jax.random.key(0), 3),
        lambda: family.realize_particles(),
        lambda: family.marginal("x"),
    ):
        with pytest.raises(ValueError, match=r"vmapped family with batch shape \(3,\)"):
            call()
    ens = _stacked(_all_objects()["Ensemble"])
    assert ens.n_particles == 5 and ens.dims == {"x": 2, "g": 3} and not ens.is_weighted
    for call in (
        lambda: ens.weights,
        lambda: ens.all_finite,
        lambda: ens.project(),
        lambda: ens["x"],
        lambda: effective_sample_size(ens),
    ):
        with pytest.raises(ValueError, match="vmapped family"):
            call()
    cmap = _stacked(_all_objects()["MatheronMap"])
    with pytest.raises(ValueError, match="vmapped family"):
        cmap(_all_objects()["Ensemble"], g=jnp.zeros(3))
    smap = _stacked(_all_objects()["SquareRootMap"])
    with pytest.raises(ValueError, match="vmapped family"):
        smap(g=jnp.zeros(3))


def test_19_construction_inside_vmap_and_agreement_with_a_loop():
    means = jnp.asarray(RNG.normal(size=(4, 3)))
    R = DensePSD(jnp.asarray(_psd(3)))
    F = jnp.asarray(RNG.normal(size=(3, 2)))
    family = jax.vmap(
        lambda m: Gaussian(
            {"x": m, "y": m}, factors={"x": F, "y": F}, block_covs={"y": R}
        )
    )(means)
    assert type(family) is Gaussian and family.batch_shape == (4,)
    y = jnp.asarray(RNG.normal(size=3))
    mapped = jax.vmap(lambda g: g.condition(y=y).mean("x"))(family)
    for i in range(4):
        one = Gaussian(
            {"x": means[i], "y": means[i]}, factors={"x": F, "y": F}, block_covs={"y": R}
        )
        _close(mapped[i], one.condition(y=y).mean("x"))
    J = 5
    particles = jnp.asarray(RNG.normal(size=(2, J, J)))  # a leading axis equal to J
    ens = jax.vmap(lambda x: Ensemble(x=x))(particles)
    assert ens.batch_shape == (2,)


def test_19_a_gradient_through_every_conditioning_operation_computes_one_svd():
    J = 5
    ens, approx, *_ = _ensemble_problem(J, {"x": 2, "g": 3})
    G = ens["g"]
    y = jnp.zeros(3)
    R = approx.block_cov("g")

    def approx_of(G):
        return ens.assign(g=G).project().add_noise(g=R)

    for name, loss in (
        (
            "condition",
            lambda G: jnp.sum(approx_of(G).condition(g=y).factor("x").to_dense()),
        ),
        ("log_density", lambda G: approx_of(G).log_density(g=y)),
        ("matheron", lambda G: jnp.sum(approx_of(G).conditional_map("g")(ens, g=y)["x"])),
        ("square root", lambda G: jnp.sum(approx_of(G).square_root_map("g")(g=y)["x"])),
    ):
        count = _count_svd(jax.make_jaxpr(jax.grad(loss))(G).jaxpr)
        assert count == 1, f"{name}: {count} SVDs"


# --- 20. reproducibility -----------------------------------------------------


def test_20_same_key_same_output_different_keys_differ():
    _, approx, *_ = _ensemble_problem(5, {"x": 2, "g": 3})
    for fn in (
        lambda k: approx.sample(k, 4)["x"],
        lambda k: approx.realize_particles(key=k)["g"],
        lambda k: approx.conditional_map("g").particle_coefficients(
            g=jnp.zeros(3), key=k
        ),
        lambda k: exact_moment_ensemble(k, approx, 10)["x"],
    ):
        a, b, other = fn(jax.random.key(1)), fn(jax.random.key(1)), fn(jax.random.key(2))
        np.testing.assert_array_equal(a, b)
        assert np.abs(np.asarray(a) - np.asarray(other)).max() > 1e-3


def test_20_the_pinned_draws_are_snapshotted():
    """Every draw is pinned by the contract, and the bit stream belongs to JAX.

    Captured under JAX 0.10.2 with x64 enabled and default PRNG settings. A
    failure here is not necessarily an EnsKit bug -- check JAX's version and
    the jax_threefry_partitionable / x64 flags first -- but it does mean every
    stochastic output of this layer changed. The definitions built from these
    primitives are checked elementwise in obligations 9 and 11.
    """
    key = jax.random.key(0)
    keys = jax.random.split(key, 2)
    # sample: normal(split(key, 1 + n_D)[i], ...); realize_particles and
    # SquareRootMap: normal(split(key, n_R)[i], ...)
    np.testing.assert_array_equal(
        jax.random.normal(keys[0], (2, 2)),
        [
            [1.8800298928929466, -0.4812149737008219],
            [0.4154572296392434, 2.381840078208049],
        ],
    )
    np.testing.assert_array_equal(
        jax.random.normal(keys[1], (2, 2)),
        [
            [-1.400884102827668, 1.4321449977776874],
            [0.6248106961344603, 0.20048727190797802],
        ],
    )
    # the maps and exact_moment_ensemble: normal(key, (n, N)) unsplit
    np.testing.assert_array_equal(
        jax.random.normal(key, (2, 3), jnp.float32),
        np.asarray(
            [
                [1.622642159461975, 2.0252647399902344, -0.4335944354534149],
                [-0.07861734926700592, 0.17609089612960815, -0.9720892310142517],
            ],
            np.float32,
        ),
    )
    # resample: uniform(key, (), dtype) and categorical(key, log_weights, shape=(n,))
    assert float(jax.random.uniform(key, (), jnp.float64)) == 0.41845711171638644
    np.testing.assert_array_equal(
        jax.random.categorical(key, jnp.asarray([0.0, -1.0, -2.0, 0.5]), shape=(6,)),
        [2, 3, 0, 1, 3, 3],
    )


# --- 21. dtypes --------------------------------------------------------------


def test_21_no_operation_promotes_float32():
    f32 = jnp.float32
    J = 5
    ens = Ensemble(
        x=jnp.asarray(RNG.normal(size=(J, 2)), f32),
        g=jnp.asarray(RNG.normal(size=(J, 3)), f32),
    )
    R = PSDDiagonal(jnp.ones(3, f32))
    approx = ens.project().add_noise(g=R)
    y = jnp.zeros(3, f32)
    key = jax.random.key(0)
    results = [
        approx.mean("x"),
        approx.condition(g=y).mean("x"),
        approx.condition(g=y).factor("x").to_dense(),
        approx.square_root_map("g")(g=y)["x"],
        approx.conditional_map("g")(ens, g=y, key=key)["x"],
        approx.conditional_map("g").particle_coefficients(g=y, key=key),
        approx.realize_particles(key=key)["g"],
        approx.sample(key, 3)["g"],
        approx.absorb("g").factor("x").to_dense(),
        approx.absorb("g").sample(key, 3)["x"],
        exact_moment_ensemble(key, approx, 10)["g"],
        approx.log_density(g=y),
        ens.project().condition(g=y).mean("x"),
        reweight(ens, jnp.zeros(J)).project().factor("x").to_dense(),
        resample(key, reweight(ens, jnp.zeros(J)))["x"],
        effective_sample_size(ens),
    ]
    for i, out in enumerate(results):
        assert out.dtype == f32, (i, out.dtype)
    assert approx.absorb("g").factor("x").ops[-1].dtype == f32  # the Zero padding
    carry = jax.lax.scan(
        lambda e, _: (e.assign(x=e["x"] + 1), None), ens, None, length=2
    )[0]
    assert carry["x"].dtype == f32


# --- 22. repr and messages ---------------------------------------------------


def test_22_repr_names_static_sizes_and_never_raises():
    objs = _all_objects()
    assert (
        repr(objs["Ensemble"])
        == "Ensemble(n_particles=5, blocks={'x': 2, 'g': 3}, weighted=False)"
    )
    assert repr(objs["weighted Ensemble"]).endswith("weighted=True)")
    assert (
        repr(objs["Gaussian"])
        == "Gaussian(blocks={'x': 2, 'g': 3}, latent_dim=8, block_covs=())"
    )
    assert repr(objs["EnsembleGaussian"]) == (
        "EnsembleGaussian(n_particles=5, blocks={'x': 2, 'g': 3}, block_covs=('g',))"
    )
    assert (
        repr(objs["MatheronMap"])
        == "MatheronMap(given=('g',), targets=('x',), latent_dim=5)"
    )
    assert (
        repr(objs["SquareRootMap"])
        == "SquareRootMap(given=('g',), targets=('x',), n_particles=5)"
    )
    for obj in objs.values():
        leaves, treedef = jax.tree_util.tree_flatten(obj)
        broken = jax.tree_util.tree_unflatten(treedef, [object()] * len(leaves))
        assert repr(broken).startswith("<") and "unprintable" in repr(broken)
        assert "Array" not in repr(obj)


def test_22_messages_name_the_object_the_method_the_block_and_the_value():
    g, _ = _problem({"x": 2, "g": 3}, 4, terms=("g",))
    with pytest.raises(ValueError) as info:
        g.condition(g=jnp.zeros(4))
    message = str(info.value)
    assert repr(g) in message and ".condition" in message and "'g'" in message
    assert "(3,)" in message and "(4,)" in message


# ===========================================================================
# Section 2 -- regression tests ported from tests/test_gauss.py
#
# Each keeps the old test's name after ``test_regression_`` and its reasoning,
# in this layer's names, as the contract's table under "Ported regression
# tests" assigns them.
# ===========================================================================


def _exact_posterior_mean(U, V, R, y) -> np.ndarray:
    """The posterior mean of the Gaussian fitted to (U, V), in exact rational
    arithmetic over the stored floats: the empirical moments are formed and
    ``(C_vv + R) x = y - v_bar`` is solved over ``Fraction``. Only for tiny
    problems."""
    J, P, N = U.shape[0], U.shape[1], V.shape[1]
    Uf, Vf, Rf = _fr(U), _fr(V), _fr(R)
    um = [sum(Uf[i][k] for i in range(J)) / J for k in range(P)]
    vm = [sum(Vf[i][k] for i in range(J)) / J for k in range(N)]
    Au = [[Uf[i][k] - um[k] for k in range(P)] for i in range(J)]
    Av = [[Vf[i][k] - vm[k] for k in range(N)] for i in range(J)]
    Cuv = [
        [sum(Au[i][a] * Av[i][b] for i in range(J)) / (J - 1) for b in range(N)]
        for a in range(P)
    ]
    M = [
        [
            sum(Av[i][a] * Av[i][b] for i in range(J)) / (J - 1) + Rf[a][b]
            for b in range(N)
        ]
        for a in range(N)
    ]
    x = _fr_solve(M, [[Fraction(float(y[k])) - vm[k]] for k in range(N)])
    return np.array(
        [float(um[a] + sum(Cuv[a][k] * x[k][0] for k in range(N))) for a in range(P)]
    )


def test_regression_the_square_root_reading_needs_a_centered_factor():
    """The square-root reading of a conditioned factor is valid only for a
    centered one, which is why ``realize_particles`` exists only on
    ``EnsembleGaussian``.

    Applied to an uncentered factor, ``m' + sqrt(k-1) F T e_j`` is a perfectly
    shaped particle set whose mean is displaced by ``sqrt(k-1) F T 1 / k`` and
    whose sample covariance falls short of the conditional's by the rank-one
    term ``(F T 1)(F T 1)^T / k``. Right shape, finite, nothing raised: a value
    check could catch it only in debug mode, and carrying alignment as a type
    makes it unrepresentable.
    """
    J, P, N = 8, 3, 4
    U, V = RNG.normal(size=(J, P)) + 3.0, RNG.normal(size=(J, N))
    R = _psd(N)
    y = jnp.asarray(RNG.normal(size=N))
    # an uncentered factor: raw particles rather than anomalies
    uncentered = Gaussian(
        {"u": jnp.asarray(U.mean(axis=0)), "g": jnp.asarray(V.mean(axis=0))},
        factors={
            "u": jnp.asarray(U.T / np.sqrt(J - 1)),
            "g": jnp.asarray(V.T / np.sqrt(J - 1)),
        },
        block_covs={"g": DensePSD(jnp.asarray(R))},
    )
    assert not hasattr(uncentered, "realize_particles")
    post = uncentered.condition(g=y)
    m_post, C_post = np.asarray(post.mean("u")), _cov_dense(post, "u")
    read = m_post + np.sqrt(J - 1) * np.asarray(post.factor("u").to_dense()).T
    mean, cov = _sample_moments(read)
    assert np.all(np.isfinite(read)), "the failure is a wrong number, not a nan"
    assert np.abs(mean - m_post).max() > 0.1 * np.abs(m_post).max()
    assert np.abs(cov - C_post).max() > 0.1 * np.abs(C_post).max()

    # the aligned reading of the same particles is exact in both moments
    approx = (
        Ensemble(u=jnp.asarray(U), g=jnp.asarray(V))
        .project()
        .add_noise(g=DensePSD(jnp.asarray(R)))
    )
    aligned = approx.condition(g=y)
    mean, cov = _sample_moments(aligned.realize_particles()["u"])
    _close(mean, aligned.mean("u"))
    _close(cov, _cov_dense(aligned, "u"))


def test_regression_independently_chosen_factors_lose_the_cross_covariance():
    """The constructor cannot check that factor rows of two blocks come from
    one factorization of the joint covariance, and the failure is silent.

    Factorizing ``C_uu`` and ``C_gg`` separately gives rows that still define
    a valid joint Gaussian -- any stacked factor does -- with the intended
    marginals and a cross-covariance that is whatever the two factorizations
    happen to imply. Conditioning then returns a finite, plausible
    conditional of a different joint. ``Ensemble.project`` and the maps
    layer's pushforward build coherent rows by their arithmetic, which is why
    they are the documented routes.
    """
    P = N = 4
    F = RNG.normal(size=(P + N, P))
    C = F @ F.T
    Cuu, Cug, Cgg = C[:P, :P], C[:P, P:], C[P:, P:]
    R = _psd(N)
    y = RNG.normal(size=N)
    zeros = {"u": jnp.zeros(P), "g": jnp.zeros(N)}
    coherent = Gaussian(
        zeros,
        factors={"u": jnp.asarray(F[:P]), "g": jnp.asarray(F[P:])},
        block_covs={"g": DensePSD(jnp.asarray(R))},
    )
    separate = Gaussian(
        zeros,
        factors={
            "u": jnp.asarray(np.linalg.cholesky(Cuu)),
            "g": jnp.asarray(np.linalg.cholesky(Cgg)),
        },
        block_covs={"g": DensePSD(jnp.asarray(R))},
    )
    for g in (coherent, separate):
        _close(_cov_dense(g.marginal("u"), "u"), Cuu, scale=np.abs(C).max())
    assert (
        np.abs(np.asarray(separate.cov("u", "g").to_dense()) - Cug).max()
        > 0.1 * np.abs(Cug).max()
    )
    m_ref = Cug @ np.linalg.solve(Cgg + R, y)
    _close(coherent.condition(g=jnp.asarray(y)).mean("u"), m_ref)
    wrong = np.asarray(separate.condition(g=jnp.asarray(y)).mean("u"))
    assert (
        np.all(np.isfinite(wrong))
        and np.abs(wrong - m_ref).max() > 0.1 * np.abs(m_ref).max()
    )


def test_regression_transporting_arbitrary_realizations_costs_one_whitening_each():
    """The two Matheron routes have genuinely different costs, and the cheaper
    one is only available where the samples are the Gaussian's own particles.

    ``particle_coefficients`` reads the particles' whitened residuals off the
    whitened factor, ``W(y - g_j) = W(y - m) - sqrt(J-1) S_j``, for ``J + 1``
    whitened vectors with the build. Calling the map on samples cannot: they
    are arbitrary data, so it whitens one residual per sample on top of the
    ``k`` for ``S``. Both are correct; only a count tells them apart, and for
    a dense whitener the difference is the dominant ``O(J N^2)`` term.
    """
    J, N = 6, 4
    noise = CountingWhitenPSD.counting(_psd(N))
    ens = Ensemble(
        u=jnp.asarray(RNG.normal(size=(J, 3))), g=jnp.asarray(RNG.normal(size=(J, N)))
    )
    approx = ens.project().add_noise(g=noise)
    y = jnp.asarray(RNG.normal(size=N))
    approx.conditional_map("g").particle_coefficients(g=y, key=jax.random.key(0))
    assert noise.count == J + 1
    noise.reset()
    approx.conditional_map("g")(ens, g=y, key=jax.random.key(0))
    assert noise.count == 2 * J


def test_regression_thin_svd_needs_the_identity_completion():
    """The naive ``U (I + Sigma^2)^-1/2 U^T`` omits the identity on the
    orthogonal complement of ``U``'s columns, and is wrong whenever
    ``rho < k``: wrong by O(1), with no exception and no shape error, since
    the naive form is a ``(k, k)`` matrix like the right one. Checked through
    ``condition`` at ``N < k``."""
    k, N = 7, 3
    S = RNG.normal(size=(k, N))
    T = _transform_of(S.T)
    w, Vec = np.linalg.eigh(np.eye(k) + S @ S.T)
    _close(T, (Vec * w**-0.5) @ Vec.T)
    U, sigma, _ = np.linalg.svd(S, full_matrices=False)
    naive = (U * (1.0 / np.sqrt(1.0 + sigma**2))) @ U.T
    assert np.abs(naive - T).max() > 0.5


def test_regression_stably_formed_invariant_beats_the_re_formed_one():
    """Obligation 3's invariant must be formed as ``T T^T + (T S)(T S)^T = I``.

    The algebraically equivalent ``T (I + S S^T) T^T = I`` re-forms the
    ``sigma_max^2``-sized intermediate whose rounding the check exists to
    detect, and no float64 implementation can meet the tolerance. The failure
    needs a *mixed* spectrum: with every singular value at one large scale,
    ``T T^T`` is itself of order ``sigma_max^-2`` and hides the loss.
    """
    k, N, sigma_max = 5, 7, 1e10
    S = _with_spectrum(k, N, [sigma_max, 1e5, 1.0, 1.0, 1.0])
    T = _transform_of(S.T)
    TS = T @ S
    stable = np.abs(T @ T.T + TS @ TS.T - np.eye(k)).max()
    re_formed = np.abs(T @ (np.eye(k) + S @ S.T) @ T.T - np.eye(k)).max()
    assert stable <= 128 * EPS * sigma_max
    assert re_formed > 1.0 and re_formed > 1e6 * stable


def test_regression_mixing_perturbation_representations_corrupts_the_update():
    """A perturbation must live in exactly one representation.

    The map writes the perturbed whitened residual as ``W(y - g_j) - eps_j``
    and never exposes ``eps``, because pushing the *same* ``eps`` through
    ``factor()`` to materialize ``g_j + L eps_j`` gives a different update:
    ``W L`` has orthonormal rows but is not the identity. Marginal statistics
    still look right, which is what makes it silent.
    """
    J, N = 6, 4
    R = _psd(N)
    Q, _ = np.linalg.qr(RNG.normal(size=(N, N)))
    term = RotatedWhitenPSD.from_matrix(R, Q)
    ens = Ensemble(
        u=jnp.asarray(RNG.normal(size=(J, 3))), g=jnp.asarray(RNG.normal(size=(J, N)))
    )
    cmap = ens.project().add_noise(g=term).conditional_map("g")
    y, key = jnp.asarray(RNG.normal(size=N)), jax.random.key(6)
    whitened_route = np.asarray(cmap(ens, g=y, key=key)["u"])
    eps = jax.random.normal(key, (J, N))
    L = term.factor()
    factor_route = np.asarray(cmap(ens.assign(g=ens["g"] + L.matvec(eps)), g=y)["u"])
    assert np.abs(whitened_route - factor_route).max() > 1e-3
    WL = _recovered_whitener(term, N) @ np.asarray(L.to_dense())
    _close(WL @ WL.T, np.eye(N))
    assert np.abs(WL - np.eye(N)).max() > 0.1


def test_regression_uncentered_transform_shifts_the_ensemble_mean():
    """``T 1 = 1`` holds only for a centered factor, and the square-root
    reading depends on it.

    Building ``S`` from raw particles rather than anomalies still produces a
    perfectly valid ``(I + S S^T)^-1/2`` -- no exception, no shape error --
    but it no longer fixes the ones vector, so the transformed anomalies stop
    summing to zero and the particles' mean is silently shifted away from the
    conditional mean. Checked on ``SquareRootMap`` and on ``realize_particles``
    after ``condition``, which center, against the uncentered transform.
    """
    from enskit.linalg import IdentityPlusGram

    J, N = 6, 4
    rng = np.random.default_rng(23)
    U = rng.normal(size=(J, 3))
    V = rng.normal(size=(J, N)) + 20.0  # a large mean makes the error unmistakable
    R = _psd(N)
    y = jnp.asarray(rng.normal(size=N))
    approx = (
        Ensemble(u=jnp.asarray(U), g=jnp.asarray(V))
        .project()
        .add_noise(g=DensePSD(jnp.asarray(R)))
    )
    m_post = np.asarray(approx.condition(g=y).mean("u"))
    for particles in (
        approx.square_root_map("g")(g=y)["u"],
        approx.condition(g=y).realize_particles()["u"],
    ):
        _close(np.asarray(particles).mean(axis=0), m_post)
    W = _recovered_whitener(approx.block_cov("g"), N)
    T_bad = np.asarray(
        IdentityPlusGram(jnp.asarray(V @ W.T / np.sqrt(J - 1))).inverse_sqrt().to_dense()
    )
    assert np.abs(T_bad @ np.ones(J) - 1.0).max() > 0.1
    shifted = m_post + T_bad @ (U - U.mean(axis=0))
    assert np.abs(shifted.mean(axis=0) - m_post).max() > 1e-3


def test_regression_gradient_is_finite_at_an_exactly_collapsed_operand():
    """Conditioning is smooth at an exactly collapsed given block, where a
    plain SVD's gradient is ``nan``; the layer routes through
    ``IdentityPlusGram``, whose derivative rules are finite there.

    The ``nan`` of a plain SVD is still asserted, so the test keeps exercising
    a genuinely degenerate operand. At ``S = 0`` the closed form of the gain
    coefficients' derivative is ``dw = dS b``, so the conditional mean moves
    by ``F_x dS b`` with ``S = (W F_c)^T``.
    """
    collapsed = jnp.zeros((3, 4))

    def plain_svd_transform(s):
        U, sigma, _ = jnp.linalg.svd(s, full_matrices=False)
        return jnp.eye(3) + (U * (1.0 / jnp.sqrt(1.0 + sigma**2) - 1.0)) @ U.T

    assert bool(
        jnp.isnan(jax.grad(lambda s: jnp.sum(plain_svd_transform(s)))(collapsed)).any()
    )

    Fx = jnp.asarray(RNG.normal(size=(2, 3)))
    y = jnp.asarray(RNG.normal(size=4))

    def mean(Fc):
        g = Gaussian(
            {"x": jnp.zeros(2), "c": jnp.zeros(4)},
            factors={"x": Fx, "c": Fc},
            block_covs={"c": PSDDiagonal(jnp.ones(4))},
        )
        return g.condition(c=y).mean("x")

    dFc = jnp.asarray(RNG.normal(size=(4, 3)))
    _, tangent = jax.jvp(mean, (collapsed.T,), (dFc,))
    _close(tangent, np.asarray(Fx) @ np.asarray(dFc).T @ np.asarray(y))
    grad = jax.grad(lambda F: jnp.sum(mean(F)))(collapsed.T)
    _close(grad, np.outer(np.asarray(y), np.asarray(Fx).sum(axis=0)))


def test_regression_singular_noise_covariance_yields_nan_without_raising():
    """A singular independent term is not detectable cheaply, so outside
    debug mode it surfaces as ``nan`` rather than an exception, on every
    conditioning path.

    Asserted deliberately, so the day it starts raising is visible here
    rather than in a caller's results. ``whiten`` is supported -- support is
    static and says nothing about values -- and returns ``inf``.
    """
    ens, approx = _singular_problem()
    singular = approx.block_cov("g")
    assert singular.supports("whiten")
    assert bool(jnp.isinf(singular.whiten(jnp.ones(3))).any())
    y = jnp.ones(3)
    for result in (
        approx.condition(g=y).mean("x"),
        approx.square_root_map("g")(g=y)["x"],
        approx.conditional_map("g")(ens, g=y)["x"],
        approx.conditional_map("g").particle_coefficients(g=y, key=jax.random.key(0)),
    ):
        assert bool(jnp.isnan(result).all())
    assert not bool(jnp.isfinite(approx.log_density(g=y)))


def test_regression_all_three_methods_report_a_nan_result_in_debug_mode():
    """The checks apply uniformly, and each message names the method whose
    check fired, never a constructor below it.

    Before this held, the old ``condition`` alone raised -- because it
    happened to route its mean through a constructor -- while the two
    sample-moving methods returned ``nan`` on identical inputs (issue #10). A
    caller who wraps a loop in ``debug_checks`` must get the same exception
    from every path. Results are built through the constructor-bypassing path
    after the result check, so no constructor's own check can fire first.
    """
    ens, approx = _singular_problem()
    y = jnp.ones(3)
    paths = _conditioning_paths(ens, approx, y)
    with debug_checks(True):
        for name, call in paths.items():
            with pytest.raises(ValueError) as info:
                call()
            message = str(info.value)
            method = {
                "map call": ".__call__",
                "square-root call": ".__call__",
            }.get(name, f".{name}")
            assert method in message, (name, message)
            assert "must be finite" not in message.split(":")[0]


def test_regression_result_checks_are_skipped_under_jit():
    """Tier 4 reads array values, so it is skipped on tracers: the checks are
    an eager-mode debugging aid, and a jitted loop gets the ``nan``
    regardless of the debug flag."""
    ens, approx = _singular_problem()
    y = jnp.ones(3)
    cmap = approx.conditional_map("g")  # built eagerly, so outside debug mode
    with debug_checks(True):
        assert bool(
            jnp.isnan(
                jax.jit(lambda a, y: a.square_root_map("g")(g=y)["x"])(approx, y)
            ).all()
        )
        mapped = jax.vmap(lambda y: cmap(ens, g=y)["x"])(jnp.stack([y, y]))
        assert bool(jnp.isnan(mapped).all())


def test_regression_sample_is_outside_the_result_check_by_design():
    """``sample`` and ``realize_particles`` are excluded because their output
    is non-finite only if a field is, and fields are validated at
    construction -- which is what makes the exclusion safe rather than an
    oversight."""
    with debug_checks(True):
        with pytest.raises(ValueError, match="PSDLowRank.F must be finite"):
            PSDLowRank(jnp.full((3, 2), jnp.nan))
        with pytest.raises(ValueError, match="mean of block 'x' must be finite"):
            Gaussian({"x": jnp.full(3, jnp.nan)}, block_covs={"x": DensePSD(jnp.eye(3))})
    silent = Gaussian({"x": jnp.zeros(3)}, factors={"x": jnp.full((3, 2), jnp.nan)})
    with debug_checks(True):
        assert bool(jnp.isnan(silent.sample(jax.random.key(0), 2)["x"]).all())
        aligned = EnsembleGaussian(
            {"x": jnp.zeros(3)}, factors={"x": jnp.zeros((3, 2))}, n_particles=2
        )
        nan_row = jax.tree_util.tree_map(lambda a: a * jnp.nan, aligned)
        assert bool(jnp.isnan(nan_row.realize_particles()["x"]).all())


def test_regression_each_update_applies_the_whitener_j_plus_one_times():
    """Whitening the factor and the residual in two calls costs ``2k``
    applications of ``W`` where ``k + 1`` are needed.

    Whitening is linear, so it commutes with differencing: one call on the
    ``k + 1`` stacked columns ``[F_c | y - m_c]`` yields every whitened
    quantity ``condition`` needs, and at ``k = J`` that is ``J + 1``; the
    particle route of the Matheron map reaches the same count. A second call
    produces correct numbers, which is why only a count catches it.
    """
    J, N = 6, 4
    noise = CountingWhitenPSD.counting(_psd(N))
    ens = Ensemble(
        u=jnp.asarray(RNG.normal(size=(J, 3))), g=jnp.asarray(RNG.normal(size=(J, N)))
    )
    approx = ens.project().add_noise(g=noise)
    y = jnp.asarray(RNG.normal(size=N))
    approx.condition(g=y)
    assert noise.count == J + 1
    noise.reset()
    approx.conditional_map("g").particle_coefficients(g=y, key=jax.random.key(0))
    assert noise.count == J + 1


def test_regression_anomalies_are_centered_before_whitening():
    """Centering and differencing must happen before whitening, and only
    accuracy shows it.

    Whitening is linear, so the two groupings give the same ``S`` in exact
    arithmetic, but centering *whitened* values makes the cancellation ratio
    ``||W g_bar|| / ||W a_j||`` rather than ``||g_bar|| / ||a_j||``, so the
    error grows with ``sqrt(kappa(R))`` when the mean lies along a precise
    direction of the noise. Both orders whiten ``J + 1`` vectors, so the count
    test passes either way; this asserts against an exact rational reference
    in the regime that separates them.
    """
    J, P, N = 5, 3, 4
    kappa = 1e10
    rng = np.random.default_rng(17)
    Q, _ = np.linalg.qr(rng.normal(size=(N, N)))
    Rm = Q @ np.diag(np.geomspace(1.0, 1.0 / kappa, N)) @ Q.T
    Rm = (Rm + Rm.T) / 2
    A = rng.normal(size=(J, N))
    V = A - A.mean(axis=0) + 1e10 * Q[:, -1]  # mean along R's most precise direction
    U = rng.normal(size=(J, P))
    term = DensePSD(jnp.asarray(Rm))
    y = rng.normal(size=N)
    exact = _exact_posterior_mean(U, V, Rm, y)
    scale = max(1.0, np.abs(exact).max())
    approx = Ensemble(u=jnp.asarray(U), g=jnp.asarray(V)).project().add_noise(g=term)
    shipped = np.asarray(approx.condition(g=jnp.asarray(y)).mean("u"))

    # the same core, whitening before centering -- the reverted grouping
    from enskit.linalg import IdentityPlusGram

    Wv = np.asarray(term.whiten(jnp.asarray(V)))
    S_bad = (Wv - Wv.mean(axis=0)) / np.sqrt(J - 1)
    r_bad = np.asarray(term.whiten(jnp.asarray(y))) - Wv.mean(axis=0)
    w = np.asarray(IdentityPlusGram(jnp.asarray(S_bad)).solve_factor(jnp.asarray(r_bad)))
    reverted = U.mean(axis=0) + (U - U.mean(axis=0)).T @ w / np.sqrt(J - 1)

    shipped_err = np.abs(shipped - exact).max() / scale
    reverted_err = np.abs(reverted - exact).max() / scale
    assert shipped_err < 1e-9, f"shipped relative error {shipped_err:.2e}"
    assert reverted_err > 1e-7, f"reverted relative error only {reverted_err:.2e}"
    assert reverted_err > 1e3 * shipped_err


def test_regression_a_collapsed_ensemble_is_exact_at_any_magnitude():
    """Anomalies of identical particles must be *exactly* zero.

    ``jnp.mean`` of ``J`` bit-identical rows sums and divides, which does not
    in general return the value it was given, so a plain subtraction leaves
    spurious anomalies of about ``eps * |g_bar|``. The gain amplifies those
    into a wrong, finite, nan-free update once the particles are large: at
    ``6e23`` they move by order 1 where they should stay put. The failure
    obligation 12 cannot see on its own, because it prescribes an exactly
    representable value whose mean happens to round back to itself.
    """
    J, N = 10, 2
    U = np.random.default_rng(0).normal(size=(J, 3))
    for magnitude in (1.0, 0.1, 6.02e23, 1e150):
        ens = Ensemble(u=jnp.asarray(U), g=jnp.full((J, N), magnitude))
        np.testing.assert_array_equal(ens.anomalies("g"), jnp.zeros((J, N)))
        np.testing.assert_array_equal(ens.mean("g"), jnp.full(N, magnitude))
        approx = ens.project().add_noise(g=DensePSD(jnp.eye(N)))
        w = approx.conditional_map("g").particle_coefficients(
            g=jnp.zeros(N), key=jax.random.key(0)
        )
        np.testing.assert_array_equal(ens["u"] + approx.factor("u").matvec(w), ens["u"])
        _close(approx.square_root_map("g")(g=jnp.zeros(N))["u"], U)


def test_regression_debug_checks_survive_a_trace_over_closed_over_arrays():
    """Tier-4 checks must be skipped inside a trace, not crash in it.

    ``jnp.isfinite`` applied to a *concrete* array while a trace is live is
    staged into that trace, so the check's bool conversion sees a tracer.
    Guarding on whether the operand is a tracer does not catch it: the
    operand is concrete. This is the documented driver shape -- values and
    terms closed over, only the particles traced -- and it used to raise
    ``TracerBoolConversionError`` from inside a debug check.
    """
    J = 4
    X = jnp.asarray(np.random.default_rng(0).normal(size=(J, 3)))
    G = jnp.asarray(np.random.default_rng(1).normal(size=(J, 2)))
    R = PSDDiagonal(jnp.ones(2))
    y = jnp.zeros(2)
    eager = np.asarray(
        Ensemble(x=X, g=G).project().add_noise(g=R).square_root_map("g")(g=y)["x"]
    )
    with debug_checks(True):
        approx = Ensemble(x=X, g=G).project().add_noise(g=R)
        _close(jax.jit(lambda: approx.square_root_map("g")(g=y)["x"])(), eager)
        jax.jit(lambda: approx.condition(g=y).mean("x"))()
        jax.jit(
            lambda: approx.conditional_map("g").particle_coefficients(
                g=y, key=jax.random.key(0)
            )
        )()
        jax.jit(lambda: approx.log_density(g=y))()

        def step(x, _):
            moved = Ensemble(x=x, g=G).project().add_noise(g=R).square_root_map("g")(g=y)
            return moved["x"], None

        out, _ = jax.lax.scan(step, X, None, length=3)
        assert bool(jnp.all(jnp.isfinite(out)))


def test_regression_check_order_with_two_simultaneous_violations():
    """The specified check order is only observable when two things are wrong.

    Supplying one bad argument at a time pins no ordering at all. Each case
    below has two violations, so exactly one error can win, and which one is
    the contract's ordering rule: the family guard; names and argument
    structure, then the key; structural conditions; capabilities; core
    shapes; values.
    """
    g, _ = _problem({"x": 2, "g": 3, "c": 2}, 4, terms=("g",))
    no_whiten = Gaussian(
        {"x": jnp.zeros(2), "g": jnp.zeros(3)},
        factors={"x": jnp.ones((2, 1)), "g": jnp.ones((3, 1))},
        block_covs={"g": PSDLowRank(jnp.ones((3, 3)))},
    )
    bad = jnp.zeros(7)
    # the family guard precedes everything
    with pytest.raises(ValueError, match="vmapped family"):
        _stacked(g).condition(z=bad)
    # names before shapes and structure
    with pytest.raises(KeyError):
        g.condition(z=jnp.zeros(1), g=bad)
    # structure (mixed) before capabilities and shapes
    with pytest.raises(ValueError, match="mixed"):
        g.condition(g=bad, c=bad)
    # capabilities before core shapes
    with pytest.raises(UnsupportedOpError):
        no_whiten.condition(g=bad)
    # core shapes before values, in debug mode
    with debug_checks(True):
        with pytest.raises(ValueError, match=r"shape \(3,\)"):
            g.condition(g=jnp.full(7, jnp.nan))
    # sample: n_particles before the key
    with pytest.raises(TypeError, match="n_particles"):
        g.sample(jax.random.PRNGKey(0), 2.0)
    # static arguments before the key's type
    with pytest.raises(ValueError, match="at least 2"):
        resample(jax.random.PRNGKey(0), Ensemble(x=jnp.zeros((3, 1))), 1)
    # log_density: whiten before logdet within a block, and block by block
    neither = PSDLowRank(jnp.asarray(RNG.normal(size=(3, 3))))  # no whiten, no logdet
    with pytest.raises(UnsupportedOpError, match="`whiten`"):
        Gaussian({"y": jnp.zeros(3)}, block_covs={"y": neither}).log_density(
            y=jnp.zeros(3)
        )
    whiten_only = WhitenOnlyPSD(jnp.linalg.cholesky(jnp.asarray(_psd(3))))
    h = Gaussian(
        {"a": jnp.zeros(3), "b": jnp.zeros(3)},
        block_covs={"a": whiten_only, "b": neither},
    )
    with pytest.raises(UnsupportedOpError, match="`logdet`"):
        h.log_density(b=jnp.zeros(3), a=jnp.zeros(3))
    # condition: the exact size conditions before "at least one target"
    exact = Gaussian(
        {"a": jnp.zeros(3), "b": jnp.zeros(2)},
        factors={"a": jnp.ones((3, 1)), "b": jnp.ones((2, 1))},
    )
    with pytest.raises(ValueError, match="N <= k"):
        exact.condition(a=jnp.zeros(3), b=jnp.zeros(2))
    # a map's call: names before the samples' structure
    ens, approx, *_ = _ensemble_problem(4, {"x": 2, "g": 3})
    with pytest.raises(KeyError):
        approx.conditional_map("g")(ens.drop("x"), z=bad)


def test_regression_anomalies_are_formed_over_the_member_axis_when_batched():
    """Means are taken over the particle axis with the axis kept, so an
    operand whose leading axis happens to equal ``J`` cannot broadcast
    against the wrong axis. Under ``vmap`` with a leading axis of size ``J``
    the subtraction right-aligns against the batch axis, and an implementation
    without ``keepdims`` returns wrong anomalies without raising."""
    J, d = 4, 3
    X = jnp.asarray(RNG.normal(size=(J, J, d)))  # a family of J ensembles of J particles
    anomalies = jax.vmap(lambda x: Ensemble(x=x).anomalies("x"))(X)
    factors = jax.vmap(lambda x: Ensemble(x=x).project().factor("x").to_dense())(X)
    for i in range(J):
        Xi = np.asarray(X[i])
        _close(anomalies[i], Xi - Xi.mean(axis=0))
        _close(factors[i], (Xi - Xi.mean(axis=0)).T / np.sqrt(J - 1))


# ===========================================================================
# Section 3 -- new regression tests
# ===========================================================================


def test_regression_the_weighted_projection_with_one_dominant_weight():
    """``1 - sum(w**2)`` computed from normalized weights loses about
    ``eps / (1 - w_max)`` relative accuracy when one weight dominates, and
    the weighted covariance inherits it; the log-space form keeps every
    digit. ``log(1 + t)`` must also be ``log1p``: a plain log-sum-exp loses
    ``eps / t`` in the same cancellation."""
    J, L = 4, 15.2
    lw = np.array([0.0, -L, -L, -L])
    X = RNG.normal(size=(J, 2))
    ens = Ensemble(x=jnp.asarray(X), log_weights=jnp.asarray(lw))
    getcontext().prec = 60
    e = [Decimal(float(v)).exp() for v in lw]
    w = [x / sum(e) for x in e]
    mean = [sum(w[j] * Decimal(float(X[j, i])) for j in range(J)) for i in range(2)]
    a = [[Decimal(float(X[j, i])) - mean[i] for i in range(2)] for j in range(J)]
    div = 1 - sum(x * x for x in w)
    C = np.array(
        [
            [
                float(sum(w[j] * a[j][p] * a[j][q] for j in range(J)) / div)
                for q in range(2)
            ]
            for p in range(2)
        ]
    )
    got = _cov_dense(ens.project(), "x")
    assert np.abs(got - C).max() <= 1e3 * EPS * np.abs(C).max()


def test_regression_exact_moments_are_joint_across_blocks():
    """Exact moments must be drawn jointly: matching each block's moments
    separately leaves the cross-covariance to sampling error, and a
    conditioning test built on such inputs compares against the wrong joint."""
    P, N, J = 3, 4, 12
    joint, ref, _ = _linear_gaussian(P, N)
    ens = exact_moment_ensemble(jax.random.key(0), joint, J)
    X = np.concatenate([np.asarray(ens["u"]), np.asarray(ens["g"])], axis=1)
    _, cov = _sample_moments(X)
    _close(cov, ref.C, factor=1e4)
    separate_u = exact_moment_ensemble(jax.random.key(1), joint.marginal("u"), J)
    separate_g = exact_moment_ensemble(jax.random.key(2), joint.marginal("g"), J)
    Xs = np.concatenate(
        [np.asarray(separate_u["u"]), np.asarray(separate_g["g"])], axis=1
    )
    _, cov_s = _sample_moments(Xs)
    cross = np.ix_(ref.idx("u"), ref.idx("g"))
    assert np.abs(cov_s[cross] - ref.C[cross]).max() > 0.1 * np.abs(ref.C[cross]).max()


def test_regression_exact_conditioning_with_j_equal_to_n():
    """Centered factor rows have rank at most ``J - 1``, so exact values with
    ``N = J`` given coordinates pass ``N <= k`` and are rank deficient by
    construction. Without the shape check, an aligned Gaussian with
    ``J = N`` returned a log density of ``-3e31`` under ``jit``, finite and
    meaningless; the check always runs because it reads only shapes."""
    J = 4
    ens = Ensemble(
        x=jnp.asarray(RNG.normal(size=(J, 2))), c=jnp.asarray(RNG.normal(size=(J, J)))
    )
    g = ens.project()
    with pytest.raises(ValueError, match="J - 1"):
        jax.jit(lambda g: g.log_density(c=jnp.zeros(J)))(g)
    ok = Ensemble(x=ens["x"], c=ens["c"][:, : J - 1]).project()
    assert bool(jnp.isfinite(ok.log_density(c=jnp.zeros(J - 1))))


def test_regression_a_marginal_keeps_the_latent_width():
    """A marginal over blocks without factor rows keeps the joint's latent
    width, so later operations on the same latent space agree on it. Were it
    to reset to 0, adding a factor row back (as the maps layer does) would be
    refused as a width mismatch, or worse, accepted on a different space."""
    g, _ = _problem({"x": 2, "z": 3}, 5, terms=("z",), rowless=("z",))
    m = g.marginal("z")
    assert m.latent_dim == 5 and m.factor("z") is None
    assert m.sample(jax.random.key(0), 3)["z"].shape == (3, 3)
    absorbed = m.absorb("z")
    assert absorbed.latent_dim == 5 + 3
    assert absorbed.factor("z").shape == (3, 8)


def test_regression_the_cancelling_log_density_form():
    """The quadratic term of the noisy log density must come from
    ``LowRankUpdate(D_c, F_c).whiten``, never ``||b||^2 - <Sb, A^-1 Sb>``,
    whose two terms nearly cancel for a typical value: at ``sigma_max = 1e8``
    the cancelling form is wrong by more than the value's own scale.
    Obligation 17 measures both against an exact reference; this pins that
    the shipped route goes through ``LowRankUpdate``."""
    N, k = 3, 4
    Fc = _axis_rows(N, k, 1e8)
    v = Fc @ RNG.normal(size=k) + RNG.normal(size=N)
    g = _identity_noise_gaussian(Fc)
    got = float(g.log_density(c=jnp.asarray(v)))
    C = LowRankUpdate(PSDDiagonal(jnp.ones(N)), Dense(jnp.asarray(Fc)))
    z = np.asarray(C.whiten(jnp.asarray(v)))
    want = -0.5 * (z @ z + float(C.logdet()) + N * np.log(2 * np.pi))
    assert got == pytest.approx(want, rel=4 * EPS)
    assert got == pytest.approx(_exact_log_density(Fc, v), rel=16 * EPS)


def test_regression_a_structured_factor_row_is_left_undensified():
    """Noisy conditioning right-multiplies a target's row by ``T`` as a
    product operator, so a structured row (a Kronecker factor, say) is never
    densified: its memory stays that of its parts. Densifying it would turn
    an ``O(d_A k_A + d_B k_B)`` row into an ``O(d_A d_B k)`` array."""
    A, B = RNG.normal(size=(3, 2)), RNG.normal(size=(4, 3))
    row = kron(Dense(jnp.asarray(A)), Dense(jnp.asarray(B)))  # (12, 6)
    Fc = RNG.normal(size=(2, 6))
    g = Gaussian(
        {"x": jnp.zeros(12), "c": jnp.zeros(2)},
        factors={"x": row, "c": jnp.asarray(Fc)},
        block_covs={"c": PSDDiagonal(jnp.ones(2))},
    )
    post = g.condition(c=jnp.ones(2))
    F = post.factor("x")
    assert isinstance(F, Product) and F.ops[0] is row
    dense, S = np.kron(A, B), Fc.T
    Fd = np.asarray(F.to_dense())
    _close(Fd @ Fd.T, dense @ np.linalg.inv(np.eye(6) + S @ S.T) @ dense.T)


def test_regression_systematic_resampling_never_selects_a_zero_weight():
    """A failed particle is marked with weight zero, and must never come back.

    The cumulative sum of the weights can round just below 1; when the last
    systematic position then falls past it, ``searchsorted`` returns ``J`` and
    a clip to ``J - 1`` selects the last particle, whatever its weight. In
    float32 with 1000 particles that is about one key in ten thousand. Here
    the last particle has weight zero and is ``nan``; key 20465 is one whose
    uniform draw lands past the rounded sum.
    """
    J = 1000
    lw = np.random.default_rng(1).normal(size=J).astype(np.float32)
    lw[-1] = -np.inf
    x = np.random.default_rng(2).normal(size=(J, 1)).astype(np.float32)
    x[-1] = np.nan
    ens = Ensemble(x=jnp.asarray(x), log_weights=jnp.asarray(lw))
    cw = jnp.cumsum(ens.weights)
    assert float(cw[-1]) < 1.0, "the precondition: the sum rounds below 1"
    key = jax.random.split(jax.random.key(0), 200000)[20465]
    u = float(jax.random.uniform(key, (), jnp.float32))
    assert (u + J - 1) / J >= float(cw[-1]), "the precondition: a position past it"
    out = resample(key, ens)
    assert bool(jnp.all(jnp.isfinite(out["x"])))


def test_regression_compress_without_factor_rows_drops_the_latent_space():
    """A marginal over blocks without rows keeps ``k``, so ``compress`` meets
    ``D_F = 0 < k``: the shared factor is empty, and the result has ``k = 0``
    rather than a failed concatenation."""
    g, _ = _problem({"u": 2, "y": 3}, 5, terms=("y",), rowless=("y",))
    out = g.marginal("y").compress()
    assert type(out) is Gaussian and out.latent_dim == 0
    assert out.block_cov("y") is g.block_cov("y")


def test_regression_a_factor_row_array_of_another_dtype_is_refused():
    """A raw float64 row on float32 means would promote every result of the
    distribution, silently; arrays share one dtype. (Operators carry no
    dtype, so a mismatched operator cannot be refused yet: issue on the
    operator dtype policy.)"""
    with pytest.raises(TypeError, match="dtype"):
        Gaussian({"u": jnp.zeros(2, jnp.float32)}, factors={"u": np.ones((2, 3))})


def test_pipe_refuses_a_family():
    J = 4
    ens, approx, *_ = _ensemble_problem(J, {"x": 2, "g": 3})
    for obj in (ens, approx):
        with pytest.raises(ValueError, match="vmapped family"):
            _stacked(obj).pipe(lambda o: o)
        assert obj.pipe(lambda o, k: k, 3) == 3


def test_conditional_map_satisfies_the_protocol():
    """``MatheronMap`` is a ``ConditionalMap``; ``SquareRootMap``, which
    couples particles, is not one by the protocol's obligation 4 even though
    it has the same attributes, and so it lacks a samples argument."""
    J = 5
    ens, approx, *_ = _ensemble_problem(J, {"x": 2, "g": 3})
    cmap = approx.conditional_map("g")
    assert isinstance(cmap, ConditionalMap) and isinstance(cmap, MatheronMap)
    perm = np.asarray([3, 0, 4, 1, 2])
    permuted = Ensemble(x=ens["x"][perm], g=ens["g"][perm])
    np.testing.assert_array_equal(
        cmap(permuted, g=jnp.zeros(3))["x"], cmap(ens, g=jnp.zeros(3))["x"][perm]
    )
    out = cmap(ens.assign(z=jnp.ones((J, 1))), g=jnp.zeros(3))
    assert out.names == ("x", "z")
