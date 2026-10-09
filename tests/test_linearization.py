"""Tests for ``Gaussian.regression`` and ``maps.statistical_linearization``.

The numbered sections follow the "Regression" obligations of the
distribution contract, then the "Statistical linearization" obligations of
the maps contract. Rules for the reference throughout:

- **The dense reference is hand-written here**: NumPy least squares over the
  particles, or closed forms over materialized covariances, never routed
  through ``enskit``.
- **Exactness tests compare against closed forms**, at a tolerance of a few
  machine epsilons times the quantity's natural scale and the problem's
  conditioning.
- **Every test draws from its own deterministic stream**, reseeded from the
  test's id.
"""

from __future__ import annotations

import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import enskit  # noqa: F401  -- enables x64 before any array exists
from enskit import kalman, maps
from enskit.distribution import (
    Ensemble,
    EnsembleGaussian,
    Gaussian,
    Regression,
    _regression,
    reweight,
)
from enskit.linalg import (
    Dense,
    DensePSD,
    LinOp,
    PSDDiagonal,
    PSDLinOp,
    UnsupportedOpError,
    debug_checks,
    linop,
)
from enskit.linalg.testing import check_operator
from enskit.maps import AdditiveNoise, Linear, Linearization, statistical_linearization

RNG = np.random.default_rng(0)
EPS = float(np.finfo(np.float64).eps)


@pytest.fixture(autouse=True)
def _reseed_rng(request):
    """Give every test its own deterministic stream, seeded from its id."""
    global RNG
    RNG = np.random.default_rng(zlib.crc32(request.node.nodeid.encode()))


def _close(got, want, scale=None, factor=1e3) -> None:
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape, f"shape {got.shape} != {want.shape}"
    scale = max(1.0, float(np.abs(want).max())) if scale is None else scale
    err = float(np.abs(got - want).max()) if got.size else 0.0
    assert err <= factor * EPS * scale, f"max abs err {err:.3e} at scale {scale:.3e}"


def _psd(n: int) -> np.ndarray:
    M = RNG.normal(size=(n, n))
    return M @ M.T + n * np.eye(n)


def _nonlinear(X: np.ndarray, d_y: int) -> np.ndarray:
    """A fixed nonlinear simulator, row by row."""
    W = np.random.default_rng(7).normal(size=(X.shape[1], d_y))
    return np.tanh(X @ W) + 0.1 * X[:, :1] ** 2


def _pairs(J: int, d_x: int, d_y: int = 3):
    X = RNG.normal(size=(J, d_x))
    return X, _nonlinear(X, d_y)


def _ols(X, Y):
    """Least squares with an unpenalized intercept: ``(A, c, residuals)``."""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    A = (np.linalg.pinv(Xc) @ Yc).T
    c = Y.mean(0) - A @ X.mean(0)
    return A, c, Y - X @ A.T - c


def _qr_count(f, *args) -> int:
    return _count(jax.make_jaxpr(f)(*args).jaxpr, "qr")


def _svd_count(f, *args) -> int:
    return _count(jax.make_jaxpr(f)(*args).jaxpr, "svd")


def _count(jaxpr, primitive: str) -> int:
    total = 0
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == primitive:
            total += 1
        for param in eqn.params.values():
            for cand in param if isinstance(param, (tuple, list)) else [param]:
                inner = getattr(cand, "jaxpr", cand)
                if hasattr(inner, "eqns"):
                    total += _count(inner, primitive)
    return total


@linop
class _WhitenOnlyPSD(PSDLinOp):
    """A diagonal PSD operator that can ``whiten`` but not ``solve``."""

    d: jax.Array

    @property
    def shape(self):
        return (self.d.shape[-1], self.d.shape[-1])

    @property
    def batch_shape(self):
        return self.d.shape[:-1]

    def _matvec(self, x):
        return self.d * x

    def _to_dense(self):
        return jnp.diag(self.d)

    def _whiten(self, x):
        return x / jnp.sqrt(self.d)


# =============================================================================
# Gaussian.regression
# =============================================================================

# --- R1. unique least squares ------------------------------------------------


def test_r1_least_squares_matches_lstsq_with_an_intercept():
    J, d_x = 40, 4
    X, Y = _pairs(J, d_x)
    reg = Ensemble(x=X, y=Y).project().regression("y", given="x")
    A, c, E = _ols(X, Y)
    assert isinstance(reg, Regression) and reg.given == ("x",) and reg.target == "y"
    _close(reg.coefficients["x"].to_dense(), A)
    _close(reg.intercept, c)
    _close(reg.residual_cov.to_dense(), E.T @ E / (J - 1))


def test_r1_regression_agrees_with_condition_at_every_value():
    J = 30
    X, Y = _pairs(J, 3)
    g = Ensemble(x=X, y=Y).project()
    reg = g.regression("y", given="x")
    for _ in range(3):
        x0 = RNG.normal(size=3)
        cond = g.condition(x=jnp.asarray(x0))
        _close(reg.coefficients["x"].matvec(x0) + reg.intercept, cond.mean("y"))
        _close(reg.residual_cov.to_dense(), cond.cov("y").to_dense())


def test_r1_a_linear_map_is_recovered_exactly_square_and_nonsymmetric():
    """Regression: a transposed coefficient passes every shape check when square."""
    J, d = 12, 4
    G = RNG.normal(size=(d, d))
    assert np.abs(G - G.T).max() > 0.1
    c = RNG.normal(size=d)
    X = RNG.normal(size=(J, d))
    reg = Ensemble(x=X, y=X @ G.T + c).project().regression("y", given="x")
    _close(reg.coefficients["x"].to_dense(), G, factor=1e4)
    _close(reg.intercept, c, factor=1e4)
    _close(reg.residual_cov.to_dense(), np.zeros((d, d)), scale=1.0, factor=1e4)


def test_r1_weighted_ensemble_is_weighted_least_squares():
    J, d_x = 25, 3
    X, Y = _pairs(J, d_x)
    lw = RNG.normal(size=J)
    ens = reweight(Ensemble(x=X, y=Y), jnp.asarray(lw))
    reg = ens.project().regression("y", given="x")
    w = np.exp(lw - lw.max())
    w /= w.sum()
    Xa = np.column_stack([np.ones(J), X])
    sw = np.sqrt(w)[:, None]
    beta = np.linalg.lstsq(sw * Xa, sw * Y, rcond=None)[0]
    _close(reg.coefficients["x"].to_dense(), beta[1:].T, factor=1e4)
    _close(reg.intercept, beta[0], factor=1e4)
    E = Y - Xa @ beta
    _close(
        reg.residual_cov.to_dense(),
        (w[:, None] * E).T @ E / (1 - np.sum(w**2)),
        factor=1e4,
    )


# --- R2. minimum norm ----------------------------------------------------------


def test_r2_minimum_norm_matches_the_centered_pseudoinverse():
    J, d_x = 8, 20
    X, Y = _pairs(J, d_x)
    reg = Ensemble(x=X, y=Y).project().regression("y", given="x", min_norm=True)
    A, c, _ = _ols(X, Y)
    _close(reg.coefficients["x"].to_dense(), A, factor=1e4)
    _close(reg.intercept, c, factor=1e4)
    _close(reg.residual_cov.to_dense(), np.zeros((3, 3)), scale=1.0, factor=1e4)


def test_r2_the_intercept_is_not_penalized():
    """Regression: ``pinv([1, X])`` also shrinks the intercept, a different answer.

    Both are minimum-norm least-squares solutions that interpolate the
    particles, so nothing about the fit itself tells them apart.
    """
    J, d_x = 10, 30
    X, Y = _pairs(J, d_x)
    reg = Ensemble(x=X, y=Y).project().regression("y", given="x", min_norm=True)
    A = np.asarray(reg.coefficients["x"].to_dense())
    wrong = (np.linalg.pinv(np.column_stack([np.ones(J), X])) @ Y)[1:].T
    assert np.abs(A - wrong).max() > 1e-3
    _close(A, _ols(X, Y)[0], factor=1e4)


def test_r2_on_a_linear_map_minimum_norm_is_g_on_the_span():
    J, d_x, d_y = 6, 15, 4
    G, c = RNG.normal(size=(d_y, d_x)), RNG.normal(size=d_y)
    X = RNG.normal(size=(J, d_x))
    reg = Ensemble(x=X, y=X @ G.T + c).project().regression("y", given="x", min_norm=True)
    Xc = X - X.mean(0)
    P = np.linalg.pinv(Xc) @ Xc  # projector onto the span of the anomalies
    _close(reg.coefficients["x"].to_dense(), G @ P, factor=1e4)


def test_r2_not_unique_without_min_norm_raises_with_both_remedies():
    X, Y = _pairs(5, 4)  # N = 4 = J - 1 is unique; 5 is not
    g = Ensemble(x=X, y=Y).project()
    g.regression("y", given="x")
    X, Y = _pairs(5, 5)
    with pytest.raises(ValueError, match=r"min_norm=True.*add_noise"):
        Ensemble(x=X, y=Y).project().regression("y", given="x")


def test_r2_both_forms_agree_at_the_boundary():
    """At ``N = J - 1`` the unique and minimum-norm computations must coincide."""
    J = 9
    X, Y = _pairs(J, J - 1)
    g = Ensemble(x=X, y=Y).project()
    A1, F1 = _regression._least_squares("t", g, ("x",), g.factor("y"))
    A2, F2 = _regression._min_norm("t", g, ("x",), g.factor("y"), True)
    _close(A1[0].to_dense(), A2[0].to_dense(), factor=1e5)
    _close(F1.to_dense(), F2.to_dense(), scale=1.0, factor=1e5)


def test_r2_a_plain_gaussian_wider_than_its_latent_space():
    """``H = I`` for a plain Gaussian with full-column-rank given rows."""
    k, N, d_y = 3, 7, 2
    Fx, Fy = RNG.normal(size=(N, k)), RNG.normal(size=(d_y, k))
    mx, my = RNG.normal(size=N), RNG.normal(size=d_y)
    g = Gaussian({"x": mx, "y": my}, factors={"x": Fx, "y": Fy})
    reg = g.regression("y", given="x", min_norm=True)
    A = Fy @ np.linalg.pinv(Fx)
    _close(reg.coefficients["x"].to_dense(), A, factor=1e4)
    _close(reg.intercept, my - A @ mx, factor=1e4)


# --- R3. ridge -------------------------------------------------------------------


@pytest.mark.parametrize("dense", [False, True])
def test_r3_a_term_on_the_inputs_is_ridge_regression(dense):
    J, d_x = 12, 20
    X, Y = _pairs(J, d_x)
    Lam = _psd(d_x) / d_x if dense else np.diag(RNG.uniform(0.2, 1.0, d_x))
    term = DensePSD(jnp.asarray(Lam)) if dense else PSDDiagonal(jnp.asarray(np.diag(Lam)))
    reg = Ensemble(x=X, y=Y).project().add_noise(x=term).regression("y", given="x")
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    Cxx, Cyx, Cyy = Xc.T @ Xc / (J - 1), Yc.T @ Xc / (J - 1), Yc.T @ Yc / (J - 1)
    A = Cyx @ np.linalg.inv(Cxx + Lam)
    _close(reg.coefficients["x"].to_dense(), A, factor=1e4)
    _close(reg.intercept, Y.mean(0) - A @ X.mean(0), factor=1e4)
    _close(reg.residual_cov.to_dense(), Cyy - A @ Cyx.T, factor=1e4)


def test_r3_noisy_given_terms_need_solve():
    X, Y = _pairs(10, 3)
    g = Ensemble(x=X, y=Y).project().add_noise(x=_WhitenOnlyPSD(jnp.ones(3)))
    with pytest.raises(UnsupportedOpError):
        g.regression("y", given="x")


# --- R4. general Gaussians and several blocks ---------------------------------


def test_r4_the_regression_of_a_linear_gaussian_joint_is_its_model_and_its_gain():
    """Regressing g on u returns (H, R); regressing u on g, the Kalman gain."""
    P, N = 4, 3
    m0, C0 = RNG.normal(size=P), _psd(P)
    H, R, c = RNG.normal(size=(N, P)), _psd(N), RNG.normal(size=N)
    prior = Gaussian.independent(u=(jnp.asarray(m0), DensePSD(jnp.asarray(C0))))
    joint = maps.pushforward(
        prior, Linear(jnp.asarray(H), jnp.asarray(c)), inputs="u", output="g"
    )
    joint = joint.add_noise(g=DensePSD(jnp.asarray(R)))
    fwd = joint.regression("g", given="u")
    _close(fwd.coefficients["u"].to_dense(), H, factor=1e4)
    _close(fwd.intercept, c, factor=1e4)
    _close(fwd.residual_cov.to_dense(), R, factor=1e4)
    back = joint.regression("u", given="g")
    S = H @ C0 @ H.T + R
    K = C0 @ H.T @ np.linalg.inv(S)
    _close(back.coefficients["g"].to_dense(), K, factor=1e4)
    _close(back.intercept, m0 - K @ (H @ m0 + c), factor=1e4)
    _close(back.residual_cov.to_dense(), C0 - K @ H @ C0, factor=1e4)


def test_r4_several_given_blocks_split_one_stacked_regression():
    J = 30
    X = RNG.normal(size=(J, 5))
    Y = _nonlinear(X, 2)
    g = Ensemble(b=X[:, 3:], y=Y, a=X[:, :3]).project()
    reg = g.regression("y", given=("a", "b"))
    assert reg.given == ("b", "a")  # block order, not the order written
    A, c, _ = _ols(X, Y)
    _close(reg.coefficients["a"].to_dense(), A[:, :3], factor=1e4)
    _close(reg.coefficients["b"].to_dense(), A[:, 3:], factor=1e4)
    _close(reg.intercept, c, factor=1e4)


def test_r4_a_target_independent_of_the_latent_space_has_zero_coefficients():
    R = _psd(2)
    g = Gaussian(
        {"x": RNG.normal(size=3), "z": RNG.normal(size=2)},
        factors={"x": RNG.normal(size=(3, 4))},
        block_covs={"z": DensePSD(jnp.asarray(R))},
    )
    reg = g.regression("z", given="x")
    _close(reg.coefficients["x"].to_dense(), np.zeros((2, 3)))
    _close(reg.intercept, g.mean("z"))
    _close(reg.residual_cov.to_dense(), R)


def test_r4_validation():
    X, Y = _pairs(10, 2)
    g = Ensemble(x=X, y=Y, z=Y).project()
    with pytest.raises(ValueError, match="both the target and given"):
        g.regression("y", given=("x", "y"))
    with pytest.raises(KeyError):
        g.regression("w", given="x")
    with pytest.raises(KeyError):
        g.regression("y", given="w")
    with pytest.raises(TypeError, match="min_norm"):
        g.regression("y", given="x", min_norm=1)
    with pytest.raises(TypeError, match="target"):
        g.regression(("y",), given="x")
    with pytest.raises(ValueError, match="mixed"):
        g.add_noise(x=PSDDiagonal(jnp.ones(2))).regression("y", given=("x", "z"))
    with pytest.raises(ValueError, match="vmapped family"):
        jax.tree.map(lambda a: jnp.stack([a, a]), g).regression("y", given="x")


def test_r4_rank_deficient_inputs_raise_in_debug_mode():
    X, Y = _pairs(10, 3)
    X[:, 2] = X[:, 0]
    g = Ensemble(x=X, y=Y).project()
    with debug_checks(), pytest.raises(ValueError, match="rank deficient"):
        g.regression("y", given="x")
    X, Y = _pairs(4, 6)
    X[:, :] = X[:, :1]  # every input coordinate the same: rank 1, not J - 1
    g = Ensemble(x=X, y=Y).project()
    with debug_checks(), pytest.raises(ValueError, match="rank deficient"):
        g.regression("y", given="x", min_norm=True)


def _dense_reference(g, target, given):
    """``C_yx C_xx^{-1}``, the intercept and ``C_yy - A C_xy``, densely."""
    Cxx = np.block([[np.asarray(g.cov(a, b).to_dense()) for b in given] for a in given])
    Cyx = np.concatenate([np.asarray(g.cov(target, b).to_dense()) for b in given], 1)
    A = Cyx @ np.linalg.inv(Cxx)
    mx = np.concatenate([np.asarray(g.mean(b)) for b in given])
    c = np.asarray(g.mean(target)) - A @ mx
    return A, c, np.asarray(g.cov(target).to_dense()) - A @ Cyx.T


def test_r4_noisy_with_a_given_block_lacking_a_factor_row_and_a_noisy_target():
    """Also: the given order written is not block order, and the block without
    a factor row needs neither ``whiten`` nor ``solve``."""
    k = 4
    g = Gaussian(
        {n: RNG.normal(size=d) for n, d in (("y", 3), ("a", 2), ("b", 2))},
        factors={"y": RNG.normal(size=(3, k)), "a": RNG.normal(size=(2, k))},
        block_covs={
            "y": DensePSD(jnp.asarray(_psd(3))),
            "a": DensePSD(jnp.asarray(_psd(2))),
            "b": _WhitenOnlyPSD(jnp.asarray(RNG.uniform(1, 2, 2))),
        },
    )
    reg = g.regression("y", given=("b", "a"))
    assert reg.given == ("a", "b")
    A, c, Omega = _dense_reference(g, "y", ("a", "b"))
    _close(reg.coefficients["a"].to_dense(), A[:, :2], factor=1e4)
    _close(reg.coefficients["b"].to_dense(), np.zeros((3, 2)))
    _close(reg.intercept, c, factor=1e4)
    _close(reg.residual_cov.to_dense(), Omega, factor=1e4)


def test_r4_with_no_latent_space_every_block_is_independent():
    R = _psd(2)
    g = Gaussian.independent(
        x=(jnp.asarray(RNG.normal(size=3)), DensePSD(jnp.asarray(_psd(3)))),
        y=(jnp.asarray(RNG.normal(size=2)), DensePSD(jnp.asarray(R))),
    )
    reg = g.regression("y", given="x")
    _close(reg.coefficients["x"].to_dense(), np.zeros((2, 3)))
    _close(reg.intercept, g.mean("y"))
    _close(reg.residual_cov.to_dense(), R)


@pytest.mark.parametrize("noisy", [False, True])
def test_r4_structured_factor_rows_are_regressed_without_densifying_the_result(noisy):
    from enskit.linalg import kron

    k = 6
    Fx = kron(
        Dense(jnp.asarray(RNG.normal(size=(2, 2)))),
        Dense(jnp.asarray(RNG.normal(size=(2, 3)))),
    )
    Fy = kron(
        Dense(jnp.asarray(RNG.normal(size=(1, 2)))),
        Dense(jnp.asarray(RNG.normal(size=(3, 3)))),
    )
    covs = {"x": DensePSD(jnp.asarray(_psd(4)))} if noisy else {}
    g = Gaussian(
        {"x": RNG.normal(size=4), "y": RNG.normal(size=3)},
        factors={"x": Fx, "y": Fy},
        block_covs=covs,
    )
    assert g.latent_dim == k
    reg = g.regression("y", given="x")
    A, c, Omega = _dense_reference(g, "y", ("x",))
    _close(reg.coefficients["x"].to_dense(), A, factor=1e5)
    _close(reg.intercept, c, factor=1e5)
    _close(reg.residual_cov.to_dense(), Omega, factor=1e5)


def test_r4_the_centered_coordinates_are_an_orthonormal_basis_of_the_complement():
    """The reflector stands for the explicit ``(J, J - 1)`` basis, never formed."""
    for J in (2, 3, 17):
        H = np.asarray(_regression._centered_coordinates(jnp.eye(J)))
        _close(H.T @ H, np.eye(J - 1))
        _close(H.T @ np.ones(J), np.zeros(J - 1), scale=1.0)


# --- R5. operators, transforms, counts ------------------------------------------


@pytest.mark.parametrize("regime", ["least_squares", "min_norm", "ridge"])
def test_r5_every_operator_passes_check_operator(regime):
    d_x = {"least_squares": 3, "min_norm": 12, "ridge": 12}[regime]
    X, Y = _pairs(8, d_x)
    g = Ensemble(x=X, y=Y).project()
    if regime == "ridge":
        g = g.add_noise(x=PSDDiagonal(jnp.full(d_x, 0.5)))
    reg = g.regression("y", given="x", min_norm=regime == "min_norm")
    assert isinstance(reg.coefficients["x"], LinOp)
    check_operator(reg.coefficients["x"])
    check_operator(reg.residual_cov)


def test_r5_one_decomposition_per_regression():
    X, Y = _pairs(8, 3)
    g = Ensemble(x=X, y=Y).project()
    gw = Ensemble(x=jnp.asarray(RNG.normal(size=(8, 12))), y=Y).project()
    gr = g.add_noise(x=PSDDiagonal(jnp.ones(3)))

    def ls(g):
        return g.regression("y", given="x")

    def mn(g):
        return g.regression("y", given="x", min_norm=True)

    assert (_qr_count(ls, g), _svd_count(ls, g)) == (1, 0)
    assert (_qr_count(mn, gw), _svd_count(mn, gw)) == (1, 0)
    assert (_qr_count(ls, gr), _svd_count(ls, gr)) == (0, 1)


@pytest.mark.parametrize("regime", ["least_squares", "min_norm", "ridge"])
def test_r5_derivatives_are_finite_and_match_a_dense_reference(regime):
    J, d_x = 8, {"least_squares": 3, "min_norm": 12, "ridge": 12}[regime]
    X0, Y0 = _pairs(J, d_x)
    lam = 0.5

    def ours(X):
        g = Ensemble(x=X, y=jnp.sin(X[:, :3]) + X[:, :1] ** 2).project()
        if regime == "ridge":
            g = g.add_noise(x=PSDDiagonal(jnp.full(d_x, lam)))
        reg = g.regression("y", given="x", min_norm=regime == "min_norm")
        return jnp.sum(reg.coefficients["x"].to_dense() ** 2) + jnp.sum(reg.intercept)

    def ref(X):
        Y = jnp.sin(X[:, :3]) + X[:, :1] ** 2
        Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        if regime == "least_squares":
            A = jnp.linalg.solve(Xc.T @ Xc, Xc.T @ Yc).T
        elif regime == "min_norm":
            A = (Xc.T @ jnp.linalg.solve(Xc @ Xc.T + jnp.ones((J, J)), Yc)).T
        else:
            A = (Yc.T @ Xc) @ jnp.linalg.inv(Xc.T @ Xc + (J - 1) * lam * jnp.eye(d_x))
        return jnp.sum(A**2) + jnp.sum(Y.mean(0) - A @ X.mean(0))

    X0 = jnp.asarray(X0)
    _close(ours(X0), ref(X0), factor=1e5)
    got, want = jax.grad(ours)(X0), jax.grad(ref)(X0)
    assert bool(jnp.all(jnp.isfinite(got)))
    _close(got, want, factor=1e6)


def test_r5_jit_and_vmap():
    X, Y = _pairs(10, 3)
    reg = Ensemble(x=X, y=Y).project().regression("y", given="x")
    jitted = jax.jit(lambda X, Y: Ensemble(x=X, y=Y).project().regression("y", given="x"))
    out = jitted(jnp.asarray(X), jnp.asarray(Y))
    assert out.given == ("x",)
    _close(out.coefficients["x"].to_dense(), reg.coefficients["x"].to_dense())
    Xs = jnp.asarray(RNG.normal(size=(3, 10, 3)))
    fam = jax.vmap(
        lambda X: Ensemble(x=X, y=jnp.sin(X)).project().regression("y", given="x")
    )(Xs)
    assert fam.batch_shape == (3,) and fam.intercept.shape == (3, 3)
    for i in range(3):
        one = Ensemble(x=Xs[i], y=jnp.sin(Xs[i])).project().regression("y", given="x")
        _close(fam.intercept[i], one.intercept)


# =============================================================================
# maps.statistical_linearization
# =============================================================================

# --- L1. the recipe ----------------------------------------------------------------


def test_l1_pushforward_then_fit_calls_the_simulator_once_and_recovers_it():
    J, d_x, d_y = 20, 4, 3
    G, c = jnp.asarray(RNG.normal(size=(d_y, d_x))), jnp.asarray(RNG.normal(size=d_y))
    calls = []

    def f(x):
        calls.append(x.shape)
        return x @ G.T + c

    ens = Ensemble(x=jnp.asarray(RNG.normal(size=(J, d_x)))).pipe(
        maps.pushforward, f, inputs="x", output="y"
    )
    fit = statistical_linearization(ens, inputs="x", output="y")
    assert calls == [(J, d_x)]
    assert isinstance(fit, Linearization) and isinstance(fit.map, Linear)
    _close(fit.map.op.to_dense(), G, factor=1e4)
    _close(fit.map.shift, c, factor=1e4)
    assert fit.residuals.shape == (J, d_y)
    _close(fit.residuals, np.zeros((J, d_y)), scale=1.0, factor=1e4)
    assert fit.map(jnp.zeros(d_x)).shape == (d_y,)  # a single input row, too


def test_l1_residuals_are_the_particles_misfit_and_their_covariance():
    J, d_x = 40, 3
    X, Y = _pairs(J, d_x)
    fit = statistical_linearization(Ensemble(x=X, y=Y), inputs="x", output="y")
    A, c, E = _ols(X, Y)
    _close(fit.residuals, E, factor=1e4)
    _close(fit.residual_cov.to_dense(), E.T @ E / (J - 1), factor=1e4)


def test_l1_several_inputs_keep_the_order_written():
    J = 30
    a, b = RNG.normal(size=(J, 2)), RNG.normal(size=(J, 3))
    Ga, Gb = RNG.normal(size=(4, 2)), RNG.normal(size=(4, 3))
    ens = Ensemble(a=a, b=b).pipe(
        maps.pushforward, lambda b, a: b @ Gb.T + a @ Ga.T, inputs=("b", "a"), output="y"
    )
    fit = statistical_linearization(ens, inputs=("b", "a"), output="y")
    assert fit.map.input_names == ("b", "a")
    _close(fit.map.op["a"].to_dense(), Ga, factor=1e4)
    _close(fit.map.op["b"].to_dense(), Gb, factor=1e4)
    # y is affine in (a, b), so the fitted map reproduces its covariances; a
    # comparison of means alone would hold for any A, since b = m_y - A m_x.
    out = maps.pushforward(ens.project(), fit.map, inputs=("b", "a"), output="z")
    _close(out.cov("z").to_dense(), out.cov("y").to_dense(), factor=1e4)
    _close(out.cov("z", "b").to_dense(), out.cov("y", "b").to_dense(), factor=1e4)


# --- L2. identities -----------------------------------------------------------------


@pytest.mark.parametrize("d_x", [3, 15])
def test_l2_the_fit_plus_its_residual_reproduces_the_projected_joint(d_x):
    J = 10
    X, Y = _pairs(J, d_x)
    ens = Ensemble(x=X, y=Y)
    fit = statistical_linearization(ens, inputs="x", output="y", min_norm=True)
    pushed = maps.pushforward(
        ens.marginal("x").project(), fit.map, inputs="x", output="y"
    )
    pushed = maps.pushforward(
        pushed, AdditiveNoise(fit.residual_cov), inputs="y", output="y"
    )
    # Omega is singular, so it is moved into the shared factor before cov(y)
    # builds a LowRankUpdate, which would need it to whiten.
    pushed = pushed.absorb("y")
    target = ens.project()
    _close(pushed.mean("y"), target.mean("y"), factor=1e4)
    _close(pushed.cov("y").to_dense(), target.cov("y").to_dense(), factor=1e4)
    _close(pushed.cov("y", "x").to_dense(), target.cov("y", "x").to_dense(), factor=1e4)


def test_l2_the_update_is_the_linear_gaussian_update_with_residual_noise():
    """The ensemble update treats the regression residual as extra noise."""
    J, d_x, d_y = 30, 3, 4
    X, Y = _pairs(J, d_x, d_y)
    R = _psd(d_y) / d_y
    y_obs = RNG.normal(size=d_y)
    ens = Ensemble(x=X, y=Y)
    post = kalman.update(
        ens,
        y=jnp.asarray(y_obs),
        noise={"y": DensePSD(jnp.asarray(R))},
        update_rule=kalman.SymmetricSquareRoot(),
    )
    A, b, E = _ols(X, Y)
    m, C = X.mean(0), np.cov(X.T)
    Omega = E.T @ E / (J - 1)
    S = A @ C @ A.T + R + Omega
    K = C @ A.T @ np.linalg.inv(S)
    _close(post.mean("x"), m + K @ (y_obs - A @ m - b), factor=1e4)
    _close(np.cov(np.asarray(post["x"]).T), C - K @ A @ C, factor=1e4)


def test_l2_for_symmetric_inputs_and_a_quadratic_the_fit_is_the_average_jacobian():
    """Stein's lemma, exactly: odd central moments vanish on an antithetic set."""
    n, d = 10, 3
    m = RNG.normal(size=d)
    D = RNG.normal(size=(n, d))
    X = np.concatenate([m + D, m - D])
    Q = RNG.normal(size=(2, d))
    fit = statistical_linearization(Ensemble(x=X, y=(X**2) @ Q.T), inputs="x", output="y")
    _close(fit.map.op.to_dense(), 2 * Q * m, factor=1e4)  # E[Df] = 2 Q diag(m)


# --- L3. Gaussians, weights, validation, transforms ----------------------------------


def test_l3_a_gaussian_gives_no_residuals_and_carries_ridge():
    J, d_x = 10, 20
    X, Y = _pairs(J, d_x)
    lam = 0.4
    g = Ensemble(x=X, y=Y).project().add_noise(x=PSDDiagonal(jnp.full(d_x, lam)))
    fit = statistical_linearization(g, inputs="x", output="y")
    assert fit.residuals is None
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    A = Yc.T @ Xc @ np.linalg.inv(Xc.T @ Xc + (J - 1) * lam * np.eye(d_x))
    _close(fit.map.op.to_dense(), A, factor=1e4)


def test_l3_a_weighted_ensemble_wider_than_its_rank_raises():
    X, Y = _pairs(6, 5)
    ens = reweight(Ensemble(x=X, y=Y), jnp.asarray(RNG.normal(size=6)))
    statistical_linearization(ens, inputs="x", output="y")
    X, Y = _pairs(6, 6)
    ens = reweight(Ensemble(x=X, y=Y), jnp.asarray(RNG.normal(size=6)))
    with pytest.raises(ValueError, match="Resample first"):
        statistical_linearization(ens, inputs="x", output="y", min_norm=True)


def test_l3_validation():
    X, Y = _pairs(10, 2)
    ens = Ensemble(x=X, y=Y)
    with pytest.raises(TypeError, match="Ensemble or a Gaussian"):
        statistical_linearization({"x": X}, inputs="x", output="y")
    with pytest.raises(KeyError):
        statistical_linearization(ens, inputs="x", output="w")
    with pytest.raises(KeyError):
        statistical_linearization(ens, inputs="w", output="y")
    with pytest.raises(ValueError, match="both an input and the output"):
        statistical_linearization(ens, inputs=("x", "y"), output="y")
    with pytest.raises(ValueError, match="at least one input"):
        statistical_linearization(ens, inputs=(), output="y")
    with pytest.raises(TypeError, match="output must be a str"):
        statistical_linearization(ens, inputs="x", output=("y",))
    with pytest.raises(TypeError, match="min_norm"):
        statistical_linearization(ens, inputs="x", output="y", min_norm="yes")
    with pytest.raises(ValueError, match="vmapped family"):
        statistical_linearization(
            jax.tree.map(lambda a: jnp.stack([a, a]), ens), inputs="x", output="y"
        )


def test_l3_jit_returns_the_same_fit_and_vmap_a_family():
    X, Y = _pairs(12, 3)
    fit = statistical_linearization(Ensemble(x=X, y=Y), inputs="x", output="y")
    out = jax.jit(
        lambda X, Y: statistical_linearization(Ensemble(x=X, y=Y), inputs="x", output="y")
    )(jnp.asarray(X), jnp.asarray(Y))
    _close(out.map.op.to_dense(), fit.map.op.to_dense())
    _close(out.residuals, fit.residuals)
    Xs = jnp.asarray(RNG.normal(size=(2, 12, 3)))
    fam = jax.vmap(
        lambda X: statistical_linearization(
            Ensemble(x=X, y=jnp.sin(X)), inputs="x", output="y"
        )
    )(Xs)
    assert fam.map.shift.shape == (2, 3) and fam.residuals.shape == (2, 12, 3)


def test_l3_an_ensemble_gaussian_stays_aligned_through_the_fit():
    X, Y = _pairs(10, 3)
    ens = Ensemble(x=X, y=Y)
    fit = statistical_linearization(ens, inputs="x", output="y")
    out = maps.pushforward(ens.marginal("x").project(), fit.map, inputs="x", output="y")
    assert isinstance(out, EnsembleGaussian)
