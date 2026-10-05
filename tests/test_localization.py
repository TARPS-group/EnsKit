"""Conformance and regression tests for domain localization in the Kalman layer.

The tests work through obligations 20 to 26 of the "Kalman contract"
(section *Localization*), then the user guide's page.

Rules for the reference throughout, as in ``tests/test_kalman.py``:

- **The dense reference is hand-written here**, in NumPy: neighborhoods are
  found by a stable sort of the distances, and each local problem is solved in
  the form of Hunt et al. (2007) for the square-root rule and with an explicit
  Kalman gain for the stochastic rule, never through ``enskit.distribution``
  or ``IdentityPlusGram``. A neighbor of weight zero is left out of the
  reference's local problem, which is what its infinite noise variance means.
- **Exactness tests compare against closed forms** at a few machine epsilons
  times the quantity's scale.
- **Every test draws from its own deterministic stream**, reseeded from the
  test's id.
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import enskit  # noqa: F401  -- enables x64 before any array exists
from enskit.distribution import Ensemble, Gaussian
from enskit.kalman import (
    DomainLocalization,
    LocalizedUpdateRule,
    Matheron,
    SymmetricSquareRoot,
    gaspari_cohn,
    gaussian_approximation,
    update,
)
from enskit.linalg import DensePSD, Identity, PSDDiagonal, debug_checks, linop
from enskit.testing import check_update_rule

RNG = np.random.default_rng(0)
EPS = float(np.finfo(np.float64).eps)
RULES = (SymmetricSquareRoot(), Matheron())


@pytest.fixture(autouse=True)
def _reseed_rng(request):
    """Give every test its own deterministic stream, seeded from its id."""
    global RNG
    RNG = np.random.default_rng(zlib.crc32(request.node.nodeid.encode()))


def _ones(r):
    """A taper of weight 1 everywhere: no localization."""
    return jnp.ones_like(r)


def _boxcar(r):
    """Weight 1 inside the radius and 0 outside: not a positive-definite function."""
    return jnp.where(r < 1.0, 1.0, 0.0)


def _periodic(n):
    def distance(point, points):
        d = jnp.abs(points[:, 0] - point[0])
        return jnp.minimum(d, n - d)

    return distance


# ---------------------------------------------------------------------------
# a problem on a line: a located block "x", a global block "th", given blocks
# ---------------------------------------------------------------------------


class _Line:
    """Particles over ``x`` (located on a line), ``th`` (global) and given blocks.

    The given blocks are linear images of the targets plus a site-local
    perturbation, so their anomalies are correlated with ``x`` near their
    sites. The noise is diagonal, one variance per given coordinate.
    """

    def __init__(self, J=9, P=14, given=(("g", 7),), seed_noise=None):
        self.J, self.P = J, P
        self.given = tuple(n for n, _ in given)
        self.dims = dict(given)
        x = RNG.normal(size=(J, P)).cumsum(axis=1) / 3
        th = RNG.normal(size=(J, 2))
        self.blocks = {"x": x, "th": th}
        self.xc = np.arange(P, dtype=float)[:, None]
        self.gc = {}
        self.r = {}
        for name, d in given:
            sites = np.sort(RNG.uniform(0, P - 1, size=d))
            near = np.clip(np.round(sites).astype(int), 0, P - 1)
            self.blocks[name] = (
                x[:, near] + 0.3 * th[:, :1] + 0.2 * RNG.normal(size=(J, d))
            )
            self.gc[name] = sites[:, None]
            self.r[name] = RNG.uniform(0.2, 1.0, size=d)
        self.names = ("x", "th") + self.given

    def ensemble(self, dtype=jnp.float64) -> Ensemble:
        return Ensemble({n: jnp.asarray(self.blocks[n], dtype) for n in self.names})

    def noise(self, dtype=jnp.float64, scale=None) -> dict:
        out = {}
        for n in self.given:
            D = PSDDiagonal(jnp.asarray(self.r[n], dtype))
            out[n] = D if scale is None else D * scale
        return out

    def values(self, dtype=jnp.float64) -> dict:
        return {n: jnp.asarray(RNG.normal(size=self.dims[n]), dtype) for n in self.given}

    def localization(self, *, radius=4.0, K=4, taper=None, order=None, bare=False):
        order = self.given if order is None else order
        gc = self.gc[self.given[0]] if bare else {n: self.gc[n] for n in order}
        return DomainLocalization(
            {"x": self.xc, "th": None}, gc, radius=radius, max_neighbors=K, taper=taper
        )


def _local_reference(prob, loc_taper, radius, K, y, *, rule, eps=None, scale=1.0):
    """The dense update of every local problem, and the global block's.

    ``eps`` is the ``(J, N)`` standard normal draw in the given blocks' block
    order. Returns ``{"x": (J, P), "th": (J, 2)}``.
    """
    J = prob.J
    G = np.concatenate([prob.blocks[n] for n in prob.given], axis=1)
    r = np.concatenate([prob.r[n] for n in prob.given]) * scale
    yv = np.concatenate([np.asarray(y[n]) for n in prob.given])
    gc = np.concatenate([prob.gc[n] for n in prob.given])[:, 0]
    Ya = G - G.mean(axis=0)

    def solve(Xa_col, xbar, xcol, idx, rho):
        keep = rho > 0
        idx, rho = idx[keep], rho[keep]
        rl = r[idx] / rho
        Y = Ya[:, idx]
        gbar = G[:, idx].mean(axis=0)
        if rule == "sqrt":
            Rinv = np.diag(1 / rl)
            Pa = np.linalg.inv((J - 1) * np.eye(J) + Y @ Rinv @ Y.T)
            wbar = Pa @ Y @ Rinv @ (yv[idx] - gbar)
            lam, V = np.linalg.eigh((J - 1) * Pa)
            W = V @ np.diag(np.sqrt(lam)) @ V.T
            return xbar + Xa_col @ wbar + W @ Xa_col
        Cxy = Xa_col @ Y / (J - 1)
        Cyy = Y.T @ Y / (J - 1)
        gain = np.linalg.solve(Cyy + np.diag(rl), Cxy)
        e = eps[:, idx] * np.sqrt(rl)
        return xcol + (yv[idx] - G[:, idx] - e) @ gain

    out = {}
    x = prob.blocks["x"]
    Xa = x - x.mean(axis=0)
    cols = []
    for p in range(prob.P):
        d = np.abs(gc - prob.xc[p, 0])
        idx = np.argsort(d, kind="stable")[:K]
        rho = np.asarray(loc_taper(jnp.asarray(d[idx] / radius)))
        cols.append(solve(Xa[:, p], x[:, p].mean(), x[:, p], idx, rho))
    out["x"] = np.stack(cols, axis=1)
    th = prob.blocks["th"]
    Ta = th - th.mean(axis=0)
    every = np.arange(len(r))
    out["th"] = np.stack(
        [
            solve(Ta[:, q], th[:, q].mean(), th[:, q], every, np.ones(len(r)))
            for q in range(th.shape[1])
        ],
        axis=1,
    )
    return out


def _eps(prob, key, dtype=jnp.float64):
    _, k_noise = jax.random.split(key)
    N = sum(prob.dims.values())
    return np.asarray(jax.random.normal(k_noise, (prob.J, N), dtype))


def _close(a, b, tol=200):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    scale = max(1.0, np.abs(b).max())
    assert np.abs(a - b).max() <= tol * EPS * scale, np.abs(a - b).max()


# ===========================================================================
# 20. gaspari_cohn
# ===========================================================================


def _gc_numpy(r):
    z = 2 * np.abs(r)
    out = np.zeros_like(z)
    a = z <= 1
    out[a] = -(z[a] ** 5) / 4 + z[a] ** 4 / 2 + 5 * z[a] ** 3 / 8 - 5 * z[a] ** 2 / 3 + 1
    b = (z > 1) & (z < 2)
    zb = z[b]
    out[b] = (
        zb**5 / 12 - zb**4 / 2 + 5 * zb**3 / 8 + 5 * zb**2 / 3 - 5 * zb + 4 - 2 / (3 * zb)
    )
    return np.maximum(out, 0)


def test_20_gaspari_cohn_against_its_formula():
    r = np.concatenate([RNG.uniform(-1.5, 1.5, size=200), [0.0, 0.5, 1.0, -1.0, 3.0]])
    got = np.asarray(gaspari_cohn(jnp.asarray(r)))
    np.testing.assert_allclose(got, _gc_numpy(r), atol=10 * EPS)
    assert float(gaspari_cohn(0.0)) == 1.0
    # zero at and beyond the support, even, in [0, 1]
    assert np.all(np.asarray(gaspari_cohn(jnp.asarray([1.0, 1.2, 7.0, -1.0]))) == 0.0)
    _close(gaspari_cohn(jnp.asarray(r)), gaspari_cohn(jnp.asarray(-r)), 1)
    assert np.all((got >= 0) & (got <= 1))
    # continuous where the branches meet
    for z0 in (0.5, 1.0):
        lo, hi = gaspari_cohn(z0 - 1e-9), gaspari_cohn(z0 + 1e-9)
        assert abs(float(lo) - float(hi)) < 1e-8


def test_20_gaspari_cohn_derivative_dtype_and_round_off():
    grad = jax.vmap(jax.grad(gaspari_cohn))(jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0, 2.0]))
    assert np.all(np.isfinite(np.asarray(grad)))
    assert float(grad[0]) == 0.0 and float(grad[-1]) == 0.0
    r32 = jnp.linspace(0.5, 1.0, 100001, dtype=jnp.float32)
    out = gaspari_cohn(r32)
    assert out.dtype == jnp.float32
    # the second branch rounds below zero in float32 without the clamp
    assert float(out.min()) >= 0.0
    assert gaspari_cohn(jnp.arange(3)).dtype == jnp.float64


# ===========================================================================
# 21. the geometry
# ===========================================================================


def test_21_neighbors_and_weights_against_a_sort():
    xc = RNG.uniform(0, 10, size=(6, 2))
    gc = {"a": RNG.uniform(0, 10, size=(5, 2)), "b": RNG.uniform(0, 10, size=(4, 2))}
    loc = DomainLocalization({"x": xc}, gc, radius=6.0, max_neighbors=5)
    pts = np.concatenate([gc["a"], gc["b"]])
    d = np.linalg.norm(xc[:, None, :] - pts[None, :, :], axis=-1)
    idx = np.argsort(d, axis=1, kind="stable")[:, :5]
    np.testing.assert_array_equal(np.asarray(loc.neighbors["x"]), idx)
    np.testing.assert_allclose(
        np.asarray(loc.weights["x"]),
        _gc_numpy(np.take_along_axis(d, idx, axis=1) / 6.0),
        atol=1e-14,
    )
    assert loc.located == ("x",) and loc.target_coords["x"].shape == (6, 2)
    assert list(loc.given_coords) == ["a", "b"]


def test_21_ties_go_to_the_lower_index_and_a_periodic_distance():
    # given sites 0..9 on a ring of 10; target at 0: sites 1 and 9 tie
    loc = DomainLocalization(
        {"x": np.zeros((1, 1))},
        np.arange(10.0)[:, None],
        radius=3.0,
        max_neighbors=4,
        distance=_periodic(10),
    )
    np.testing.assert_array_equal(np.asarray(loc.neighbors["x"])[0], [0, 1, 9, 2])
    w = np.asarray(loc.weights["x"])[0]
    np.testing.assert_allclose(w, _gc_numpy(np.array([0, 1, 1, 2]) / 3.0), atol=1e-14)
    # integer locations become floating
    loc = DomainLocalization(
        {"x": np.arange(3)[:, None]}, np.arange(4)[:, None], radius=2, max_neighbors=2
    )
    assert jnp.issubdtype(loc.weights["x"].dtype, jnp.floating)


@pytest.mark.parametrize(
    "kwargs, exc, match",
    [
        (dict(target_coords=[("x", np.zeros((2, 1)))]), TypeError, "mapping"),
        (dict(target_coords={1: np.zeros((2, 1))}), TypeError, "must be str"),
        (dict(target_coords={"x": np.zeros((2, 1)) * 1j}), TypeError, "real"),
        (dict(target_coords={"x": np.zeros(2)}), ValueError, "rows, q"),
        (dict(target_coords={"x": np.zeros((0, 1))}), ValueError, "rows, q"),
        (
            dict(target_coords={"x": np.zeros((2, 2))}),
            ValueError,
            "same number of columns",
        ),
        (dict(target_coords={"x": None}), ValueError, "no target block has locations"),
        (dict(given_coords={}), ValueError, "names no block"),
        (dict(radius=np.ones(2)), ValueError, "scalar"),
        (dict(max_neighbors=2.0), TypeError, "int"),
        (dict(max_neighbors=True), TypeError, "int"),
        (dict(max_neighbors=0), ValueError, "from 1"),
        (dict(max_neighbors=4), ValueError, "from 1"),
        (dict(taper=3), TypeError, "callable"),
        (dict(distance="euclid"), TypeError, "callable"),
        (dict(distance=lambda p, ps: ps), ValueError, "distance must map"),
        (dict(taper=lambda r: r[..., :1]), ValueError, "taper must return"),
    ],
)
def test_21_construction_checks(kwargs, exc, match):
    args = dict(
        target_coords={"x": np.zeros((2, 1))},
        given_coords=np.arange(3.0)[:, None],
        radius=1.0,
        max_neighbors=2,
    )
    args.update(kwargs)
    with pytest.raises(exc, match=match):
        DomainLocalization(args.pop("target_coords"), args.pop("given_coords"), **args)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(target_coords={"x": np.array([[np.nan]])}), "must be finite"),
        (dict(radius=0.0), "finite and positive"),
        (dict(radius=np.inf), "finite and positive"),
        (dict(distance=lambda p, ps: -jnp.abs(ps[:, 0] - p[0])), "nonnegative"),
        (dict(taper=lambda r: 1.5 * jnp.ones_like(r)), "above 1"),
        (dict(taper=lambda r: -jnp.ones_like(r)), r"in \[0, 1\]"),
    ],
)
def test_21_debug_mode_construction_checks(kwargs, match):
    args = dict(
        target_coords={"x": np.zeros((1, 1))},
        given_coords=np.arange(3.0)[:, None],
        radius=1.0,
        max_neighbors=2,
    )
    args.update(kwargs)
    tc, gc = args.pop("target_coords"), args.pop("given_coords")
    DomainLocalization(tc, gc, **args)  # checks off: no error
    with debug_checks(), pytest.raises(ValueError, match=match):
        DomainLocalization(tc, gc, **args)


# ===========================================================================
# 22. exactness without localization
# ===========================================================================


@pytest.mark.parametrize("rule", RULES, ids=repr)
def test_22_no_localization_is_the_global_update(rule):
    prob = _Line(given=(("g", 5), ("h", 3)))
    ens, noise, y = prob.ensemble(), prob.noise(), prob.values()
    loc = prob.localization(K=8, taper=_ones, order=("h", "g"))
    key = jax.random.key(3)
    want = update(ens, y, noise=noise, update_rule=rule, key=key)
    got = update(ens, y, noise=noise, update_rule=LocalizedUpdateRule(rule, loc), key=key)
    assert got.names == want.names == ("x", "th")
    for n in want.names:
        _close(got[n], want[n], 1000)


@pytest.mark.parametrize("rule", RULES, ids=repr)
def test_22_both_rules_pass_check_update_rule(rule):
    loc = DomainLocalization(
        {"u": np.arange(3.0)[:, None], "v": None},
        np.arange(4.0)[:, None],
        radius=1.0,
        max_neighbors=4,
        taper=_ones,
    )
    check_update_rule(LocalizedUpdateRule(rule, loc), diagonal_noise=True)


# ===========================================================================
# 23. exactness of every local problem
# ===========================================================================


@pytest.mark.parametrize("scale", (None, 0.5))
def test_23_square_root_solves_every_local_problem(scale):
    prob = _Line(given=(("g", 6), ("h", 5)))
    radius, K = 3.0, 5
    loc = prob.localization(radius=radius, K=K, order=("h", "g"))
    assert np.any(np.asarray(loc.weights["x"]) == 0), "no neighbor is masked"
    y = prob.values()
    got = update(
        prob.ensemble(),
        y,
        noise=prob.noise(scale=scale),
        update_rule=LocalizedUpdateRule(SymmetricSquareRoot(), loc),
    )
    want = _local_reference(
        prob,
        gaspari_cohn,
        radius,
        K,
        y,
        rule="sqrt",
        scale=1.0 if scale is None else scale,
    )
    for n in ("x", "th"):
        _close(got[n], want[n], 2000)


@pytest.mark.parametrize("scale", (None, 0.5))
def test_23_matheron_solves_every_local_problem_with_one_draw(scale):
    prob = _Line(given=(("g", 6), ("h", 5)))
    radius, K = 3.0, 5
    loc = prob.localization(radius=radius, K=K, order=("h", "g"))
    y, key = prob.values(), jax.random.key(11)
    got = update(
        prob.ensemble(),
        y,
        noise=prob.noise(scale=scale),
        update_rule=LocalizedUpdateRule(Matheron(), loc),
        key=key,
    )
    want = _local_reference(
        prob,
        gaspari_cohn,
        radius,
        K,
        y,
        rule="matheron",
        eps=_eps(prob, key),
        scale=1.0 if scale is None else scale,
    )
    for n in ("x", "th"):
        _close(got[n], want[n], 2000)


def test_23_bare_given_coords_and_identity_noise():
    prob = _Line()
    prob.r["g"] = np.ones(prob.dims["g"])
    loc = prob.localization(radius=3.0, K=4, bare=True)
    y = prob.values()
    got = update(
        prob.ensemble(),
        y,
        noise={"g": Identity(prob.dims["g"])},
        update_rule=LocalizedUpdateRule(SymmetricSquareRoot(), loc),
    )
    want = _local_reference(prob, gaspari_cohn, 3.0, 4, y, rule="sqrt")
    _close(got["x"], want["x"], 2000)


# ===========================================================================
# 24. the hazards
# ===========================================================================


@pytest.mark.parametrize("rule", RULES, ids=repr)
def test_24_regression_a_global_target_keeps_its_pooling(rule):
    """A block without locations is updated by every given coordinate at weight 1.

    Tapering a globally shared parameter would destroy the pooling over the
    whole domain that makes it global; with nothing raised, it would just be
    estimated from too little data. So the global block must come out exactly
    as the wrapped rule alone updates it, whatever the localization does to
    the located block.
    """
    prob = _Line()
    ens, noise, y = prob.ensemble(), prob.noise(), prob.values()
    key = jax.random.key(5)
    loc = prob.localization(radius=2.0, K=2)
    want = update(ens, y, noise=noise, update_rule=rule, key=key)
    got = update(ens, y, noise=noise, update_rule=LocalizedUpdateRule(rule, loc), key=key)
    _close(got["th"], want["th"], 100)
    assert np.abs(np.asarray(got["x"]) - np.asarray(want["x"])).max() > 1e-3


def test_24_regression_a_forgotten_target_raises():
    """A target missing from ``target_coords`` raises rather than going global.

    The design's stubs let an omitted block mean ``None``; a block forgotten
    or misspelled at a call site then got the global update, a plausible
    answer with nothing raised.
    """
    prob = _Line()
    ens = prob.ensemble()
    loc = DomainLocalization({"x": prob.xc}, prob.gc["g"], radius=3.0, max_neighbors=3)
    with pytest.raises(ValueError, match=r"\('th',\) are missing from target_coords"):
        update(
            ens,
            prob.values(),
            noise=prob.noise(),
            update_rule=LocalizedUpdateRule(SymmetricSquareRoot(), loc),
        )


@pytest.mark.parametrize("rule", RULES, ids=repr)
def test_24_regression_a_taper_that_is_not_positive_definite_is_valid(rule):
    """A boxcar taper is not a positive-definite function, and is still exact here.

    The positive-definiteness requirement on a taper is covariance
    localization's, where the taper multiplies a covariance. In domain
    localization a weight only scales a noise variance, so each local problem
    is still a Gaussian one, and the update solves it.
    """
    prob = _Line()
    loc = prob.localization(radius=2.5, K=5, taper=_boxcar)
    assert set(np.unique(np.asarray(loc.weights["x"]))) == {0.0, 1.0}
    y, key = prob.values(), jax.random.key(2)
    got = update(
        prob.ensemble(),
        y,
        noise=prob.noise(),
        update_rule=LocalizedUpdateRule(rule, loc),
        key=key,
    )
    name = "sqrt" if isinstance(rule, SymmetricSquareRoot) else "matheron"
    want = _local_reference(prob, _boxcar, 2.5, 5, y, rule=name, eps=_eps(prob, key))
    _close(got["x"], want["x"], 2000)


def test_24_regression_the_shared_draw_has_each_local_problems_variance():
    """The local residual is sqrt(rho) W(y - g) - eps, not sqrt(rho) (W(y - g) - eps).

    The design prototype scaled the draw by the taper too, which perturbs with
    the untapered variance where the local problem's noise has r / rho, so
    each coordinate's spread fell short of its local problem's. Given the
    particles, the localized coordinate's variance over keys must be the
    local problem's gain-weighted noise variance.
    """
    prob = _Line(J=7, P=6)
    loc = prob.localization(radius=3.0, K=4)
    approx = gaussian_approximation(prob.ensemble(), prob.noise())
    step = LocalizedUpdateRule(Matheron(), loc).build(prob.ensemble(), approx, "g")
    y = prob.values()
    keys = jax.random.split(jax.random.key(0), 4000)
    xs = np.asarray(jax.jit(jax.vmap(lambda k: step(y, key=k)["x"]))(keys))
    # the reference: particle 0, coordinate 2, variance over the draw
    p, j = 2, 0
    G, r = prob.blocks["g"], prob.r["g"]
    d = np.abs(prob.gc["g"][:, 0] - p)
    idx = np.argsort(d, kind="stable")[:4]
    rho = _gc_numpy(d[idx] / 3.0)
    keep = rho > 0
    idx, rho = idx[keep], rho[keep]
    Xa = prob.blocks["x"][:, p] - prob.blocks["x"][:, p].mean()
    Y = G[:, idx] - G[:, idx].mean(axis=0)
    rl = r[idx] / rho
    gain = np.linalg.solve(Y.T @ Y / 6 + np.diag(rl), Xa @ Y / 6)
    want = gain @ np.diag(rl) @ gain
    var = xs[:, j, p].var(ddof=1)
    assert abs(var - want) < 5 * want * np.sqrt(2 / 3999), (var, want)
    short = gain @ np.diag(r[idx]) @ gain
    assert abs(var - short) > 5 * want * np.sqrt(2 / 3999), "the test cannot tell"


def test_24_a_weight_above_one_raises_in_debug_mode():
    with debug_checks(), pytest.raises(ValueError, match="above 1"):
        DomainLocalization(
            {"x": np.zeros((2, 1))},
            np.zeros((3, 1)),
            radius=1.0,
            max_neighbors=2,
            taper=lambda r: 2.0 - r,
        )


def test_24_correlated_noise_is_refused():
    prob = _Line()
    ens = prob.ensemble()
    d = prob.dims["g"]
    M = RNG.normal(size=(d, d))
    with pytest.raises(TypeError, match="row-local"):
        update(
            ens,
            prob.values(),
            noise={"g": DensePSD(jnp.asarray(M @ M.T + np.eye(d)))},
            update_rule=LocalizedUpdateRule(Matheron(), prob.localization()),
            key=jax.random.key(0),
        )


def test_24_build_checks():
    prob = _Line(given=(("g", 5), ("h", 3)))
    ens, noise = prob.ensemble(), prob.noise()
    approx = gaussian_approximation(ens, noise)
    rule = LocalizedUpdateRule(SymmetricSquareRoot(), prob.localization())
    given = ("g", "h")

    with pytest.raises(TypeError, match="SymmetricSquareRoot"):
        LocalizedUpdateRule(object(), prob.localization())
    with pytest.raises(TypeError, match="DomainLocalization"):
        LocalizedUpdateRule(Matheron(), {"x": 1})
    # a plain Gaussian
    plain = Gaussian(
        {n: approx.mean(n) for n in approx.names},
        factors={n: approx.factor(n) for n in approx.names},
        block_covs=noise,
    )
    with pytest.raises(ValueError, match="EnsembleGaussian"):
        rule.build(ens, plain, given)
    # given blocks other than the localization's
    with pytest.raises(ValueError, match="locates the given blocks"):
        rule.build(
            ens.drop("h"), gaussian_approximation(ens.drop("h"), {"g": noise["g"]}), "g"
        )
    bare = LocalizedUpdateRule(SymmetricSquareRoot(), prob.localization(bare=True))
    with pytest.raises(ValueError, match="one bare array"):
        bare.build(ens, approx, given)
    # target names
    odd = DomainLocalization(
        {"x": prob.xc, "th": None, "z": None},
        {n: prob.gc[n] for n in given},
        radius=3.0,
        max_neighbors=3,
    )
    with pytest.raises(KeyError, match="'z', which is not a block"):
        LocalizedUpdateRule(Matheron(), odd).build(ens, approx, given)
    odd = DomainLocalization(
        {"x": prob.xc, "th": None, "h": None},
        {n: prob.gc[n] for n in given},
        radius=3.0,
        max_neighbors=3,
    )
    with pytest.raises(ValueError, match="'h', which is a given block"):
        LocalizedUpdateRule(Matheron(), odd).build(ens, approx, given)
    # wrong rows
    odd = DomainLocalization(
        {"x": prob.xc[:-1], "th": None},
        {n: prob.gc[n] for n in given},
        radius=3.0,
        max_neighbors=3,
    )
    with pytest.raises(ValueError, match=r"target_coords\['x'\] has 13 rows"):
        LocalizedUpdateRule(Matheron(), odd).build(ens, approx, given)
    # a target with a term, a given block without one
    with pytest.raises(ValueError, match="independent terms"):
        rule.build(ens, approx.add_noise(th=PSDDiagonal(jnp.ones(2))), given)
    exact = gaussian_approximation(ens, {"g": noise["g"]})
    with pytest.raises(ValueError, match="'h' has no independent term"):
        rule.build(ens, exact, given)
    # the order: the family guard before everything
    family = jax.tree_util.tree_map(lambda leaf: jnp.stack([leaf] * 2), ens)
    with pytest.raises(ValueError, match="vmapped family"):
        rule.build(family, approx, "nope")
    with pytest.raises(ValueError, match="vmapped family"):
        LocalizedUpdateRule(
            Matheron(),
            jax.tree_util.tree_map(
                lambda leaf: jnp.stack([leaf] * 2), prob.localization()
            ),
        )


def test_24_call_checks():
    prob = _Line()
    ens, noise, y = prob.ensemble(), prob.noise(), prob.values()
    approx = gaussian_approximation(ens, noise)
    loc = prob.localization()
    step = LocalizedUpdateRule(Matheron(), loc).build(ens, approx, "g")
    with pytest.raises(ValueError, match="key is required"):
        step(y)
    with pytest.raises(TypeError, match="typed key"):
        step(y, key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="missing"):
        step({}, key=jax.random.key(0))
    with pytest.raises(KeyError, match="not a given block"):
        step(y, x=y["g"], key=jax.random.key(0))
    with pytest.raises(ValueError, match=r"shape \(7,\)"):
        step(g=y["g"][:-1], key=jax.random.key(0))
    sq = LocalizedUpdateRule(SymmetricSquareRoot(), loc).build(ens, approx, "g")
    _close(sq(y)["x"], sq(y, key=jax.random.key(4))["x"], 0)
    with debug_checks(), pytest.raises(ValueError, match="must be finite"):
        sq(g=y["g"].at[0].set(jnp.nan))


# ===========================================================================
# 25. the span
# ===========================================================================


def test_25_localized_increments_leave_the_span_of_the_anomalies():
    prob = _Line(J=6, P=40, given=(("g", 20),))
    ens, noise, y = prob.ensemble(), prob.noise(), prob.values()
    x = prob.blocks["x"]
    basis, _ = np.linalg.qr((x - x.mean(axis=0)).T)  # (P, J) span of the anomalies

    def outside(post):
        inc = np.asarray(post["x"]).mean(axis=0) - x.mean(axis=0)
        return np.linalg.norm(inc - basis @ (basis.T @ inc)) / np.linalg.norm(inc)

    loc = prob.localization(radius=4.0, K=4)
    for rule in RULES:
        key = jax.random.key(0)
        assert outside(update(ens, y, noise=noise, update_rule=rule, key=key)) < 1e-12
        local = update(
            ens, y, noise=noise, update_rule=LocalizedUpdateRule(rule, loc), key=key
        )
        assert outside(local) > 0.1


# ===========================================================================
# 26. counts, derivatives, JAX, repr
# ===========================================================================


@linop
class CountingDiagonal(PSDDiagonal):
    """A diagonal noise that records how many vectors it has whitened."""

    def _whiten(self, x: Array) -> Array:
        log = self.__dict__.get("log")
        if log is not None:
            log.append(1 if x.ndim == 1 else int(np.prod(x.shape[:-1])))
        return super()._whiten(x)


def test_26_whitened_vectors_per_update():
    prob = _Line()
    ens, y = prob.ensemble(), prob.values()
    noise = CountingDiagonal(jnp.asarray(prob.r["g"]))
    log = []
    object.__setattr__(noise, "log", log)
    approx = gaussian_approximation(ens, {"g": noise})
    for rule in RULES:
        lr = LocalizedUpdateRule(rule, prob.localization())
        log.clear()
        step = lr.build(ens, approx, "g")
        assert sum(log) == prob.J
        log.clear()
        step(y, key=jax.random.key(0))
        assert sum(log) == 1
        with debug_checks():
            log.clear()
            lr.build(ens, approx, "g")(y, key=jax.random.key(0))
            assert sum(log) == prob.J + 1


def _count_svd(jaxpr) -> int:
    total = 0
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "svd":
            total += 1
        for param in eqn.params.values():
            for candidate in param if isinstance(param, (tuple, list)) else [param]:
                inner = getattr(candidate, "jaxpr", candidate)
                if hasattr(inner, "eqns"):
                    total += _count_svd(inner)
    return total


@pytest.mark.parametrize("rule", RULES, ids=repr)
def test_26_svds_per_build_and_call(rule):
    prob = _Line()
    ens, noise, y = prob.ensemble(), prob.noise(), prob.values()
    key = jax.random.key(0)
    for tc, want in (
        ({"x": prob.xc, "th": None}, 2),  # one batched local svd, one global
        ({"x": prob.xc, "th": np.zeros((2, 1))}, 2),  # two located blocks, no global
    ):
        loc = DomainLocalization(tc, prob.gc["g"], radius=3.0, max_neighbors=3)
        lr = LocalizedUpdateRule(rule, loc)

        def run(e, lr=lr):
            return lr.build(e, gaussian_approximation(e, noise), "g")(y, key=key)["x"]

        assert _count_svd(jax.make_jaxpr(run)(ens).jaxpr) == want
        step = lr.build(ens, gaussian_approximation(ens, noise), "g")
        assert _count_svd(jax.make_jaxpr(lambda s: s(y, key=key)["x"])(step).jaxpr) == 0


@pytest.mark.parametrize("rule", RULES, ids=repr)
def test_26_derivatives_are_finite_at_zero_weights_and_correct(rule):
    prob = _Line()
    noise = prob.noise()
    y = prob.values()["g"]
    key = jax.random.key(1)
    xg = jnp.asarray(prob.blocks["g"])

    def f(y, g, radius):
        loc = DomainLocalization(
            {"x": prob.xc, "th": None}, prob.gc["g"], radius=radius, max_neighbors=5
        )
        ens = Ensemble(
            x=jnp.asarray(prob.blocks["x"]), th=jnp.asarray(prob.blocks["th"]), g=g
        )
        out = update(
            ens, g=y, noise=noise, update_rule=LocalizedUpdateRule(rule, loc), key=key
        )
        return jnp.sum(jnp.sin(out["x"])) + jnp.sum(out["th"] ** 2)

    loc = DomainLocalization({"x": prob.xc}, prob.gc["g"], radius=2.5, max_neighbors=5)
    assert np.any(np.asarray(loc.weights["x"]) == 0), "no zero weight"
    grads = jax.jit(jax.grad(f, argnums=(0, 1, 2)))(y, xg, 2.5)
    for g in grads:
        assert np.all(np.isfinite(np.asarray(g)))
    for i, arg in enumerate((y, xg, jnp.asarray(2.5))):
        v = jnp.asarray(RNG.normal(size=jnp.shape(arg)))
        h = 1e-6
        args = [y, xg, 2.5]
        lo, hi = list(args), list(args)
        lo[i], hi[i] = arg - h * v, arg + h * v
        fd = (f(*hi) - f(*lo)) / (2 * h)
        an = jnp.sum(grads[i] * v)
        assert abs(float(fd - an)) < 1e-6 * max(1.0, abs(float(an))), (i, fd, an)


@pytest.mark.parametrize("rule", RULES, ids=repr)
def test_26_jax_pytrees_jit_and_vmap(rule):
    prob = _Line()
    ens, noise = prob.ensemble(), prob.noise()
    loc = prob.localization()
    lr = LocalizedUpdateRule(rule, loc)
    approx = gaussian_approximation(ens, noise)
    step = lr.build(ens, approx, "g")
    key = jax.random.key(0)
    y = prob.values()
    want = step(y, key=key)
    for obj in (loc, lr, step):
        leaves, tree = jax.tree_util.tree_flatten(obj)
        again = jax.tree_util.tree_unflatten(tree, leaves)
        assert type(again) is type(obj) and repr(again) == repr(obj)
    leaves, tree = jax.tree_util.tree_flatten(step)
    _close(jax.tree_util.tree_unflatten(tree, leaves)(y, key=key)["x"], want["x"], 0)
    # the rule and the localization cross jit as arguments
    jitted = jax.jit(lambda r, e, y: update(e, y, noise=noise, update_rule=r, key=key))(
        lr, ens, y
    )
    _close(jitted["x"], want["x"], 1e3)
    ys = jnp.stack([y["g"], y["g"] + 1.0])
    batched = jax.vmap(lambda v: step(g=v, key=key)["x"])(ys)
    _close(batched[1], step(g=ys[1], key=key)["x"], 1e3)
    steps = jax.tree_util.tree_map(lambda leaf: jnp.stack([leaf] * 2), step)
    with pytest.raises(ValueError, match="vmapped family"):
        steps(y, key=key)


@pytest.mark.parametrize("rule", RULES, ids=repr)
def test_26_float32_stays_float32(rule):
    prob = _Line()
    ens = prob.ensemble(jnp.float32)
    out = update(
        ens,
        prob.values(jnp.float32),
        noise=prob.noise(jnp.float32),
        update_rule=LocalizedUpdateRule(rule, prob.localization()),
        key=jax.random.key(0),
    )
    assert all(out[n].dtype == jnp.float32 for n in out.names)
    assert all(bool(jnp.all(jnp.isfinite(out[n]))) for n in out.names)


def test_26_reprs():
    prob = _Line()
    loc = DomainLocalization(
        {"x": prob.xc, "th": None}, {"g": prob.gc["g"]}, radius=3.0, max_neighbors=4
    )
    assert repr(loc) == (
        "DomainLocalization(target_dims={'x': 14, 'th': None}, given_dims={'g': 7}, "
        "coord_dim=1, max_neighbors=4, taper=gaspari_cohn)"
    )
    lr = LocalizedUpdateRule(Matheron(), loc)
    assert repr(lr) == "LocalizedUpdateRule(Matheron(), DomainLocalization(...))"
    step = lr.build(
        prob.ensemble(), gaussian_approximation(prob.ensemble(), prob.noise()), "g"
    )
    assert repr(step) == (
        "LocalizedUpdate(rule=Matheron(), given=('g',), targets=('x', 'th'), "
        "n_particles=9, located=('x',))"
    )
    assert "given_dims=7" in repr(prob.localization(bare=True))


def test_26_regression_a_promoted_noise_raises_rather_than_changing_dtype():
    """A scaled float32 noise must not give a float64 result silently.

    The operator layer promotes a scalar scale to float64, so a float32
    ``PSDDiagonal`` times 0.5 whitens to float64. The localized result's dtype
    followed the whitened factor's, and the update returned float64 blocks for
    float32 particles: a run would be demoted to float64 with nothing raised.
    """
    prob = _Line()
    ens = prob.ensemble(jnp.float32)
    noise = prob.noise(jnp.float32, scale=0.5)
    assert noise["g"].whiten(jnp.ones(prob.dims["g"], jnp.float32)).dtype != jnp.float32
    for rule in RULES:
        with pytest.raises(TypeError, match="whitens to dtype float64"):
            update(
                ens,
                prob.values(jnp.float32),
                noise=noise,
                update_rule=LocalizedUpdateRule(rule, prob.localization()),
                key=jax.random.key(0),
            )


def test_26_regression_collocated_locations_have_finite_derivatives():
    """A target located exactly at a given coordinate has finite derivatives.

    The Euclidean distance's square root has an infinite derivative at zero,
    which gave ``nan`` derivatives in both the target and the given locations
    whenever an observation sat on a grid point, the common case.
    """
    xc = jnp.arange(3.0)[:, None]

    def weights(xc, gc):
        loc = DomainLocalization({"x": xc}, gc, radius=2.0, max_neighbors=2)
        return jnp.sum(loc.weights["x"])

    g_x, g_g = jax.grad(weights, argnums=(0, 1))(xc, jnp.asarray([[1.0], [2.5]]))
    assert np.all(np.isfinite(np.asarray(g_x))) and np.all(np.isfinite(np.asarray(g_g)))


def test_26_regression_fresh_callables_do_not_retrace():
    """A taper or distance passed as a fresh lambda does not make ``jit`` retrace.

    They were static fields, compared by identity, so every new localization
    with an equal lambda compiled the update again.
    """
    prob = _Line()
    ens, noise, y = prob.ensemble(), prob.noise(), prob.values()
    traces = []

    @jax.jit
    def run(loc):
        traces.append(1)
        rule = LocalizedUpdateRule(SymmetricSquareRoot(), loc)
        return update(ens, y, noise=noise, update_rule=rule)["x"]

    for _ in range(3):
        run(prob.localization(taper=lambda r: gaspari_cohn(r)))
    assert len(traces) == 1


def test_26_more_construction_checks():
    xc = np.zeros((2, 1))
    with pytest.raises(TypeError, match="holds None"):
        DomainLocalization({"x": xc}, {"g": None}, radius=1.0, max_neighbors=1)
    # type errors come before the empty-mapping check
    with pytest.raises(TypeError, match="max_neighbors must be an int"):
        DomainLocalization({"x": xc}, {}, radius=1.0, max_neighbors="3")
    loc = DomainLocalization(
        {"x": xc}, np.zeros((3, 1)), radius=1.0, max_neighbors=np.int64(2)
    )
    assert loc.max_neighbors == 2 and type(loc.max_neighbors) is int


# ===========================================================================
# the user guide's page
# ===========================================================================


def test_the_user_guide_page_runs_and_says_what_it_does():
    """Every Python block of ``docs/user-guide/localization.md``, in order.

    The page's claims are then checked against what the blocks computed.
    """
    page = Path(__file__).parents[1] / "docs" / "user-guide" / "localization.md"
    blocks = re.findall(r"```python\n(.*?)```", page.read_text(), re.S)
    assert len(blocks) >= 4
    ns: dict = {}
    for block in blocks:
        exec(compile(block, str(page), "exec"), ns)
    # the global update is confined to the span; the localized one is not
    assert ns["outside_global"] < 1e-10 and ns["outside_local"] > 0.5  # "most"
    # localization brings the mean much closer to the exact posterior's, and
    # the spread from collapsed to about right; the page quotes these numbers
    assert abs(ns["error_global"] - 0.50) < 0.01 and abs(ns["error_local"] - 0.07) < 0.01
    assert abs(ns["spread_global"] - 0.15) < 0.01
    assert abs(ns["exact_spread"] - 0.21) < 0.01
    assert abs(ns["spread_local"] - ns["exact_spread"]) < 0.01
    assert ns["post"].names == ("x",)
    assert ns["both"].target_names == ("x", "offset")
    assert ns["ring_localization"].neighbors["x"].shape == (40, 10)
