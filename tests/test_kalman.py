"""Conformance and regression tests for the Kalman layer.

The file has three sections. The first works through the numbered conformance
obligations of the "Kalman contract"; the second holds the regression tests
ported from ``tests/test_eki.py`` under the contract's table, each keeping its
old docstring's reasoning; the third holds the new regression tests the
contract lists. Regression tests document why a rule exists, and deleting one
as redundant loses that.

Rules for the reference throughout:

- **The dense reference is hand-written here**, in NumPy, from the arrays a
  problem was built from: sample moments, gains and draws are recomputed
  without calling ``enskit.distribution`` or ``IdentityPlusGram``.
  ``exact_moment_ensemble`` may build inputs; it is never the reference.
- **Exactness tests compare against closed forms** at a few machine epsilons
  times the quantity's scale; statistical tests compare against standard
  errors computed from the known variance of the estimator.
- **Every test draws from its own deterministic stream**, reseeded from the
  test's id.
"""

from __future__ import annotations

import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import enskit  # noqa: F401  -- enables x64 before any array exists
from enskit import kalman
from enskit.distribution import (
    Ensemble,
    EnsembleGaussian,
    Gaussian,
    SquareRootMap,
    exact_moment_ensemble,
    resample,
    reweight,
)
from enskit.kalman import (
    Matheron,
    ParticleUpdate,
    SymmetricSquareRoot,
    UpdateRule,
    gaussian_approximation,
    inflate_additive,
    inflate_multiplicative,
    relax_to_prior_perturbations,
    relax_to_prior_spread,
    update,
)
from enskit.linalg import (
    DensePSD,
    PSDDiagonal,
    PSDLinOp,
    Triangular,
    UnsupportedOpError,
    debug_checks,
    dense_matvec,
    linop,
    tri_solve,
)

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
    return M @ M.T / n + np.eye(n)


@linop
class CountingWhitenPSD(PSDLinOp):
    """A whitening operator that records how many vectors it has whitened.

    The count is a plain list on the instance, invisible to the pytree
    machinery; only the instance :meth:`counting` builds counts.
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


@linop
class WhitenOnlyPSD(PSDLinOp):
    """A PSD operator that whitens, but has no ``factor``."""

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
class FactorOnlyPSD(PSDLinOp):
    """A PSD operator with a ``factor`` but no ``whiten``."""

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


def test_local_operators_satisfy_the_operator_contract():
    """The fixtures are contract-valid, so failures cannot be blamed on them."""
    from enskit.linalg.testing import check_operator

    L = jnp.linalg.cholesky(jnp.asarray(_psd(4)))
    check_operator(CountingWhitenPSD.counting(_psd(4)))
    check_operator(WhitenOnlyPSD(L))
    check_operator(FactorOnlyPSD(L))


# ---------------------------------------------------------------------------
# the problems and the hand-written dense reference
# ---------------------------------------------------------------------------


class _Problem:
    """A linear-Gaussian joint over targets and given blocks, as NumPy arrays.

    ``u`` has a Gaussian prior with factor ``L0``; each further target ``v``
    is ``B u``; each given block is ``H u`` observed with noise ``R``. The
    joint without noise has latent width ``P``.
    """

    def __init__(self, P=4, targets=(), given=(("g", 6),), J=32, noise_scale=0.3):
        self.P, self.J = P, J
        self.L0 = RNG.normal(size=(P, P)) + 2 * np.eye(P)
        self.m0 = RNG.normal(size=P)
        self.maps = {"u": np.eye(P)}
        for name, d in targets:
            self.maps[name] = RNG.normal(size=(d, P))
        self.target_names = tuple(self.maps)
        self.given_names = tuple(n for n, _ in given)
        for name, d in given:
            self.maps[name] = RNG.normal(size=(d, P)) * 0.5
        self.R = {name: noise_scale**2 * _psd(d) for name, d in given}
        self.names = self.target_names + self.given_names
        self.joint = Gaussian(
            {n: jnp.asarray(self.maps[n] @ self.m0) for n in self.names},
            factors={n: jnp.asarray(self.maps[n] @ self.L0) for n in self.names},
        )
        self.y = {
            n: self.maps[n] @ (self.m0 + self.L0 @ RNG.normal(size=P))
            + RNG.normal(size=d)
            for n, d in given
        }

    def noise(self, op=DensePSD):
        return {n: op(jnp.asarray(R)) for n, R in self.R.items()}

    def values(self):
        return {n: jnp.asarray(v) for n, v in self.y.items()}

    def exact_ensemble(self, key=0):
        return exact_moment_ensemble(jax.random.key(key), self.joint, self.J)

    def random_ensemble(self):
        u = self.m0 + RNG.normal(size=(self.J, self.P)) @ self.L0.T
        return Ensemble({n: jnp.asarray(u @ self.maps[n].T) for n in self.names})

    def exact_posterior(self, noisy=True):
        """The exact conditional moments of the targets, densely."""
        H = np.concatenate([self.maps[n] for n in self.given_names])
        T = np.concatenate([self.maps[n] for n in self.target_names])
        C0 = self.L0 @ self.L0.T
        Cgg = H @ C0 @ H.T
        if noisy:
            Cgg = Cgg + _block_diag([self.R[n] for n in self.given_names])
        K = T @ C0 @ H.T @ np.linalg.inv(Cgg)
        y = np.concatenate([self.y[n] for n in self.given_names])
        mean = T @ self.m0 + K @ (y - H @ self.m0)
        cov = T @ C0 @ T.T - K @ H @ C0 @ T.T
        return mean, cov


def _block_diag(blocks):
    n = sum(b.shape[0] for b in blocks)
    out, i = np.zeros((n, n)), 0
    for b in blocks:
        d = b.shape[0]
        out[i : i + d, i : i + d] = b
        i += d
    return out


def _stack(ens: Ensemble, names) -> np.ndarray:
    return np.concatenate([np.asarray(ens[n], np.float64) for n in names], axis=1)


def _moments(x: np.ndarray):
    """Sample mean and covariance (divisor J - 1) of a row-wise ensemble."""
    a = x - x.mean(axis=0)
    return x.mean(axis=0), a.T @ a / (x.shape[0] - 1)


def _close(got, want, factor=1e3, scale=None):
    got, want = np.asarray(got), np.asarray(want)
    scale = max(1.0, np.abs(want).max()) if scale is None else scale
    np.testing.assert_allclose(got, want, rtol=0, atol=factor * EPS * scale)


def _dense_gain(ens: Ensemble, targets, given, R: np.ndarray) -> np.ndarray:
    """The sample gain K = C_xg (C_gg + R)^-1, from the particles alone."""
    x, g = _stack(ens, targets), _stack(ens, given)
    J = x.shape[0]
    ax, ag = x - x.mean(axis=0), g - g.mean(axis=0)
    Cxg, Cgg = ax.T @ ag / (J - 1), ag.T @ ag / (J - 1)
    return Cxg @ np.linalg.inv(Cgg + R)


def _matheron_reference(prob, ens, key, *, target_terms=None):
    """Each particle's dense update, with the draws recomputed from the key."""
    targets, given = prob.target_names, prob.given_names
    R = _block_diag([prob.R[n] for n in given])
    K = _dense_gain(ens, targets, given, R)
    J = ens.n_particles
    k_targets, k_noise = jax.random.split(key)
    N = R.shape[0]
    eps = np.asarray(jax.random.normal(k_noise, (J, N)))
    # the whitener each term applies, recovered as whiten(I) transposed; the
    # stacked whitener is block diagonal
    W = _block_diag(
        [
            np.asarray(
                DensePSD(jnp.asarray(prob.R[n])).whiten(jnp.eye(prob.R[n].shape[0]))
            ).T
            for n in given
        ]
    )
    e = np.linalg.solve(W, eps.T).T
    y = np.concatenate([prob.y[n] for n in given])
    x = _stack(ens, targets) + (y - _stack(ens, given) - e) @ K.T
    out, i = {}, 0
    terms = target_terms or {}
    sampled = [n for n in targets if n in terms]
    keys = jax.random.split(k_targets, len(sampled)) if sampled else None
    for n in targets:
        d = ens.dims[n]
        out[n] = x[:, i : i + d]
        if n in terms:
            L = np.linalg.cholesky(terms[n])
            z = np.asarray(jax.random.normal(keys[sampled.index(n)], (J, d)))
            out[n] = out[n] + z @ L.T
        i += d
    return out


# ===========================================================================
# Section 1 -- the conformance obligations
# ===========================================================================

# --- 1. the approximation ----------------------------------------------------


def test_1_gaussian_approximation_is_project_plus_noise():
    prob = _Problem(targets=(("v", 2),))
    ens = prob.random_ensemble()
    noise = prob.noise()
    approx = gaussian_approximation(ens, noise)
    assert isinstance(approx, EnsembleGaussian)
    assert approx.n_particles == prob.J
    assert approx.block_cov("g") is noise["g"]
    for n in ("u", "v"):
        assert approx.block_cov(n) is None
    x = _stack(ens, prob.names)
    mean, cov = _moments(x)
    dims = [ens.dims[n] for n in prob.names]
    cov[-dims[-1] :, -dims[-1] :] += prob.R["g"]
    got_mean = np.concatenate([np.asarray(approx.mean(n)) for n in prob.names])
    got_cov = np.block(
        [
            [np.asarray(approx.cov(a, b).to_dense()) for b in prob.names]
            for a in prob.names
        ]
    )
    _close(got_mean, mean)
    _close(got_cov, cov)
    # aligned: realizing without the noise term returns the particles
    back = approx.realize_particles(exclude_block_covs=("g",))
    for n in prob.names:
        _close(back[n], ens[n])


def test_1_without_noise_and_weighted():
    prob = _Problem()
    ens = prob.random_ensemble()
    plain = gaussian_approximation(ens)
    assert isinstance(plain, EnsembleGaussian)
    assert all(plain.block_cov(n) is None for n in plain.names)
    assert isinstance(gaussian_approximation(ens, {}), EnsembleGaussian)
    weighted = reweight(ens, jnp.asarray(RNG.normal(size=prob.J)))
    approx = gaussian_approximation(weighted, prob.noise())
    assert type(approx) is Gaussian
    assert approx.block_cov("g") is not None


# --- 2. square-root exactness ------------------------------------------------

SHAPES = [
    pytest.param((), (("g", 6),), id="one-target-one-given"),
    pytest.param((("v", 3),), (("g", 6),), id="two-targets"),
    pytest.param((("v", 2),), (("g", 3), ("h", 4)), id="two-given"),
]


@pytest.mark.parametrize(("targets", "given"), SHAPES)
def test_2_square_root_is_exact_on_exact_moment_particles(targets, given):
    prob = _Problem(targets=targets, given=given)
    ens = prob.exact_ensemble()
    post = update(
        ens, prob.values(), noise=prob.noise(), update_rule=SymmetricSquareRoot()
    )
    assert post.names == prob.target_names
    mean, cov = prob.exact_posterior()
    got_mean, got_cov = _moments(_stack(post, prob.target_names))
    _close(got_mean, mean, 1e4)
    _close(got_cov, cov, 1e4)


def test_2_square_root_with_exact_values():
    prob = _Problem(P=5, given=(("g", 3),), J=12)
    ens = prob.exact_ensemble()
    post = update(ens, prob.values(), update_rule=SymmetricSquareRoot())
    mean, cov = prob.exact_posterior(noisy=False)
    got_mean, got_cov = _moments(_stack(post, ("u",)))
    _close(got_mean, mean, 1e4)
    _close(got_cov, cov, 1e4)
    assert isinstance(
        SymmetricSquareRoot().build(ens, gaussian_approximation(ens), "g"),
        ParticleUpdate,
    )


def test_2_a_target_term_needs_a_key_and_is_drawn():
    prob = _Problem()
    ens = prob.exact_ensemble()
    Q = PSDDiagonal(jnp.full(prob.P, 4.0))
    approx = gaussian_approximation(ens, prob.noise()).add_noise(u=Q)
    step = SymmetricSquareRoot().build(ens, approx, "g")
    assert isinstance(step, SquareRootMap)
    with pytest.raises(ValueError, match="key"):
        step(prob.values())
    a = step(prob.values(), key=jax.random.key(0))
    b = SymmetricSquareRoot().build(ens, gaussian_approximation(ens, prob.noise()), "g")(
        prob.values()
    )
    keys = jax.random.split(jax.random.key(0), 1)
    draw = np.asarray(jax.random.normal(keys[0], (prob.J, prob.P))) * 2.0
    _close(a["u"], np.asarray(b["u"]) + draw, 1e3)


# --- 3. Matheron against dense -----------------------------------------------


@pytest.mark.parametrize(("targets", "given"), SHAPES)
@pytest.mark.parametrize("path", ["aligned", "general"])
def test_3_matheron_equals_the_dense_perturbed_update(targets, given, path):
    prob = _Problem(targets=targets, given=given)
    ens = prob.random_ensemble()
    approx = gaussian_approximation(ens, prob.noise())
    if path == "general":
        approx = _unaligned(approx)
    key = jax.random.key(11)
    step = Matheron().build(ens, approx, prob.given_names)
    assert step.aligned == (path == "aligned")
    got = step(prob.values(), key=key)
    want = _matheron_reference(prob, ens, key)
    for n in prob.target_names:
        _close(got[n], want[n], 1e5)


def test_3_matheron_with_a_target_term_against_dense():
    prob = _Problem(targets=(("v", 2),))
    ens = prob.random_ensemble()
    Qu, Qv = _psd(prob.P), _psd(2)
    approx = gaussian_approximation(ens, prob.noise()).add_noise(
        u=DensePSD(jnp.asarray(Qu)), v=DensePSD(jnp.asarray(Qv))
    )
    key = jax.random.key(3)
    for a in (approx, _unaligned(approx)):
        got = Matheron().build(ens, a, "g")(prob.values(), key=key)
        want = _matheron_reference(prob, ens, key, target_terms={"u": Qu, "v": Qv})
        for n in prob.target_names:
            _close(got[n], want[n], 1e5)


def test_3_the_two_paths_agree_for_the_same_key():
    prob = _Problem(targets=(("v", 2),))
    ens = prob.random_ensemble()
    approx = gaussian_approximation(ens, prob.noise())
    key = jax.random.key(5)
    fast = Matheron().build(ens, approx, "g")(prob.values(), key=key)
    slow = Matheron().build(ens, _unaligned(approx), "g")(prob.values(), key=key)
    for n in prob.target_names:
        _close(fast[n], slow[n], 1e4)


def test_3_matheron_is_unbiased_over_keys():
    """The sample moments average to the fitted conditional's, within errors.

    On exact-moment particles the fitted Gaussian is the joint, so the target
    is the exact conditional. The sample mean's variance over the key is
    ``K R K^T / J`` exactly; the covariance's standard error is estimated
    from the replicates.
    """
    prob = _Problem(J=16)
    ens = prob.exact_ensemble()
    step = Matheron().build(ens, gaussian_approximation(ens, prob.noise()), "g")
    M = 400
    keys = jax.random.split(jax.random.key(2), M)
    xs = np.stack([np.asarray(step(prob.values(), key=k)["u"]) for k in keys])
    means = xs.mean(axis=1)
    covs = np.stack([_moments(x)[1] for x in xs])
    mean, cov = prob.exact_posterior()
    K = _dense_gain(ens, ("u",), ("g",), prob.R["g"])
    se_mean = np.sqrt(np.diag(K @ prob.R["g"] @ K.T) / prob.J / M)
    assert np.all(np.abs(means.mean(axis=0) - mean) < 5 * se_mean)
    se_cov = covs.std(axis=0, ddof=1) / np.sqrt(M)
    assert np.all(np.abs(covs.mean(axis=0) - cov) < 5 * se_cov + 1e-12)
    # the sample mean's spread over keys is the stated variance
    ratio = means.var(axis=0, ddof=1) / (np.diag(K @ prob.R["g"] @ K.T) / prob.J)
    assert np.all(np.abs(ratio - 1) < 0.25)


# --- 4. the one-call form ----------------------------------------------------


def test_4_update_is_the_three_stages_bitwise():
    prob = _Problem(targets=(("v", 2),))
    ens = prob.random_ensemble()
    key = jax.random.key(9)
    for rule, k in ((Matheron(), key), (SymmetricSquareRoot(), None)):
        one = update(ens, prob.values(), noise=prob.noise(), update_rule=rule, key=k)
        approx = gaussian_approximation(ens, prob.noise())
        three = rule.build(ens, approx, ("g",))(prob.values(), key=k)
        assert one.names == three.names == ("u", "v")
        for n in one.names:
            np.testing.assert_array_equal(one[n], three[n])
    # keywords and the mapping are the same call
    a = update(
        ens, g=prob.values()["g"], noise=prob.noise(), update_rule=Matheron(), key=key
    )
    np.testing.assert_array_equal(
        a["u"],
        update(ens, prob.values(), noise=prob.noise(), update_rule=Matheron(), key=key)[
            "u"
        ],
    )


def test_4_the_approximation_hook_receives_the_ensemble_and_a_dict():
    prob = _Problem()
    ens = prob.random_ensemble()
    seen = []

    def hook(e, noise):
        seen.append((e, noise))
        return gaussian_approximation(e, noise)

    update(
        ens,
        prob.values(),
        noise=prob.noise(),
        update_rule=Matheron(),
        approximation=hook,
        key=jax.random.key(0),
    )
    update(
        ens.assign(g=ens["g"][:, :3]),
        g=prob.values()["g"][:3],
        update_rule=SymmetricSquareRoot(),
        approximation=hook,
    )
    assert seen[0][0] is ens
    assert type(seen[0][1]) is dict and set(seen[0][1]) == {"g"}
    assert seen[1][1] == {}


# --- 5. value-free builds ----------------------------------------------------


@pytest.mark.parametrize("which", ["sqrt", "sqrt-exact", "aligned", "general"])
def test_5_one_build_serves_several_values(which):
    prob = _Problem(P=5, given=(("g", 3),), J=12)
    ens = prob.random_ensemble()
    noise = {} if which == "sqrt-exact" else prob.noise()
    approx = gaussian_approximation(ens, noise)
    if which == "general":
        approx = _unaligned(approx)
    rule = SymmetricSquareRoot() if which.startswith("sqrt") else Matheron()
    key = jax.random.key(1)
    step = rule.build(ens, approx, "g")
    for _ in range(2):
        y = jnp.asarray(RNG.normal(size=3))
        got = step(g=y, key=key)
        fresh = rule.build(ens, approx, "g")(g=y, key=key)
        np.testing.assert_array_equal(got["u"], fresh["u"])


# --- 6. inflation and relaxation ---------------------------------------------


@pytest.mark.parametrize("weighted", [False, True])
def test_6_multiplicative_inflation(weighted):
    x = RNG.normal(size=(10, 3)) * 2 + 5
    lw = RNG.normal(size=10) if weighted else None
    ens = Ensemble(
        x=jnp.asarray(x),
        z=jnp.asarray(RNG.normal(size=(10, 1))),
        log_weights=None if lw is None else jnp.asarray(lw),
    )
    out = inflate_multiplicative(ens, 1.3, "x")
    w = (
        np.full(10, 0.1)
        if lw is None
        else np.exp(lw - lw.max()) / np.exp(lw - lw.max()).sum()
    )
    mean = w @ x
    _close(out["x"], mean + 1.3 * (x - mean), 64)
    np.testing.assert_array_equal(out["z"], ens["z"])
    assert out.is_weighted == weighted
    _close(out.mean("x"), ens.mean("x"), 64)
    _close(out.cov("x").to_dense(), 1.69 * np.asarray(ens.cov("x").to_dense()), 256)


def test_6_additive_inflation_matches_its_pinned_definition():
    J = 8
    x = RNG.normal(size=(J, 3))
    z = RNG.normal(size=(J, 2))
    Qx, Qz = _psd(3), _psd(2)
    ens = Ensemble(z=jnp.asarray(z), x=jnp.asarray(x))
    key = jax.random.key(4)
    got = inflate_additive(
        key, ens, {"x": DensePSD(jnp.asarray(Qx))}, z=DensePSD(jnp.asarray(Qz))
    )
    keys = jax.random.split(key, 2)  # z first: the ensemble's block order
    for name, Q, k, base in (("z", Qz, keys[0], z), ("x", Qx, keys[1], x)):
        eps = np.asarray(jax.random.normal(k, (J, Q.shape[0]))) @ np.linalg.cholesky(Q).T
        _close(got[name], base + eps - eps.mean(axis=0), 64)
        _close(got.mean(name), base.mean(axis=0), 64)
    # weighted: centered by the weighted mean, which is preserved
    lw = jnp.asarray(RNG.normal(size=J))
    wens = Ensemble(x=jnp.asarray(x), log_weights=lw)
    wgot = inflate_additive(key, wens, x=DensePSD(jnp.asarray(Qx)))
    _close(wgot.mean("x"), wens.mean("x"), 64)
    assert wgot.is_weighted


def test_6_additive_inflation_grows_the_covariance_by_q_on_average():
    J, M = 6, 3000
    x = RNG.normal(size=(J, 2))
    Q = _psd(2)
    ens = Ensemble(x=jnp.asarray(x))
    f = jax.jit(
        jax.vmap(lambda k: inflate_additive(k, ens, x=DensePSD(jnp.asarray(Q)))["x"])
    )
    xs = np.asarray(f(jax.random.split(jax.random.key(0), M)))
    grow = np.stack([_moments(a)[1] for a in xs]) - _moments(x)[1]
    se = grow.std(axis=0, ddof=1) / np.sqrt(M)
    assert np.all(np.abs(grow.mean(axis=0) - Q) < 5 * se)


def test_6_relaxations_against_their_formulas():
    J = 9
    prior = Ensemble(
        x=jnp.asarray(RNG.normal(size=(J, 3)) * 3), g=jnp.asarray(RNG.normal(size=(J, 2)))
    )
    post = Ensemble(x=jnp.asarray(RNG.normal(size=(J, 3)) + 1))
    a_pr = np.asarray(prior["x"]) - np.asarray(prior["x"]).mean(axis=0)
    a_po = np.asarray(post["x"]) - np.asarray(post["x"]).mean(axis=0)
    m_po = np.asarray(post["x"]).mean(axis=0)
    s_pr = np.sqrt((a_pr**2).sum(axis=0) / (J - 1))
    s_po = np.sqrt((a_po**2).sum(axis=0) / (J - 1))
    alpha = 0.4
    rtps = relax_to_prior_spread(prior, post, alpha)
    _close(rtps["x"], m_po + a_po * (1 + alpha * (s_pr - s_po) / s_po), 64)
    rtpp = relax_to_prior_perturbations(prior, post, alpha)
    _close(rtpp["x"], m_po + (1 - alpha) * a_po + alpha * a_pr, 64)
    # alpha = 1 restores the prior's spread; alpha = 0 changes nothing
    _close(
        np.asarray(relax_to_prior_spread(prior, post, 1.0)["x"]).std(axis=0, ddof=1),
        s_pr,
        64,
    )
    _close(relax_to_prior_perturbations(prior, post, 0.0)["x"], post["x"], 16)
    assert rtps.names == ("x",)


def test_6_rtps_leaves_a_zero_spread_coordinate_unchanged():
    J = 6
    x = RNG.normal(size=(J, 2))
    x[:, 1] = 7.0
    prior = Ensemble(x=jnp.asarray(RNG.normal(size=(J, 2))))
    post = Ensemble(x=jnp.asarray(x))
    out = relax_to_prior_spread(prior, post, 0.5)
    np.testing.assert_array_equal(out["x"][:, 1], np.full(J, 7.0))
    grad = jax.grad(lambda a: jnp.sum(relax_to_prior_spread(prior, post, a)["x"] ** 2))(
        0.5
    )
    assert np.isfinite(grad)
    grad_x = jax.grad(
        lambda p: jnp.sum(relax_to_prior_spread(prior, Ensemble(x=p), 0.5)["x"] ** 2)
    )(jnp.asarray(x))
    assert np.all(np.isfinite(grad_x))


# --- 7. counts ---------------------------------------------------------------


def test_7_whitened_vectors_per_update():
    J, N = 10, 3
    ens = Ensemble(
        u=jnp.asarray(RNG.normal(size=(J, 4))), g=jnp.asarray(RNG.normal(size=(J, N)))
    )
    noise = CountingWhitenPSD.counting(_psd(N))
    y = jnp.asarray(RNG.normal(size=N))
    key = jax.random.key(0)

    def count(call):
        noise.reset()
        call()
        return noise.count

    assert count(lambda: gaussian_approximation(ens, {"g": noise})) == 0
    approx = gaussian_approximation(ens, {"g": noise})
    assert count(lambda: Matheron().build(ens, approx, "g")(g=y, key=key)) == J + 1
    assert (
        count(
            lambda: update(ens, g=y, noise={"g": noise}, update_rule=Matheron(), key=key)
        )
        == J + 1
    )
    assert count(lambda: SymmetricSquareRoot().build(ens, approx, "g")(g=y)) == J + 1
    step = Matheron().build(ens, approx, "g")
    assert count(lambda: step(g=y, key=key)) == 1
    # the general path: k at build, J at call; here k = J + 3 after absorbing
    wide = _unaligned(approx, extra=3)
    assert wide.latent_dim == J + 3
    assert count(lambda: Matheron().build(ens, wide, "g")(g=y, key=key)) == (J + 3) + J
    # with debug checks on, the alignment check whitens nothing
    with debug_checks():
        assert count(lambda: Matheron().build(ens, approx, "g")(g=y, key=key)) == J + 1


def test_7_one_svd_per_build_none_per_call():
    prob = _Problem(P=5, given=(("g", 3),), J=12)
    ens = prob.random_ensemble()
    approx = gaussian_approximation(ens, prob.noise())
    key = jax.random.key(0)
    y = prob.values()
    for name, a, rule in (
        ("aligned", approx, Matheron()),
        ("general", _unaligned(approx), Matheron()),
        ("sqrt", approx, SymmetricSquareRoot()),
    ):
        whole = _count_svd(
            jax.make_jaxpr(lambda a, rule=rule: rule.build(ens, a, "g")(y, key=key)["u"])(
                a
            ).jaxpr
        )
        assert whole == 1, f"{name}: {whole} SVDs per build and call"
        step = rule.build(ens, a, "g")
        per_call = _count_svd(jax.make_jaxpr(lambda s: s(y, key=key)["u"])(step).jaxpr)
        assert per_call == 0, f"{name}: {per_call} SVDs per call"


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


# --- 8. alignment ------------------------------------------------------------


@pytest.mark.parametrize("scale", [1.0, 1e8])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_8_the_alignment_check_passes_on_the_default_approximation(scale, dtype):
    prob = _Problem()
    ens = prob.random_ensemble()
    ens = Ensemble({n: (scale * ens[n] + scale).astype(dtype) for n in ens.names})
    noise = {"g": DensePSD(jnp.asarray(prob.R["g"] * scale**2, dtype))}
    with debug_checks():
        approx = gaussian_approximation(ens, noise)
        for rule in (Matheron(), SymmetricSquareRoot()):
            rule.build(ens, approx, "g")


@pytest.mark.parametrize("rule", [Matheron(), SymmetricSquareRoot()], ids=repr)
def test_8_the_alignment_check_catches_a_misaligned_approximation(rule):
    prob = _Problem()
    ens = prob.random_ensemble()
    other = prob.random_ensemble()
    foreign = gaussian_approximation(other, prob.noise())
    approx = gaussian_approximation(ens, prob.noise())
    inflated = EnsembleGaussian(
        {n: approx.mean(n) for n in approx.names},
        factors={n: approx.factor(n).to_dense() * 1.1 for n in approx.names},
        block_covs={"g": prob.noise()["g"]},
        n_particles=prob.J,
    )
    for bad in (foreign, inflated):
        rule.build(ens, bad, "g")  # outside debug mode: trusted
        with debug_checks(), pytest.raises(ValueError, match="not aligned"):
            rule.build(ens, bad, "g")
        with debug_checks():  # never under a trace
            jax.jit(
                lambda e, a: rule.build(e, a, "g")(
                    {"g": prob.values()["g"]}, key=jax.random.key(0)
                )["u"]
            )(ens, bad)


# --- 9. failed and weighted particles ----------------------------------------


def test_9_a_failed_particle_raises_in_debug_mode_and_is_nan_otherwise():
    prob = _Problem()
    ens = prob.random_ensemble()
    bad = ens.assign(g=ens["g"].at[4, 1].set(jnp.nan))
    approx = gaussian_approximation(ens, prob.noise())
    for call in (
        lambda: update(
            bad,
            prob.values(),
            noise=prob.noise(),
            update_rule=Matheron(),
            key=jax.random.key(0),
        ),
        lambda: Matheron().build(bad, approx, "g"),
        lambda: SymmetricSquareRoot().build(bad, approx, "g"),
    ):
        with debug_checks(), pytest.raises(ValueError, match="block 'g'.*reweight"):
            call()
    out = update(
        bad,
        prob.values(),
        noise=prob.noise(),
        update_rule=Matheron(),
        key=jax.random.key(0),
    )
    assert bool(jnp.all(jnp.isnan(out["u"])))


def test_9_the_documented_remedy_updates_cleanly():
    prob = _Problem()
    ens = prob.random_ensemble()
    bad = ens.assign(g=ens["g"].at[4, 1].set(jnp.inf))
    kept = resample(
        jax.random.key(1), reweight(bad, jnp.where(bad.all_finite, 0.0, -jnp.inf))
    )
    with debug_checks():
        out = update(
            kept,
            prob.values(),
            noise=prob.noise(),
            update_rule=Matheron(),
            key=jax.random.key(0),
        )
    assert bool(jnp.all(jnp.isfinite(out["u"])))


def test_9_a_weighted_ensemble_is_refused_with_the_resample_message():
    prob = _Problem()
    ens = reweight(prob.random_ensemble(), jnp.zeros(prob.J))
    approx = gaussian_approximation(prob.random_ensemble(), prob.noise())
    for call in (
        lambda: update(
            ens,
            prob.values(),
            noise=prob.noise(),
            update_rule=Matheron(),
            key=jax.random.key(0),
        ),
        lambda: Matheron().build(ens, approx, "g"),
        lambda: SymmetricSquareRoot().build(ens, approx, "g"),
        lambda: relax_to_prior_spread(ens, ens, 0.5),
    ):
        with pytest.raises(ValueError, match="(?i)resample|unweighted"):
            call()


# --- 10. validation ----------------------------------------------------------


def test_10_build_argument_rules():
    prob = _Problem(targets=(("v", 2),))
    ens = prob.random_ensemble()
    approx = gaussian_approximation(ens, prob.noise())
    for rule in (Matheron(), SymmetricSquareRoot()):
        with pytest.raises(TypeError, match="Ensemble"):
            rule.build(ens["u"], approx, "g")
        with pytest.raises(TypeError, match="Gaussian"):
            rule.build(ens, ens, "g")
        with pytest.raises(KeyError, match="nope"):
            rule.build(ens, approx, "nope")
        with pytest.raises(ValueError, match="more than once"):
            rule.build(ens, approx, ("g", "g"))
        with pytest.raises(ValueError, match="at least one given"):
            rule.build(ens, approx, ())
        with pytest.raises(ValueError, match="blocks"):
            rule.build(ens.drop("v"), approx, "g")
        only_given = ens.marginal("g")
        with pytest.raises(ValueError, match="target"):
            rule.build(only_given, gaussian_approximation(only_given, prob.noise()), "g")
        with pytest.raises(TypeError, match="dtype"):
            rule.build(
                Ensemble({n: ens[n].astype(jnp.float32) for n in ens.names}), approx, "g"
            )
    # the rules' own conditions
    exact = gaussian_approximation(ens)
    with pytest.raises(ValueError, match="no independent term"):
        Matheron().build(ens, exact, "g")
    with pytest.raises(ValueError, match="EnsembleGaussian"):
        SymmetricSquareRoot().build(ens, _unaligned(approx), "g")
    with pytest.raises(ValueError, match="EnsembleGaussian"):
        SymmetricSquareRoot().build(
            ens,
            gaussian_approximation(
                Ensemble({n: ens[n][:-1] for n in ens.names}), prob.noise()
            ),
            "g",
        )
    mixed = gaussian_approximation(ens, {"v": PSDDiagonal(jnp.ones(2))})
    with pytest.raises(ValueError, match="mix"):
        SymmetricSquareRoot().build(ens, mixed, ("v", "g"))
    small = Ensemble({n: ens[n][:6] for n in ens.names})
    with pytest.raises(ValueError, match="more than 6 particles"):
        SymmetricSquareRoot().build(small, gaussian_approximation(small), "g")
    # a missing key at call
    with pytest.raises(ValueError, match="key"):
        Matheron().build(ens, approx, "g")(prob.values())
    with pytest.raises(TypeError, match="typed key"):
        Matheron().build(ens, approx, "g")(prob.values(), key=jax.random.PRNGKey(0))


def test_10_update_argument_rules():
    prob = _Problem()
    ens = prob.random_ensemble()
    y = prob.values()
    kw = dict(noise=prob.noise(), key=jax.random.key(0))
    with pytest.raises(TypeError, match="build"):
        update(ens, y, update_rule=kalman.update, **kw)
    with pytest.raises(ValueError, match="at least one"):
        update(ens, update_rule=Matheron(), **kw)
    with pytest.raises(KeyError, match="nope"):
        update(ens, nope=y["g"], update_rule=Matheron(), **kw)
    with pytest.raises(TypeError, match="twice"):
        update(ens, y, g=y["g"], update_rule=Matheron(), **kw)
    with pytest.raises(KeyError, match="zzz"):
        update(ens, y, update_rule=Matheron(), noise={"zzz": prob.noise()["g"]})
    with pytest.raises(ValueError, match="not a given block"):
        update(ens, y, update_rule=Matheron(), noise={"u": PSDDiagonal(jnp.ones(prob.P))})
    with pytest.raises(TypeError, match="callable"):
        update(ens, y, update_rule=Matheron(), approximation=3, **kw)
    with pytest.raises(TypeError, match="Gaussian"):
        update(ens, y, update_rule=Matheron(), approximation=lambda e, n: e, **kw)
    with pytest.raises(ValueError, match="target"):
        update(ens.marginal("g"), y, update_rule=Matheron(), **kw)
    with pytest.raises(TypeError, match="Ensemble"):
        update(ens["g"], y, update_rule=Matheron(), **kw)


def test_10_update_checks_its_rules_result():
    prob = _Problem(targets=(("v", 2),))
    ens = prob.random_ensemble()

    class Bad:
        def __init__(self, change):
            self.change = change

        def build(self, particles, approximation, given):
            inner = SymmetricSquareRoot().build(particles, approximation, given)
            return lambda values=None, /, *, key=None, **kw: self.change(
                inner(values, **kw)
            )

    cases = [
        (lambda out: out["u"], TypeError, "not an Ensemble"),
        (lambda out: out.marginal("u"), ValueError, "target blocks"),
        (lambda out: reweight(out, jnp.zeros(prob.J)), ValueError, "weighted"),
        (
            lambda out: Ensemble({n: out[n][:-1] for n in out.names}),
            ValueError,
            "particles",
        ),
        (
            lambda out: Ensemble({n: out[n].astype(jnp.float32) for n in out.names}),
            TypeError,
            "dtype",
        ),
    ]
    for change, error, match in cases:
        with pytest.raises(error, match=match):
            update(ens, prob.values(), noise=prob.noise(), update_rule=Bad(change))


def test_10_the_build_check_order_with_two_simultaneous_violations():
    """One violation at a time pins no order; two at once do."""
    prob = _Problem()
    ens = prob.random_ensemble()
    approx = gaussian_approximation(ens, prob.noise())
    weighted = reweight(ens, jnp.zeros(prob.J))
    # unknown name before weights
    with pytest.raises(KeyError):
        Matheron().build(weighted, approx, "nope")
    # weights before block agreement
    with pytest.raises(ValueError, match="weighted"):
        Matheron().build(weighted.drop("u").assign(w=ens["u"]), approx, "g")
    # structural condition before capabilities
    no_whiten = gaussian_approximation(
        ens, {"g": FactorOnlyPSD(jnp.linalg.cholesky(jnp.asarray(prob.R["g"])))}
    )
    with pytest.raises(UnsupportedOpError):
        Matheron().build(ens, no_whiten, "g")
    with pytest.raises(ValueError, match="EnsembleGaussian"):
        SymmetricSquareRoot().build(ens, _unaligned(no_whiten), "g")
    # capabilities before tier 4
    bad = ens.assign(g=ens["g"].at[0, 0].set(jnp.nan))
    with debug_checks(), pytest.raises(UnsupportedOpError):
        Matheron().build(bad, no_whiten, "g")
    # family guard before everything
    family = jax.tree_util.tree_map(lambda leaf: jnp.stack([leaf] * 2), ens)
    with pytest.raises(ValueError, match="vmapped family"):
        Matheron().build(family, approx, "nope")


def test_10_inflation_argument_rules():
    ens = Ensemble(x=jnp.asarray(RNG.normal(size=(5, 3))))
    with pytest.raises(KeyError):
        inflate_multiplicative(ens, 1.1, "y")
    with pytest.raises(ValueError, match="more than once"):
        inflate_multiplicative(ens, 1.1, ("x", "x"))
    with pytest.raises(ValueError, match="scalar"):
        inflate_multiplicative(ens, jnp.ones(3))
    with debug_checks(), pytest.raises(ValueError, match="positive"):
        inflate_multiplicative(ens, -1.0)
    with pytest.raises(ValueError, match="at least one"):
        inflate_additive(jax.random.key(0), ens)
    with pytest.raises(TypeError, match="PSDLinOp"):
        inflate_additive(jax.random.key(0), ens, x=jnp.eye(3))
    with pytest.raises(ValueError, match="side"):
        inflate_additive(jax.random.key(0), ens, x=PSDDiagonal(jnp.ones(2)))
    with pytest.raises(ValueError, match="key"):
        inflate_additive(None, ens, x=PSDDiagonal(jnp.ones(3)))
    with pytest.raises(TypeError, match="typed key"):
        inflate_additive(jax.random.PRNGKey(0), ens, x=PSDDiagonal(jnp.ones(3)))
    other = Ensemble(x=jnp.asarray(RNG.normal(size=(6, 3))))
    with pytest.raises(ValueError, match="particles"):
        relax_to_prior_spread(other, ens, 0.5)
    with pytest.raises(KeyError, match="prior"):
        relax_to_prior_perturbations(Ensemble(z=ens["x"]), ens, 0.5)
    with pytest.raises(ValueError, match="dimension"):
        relax_to_prior_perturbations(Ensemble(x=ens["x"][:, :2]), ens, 0.5)
    with pytest.raises(TypeError, match="dtype"):
        relax_to_prior_perturbations(Ensemble(x=ens["x"].astype(jnp.float32)), ens, 0.5)
    with pytest.raises(ValueError, match="scalar"):
        relax_to_prior_spread(ens, ens, jnp.ones(3))
    with debug_checks(), pytest.raises(ValueError, match=r"\[0, 1\]"):
        relax_to_prior_spread(ens, ens, 1.5)
    bad = ens.assign(x=ens["x"].at[0, 0].set(jnp.nan))
    for call in (
        lambda: inflate_multiplicative(bad, 1.1),
        lambda: inflate_additive(jax.random.key(0), bad, x=PSDDiagonal(jnp.ones(3))),
        lambda: relax_to_prior_spread(ens, bad, 0.5),
    ):
        with debug_checks(), pytest.raises(ValueError, match="not finite"):
            call()


# --- 11. capabilities --------------------------------------------------------


def test_11_missing_capabilities_raise_before_any_work():
    prob = _Problem()
    ens = prob.random_ensemble()
    L = jnp.linalg.cholesky(jnp.asarray(prob.R["g"]))
    no_whiten = gaussian_approximation(ens, {"g": FactorOnlyPSD(L)})
    for rule in (Matheron(), SymmetricSquareRoot()):
        with pytest.raises(UnsupportedOpError, match="whiten"):
            rule.build(ens, no_whiten, "g")
    no_factor = gaussian_approximation(ens, prob.noise()).add_noise(
        u=WhitenOnlyPSD(jnp.eye(prob.P))
    )
    for rule in (Matheron(), SymmetricSquareRoot()):
        with pytest.raises(UnsupportedOpError, match="factor"):
            rule.build(ens, no_factor, "g")
    with pytest.raises(UnsupportedOpError, match="factor"):
        inflate_additive(jax.random.key(0), ens, u=WhitenOnlyPSD(jnp.eye(prob.P)))


# --- 12. diagnostics ---------------------------------------------------------


def test_12_a_singular_noise_covariance_is_reported_through_every_rule():
    prob = _Problem()
    ens = prob.random_ensemble()
    with debug_checks(False):
        singular = PSDDiagonal(jnp.asarray([1.0, 0.0, 1.0, 1.0, 1.0, 1.0]))
    for rule, key in ((Matheron(), jax.random.key(0)), (SymmetricSquareRoot(), None)):
        with debug_checks(), pytest.raises(ValueError, match="given block 'g'"):
            update(ens, prob.values(), noise={"g": singular}, update_rule=rule, key=key)
        out = update(ens, prob.values(), noise={"g": singular}, update_rule=rule, key=key)
        assert not bool(jnp.all(jnp.isfinite(out["u"])))


# --- 13. derivatives ---------------------------------------------------------


def _fd(f, x, h=1e-6):
    x = np.asarray(x, np.float64)
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(jnp.asarray(x + e)) - f(jnp.asarray(x - e))) / (2 * h)
    return g


@pytest.mark.parametrize("rule", [Matheron(), SymmetricSquareRoot()], ids=repr)
def test_13_first_derivatives_agree_with_finite_differences(rule):
    prob = _Problem(P=3, given=(("g", 2),), J=8)
    ens = prob.random_ensemble()
    key = jax.random.key(0)
    R = jnp.asarray(prob.R["g"])

    def by_value(y):
        return jnp.sum(
            update(ens, g=y, noise={"g": DensePSD(R)}, update_rule=rule, key=key)["u"]
            ** 2
        )

    def by_scale(s):
        return jnp.sum(
            update(
                ens,
                prob.values(),
                noise={"g": DensePSD(R) * s},
                update_rule=rule,
                key=key,
            )["u"]
            ** 2
        )

    def by_particles(u):
        e = ens.assign(u=u)
        return jnp.sum(
            update(e, prob.values(), noise={"g": DensePSD(R)}, update_rule=rule, key=key)[
                "u"
            ]
            ** 2
        )

    y = prob.values()["g"]
    np.testing.assert_allclose(
        jax.grad(by_value)(y), _fd(by_value, y), rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(by_scale))(1.3), _fd(by_scale, 1.3), rtol=1e-5
    )
    u = ens["u"]
    np.testing.assert_allclose(
        jax.grad(by_particles)(u), _fd(by_particles, u), rtol=1e-5, atol=1e-7
    )


@pytest.mark.parametrize("rule", [Matheron(), SymmetricSquareRoot()], ids=repr)
def test_13_derivatives_are_finite_at_a_collapsed_ensemble(rule):
    J, N = 6, 3
    ens = Ensemble(u=jnp.ones((J, 2)), g=jnp.full((J, N), 2.0))
    key = jax.random.key(0)

    def f(s):
        return jnp.sum(
            update(
                ens,
                g=jnp.zeros(N),
                noise={"g": PSDDiagonal(jnp.ones(N)) * s},
                update_rule=rule,
                key=key,
            )["u"]
        )

    for grad in (jax.grad(f)(1.0), jax.jvp(f, (1.0,), (1.0,))[1]):
        assert np.isfinite(grad)


# --- 14. JAX -----------------------------------------------------------------


def test_14_rules_and_built_updates_round_trip_through_flatten():
    prob = _Problem(P=5, given=(("g", 3),), J=12)
    ens = prob.random_ensemble()
    approx = gaussian_approximation(ens, prob.noise())
    key = jax.random.key(0)
    built = [
        Matheron().build(ens, approx, "g"),
        Matheron().build(ens, _unaligned(approx), "g"),
        SymmetricSquareRoot().build(ens, approx, "g"),
        SymmetricSquareRoot().build(ens, gaussian_approximation(ens), "g"),
    ]
    for obj in built + [Matheron(), SymmetricSquareRoot()]:
        leaves, tree = jax.tree_util.tree_flatten(obj)
        back = jax.tree_util.tree_unflatten(tree, leaves)
        assert type(back) is type(obj)
    for step in built:
        back = jax.tree_util.tree_unflatten(*reversed(jax.tree_util.tree_flatten(step)))
        np.testing.assert_array_equal(
            back(prob.values(), key=key)["u"], step(prob.values(), key=key)["u"]
        )
    # a built update crosses jit as an argument, and so does a rule
    step = built[0]
    out = jax.jit(lambda s, y: s(g=y, key=key)["u"])(step, prob.values()["g"])
    _close(out, step(prob.values(), key=key)["u"], 1e4)
    jax.jit(lambda r, e, a: r.build(e, a, "g")(prob.values(), key=key)["u"])(
        Matheron(), ens, approx
    )


@pytest.mark.parametrize("rule", [Matheron(), SymmetricSquareRoot()], ids=repr)
def test_14_jit_and_vmap_over_values_agree_with_a_loop(rule):
    prob = _Problem()
    ens = prob.random_ensemble()
    key = jax.random.key(0)
    noise = prob.noise()
    ys = jnp.asarray(RNG.normal(size=(3, 6)))

    def run(y):
        return update(ens, g=y, noise=noise, update_rule=rule, key=key)["u"]

    batched = jax.jit(jax.vmap(run))(ys)
    for i in range(3):
        _close(batched[i], run(ys[i]), 1e4)


def test_14_a_vmapped_family_is_refused():
    prob = _Problem()
    ens = prob.random_ensemble()
    family = jax.tree_util.tree_map(lambda leaf: jnp.stack([leaf] * 2), ens)
    step = Matheron().build(ens, gaussian_approximation(ens, prob.noise()), "g")
    steps = jax.tree_util.tree_map(lambda leaf: jnp.stack([leaf] * 2), step)
    for call in (
        lambda: update(
            family,
            prob.values(),
            noise=prob.noise(),
            update_rule=Matheron(),
            key=jax.random.key(0),
        ),
        lambda: steps(prob.values(), key=jax.random.key(0)),
        lambda: inflate_multiplicative(family, 1.1),
        lambda: relax_to_prior_spread(family, family, 0.5),
    ):
        with pytest.raises(ValueError, match="vmapped family"):
            call()
    assert repr(steps).startswith("vmapped(MatheronUpdate(")


# --- 15. dtypes --------------------------------------------------------------


def test_15_float32_stays_float32():
    prob = _Problem()
    ens = prob.random_ensemble()
    ens = Ensemble({n: ens[n].astype(jnp.float32) for n in ens.names})
    noise = {"g": DensePSD(jnp.asarray(prob.R["g"], jnp.float32))}
    y = {"g": jnp.asarray(prob.y["g"], jnp.float32)}
    key = jax.random.key(0)
    outs = [
        update(ens, y, noise=noise, update_rule=Matheron(), key=key),
        update(ens, y, noise=noise, update_rule=SymmetricSquareRoot()),
        Matheron().build(ens, _unaligned(gaussian_approximation(ens, noise)), "g")(
            y, key=key
        ),
        inflate_multiplicative(ens, 1.05),
        inflate_additive(key, ens, u=PSDDiagonal(jnp.ones(prob.P, jnp.float32))),
        relax_to_prior_spread(ens, ens.marginal("u"), 0.5),
        relax_to_prior_perturbations(ens, ens.marginal("u"), 0.5),
    ]
    for out in outs:
        for n in out.names:
            assert out[n].dtype == jnp.float32, (out, n)


# --- 16. reproducibility -----------------------------------------------------


def test_16_same_key_same_output_different_keys_differ():
    prob = _Problem()
    ens = prob.random_ensemble()
    approx = gaussian_approximation(ens, prob.noise())
    step = Matheron().build(ens, approx, "g")
    for fn in (
        lambda k: step(prob.values(), key=k)["u"],
        lambda k: inflate_additive(k, ens, u=PSDDiagonal(jnp.ones(prob.P)))["u"],
    ):
        a, b, other = fn(jax.random.key(1)), fn(jax.random.key(1)), fn(jax.random.key(2))
        np.testing.assert_array_equal(a, b)
        assert np.abs(np.asarray(a) - np.asarray(other)).max() > 1e-3


def test_16_the_pinned_draws_are_snapshotted():
    """The composed draws of the two stochastic functions, on fixed inputs.

    Captured under the JAX of ``uv.lock`` with x64 enabled and default PRNG
    settings. A failure means every stochastic output of this layer changed:
    check JAX's version and PRNG flags before suspecting EnsKit. The
    definitions are checked elementwise in obligations 3 and 6.
    """
    ens = Ensemble(
        u=jnp.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]]),
        g=jnp.asarray([[0.5], [-0.5], [1.0]]),
    )
    approx = gaussian_approximation(ens, {"g": PSDDiagonal(jnp.ones(1))}).add_noise(
        u=PSDDiagonal(jnp.ones(2))
    )
    out = Matheron().build(ens, approx, "g")(g=jnp.asarray([0.25]), key=jax.random.key(0))
    np.testing.assert_allclose(np.asarray(out["u"]), MATHERON_SNAPSHOT, rtol=1e-12)
    add = inflate_additive(jax.random.key(0), ens, u=PSDDiagonal(jnp.ones(2)))
    np.testing.assert_allclose(np.asarray(add["u"]), ADDITIVE_SNAPSHOT, rtol=1e-12)


MATHERON_SNAPSHOT = [
    [-0.816732895041071, 2.4689244868849594],
    [2.416760697148697, 0.7417469669152286],
    [2.343602644408292, 3.144207808981837],
]
ADDITIVE_SNAPSHOT = [
    [1.3066565353954127, 0.008757836082834425],
    [0.8420838721417094, 1.8718128879917053],
    [0.8512595924628776, 1.1194292759254605],
]


# --- 17. repr ----------------------------------------------------------------


def test_17_reprs():
    prob = _Problem(P=5, given=(("g", 3),), J=12)
    ens = prob.random_ensemble()
    approx = gaussian_approximation(ens, prob.noise())
    assert repr(Matheron()) == "Matheron()"
    assert repr(SymmetricSquareRoot()) == "SymmetricSquareRoot()"
    assert repr(Matheron().build(ens, approx, "g")) == (
        "MatheronUpdate(given=('g',), targets=('u',), n_particles=12, aligned=True)"
    )
    assert repr(Matheron().build(ens, _unaligned(approx), "g")).endswith("aligned=False)")
    assert repr(SymmetricSquareRoot().build(ens, gaussian_approximation(ens), "g")) == (
        "ExactSquareRootUpdate(given=('g',), targets=('u',), n_particles=12)"
    )


# --- 19. examples ------------------------------------------------------------


def test_19_example_3_one_update():
    """Example 3 of the design: one update, one call, against the exact answer."""
    prob = _Problem(P=4, given=(("g", 6),), J=32)
    ens = prob.exact_ensemble()
    sqrt_post = update(
        ens, g=prob.values()["g"], noise=prob.noise(), update_rule=SymmetricSquareRoot()
    )
    pert_post = update(
        ens,
        g=prob.values()["g"],
        noise=prob.noise(),
        update_rule=Matheron(),
        key=jax.random.key(1),
    )
    exact = prob.joint.add_noise(g=prob.noise()["g"]).condition(g=prob.values()["g"])
    assert repr(sqrt_post) == "Ensemble(n_particles=32, blocks={'u': 4}, weighted=False)"
    assert jnp.allclose(sqrt_post.mean("u"), exact.mean("u"), atol=1e-12)
    assert jnp.allclose(
        sqrt_post.cov("u").to_dense(), exact.cov("u").to_dense(), atol=1e-12
    )
    assert jnp.allclose(pert_post.mean("u"), exact.mean("u"), atol=0.3)


class DEnKF:
    """Example 14's rule: full gain on the mean, half gain on the anomalies.

    Particle j moves by K (y* - g_bar - (g_j - g_bar) / 2): the approximation's
    Matheron map, applied with no noise to particles whose given blocks are
    pulled halfway to their mean (Sakov & Oke, 2008).
    """

    def build(self, particles, approximation, given):
        cmap = approximation.conditional_map(given)
        halfway = particles.assign(
            {c: particles.mean(c) + 0.5 * particles.anomalies(c) for c in given}
        )

        def call(values=None, /, *, key=None, **block_values):
            return cmap(halfway, values, **block_values).marginal(*cmap.targets)

        return call


def test_19_example_14_a_user_rule():
    prob = _Problem(P=4, given=(("g", 6),), J=32)
    ens = prob.exact_ensemble()
    assert isinstance(DEnKF(), UpdateRule)
    post = update(ens, g=prob.values()["g"], noise=prob.noise(), update_rule=DEnKF())
    exact = update(
        ens, g=prob.values()["g"], noise=prob.noise(), update_rule=SymmetricSquareRoot()
    )
    assert jnp.allclose(post.mean("u"), exact.mean("u"), atol=1e-12)
    assert jnp.all(
        jnp.diag(post.cov("u").to_dense()) >= jnp.diag(exact.cov("u").to_dense())
    )


# ===========================================================================
# Section 2 -- regression tests ported from tests/test_eki.py
# ===========================================================================


def test_regression_inflation_scales_the_anomalies_not_the_covariance():
    """``anomaly_scale`` is the argument's name because the two conventions differ.

    A caller passing an intended *variance* inflation of 1.2 gets 1.44. The
    error is invisible at the small values normally used, where r and
    sqrt(gamma) barely differ, and severe at large ones, which is why the
    name pins the convention and this test pins the name.
    """
    ens = Ensemble(x=jnp.asarray(RNG.normal(size=(12, 3))))
    before = np.asarray(ens.cov("x").to_dense())
    after = np.asarray(inflate_multiplicative(ens, anomaly_scale=1.2).cov("x").to_dense())
    assert np.abs(after - 1.44 * before).max() < 64 * EPS * np.abs(before).max()
    assert np.abs(after - 1.2 * before).max() > 1e-3 * np.abs(before).max()


def test_regression_a_vmapped_inflation_refuses_rather_than_broadcasting():
    """A scale whose length equals the dimension would otherwise inflate per coordinate.

    Shape-correct, exception-free, and wrong: each coordinate gets a
    different factor. The old policy refused a vmapped family of itself; the
    functions here refuse a non-scalar scale, and a vmapped ensemble.
    """
    ens = Ensemble(x=jnp.arange(18.0).reshape(6, 3))
    with pytest.raises(ValueError, match="scalar"):
        inflate_multiplicative(ens, jnp.asarray([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="scalar"):
        relax_to_prior_perturbations(ens, ens, jnp.asarray([0.1, 0.2, 0.3]))
    family = jax.tree_util.tree_map(lambda leaf: jnp.stack([leaf] * 3), ens)
    with pytest.raises(ValueError, match="vmapped family"):
        inflate_multiplicative(family, 1.1)
    # a family of scales is a vmap over the call
    out = jax.vmap(lambda s: inflate_multiplicative(ens, s)["x"])(jnp.asarray([1.0, 2.0]))
    _close(out[1], inflate_multiplicative(ens, 2.0)["x"], 16)


def test_regression_a_float32_update_cannot_quietly_demote_a_run():
    """Every downstream computation would still pass at its own tolerance.

    An update's result becomes the next state of an algorithm, so a rule
    returning float32 particles from float64 ones would demote every later
    update. ``update`` refuses it, naming the rule.
    """
    prob = _Problem()
    ens = prob.random_ensemble()

    class Demoting:
        def build(self, particles, approximation, given):
            inner = SymmetricSquareRoot().build(particles, approximation, given)

            def call(values=None, /, *, key=None, **kw):
                out = inner(values, **kw)
                return Ensemble({n: out[n].astype(jnp.float32) for n in out.names})

            return call

    with pytest.raises(TypeError, match="Demoting.*float32"):
        update(ens, prob.values(), noise=prob.noise(), update_rule=Demoting())


def test_regression_multiplicative_stays_in_the_span_and_additive_leaves_it():
    """The executable form of the subspace property, in both directions.

    Five coordinates with rank-3 anomalies, so the span is a proper subspace
    and "leaving it" is a statement with content. An update and multiplicative
    inflation recombine anomalies; additive inflation adds new directions.
    """
    J = 4
    base = RNG.normal(size=(J, 3)) @ RNG.normal(size=(3, 5))
    ens = Ensemble(x=jnp.asarray(base), g=jnp.asarray(RNG.normal(size=(J, 2))))
    anomalies = base - base.mean(axis=0)
    basis = np.linalg.svd(anomalies.T, full_matrices=False)[0][:, :3]

    def leaves(x):
        moved = np.asarray(x) - base.mean(axis=0)
        return np.abs(moved - moved @ basis @ basis.T).max()

    post = update(
        ens,
        g=jnp.zeros(2),
        noise={"g": PSDDiagonal(jnp.ones(2))},
        update_rule=SymmetricSquareRoot(),
    )
    assert leaves(post["x"]) < 1e-9
    assert leaves(inflate_multiplicative(ens, 1.05)["x"]) < 1e-9
    assert (
        leaves(
            inflate_additive(jax.random.key(0), ens, x=PSDDiagonal(jnp.full(5, 0.05)))[
                "x"
            ]
        )
        > 1e-3
    )


# ===========================================================================
# Section 3 -- new regression tests
# ===========================================================================


def test_regression_matheron_draws_the_target_term():
    """The design review's prototype dropped it: a variance of 0.044 against 4.04.

    ``MatheronMap`` passes a target's independent term through unsampled, so
    the stochastic rule must draw it. Without the draw the updated spread
    omits ``D_x`` entirely, and the sample mean still looks right.
    """
    prob = _Problem(J=200)
    ens = prob.exact_ensemble()
    Q = PSDDiagonal(jnp.full(prob.P, 4.0))
    approx = gaussian_approximation(ens, prob.noise()).add_noise(u=Q)
    exact_var = np.diag(np.asarray(approx.condition(prob.values()).cov("u").to_dense()))
    assert np.all(exact_var > 4.0)
    step = Matheron().build(ens, approx, "g")
    var = np.mean(
        [
            np.var(
                np.asarray(step(prob.values(), key=jax.random.key(s))["u"]),
                axis=0,
                ddof=1,
            )
            for s in range(20)
        ],
        axis=0,
    )
    np.testing.assert_allclose(var, exact_var, rtol=0.1)


def test_regression_the_aligned_and_general_paths_agree():
    """Only a count tells them apart, so their agreement is tested directly.

    The review's check 1: the same joint, alignment declared and not, the
    same key.
    """
    prob = _Problem(J=200)
    ens = prob.exact_ensemble()
    approx = gaussian_approximation(ens, prob.noise())
    key = jax.random.key(5)
    fast = Matheron().build(ens, approx, "g")(prob.values(), key=key)
    general = Matheron().build(ens, _unaligned(approx), "g")(prob.values(), key=key)
    assert np.abs(np.asarray(fast["u"]) - np.asarray(general["u"])).max() < 1e-10


def test_regression_a_python_float_scale_does_not_promote_float32():
    """``jnp.asarray(1.02)`` is a strong float64 under x64 and would promote."""
    ens = Ensemble(x=jnp.asarray(RNG.normal(size=(5, 2)), jnp.float32))
    assert inflate_multiplicative(ens, 1.02)["x"].dtype == jnp.float32
    assert inflate_multiplicative(ens, jnp.asarray(1.02))["x"].dtype == jnp.float32
    assert relax_to_prior_spread(ens, ens, jnp.asarray(0.5))["x"].dtype == jnp.float32


def test_regression_the_alignment_check_catches_foreign_particles():
    """An ``EnsembleGaussian`` of the right count from other particles.

    Without the check, ``SymmetricSquareRoot`` returns the other particles'
    update and ``Matheron`` reads the other particles' residuals, each with
    plausible statistics and nothing raised.
    """
    prob = _Problem()
    ens, other = prob.random_ensemble(), prob.random_ensemble()
    foreign = gaussian_approximation(other, prob.noise())
    silent = SymmetricSquareRoot().build(ens, foreign, "g")(prob.values())
    np.testing.assert_allclose(
        silent["u"],
        SymmetricSquareRoot().build(other, foreign, "g")(prob.values())["u"],
    )
    with debug_checks(), pytest.raises(ValueError, match="block 'u'"):
        SymmetricSquareRoot().build(ens, foreign, "g")


def test_regression_a_term_of_another_dtype_cannot_mix_the_result():
    """A float64 term on float32 particles gave a float32 and a float64 block.

    The adversarial review's finding: ``add_noise`` accepts the term,
    operators carry no dtype (#33), and the result is assembled without the
    ``Ensemble`` constructor, so nothing raised until a later call refused the
    mixed ensemble far from the cause. ``Matheron`` and ``update`` now refuse
    it, naming the block.
    """
    prob = _Problem(targets=(("v", 2),))
    ens = prob.random_ensemble()
    ens = Ensemble({n: ens[n].astype(jnp.float32) for n in ens.names})
    noise = {"g": DensePSD(jnp.asarray(prob.R["g"], jnp.float32))}
    y = {"g": jnp.asarray(prob.y["g"], jnp.float32)}

    def mixed(e, n):
        return gaussian_approximation(e, n).add_noise(v=PSDDiagonal(jnp.ones(2)))

    key = jax.random.key(0)
    with pytest.raises(TypeError, match="block 'v'"):
        Matheron().build(ens, mixed(ens, noise), "g")(y, key=key)
    for rule in (Matheron(), SymmetricSquareRoot()):
        with pytest.raises(TypeError, match="block 'v'.*float64"):
            update(ens, y, noise=noise, update_rule=rule, approximation=mixed, key=key)
    with pytest.raises(TypeError, match="dtype"):
        inflate_additive(key, ens, u=PSDDiagonal(jnp.ones(prob.P)))


def test_regression_update_checks_the_result_blocks_exactly():
    """Order and dimensions too, not only the set of names (the review's finding)."""
    prob = _Problem(targets=(("v", 2),))
    ens = prob.random_ensemble()

    class Changes:
        def __init__(self, change):
            self.change = change

        def build(self, particles, approximation, given):
            inner = SymmetricSquareRoot().build(particles, approximation, given)
            return lambda values=None, /, *, key=None, **kw: self.change(inner(values))

    with pytest.raises(ValueError, match="order"):
        update(
            ens,
            prob.values(),
            noise=prob.noise(),
            update_rule=Changes(lambda o: o.marginal("v", "u")),
        )
    with pytest.raises(ValueError, match="dimension"):
        update(
            ens,
            prob.values(),
            noise=prob.noise(),
            update_rule=Changes(lambda o: Ensemble(u=o["u"][:, :2], v=o["v"])),
        )
    with pytest.raises(TypeError, match="PSDLinOp"):
        update(
            ens,
            prob.values(),
            noise={"g": jnp.eye(6)},
            update_rule=Matheron(),
            approximation=lambda e, n: gaussian_approximation(e, prob.noise()),
            key=jax.random.key(0),
        )


def test_regression_a_zero_weight_failed_particle_passes_the_inflation_checks():
    """The functions that accept weighted ensembles skip particles of weight zero."""
    x = RNG.normal(size=(6, 2))
    x[2, 0] = np.nan
    ens = Ensemble(x=jnp.asarray(x), log_weights=jnp.asarray([0, 0, -np.inf, 0, 0, 0.0]))
    with debug_checks():
        out = inflate_multiplicative(ens, 1.1)
        inflate_additive(jax.random.key(0), ens, x=PSDDiagonal(jnp.ones(2)))
    assert bool(jnp.all(jnp.isfinite(out.mean("x"))))
    unweighted = Ensemble(x=jnp.asarray(x))
    with debug_checks(), pytest.raises(ValueError, match="not finite"):
        inflate_multiplicative(unweighted, 1.1)


def test_regression_alignment_admits_a_linear_map_of_an_aligned_block():
    """The review's repro: float32, an offset of 1e5, a differencing map.

    The derived block's round-off scales with the source block's magnitude,
    not its own, so a bound built from the block alone rejected a correctly
    aligned approximation (gap 6.1e-3 against 2.5e-3).
    """
    J, d = 32, 20
    u = (1e5 + RNG.normal(size=(J, d))).astype(np.float32)
    H = (np.eye(d) - np.eye(d, k=1))[:-1].astype(np.float32)  # pure differences
    ens = Ensemble(u=jnp.asarray(u), g=jnp.asarray(u) @ jnp.asarray(H).T)
    fit = ens.marginal("u").project()
    Hj = jnp.asarray(H)
    F = fit.factor("u").to_dense()
    approx = EnsembleGaussian(
        {"u": fit.mean("u"), "g": Hj @ fit.mean("u")},
        factors={"u": F, "g": Hj @ F},
        block_covs={"g": PSDDiagonal(jnp.ones(d - 1, jnp.float32))},
        n_particles=J,
    )
    with debug_checks():
        Matheron().build(ens, approx, "g")
        SymmetricSquareRoot().build(ens, approx, "g")


def test_inflate_additive_checks_the_covariances_before_the_key():
    ens = Ensemble(x=jnp.asarray(RNG.normal(size=(5, 3))))
    with pytest.raises(TypeError, match="PSDLinOp"):
        inflate_additive(None, ens, x=jnp.eye(3))


def test_public_names_report_the_package_they_are_imported_from():
    for name in kalman.__all__:
        assert getattr(kalman, name).__module__ == "enskit.kalman"


# ---------------------------------------------------------------------------
# helpers used above
# ---------------------------------------------------------------------------


def _unaligned(approx, extra: int = 0) -> Gaussian:
    """The same joint as a plain ``Gaussian``, so alignment is not declared.

    With ``extra``, the latent space grows by that many zero columns, which
    changes nothing but the width the general path whitens.
    """
    J = approx.latent_dim
    factors = {}
    for n in approx.names:
        F = approx.factor(n)
        if F is None:
            continue
        F = F.to_dense()
        if extra:
            F = jnp.concatenate([F, jnp.zeros((F.shape[0], extra), F.dtype)], axis=1)
        factors[n] = F
    return Gaussian(
        {n: approx.mean(n) for n in approx.names},
        factors=factors,
        block_covs={
            n: approx.block_cov(n)
            for n in approx.names
            if approx.block_cov(n) is not None
        },
        latent_dim=J + extra,
    )
