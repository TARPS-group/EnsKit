"""Conformance and regression tests for the maps layer and ``check_simulator``.

The file works through the numbered conformance obligations of the "Maps
contract", then holds the regression tests: those ported from
``tests/test_toy.py`` under the contract's table, keeping their old
docstrings' reasoning, and the new ones. Regression tests document why a rule
exists, and deleting one as redundant loses that.

Rules for the reference throughout:

- **The dense reference is hand-written here**, from means, anomalies and
  materialized operators, never routed through ``enskit.maps``.
- **Exactness tests compare against closed forms**, at a tolerance of a few
  machine epsilons times the quantity's natural scale.
- **Every test draws from its own deterministic stream**, reseeded from the
  test's id.
"""

from __future__ import annotations

import re
import subprocess
import sys
import warnings
import zlib
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import enskit  # noqa: F401  -- enables x64 before any array exists
from enskit import maps, toy
from enskit.distribution import Ensemble, EnsembleGaussian, Gaussian, reweight
from enskit.linalg import (
    Dense,
    DensePSD,
    PSDDiagonal,
    PSDLinOp,
    UnsupportedOpError,
    debug_checks,
    dense_matvec,
    linop,
)
from enskit.maps import AdditiveNoise, BlackBox, Linear, StructuredMap, pushforward
from enskit.testing import check_simulator

RNG = np.random.default_rng(0)
EPS = float(np.finfo(np.float64).eps)


@pytest.fixture(autouse=True)
def _reseed_rng(request):
    """Give every test its own deterministic stream, seeded from its id."""
    global RNG
    RNG = np.random.default_rng(zlib.crc32(request.node.nodeid.encode()))


# ---------------------------------------------------------------------------
# fixtures and the hand-written reference
# ---------------------------------------------------------------------------


@linop
class MatvecOnlyPSD(PSDLinOp):
    """A PSD operator that applies and whitens but cannot ``factor``."""

    C: Array

    @property
    def shape(self) -> tuple[int, int]:
        n = self.C.shape[-1]
        return (n, n)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.C.shape[:-2])

    def _matvec(self, x: Array) -> Array:
        return dense_matvec(self.C, x)

    def _to_dense(self) -> Array:
        return self.C


def _psd(n: int) -> np.ndarray:
    M = RNG.normal(size=(n, n))
    return M @ M.T + n * np.eye(n)


def _arr(*shape) -> Array:
    return jnp.asarray(RNG.normal(size=shape))


def _close(got, want, scale=None, factor=1e3) -> None:
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape, f"shape {got.shape} != {want.shape}"
    scale = max(1.0, float(np.abs(want).max())) if scale is None else scale
    err = float(np.abs(got - want).max()) if got.size else 0.0
    assert err <= factor * EPS * scale, f"max abs err {err:.3e} at scale {scale:.3e}"


def _stacked(obj, reps: int = 2):
    """A vmapped family: ``obj``'s leaves stacked along a new leading axis."""
    return jax.tree.map(lambda a: jnp.stack([a] * reps), obj)


def _dense_row(g, name) -> np.ndarray:
    F = g.factor(name)
    if F is None:
        return np.zeros((g.dims[name], g.latent_dim))
    return np.asarray(F.to_dense())


def _dense_cov(g, a, b=None) -> np.ndarray:
    """``cov(a, b)`` of a Gaussian from its materialized parts."""
    b = a if b is None else b
    C = _dense_row(g, a) @ _dense_row(g, b).T
    if a == b and g.block_cov(a) is not None:
        C = C + np.asarray(g.block_cov(a).to_dense())
    return C


def _sample_cov(xa, xb) -> np.ndarray:
    xa, xb = np.asarray(xa), np.asarray(xb)
    da, db = xa - xa.mean(0), xb - xb.mean(0)
    return da.T @ db / (xa.shape[0] - 1)


def _prior_with_partner(d: int = 4, dz: int = 2):
    """A Gaussian over ``x`` (with a term and a row) and ``z`` correlated with it."""
    k = 3
    Fx, Fz = _arr(d, k), _arr(dz, k)
    return Gaussian(
        {"x": _arr(d), "z": _arr(dz)},
        factors={"x": Fx, "z": Fz},
        block_covs={"x": DensePSD(jnp.asarray(_psd(d)))},
    )


# ===========================================================================
# 1. names and structure
# ===========================================================================


def test_1_outputs_replace_inputs_in_place_or_are_appended():
    J = 6
    u, z = _arr(J, 3), _arr(J, 2)
    ens = reweight(Ensemble(u=u, z=z), jnp.asarray(RNG.normal(size=J)))
    G = _arr(4, 3)

    new = pushforward(ens, lambda u: u @ G.T, inputs="u", output="g")
    assert new.names == ("u", "z", "g")
    assert new.n_particles == J and new.is_weighted
    np.testing.assert_array_equal(new.log_weights, ens.log_weights)
    assert new["u"] is ens["u"] and new["z"] is ens["z"]

    # A replacement keeps its position, and may change its dimension.
    replaced = pushforward(ens, lambda u: u @ G.T, inputs="u", output="u")
    assert replaced.names == ("u", "z") and replaced.dims == {"u": 4, "z": 2}
    _close(replaced["u"], np.asarray(u) @ np.asarray(G).T)

    # Several outputs: replaced ones in place, new ones appended in order.
    both = pushforward(
        ens, lambda u, z: (z + 1.0, u[:, :1], u), inputs=("u", "z"),
        output=("z", "w", "v"),
    )
    assert both.names == ("u", "z", "w", "v")
    np.testing.assert_array_equal(both["z"], ens["z"] + 1.0)
    np.testing.assert_array_equal(both["w"], ens["u"][:, :1])
    np.testing.assert_array_equal(both["v"], ens["u"])


def test_1_inputs_default_to_every_block_in_block_order():
    ens = Ensemble({"b": _arr(5, 2), "a": _arr(5, 3)})
    seen = []

    def record(*xs):
        seen.append(xs)
        return xs[0]

    pushforward(ens, record, output="c")
    (xs,) = seen
    assert [x.shape for x in xs] == [(5, 2), (5, 3)]
    assert xs[0] is ens["b"] and xs[1] is ens["a"]


def test_1_a_str_is_one_array_and_a_sequence_of_one_is_a_container():
    ens = Ensemble(u=_arr(5, 3))
    one = pushforward(ens, lambda u: 2.0 * u, inputs="u", output="g")
    container = pushforward(ens, lambda u: (2.0 * u,), inputs="u", output=["g"])
    np.testing.assert_array_equal(one["g"], container["g"])
    with pytest.raises(ValueError, match="returned shape"):
        pushforward(ens, lambda u: (2.0 * u,), inputs="u", output="g")


def test_1_a_gaussian_follows_the_same_rules():
    g = _prior_with_partner()
    A = _arr(5, 4)
    new = pushforward(g, Linear(A), inputs="x", output="y")
    assert new.names == ("x", "z", "y")
    replaced = pushforward(g, Linear(A), inputs="x", output="x")
    assert replaced.names == ("x", "z") and replaced.dims == {"x": 5, "z": 2}


# ===========================================================================
# 2. the call
# ===========================================================================


def test_2_one_call_with_every_particle_in_the_order_of_inputs():
    """Positional, concrete, read-only, and every row, failed ones included.

    The weighted ensemble holds a failed particle at weight zero: the layer
    does not select rows, so the simulator sees it like any other.
    """
    J = 5
    u, z = np.array(_arr(J, 3)), np.array(_arr(J, 2))
    u[2] = np.nan
    ens = Ensemble(u=jnp.asarray(u), z=jnp.asarray(z))
    ens = reweight(ens, jnp.where(ens.all_finite, 0.0, -jnp.inf))
    calls = []

    def record(zz, uu):
        calls.append((zz, uu))
        assert isinstance(zz, jax.Array) and not isinstance(zz, jax.core.Tracer)
        assert not np.asarray(zz).flags.writeable
        return jnp.concatenate([zz, uu], axis=-1)

    out = pushforward(ens, record, inputs=("z", "u"), output="g")
    assert len(calls) == 1
    zz, uu = calls[0]
    assert zz.shape == (J, 2) and uu.shape == (J, 3)
    np.testing.assert_array_equal(np.asarray(uu), u)
    assert not np.isfinite(np.asarray(out["g"])[2]).all()


# ===========================================================================
# 3. the containers are a promise
# ===========================================================================


def test_3_the_containers_are_a_promise_not_a_tolerance():
    """A jax array, a NumPy array and a nested list give bit-identical results.

    Ported from ``tests/test_eki.py``'s test 28. The numbers are computed once,
    so this tests the container and not whether NumPy and XLA agree bit for
    bit.
    """
    ens = Ensemble(u=_arr(6, 3))
    values = np.asarray(ens["u"]) @ np.asarray(_arr(4, 3)).T
    results = [
        pushforward(ens, lambda u, f=f: f(values), inputs="u", output="g")["g"]
        for f in (jnp.asarray, np.asarray, lambda v: v.tolist())
    ]
    for other in results[1:]:
        np.testing.assert_array_equal(np.asarray(other), np.asarray(results[0]))
        assert other.dtype == jnp.float64


# ===========================================================================
# 4. several outputs
# ===========================================================================


class _Fields(NamedTuple):
    b: Array
    a: Array


def test_4_tuple_mapping_and_namedtuple_give_the_same_result():
    ens = Ensemble(u=_arr(5, 3))
    a, b = (lambda u: u[:, :2]), (lambda u: 3.0 * u)
    forms = [
        lambda u: (a(u), b(u)),
        lambda u: [a(u), b(u)],
        lambda u: {"b": b(u), "a": a(u)},
        lambda u: _Fields(b=b(u), a=a(u)),  # field order differs from output's
    ]
    results = [pushforward(ens, f, inputs="u", output=("a", "b")) for f in forms]
    for r in results:
        assert r.names == ("u", "a", "b")
        np.testing.assert_array_equal(r["a"], results[0]["a"])
        np.testing.assert_array_equal(r["b"], results[0]["b"])


def test_4_a_container_that_does_not_match_the_names_raises():
    ens = Ensemble(u=_arr(5, 3))
    out = ("a", "b")
    with pytest.raises(ValueError, match="returned 3 arrays"):
        pushforward(ens, lambda u: (u, u, u), inputs="u", output=out)
    with pytest.raises(ValueError, match="keyed"):
        pushforward(ens, lambda u: {"a": u, "c": u}, inputs="u", output=out)
    with pytest.raises(ValueError, match="keyed"):
        pushforward(ens, lambda u: {"a": u}, inputs="u", output=out)
    with pytest.raises(ValueError, match="tuple in that order"):
        pushforward(ens, lambda u: u, inputs="u", output=out)


# ===========================================================================
# 5. the dtype rule (#19)
# ===========================================================================


def test_5_a_narrower_output_is_promoted_with_one_warning_per_call():
    """Two outputs promoted by one call give one warning naming both."""
    ens = Ensemble(u=_arr(5, 3))

    def narrow(u):
        return u.astype(jnp.float32), (2.0 * u).astype(jnp.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = pushforward(ens, narrow, inputs="u", output=("a", "b"))
    promotions = [w for w in caught if "promoted" in str(w.message)]
    assert len(promotions) == 1
    message = str(promotions[0].message)
    assert "narrow" in message and "'a'" in message and "'b'" in message
    assert "float32" in message
    assert promotions[0].filename == __file__, "the warning points at the caller"
    assert out["a"].dtype == out["b"].dtype == jnp.float64

    # Under jit the warning is issued when tracing, once.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f = jax.jit(lambda e: pushforward(e, narrow, inputs="u", output=("a", "b")))
        f(ens)
        f(ens)
    assert len([w for w in caught if "promoted" in str(w.message)]) == 1


def test_5_a_wider_output_raises_naming_the_simulator():
    """Issue #19, settled: a wider return names the simulator, not a later layer.

    Ported from ``tests/test_eki.py``'s ``test_29_promotion_only_ever_widens``,
    which pinned the old behavior: a wider return passed through, and the run
    then died at the update's dtype check with an error naming an update rule
    the caller never wrote.
    """
    ens = Ensemble(u=_arr(5, 3).astype(jnp.float32))

    def widening_simulator(u):
        return np.asarray(u).astype(np.float64)

    with pytest.raises(ValueError, match="widening_simulator.*float64.*wider"):
        pushforward(ens, widening_simulator, inputs="u", output="g")


@pytest.mark.parametrize(
    "dtype, fragment",
    [
        (jnp.int64, "not a real floating"),
        (jnp.bool_, "not a real floating"),
        (jnp.complex128, "not a real floating"),
    ],
)
def test_5_integer_boolean_and_complex_outputs_raise(dtype, fragment):
    ens = Ensemble(u=_arr(5, 3))
    with pytest.raises(ValueError, match=fragment):
        pushforward(ens, lambda u: u.astype(dtype), inputs="u", output="g")


def test_5_an_incomparable_dtype_raises():
    """float16 and bfloat16 promote to neither: refused, not guessed."""
    ens = Ensemble(u=_arr(5, 3).astype(jnp.float16))
    with pytest.raises(ValueError, match="does not promote"):
        pushforward(ens, lambda u: u.astype(jnp.bfloat16), inputs="u", output="g")


def test_5_a_nested_list_is_read_in_the_ensemble_dtype():
    """A list carries no precision of its own, so a float32 ensemble takes one.

    Without this, ``jnp.asarray`` read every list as float64, and a float32
    ensemble refused it as wider than itself.
    """
    ens = Ensemble(u=_arr(5, 3).astype(jnp.float32))
    as_array = pushforward(ens, lambda u: 2 * u, inputs="u", output="g")
    as_list = pushforward(
        ens, lambda u: np.asarray(2 * u).tolist(), inputs="u", output="g"
    )
    assert as_list["g"].dtype == jnp.float32
    np.testing.assert_array_equal(as_list["g"], as_array["g"])
    with pytest.raises(ValueError, match="not a real floating"):
        pushforward(ens, lambda u: [[1, 2, 3]] * 5, inputs="u", output="g")


def test_5_a_wider_numpy_return_raises_with_x64_off_too():
    """The dtype is read before conversion, which would demote it silently.

    With x64 off, ``jnp.asarray`` turns a float64 NumPy array into float32
    with no warning, so a rule applied after it never saw the wider return.
    Checked in a fresh interpreter, since x64 is a process-wide setting.
    """
    program = (
        "import enskit, jax, jax.numpy as jnp, numpy as np\n"
        "jax.config.update('jax_enable_x64', False)\n"
        "from enskit import maps\n"
        "from enskit.distribution import Ensemble\n"
        "ens = Ensemble(u=jnp.ones((4, 2)))\n"
        "assert ens['u'].dtype == jnp.float32\n"
        "try:\n"
        "    maps.pushforward(ens, lambda u: np.asarray(u, np.float64) / 3,\n"
        "                     inputs='u', output='g')\n"
        "except ValueError as e:\n"
        "    print('raised', 'wider' in str(e))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "raised True", result.stdout + result.stderr


def test_5_the_warning_names_the_callers_line_through_pipe():
    ens = Ensemble(u=_arr(5, 3))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ens.pipe(pushforward, lambda u: u.astype(jnp.float32), inputs="u", output="g")
    (w,) = [w for w in caught if "promoted" in str(w.message)]
    assert w.filename == __file__


def test_5_the_structured_maps_say_what_to_change():
    ens = Ensemble(x=_arr(5, 3).astype(jnp.float32))
    with pytest.raises(ValueError, match="noise covariance the ensemble's dtype"):
        pushforward(ens, AdditiveNoise(PSDDiagonal(jnp.ones(3))), inputs="x",
                    output="y", key=jax.random.key(0))
    with pytest.raises(ValueError, match="operators and the shift"):
        pushforward(ens, Linear(jnp.eye(3)), inputs="x", output="y")


# ===========================================================================
# 6. failed rows are written through
# ===========================================================================


def test_6_failed_rows_are_written_through_in_debug_mode_too():
    ens = Ensemble(u=_arr(6, 2))

    def failing(u):
        return jnp.where(u[:, :1] > 0, u, jnp.nan)

    expected = np.asarray(ens["u"])[:, 0] > 0
    assert expected.any() and not expected.all(), "vacuous: no failure or all failed"
    with debug_checks():
        out = pushforward(ens, failing, inputs="u", output="g")
    np.testing.assert_array_equal(np.asarray(out.all_finite), expected)


# ===========================================================================
# 7. needs_key
# ===========================================================================


def test_7_a_simulator_that_needs_a_key_gets_it_whole_and_first():
    ens = Ensemble(u=_arr(5, 3))
    seen = []

    def noisy(key, u):
        seen.append(key)
        return u + jax.random.normal(key, u.shape, u.dtype)

    noisy.needs_key = True
    key = jax.random.key(7)
    a = pushforward(ens, noisy, inputs="u", output="g", key=key)
    b = pushforward(ens, noisy, inputs="u", output="g", key=key)
    assert seen[0] is key
    np.testing.assert_array_equal(a["g"], b["g"])
    with pytest.raises(ValueError, match="noisy declares needs_key"):
        pushforward(ens, noisy, inputs="u", output="g")


def test_7_a_key_is_type_checked_and_otherwise_ignored():
    ens = Ensemble(u=_arr(5, 3))
    plain = pushforward(ens, lambda u: u, inputs="u", output="g")
    keyed = pushforward(ens, lambda u: u, inputs="u", output="g", key=jax.random.key(0))
    np.testing.assert_array_equal(plain["g"], keyed["g"])
    with pytest.raises(TypeError, match="typed key"):
        pushforward(ens, lambda u: u, inputs="u", output="g", key=jax.random.PRNGKey(0))

    def truthy(u):
        return u

    truthy.needs_key = 1  # only True declares it
    pushforward(ens, truthy, inputs="u", output="g")


# ===========================================================================
# 8. Linear on an ensemble
# ===========================================================================


def test_8_linear_on_an_ensemble_is_exact_per_particle():
    J = 7
    u, z = _arr(J, 3), _arr(J, 2)
    A, B, c = _arr(4, 3), _arr(4, 2), _arr(4)
    ens = Ensemble(u=u, z=z)
    out = pushforward(ens, Linear(A, shift=c), inputs="u", output="g")
    _close(out["g"], np.asarray(u) @ np.asarray(A).T + np.asarray(c))
    lin = Linear({"u": A, "z": Dense(B)})
    both = pushforward(ens, lin, inputs=("u", "z"), output="g")
    _close(both["g"], np.asarray(u) @ np.asarray(A).T + np.asarray(z) @ np.asarray(B).T)


def test_8_pushing_particles_then_projecting_equals_projecting_then_pushing():
    """Sample covariances are linear in the particles, so the routes agree exactly."""
    J = 9
    ens = Ensemble(u=_arr(J, 3), z=_arr(J, 2))
    lin = Linear({"u": _arr(4, 3), "z": _arr(4, 2)}, shift=_arr(4))
    first = pushforward(ens, lin, inputs=("u", "z"), output="g").project()
    second = pushforward(ens.project(), lin, inputs=("u", "z"), output="g")
    assert isinstance(second, EnsembleGaussian)
    _close(second.mean("g"), first.mean("g"))
    for other in ("u", "z", "g"):
        _close(_dense_cov(second, "g", other), _dense_cov(first, "g", other))


# ===========================================================================
# 9. Linear on a Gaussian
# ===========================================================================


def test_9_linear_on_an_aligned_block_keeps_the_kind_and_the_operators():
    J = 8
    ens = Ensemble(x=_arr(J, 4), z=_arr(J, 2))
    approx = ens.project()
    A, c = _arr(5, 4), _arr(5)
    out = pushforward(approx, Linear(A, shift=c), inputs="x", output="y")
    assert isinstance(out, EnsembleGaussian) and out.n_particles == J
    assert type(out.factor("y")).__name__ == "Product", "the row was densified"
    An = np.asarray(A)
    _close(out.mean("y"), An @ np.asarray(approx.mean("x")) + np.asarray(c))
    Cxx = _dense_cov(approx, "x")
    _close(_dense_cov(out, "y"), An @ Cxx @ An.T)
    _close(_dense_cov(out, "y", "x"), An @ Cxx)
    _close(_dense_cov(out, "y", "z"), An @ _dense_cov(approx, "x", "z"))


def test_9_linear_absorbs_an_input_term_and_keeps_every_covariance():
    g = _prior_with_partner()
    A, c = _arr(5, 4), _arr(5)
    out = pushforward(g, Linear(A, shift=c), inputs="x", output="y")
    assert type(out) is Gaussian and out.latent_dim == g.latent_dim + 4
    An = np.asarray(A)
    _close(out.mean("y"), An @ np.asarray(g.mean("x")) + np.asarray(c))
    Cxx = _dense_cov(g, "x")
    _close(_dense_cov(out, "y"), An @ Cxx @ An.T)
    _close(_dense_cov(out, "y", "x"), An @ Cxx)
    _close(_dense_cov(out, "y", "z"), An @ _dense_cov(g, "x", "z"))
    _close(_dense_cov(out, "x"), Cxx)


def test_9_linear_with_several_inputs_matches_the_closed_form():
    g = _prior_with_partner()
    A, B = _arr(3, 4), _arr(3, 2)
    out = pushforward(g, Linear({"x": A, "z": B}), inputs=("x", "z"), output="y")
    An, Bn = np.asarray(A), np.asarray(B)
    Cxx, Czz, Cxz = _dense_cov(g, "x"), _dense_cov(g, "z"), _dense_cov(g, "x", "z")
    _close(out.mean("y"), An @ np.asarray(g.mean("x")) + Bn @ np.asarray(g.mean("z")))
    _close(
        _dense_cov(out, "y"),
        An @ Cxx @ An.T + Bn @ Czz @ Bn.T + An @ Cxz @ Bn.T + Bn @ Cxz.T @ An.T,
    )
    _close(_dense_cov(out, "y", "x"), An @ Cxx + Bn @ Cxz.T)
    _close(_dense_cov(out, "y", "z"), An @ Cxz + Bn @ Czz)


def test_9_several_inputs_out_of_order_with_terms_replacing_one():
    """Inputs not in block order, two with terms, the output replacing one.

    The new row is assembled from the inputs in the order given, while
    ``absorb`` works in block order; a mix-up between the two gives every
    covariance of the wrong block, with nothing raised.
    """
    k = 2
    g = Gaussian(
        {"a": _arr(3), "b": _arr(2), "c": _arr(4)},
        factors={"a": _arr(3, k), "c": _arr(4, k)},
        block_covs={"b": DensePSD(jnp.asarray(_psd(2))),
                    "c": DensePSD(jnp.asarray(_psd(4)))},
    )
    Ac, Ab = _arr(3, 4), _arr(3, 2)
    out = pushforward(g, Linear({"c": Ac, "b": Ab}), inputs=("c", "b"), output="b")
    assert out.names == ("a", "b", "c") and out.dims["b"] == 3
    Acn, Abn = np.asarray(Ac), np.asarray(Ab)
    C = {(p, q): _dense_cov(g, p, q) for p in "abc" for q in "abc"}
    _close(out.mean("b"), Acn @ np.asarray(g.mean("c")) + Abn @ np.asarray(g.mean("b")))
    _close(
        _dense_cov(out, "b"),
        Acn @ C["c", "c"] @ Acn.T + Abn @ C["b", "b"] @ Abn.T
        + Acn @ C["c", "b"] @ Abn.T + Abn @ C["b", "c"] @ Acn.T,
    )
    _close(_dense_cov(out, "b", "a"), Acn @ C["c", "a"] + Abn @ C["b", "a"])
    _close(_dense_cov(out, "b", "c"), Acn @ C["c", "c"] + Abn @ C["b", "c"])
    _close(_dense_cov(out, "c"), C["c", "c"])


def test_9_linear_replacing_its_input_matches_the_closed_form():
    g = _prior_with_partner()
    A = _arr(5, 4)
    out = pushforward(g, Linear(A), inputs="x", output="x")
    An = np.asarray(A)
    assert out.block_cov("x") is None
    _close(_dense_cov(out, "x"), An @ _dense_cov(g, "x") @ An.T)
    _close(_dense_cov(out, "x", "z"), An @ _dense_cov(g, "x", "z"))


def test_9_a_prior_pushed_and_conditioned_is_the_linear_gaussian_posterior():
    P, N = 4, 3
    m0, C0, R = _arr(P), _psd(P), _psd(N)
    A, c = np.asarray(_arr(N, P)), np.asarray(_arr(N))
    y = np.asarray(_arr(N))
    prior = Gaussian.independent(u=(m0, DensePSD(jnp.asarray(C0))))
    joint = (
        prior.pipe(pushforward, Linear(A, shift=c), inputs="u", output="g")
        .pipe(pushforward, AdditiveNoise(DensePSD(jnp.asarray(R))), inputs="g",
              output="y")
    )
    post = joint.condition(y=y).marginal("u")
    S = A @ C0 @ A.T + R
    K = C0 @ A.T @ np.linalg.inv(S)
    _close(post.mean("u"), np.asarray(m0) + K @ (y - A @ np.asarray(m0) - c),
           factor=1e4)
    _close(_dense_cov(post, "u"), C0 - K @ A @ C0, factor=1e4)


# ===========================================================================
# 10. AdditiveNoise on a Gaussian
# ===========================================================================


def test_10_additive_noise_replacing_its_input_is_add_noise():
    J = 6
    approx = Ensemble(x=_arr(J, 3)).project()
    R = DensePSD(jnp.asarray(_psd(3)))
    out = pushforward(approx, AdditiveNoise(R), inputs="x", output="x")
    ref = approx.add_noise(x=R)
    assert type(out) is type(ref) is EnsembleGaussian
    _close(_dense_cov(out, "x"), _dense_cov(ref, "x"))


def test_10_additive_noise_to_a_new_block_matches_the_closed_form():
    R = _psd(4)
    noise = AdditiveNoise(DensePSD(jnp.asarray(R)))

    # The input has a term: absorbed, so the copy stays correlated through it.
    g = _prior_with_partner()
    out = pushforward(g, noise, inputs="x", output="y")
    assert type(out) is Gaussian
    Cxx = _dense_cov(g, "x")
    _close(out.mean("y"), g.mean("x"))
    _close(_dense_cov(out, "y"), Cxx + R)
    _close(_dense_cov(out, "y", "x"), Cxx)
    _close(_dense_cov(out, "y", "z"), _dense_cov(g, "x", "z"))

    # The input has no term: the latent space, and so the kind, is kept.
    approx = Ensemble(x=_arr(7, 4), z=_arr(7, 2)).project()
    out = pushforward(approx, noise, inputs="x", output="y")
    assert isinstance(out, EnsembleGaussian) and out.factor("y") is approx.factor("x")
    _close(_dense_cov(out, "y"), _dense_cov(approx, "x") + R)
    _close(_dense_cov(out, "y", "x"), _dense_cov(approx, "x"))


# ===========================================================================
# 11. AdditiveNoise on an ensemble
# ===========================================================================


def test_11_additive_noise_on_an_ensemble_is_the_pinned_draw():
    J = 6
    ens = Ensemble(x=_arr(J, 3))
    R = DensePSD(jnp.asarray(_psd(3)))
    key = jax.random.key(3)
    out = pushforward(ens, AdditiveNoise(R), inputs="x", output="y", key=key)
    eta = jax.random.normal(key, (J, 3), jnp.float64)
    np.testing.assert_array_equal(out["y"], ens["x"] + R.factor().matvec(eta))
    np.testing.assert_array_equal(out["x"], ens["x"])


def test_11_a_missing_key_or_factor_raises_before_drawing():
    ens = Ensemble(x=_arr(5, 3))
    with pytest.raises(ValueError, match="key is required"):
        pushforward(ens, AdditiveNoise(PSDDiagonal(jnp.ones(3))), inputs="x",
                    output="y")
    no_factor = AdditiveNoise(MatvecOnlyPSD(jnp.asarray(_psd(3))))
    with pytest.raises(UnsupportedOpError, match="factor"):
        pushforward(ens, no_factor, inputs="x", output="y", key=jax.random.key(0))


# ===========================================================================
# 12. BlackBox
# ===========================================================================


def _host_model(G):
    calls = []

    def simulate(u):
        calls.append(u)
        return np.tanh(np.asarray(u)) @ G.T

    return simulate, calls


def test_12_blackbox_matches_the_plain_callable_eagerly_under_jit_and_vmap():
    G = np.asarray(_arr(4, 3))
    simulate, calls = _host_model(G)
    box = BlackBox(simulate, 4)
    ens = Ensemble(u=_arr(5, 3))
    plain = pushforward(ens, simulate, inputs="u", output="g")["g"]
    eager = pushforward(ens, box, inputs="u", output="g")["g"]
    jitted = jax.jit(lambda e: pushforward(e, box, inputs="u", output="g")["g"])(ens)
    _close(eager, plain)
    _close(jitted, plain)

    family = _arr(3, 5, 3)
    calls.clear()
    mapped = jax.vmap(lambda u: pushforward(Ensemble(u=u), box, inputs="u",
                                            output="g")["g"])(family)
    assert len(calls) == 3 and all(c.shape == (5, 3) for c in calls)
    for i in range(3):
        _close(mapped[i], np.tanh(np.asarray(family[i])) @ G.T)


def test_12_blackbox_has_a_zero_derivative_in_both_modes():
    G = np.asarray(_arr(4, 3))
    box = BlackBox(_host_model(G)[0], 4)
    u = _arr(5, 3)

    def loss(u):
        return jnp.sum(pushforward(Ensemble(u=u), box, inputs="u", output="g")["g"] ** 2)

    for grad in (jax.grad(loss), jax.jit(jax.grad(loss)), jax.jacfwd(loss)):
        np.testing.assert_array_equal(grad(u), np.zeros((5, 3)))
    _, tangent = jax.jvp(lambda u: box(u), (u,), (jnp.ones_like(u),))
    np.testing.assert_array_equal(tangent, np.zeros((5, 4)))


def test_12_blackbox_with_a_key_and_several_outputs():
    seen = []

    def simulate(key_data, u):
        seen.append(key_data)
        return u[:, :1], 2.0 * u

    box = BlackBox(simulate, (1, 3), needs_key=True)
    assert box.needs_key is True
    ens = Ensemble(u=_arr(5, 3))
    key = jax.random.key(11)
    out = pushforward(ens, box, inputs="u", output=("a", "b"), key=key)
    np.testing.assert_array_equal(seen[0], np.asarray(jax.random.key_data(key)))
    _close(out["b"], 2.0 * np.asarray(ens["u"]))
    with pytest.raises(ValueError, match="needs_key"):
        pushforward(ens, box, inputs="u", output=("a", "b"))


@pytest.mark.parametrize(
    "returned, fragment",
    [
        (lambda u: u[:, :2], r"returned shape \(5, 2\)"),
        (lambda u: u.astype(np.int64), "not a real floating"),
        (lambda u: (u, u), "returned shape"),
    ],
)
def test_12_the_callback_holds_the_return_to_the_contract(returned, fragment):
    box = BlackBox(returned, 3)
    with pytest.raises(jax.errors.JaxRuntimeError, match=fragment):
        box(_arr(5, 3))


def test_12_the_callback_refuses_a_wider_return_and_promotes_a_narrower_one():
    u32 = _arr(5, 3).astype(jnp.float32)
    with pytest.raises(jax.errors.JaxRuntimeError, match="wider"):
        BlackBox(lambda u: u.astype(np.float64), 3)(u32)
    with pytest.warns(UserWarning, match="promoted"):
        out = BlackBox(lambda u: u.astype(np.float32), 3)(_arr(5, 3))
    assert out.dtype == jnp.float64
    assert BlackBox(lambda u: u, 3, dtype=jnp.float32)(u32).dtype == jnp.float32

    # Two narrow outputs, one call: one warning naming both.
    two = BlackBox(lambda u: (u.astype(np.float32), u.astype(np.float32)), (3, 3))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        two(_arr(5, 3))
    (w,) = [w for w in caught if "promoted" in str(w.message)]
    assert "output 0" in str(w.message) and "output 1" in str(w.message)


# ===========================================================================
# 13. differentiability
# ===========================================================================


def test_13_a_jax_simulator_is_differentiated_row_by_row():
    G = _arr(4, 3)
    u = _arr(5, 3)

    def loss(u):
        return jnp.sum(pushforward(Ensemble(u=u), lambda u: u @ G.T, inputs="u",
                                   output="g")["g"])

    _close(jax.grad(loss)(u), np.tile(np.asarray(G).sum(0), (5, 1)))


def test_13_linear_on_a_gaussian_differentiates_in_the_operator_and_shift():
    P = 3
    C0 = _psd(P)
    prior = Gaussian.independent(u=(_arr(P), DensePSD(jnp.asarray(C0))))

    def trace_cov(A):
        out = pushforward(prior, Linear(Dense(A)), inputs="u", output="g")
        return jnp.trace(out.cov("g").to_dense())

    def mean_sum(c):
        out = pushforward(prior, Linear(jnp.eye(P), shift=c), inputs="u", output="g")
        return jnp.sum(out.mean("g"))

    A = _arr(4, P)
    _close(jax.grad(trace_cov)(A), 2.0 * np.asarray(A) @ C0, factor=1e4)
    _close(jax.grad(mean_sum)(_arr(P)), np.ones(P))


def test_13_additive_noise_on_an_ensemble_is_reparameterized():
    J, d = 6, 3
    ens = Ensemble(x=_arr(J, d))
    key = jax.random.key(5)

    def total(s):
        out = pushforward(ens, AdditiveNoise(PSDDiagonal(s)), inputs="x", output="y",
                          key=key)
        return jnp.sum(out["y"])

    s = jnp.asarray(RNG.uniform(0.5, 2.0, size=d))
    eta = np.asarray(jax.random.normal(key, (J, d), jnp.float64))
    _close(jax.grad(total)(s), eta.sum(0) / (2.0 * np.sqrt(np.asarray(s))))


# ===========================================================================
# 14. JAX integration
# ===========================================================================


def test_14_pushforward_under_jit_and_vmap():
    G = _arr(4, 3)
    ens = Ensemble(u=_arr(5, 3))
    eager = pushforward(ens, Linear(G), inputs="u", output="g")
    jitted = jax.jit(lambda e: pushforward(e, Linear(G), inputs="u", output="g"))(ens)
    np.testing.assert_allclose(jitted["g"], eager["g"], rtol=0, atol=1e-14)

    family = _arr(3, 5, 3)
    mapped = jax.vmap(
        lambda u: pushforward(Ensemble(u=u), lambda u: u @ G.T, inputs="u",
                              output="g")["g"]
    )(family)
    _close(mapped, np.asarray(family) @ np.asarray(G).T)


def test_14_a_linear_map_crosses_jit_as_data_without_recompiling():
    traces = []

    @jax.jit
    def push(g, lin):
        traces.append(1)
        return pushforward(g, lin, inputs="u", output="y").mean("y")

    prior = Gaussian.independent(u=(_arr(3), DensePSD(jnp.asarray(_psd(3)))))
    for _ in range(3):
        push(prior, Linear(_arr(4, 3), shift=_arr(4)))
    assert len(traces) == 1
    push(prior, Linear(_arr(4, 3)))  # no shift: a different tree structure
    assert len(traces) == 2


# ===========================================================================
# 15. a user-written StructuredMap
# ===========================================================================


class _Scale:
    """``x -> s x``, exact on Gaussians: a minimal user StructuredMap."""

    def __init__(self, s):
        self.s = s

    def push_ensemble(self, ensemble, inputs, output, key):
        (x,) = inputs
        return ensemble.assign({output: self.s * ensemble[x]})

    def push_gaussian(self, gaussian, inputs, output):
        (x,) = inputs
        return pushforward(gaussian, Linear(self.s * jnp.eye(gaussian.dims[x])),
                           inputs=x, output=output)


class _Dropper:
    """Breaks rule 1 of the protocol: loses a block."""

    def push_ensemble(self, ensemble, inputs, output, key):
        return ensemble.marginal(inputs[0])

    def push_gaussian(self, gaussian, inputs, output):
        return Ensemble(x=jnp.zeros((2, 1)))


def test_15_a_user_structured_map_is_dispatched_to_and_checked():
    s = jnp.asarray(3.0)
    assert isinstance(_Scale(s), StructuredMap)
    ens = Ensemble(x=_arr(5, 2), z=_arr(5, 1))
    out = pushforward(ens, _Scale(s), inputs="x", output="y")
    _close(out["y"], 3.0 * np.asarray(ens["x"]))
    g = pushforward(ens.project(), _Scale(s), inputs="x", output="y")
    _close(_dense_cov(g, "y"), 9.0 * _dense_cov(ens.project(), "x"))

    with pytest.raises(ValueError, match="returned blocks"):
        pushforward(ens, _Dropper(), inputs="x", output="y")
    with pytest.raises(TypeError, match="returned Ensemble for a"):
        pushforward(ens.project(), _Dropper(), inputs="x", output="y")


# ===========================================================================
# 16. validation
# ===========================================================================


def _validation_cases():
    ens = Ensemble(u=_arr(5, 3), z=_arr(5, 2))
    g = ens.project()
    ident = lambda u: u  # noqa: E731
    fam = _stacked(ens)
    return [
        ("not a dist", lambda: pushforward(np.ones((5, 3)), ident, output="g"),
         TypeError, "Ensemble or a Gaussian"),
        ("family", lambda: pushforward(fam, ident, inputs="u", output="g"),
         ValueError, "vmapped family"),
        ("output type", lambda: pushforward(ens, ident, inputs="u", output=3),
         TypeError, "output must be"),
        ("output entry", lambda: pushforward(ens, ident, inputs="u", output=("a", 1)),
         TypeError, "must be a str"),
        ("inputs type", lambda: pushforward(ens, ident, inputs=3, output="g"),
         TypeError, "inputs must be"),
        ("no output", lambda: pushforward(ens, ident, inputs="u", output=()),
         ValueError, "at least one output"),
        ("no input", lambda: pushforward(ens, ident, inputs=(), output="g"),
         ValueError, "at least one input"),
        ("repeated output",
         lambda: pushforward(ens, ident, inputs="u", output=("a", "a")),
         ValueError, "more than once"),
        ("repeated input",
         lambda: pushforward(ens, ident, inputs=("u", "u"), output="g"),
         ValueError, "more than once"),
        ("unknown input", lambda: pushforward(ens, ident, inputs="w", output="g"),
         KeyError, "not a block"),
        ("overwrite", lambda: pushforward(ens, ident, inputs="u", output="z"),
         ValueError, "drop it first"),
        ("raw key",
         lambda: pushforward(ens, ident, inputs="u", output="g",
                             key=jax.random.PRNGKey(0)),
         TypeError, "typed key"),
        ("gaussian callable", lambda: pushforward(g, ident, inputs="u", output="g"),
         TypeError, "only be pushed through a StructuredMap"),
        ("structured several",
         lambda: pushforward(ens, Linear(jnp.eye(3)), inputs="u", output=("a", "b")),
         ValueError, "produces one block"),
        ("not callable", lambda: pushforward(ens, 3.0, inputs="u", output="g"),
         TypeError, "callable or a StructuredMap"),
        ("linear count",
         lambda: pushforward(ens, Linear(jnp.eye(3)), inputs=("u", "z"), output="g"),
         ValueError, "takes 1 input"),
        ("linear names",
         lambda: pushforward(ens, Linear({"z": jnp.ones((2, 2)), "u": jnp.eye(2, 3)}),
                             inputs=("u", "z"), output="g"),
         ValueError, "built for the inputs"),
        ("linear dims",
         lambda: pushforward(g, Linear(jnp.eye(2)), inputs="u", output="g"),
         ValueError, "of dimension 3"),
        ("noise count",
         lambda: pushforward(g, AdditiveNoise(PSDDiagonal(jnp.ones(3))),
                             inputs=("u", "z"), output="y"),
         ValueError, "exactly one input"),
        ("noise dims",
         lambda: pushforward(g, AdditiveNoise(PSDDiagonal(jnp.ones(2))), inputs="u",
                             output="y"),
         ValueError, "dimension 2 to block 'u'"),
        ("not array-like",
         lambda: pushforward(ens, lambda u: object(), inputs="u", output="g"),
         ValueError, "not array-like"),
        ("one dimensional",
         lambda: pushforward(ens, lambda u: u[:, 0], inputs="u", output="g"),
         ValueError, r"\(5, 1\)"),
        ("wrong length",
         lambda: pushforward(ens, lambda u: u[:4], inputs="u", output="g"),
         ValueError, "returned shape"),
        ("no columns",
         lambda: pushforward(ens, lambda u: u[:, :0], inputs="u", output="g"),
         ValueError, "d_out >= 1"),
        ("linear wider",
         lambda: pushforward(Ensemble(u=_arr(5, 3).astype(jnp.float32)),
                             Linear(jnp.eye(3)), inputs="u", output="g"),
         ValueError, "Linear returned dtype float64"),
        ("linear wider on a gaussian",
         lambda: pushforward(
             Gaussian({"u": jnp.zeros(3, jnp.float32)},
                      block_covs={"u": PSDDiagonal(jnp.ones(3, jnp.float32))}),
             Linear(jnp.eye(3)), inputs="u", output="g"),
         ValueError, "differs from the Gaussian's"),
        ("noise key", lambda: pushforward(ens, AdditiveNoise(PSDDiagonal(jnp.ones(3))),
                                          inputs="u", output="y"),
         ValueError, "key is required"),
        ("needs key, missing",
         lambda: pushforward(ens, BlackBox(lambda k, u: u, 3, needs_key=True),
                             inputs="u", output="g"),
         ValueError, "declares needs_key"),
    ]


_CASES = [case[0] for case in _validation_cases()]


@pytest.mark.parametrize("name", _CASES)
def test_16_every_row_of_the_validation_table(name):
    (_, call, exc, fragment), = [c for c in _validation_cases() if c[0] == name]
    with pytest.raises(exc, match=fragment):
        call()


@pytest.mark.parametrize(
    "build, exc, fragment",
    [
        (lambda: Linear("nope"), TypeError, "LinOp or a 2-D array"),
        (lambda: Linear(jnp.ones((2, 2), jnp.int32)), TypeError, "real floating"),
        (lambda: Linear(jnp.ones(3)), ValueError, "2-D array"),
        (lambda: Linear({}), ValueError, "empty"),
        (lambda: Linear({1: jnp.eye(2)}), TypeError, "must be str"),
        (lambda: Linear({"a": jnp.eye(2), "b": jnp.ones((3, 2))}), ValueError,
         "disagree"),
        (lambda: Linear(jnp.eye(2), shift=jnp.ones(3)), ValueError, r"shape \(2,\)"),
        (lambda: Linear(jnp.eye(2), shift=jnp.ones(2, jnp.int32)), TypeError,
         "real floating"),
        (lambda: Linear(_stacked(Dense(jnp.ones((3, 3))))), ValueError,
         "vmapped family"),
        (lambda: AdditiveNoise(Dense(jnp.eye(2))), TypeError, "PSDLinOp"),
        (lambda: AdditiveNoise(_stacked(PSDDiagonal(jnp.ones(3)))), ValueError,
         "vmapped family"),
        (lambda: BlackBox(3, 2), TypeError, "callable"),
        (lambda: BlackBox(len, 2.0), TypeError, "output_dim"),
        (lambda: BlackBox(len, True), TypeError, "output_dim"),
        (lambda: BlackBox(len, ()), TypeError, "output_dim"),
        (lambda: BlackBox(len, (2, 0)), ValueError, "at least 1"),
        (lambda: BlackBox(len, 2, dtype=jnp.int32), TypeError, "real floating"),
        (lambda: BlackBox(len, 2, needs_key=1), TypeError, "bool"),
    ],
)
def test_16_constructors_validate(build, exc, fragment):
    with pytest.raises(exc, match=fragment):
        build()


def test_16_direct_calls_validate():
    with pytest.raises(TypeError, match="takes 1 input"):
        Linear(jnp.eye(2))(jnp.ones((3, 2)), jnp.ones((3, 2)))
    box = BlackBox(lambda u: u, 2)
    with pytest.raises(TypeError, match="at least one input"):
        box()
    with pytest.raises(ValueError, match="2-D"):
        box(jnp.ones(2))
    keyed = BlackBox(lambda k, u: u, 2, needs_key=True)
    with pytest.raises(TypeError, match="typed key"):
        keyed(jnp.ones((3, 2)), jnp.ones((3, 2)))


def test_16_the_layer_names_its_public_objects_and_reprs():
    for name in maps.__all__:
        assert getattr(maps, name).__module__ == "enskit.maps"
    assert check_simulator.__module__ == "enskit.testing"
    assert repr(Linear(jnp.ones((6, 4)), shift=jnp.ones(6))) == (
        "Linear(shape=(6, 4), shift=True)"
    )
    assert repr(Linear({"x": jnp.ones((6, 4)), "z": jnp.ones((6, 2))})) == (
        "Linear(inputs={'x': 4, 'z': 2}, output_dim=6, shift=False)"
    )
    assert repr(AdditiveNoise(PSDDiagonal(jnp.ones(6)))) == "AdditiveNoise(dim=6)"

    def simulate(u):
        return u

    assert repr(BlackBox(simulate, 10)) == (
        "BlackBox(f=simulate, output_dim=10, needs_key=False)"
    )


def test_16_no_layer_imports_the_testing_module():
    """``enskit.testing`` sits on top, beside the toy module, imported by nothing.

    Checked in a fresh interpreter, since this one has already imported it.
    """
    program = (
        "import sys; import enskit, enskit.linalg, enskit.distribution, enskit.maps; "
        "print('enskit.testing' in sys.modules, 'enskit.toy' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False False", result.stdout


# ===========================================================================
# 17. check_simulator
# ===========================================================================

#: One instance of each toy problem, with the sizes it should answer at.
PROBLEMS = [
    ("linear_gaussian", toy.linear_gaussian(), 4, 8),
    ("exponential_decay", toy.exponential_decay(), 2, 12),
    ("restricted_decay", toy.restricted_decay(), 2, 12),
]


@pytest.mark.parametrize("name, problem, d_in, d_out", PROBLEMS,
                         ids=[p[0] for p in PROBLEMS])
def test_17_every_toy_simulator_passes_check_simulator(name, problem, d_in, d_out):
    """The harness a user runs their own simulator through, run on ours.

    Ported from ``test_4_every_model_passes_check_forward_model``.
    """
    check_simulator(problem.forward, d_in, d_out)


def _decay():
    times = jnp.linspace(0.25, 1.0, 4)
    return jax.vmap(lambda u: u[0] * jnp.exp(-u[1] * times)), times


def test_17_check_simulator_rejects_the_defects_it_claims_to_catch():
    """A checker with no failing case is worthless, so each check gets one.

    Ported from ``test_4_check_forward_model_rejects_the_defects_it_claims_to_catch``.
    The two row-coupling cases are the valuable ones: that defect is the one
    the contract calls undetectable inside a pushforward. They are separate
    cases because a symmetric coupling survives a permutation of the
    particles and only the subset comparison catches it.
    """
    decay, times = _decay()

    def normalized(u):
        # Symmetric coupling: survives a permutation; only the subset sees it.
        return decay(u - jnp.mean(u, axis=0))

    def ordered(u):
        # Order-dependent coupling: a running total down the rows.
        return decay(jnp.cumsum(u, axis=0))

    def per_particle(u):
        # Written for one particle and handed them all: one output vector.
        p = u[0]
        return p[0] * jnp.exp(-p[1] * times)

    def narrow(u):
        return decay(u).astype(jnp.float32)

    def integer(u):
        return jnp.zeros((u.shape[0], times.size), dtype=jnp.int64)

    def stochastic(u):
        key = jax.random.key(int(np.random.default_rng().integers(1 << 30)))
        return decay(u) + jax.random.normal(key, (u.shape[0], 4))

    for f, fragment in [
        (ordered, "permuting the particles changed more than the order"),
        (normalized, "alongside different particles"),
        (per_particle, "returned shape"),
        (narrow, "float32"),
        (integer, "not a real floating"),
        (stochastic, "not deterministic"),
    ]:
        with pytest.raises(AssertionError, match=fragment):
            check_simulator(f, 2, 4)

    # The declaration suppresses exactly the two checks a stochastic simulator
    # cannot satisfy, and nothing else.
    check_simulator(stochastic, 2, 4, stochastic=True)
    for f, fragment in [
        (per_particle, "returned shape"),
        (narrow, "float32"),
        (integer, "not a real floating"),
    ]:
        with pytest.raises(AssertionError, match=fragment):
            check_simulator(f, 2, 4, stochastic=True)


def test_17_check_simulator_checks_the_second_size_and_the_argument():
    """The second ensemble size, the number of calls, and the argument.

    Ported from
    ``test_4_check_forward_model_checks_the_second_ensemble_size_and_the_argument``.
    A simulator that answers at one ensemble size and not another passes
    every other check, and the docstring promises five calls on
    ``jax.Array`` arguments.
    """
    decay, _ = _decay()

    def fixed_size(u):
        # A wrapper that preallocated for one ensemble size, as a subprocess
        # wrapper naturally does, and was then handed another.
        out = np.full((6, 4), np.nan)
        rows = min(u.shape[0], 6)
        out[:rows] = np.asarray(decay(u))[:rows]
        return out[: u.shape[0]]

    with pytest.raises(AssertionError, match="returned shape"):
        check_simulator(fixed_size, 2, 4)

    seen = []

    def recording(u):
        seen.append(u)
        return decay(u)

    check_simulator(recording, 2, 4)
    assert len(seen) == 5, "the docstring promises five calls"
    assert {tuple(a.shape) for a in seen} == {(6, 2), (7, 2), (2, 2)}
    for argument in seen:
        assert isinstance(argument, jax.Array)
        assert not np.asarray(argument).flags.writeable


def test_17_check_simulator_compares_where_the_failures_are():
    """Ported from ``test_4_check_forward_model_compares_where_the_failures_are``.

    The non-finite pattern is compared, not only the finite values: a domain
    that depends on the other particles is a coupling like any other.
    """
    decay, _ = _decay()

    def moving_domain(u):
        # Fails whichever particle has the smallest first input, which
        # depends on the company it is in.
        return decay(u).at[jnp.argmin(u[:, 0])].set(jnp.nan)

    with pytest.raises(AssertionError, match="non-finite entries"):
        check_simulator(moving_domain, 2, 4)


def test_17_check_simulator_accepts_a_failing_simulator_and_a_numpy_one():
    """Ported from ``test_4_check_forward_model_accepts_a_failing_model_and_a_numpy_one``.

    The failing simulator is the case the nan-aware comparison exists for:
    two permuted ``nan`` rows are not equal under the ordinary comparison, so
    a naive checker would reject every simulator that can fail.
    """
    failing = toy.restricted_decay()
    seed = 0  # four of six particles succeed
    # The first inputs check_simulator draws, for this seed.
    probe = jnp.asarray(np.random.default_rng(seed).normal(size=(6, 2)))
    outputs = np.asarray(failing.forward(probe))
    assert not np.isfinite(outputs).all(), "no particle failed, so this is vacuous"
    assert np.isfinite(outputs).any(), "every particle failed, so this is vacuous"
    check_simulator(failing.forward, 2, 12, seed=seed)

    times = np.linspace(0.25, 1.0, 4)

    def numpy_simulator(u):
        rows = np.asarray(u)  # a read-only view; only read
        return [[float(p[0] * np.exp(-p[1] * t)) for t in times] for p in rows]

    check_simulator(numpy_simulator, 2, 4)


def test_17_check_simulator_takes_several_inputs_and_outputs():
    def simulate(x, z):
        return {"sum": x[:, :2] + z, "x": 2.0 * x}

    check_simulator(simulate, (3, 2), {"sum": 2, "x": 3})

    def coupled(x, z):
        return {"sum": x[:, :2] + z, "x": x - jnp.mean(x, axis=0)}

    with pytest.raises(AssertionError, match="output 'x'"):
        check_simulator(coupled, [3, 2], {"sum": 2, "x": 3})


def test_17_check_simulator_honors_needs_key():
    """Determinism given the key is checked; row independence is skipped."""
    calls = []

    def keyed(key, u):
        calls.append(key)
        return u + jax.random.normal(key, u.shape, u.dtype)

    keyed.needs_key = True
    check_simulator(keyed, 3, 3)
    assert len(calls) == 3

    def ignores_its_key(key, u):
        key = jax.random.key(int(np.random.default_rng().integers(1 << 30)))
        return u + jax.random.normal(key, u.shape, u.dtype)

    ignores_its_key.needs_key = True
    with pytest.raises(AssertionError, match="not deterministic"):
        check_simulator(ignores_its_key, 3, 3)
    with pytest.raises(AssertionError, match="returned shape"):
        check_simulator(keyed, 3, 4)


def test_17_check_simulator_checks_a_blackbox_as_it_checks_a_function():
    """Its checks run inside the callback, so their failures are JAX's errors.

    A narrow BlackBox used to pass, its promotion warning unrecognized, and a
    wrong shape raised ``JaxRuntimeError`` rather than ``AssertionError``.
    """
    check_simulator(BlackBox(lambda u: 2.0 * np.asarray(u), 2), 2, 2)
    with pytest.raises(AssertionError, match="promoted"):
        check_simulator(BlackBox(lambda u: np.asarray(u, np.float32), 2), 2, 2)
    with pytest.raises(AssertionError, match="returned shape"):
        check_simulator(BlackBox(lambda u: np.asarray(u)[:, :1], 2), 2, 2)


def test_17_check_simulator_validates_its_own_arguments():
    with pytest.raises(TypeError, match="input_dims"):
        check_simulator(lambda u: u, 2.0, 2)
    with pytest.raises(ValueError, match="output_dims"):
        check_simulator(lambda u: u, 2, 0)


# ===========================================================================
# regressions ported from tests/test_toy.py
# ===========================================================================


def test_regression_check_simulator_rejects_a_coupling_in_a_small_output():
    """A per-element tolerance, not one global scale set by the largest value.

    Ported from
    ``test_12_regression_the_checker_rejects_a_coupling_in_a_small_observable``.
    With a single global scale, a simulator whose outputs span orders of
    magnitude could couple its small ones freely: a 50% coupling in an O(1)
    component was invisible beside a component of size 1e8, while the same
    coupling alone was caught with a margin of 6e5.
    """

    def mixed(u):
        big = 1e8 * u[:, 0]
        small = u[:, 1] + 0.5 * jnp.mean(u[:, 1])
        return jnp.stack([big, small], axis=-1)

    with pytest.raises(AssertionError, match="alongside different particles"):
        check_simulator(mixed, 2, 2)


def test_regression_check_simulator_refuses_too_few_particles():
    """At J < 3 both row-independence comparisons are vacuous, so it raises.

    Ported from ``test_12_regression_the_checker_refuses_an_ensemble_too_small_to_check``.
    At J = 1 an out-of-bounds subset index is silently clamped by JAX, making
    the comparison a tautology, and a definitively coupled model passed. At
    J = 2 the subset is one particle twice and a fair permutation is the
    identity for most seeds.
    """

    def coupled(u):
        return u - jnp.mean(u, axis=0)

    for n in (1, 2):
        with pytest.raises(ValueError, match="n_particles must be an int of at least 3"):
            check_simulator(coupled, 2, 2, n_particles=n)
    with pytest.raises(AssertionError):
        check_simulator(coupled, 2, 2, n_particles=3)


@pytest.mark.parametrize("seed", range(12))
def test_regression_check_simulator_permutation_is_never_the_identity(seed):
    """A fair draw returns the identity often at small J, asserting nothing.

    Ported from ``test_12_regression_the_permutation_is_never_the_identity``.
    """

    def ordered(u):
        return jnp.cumsum(u, axis=0)

    with pytest.raises(AssertionError, match="permuting the particles"):
        check_simulator(ordered, 2, 2, n_particles=3, seed=seed)


# ===========================================================================
# the user guide's page
# ===========================================================================


def test_the_user_guide_page_runs_and_says_what_it_does():
    """Every Python block of ``docs/user-guide/maps.md``, in order, in one namespace.

    The page's claims are then checked against what the blocks built, so a
    block that runs but no longer shows what the prose says fails here.
    """
    page = Path(__file__).parents[1] / "docs" / "user-guide" / "maps.md"
    blocks = re.findall(r"```python\n(.*?)```", page.read_text(), re.S)
    assert len(blocks) >= 8
    ns: dict = {}
    for block in blocks:
        exec(compile(block, str(page), "exec"), ns)

    assert ns["ens"].names == ("u", "g")
    assert ns["moved"].names == ("u", "g")
    assert ns["ens2"].names == ("u", "g", "rate")
    failed = ~np.asarray(ns["tried"].all_finite)
    assert failed.any() and not failed.all(), "the failure example is vacuous"
    np.testing.assert_array_equal(np.asarray(ns["kept"].weights)[failed], 0.0)
    _close(ns["traced"]["g"], ns["ens"]["g"])
    assert isinstance(ns["exact"], EnsembleGaussian)

    # The linearization is least squares of g on u, and residuals are its misfit.
    U, G = np.asarray(ns["ens"]["u"]), np.asarray(ns["ens"]["g"])
    Ua = np.column_stack([np.ones(len(U)), U])
    beta = np.linalg.lstsq(Ua, G, rcond=None)[0]
    _close(ns["fit"].map.op.to_dense(), beta[1:].T, factor=1e4)
    _close(ns["predicted"], Ua @ beta, factor=1e4)
    _close(ns["fit"].residuals, G - Ua @ beta, factor=1e4)
    assert type(ns["sampled"]) is EnsembleGaussian

    # The joint conditioned on y is the linear-Gaussian posterior.
    H, R = np.asarray(ns["H"]), 0.1 * np.eye(3)
    y = np.array([0.3, -0.2, 0.5])
    K = H.T @ np.linalg.inv(H @ H.T + R)
    post = ns["posterior"]
    _close(post.mean("x"), K @ y)
    _close(_dense_cov(post, "x"), np.eye(2) - K @ H)
