r"""The conformance check for update rules."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..distribution import Ensemble, Gaussian, exact_moment_ensemble
from ..kalman import gaussian_approximation
from ..linalg import DensePSD

__all__ = ["check_update_rule"]

#: Replicates averaged for a stochastic rule's moments.
_N_REPLICATES = 256

#: How many estimated standard errors a stochastic average may stray.
_Z = 5.0


def check_update_rule(rule, *, key=None, exact_covariance: bool = True) -> None:
    r"""Check an :class:`~enskit.kalman.UpdateRule` against the update contract.

    Builds a linear-Gaussian fixture: a parameter block ``"u"`` of dimension
    3 with a Gaussian prior, a second target ``"v" = B u`` of dimension 2,
    and a given block ``"g" = H u`` of dimension 4 observed with noise
    :math:`R`; :math:`J = 12` particles whose sample moments equal the joint's
    exactly; the approximation
    :func:`~enskit.kalman.gaussian_approximation`; and the exact conditional
    of ``("u", "v")`` at a value :math:`y^*`, from
    :meth:`~enskit.distribution.Gaussian.condition`. Then checks, in order:

    1. ``rule.build(particles, approximation, ("g",))`` returns a callable
       whose result is an unweighted :class:`~enskit.distribution.Ensemble`
       over ``("u", "v")``, with :math:`J` particles, the particles' dtype and
       finite values; a float32 fixture stays float32; a missing value, a
       value for a block that is not given and a misshapen value each raise.
    2. Whether the update is stochastic (two keys give different results);
       if so, the same key twice gives the same result, and a call without a
       key raises ``ValueError``.
    3. One build called at two values equals a fresh build at each.
    4. The sample mean of ``("u", "v")`` equals the exact conditional mean,
       and, when ``exact_covariance``, the sample covariance (divisor
       :math:`J - 1`) the exact conditional covariance: to round-off for a
       deterministic update; for a stochastic one, the average over 256 keys
       within five estimated standard errors.
    5. Building and calling under :func:`jax.jit` agrees with the eager
       result, and :func:`jax.vmap` over two values agrees with a loop.

    The fixture has one given block and no target with an independent term,
    so the check says nothing about how a rule treats several given blocks or
    draws a target's term; test those with the rule's own tests.

    Parameters
    ----------
    rule : UpdateRule
        The rule to check.
    key : jax.random key, optional
        Keyword-only. Seeds the keys passed to the update; a fixed key by
        default, so the check is reproducible.
    exact_covariance : bool
        Keyword-only. Whether the rule claims the exact conditional
        covariance on a linear-Gaussian problem. The deterministic EnKF, for
        instance, does not.

    Raises
    ------
    AssertionError
        Naming the obligation that failed.
    """
    key = jax.random.key(20261004) if key is None else key
    fix = _fixture(jnp.float64)
    keys = jax.random.split(key, 4 + _N_REPLICATES)

    # 1. structure
    upd = _build(rule, fix)
    out = _call(upd, fix["y"], keys[0], "1")
    _check_structure(out, fix, "1")
    fix32 = _fixture(jnp.float32)
    out32 = _call(_build(rule, fix32), fix32["y"], keys[0], "1")
    _check_structure(out32, fix32, "1 (float32)")

    for bad, what in (
        ({}, "a missing value"),
        ({"g": fix["y"], "u": fix["y"][:3]}, "a value for a block that is not given"),
        ({"g": fix["y"][:-1]}, "a value of the wrong shape"),
    ):
        try:
            upd(bad, key=keys[0])
        except (ValueError, KeyError, TypeError):
            continue
        raise AssertionError(f"check_update_rule, obligation 1: {what} did not raise")

    # 2. keys
    other = _call(upd, fix["y"], keys[1], "2")
    stochastic = not _same(out, other)
    if stochastic:
        again = _call(upd, fix["y"], keys[0], "2")
        _expect(_same(out, again), "2", "the same key twice gave different results")
        try:
            upd({"g": fix["y"]})
        except ValueError:
            pass
        else:
            raise AssertionError(
                "check_update_rule, obligation 2: a stochastic update called without "
                "a key did not raise ValueError"
            )

    # 3. value-free builds
    y_b = fix["y"] + 0.5
    reused = (_call(upd, fix["y"], keys[2], "3"), _call(upd, y_b, keys[2], "3"))
    fresh = (
        _call(_build(rule, fix), fix["y"], keys[2], "3"),
        _call(_build(rule, fix), y_b, keys[2], "3"),
    )
    for a, b in zip(reused, fresh, strict=True):
        _expect(
            _close(a, b, 1e3),
            "3",
            "a build called at a second value disagrees with a fresh build at that "
            "value, so the build depends on the values",
        )

    # 4. exactness
    want_m, want_C = fix["mean"], fix["cov"]
    scale_m = max(1.0, float(np.abs(want_m).max()))
    scale_C = max(1.0, float(np.abs(want_C).max()))
    if not stochastic:
        m, C = _moments(out)
        _expect(
            np.abs(m - want_m).max() <= 1e-8 * scale_m,
            "4",
            f"the sample mean differs from the exact conditional mean by "
            f"{np.abs(m - want_m).max():.3g}",
        )
        if exact_covariance:
            _expect(
                np.abs(C - want_C).max() <= 1e-8 * scale_C,
                "4",
                f"the sample covariance differs from the exact conditional covariance "
                f"by {np.abs(C - want_C).max():.3g}",
            )
    else:
        reps = [_moments(_call(upd, fix["y"], k, "4")) for k in keys[4:]]
        ms = np.stack([r[0] for r in reps])
        Cs = np.stack([r[1] for r in reps])
        for what, vals, want, scale, check in (
            ("mean", ms, want_m, scale_m, True),
            ("covariance", Cs, want_C, scale_C, exact_covariance),
        ):
            if not check:
                continue
            avg = vals.mean(axis=0)
            se = vals.std(axis=0, ddof=1) / np.sqrt(len(vals))
            excess = np.abs(avg - want) - (_Z * se + 1e-8 * scale)
            _expect(
                np.all(excess <= 0),
                "4",
                f"the sample {what} averaged over {len(vals)} keys differs from the "
                f"exact conditional {what} by more than {_Z:g} standard errors "
                f"(worst excess {excess.max():.3g})",
            )

    # 5. jit and vmap
    def run(y, k):
        return _build(rule, fix)({"g": y}, key=k)

    try:
        jitted = jax.jit(run)(fix["y"], keys[3])
    except AssertionError:
        raise
    except Exception as e:  # noqa: BLE001 -- any failure to trace is the finding
        raise AssertionError(
            f"check_update_rule, obligation 5: jax.jit failed: {e!r}"
        ) from e
    _expect(
        _close(jitted, _call(upd, fix["y"], keys[3], "5"), 1e6),
        "5",
        "the jit-compiled build and call disagree with the eager result",
    )
    ys = jnp.stack([fix["y"], y_b])
    try:
        batched = jax.vmap(lambda y: run(y, keys[3]))(ys)
    except Exception as e:  # noqa: BLE001
        raise AssertionError(
            f"check_update_rule, obligation 5: jax.vmap failed: {e!r}"
        ) from e
    for i, y in enumerate((fix["y"], y_b)):
        single = _call(upd, y, keys[3], "5")
        member = jax.tree_util.tree_map(lambda leaf, i=i: leaf[i], batched)
        _expect(
            _close(member, single, 1e6),
            "5",
            "jax.vmap over the values disagrees with a loop",
        )


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------


def _fixture(dtype) -> dict:
    """The linear-Gaussian fixture, its particles, approximation and answer."""
    rng = np.random.default_rng(1729)
    P, Dv, N, J = 3, 2, 4, 12
    L0 = rng.normal(size=(P, P)) + 2 * np.eye(P)
    m0 = rng.normal(size=P)
    B = rng.normal(size=(Dv, P))
    H = rng.normal(size=(N, P))
    M = rng.normal(size=(N, N))
    R = 0.25 * (M @ M.T / N + np.eye(N))
    y = H @ (m0 + L0 @ rng.normal(size=P)) + rng.normal(size=N)

    def arr(a):
        return jnp.asarray(a, dtype)

    joint = Gaussian(
        {"u": arr(m0), "v": arr(B @ m0), "g": arr(H @ m0)},
        factors={"u": arr(L0), "v": arr(B @ L0), "g": arr(H @ L0)},
    )
    particles = exact_moment_ensemble(jax.random.key(0), joint, J)
    noise = DensePSD(arr(R))
    approximation = gaussian_approximation(particles, {"g": noise})
    exact = joint.add_noise(g=noise).condition(g=arr(y))
    mean = np.concatenate([np.asarray(exact.mean(n), np.float64) for n in ("u", "v")])
    cov = np.block(
        [
            [np.asarray(exact.cov(a, b).to_dense(), np.float64) for b in ("u", "v")]
            for a in ("u", "v")
        ]
    )
    return {
        "particles": particles,
        "approximation": approximation,
        "y": arr(y),
        "mean": mean,
        "cov": cov,
        "dtype": jnp.dtype(dtype),
        "J": J,
    }


def _expect(ok, obligation: str, message: str) -> None:
    if not bool(ok):
        raise AssertionError(f"check_update_rule, obligation {obligation}: {message}")


def _build(rule, fix):
    try:
        upd = rule.build(fix["particles"], fix["approximation"], ("g",))
    except Exception as e:  # noqa: BLE001
        raise AssertionError(
            f"check_update_rule, obligation 1: build raised on a valid fixture: {e!r}"
        ) from e
    _expect(callable(upd), "1", f"build returned {type(upd).__name__}, not a callable")
    return upd


def _call(upd, y, key, obligation: str):
    try:
        return upd({"g": y}, key=key)
    except Exception as e:  # noqa: BLE001
        raise AssertionError(
            f"check_update_rule, obligation {obligation}: calling the built update "
            f"raised: {e!r}"
        ) from e


def _check_structure(out, fix, obligation: str) -> None:
    _expect(
        isinstance(out, Ensemble),
        obligation,
        f"the update returned {type(out).__name__}, not an Ensemble",
    )
    _expect(
        out.names == ("u", "v"),
        obligation,
        f"the result's blocks are {out.names}; the targets, in the approximation's "
        f"order, are ('u', 'v')",
    )
    _expect(not out.is_weighted, obligation, "the result is weighted")
    _expect(
        out.n_particles == fix["J"],
        obligation,
        f"the result has {out.n_particles} particles, not {fix['J']}",
    )
    for name in out.names:
        _expect(
            out[name].dtype == fix["dtype"],
            obligation,
            f"block {name!r} has dtype {out[name].dtype}; the particles have "
            f"{fix['dtype']}",
        )
        _expect(
            bool(jnp.all(jnp.isfinite(out[name]))),
            obligation,
            f"block {name!r} is not finite",
        )


def _same(a: Ensemble, b: Ensemble) -> bool:
    return all(bool(jnp.array_equal(a[n], b[n])) for n in a.names)


def _close(a: Ensemble, b: Ensemble, factor: float) -> bool:
    eps = float(jnp.finfo(a[a.names[0]].dtype).eps)
    for n in a.names:
        x, z = np.asarray(a[n]), np.asarray(b[n])
        if np.abs(x - z).max() > factor * eps * max(1.0, np.abs(z).max()):
            return False
    return True


def _moments(out: Ensemble):
    x = np.concatenate([np.asarray(out[n], np.float64) for n in ("u", "v")], axis=1)
    a = x - x.mean(axis=0)
    return x.mean(axis=0), a.T @ a / (x.shape[0] - 1)
