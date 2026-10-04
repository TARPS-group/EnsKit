r"""The conformance check for conditional maps."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..distribution import Ensemble, Gaussian, exact_moment_ensemble, reweight

__all__ = ["check_conditional_map"]

#: The name of the block the check adds to the samples, to see it passed through.
_EXTRA = "_check_conditional_map_extra"


#: Replicates averaged for the keyed check, and its standard-error multiple.
_N_REPLICATES = 256
_Z = 5.0


def check_conditional_map(
    cmap, joint: Gaussian, given, *, tol: float = 1e-8, keyed_noise_free: bool = False
) -> None:
    r"""Check a :class:`~enskit.distribution.ConditionalMap` against exact conditioning.

    ``cmap`` is a map built from the Gaussian ``joint`` for the blocks named
    ``given``, each of which has an independent term. The check draws samples
    of ``joint`` whose sample mean and covariance (divisor :math:`J - 1`)
    equal the joint's exactly, the given blocks with their noise
    (:func:`~enskit.distribution.exact_moment_ensemble`), and checks, in
    order:

    1. ``cmap.given`` names the blocks of ``given``, and ``cmap.targets`` is
       disjoint from it, all blocks of ``joint``.
    2. A missing value, a value for a block that is not given, and a value of
       the wrong shape each raise.
    3. On samples carrying an extra block and log weights, the result has the
       targets replaced, the given blocks dropped, the extra block unchanged
       and in position, the weights kept, and the samples' count; samples
       missing a target raise.
    4. Without a key, permuting the samples permutes the result exactly.
    5. The same key twice gives the same result.
    6. On the unweighted samples, the result's sample mean and covariance of
       the targets, cross-covariances included, match
       ``joint.condition(values)`` within ``tol`` relative to their scale:

       .. math::

           \lvert \hat m - m' \rvert \le \mathrm{tol}\,\max(1, \lvert m'\rvert_\infty),
           \qquad
           \lvert \hat C - C' \rvert \le \mathrm{tol}\,\max(1, \lvert C'\rvert_\infty).

    7. Only with ``keyed_noise_free``: called with a key on samples of the
       joint *without* the given blocks' noise, the result's sample moments,
       averaged over 256 keys, match the conditional's within five estimated
       standard errors plus ``tol``. This is how
       :class:`~enskit.distribution.MatheronMap` reads a key (the samples are
       noise-free and the map draws the noise), and what the stochastic
       update rule relies on; the protocol itself does not require it.

    An affine map, such as :class:`~enskit.distribution.MatheronMap`, is exact
    to round-off on such samples; an approximate map passes a looser ``tol``.

    Parameters
    ----------
    cmap : ConditionalMap
        The map to check.
    joint : ~enskit.distribution.Gaussian
        The joint the map was built from.
    given : str or sequence of str
        The given blocks' names.
    tol : float
        Keyword-only. The relative tolerance of obligations 6 and 7.
    keyed_noise_free : bool
        Keyword-only. Whether the map reads a key as "the samples are
        noise-free; draw the given blocks' noise", which obligation 7 then
        checks. Every given block must have a factor row.

    Raises
    ------
    AssertionError
        Naming the obligation that failed.
    """
    given = (given,) if isinstance(given, str) else tuple(given)
    rng = np.random.default_rng(31415)

    # 1. names
    _expect(
        set(cmap.given) == set(given) and len(cmap.given) == len(given),
        "1",
        f"cmap.given is {cmap.given}; the given blocks are {given}",
    )
    targets = tuple(cmap.targets)
    _expect(
        not set(targets) & set(given),
        "1",
        f"cmap.targets {targets} overlaps the given blocks {given}",
    )
    _expect(
        all(n in joint.names for n in targets + given),
        "1",
        f"cmap names blocks that are not in the joint {joint.names}",
    )

    sources = joint.latent_dim + sum(
        joint.block_cov(n).factor().shape[1]
        for n in joint.names
        if joint.block_cov(n) is not None
    )
    n = sources + 8
    samples = exact_moment_ensemble(jax.random.key(0), joint, n)
    dtype = samples[samples.names[0]].dtype
    values = {
        c: joint.mean(c) + jnp.asarray(rng.normal(size=joint.dims[c]), dtype)
        for c in given
    }

    # 2. values
    first = given[0]
    _raises(
        lambda: cmap(samples, {c: v for c, v in values.items() if c != first}),
        "2",
        "a missing value",
    )
    _raises(
        lambda: cmap(samples, {**values, targets[0]: samples[targets[0]][0]}),
        "2",
        "a value for a block that is not given",
    )
    _raises(
        lambda: cmap(samples, {**values, first: jnp.zeros(joint.dims[first] + 1, dtype)}),
        "2",
        "a value of the wrong shape",
    )

    # 3. structure, on weighted samples carrying an extra block
    extra = jnp.asarray(rng.normal(size=(n, 2)), dtype)
    weighted = reweight(
        samples.assign({_EXTRA: extra}), jnp.asarray(rng.normal(size=n), dtype)
    )
    out = _call(lambda: cmap(weighted, values), "3")
    want_names = tuple(b for b in weighted.names if b not in given)
    _expect(isinstance(out, Ensemble), "3", f"the map returned {type(out).__name__}")
    _expect(
        out.names == want_names,
        "3",
        f"the result's blocks are {out.names}; expected {want_names}: the targets "
        f"replaced, the given blocks dropped, other blocks in position",
    )
    _expect(
        out.n_particles == n, "3", f"the result has {out.n_particles} samples, not {n}"
    )
    _expect(
        bool(jnp.array_equal(out[_EXTRA], extra)),
        "3",
        "a block that is neither given nor a target was changed",
    )
    _expect(
        out.is_weighted and bool(jnp.array_equal(out.log_weights, weighted.log_weights)),
        "3",
        "the weights were not kept",
    )
    _raises(
        lambda: cmap(samples.drop(targets[0]), values), "3", "samples missing a target"
    )

    # 4. pointwise
    plain = _call(lambda: cmap(samples, values), "4")
    perm = rng.permutation(n)
    permuted = Ensemble({b: samples[b][perm] for b in samples.names})
    moved = _call(lambda: cmap(permuted, values), "4")
    for b in targets:
        _expect(
            bool(jnp.array_equal(moved[b], plain[b][perm])),
            "4",
            f"permuting the samples did not permute block {b!r} of the result: the "
            f"map couples samples",
        )

    # 5. keys
    key = jax.random.key(7)
    a = _call(lambda: cmap(samples, values, key=key), "5")
    b = _call(lambda: cmap(samples, values, key=key), "5")
    _expect(
        all(bool(jnp.array_equal(a[t], b[t])) for t in targets),
        "5",
        "the same key twice gave different results",
    )

    # 6. agreement with condition
    exact = joint.condition(values)
    want_m = np.concatenate([np.asarray(exact.mean(t), np.float64) for t in targets])
    want_C = np.block(
        [
            [np.asarray(exact.cov(s, t).to_dense(), np.float64) for t in targets]
            for s in targets
        ]
    )
    x = np.concatenate([np.asarray(plain[t], np.float64) for t in targets], axis=1)
    dev = x - x.mean(axis=0)
    got_m, got_C = x.mean(axis=0), dev.T @ dev / (n - 1)
    err_m = np.abs(got_m - want_m).max()
    err_C = np.abs(got_C - want_C).max()
    _expect(
        err_m <= tol * max(1.0, np.abs(want_m).max()),
        "6",
        f"the transported sample mean differs from the conditional mean by {err_m:.3g}",
    )
    _expect(
        err_C <= tol * max(1.0, np.abs(want_C).max()),
        "6",
        f"the transported sample covariance differs from the conditional covariance "
        f"by {err_C:.3g}",
    )

    # 7. the keyed path, on noise-free samples
    if not keyed_noise_free:
        return
    rowless = tuple(c for c in given if joint.factor(c) is None)
    _expect(
        not rowless,
        "7",
        f"keyed_noise_free needs every given block to have a factor row; {rowless} "
        f"have none",
    )
    noise_free = Gaussian(
        {b: joint.mean(b) for b in joint.names},
        factors={b: joint.factor(b) for b in joint.names if joint.factor(b) is not None},
        block_covs={
            b: joint.block_cov(b)
            for b in joint.names
            if b not in given and joint.block_cov(b) is not None
        },
        latent_dim=joint.latent_dim,
    )
    clean = exact_moment_ensemble(jax.random.key(1), noise_free, n)
    ms, Cs = [], []
    for k in jax.random.split(jax.random.key(2), _N_REPLICATES):
        out = _call(lambda k=k: cmap(clean, values, key=k), "7")
        x = np.concatenate([np.asarray(out[t], np.float64) for t in targets], axis=1)
        dev = x - x.mean(axis=0)
        ms.append(x.mean(axis=0))
        Cs.append(dev.T @ dev / (n - 1))
    for what, vals, want in (
        ("mean", np.stack(ms), want_m),
        ("covariance", np.stack(Cs), want_C),
    ):
        avg = vals.mean(axis=0)
        se = vals.std(axis=0, ddof=1) / np.sqrt(len(vals))
        excess = np.abs(avg - want) - (_Z * se + tol * max(1.0, np.abs(want).max()))
        _expect(
            np.all(excess <= 0),
            "7",
            f"with a key on noise-free samples, the sample {what} averaged over "
            f"{len(vals)} keys differs from the conditional {what} by more than "
            f"{_Z:g} standard errors (worst excess {excess.max():.3g})",
        )


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------


def _expect(ok, obligation: str, message: str) -> None:
    if not bool(ok):
        raise AssertionError(f"check_conditional_map, obligation {obligation}: {message}")


def _call(f, obligation: str):
    try:
        return f()
    except Exception as e:  # noqa: BLE001
        raise AssertionError(
            f"check_conditional_map, obligation {obligation}: the map raised on valid "
            f"arguments: {e!r}"
        ) from e


def _raises(f, obligation: str, what: str) -> None:
    try:
        f()
    except (ValueError, KeyError, TypeError):
        return
    raise AssertionError(
        f"check_conditional_map, obligation {obligation}: {what} did not raise"
    )
