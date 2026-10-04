"""Helpers shared by the Kalman layer's modules.

Private to :mod:`enskit.kalman`: block arguments, keys, the family guard, the
build arguments' checks, the debug-mode checks on particles, the alignment
check, and the pytree registration of the layer's classes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp

from ..distribution import Ensemble, EnsembleGaussian, Gaussian
from ..linalg import value_check

#: The remedies for a failed particle, as every message states them.
FAILED_REMEDY = (
    "Failed particles are found with Ensemble.all_finite. Repair them, or drop "
    "them with resample(key, reweight(ensemble, jnp.where(ensemble.all_finite, "
    "0.0, -jnp.inf))), which returns an unweighted ensemble of finite particles."
)

#: Why a weighted ensemble is refused, and what to do.
WEIGHTED_REMEDY = (
    "Updates take unweighted particles: a weighted ensemble's projection is not "
    "aligned with its particles. Resample first, with "
    "enskit.distribution.resample."
)


def pytree_class(*, data: tuple[str, ...], meta: tuple[str, ...]):
    """Register a class as a pytree with explicitly declared fields.

    ``data`` fields are children, which may be any pytrees (distributions and
    maps included); ``meta`` fields are static and hashable. Unflattening
    bypasses the constructor.
    """

    def register(cls):
        def flatten(obj):
            return (
                [getattr(obj, n) for n in data],
                tuple(getattr(obj, n) for n in meta),
            )

        def unflatten(aux, children):
            obj = object.__new__(cls)
            for name, value in zip(data, children, strict=True):
                object.__setattr__(obj, name, value)
            for name, value in zip(meta, aux, strict=True):
                object.__setattr__(obj, name, value)
            return obj

        jax.tree_util.register_pytree_node(cls, flatten, unflatten)
        return cls

    return register


def build(cls: type, **fields):
    """Build an instance without running a constructor."""
    obj = object.__new__(cls)
    for name, value in fields.items():
        object.__setattr__(obj, name, value)
    return obj


def guard(obj, where: str) -> None:
    """Refuse a vmapped family, before any other check."""
    batch = getattr(obj, "batch_shape", ())
    if batch != ():
        raise ValueError(
            f"{where}: {obj!r} is a vmapped family with batch shape {batch}, which "
            f"cannot be used directly; apply it under jax.vmap, one member at a time."
        )


def merge_blocks(where: str, mapping, kwargs: dict) -> dict:
    """Merge a positional mapping and keywords into one ordered dict."""
    out: dict = {}
    if mapping is not None:
        if not isinstance(mapping, Mapping):
            raise TypeError(
                f"{where}: the positional argument must be a mapping from block "
                f"name to value, got {type(mapping).__name__}"
            )
        for name, value in mapping.items():
            if not isinstance(name, str):
                raise TypeError(
                    f"{where}: block names must be str, got {type(name).__name__}"
                )
            out[name] = value
    for name, value in kwargs.items():
        if name in out:
            raise TypeError(
                f"{where}: block {name!r} is given twice, in the mapping and as a keyword"
            )
        out[name] = value
    return out


def check_key(where: str, key, *, required: bool, why: str = "") -> None:
    """Check a key's type, then its presence where one is needed."""
    if key is not None and not (
        isinstance(key, jax.Array) and jnp.issubdtype(key.dtype, jax.dtypes.prng_key)
    ):
        raise TypeError(
            f"{where}: key must be a typed key from jax.random.key, got "
            f"{getattr(key, 'dtype', type(key).__name__)}. A raw uint32 key from "
            f"jax.random.PRNGKey is refused: its trailing shape makes a family of "
            f"keys ambiguous."
        )
    if key is None and required:
        raise ValueError(f"{where}: a key is required{why}")


def name_list(where: str, names: tuple[str, ...], selected) -> tuple[str, ...]:
    """Normalize a ``str`` or a sequence of ``str``: each a block, none repeated.

    Returns the names in the order given.
    """
    if isinstance(selected, str):
        selected = (selected,)
    try:
        selected = tuple(selected)
    except TypeError:
        raise TypeError(
            f"{where}: expected a block name or a sequence of block names, got "
            f"{type(selected).__name__}"
        ) from None
    for name in selected:
        if not isinstance(name, str):
            raise TypeError(
                f"{where}: block names must be str, got {type(name).__name__}"
            )
        if name not in names:
            raise KeyError(f"{where}: no block named {name!r}; the blocks are {names}")
    if len(set(selected)) != len(selected):
        repeated = next(n for n in selected if selected.count(n) > 1)
        raise ValueError(f"{where}: block {repeated!r} is named more than once")
    return selected


def check_ensemble(where: str, ensemble, what: str = "ensemble") -> None:
    """An :class:`Ensemble`, not a vmapped family."""
    if not isinstance(ensemble, Ensemble):
        raise TypeError(
            f"{where}: {what} must be an Ensemble, got {type(ensemble).__name__}"
        )


def check_unweighted(where: str, ensemble: Ensemble, what: str = "the ensemble") -> None:
    if ensemble.is_weighted:
        raise ValueError(f"{where}: {what} is weighted. {WEIGHTED_REMEDY}")


def check_build_arguments(where: str, particles, approximation, given):
    """The build arguments of a shipped rule; returns ``(given, targets)``.

    Both in the approximation's block order. Runs steps 1 to 6 of the
    contract's build checks; tier 4 is :func:`check_particles_finite` and
    :func:`check_alignment`, called by the rule after its own checks.
    """
    guard(particles, where)
    guard(approximation, where)
    check_ensemble(where, particles, "particles")
    if not isinstance(approximation, Gaussian):
        raise TypeError(
            f"{where}: approximation must be a Gaussian, got "
            f"{type(approximation).__name__}"
        )
    selected = name_list(where, approximation.names, given)
    if not selected:
        raise ValueError(f"{where}: at least one given block is required")
    check_unweighted(where, particles, "particles")
    if set(approximation.names) != set(particles.names):
        raise ValueError(
            f"{where}: the approximation's blocks {approximation.names} must be the "
            f"particles' blocks {particles.names}"
        )
    for name in approximation.names:
        if approximation.dims[name] != particles.dims[name]:
            raise ValueError(
                f"{where}: block {name!r} has dimension {approximation.dims[name]} in "
                f"the approximation and {particles.dims[name]} in the particles"
            )
    a_dtype = approximation.mean(approximation.names[0]).dtype
    p_dtype = particles[particles.names[0]].dtype
    if a_dtype != p_dtype:
        raise TypeError(
            f"{where}: the approximation has dtype {a_dtype}, the particles {p_dtype}"
        )
    given = tuple(n for n in approximation.names if n in selected)
    targets = tuple(n for n in approximation.names if n not in selected)
    if not targets:
        raise ValueError(
            f"{where}: every block is given; at least one target block must remain"
        )
    return given, targets


def is_aligned_type(particles: Ensemble, approximation: Gaussian) -> bool:
    """The approximation's type promises alignment with these particles."""
    return (
        isinstance(approximation, EnsembleGaussian)
        and approximation.n_particles == particles.n_particles
    )


def check_particles_finite(where: str, ensemble: Ensemble, names=None) -> None:
    """Debug-mode check: every particle of positive weight is finite."""
    names = ensemble.names if names is None else names
    for name in names:
        x = ensemble[name]

        def failed(a, name=name):
            out = ~jnp.all(jnp.isfinite(a), axis=-1)
            if ensemble.is_weighted:
                out = out & (ensemble.weights > 0)
            return out

        lazy_value_check(
            x,
            lambda a, failed=failed: ~jnp.any(failed(a)),
            lambda x=x, name=name, failed=failed: (
                f"{where}: {int(jnp.sum(failed(x)))} particle(s) "
                f"{'of positive weight ' if ensemble.is_weighted else ''}are not "
                f"finite in block {name!r}. {FAILED_REMEDY}"
            ),
        )


def check_alignment(where: str, particles: Ensemble, approximation) -> None:
    """The alignment check: the approximation realizes these particles.

    Debug mode only, on concrete arrays: the realization runs inside the
    check's predicate, so it is neither computed with checks off nor read
    inside a trace.
    """
    J = particles.n_particles
    cache: list = []

    def measure():
        if not cache:
            termed = tuple(
                n for n in approximation.names if approximation.block_cov(n) is not None
            )
            realized = approximation.realize_particles(exclude_block_covs=termed)
            # The round-off of a block built from another (a linear map of an
            # aligned block) scales with the source's magnitude, so the
            # magnitude term takes the largest over every block.
            size = jnp.max(
                jnp.stack([jnp.max(jnp.abs(particles[n])) for n in particles.names])
            )
            eps = float(jnp.finfo(particles[particles.names[0]].dtype).eps)
            for name in approximation.names:
                gap = jnp.max(jnp.abs(realized[name] - particles[name]))
                spread = jnp.max(jnp.abs(particles.anomalies(name)))
                bound = math.sqrt(eps) * spread + 64 * J * eps * size
                cache.append((name, gap, bound))
        return cache

    def message():
        name, gap, bound = next((n, g, b) for n, g, b in measure() if not bool(g <= b))
        return (
            f"{where}: the approximation is an EnsembleGaussian with the particles' "
            f"count, but it is not aligned with them: its realized particles differ "
            f"from the particles in block {name!r} by {float(gap):.3g}, beyond the "
            f"round-off bound {float(bound):.3g}. Build it with "
            f"gaussian_approximation from these particles, or pass a modified "
            f"approximation as a plain Gaussian. To inflate the spread, apply "
            f"inflate_multiplicative to the particles before the update."
        )

    lazy_value_check(
        particles[particles.names[0]],
        lambda _: jnp.all(jnp.stack([g <= b for _, g, b in measure()])),
        message,
    )


def check_scalar(where: str, name: str, value, dtype):
    """A real scalar (a Python number or a 0-d array), in ``dtype``."""
    arr = jnp.asarray(value)
    if not (
        jnp.issubdtype(arr.dtype, jnp.floating) or jnp.issubdtype(arr.dtype, jnp.integer)
    ):
        raise ValueError(f"{where}: {name} must be a real scalar, got dtype {arr.dtype}")
    if arr.ndim != 0:
        raise ValueError(
            f"{where}: {name} must be a scalar, got shape {arr.shape}. An array would "
            f"broadcast and scale each coordinate by a different factor; a family of "
            f"scalars is a jax.vmap over this call."
        )
    return arr.astype(dtype)


def result_check(where: str, what: str, value) -> None:
    """Debug-mode check that an updated block is finite."""
    value_check(
        value,
        lambda a: bool(jnp.all(jnp.isfinite(a))),
        f"{where}: the {what} must be finite. The inputs were finite, so this means "
        f"overflow in applying a target factor row or a decomposition that failed.",
    )


def lazy_value_check(x, predicate, message) -> None:
    """:func:`~enskit.linalg.value_check` with a message built only on failure."""
    failed = []

    def check(a):
        outcome = predicate(a)
        if isinstance(outcome, jax.core.Tracer):
            return outcome
        if not bool(outcome):
            failed.append(True)
        return True

    value_check(x, check, "")
    if failed:
        raise ValueError(message())


def safe_repr(build_text, type_name: str, batch_shape) -> str:
    """Render a repr, wrapping a family; a marker form if sizes are unreadable."""
    try:
        base = build_text()
        batch = batch_shape()
    except Exception:
        return f"<{type_name} (unprintable leaves)>"
    return f"vmapped({base}, batch={batch})" if batch != () else base
