"""Helpers shared by the algorithm layer's modules.

Private to :mod:`enskit.algorithms`: the pytree registration of the layer's
classes, the family guard, typed-key checks, and the static check every
policy's output goes through.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..distribution import Ensemble


def pytree_class(*, data: tuple[str, ...], meta: tuple[str, ...]):
    """Register a class as a pytree with explicitly declared fields.

    ``data`` fields are children, which may be any pytrees (ensembles and
    operators included); ``meta`` fields are static and hashable.
    Unflattening bypasses the constructor.
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

        def setattr_frozen(self, name, value):
            raise AttributeError(f"{cls.__name__} is immutable; cannot set {name!r}")

        jax.tree_util.register_pytree_node(cls, flatten, unflatten)
        if "__setattr__" not in cls.__dict__:
            cls.__setattr__ = setattr_frozen
        return cls

    return register


def set_fields(obj, **fields) -> None:
    """Set fields on an immutable instance, from its own constructor."""
    for name, value in fields.items():
        object.__setattr__(obj, name, value)


def guard(obj, where: str) -> None:
    """Refuse a vmapped family, before any other check."""
    batch = getattr(obj, "batch_shape", ())
    if batch != ():
        raise ValueError(
            f"{where}: {obj!r} is a vmapped family with batch shape {batch}, which "
            f"cannot be used directly; apply it under jax.vmap, one member at a time."
        )


def broadcast_batch(where: str, *shapes: tuple[int, ...]) -> tuple[int, ...]:
    """The broadcast of several leaves' batch shapes; ``ValueError`` on a mismatch."""
    try:
        return tuple(jnp.broadcast_shapes(*shapes)) if shapes else ()
    except ValueError as exc:
        raise ValueError(
            f"{where}: the fields' batch shapes {shapes} do not broadcast"
        ) from exc


def is_typed_key(key) -> bool:
    """Whether ``key`` is a typed key from :func:`jax.random.key`."""
    return isinstance(key, jax.Array) and jnp.issubdtype(
        key.dtype, jax.dtypes.prng_key
    )


def check_key(where: str, key) -> None:
    """Require a single typed key."""
    if not is_typed_key(key):
        raise TypeError(
            f"{where}: key must be a typed key from jax.random.key, got "
            f"{getattr(key, 'dtype', type(key).__name__)}. A raw uint32 key from "
            f"jax.random.PRNGKey has shape (2,), which would make a family of "
            f"keys ambiguous; convert it with jax.random.wrap_key_data."
        )
    if key.shape != ():
        raise ValueError(
            f"{where}: key must be a single key of shape (), got {key.shape}"
        )


def check_policy_output(where: str, who: str, got, like: Ensemble) -> None:
    """A policy's ensemble must match the one it replaces, statically.

    Same type, unweighted, the same blocks in the same order, the same
    dimensions, particle count and dtype. A policy returning another dtype
    would demote every later step with nothing else raising.
    """
    if not isinstance(got, Ensemble):
        raise TypeError(
            f"{where}: {who} must return an Ensemble, got {type(got).__name__}"
        )
    if got.batch_shape != ():
        raise ValueError(f"{where}: {who} returned a vmapped family, {got!r}")
    if got.is_weighted:
        raise ValueError(f"{where}: {who} returned a weighted ensemble; it must not")
    if got.names != like.names or got.dims != like.dims:
        raise ValueError(
            f"{where}: {who} returned blocks {got.dims}, expected {like.dims} in "
            f"that order"
        )
    if got.n_particles != like.n_particles:
        raise ValueError(
            f"{where}: {who} returned {got.n_particles} particles, expected "
            f"{like.n_particles}"
        )
    want = like[like.names[0]].dtype
    for name in got.names:
        if got[name].dtype != want:
            raise TypeError(
                f"{where}: {who} returned block {name!r} of dtype {got[name].dtype}, "
                f"expected {want}. A policy that changes the dtype demotes every "
                f"later step, and every later check still passes at its own "
                f"tolerance."
            )


def safe_repr(build_text, type_name: str, batch_shape) -> str:
    """``build_text()``, wrapped as a family when batched; never raises."""
    try:
        text = build_text()
        batch = batch_shape()
    except Exception:
        return f"<{type_name} (unprintable)>"
    return f"vmapped({text}, batch={batch})" if batch != () else text
