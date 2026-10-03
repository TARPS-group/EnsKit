"""Prototype maps layer: structured maps and pushforward."""
from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .distribution import Ensemble, Gaussian, _pytree
from .linalg import Dense, LinOp, PSDLinOp


@_pytree(data=("op", "shift"), meta=())
class Linear:
    """x -> A x + shift, with A a LinOp."""

    def __init__(self, op: LinOp, shift=None):
        object.__setattr__(self, "op", op if isinstance(op, LinOp) else Dense(jnp.asarray(op)))
        object.__setattr__(self, "shift", None if shift is None else jnp.asarray(shift))

    def __call__(self, x):
        y = self.op.matvec(x)
        return y if self.shift is None else y + self.shift


@_pytree(data=("cov",), meta=())
class AdditiveNoise:
    """x -> x + e, e ~ N(0, cov), independent of x and of everything else."""

    def __init__(self, cov: PSDLinOp):
        object.__setattr__(self, "cov", cov)


class BlackBox:
    """A host-side simulator made traceable: pure_callback + zero derivative."""

    def __init__(self, f: Callable, output_dim: int, *, vectorized: bool = True):
        self.f, self.output_dim, self.vectorized = f, output_dim, vectorized

    def __call__(self, *xs):
        n = xs[0].shape[0]
        shape = jax.ShapeDtypeStruct((n, self.output_dim), xs[0].dtype)
        f = self.f

        @jax.custom_jvp
        def call(*xs):
            return jax.pure_callback(lambda *a: np.asarray(f(*a), dtype=shape.dtype), shape, *xs,
                                    vmap_method="sequential")

        @call.defjvp
        def _(primals, tangents):
            return call(*primals), jnp.zeros(shape.shape, shape.dtype)

        return call(*xs)


def _names(x) -> tuple[str, ...]:
    return (x,) if isinstance(x, str) else tuple(x)


def pushforward(dist, f, *, output: str, inputs: str | Sequence[str] | None = None, key=None):
    """Add (or replace) block ``output`` = f(inputs) to an Ensemble or a Gaussian."""
    inputs = dist.names if inputs is None else _names(inputs)
    if output in dist.names and output not in inputs:
        raise ValueError(f"pushforward: block {output!r} exists and is not an input; "
                         f"drop it first")
    if isinstance(dist, Ensemble):
        return _push_ensemble(dist, f, inputs, output, key)
    if isinstance(dist, Gaussian):
        return _push_gaussian(dist, f, inputs, output, key)
    raise TypeError(f"pushforward: unsupported distribution {type(dist).__name__}")


def _push_ensemble(ens, f, inputs, output, key):
    xs = [ens[n] for n in inputs]
    if isinstance(f, AdditiveNoise):
        if key is None:
            raise ValueError("pushforward: AdditiveNoise on an Ensemble needs a key")
        (x,) = xs
        L = f.cov.factor()
        y = x + L.matvec(jax.random.normal(key, (ens.n_particles, L.shape[1])))
    else:
        y = f(*xs)
    y = jnp.asarray(y)
    if y.ndim != 2 or y.shape[0] != ens.n_particles:
        raise ValueError(f"pushforward: simulator returned shape {y.shape}, "
                         f"expected ({ens.n_particles}, d_out)")
    return ens.assign({output: y})


def _push_gaussian(g, f, inputs, output, key):
    if len(inputs) != 1:
        raise NotImplementedError("prototype: one input block for Gaussian pushforward")
    (src,) = inputs
    if isinstance(f, Linear):
        g = g.absorb(src)
        i = g.names.index(src)
        m = f(g.means[i])
        F = g.factors[i]
        newF = None if F is None else Dense(f.op.matmat(F.to_dense()))
        if output == src:
            return g.with_block(output, m, newF)
        return g.with_block(output, m, newF)
    if isinstance(f, AdditiveNoise):
        if output == src:
            return g.add_noise({src: f.cov})
        g = g.absorb(src)
        i = g.names.index(src)
        return g.with_block(output, g.means[i], g.factors[i]).add_noise({output: f.cov})
    raise TypeError(
        f"pushforward: a Gaussian can only be pushed through Linear or AdditiveNoise, "
        f"not {type(f).__name__}; sample an Ensemble, push it, and project instead")
