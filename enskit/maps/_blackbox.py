r""":class:`BlackBox`, a host-side simulator made traceable."""
from __future__ import annotations

import functools
import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..linalg import static_field
from . import _common as c

__all__ = ["BlackBox"]


@c.map_class
class BlackBox:
    r"""A host-side simulator made safe to call inside ``jit``, ``vmap`` and ``grad``.

    Wraps ``f`` in :func:`jax.pure_callback`, so a NumPy code, a subprocess
    or a scheduler submission can appear inside a traced function, and stops
    the gradient at its inputs, so that differentiation treats its outputs as
    constants:

    .. math::

        \frac{\partial\, \mathrm{BlackBox}(f)(x)}{\partial x} = 0 .

    Outside any trace, a plain callable needs no wrapper.

    Calling ``bb(*inputs)`` takes one or more ``(n, d_in)`` arrays and returns
    ``(n, d_out)`` of the result dtype, or a tuple of them when
    ``output_dim`` is a tuple. ``f`` receives NumPy arrays and must satisfy
    the simulator contract of :func:`pushforward`; its return is checked
    against that contract inside the callback: the shape, the number of
    outputs, and the dtype rule against the result dtype (a narrower
    floating return is promoted with a warning; a wider, integer or complex
    one raises). Under :func:`jax.vmap` the callback uses
    ``vmap_method="sequential"``: ``f`` is called once per member of the
    mapped family, each time with ordinary ``(n, d_in)`` arrays.

    Parameters
    ----------
    f : callable
        The simulator. Receives NumPy arrays, and returns an array-like, or a
        tuple of them when ``output_dim`` is a tuple.
    output_dim : int or tuple of int
        ``d_out``, each at least 1, needed to declare the callback's result
        shapes before ``f`` runs. A tuple declares several outputs, returned
        by ``f`` as a tuple in that order.
    dtype : dtype, optional
        Keyword-only. The result dtype, a real floating dtype; by default the
        dtype of the first input at each call.
    needs_key : bool
        Keyword-only. When ``True``, the box is called as ``bb(key, *inputs)``
        with a typed key, and calls ``f(key_data, *inputs)`` with
        ``key_data`` the ``uint32`` array :func:`jax.random.key_data`
        returns; :func:`pushforward` then requires a key.

    Raises
    ------
    TypeError
        If ``f`` is not callable, ``output_dim`` is not an ``int`` or a tuple
        of them, ``dtype`` is not a real floating dtype, or ``needs_key`` is
        not a ``bool``.
    ValueError
        If an output dimension is below 1.

    Notes
    -----
    A zero derivative is a statement about the gradient being computed, not
    about the simulator. A gradient through a computation that calls a box
    at points depending on the differentiated parameter is a partial
    derivative with those outputs held fixed: the derivative wanted when the
    parameter acts elsewhere and the outputs are data, not the derivative of
    the simulator.

    An exception inside the callback, from the checks above or from ``f``
    itself, is raised by JAX as :class:`jax.errors.JaxRuntimeError` carrying
    the original message, eagerly as well as under ``jit``. ``f`` should
    catch its own failures and return non-finite rows instead.
    """

    f: Callable = static_field()
    output_dim: int | tuple[int, ...] = static_field()
    dtype: object = static_field()
    needs_key: bool = static_field()

    def __init__(self, f, output_dim, *, dtype=None, needs_key=False) -> None:
        where = "BlackBox"
        if not callable(f):
            raise TypeError(f"{where}: f must be callable, got {type(f).__name__}")
        dims = output_dim if isinstance(output_dim, tuple) else (output_dim,)
        if not dims or any(type(d) is not int for d in dims):
            raise TypeError(
                f"{where}: output_dim must be a Python int or a non-empty tuple of "
                f"them, got {output_dim!r}"
            )
        if any(d < 1 for d in dims):
            raise ValueError(f"{where}: every output dimension must be at least 1, "
                             f"got {output_dim!r}")
        if dtype is not None:
            try:
                dtype = jnp.dtype(dtype)
            except TypeError as exc:
                raise TypeError(f"{where}: dtype {dtype!r} is not a dtype") from exc
            if not jnp.issubdtype(dtype, jnp.floating):
                raise TypeError(
                    f"{where}: dtype must be a real floating dtype, got {dtype}"
                )
        if type(needs_key) is not bool:
            raise TypeError(
                f"{where}: needs_key must be a bool, got {type(needs_key).__name__}"
            )
        object.__setattr__(self, "f", f)
        object.__setattr__(self, "output_dim", output_dim)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "needs_key", needs_key)

    def __call__(self, *args):
        """Call ``f`` through the callback; see the class docstring."""
        where = repr(self)
        operands = list(args)
        if self.needs_key:
            if not operands:
                raise TypeError(f"{where}: needs_key is set, so the key comes first")
            key = operands.pop(0)
            c.check_key(where, key, required=True)
        if not operands:
            raise TypeError(f"{where}: takes at least one input array")
        xs = [jnp.asarray(x) for x in operands]
        n = xs[0].shape[0] if xs[0].ndim == 2 else None
        for x in xs:
            if x.ndim != 2 or x.shape[0] != n:
                raise ValueError(
                    f"{where}: the inputs must be 2-D with one leading dimension, got "
                    f"shapes {[tuple(y.shape) for y in xs]}"
                )
        dtype = self.dtype if self.dtype is not None else xs[0].dtype
        several = isinstance(self.output_dim, tuple)
        dims = self.output_dim if several else (self.output_dim,)
        shapes = tuple(jax.ShapeDtypeStruct((n, d), dtype) for d in dims)
        host = functools.partial(
            _host_call, self.f, c.describe(self.f), dims, n, dtype, several
        )
        operands = [jax.lax.stop_gradient(x) for x in xs]
        if self.needs_key:
            operands.insert(0, jax.random.key_data(key))
        out = jax.pure_callback(
            host, shapes if several else shapes[0], *operands, vmap_method="sequential"
        )
        return tuple(out) if several else out

    def __repr__(self) -> str:
        try:
            return (
                f"BlackBox(f={c.describe(self.f)}, output_dim={self.output_dim}, "
                f"needs_key={self.needs_key})"
            )
        except Exception:
            return "BlackBox(<unreadable>)"


# -- private -----------------------------------------------------------------------


def _host_call(f, who, dims, n, dtype, several, *arrays):
    """Run ``f`` on the host and hold its return to the simulator contract."""
    where = f"BlackBox({who})"
    returned = f(*arrays)
    if several:
        if not isinstance(returned, (tuple, list)) or len(returned) != len(dims):
            raise ValueError(
                f"{where}: declared {len(dims)} outputs, so f must return a tuple of "
                f"{len(dims)} arrays; got {type(returned).__name__}"
            )
        values = list(returned)
    else:
        values = [returned]
    out, promoted = [], []
    for i, (value, d) in enumerate(zip(values, dims, strict=True)):
        arr = np.asarray(value)
        if arr.shape != (n, d):
            raise ValueError(
                f"{where}: f returned shape {arr.shape} for output {i}, expected "
                f"({n}, {d})"
            )
        verdict = c.dtype_rule(arr.dtype, dtype)
        if verdict == c.REFUSE:
            raise ValueError(c.refusal(where, "f", f"output {i}", arr.dtype, dtype))
        if verdict == c.PROMOTE:
            promoted.append(f"output {i} from {arr.dtype}")
        out.append(arr.astype(dtype, copy=False))
    if promoted:
        warnings.warn(
            f"{where}: f returned a narrower dtype than {dtype}, so its output was "
            f"promoted ({', '.join(promoted)}). Promotion does not recover the "
            f"digits f did not compute.",
            UserWarning,
            stacklevel=2,
        )
    return tuple(out) if several else out[0]
