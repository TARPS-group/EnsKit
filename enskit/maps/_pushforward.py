r""":func:`pushforward` and the :class:`StructuredMap` protocol."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from ..distribution import Ensemble, Gaussian
from . import _common as c

__all__ = ["pushforward", "StructuredMap"]


def pushforward(dist, f, *, output, inputs=None, key=None):
    r"""The distribution of ``f(inputs)``, as block ``output`` of the result.

    .. math::

        (x_1, \dots, x_n) \sim p \quad\Longrightarrow\quad
        \big(x_1, \dots, x_n,\ f(x_{b_1}, \dots, x_{b_m})\big),

    for input blocks :math:`b_1, \dots, b_m`. The joint relation between the
    inputs and the output is kept: particle by particle on an
    :class:`~enskit.distribution.Ensemble`, and in the shared factor on a
    :class:`~enskit.distribution.Gaussian`.

    ============== =================== =========================================
    ``dist``       ``f``               what happens
    ============== =================== =========================================
    ``Ensemble``   a simulator         ``f`` is called once with every particle
    ``Ensemble``   a StructuredMap     ``f.push_ensemble(dist, inputs, output, key)``
    ``Gaussian``   a StructuredMap     ``f.push_gaussian(dist, inputs, output)``
    ``Gaussian``   anything else       ``TypeError``
    ============== =================== =========================================

    A simulator is any callable. It receives one positional ``(J, d_b)``
    array per input block, in the order of ``inputs``, and returns a
    ``(J, d_out)`` array-like whose row :math:`j` depends only on row
    :math:`j` of the inputs. A failed particle is a non-finite row, which is
    written to the result as returned; the callable catches its own
    exceptions. A simulator whose attribute ``needs_key`` is ``True`` is
    called as ``f(key, *inputs)``.

    Parameters
    ----------
    dist : enskit.distribution.Ensemble or enskit.distribution.Gaussian
        The distribution to push forward.
    f : callable or StructuredMap
        The map. A :class:`~enskit.distribution.Gaussian` takes only a
        :class:`StructuredMap` (:class:`Linear`, :class:`AdditiveNoise`),
        because only those keep a Gaussian Gaussian.
    output : str or sequence of str
        Keyword-only. The name of the result block. If it is one of
        ``inputs`` the block is replaced in its position; otherwise it must
        be new, and is appended. With a sequence of names, a simulator is
        called once and returns one array per name: a tuple in the same
        order, or a mapping or ``NamedTuple`` keyed by the names.
    inputs : str or sequence of str, optional
        Keyword-only. The blocks passed to ``f``, in this order. Defaults to
        every block of ``dist``, in block order.
    key : jax.random key, optional
        Keyword-only. Required when the map draws randomness on an ensemble:
        an :class:`AdditiveNoise`, or a simulator that declares
        ``needs_key``. Otherwise ignored, though type-checked.

    Returns
    -------
    enskit.distribution.Ensemble or enskit.distribution.Gaussian
        The type of ``dist``, with every block that is not an output kept.
        An ensemble keeps its weights. An
        :class:`~enskit.distribution.EnsembleGaussian` stays one unless an
        independent term had to be absorbed.

    Raises
    ------
    TypeError
        If ``dist`` is not a distribution, a name is not a ``str``, the key
        is not a typed key, ``f`` is neither callable nor a
        :class:`StructuredMap`, or ``f`` is not a :class:`StructuredMap` and
        ``dist`` is a Gaussian.
    KeyError
        If an input is not a block.
    ValueError
        If there is no output or input, a name is repeated, an output names
        a block that is not an input, a needed key is missing, a map does
        not fit its inputs, or a simulator's return is not the container
        expected or not ``(J, d_out)``. Also if an output's dtype is
        integer, boolean, complex, or wider than the ensemble's; a narrower
        floating dtype is promoted, with one ``UserWarning`` per call.
    UnsupportedOpError
        If a structured map needs an operator capability, such as
        ``factor``, that an operator lacks.

    Notes
    -----
    Nothing inside this function can detect a simulator whose rows are
    coupled; :func:`enskit.testing.check_simulator` checks it from outside.
    Under :func:`jax.jit` a host-side simulator receives tracers and fails;
    wrap it in :class:`BlackBox`. The promotion warning is issued when a
    function is traced, not at every call of the compiled function.

    The order of operations chooses the estimator: pushing an ensemble
    through :class:`AdditiveNoise` and projecting estimates the output's
    covariance from sampled noise, while projecting first and pushing the
    Gaussian adds the noise covariance exactly.
    """
    where = "pushforward"
    if not isinstance(dist, (Ensemble, Gaussian)):
        raise TypeError(
            f"{where}: dist must be an Ensemble or a Gaussian, got {type(dist).__name__}"
        )
    if dist.batch_shape != ():
        raise ValueError(
            f"{where}: {dist!r} is a vmapped family with batch shape "
            f"{dist.batch_shape}, which cannot be pushed forward directly; apply "
            f"pushforward under jax.vmap, one member at a time."
        )
    outputs, several = _output_names(where, output)
    inputs = _input_names(where, dist, inputs)
    for name in outputs:
        if name in dist.names and name not in inputs:
            raise ValueError(
                f"{where}: output {name!r} is already a block and is not an input, "
                f"so it would be overwritten; drop it first, or name it in inputs to "
                f"replace it"
            )
    c.check_key(where, key, required=False)
    structured = isinstance(f, StructuredMap)
    if isinstance(dist, Gaussian) and not structured:
        raise TypeError(
            f"{where}: a Gaussian can only be pushed through a StructuredMap (Linear, "
            f"AdditiveNoise), since any other map makes it non-Gaussian; got "
            f"{c.describe(f)}. Sample an Ensemble, push it, and project instead."
        )
    if not structured and not callable(f):
        raise TypeError(
            f"{where}: the map must be callable or a StructuredMap, got "
            f"{type(f).__name__}"
        )
    if structured and several:
        raise ValueError(
            f"{where}: {c.describe(f)} is a StructuredMap, which produces one block; "
            f"got outputs {outputs}"
        )
    if not structured:
        return _push_simulator(where, dist, f, inputs, outputs, several, key)
    if isinstance(dist, Ensemble):
        result = f.push_ensemble(dist, inputs, outputs[0], key)
    else:
        result = f.push_gaussian(dist, inputs, outputs[0])
    _check_result(where, f, dist, result, outputs[0])
    return result


@runtime_checkable
class StructuredMap(Protocol):
    """A map that can also act exactly on a :class:`~enskit.distribution.Gaussian`.

    The extension point for maps with structure: implement both methods and
    :func:`pushforward` dispatches to them. :class:`Linear` and
    :class:`AdditiveNoise` are the shipped instances; linearizations and
    sigma-point rules are future ones.

    :func:`pushforward` calls the methods with ``inputs`` a non-empty tuple
    of distinct block names, ``output`` a single name that is either one of
    ``inputs`` or not a block, and ``key`` a typed key or ``None``, all
    checked. An implementation must:

    1. return the distribution's type, with every block other than
       ``output`` kept in its position, and ``output`` replaced in place or
       appended last;
    2. raise ``ValueError`` for inputs it cannot take and for a missing key it
       needs, and the operator layer's ``UnsupportedOpError``, unmodified,
       for a capability an operator lacks, all before any work;
    3. be exact on a Gaussian, or document what it approximates;
    4. agree in distribution between the two methods: pushing samples of a
       Gaussian through ``push_ensemble`` draws from the distribution
       ``push_gaussian`` returns;
    5. be a pytree when it holds arrays or operators.

    ``isinstance(f, StructuredMap)`` checks only that both methods exist.
    """

    def push_ensemble(
        self, ensemble: Ensemble, inputs: tuple[str, ...], output: str, key
    ) -> Ensemble:
        """Push the particles of ``ensemble`` through the map."""
        ...

    def push_gaussian(
        self, gaussian: Gaussian, inputs: tuple[str, ...], output: str
    ) -> Gaussian:
        """Push ``gaussian`` through the map, exactly."""
        ...


# -- private -----------------------------------------------------------------------


def _output_names(where: str, output) -> tuple[tuple[str, ...], bool]:
    """The output names, and whether the simulator returns a container of them."""
    if isinstance(output, str):
        return (output,), False
    if not isinstance(output, Sequence):
        raise TypeError(
            f"{where}: output must be a str or a sequence of str, got "
            f"{type(output).__name__}"
        )
    names = tuple(output)
    if not names:
        raise ValueError(f"{where}: at least one output name is required")
    _check_names(where, "output", names)
    return names, True


def _input_names(where: str, dist, inputs) -> tuple[str, ...]:
    if inputs is None:
        return dist.names
    if isinstance(inputs, str):
        names = (inputs,)
    elif isinstance(inputs, Sequence):
        names = tuple(inputs)
    else:
        raise TypeError(
            f"{where}: inputs must be a str or a sequence of str, got "
            f"{type(inputs).__name__}"
        )
    if not names:
        raise ValueError(f"{where}: at least one input block is required")
    _check_names(where, "inputs", names)
    unknown = [n for n in names if n not in dist.names]
    if unknown:
        raise KeyError(
            f"{where}: input {unknown[0]!r} is not a block; the blocks are {dist.names}"
        )
    return names


def _check_names(where: str, what: str, names: tuple) -> None:
    for name in names:
        if not isinstance(name, str):
            raise TypeError(
                f"{where}: every name in {what} must be a str, got "
                f"{type(name).__name__}"
            )
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"{where}: {what} names {repeated} more than once")


def _push_simulator(where, ens, f, inputs, outputs, several, key) -> Ensemble:
    who = c.describe(f)
    xs = [ens[name] for name in inputs]
    if c.needs_key(f):
        c.check_key(where, key, required=True, why=f": {who} declares needs_key")
        returned = f(key, *xs)
    else:
        returned = f(*xs)
    values = _unpack(where, who, returned, outputs, several)
    dtype = xs[0].dtype
    arrays, promoted = {}, {}
    for name in outputs:
        arr, was = c.output_array(where, who, name, values[name], ens.n_particles, dtype)
        arrays[name] = arr
        if was is not None:
            promoted[name] = was
    if promoted:
        c.warn_promoted(where, who, promoted, dtype)
    return ens.assign(arrays)


def _unpack(where: str, who: str, returned, outputs, several) -> dict:
    """The returned value of each output, by name."""
    if not several:
        return {outputs[0]: returned}
    if isinstance(returned, Mapping):
        keys = tuple(returned)
    elif isinstance(returned, tuple) and hasattr(returned, "_fields"):
        keys = tuple(returned._fields)
    elif isinstance(returned, (tuple, list)):
        if len(returned) != len(outputs):
            raise ValueError(
                f"{where}: {who} returned {len(returned)} arrays for the "
                f"{len(outputs)} outputs {outputs}"
            )
        return dict(zip(outputs, returned, strict=True))
    else:
        raise ValueError(
            f"{where}: {who} returned {type(returned).__name__} for the outputs "
            f"{outputs}; several outputs are returned as a tuple in that order, or "
            f"as a mapping or NamedTuple keyed by the names"
        )
    if set(keys) != set(outputs) or len(keys) != len(outputs):
        raise ValueError(
            f"{where}: {who} returned outputs keyed {keys}, expected exactly {outputs}"
        )
    if isinstance(returned, Mapping):
        return {name: returned[name] for name in outputs}
    return {name: getattr(returned, name) for name in outputs}


def _check_result(where: str, f, dist, result, output: str) -> None:
    """A structured map's result has the type and blocks the rules give."""
    who = c.describe(f)
    kind = Ensemble if isinstance(dist, Ensemble) else Gaussian
    if not isinstance(result, kind):
        raise TypeError(
            f"{where}: {who} returned {type(result).__name__} for a "
            f"{type(dist).__name__}; a StructuredMap returns the type it is given"
        )
    expected = dist.names if output in dist.names else (*dist.names, output)
    if result.names != expected:
        raise ValueError(
            f"{where}: {who} returned blocks {result.names}, expected {expected}: "
            f"every other block kept in its position, and {output!r} replaced in "
            f"place or appended"
        )
    if kind is Ensemble and result.n_particles != dist.n_particles:
        raise ValueError(
            f"{where}: {who} returned {result.n_particles} particles for "
            f"{dist.n_particles}"
        )
