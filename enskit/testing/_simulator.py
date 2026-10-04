""":func:`check_simulator`, the simulator contract checked from outside."""
from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..distribution import Ensemble
from ..maps import pushforward

__all__ = ["check_simulator"]

#: The relative tolerance of the subset comparison, per element.
_RTOL = 1e-8


def check_simulator(
    f,
    input_dims,
    output_dims,
    *,
    n_particles: int = 6,
    seed: int = 0,
    stochastic: bool = False,
) -> None:
    """Check a simulator of your own against the simulator contract.

    A simulator is any callable that :func:`enskit.maps.pushforward` calls
    with one ``(J, d_in)`` array per input block and that returns
    ``(J, d_out)``; there is no base class and nothing to register. This
    calls ``f`` through :func:`~enskit.maps.pushforward`, on ensembles of
    pseudo-random inputs, so it checks the callable as the layer will call
    it, including **row independence**, which nothing inside a pushforward
    detects.

    ============================================== ==========================
    checked                                        not checked
    ============================================== ==========================
    the container, shapes and dtypes, at ``J`` and failure signaling on the
    ``J + 1`` particles; a narrower dtype fails    simulator's own error
    here although a pushforward promotes it        paths
    determinism, by calling twice                  whether a non-finite row
    row independence, by permuting the particles   *should* have been
    and by calling on a subset of them             produced; cost, side
                                                   effects
    ============================================== ==========================

    **This calls the simulator five times**: twice when
    ``stochastic=True``, three times when it declares ``needs_key``. For a
    real simulator that is five runs of the expensive thing, so check a
    cheap configuration of it: a coarse grid, a short horizon, a stub
    solver.

    Parameters
    ----------
    f : callable
        The simulator.
    input_dims : int or sequence of int
        The dimension of each positional input, at least 1.
    output_dims : int or Mapping[str, int]
        For a simulator returning one array, its dimension. For one
        returning several, a mapping from output name to dimension; the
        simulator returns a tuple in the mapping's order, or a mapping or
        ``NamedTuple`` keyed by the names.
    n_particles : int
        Keyword-only. The ensemble size :math:`J`, at least 3; the simulator
        is also called at ``J + 1``, since one whose rows are independent
        cannot depend on how many there are.
    seed : int
        Keyword-only. Seeds the NumPy generator that fills the inputs, and
        the key passed to a simulator that declares ``needs_key``.
    stochastic : bool
        Keyword-only. Declare that the simulator is not deterministic,
        skipping the determinism and row-independence checks.

    Raises
    ------
    AssertionError
        On any violation, naming which.
    TypeError
        If a dimension is not an ``int``.
    ValueError
        If ``n_particles`` is not an ``int`` of at least 3, or a dimension is
        below 1.

    Notes
    -----
    Non-finite outputs are *not* a violation: a non-finite row is how a
    simulator signals a failed particle, so the comparisons treat two
    ``nan`` s as equal and compare where the non-finite entries are exactly.
    A simulator that fails for every particle still passes, and is still
    useless; read the outputs, not only this check's silence.

    Row independence is checked twice, because the two ways of breaking it
    are caught by different comparisons. Permuting the particles must
    permute the outputs **bit-exactly**, since the rows are the same set
    either way; this catches a coupling that depends on order, such as a
    running total down the rows. A *symmetric* coupling, such as normalizing
    across the batch, survives a permutation, so the second comparison calls
    the simulator on two of the particles without the others and compares
    to a tolerance of ``1e-8 * max(1, |value|)`` **per element**: a differently
    shaped batch may legitimately round differently, while a coupling
    changes the answer by :math:`O(1)`. A single scale set by the largest
    output would let a coupling in a small output hide beside a large one.
    Both comparisons are necessary conditions, not sufficient ones.

    A simulator that declares ``needs_key`` is called with a fixed key. Its
    determinism given that key is checked, and the row-independence checks
    are skipped, since its draws are laid out by row: a permutation of the
    particles legitimately changes which particle gets which draw.
    """
    where = "check_simulator"
    if type(n_particles) is not int or n_particles < 3:
        raise ValueError(
            f"{where}: n_particles must be an int of at least 3, got "
            f"{n_particles!r}. Both row-independence comparisons need a subset that "
            f"is smaller than the ensemble and a permutation that is not the "
            f"identity, and below 3 particles neither exists: the checks would "
            f"pass a coupled simulator."
        )
    in_dims = _dims(where, "input_dims", input_dims)
    if isinstance(output_dims, Mapping):
        out_names, output = tuple(output_dims), tuple(output_dims)
        sizes = _dims(where, "output_dims", tuple(output_dims.values()))
    else:
        out_names, output = ("output",), "output"
        sizes = _dims(where, "output_dims", output_dims)
    out_dims = dict(zip(out_names, sizes, strict=True))
    in_names = tuple(f"input {i}" for i in range(len(in_dims)))
    if set(in_names) & set(out_names):
        in_names = tuple(f"_{n}" for n in in_names)
    who = getattr(f, "__name__", None) or repr(f)
    keyed = getattr(f, "needs_key", False) is True
    key = jax.random.key(seed) if keyed else None
    rng = np.random.default_rng(seed)

    def draw(n):
        return {
            name: jnp.asarray(rng.normal(size=(n, d)))
            for name, d in zip(in_names, in_dims, strict=True)
        }

    def call(blocks):
        with warnings.catch_warnings():
            # Both pushforward's promotion and a BlackBox's, inside its callback.
            warnings.filterwarnings(
                "error", message=r".*promoted", category=UserWarning
            )
            try:
                result = pushforward(
                    Ensemble(blocks), f, output=output, inputs=in_names, key=key
                )
            except UserWarning as w:
                raise AssertionError(
                    f"{who}: {w}. A narrower return is promoted by a pushforward, "
                    f"with a warning, and has already lost the digits by then"
                ) from None
            except ValueError as exc:
                if not str(exc).startswith("pushforward:"):
                    raise
                raise AssertionError(f"{who}: {exc}") from None
            except jax.errors.JaxRuntimeError as exc:
                # A BlackBox's checks, or its promotion warning, raised inside
                # the callback.
                raise AssertionError(f"{who}: {exc}") from None
        n = result.n_particles
        for name in out_names:
            assert result.dims[name] == out_dims[name], (
                f"{who}: called with {n} particles and returned shape "
                f"({n}, {result.dims[name]}) for output {name!r}, expected "
                f"({n}, {out_dims[name]})"
            )
        return {name: np.asarray(result[name]) for name in out_names}

    blocks = draw(n_particles)
    outputs = call(blocks)
    call(draw(n_particles + 1))
    if stochastic:
        return

    # Determinism first: a stochastic simulator breaks the permutation check
    # too, and "not deterministic" is the diagnosis it should get.
    again = call(blocks)
    for name in out_names:
        _identical_allowing_nan(
            again[name],
            outputs[name],
            f"{who}: output {name!r} is not deterministic across two calls on the "
            f"same inputs{' and key' if keyed else ''}. A stochastic simulator is "
            f"legal; declare it with stochastic=True, or give it a key with "
            f"needs_key",
        )
    if keyed:
        return
    # Never the identity: comparing a result with itself asserts nothing, and
    # at small n_particles a fair draw returns it often.
    perm = rng.permutation(n_particles)
    while np.array_equal(perm, np.arange(n_particles)):
        perm = rng.permutation(n_particles)
    permuted = call({n: a[perm] for n, a in blocks.items()})
    for name in out_names:
        _identical_allowing_nan(
            permuted[name],
            outputs[name][perm],
            f"{who}: row j of output {name!r} does not depend on row j of the "
            f"inputs alone: permuting the particles changed more than the order of "
            f"the outputs, so the simulator is order-dependent across rows. A "
            f"shared accumulator written in row order does this, and no "
            f"pushforward can detect it",
        )
    subset = np.asarray([1, n_particles - 1])
    alone = call({n: a[subset] for n, a in blocks.items()})
    for name in out_names:
        _close_allowing_nan(
            alone[name],
            outputs[name][subset],
            f"{who}: a particle's output {name!r} changed when it was computed "
            f"alongside different particles, so row j does not depend on row j of "
            f"the inputs alone. A simulator that normalizes across the batch does "
            f"this, and no pushforward can detect it",
        )


# -- private -----------------------------------------------------------------------


def _dims(where: str, what: str, dims) -> tuple[int, ...]:
    """A positive int, or a non-empty sequence of them, as a tuple."""
    if isinstance(dims, Sequence) and not isinstance(dims, str):
        dims = tuple(dims)
    else:
        dims = (dims,)
    if not dims or any(type(d) is not int for d in dims):
        raise TypeError(f"{where}: {what} must be ints, got {dims!r}")
    if any(d < 1 for d in dims):
        raise ValueError(f"{where}: {what} must be at least 1, got {dims!r}")
    return dims


def _identical_allowing_nan(got, want, what: str) -> None:
    """Bit-identity, counting two ``nan`` s as equal: failed rows are legal."""
    assert got.shape == want.shape, f"{what} (shape {got.shape} != {want.shape})"
    assert np.array_equal(got, want, equal_nan=True), what


def _close_allowing_nan(got, want, what: str) -> None:
    """Agreement to a per-element tolerance, the non-finite pattern exactly."""
    assert got.shape == want.shape, f"{what} (shape {got.shape} != {want.shape})"
    finite = np.isfinite(want)
    assert np.array_equal(finite, np.isfinite(got)), (
        f"{what} (the non-finite entries are in different places)"
    )
    if not finite.any():
        return
    tolerance = _RTOL * np.maximum(1.0, np.abs(want[finite]))
    excess = np.abs(got[finite] - want[finite]) - tolerance
    assert excess.max() <= 0.0, (
        f"{what} (worst element exceeds its tolerance by {float(excess.max()):.3e})"
    )
