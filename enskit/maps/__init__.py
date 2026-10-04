r"""Maps between named blocks, and pushing distributions through them.

A *map* takes the values of some blocks to the value of a new block, or a
replacement for one of them. Any callable is a map; the classes here add
structure, which :func:`pushforward` uses to act exactly on a
:class:`~enskit.distribution.Gaussian`, or traceability.

======================== ======================================================
object                   is
======================== ======================================================
:func:`pushforward`      the distribution of ``f(inputs)``, added as a block
:class:`StructuredMap`   protocol: a map that also pushes a Gaussian exactly
:class:`Linear`          :math:`x \mapsto Ax + c` for operators :math:`A`:
                         exact on Gaussians
:class:`AdditiveNoise`   :math:`x \mapsto x + e`, :math:`e \sim \mathcal N(0, R)`:
                         exact on Gaussians
:class:`BlackBox`        a host-side simulator made traceable, with a zero
                         derivative
======================== ======================================================

**The simulator contract.** A *simulator* is any callable used as a map on an
:class:`~enskit.distribution.Ensemble`. It is called **once per
pushforward with every particle**, and receives one positional
``(n_particles, d_in)`` array per input block, in the order of ``inputs``.
It returns a ``(n_particles, d_out)`` array-like (a ``jax.Array``, a NumPy
array or a nested list), or, for several outputs, a tuple of them or a
mapping or ``NamedTuple`` keyed by the output names.

- **Row** :math:`j` **of every output depends only on row** :math:`j` **of
  the inputs.** Nothing inside :func:`pushforward` can detect a violation;
  :func:`enskit.testing.check_simulator` checks it from outside.
- **A failed particle is a non-finite row.** The callable catches its own
  crashes and timeouts and returns ``nan`` there; an exception that escapes
  it propagates, and every particle's result is lost with it.
- **The dtype is the ensemble's.** A narrower floating return is promoted,
  with one warning per call; an integer, complex or wider one raises,
  naming the simulator.
- **Determinism is not required.** A simulator that draws from a JAX key
  declares ``needs_key = True`` and is called as ``f(key, *inputs)``.

Distributing calls across processes or machines is the simulator's
business, done inside the callable.

Notes
-----
The behavior of this module is specified by the "Maps contract" page of the
documentation, which is normative. :class:`Linear`, :class:`AdditiveNoise`
and :class:`BlackBox` are frozen pytrees, as the distributions are: a map
holding operators crosses a ``jit`` boundary as data.
"""
from ._blackbox import BlackBox
from ._pushforward import StructuredMap, pushforward
from ._structured import AdditiveNoise, Linear

__all__ = [
    "pushforward",
    "StructuredMap",
    "Linear",
    "AdditiveNoise",
    "BlackBox",
]

# The modules above are private, so the public names report the package they
# are imported from, in tracebacks, ``type()`` and pickles.
for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
