"""Maps between named blocks, and pushing distributions through them.

A *map* takes the values of some blocks to the value of a new (or replaced)
block. Any callable is a map; a few wrapper classes add nothing but
structure, which :func:`pushforward` uses to act exactly on a
:class:`~enskit.distribution.Gaussian` and to estimate more accurately from an
:class:`~enskit.distribution.Ensemble`.

========================= =====================================================
object                    is
========================= =====================================================
:func:`pushforward`       the distribution of ``f(inputs)``, added as a block
:class:`Linear`           ``x -> A x + c`` for operators ``A``: exact on Gaussians
:class:`AdditiveNoise`    ``x -> x + e``, ``e ~ N(0, R)``: exact on Gaussians
:class:`BlackBox`         a host-side simulator made traceable, with outputs
                          treated as constants by differentiation
:class:`StructuredMap`    protocol for maps that know how to push a Gaussian
========================= =====================================================

**The simulator contract.** A plain callable used as a map receives one
positional ``(n_particles, d_in)`` array per input block, in the order of
``inputs``, and returns a ``(n_particles, d_out)`` array-like. It is called once
per pushforward with every particle. Row ``j`` of the output must depend only on
row ``j`` of the inputs; nothing can detect a violation, and
:func:`enskit.testing.check_simulator` checks it from outside. A particle whose
evaluation failed is signalled by a **non-finite row**: the callable catches
its own crashes and timeouts and returns ``nan`` there. Determinism is not
required. Distributing evaluations across processes or machines is the
caller's business: do it inside the callable.
"""


def pushforward(dist, f, *, output: str, inputs: str | Sequence[str] | None = None,
                key=None):
    """The distribution of ``f(inputs)``, as block ``output`` of the result.

    Parameters
    ----------
    dist : Ensemble or Gaussian
        The distribution to push forward.
    f : callable or StructuredMap
        The map. On an :class:`~enskit.distribution.Ensemble` any callable works; on a
        :class:`~enskit.distribution.Gaussian` only a :class:`StructuredMap`
        (:class:`Linear`, :class:`AdditiveNoise`) does, because only those keep
        a Gaussian Gaussian.
    output : str or sequence of str
        Keyword-only. The name of the result block. If it is one of
        ``inputs`` the block is replaced (an in-place map ``x -> M(x)``);
        otherwise it must be new and is appended. With several names the
        simulator is called once and returns one array per name: a tuple in
        the same order, or a mapping or ``NamedTuple`` keyed by the names
        (checked against ``output``). The result carries every output as a
        named block.
    inputs : str or sequence of str, optional
        Keyword-only. The blocks passed to ``f``, positionally, in this order.
        Defaults to every block of ``dist``.
    key : jax.random key, optional
        Keyword-only. Required when ``f`` draws randomness on an ensemble
        (:class:`AdditiveNoise`, or a stochastic callable declared with
        ``needs_key=True`` on :class:`BlackBox`).

    Returns
    -------
    Ensemble or Gaussian
        The same type as ``dist``, with every input block kept (unless
        replaced) and ``output`` added. On an ensemble the joint relation
        between inputs and output is kept particle by particle; on a Gaussian it is
        kept in the shared factor.

    Raises
    ------
    TypeError
        If ``f`` cannot act on a Gaussian.
    ValueError
        If ``output`` exists and is not an input; if the simulator's output is
        not ``(n_particles, d_out)``; if its dtype is integer or complex, or
        wider than the ensemble's (narrower floating dtypes are promoted with a
        warning); if a needed key is missing.

    Notes
    -----
    Exactness on Gaussians: a :class:`Linear` map of block ``x`` gives the new
    block the factor row ``A F_x`` and mean ``A m_x + c``; if ``x`` has an
    independent term it is first absorbed into the factor, so that the new
    block stays correlated with ``x``. :class:`AdditiveNoise` adds an
    independent term. Order matters on ensembles: pushing an ensemble through
    ``AdditiveNoise`` samples the noise, while pushing its projection adds the
    noise covariance exactly, a lower-variance estimate of the same joint.
    """


class StructuredMap(Protocol):
    """A map that can also act exactly on a :class:`~enskit.distribution.Gaussian`.

    The extension point for maps with structure: implement both methods and
    :func:`pushforward` dispatches to them. Linearizations and sigma-point
    rules are future implementations.
    """

    def push_ensemble(self, ensemble: Ensemble, inputs: tuple[str, ...], output: str,
                      key) -> Ensemble: ...

    def push_gaussian(self, gaussian: Gaussian, inputs: tuple[str, ...],
                      output: str) -> Gaussian: ...


class Linear:
    r"""An affine map with a structured operator.

    .. math::

        x \mapsto A x + c, \qquad\text{or}\qquad (x_1, \dots, x_m) \mapsto \sum_b A_b x_b + c .


    Calling it applies ``A`` to a batch of row vectors, so it is also an
    ordinary simulator. As a structured map it pushes Gaussians exactly: the
    output block's factor row is ``A F_x`` (a product operator) and its mean
    ``A m_x + shift``; an input block with an independent term has it absorbed
    first (:meth:`~enskit.distribution.Gaussian.absorb`), so the output stays correlated
    with the input.

    Parameters
    ----------
    op : LinOp or Mapping[str, LinOp]
        ``A``, of shape ``(d_out, d_in)``. With several inputs, a mapping from
        input block name to that block's operator: the map is
        ``sum_b A_b x_b + shift``. Called directly, it takes the inputs
        positionally in the mapping's order, and :func:`pushforward` requires
        ``inputs`` to list the same names in the same order.
    shift : Array, optional
        ``(d_out,)``.

    Methods
    -------
    __call__(*xs) -> Array
        ``(n, d_out)`` for ``(n, d_b)`` inputs.
    push_ensemble, push_gaussian
        The :class:`StructuredMap` methods.
    """


class AdditiveNoise:
    r"""Independent additive Gaussian noise.

    .. math::

        x \mapsto x + e, \qquad e \sim \mathcal N(0, R) \text{ independent of everything,}

    with :math:`R` = ``cov``.


    On a Gaussian it adds an independent term (exact, no sampling); when the
    output replaces the input this is :meth:`~enskit.distribution.Gaussian.add_noise`,
    otherwise the input is copied (absorbing any independent term it has) and
    the copy gets the noise. On an ensemble it draws ``e`` for each particle,
    needs a key, and ``cov`` must support ``factor``. It is not a callable
    simulator.

    Parameters
    ----------
    cov : PSDLinOp
        The noise covariance.
    """


class BlackBox:
    """A host-side simulator made safe to call inside ``jit``, ``vmap`` and ``grad``.

    Wraps ``f`` with :func:`jax.pure_callback`, so a NumPy code, a subprocess
    or a scheduler submission can appear inside a traced function, and gives
    it a zero derivative, so differentiation treats its outputs as constants.
    Outside any trace, a plain callable needs no wrapper.

    Under :func:`jax.vmap` the callback uses ``vmap_method="sequential"``: ``f``
    is called once per element of the mapped family, each time with ordinary
    ``(n_particles, d_in)`` arrays, so the simulator contract holds per call.

    Parameters
    ----------
    f : callable
        Satisfies the simulator contract; receives NumPy arrays.
    output_dim : int
        ``d_out``, needed to declare the callback's result shape.
    dtype : dtype, optional
        Keyword-only. The result dtype; defaults to the inputs'.
    needs_key : bool
        Keyword-only. When true, ``f`` is called as ``f(key_data, *inputs)``
        and a key must be passed to :func:`pushforward`.

    Notes
    -----
    A zero derivative is a statement about the gradient being computed, not
    about the simulator: a gradient through a run that evaluates a black box
    at points that depend on the differentiated parameter is a partial
    derivative with those evaluations held fixed. The differentiability
    contract states when that is the derivative a caller wants.
    """
