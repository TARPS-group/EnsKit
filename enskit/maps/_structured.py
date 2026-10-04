r"""The two shipped structured maps, :class:`Linear` and :class:`AdditiveNoise`."""
from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax import Array

from ..distribution import Ensemble, EnsembleGaussian, Gaussian
from ..linalg import Dense, LinOp, PSDLinOp, hstack, product, static_field
from . import _common as c

__all__ = ["Linear", "AdditiveNoise"]


@c.map_class
class Linear:
    r"""An affine map with a structured operator.

    .. math::

        x \mapsto A x + c, \qquad\text{or}\qquad
        (x_1, \dots, x_m) \mapsto \sum_{b=1}^{m} A_b x_b + c ,

    for operators :math:`A_b` of shape ``(d_out, d_b)`` and a shift
    :math:`c` of shape ``(d_out,)``.

    Calling it applies the map to a batch of row vectors, so it is also an
    ordinary simulator. As a :class:`StructuredMap` it pushes a Gaussian
    exactly: each input block's independent term, if it has one, is first
    absorbed into the shared factor
    (:meth:`~enskit.distribution.Gaussian.absorb`), so that the output stays
    correlated with it; then the output block gets

    .. math::

        m_y = \sum_b A_b m_b + c, \qquad F_y = \sum_b A_b F_b, \qquad D_y = 0,

    with :math:`F_y` held as a product operator, so a structured
    :math:`A_b` or :math:`F_b` is never densified. When nothing is absorbed,
    an :class:`~enskit.distribution.EnsembleGaussian` stays one.

    Parameters
    ----------
    op : LinOp, Array, or Mapping[str, LinOp or Array]
        :math:`A`, of shape ``(d_out, d_in)``; a 2-D array is wrapped in
        :class:`~enskit.linalg.Dense`. With several inputs, a mapping from
        input block name to that block's operator, all with one ``d_out``.
        Called directly, the map takes its inputs positionally in the
        mapping's order, and :func:`pushforward` requires ``inputs`` to name
        the same blocks in the same order.
    shift : Array, optional
        :math:`c`, a real floating ``(d_out,)`` array, kept as ``shift``
        (``None`` when absent). The mapping's keys are kept as
        ``input_names``, which is ``None`` for a single operator.

    Raises
    ------
    TypeError
        If an operator is not a :class:`~enskit.linalg.LinOp` or a floating
        2-D array, a mapping key is not a ``str``, or the shift is not a real
        floating array.
    ValueError
        If the mapping is empty, the operators disagree on ``d_out``, the
        shift is not ``(d_out,)``, or an operator is a vmapped family.

    Notes
    -----
    For an ensemble's projection, pushing the particles through :math:`A`
    and projecting gives the same moments as pushing the projection, since
    sample covariances are linear in the particles. The structured route
    pays off when the Gaussian is not a sample covariance: a prior, a static
    or hybrid covariance, or the result of earlier exact operations.
    """

    input_names: tuple[str, ...] | None = static_field()
    _ops: tuple[LinOp, ...]
    shift: Array | None

    def __init__(self, op, shift=None) -> None:
        where = "Linear"
        if isinstance(op, Mapping):
            if not op:
                raise ValueError(f"{where}: the mapping of operators is empty")
            for name in op:
                if not isinstance(name, str):
                    raise TypeError(
                        f"{where}: input names must be str, got {type(name).__name__}"
                    )
            names = tuple(op)
            ops = tuple(_as_operator(where, f"operator of {n!r}", op[n]) for n in names)
        else:
            names = None
            ops = (_as_operator(where, "op", op),)
        d_out = ops[0].shape[0]
        for A in ops[1:]:
            if A.shape[0] != d_out:
                raise ValueError(
                    f"{where}: the operators disagree on the output dimension: "
                    f"{[B.shape[0] for B in ops]}"
                )
        if shift is not None:
            shift = jnp.asarray(shift)
            if not jnp.issubdtype(shift.dtype, jnp.floating):
                raise TypeError(
                    f"{where}: shift must be a real floating array, got dtype "
                    f"{shift.dtype}"
                )
            if shift.shape != (d_out,):
                raise ValueError(
                    f"{where}: shift must have shape ({d_out},), got {shift.shape}"
                )
        object.__setattr__(self, "input_names", names)
        object.__setattr__(self, "_ops", ops)
        object.__setattr__(self, "shift", shift)

    @property
    def op(self) -> LinOp | dict[str, LinOp]:
        """The operator, or the operators keyed by input name, in order."""
        if self.input_names is None:
            return self._ops[0]
        return dict(zip(self.input_names, self._ops, strict=True))

    @property
    def output_dim(self) -> int:
        """``d_out``."""
        return self._ops[0].shape[0]

    def __call__(self, *xs) -> Array:
        """Apply the map to inputs ``(..., d_b)`` with one leading shape.

        Returns
        -------
        Array
            ``(..., d_out)``.

        Raises
        ------
        TypeError
            If the number of inputs is not the number of operators.
        ValueError
            If an input's trailing dimension is not its operator's.
        """
        if len(xs) != len(self._ops):
            raise TypeError(
                f"{self!r}: takes {len(self._ops)} input(s), got {len(xs)}"
            )
        y = self._ops[0].matvec(xs[0])
        for A, x in zip(self._ops[1:], xs[1:], strict=True):
            y = y + A.matvec(x)
        return y if self.shift is None else y + self.shift

    def push_ensemble(self, ensemble: Ensemble, inputs, output: str, key) -> Ensemble:
        """Apply the map to every particle; the :class:`StructuredMap` method."""
        where = "pushforward"
        self._fit(where, ensemble.dims, inputs)
        y = self(*[ensemble[n] for n in inputs])
        dtype = ensemble[inputs[0]].dtype
        remedy = "Give the operators and the shift the ensemble's dtype"
        arr, _ = c.output_array(
            where, "Linear", output, y, ensemble.n_particles, dtype, remedy
        )
        return ensemble.assign({output: arr})

    def push_gaussian(self, gaussian: Gaussian, inputs, output: str) -> Gaussian:
        """Push a Gaussian exactly; the :class:`StructuredMap` method."""
        where = "pushforward"
        self._fit(where, gaussian.dims, inputs)
        absorbed = tuple(n for n in inputs if gaussian.block_cov(n) is not None)
        base = gaussian.absorb(*absorbed) if absorbed else gaussian
        dtype = base.mean(base.names[0]).dtype
        mean = self(*[base.mean(n) for n in inputs])
        if mean.dtype != dtype:
            raise ValueError(
                f"{where}: Linear gives the output {output!r} dtype {mean.dtype}, "
                f"which differs from the Gaussian's {dtype}; give the operators "
                f"and the shift the Gaussian's dtype"
            )
        rows = [base.factor(n) for n in inputs]
        if len(rows) == 1:
            row = product(self._ops[0], rows[0])
        else:
            row = product(hstack(*self._ops), hstack(*(F.T for F in rows)).T)
        aligned = isinstance(gaussian, EnsembleGaussian) and not absorbed
        return c.with_block(base, output, mean, row, None, aligned=aligned)

    def __repr__(self) -> str:
        try:
            shift = self.shift is not None
            if self.input_names is None:
                return f"Linear(shape={self._ops[0].shape}, shift={shift})"
            pairs = zip(self.input_names, self._ops, strict=True)
            dims = {n: A.shape[1] for n, A in pairs}
            return f"Linear(inputs={dims}, output_dim={self.output_dim}, shift={shift})"
        except Exception:
            return "Linear(<unreadable>)"

    # -- private -------------------------------------------------------------------
    def _fit(self, where: str, dims: dict, inputs) -> None:
        if len(inputs) != len(self._ops):
            raise ValueError(
                f"{where}: {self!r} takes {len(self._ops)} input block(s), got {inputs}"
            )
        if self.input_names is not None and tuple(inputs) != self.input_names:
            raise ValueError(
                f"{where}: {self!r} was built for the inputs {self.input_names}, in "
                f"that order; got inputs {tuple(inputs)}"
            )
        for name, A in zip(inputs, self._ops, strict=True):
            if A.shape[1] != dims[name]:
                raise ValueError(
                    f"{where}: {self!r} applies an operator of shape {A.shape} to "
                    f"block {name!r} of dimension {dims[name]}"
                )
        _check_unbatched(where, self, (*self._ops, self.shift))


@c.map_class
class AdditiveNoise:
    r"""Independent additive Gaussian noise.

    .. math::

        x \mapsto x + e, \qquad
        e \sim \mathcal N(0, R) \ \text{independent of everything else,}

    with :math:`R` = ``cov``. It takes exactly one input, of dimension
    ``cov.dim``.

    On a Gaussian it is exact, with no sampling. Replacing its input, it is
    :meth:`~enskit.distribution.Gaussian.add_noise`. To a new block it copies
    the input and gives the copy :math:`R` as its independent term,

    .. math::

        m_y = m_x, \qquad F_y = F_x, \qquad D_y = R,

    after absorbing the input's own independent term, if it has one, so that
    the copy stays correlated with it; an
    :class:`~enskit.distribution.EnsembleGaussian` then becomes a plain
    Gaussian, and otherwise stays one.

    On an ensemble it draws the noise, :math:`y_j = x_j + L\eta_j` with
    :math:`L` = ``cov.factor()`` and :math:`\eta_j \sim \mathcal N(0, I)`,
    which needs a key and ``factor``. It is not callable: it is not a
    function of its input.

    Parameters
    ----------
    cov : PSDLinOp
        The noise covariance :math:`R`.

    Raises
    ------
    TypeError
        If ``cov`` is not a :class:`~enskit.linalg.PSDLinOp`.
    ValueError
        If ``cov`` is a vmapped family.

    Notes
    -----
    The order of operations chooses the estimator. Pushing an ensemble
    through this map and projecting estimates the output's covariance from
    sampled noise; projecting first and pushing the Gaussian adds :math:`R`
    exactly, a lower-variance estimate of the same joint, which needs only
    ``whiten`` of :math:`R` when the output is later conditioned on.

    References
    ----------
    Burgers, G., van Leeuwen, P. J. & Evensen, G. (1998). Analysis scheme in
    the ensemble Kalman filter. *Monthly Weather Review*, 126(6), 1719–1724.
    """

    cov: PSDLinOp

    def __init__(self, cov) -> None:
        if not isinstance(cov, PSDLinOp):
            raise TypeError(
                f"AdditiveNoise: cov must be a PSDLinOp, got {type(cov).__name__}"
            )
        if cov.batch_shape != ():
            raise ValueError(
                f"AdditiveNoise: cov {cov!r} is a vmapped family with batch shape "
                f"{cov.batch_shape}; build the map inside jax.vmap instead"
            )
        object.__setattr__(self, "cov", cov)

    @property
    def dim(self) -> int:
        """The dimension of the input and the output."""
        return self.cov.shape[0]

    def push_ensemble(self, ensemble: Ensemble, inputs, output: str, key) -> Ensemble:
        """Draw the noise for every particle; the :class:`StructuredMap` method."""
        where = "pushforward"
        x = self._fit(where, ensemble.dims, inputs)
        why = ": AdditiveNoise draws the noise on an ensemble"
        c.check_key(where, key, required=True, why=why)
        L = self.cov.factor()
        dtype = ensemble[x].dtype
        eta = jax.random.normal(key, (ensemble.n_particles, L.shape[1]), dtype)
        y = ensemble[x] + L.matvec(eta)
        remedy = "Give the noise covariance the ensemble's dtype"
        arr, _ = c.output_array(
            where, "AdditiveNoise", output, y, ensemble.n_particles, dtype, remedy
        )
        return ensemble.assign({output: arr})

    def push_gaussian(self, gaussian: Gaussian, inputs, output: str) -> Gaussian:
        """Add the noise exactly; the :class:`StructuredMap` method."""
        where = "pushforward"
        x = self._fit(where, gaussian.dims, inputs)
        if output == x:
            return gaussian.add_noise({x: self.cov})
        has_term = gaussian.block_cov(x) is not None
        base = gaussian.absorb(x) if has_term else gaussian
        aligned = isinstance(gaussian, EnsembleGaussian) and not has_term
        return c.with_block(
            base, output, base.mean(x), base.factor(x), self.cov, aligned=aligned
        )

    def __repr__(self) -> str:
        try:
            return f"AdditiveNoise(dim={self.dim})"
        except Exception:
            return "AdditiveNoise(<unreadable>)"

    # -- private -------------------------------------------------------------------
    def _fit(self, where: str, dims: dict, inputs) -> str:
        if len(inputs) != 1:
            raise ValueError(
                f"{where}: AdditiveNoise takes exactly one input block, got {inputs}"
            )
        (x,) = inputs
        if dims[x] != self.dim:
            raise ValueError(
                f"{where}: {self!r} adds noise of dimension {self.dim} to block {x!r} "
                f"of dimension {dims[x]}"
            )
        _check_unbatched(where, self, (self.cov,))
        return x


# -- private -----------------------------------------------------------------------


def _as_operator(where: str, what: str, op) -> LinOp:
    """A ``LinOp`` as given, or a floating 2-D array wrapped in ``Dense``."""
    if isinstance(op, LinOp):
        A = op
    else:
        try:
            arr = jnp.asarray(op)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{where}: {what} must be a LinOp or a 2-D array, got "
                f"{type(op).__name__}"
            ) from exc
        if not jnp.issubdtype(arr.dtype, jnp.floating):
            raise TypeError(
                f"{where}: {what} must be a real floating array, got dtype {arr.dtype}"
            )
        if arr.ndim != 2:
            raise ValueError(
                f"{where}: {what} must be a 2-D array, got shape {arr.shape}"
            )
        A = Dense(arr)
    if A.batch_shape != ():
        raise ValueError(
            f"{where}: {what} {A!r} is a vmapped family with batch shape "
            f"{A.batch_shape}; build the map inside jax.vmap instead"
        )
    return A


def _check_unbatched(where: str, owner, leaves) -> None:
    """Refuse a map rebuilt from stacked leaves, outside the ``vmap`` it came from."""
    for leaf in leaves:
        if leaf is None:
            continue
        batch = leaf.batch_shape if isinstance(leaf, LinOp) else leaf.shape[:-1]
        if batch != ():
            raise ValueError(
                f"{where}: {owner!r} is a vmapped family with batch shape {batch}, "
                f"which cannot be used directly; apply it under jax.vmap"
            )
