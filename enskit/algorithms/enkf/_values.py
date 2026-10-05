r"""The filter's result, its error, and the prediction block's name."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array

from ...distribution import Ensemble

__all__ = ["PREDICTION", "FilterResult", "EnKFError"]

#: The name of the block an analysis predicts the observation into. An
#: ensemble passed to :func:`analysis` or :func:`filter` may not use it.
PREDICTION = "prediction"


@dataclass(frozen=True, eq=False, repr=False)
class FilterResult:
    r"""What :func:`filter` returns. A frozen dataclass, not a pytree.

    Parameters
    ----------
    ensemble : Ensemble
        Keyword-only. The analysis particles at the last time: the ensemble
        to continue from.
    means : dict[str, Array]
        Keyword-only. For every block of ``ensemble``, the ``(T, d)``
        analysis means, row :math:`t` after the analysis of observation
        :math:`t`.
    log_evidence : Array
        Keyword-only. The ``(T,)`` one-step log evidences,
        :math:`\log \hat p(y_t \mid y_{1:t-1})`.
    ensembles : tuple of Ensemble, optional
        Keyword-only. Every analysis ensemble, in time order, when the filter
        ran with ``keep_ensembles=True``; ``None`` otherwise.

    Raises
    ------
    TypeError
        If a field has the wrong type.
    ValueError
        If the means, the log evidences and the ensembles disagree on the
        number of times, or the means' blocks are not the ensemble's.
    """

    ensemble: Ensemble = field(kw_only=True)
    means: dict[str, Array] = field(kw_only=True)
    log_evidence: Array = field(kw_only=True)
    ensembles: tuple[Ensemble, ...] | None = field(kw_only=True, default=None)

    def __post_init__(self) -> None:
        where = "FilterResult"
        if not isinstance(self.ensemble, Ensemble):
            raise TypeError(
                f"{where}.ensemble: must be an Ensemble, got "
                f"{type(self.ensemble).__name__}"
            )
        if not isinstance(self.means, dict):
            raise TypeError(
                f"{where}.means: must be a dict from block name to array, got "
                f"{type(self.means).__name__}"
            )
        if tuple(self.means) != self.ensemble.names:
            raise ValueError(
                f"{where}.means: has blocks {tuple(self.means)}, but the ensemble has "
                f"{self.ensemble.names}"
            )
        shape = getattr(self.log_evidence, "shape", None)
        if shape is None or len(shape) != 1:
            raise ValueError(
                f"{where}.log_evidence: must be a (T,) array, got "
                f"{shape if shape is not None else type(self.log_evidence).__name__}"
            )
        n_times = shape[0]
        for name, mean in self.means.items():
            want = (n_times, self.ensemble.dims[name])
            if tuple(getattr(mean, "shape", ())) != want:
                raise ValueError(
                    f"{where}.means[{name!r}]: expected shape {want}, got "
                    f"{getattr(mean, 'shape', type(mean).__name__)}"
                )
        if self.ensembles is not None:
            if not isinstance(self.ensembles, tuple) or not all(
                isinstance(e, Ensemble) for e in self.ensembles
            ):
                raise TypeError(f"{where}.ensembles: must be a tuple of Ensemble or None")
            if len(self.ensembles) != n_times:
                raise ValueError(
                    f"{where}.ensembles: holds {len(self.ensembles)} ensembles for "
                    f"{n_times} times"
                )

    @property
    def n_times(self) -> int:
        """The number of observations the filter assimilated, :math:`T`."""
        return int(self.log_evidence.shape[0])

    @property
    def total_log_evidence(self) -> Array:
        r"""The sum of the one-step log evidences, :math:`\log \hat p(y_{1:T})`."""
        return jnp.sum(self.log_evidence)

    def __repr__(self) -> str:
        """As ``FilterResult(n_times=300, blocks=('x',))``; never raises."""
        try:
            return f"FilterResult(n_times={self.n_times}, blocks={self.ensemble.names})"
        except Exception:
            return "<FilterResult (unprintable)>"


class EnKFError(RuntimeError):
    """A filter cannot continue: a particle stopped being finite.

    Raised by :func:`filter` when the transition, the inflation, the
    observation model, the update or the relaxation returns a particle with a
    non-finite entry. Every later analysis would be ``nan``, so the filter
    stops at that time.

    Attributes
    ----------
    time : int
        The time being assimilated when it failed, counted from the filter's
        ``start_time``: row ``time - start_time`` of its observations.
    key : jax.random key or None
        The filter's key at the start of that time, before its split.
    result : FilterResult
        The times before ``time``: its ``ensemble`` is the last good analysis
        ensemble, or the initial one if the first time failed.

    Notes
    -----
    The attributes make a caught error resumable, with the draws and the
    times the uninterrupted filter would have used::

        try:
            result = enkf.filter(ens, ys, key=key, ...)
        except enkf.EnKFError as exc:
            done = exc.result
            # after changing whatever made it fail:
            rest = enkf.filter(done.ensemble, ys[exc.time:], key=exc.key,
                               start_time=exc.time, ...)
    """

    def __init__(self, message: str, *, time: int, key, result: FilterResult) -> None:
        super().__init__(message)
        self.time = time
        self.key = key
        self.result = result
