"""Prototype inflation and relaxation policies shared by both drivers."""
from __future__ import annotations

import dataclasses

from .. import kalman


@dataclasses.dataclass(frozen=True)
class MultiplicativeInflation:
    """Before each update: x -> mean + anomaly_scale (x - mean), every block."""
    anomaly_scale: float

    def __call__(self, key, *, ensemble, **_):
        return kalman.inflate_multiplicative(ensemble, self.anomaly_scale)


@dataclasses.dataclass(frozen=True)
class RelaxToPriorSpread:
    """After each update: RTPS toward the ensemble the update started from."""
    alpha: float

    def __call__(self, *, prior, posterior, **_):
        return kalman.relax_to_prior_spread(prior, posterior, self.alpha)
