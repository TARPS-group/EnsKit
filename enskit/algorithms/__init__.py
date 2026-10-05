r"""Algorithms built from the layers below: the EKI driver and its policies.

This module provides the algorithms themselves and the policies they share.

==================================== ==========================================
object                               is
==================================== ==========================================
:mod:`enskit.algorithms.eki`         Ensemble Kalman Inversion: tempering
                                     schedules, stopping rules and the driver
:class:`Inflation`                   protocol: a transformation of the
                                     particles before they are used
:class:`Relaxation`                  protocol: a transformation of an update's
                                     result
:class:`MultiplicativeInflation`     scale the anomalies
:class:`AdditiveInflation`           add centered Gaussian draws
:class:`RelaxToPriorSpread`          RTPS: relax the spread toward the prior's
:class:`RelaxToPriorPerturbations`   RTPP: blend posterior and prior anomalies
==================================== ==========================================

A driver calls an inflation on the particles just before it uses them (EKI:
before each evaluation of the forward model), as
``inflation(key, ensemble=..., **context)``, and a relaxation just after each
update, as ``relaxation(prior=..., posterior=..., **context)``. Each driver
passes the context it has as keywords (``step`` and ``beta`` for EKI), and a
policy ignores what it does not use. The four classes wrap the functions of
:mod:`enskit.kalman` in that calling convention; any callable with it is a
policy too.

Notes
-----
This is the one layer that speaks of runs, steps, levels, schedules,
observations and time; the layers below know only distributions, maps and
updates. Inflation and relaxation leave the target of a run on purpose: a run
that uses them is no longer a tempering ladder for the targets of
:mod:`enskit.algorithms.eki`.
"""

from . import eki
from ._policies import (
    AdditiveInflation,
    Inflation,
    MultiplicativeInflation,
    Relaxation,
    RelaxToPriorPerturbations,
    RelaxToPriorSpread,
)

__all__ = [
    "eki",
    "Inflation",
    "Relaxation",
    "MultiplicativeInflation",
    "AdditiveInflation",
    "RelaxToPriorSpread",
    "RelaxToPriorPerturbations",
]

for _name in __all__[1:]:
    globals()[_name].__module__ = __name__
del _name
