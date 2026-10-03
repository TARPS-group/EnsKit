"""Inflation and relaxation policies, shared by every driver (``enskit.algorithms``).

A driver calls an *inflation* on the particles just before they are used (EKI:
before each evaluation; EnKF: after each forecast) and a *relaxation* just
after each update. Both are plain callables; these classes wrap the
functions in :mod:`enskit.kalman` with the drivers' calling convention. Every
driver passes the context it has (``step`` and ``beta`` for EKI, ``time`` for
EnKF) as keywords, and a policy ignores what it does not use.
"""


class Inflation(Protocol):
    def __call__(self, key, *, ensemble: Ensemble, **context) -> Ensemble:
        """Return the inflated particles; same blocks, particles and dtype."""


class Relaxation(Protocol):
    def __call__(self, *, prior: Ensemble, posterior: Ensemble, **context) -> Ensemble:
        """Return relaxed posterior particles; ``prior`` is the update's input."""


class MultiplicativeInflation:
    """``kalman.inflate_multiplicative`` on every block (or ``names``).

    Parameters
    ----------
    anomaly_scale : float
        Scales anomalies; the covariance scales by its square, so 1.05 is
        about a 10% variance inflation. Stored as a 0-d array, so a traced
        value flows through.
    names : tuple of str, optional
        Keyword-only. Blocks to inflate; all by default.
    """


class AdditiveInflation:
    """``kalman.inflate_additive`` with one covariance per block.

    Parameters
    ----------
    covs : Mapping[str, PSDLinOp]
        Each must support ``factor``.
    """


class RelaxToPriorSpread:
    """``kalman.relax_to_prior_spread`` with weight ``alpha`` in ``[0, 1]``."""


class RelaxToPriorPerturbations:
    """``kalman.relax_to_prior_perturbations`` with weight ``alpha`` in ``[0, 1]``."""
