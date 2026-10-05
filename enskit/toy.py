"""Toy problems: small models, with the priors and data to drive them.

Three inverse problems and two state-space problems, for the package's own
tests and for the documentation. An inverse problem bundles a forward model
with a prior, a noise covariance, synthetic data and the parameters that
generated them; a state-space problem bundles a transition and an observation
model with an initial distribution, a noise covariance, a sequence of
synthetic observations and the states that generated them. A tutorial or a
test spends two lines on setup rather than twelve.

============================ ==================================================
factory                      builds
============================ ==================================================
:func:`linear_gaussian`      :class:`LinearGaussian` — a linear model, whose
                             posterior is available in closed form, at any
                             pair of dimensions
:func:`exponential_decay`    :class:`ExponentialDecay` — two parameters, mildly
                             nonlinear
:func:`restricted_decay`     :class:`RestrictedDecay` — the same decay model
                             with a valid domain, so particles outside it fail
:func:`lorenz96`             :class:`Lorenz96` — a chaotic state on a ring of
                             sites, half of them observed, with site
                             coordinates for localization
:func:`linear_state_space`   :class:`LinearStateSpace` — linear dynamics and
                             observations, whose exact filter is available
============================ ==================================================

:func:`lorenz96_step` is the Lorenz-96 transition as a function, for a
transition that reads more blocks than the state.

**These are not production models, and they are not an interface.** EnsKit
ships no forward models for real use and defines no base class, protocol or
registry for one: a forward model, a transition or an observation model is
any simulator in the sense of :func:`enskit.maps.pushforward`, here taking a
``(J, P)`` array of parameters to ``(J, N)`` predictions, or ``(J, d)``
states to ``(J, d)`` states. What these classes exemplify is that callable
and the problem around it, nothing more.

Using one::

    import jax
    from enskit import kalman, toy
    from enskit.algorithms import eki

    problem = toy.exponential_decay()
    state = eki.EKIState.from_prior(jax.random.key(0), problem.prior, n_particles=64)
    result = eki.run(state, problem.forward, problem.y, problem.noise_cov,
                     update_rule=kalman.Matheron(),
                     schedule=eki.AdaptiveESSSchedule())

    result.mean("u"), problem.u_true      # the answer, and what generated the data

Conventions shared by all five:

- **The parameters are the block** ``"u"`` **and the state the block**
  ``"x"``. Each prior or initial distribution is a
  :class:`~enskit.distribution.Gaussian` over that one block, and the models
  are called with its ``(J, P)`` or ``(J, d)`` array.
- **A problem is a frozen dataclass of plain public values, and is not
  callable.** Pass its pieces (``forward``, ``y`` and ``noise_cov``; or
  ``transition``, ``observe``, ``noise_cov`` and ``observations``) as
  separate arguments; nothing in :mod:`enskit.algorithms` accepts a problem
  object in their place.
- **Every field is a value a caller could have written themselves**, so a
  problem can be modified by constructing the class directly rather than
  through its factory — with a different noise covariance, for instance.
- **A problem has no ensemble size.** :math:`J` is chosen where the initial
  particles are drawn, and a forward model is independent of it.
- **The synthetic data are fixed at construction** from the factory's
  ``seed``, so a docs build or a test sees the same numbers every time — at
  the package's default float64. With JAX's x64 mode disabled, the draws
  consume randomness differently and every factory builds a *different*
  problem, not merely a less precise one.
- **These models happen to be pure JAX**, hence ``jit``-able, ``vmap``-pable
  and differentiable. That is a convenience of *these* models, chosen to keep
  the test suite cheap, and **not** an obligation on yours: the driver loop is
  ordinary Python precisely so that a subprocess, a scheduler submission or a
  legacy binary is a legal forward model. The simulator contract of
  :mod:`enskit.maps` states the whole obligation.

Notes
-----
Nothing in the layers imports this module, and nothing should: it depends on
them, so an import in the other direction would make toy problems
load-bearing for the library.

Row independence — row :math:`j` of a model's return depending only on row
:math:`j` of its argument — is the one requirement a forward model must meet
that nothing in a run detects. Each model here is written so that the property
is visible rather than asserted: the linear model applies an operator to the
trailing axis, and the two decay models are :func:`jax.vmap` of a function of
one particle. :func:`enskit.testing.check_simulator` tests the property from
outside, on a model of your own.

The closed-form posterior of :class:`LinearGaussian` is not computed here. It
is :func:`enskit.maps.pushforward` through :class:`enskit.maps.Linear`
followed by :meth:`~enskit.distribution.Gaussian.add_noise` and
:meth:`~enskit.distribution.Gaussian.condition`, which is where the
mathematics belongs and which is worth seeing: checking a run against a known
answer does not mean writing your own algebra.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import Array

from . import maps
from .distribution import Gaussian
from .linalg import Dense, LinOp, PSDDiagonal, PSDLinOp, value_check

__all__ = [
    "linear_gaussian",
    "LinearGaussian",
    "exponential_decay",
    "ExponentialDecay",
    "restricted_decay",
    "RestrictedDecay",
    "lorenz96",
    "lorenz96_step",
    "Lorenz96",
    "linear_state_space",
    "LinearStateSpace",
]

# The largest array LinearGaussian.posterior will build, in elements. The
# conditioning forms a (P, k) factor and a (k, k) transform for a prior factor
# of width k, so the cap is on the larger of the two products, and the peak is
# a small multiple of it -- see LinearGaussian.posterior's Notes.
_MAX_POSTERIOR_ELEMENTS = 20_000_000

# The steps the Lorenz-96 truth is run before its first recorded state, from
# a start near the unstable fixed point x = F onto the attractor.
_LORENZ96_SPIN_UP = 1000

# The state-space problems derive their keys from fold_in(key(seed), this),
# never from key(seed) itself: with JAX's partitionable threefry,
# fold_in(k, i) equals split(k, n)[i] for every n > i, so a key folded with
# a small integer is one a caller's split of the same seed also produces.
_TOY_STREAM = 0x746F79


# ---------------------------------------------------------------------------
# the linear-Gaussian problem
# ---------------------------------------------------------------------------


def linear_gaussian(
    *,
    parameter_dim: int = 4,
    data_dim: int = 8,
    prior_std: float = 1.0,
    noise_std: float = 0.1,
    seed: int = 0,
) -> LinearGaussian:
    """A :class:`LinearGaussian` problem at a chosen pair of dimensions.

    The map's entries are drawn :math:`N(0, 1/P)`, so a prediction is of the
    same order as ``prior_std`` whatever ``P`` is, and the signal-to-noise
    ratio is about ``prior_std / noise_std`` at every size. The true
    parameters are a draw from the prior, and the observation is
    :math:`G u_\\star` plus a draw from the observation error — so the problem
    is well specified, and the posterior really does concentrate near
    ``u_true`` where the data can see it.

    Parameters
    ----------
    parameter_dim, data_dim
        The number of parameters :math:`P` and predictions :math:`N`.
        Keyword-only.
    prior_std, noise_std
        The prior standard deviation, the same in every parameter, and the
        observation error standard deviation, the same in every prediction.
        Both positive. Keyword-only.
    seed
        Seeds the map, the true parameters and the observation error.
        Keyword-only.

    Returns
    -------
    LinearGaussian

    Raises
    ------
    ValueError
        If a dimension is below 1, or a standard deviation is not positive.

    Notes
    -----
    Pass ``parameter_dim=2000, data_dim=40`` for a problem where the parameter
    dimension far exceeds any affordable ensemble size — the regime in which a run can
    only represent a :math:`J-1`-dimensional subspace of the answer, and in
    which comparing against :meth:`LinearGaussian.posterior` is the only way
    to see that.
    """
    _check_dim("linear_gaussian", "parameter_dim", parameter_dim)
    _check_dim("linear_gaussian", "data_dim", data_dim)
    _check_scale("linear_gaussian", "prior_std", prior_std)
    _check_scale("linear_gaussian", "noise_std", noise_std)
    key_map, key_truth, key_noise = jax.random.split(jax.random.key(seed), 3)
    G = Dense(
        jax.random.normal(key_map, (data_dim, parameter_dim)) / jnp.sqrt(parameter_dim)
    )
    prior_cov = PSDDiagonal(jnp.full(parameter_dim, float(prior_std) ** 2))
    prior = Gaussian({"u": jnp.zeros(parameter_dim)}, block_covs={"u": prior_cov})
    u_true = prior_cov.factor().matvec(
        jax.random.normal(key_truth, (1, parameter_dim))
    )[0]
    error = noise_std * jax.random.normal(key_noise, (data_dim,))
    return LinearGaussian(
        G=G,
        prior=prior,
        noise_cov=PSDDiagonal(jnp.full(data_dim, float(noise_std) ** 2)),
        y=G.matvec(u_true) + error,
        u_true=u_true,
    )


@dataclass(frozen=True, eq=False, repr=False)
class LinearGaussian:
    """A linear forward model :math:`v = Gu`, and the problem around it.

    The one problem here whose posterior is available exactly, which makes it
    the problem to check a run against: with a Gaussian prior and a linear
    model, :meth:`posterior` is the answer the run is approximating.

    Build one with :func:`linear_gaussian`, which draws :math:`G`, the true
    parameters and the observation from a seed. Construct the class directly
    to supply any of them yourself.

    Parameters
    ----------
    G
        The linear map, a :class:`~enskit.linalg.LinOp` of shape ``(N, P)``.
    prior
        The prior over the parameters, a
        :class:`~enskit.distribution.Gaussian` with the one block ``"u"`` of
        dimension ``P``.
    noise_cov
        The observation error covariance, a
        :class:`~enskit.linalg.PSDLinOp` of side ``N``.
    y
        The observation, a ``(N,)`` array.
    u_true
        The parameters the observation was generated from, a ``(P,)`` array.

    Raises
    ------
    ValueError
        If the five fields disagree on ``P`` or ``N``.
    TypeError
        If ``G`` is not a :class:`~enskit.linalg.LinOp`, ``prior`` not a
        :class:`~enskit.distribution.Gaussian`, or ``noise_cov`` not a
        :class:`~enskit.linalg.PSDLinOp`.
    """

    G: LinOp = field(kw_only=True)
    prior: Gaussian = field(kw_only=True)
    noise_cov: PSDLinOp = field(kw_only=True)
    y: Array = field(kw_only=True)
    u_true: Array = field(kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.G, LinOp):
            raise TypeError(
                f"LinearGaussian.G: must be an enskit.linalg.LinOp, got "
                f"{type(self.G).__name__}. Wrap a dense matrix as "
                f"enskit.linalg.Dense(G); the closed-form posterior needs an "
                f"operator."
            )
        _check_not_family("LinearGaussian", "G", self.G)
        _check_problem(
            "LinearGaussian",
            parameter_dim=self.G.shape[1],
            data_dim=self.G.shape[0],
            prior=self.prior,
            noise_cov=self.noise_cov,
            y=self.y,
            u_true=self.u_true,
        )

    @property
    def parameter_dim(self) -> int:
        """The number of parameters :math:`P`."""
        return int(self.G.shape[1])

    @property
    def data_dim(self) -> int:
        """The number of predictions :math:`N`."""
        return int(self.G.shape[0])

    def forward(self, ensemble) -> Array:
        """The forward model: ``(J, P)`` parameters in, ``(J, N)`` predictions out.

        Parameters
        ----------
        ensemble
            The parameters to evaluate, a ``(J, P)`` array — every particle's
            block ``"u"``, one per row, not a single parameter vector.

        Returns
        -------
        Array
            The predictions, ``(J, N)``, row :math:`j` from row :math:`j`.

        Raises
        ------
        ValueError
            If ``ensemble`` is not exactly two-dimensional with ``P`` columns.

        Notes
        -----
        ``G.matvec`` contracts the *trailing* axis, so the leading axis is
        carried through as a batch and the particles stay in their rows. Written
        with a dense array instead, the same contraction is ``ensemble @ G.T``;
        ``ensemble @ G`` raises for a rectangular ``G`` and silently returns
        the transposed model's predictions for a square one.

        ``jit``-able and ``vmap``-pable, as a convenience of these models
        rather than a requirement on yours — see :mod:`enskit.toy`.
        """
        return self.G.matvec(
            _check_ensemble("LinearGaussian", ensemble, self.parameter_dim)
        )

    def __repr__(self) -> str:
        """As ``LinearGaussian(parameter_dim=4, data_dim=8)``; never raises."""
        try:
            return (
                f"LinearGaussian(parameter_dim={self.parameter_dim}, "
                f"data_dim={self.data_dim})"
            )
        except Exception:
            return "<LinearGaussian (unprintable fields)>"

    def posterior(self, beta: float = 1.0) -> Gaussian:
        """The exact posterior at tempering level :math:`\\beta`: a closed form.

        The posterior of the linear-Gaussian problem, conditioning on
        :math:`y` with the noise covariance :math:`R/\\beta` — so ``beta=1.0``
        is the Bayesian posterior and a smaller value is the intermediate
        target a run passes through on the way to it.

        Three operations of the layers below::

            joint = self.prior.pipe(maps.pushforward, maps.Linear(self.G),
                                    inputs="u", output="g")
            return joint.add_noise(g=self.noise_cov / beta).condition(g=self.y)

        Copy them to reach the rest of that object: ``joint`` is the joint
        Gaussian of the parameters and the noise-free predictions, and its
        marginal ``"g"`` the prior predictive distribution of them.

        Parameters
        ----------
        beta
            The tempering level :math:`\\beta`, a positive finite scalar.
            Defaults to 1.0. Named as the state and every policy name it, so
            ``problem.posterior(result.beta)`` reads directly.

        Returns
        -------
        Gaussian
            The posterior, a :class:`~enskit.distribution.Gaussian` over the
            block ``"u"``. Its covariance holds a ``(P, k)`` factor, for ``k``
            the width of the prior covariance's factor — never a dense
            ``P``-by-``P`` matrix. Directly comparable with
            :meth:`~enskit.distribution.Ensemble.project` of a run's final
            ensemble.

        Raises
        ------
        ValueError
            If ``beta`` is not positive and finite, or if the posterior
            factor would exceed this module's element budget.
        UnsupportedOpError
            If the prior covariance does not support ``factor``, or the noise
            covariance does not support ``whiten``.

        Notes
        -----
        Nothing is inverted, so a *singular* prior covariance is fine here —
        the case a precision-form posterior
        :math:`(C_0^{-1} + \\beta G^\\top R^{-1} G)^{-1}` cannot express at
        all.

        The cost is set by ``P`` and the width ``k`` of the prior's factor.
        At a full-rank prior in 2000 dimensions the posterior factor is
        ``(2000, 2000)``, or 32 MB, and the conditioning forms ``(k, k)``
        intermediates beside it. The guard is on the larger of ``P * k`` and
        ``k * k``, capped at 20 million elements, and it raises rather than
        allocating.

        The run itself has no such limit, so a high-dimensional problem can be
        inverted where its closed form cannot be written down. The three
        operations above bypass the guard, deliberately: it is a guard on a
        toy convenience, not a limit of the distribution layer.
        """
        beta = float(beta)
        if not (beta > 0.0) or not math.isfinite(beta):
            raise ValueError(
                f"LinearGaussian.posterior: beta must be positive and finite, "
                f"got {beta}. The target at beta 0 is the prior itself, which "
                f"is the `prior` field."
            )
        latent_dim = self.prior.latent_dim
        term = self.prior.block_cov("u")
        if term is not None:
            latent_dim += term.factor().shape[1]
        # The conditioning forms a (P, k) factor *and* a (k, k) transform, so
        # bound the larger of the two: a prior factor wider than P would sail
        # past a cap on P * k while building a k * k array.
        elements = max(self.parameter_dim, latent_dim) * latent_dim
        if elements > _MAX_POSTERIOR_ELEMENTS:
            raise ValueError(
                f"LinearGaussian.posterior: the largest array it would build "
                f"is {max(self.parameter_dim, latent_dim)}-by-{latent_dim}, "
                f"{elements} elements, above this module's budget of "
                f"{_MAX_POSTERIOR_ELEMENTS}. The closed form is not available "
                f"at this size; the run is."
            )
        joint = self.prior.pipe(
            maps.pushforward, maps.Linear(self.G), inputs="u", output="g"
        )
        return joint.add_noise(g=self.noise_cov / beta).condition(g=self.y)


# ---------------------------------------------------------------------------
# the decay problem, and its restricted variant
# ---------------------------------------------------------------------------


def exponential_decay(
    *,
    n_times: int = 12,
    t_max: float = 3.0,
    noise_std: float = 0.02,
    seed: int = 0,
) -> ExponentialDecay:
    """An :class:`ExponentialDecay` problem on an evenly spaced set of points.

    The points are ``n_times`` values evenly spaced over ``(0, t_max]``, the
    true parameters are ``(2.0, 1.5)``, and the prior is
    :math:`\\mathcal{N}\\bigl((1, 1), I\\bigr)`. The same functional form as
    the user guide's "Writing a forward model" page, at a faster true rate, a
    wider prior over the rate, a five times narrower observation error and
    twelve observation points rather than three.

    Parameters
    ----------
    n_times
        The number of observation points :math:`N`. Keyword-only.
    t_max
        The last point. Keyword-only.
    noise_std
        The observation error standard deviation, the same at every point.
        Keyword-only.
    seed
        Seeds the observation error. Keyword-only.

    Returns
    -------
    ExponentialDecay

    Raises
    ------
    TypeError
        If ``n_times`` is not an ``int``.
    ValueError
        If ``n_times`` is below 1, or ``t_max`` or ``noise_std`` is not
        positive and finite.

    Notes
    -----
    The defaults are chosen so that assimilating the observation in one unit
    step and assimilating it gradually reach *reliably* different answers
    rather than coincidentally different ones. Measured over the eight
    observation seeds 0 to 7, against
    :class:`~enskit.algorithms.eki.AdaptiveESSSchedule` at 64 particles with
    :class:`~enskit.kalman.SymmetricSquareRoot`: the two posterior means
    differ by between 0.088 and 0.35 in the rate, and the gradual answer is
    nearer ``u_true`` at every seed, by a factor between 2.7 and 27.
    ``tests/test_toy.py`` asserts both over all eight.
    """
    times, prior, noise_cov, y, u_true = _decay_problem(
        n_times=n_times,
        t_max=t_max,
        noise_std=noise_std,
        seed=seed,
        where="exponential_decay",
    )
    return ExponentialDecay(
        times=times, prior=prior, noise_cov=noise_cov, y=y, u_true=u_true
    )


@dataclass(frozen=True, eq=False, repr=False)
class ExponentialDecay:
    """Two parameters, mildly nonlinear: an amplitude and a decay rate.

    .. math::

        v_i(u) = u_0 \\, e^{-u_1 t_i}, \\qquad i = 1, \\dots, N ,

    for a fixed set of points :math:`t_i`. Two parameters, so results print on
    one line and plot in a plane, and nonlinear enough that how gradually the
    observation is assimilated changes the answer. There is no closed-form
    posterior — use :class:`LinearGaussian` where an exact answer is the
    point.

    Build one with :func:`exponential_decay`.

    Parameters
    ----------
    times
        The points :math:`t_i`, a ``(N,)`` array.
    prior
        The prior over ``(amplitude, rate)``, a
        :class:`~enskit.distribution.Gaussian` over the one block ``"u"`` of
        dimension 2.
    noise_cov
        The observation error covariance, a
        :class:`~enskit.linalg.PSDLinOp` of side ``N``.
    y
        The observation, a ``(N,)`` array.
    u_true
        The parameters the observation was generated from, a ``(2,)`` array.

    Raises
    ------
    ValueError
        If ``times`` is not rank 1, or the fields disagree on ``N`` or on the
        two parameters.
    TypeError
        If ``prior`` is not a :class:`~enskit.distribution.Gaussian` or ``noise_cov``
        not a :class:`~enskit.linalg.PSDLinOp`.
    """

    times: Array = field(kw_only=True)
    prior: Gaussian = field(kw_only=True)
    noise_cov: PSDLinOp = field(kw_only=True)
    y: Array = field(kw_only=True)
    u_true: Array = field(kw_only=True)

    def __post_init__(self) -> None:
        _check_times("ExponentialDecay", self.times)
        _check_problem(
            "ExponentialDecay",
            parameter_dim=2,
            data_dim=int(self.times.shape[0]),
            prior=self.prior,
            noise_cov=self.noise_cov,
            y=self.y,
            u_true=self.u_true,
        )

    @property
    def parameter_dim(self) -> int:
        """The number of parameters, 2."""
        return 2

    @property
    def data_dim(self) -> int:
        """The number of predictions :math:`N`."""
        return int(self.times.shape[0])

    def forward(self, ensemble) -> Array:
        """The forward model: ``(J, P)`` parameters in, ``(J, N)`` predictions out.

        Parameters
        ----------
        ensemble
            The parameters to evaluate, a ``(J, 2)`` array — the whole
            ensemble's block ``"u"``, one particle per row, not a single parameter
            vector.

        Returns
        -------
        Array
            The predictions, ``(J, N)``, row :math:`j` from row :math:`j`.

        Raises
        ------
        ValueError
            If ``ensemble`` is not exactly two-dimensional with 2 columns.

        Notes
        -----
        This is :func:`jax.vmap` of a function of one particle, which is the
        wrapper a model written for a single parameter vector needs — and
        which cannot couple the rows even in principle.

        ``jit``-able and ``vmap``-pable, as a convenience of these models
        rather than a requirement on yours — see :mod:`enskit.toy`.
        """
        _check_ensemble("ExponentialDecay", ensemble, 2)
        return jax.vmap(_decay, in_axes=(0, None))(ensemble, self.times)

    def __repr__(self) -> str:
        """As ``ExponentialDecay(data_dim=12)``; never raises."""
        try:
            return f"ExponentialDecay(data_dim={self.data_dim})"
        except Exception:
            return "<ExponentialDecay (unprintable fields)>"


def restricted_decay(
    *,
    n_times: int = 12,
    t_max: float = 3.0,
    noise_std: float = 0.02,
    rate_floor: float = 0.0,
    seed: int = 0,
) -> RestrictedDecay:
    """A :class:`RestrictedDecay` problem: :func:`exponential_decay`'s, with a domain.

    The same problem in every other respect, so a run against it can be
    compared with one against :func:`exponential_decay` directly. At the
    default floor the prior puts about 16% of its mass outside the valid
    domain, so a few particles of any ensemble drawn from the prior fail, and
    none once the ensemble has concentrated above the floor.

    Parameters
    ----------
    n_times, t_max, noise_std, seed
        As :func:`exponential_decay`. Keyword-only.
    rate_floor
        The domain boundary; the model is defined where the rate exceeds it.
        Raise it toward the prior mean to fail more particles. It must stay
        strictly below the true rate of 1.5, or the observation would have
        been generated where the model does not evaluate. Keyword-only.

    Returns
    -------
    RestrictedDecay

    Raises
    ------
    ValueError
        As :func:`exponential_decay`, and if ``rate_floor`` is at or above the
        true rate.
    """
    times, prior, noise_cov, y, u_true = _decay_problem(
        n_times=n_times,
        t_max=t_max,
        noise_std=noise_std,
        seed=seed,
        where="restricted_decay",
    )
    return RestrictedDecay(
        times=times,
        prior=prior,
        noise_cov=noise_cov,
        y=y,
        u_true=u_true,
        rate_floor=float(rate_floor),
    )


@dataclass(frozen=True, eq=False, repr=False)
class RestrictedDecay:
    """:class:`ExponentialDecay` with a valid domain, so some particles fail.

    The same model, defined only where the decay rate exceeds
    ``rate_floor``:

    .. math::

        v_i(u) = \\begin{cases}
            u_0 \\, e^{-u_1 t_i} & u_1 > \\texttt{rate\\_floor}, \\\\
            \\texttt{nan} & \\text{otherwise.}
        \\end{cases}

    A non-finite prediction row is how a forward model signals that a particle
    failed, and a solver refusing to run or diverging on an out-of-domain
    parameter is the ordinary reason for one. Which particles fail is a
    deterministic function of the ensemble, so a run is reproducible; the
    *fraction* that fail is set by how much prior mass sits below the floor,
    which makes it a property of the problem rather than an injected rate.

    Build one with :func:`restricted_decay`.

    Parameters
    ----------
    times, prior, noise_cov, y, u_true
        As :class:`ExponentialDecay`.
    rate_floor
        The domain boundary: the model is defined where the rate, the second
        parameter, is strictly greater than this. A Python ``float``.

    Raises
    ------
    ValueError
        As :class:`ExponentialDecay`, and if ``rate_floor`` is not finite, or
        if ``u_true`` is outside the valid domain — a problem whose own true
        parameters fail is a mistake rather than a test case.
    TypeError
        As :class:`ExponentialDecay`.

    Notes
    -----
    Under a Gaussian prior with rate mean :math:`m_1` and standard deviation
    :math:`\\sigma_1`, the expected fraction of failing particles is

    .. math::

        \\Phi\\!\\left(
            \\frac{\\texttt{rate\\_floor} - m_1}{\\sigma_1}
        \\right) ,

    so a target failure rate is reached by moving either the floor or the
    prior. The realized fraction varies with the ensemble draw, and falls to
    zero once an ensemble has concentrated above the floor.
    """

    times: Array = field(kw_only=True)
    prior: Gaussian = field(kw_only=True)
    noise_cov: PSDLinOp = field(kw_only=True)
    y: Array = field(kw_only=True)
    u_true: Array = field(kw_only=True)
    rate_floor: float = field(kw_only=True)

    def __post_init__(self) -> None:
        _check_times("RestrictedDecay", self.times)
        _check_problem(
            "RestrictedDecay",
            parameter_dim=2,
            data_dim=int(self.times.shape[0]),
            prior=self.prior,
            noise_cov=self.noise_cov,
            y=self.y,
            u_true=self.u_true,
        )
        if isinstance(self.rate_floor, bool) or not isinstance(
            self.rate_floor, (int, float)
        ):
            raise TypeError(
                f"RestrictedDecay.rate_floor: must be a Python float, got "
                f"{type(self.rate_floor).__name__}. A string or a bool would "
                f"pass a coercion check and then move the domain boundary, or "
                f"fail inside the model with an error naming a tracer."
            )
        floor = float(self.rate_floor)
        if not math.isfinite(floor):
            raise ValueError(
                f"RestrictedDecay.rate_floor: must be finite, got {floor}"
            )
        # Store the coerced value, so the field is the Python float the class
        # documents rather than whatever numeric type was passed.
        object.__setattr__(self, "rate_floor", floor)
        if not float(self.u_true[1]) > floor:
            raise ValueError(
                f"RestrictedDecay: u_true has rate {float(self.u_true[1])}, which "
                f"is outside the valid domain rate > {floor}, so the "
                f"observation was generated where the model does not evaluate"
            )

    @property
    def parameter_dim(self) -> int:
        """The number of parameters, 2."""
        return 2

    @property
    def data_dim(self) -> int:
        """The number of predictions :math:`N`."""
        return int(self.times.shape[0])

    def forward(self, ensemble) -> Array:
        """The forward model: ``(J, P)`` in, ``(J, N)`` out, some rows ``nan``.

        Parameters
        ----------
        ensemble
            The parameters to evaluate, a ``(J, 2)`` array — the whole
            ensemble's block ``"u"``, one particle per row.

        Returns
        -------
        Array
            The predictions, ``(J, N)``. A particle whose rate is not above
            ``rate_floor`` gets a wholly non-finite row, which a run reads as
            a failed particle.

        Raises
        ------
        ValueError
            If ``ensemble`` is not exactly two-dimensional with 2 columns.

        Notes
        -----
        The rows are built by :func:`jax.vmap` of a function of one particle, as
        in :class:`ExponentialDecay`.

        A row is wholly non-finite when the particle is outside the domain. It
        can *also* go partly non-finite inside the domain, for a
        ``rate_floor`` far below zero: :math:`e^{-u_1 t}` overflows to
        ``inf`` below a rate of about :math:`-236` at the default ``t_max``.
        Either way the particle counts as failed, since a run requires every
        entry of a row to be finite. The shipped prior puts that region 237
        standard deviations away, so the default problem never reaches it.

        Returning these same numbers as a NumPy array, or as a nested Python
        list, would give a bit-identical run: the return may be any array-like
        of the right shape. That is what lets a wrapper around an external
        code assemble its rows in Python without converting — and it is how a
        real failing model, which is not pure JAX, meets the same obligation
        this one meets with :func:`jax.numpy.where`.
        """
        _check_ensemble("RestrictedDecay", ensemble, 2)
        return jax.vmap(_restricted_decay, in_axes=(0, None, None))(
            ensemble, self.times, self.rate_floor
        )

    def __repr__(self) -> str:
        """As ``RestrictedDecay(data_dim=12, rate_floor=0.0)``; never raises."""
        try:
            return (
                f"RestrictedDecay(data_dim={self.data_dim}, "
                f"rate_floor={self.rate_floor})"
            )
        except Exception:
            return "<RestrictedDecay (unprintable fields)>"


# ---------------------------------------------------------------------------
# the Lorenz-96 problem
# ---------------------------------------------------------------------------


def lorenz96(
    *,
    state_dim: int = 40,
    n_times: int = 300,
    obs_every: int = 2,
    noise_std: float = 1.0,
    initial_std: float = 1.0,
    forcing: float = 8.0,
    dt: float = 0.05,
    seed: int = 0,
) -> Lorenz96:
    r"""A :class:`Lorenz96` problem: a chaotic state on a ring, half observed.

    The true state starts at :math:`F + 0.01\,z` for a standard normal
    :math:`z \in \mathbb R^d`, is run for :math:`1000` steps onto the
    attractor, and then for ``n_times`` further steps, each one
    :func:`lorenz96_step`. Every ``obs_every``-th site, starting at site 0, is
    observed after every step with independent noise of standard deviation
    ``noise_std``. The initial distribution is centered on the true state
    before the first of those steps:

    .. math::

        x_0 \sim \mathcal N\big(x_0^\dagger,\ \sigma_0^2 I_d\big), \qquad
        y_t = H x_t^\dagger + e_t, \quad e_t \sim \mathcal N(0, \sigma^2 I_N),
        \qquad t = 1, \dots, T,

    with :math:`\sigma_0` = ``initial_std``, :math:`\sigma` = ``noise_std``,
    and :math:`H` the selection of the observed sites.

    Parameters
    ----------
    state_dim
        The number of sites :math:`d`, at least 4. Keyword-only.
    n_times
        The number of observation times :math:`T`. Keyword-only.
    obs_every
        The spacing of the observed sites, from 1 (every site) to
        ``state_dim``. Keyword-only.
    noise_std, initial_std
        The observation noise and initial standard deviations, positive.
        Keyword-only.
    forcing
        :math:`F`, finite. Keyword-only.
    dt
        The step length, positive. Keyword-only.
    seed
        Seeds the starting point and the observation noise. Keyword-only.

    Returns
    -------
    Lorenz96

    Raises
    ------
    TypeError
        If a size is not an ``int``.
    ValueError
        If ``state_dim`` is below 4, ``n_times`` below 1, ``obs_every``
        outside ``[1, state_dim]``, a standard deviation or ``dt`` not
        positive and finite, or ``forcing`` not finite.

    Notes
    -----
    At the defaults, an ensemble Kalman filter with 40 particles and
    :class:`~enskit.kalman.Matheron` tracks the truth with an analysis error
    well below the observation noise; with 10 particles it needs
    localization (:class:`~enskit.kalman.LocalizedUpdateRule`), with which
    :attr:`Lorenz96.coords` and :attr:`Lorenz96.obs_coords` are meant to be
    used.

    The keys are split from ``jax.random.fold_in(jax.random.key(seed),
    0x746F79)``, so no key a caller splits from ``jax.random.key(seed)`` into
    fewer than ``0x746F79`` (about 7.6 million) keys reproduces the problem's
    draws.

    References
    ----------
    Lorenz, E. N. (1996). Predictability: a problem partly solved. In
    *Proceedings of the ECMWF Seminar on Predictability*, vol. 1, 1–18.
    Reading, UK.
    """
    where = "lorenz96"
    _check_dim(where, "state_dim", state_dim)
    if state_dim < 4:
        raise ValueError(
            f"{where}: state_dim must be at least 4, got {state_dim}; the "
            f"equations read the sites i - 2, i - 1 and i + 1, which must be "
            f"distinct from i"
        )
    _check_dim(where, "n_times", n_times)
    _check_dim(where, "obs_every", obs_every)
    if obs_every > state_dim:
        raise ValueError(
            f"{where}: obs_every must be at most state_dim = {state_dim}, got "
            f"{obs_every}"
        )
    _check_scale(where, "noise_std", noise_std)
    _check_scale(where, "initial_std", initial_std)
    _check_scale(where, "dt", dt)
    forcing = _check_finite(where, "forcing", forcing)
    key_start, key_noise = _toy_keys(seed, 2)
    start = forcing + 0.01 * jax.random.normal(key_start, (state_dim,))
    x0, truth = _lorenz96_trajectory(start, forcing, float(dt), n_times)
    observed = jnp.arange(0, state_dim, obs_every)
    data_dim = int(observed.shape[0])
    error = noise_std * jax.random.normal(key_noise, (n_times, data_dim))
    return Lorenz96(
        forcing=forcing,
        dt=float(dt),
        initial=Gaussian(
            {"x": x0},
            block_covs={"x": PSDDiagonal(jnp.full(state_dim, float(initial_std) ** 2))},
        ),
        observe=maps.Linear(Dense(jnp.eye(state_dim)[observed])),
        noise_cov=PSDDiagonal(jnp.full(data_dim, float(noise_std) ** 2)),
        observations=truth[:, observed] + error,
        truth=truth,
        obs_coords=observed.astype(jnp.result_type(float))[:, None],
    )


def lorenz96_step(x, forcing=8.0, dt=0.05) -> Array:
    r"""One fourth-order Runge–Kutta step of the Lorenz-96 equations.

    On the trailing axis of ``x``, sites :math:`i = 0, \dots, d - 1` with
    indices taken modulo :math:`d`:

    .. math::

        f_i(x) = (x_{i+1} - x_{i-2})\, x_{i-1} - x_i + F, \qquad
        x \mapsto x + \tfrac{\Delta t}{6}\,(k_1 + 2k_2 + 2k_3 + k_4),

    with :math:`k_1 = f(x)`, :math:`k_2 = f(x + \tfrac{\Delta t}{2} k_1)`,
    :math:`k_3 = f(x + \tfrac{\Delta t}{2} k_2)` and
    :math:`k_4 = f(x + \Delta t\, k_3)`.

    Parameters
    ----------
    x : Array
        ``(..., d)``; leading axes are a batch, so a ``(J, d)`` ensemble block
        is advanced row by row.
    forcing : float or Array
        :math:`F`, a scalar, or an array broadcasting against ``x``: a
        ``(J, 1)`` block gives each particle its own forcing.
    dt : float
        :math:`\Delta t`.

    Returns
    -------
    Array
        ``x`` advanced by one step, of the broadcast shape.

    References
    ----------
    Lorenz, E. N. (1996). Predictability: a problem partly solved. In
    *Proceedings of the ECMWF Seminar on Predictability*, vol. 1, 1–18.
    Reading, UK.
    """
    return _lorenz96_step(jnp.asarray(x), jnp.asarray(forcing), jnp.asarray(dt))


@dataclass(frozen=True, eq=False, repr=False)
class Lorenz96:
    r"""The Lorenz-96 system on a ring of sites, observed at a subset of them.

    .. math::

        x_t = M(x_{t-1}), \qquad y_t = H x_t + e_t, \quad
        e_t \sim \mathcal N(0, R), \qquad t = 1, \dots, T,

    with :math:`M` one :func:`lorenz96_step` of length ``dt`` at forcing
    ``forcing``, no transition noise, and :math:`H` a selection of sites. The
    state is the block ``"x"``. Build one with :func:`lorenz96`.

    Parameters
    ----------
    forcing, dt
        :math:`F` and :math:`\Delta t`, Python floats. Keyword-only.
    initial
        The distribution of :math:`x_0`, a
        :class:`~enskit.distribution.Gaussian` with the one block ``"x"`` of
        dimension ``d``. Keyword-only.
    observe
        :math:`H`, a :class:`~enskit.maps.Linear` from ``d`` to ``N``.
        Keyword-only.
    noise_cov
        :math:`R`, a :class:`~enskit.linalg.PSDLinOp` of side ``N``.
        Keyword-only.
    observations
        :math:`y_1, \dots, y_T`, a ``(T, N)`` array. Keyword-only.
    truth
        :math:`x_1^\dagger, \dots, x_T^\dagger`, the states the observations
        were generated from, a ``(T, d)`` array: row :math:`t` goes with
        observation row :math:`t`. Keyword-only.
    obs_coords
        Where each observed value lives, a ``(N, 1)`` array of site indices,
        for :class:`~enskit.kalman.DomainLocalization`. Keyword-only.

    Raises
    ------
    TypeError
        If a field has the wrong type.
    ValueError
        If the fields disagree on ``d``, ``N`` or ``T``.
    """

    forcing: float = field(kw_only=True)
    dt: float = field(kw_only=True)
    initial: Gaussian = field(kw_only=True)
    observe: maps.Linear = field(kw_only=True)
    noise_cov: PSDLinOp = field(kw_only=True)
    observations: Array = field(kw_only=True)
    truth: Array = field(kw_only=True)
    obs_coords: Array = field(kw_only=True)

    def __post_init__(self) -> None:
        name = "Lorenz96"
        _check_finite(name, "forcing", self.forcing)
        _check_scale(name, "dt", self.dt)
        state_dim, data_dim, n_times = _check_state_space(
            name,
            initial=self.initial,
            observe=self.observe,
            noise_cov=self.noise_cov,
            observations=self.observations,
            truth=self.truth,
        )
        if state_dim < 4:
            raise ValueError(
                f"{name}: the state must have at least 4 sites, got {state_dim}"
            )
        _check_array_field(name, "obs_coords", self.obs_coords, (data_dim, 1))

    @property
    def state_dim(self) -> int:
        """The number of sites :math:`d`."""
        return int(self.truth.shape[1])

    @property
    def data_dim(self) -> int:
        """The number of observed values per time, :math:`N`."""
        return int(self.observations.shape[1])

    @property
    def n_times(self) -> int:
        """The number of observation times :math:`T`."""
        return int(self.observations.shape[0])

    @property
    def coords(self) -> Array:
        """Where each state coordinate lives: the ``(d, 1)`` site indices."""
        return jnp.arange(self.state_dim, dtype=jnp.result_type(float))[:, None]

    def transition(self, x) -> Array:
        """The transition :math:`M`: ``(J, d)`` states in, ``(J, d)`` out.

        Parameters
        ----------
        x
            Every particle's state, one per row.

        Returns
        -------
        Array
            ``(J, d)``, each row advanced by one :func:`lorenz96_step`.

        Raises
        ------
        ValueError
            If ``x`` is not exactly two-dimensional with ``d`` columns.
        """
        x = _check_ensemble("Lorenz96", x, self.state_dim, method="transition")
        return lorenz96_step(x, self.forcing, self.dt)

    def __repr__(self) -> str:
        """As ``Lorenz96(state_dim=40, data_dim=20, n_times=300)``; never raises."""
        try:
            return (
                f"Lorenz96(state_dim={self.state_dim}, data_dim={self.data_dim}, "
                f"n_times={self.n_times})"
            )
        except Exception:
            return "<Lorenz96 (unprintable fields)>"


# ---------------------------------------------------------------------------
# the linear state-space problem
# ---------------------------------------------------------------------------


def linear_state_space(
    *,
    state_dim: int = 3,
    data_dim: int = 2,
    n_times: int = 20,
    transition_noise_std: float = 0.3,
    noise_std: float = 0.5,
    initial_std: float = 1.0,
    seed: int = 0,
) -> LinearStateSpace:
    r"""A :class:`LinearStateSpace` problem with random, stable dynamics.

    The transition is :math:`A = 0.95\,U` for a random orthogonal :math:`U`,
    so every eigenvalue has modulus 0.95; :math:`H` has independent
    :math:`\mathcal N(0, 1/d)` entries. The true initial state is a draw from
    the initial distribution, and the truth and observations follow the model
    with the stated noise:

    .. math::

        x_0 \sim \mathcal N(0, \sigma_0^2 I_d), \qquad
        Q = \sigma_q^2 I_d, \qquad R = \sigma^2 I_N,

    with :math:`\sigma_0` = ``initial_std``, :math:`\sigma_q` =
    ``transition_noise_std`` and :math:`\sigma` = ``noise_std``.

    Parameters
    ----------
    state_dim, data_dim
        :math:`d` and :math:`N`. Keyword-only.
    n_times
        The number of observation times :math:`T`. Keyword-only.
    transition_noise_std
        :math:`\sigma_q`, finite and not negative; ``0`` gives no transition
        noise, and ``transition_noise`` is then ``None``. Keyword-only.
    noise_std, initial_std
        Positive. Keyword-only.
    seed
        Seeds the matrices, the true initial state and every noise draw.
        Keyword-only.

    Returns
    -------
    LinearStateSpace

    Raises
    ------
    TypeError
        If a size is not an ``int``.
    ValueError
        If a size is below 1, or a standard deviation is out of range.

    Notes
    -----
    With ``transition_noise_std=0``, an ensemble Kalman filter with the
    symmetric square-root update, started from particles whose sample moments
    equal the initial distribution's, reproduces :meth:`exact_filter` to
    round-off whenever the ensemble has at least :math:`d + 1` particles:
    the sample moments then pass through every step exactly.
    """
    where = "linear_state_space"
    _check_dim(where, "state_dim", state_dim)
    _check_dim(where, "data_dim", data_dim)
    _check_dim(where, "n_times", n_times)
    _check_scale(where, "noise_std", noise_std)
    _check_scale(where, "initial_std", initial_std)
    q = _check_finite(where, "transition_noise_std", transition_noise_std)
    if q < 0.0:
        raise ValueError(
            f"{where}: transition_noise_std must not be negative, got "
            f"{transition_noise_std}"
        )
    k_a, k_h, k_x0, k_q, k_r = _toy_keys(seed, 5)
    U, _ = jnp.linalg.qr(jax.random.normal(k_a, (state_dim, state_dim)))
    A = 0.95 * U
    H = jax.random.normal(k_h, (data_dim, state_dim)) / math.sqrt(state_dim)
    x0 = initial_std * jax.random.normal(k_x0, (state_dim,))
    shocks = q * jax.random.normal(k_q, (n_times, state_dim))
    truth = _linear_trajectory(A, x0, shocks)
    error = noise_std * jax.random.normal(k_r, (n_times, data_dim))
    return LinearStateSpace(
        transition=maps.Linear(Dense(A)),
        transition_noise=None if q == 0.0 else PSDDiagonal(jnp.full(state_dim, q**2)),
        observe=maps.Linear(Dense(H)),
        noise_cov=PSDDiagonal(jnp.full(data_dim, float(noise_std) ** 2)),
        initial=Gaussian(
            {"x": jnp.zeros(state_dim)},
            block_covs={"x": PSDDiagonal(jnp.full(state_dim, float(initial_std) ** 2))},
        ),
        observations=truth @ H.T + error,
        truth=truth,
    )


@dataclass(frozen=True, eq=False, repr=False)
class LinearStateSpace:
    r"""Linear dynamics and linear observations, all noise Gaussian.

    .. math::

        x_0 \sim \mathcal N(m_0, P_0), \qquad
        x_t = A x_{t-1} + w_t, \quad w_t \sim \mathcal N(0, Q), \qquad
        y_t = H x_t + e_t, \quad e_t \sim \mathcal N(0, R),

    for :math:`t = 1, \dots, T`. The filtering distributions are Gaussian and
    :meth:`exact_filter` computes them, which makes this the problem to check
    an ensemble filter against. The state is the block ``"x"``. Build one with
    :func:`linear_state_space`.

    Parameters
    ----------
    transition
        :math:`A`, a :class:`~enskit.maps.Linear` from ``d`` to ``d``.
        Keyword-only.
    transition_noise
        :math:`Q`, a :class:`~enskit.linalg.PSDLinOp` of side ``d``, or
        ``None`` for none. Keyword-only.
    observe
        :math:`H`, a :class:`~enskit.maps.Linear` from ``d`` to ``N``.
        Keyword-only.
    noise_cov
        :math:`R`, of side ``N``. Keyword-only.
    initial
        :math:`\mathcal N(m_0, P_0)`, a
        :class:`~enskit.distribution.Gaussian` with the one block ``"x"``.
        Keyword-only.
    observations
        A ``(T, N)`` array. Keyword-only.
    truth
        The states the observations were generated from, ``(T, d)``.
        Keyword-only.

    Raises
    ------
    TypeError
        If a field has the wrong type.
    ValueError
        If the fields disagree on ``d``, ``N`` or ``T``.
    """

    transition: maps.Linear = field(kw_only=True)
    transition_noise: PSDLinOp | None = field(kw_only=True)
    observe: maps.Linear = field(kw_only=True)
    noise_cov: PSDLinOp = field(kw_only=True)
    initial: Gaussian = field(kw_only=True)
    observations: Array = field(kw_only=True)
    truth: Array = field(kw_only=True)

    def __post_init__(self) -> None:
        name = "LinearStateSpace"
        state_dim, _, _ = _check_state_space(
            name,
            initial=self.initial,
            observe=self.observe,
            noise_cov=self.noise_cov,
            observations=self.observations,
            truth=self.truth,
        )
        _check_linear_map(name, "transition", self.transition, state_dim, state_dim)
        if self.transition_noise is not None:
            if not isinstance(self.transition_noise, PSDLinOp):
                raise TypeError(
                    f"{name}.transition_noise: must be an enskit.linalg.PSDLinOp or "
                    f"None, got {type(self.transition_noise).__name__}"
                )
            _check_not_family(name, "transition_noise", self.transition_noise)
            if self.transition_noise.shape[0] != state_dim:
                raise ValueError(
                    f"{name}: {self.transition_noise!r} has side "
                    f"{self.transition_noise.shape[0]}, but the state has dimension "
                    f"{state_dim}"
                )

    @property
    def state_dim(self) -> int:
        """The state dimension :math:`d`."""
        return int(self.truth.shape[1])

    @property
    def data_dim(self) -> int:
        """The number of observed values per time, :math:`N`."""
        return int(self.observations.shape[1])

    @property
    def n_times(self) -> int:
        """The number of observation times :math:`T`."""
        return int(self.observations.shape[0])

    def exact_filter(self) -> tuple[tuple[Gaussian, ...], Array]:
        r"""The exact filtering distributions and one-step log evidences.

        The Kalman recursion, with :math:`m_0, P_0` the initial mean and
        covariance:

        .. math::

            \begin{aligned}
            m_t^- &= A m_{t-1}, & P_t^- &= A P_{t-1} A^\top + Q, \\
            S_t &= H P_t^- H^\top + R, & K_t &= P_t^- H^\top S_t^{-1}, \\
            m_t &= m_t^- + K_t (y_t - H m_t^-), & P_t &= P_t^- - K_t S_t K_t^\top,
            \end{aligned}

        and the log evidence of time :math:`t` is
        :math:`\log \mathcal N(y_t;\ H m_t^-,\ S_t)`, whose sum over
        :math:`t` is :math:`\log p(y_{1:T})`.

        Computed with the operations of the layers below, one time at a
        time::

            belief = belief.pipe(maps.pushforward, self.transition,
                                 inputs="x", output="x")
            belief = belief.pipe(maps.pushforward,
                                 maps.AdditiveNoise(self.transition_noise),
                                 inputs="x", output="x")      # when Q is given
            joint = belief.pipe(maps.pushforward, self.observe,
                                inputs="x", output="y").add_noise(y=self.noise_cov)
            log_evidence = joint.log_density(y=y)
            belief = joint.condition(y=y).compress()

        Returns
        -------
        distributions : tuple of Gaussian
            :math:`\mathcal N(m_t, P_t)` over the block ``"x"``, for
            :math:`t = 1, \dots, T`.
        log_evidence : Array
            ``(T,)``.

        Notes
        -----
        ``compress`` after each conditioning keeps the latent width at most
        ``d``; without it, each absorbed :math:`Q` would widen the factor by
        its own width at every time.
        """
        belief = self.initial
        distributions, log_evidence = [], []
        for y in self.observations:
            belief = belief.pipe(
                maps.pushforward, self.transition, inputs="x", output="x"
            )
            if self.transition_noise is not None:
                belief = belief.pipe(
                    maps.pushforward,
                    maps.AdditiveNoise(self.transition_noise),
                    inputs="x",
                    output="x",
                )
            joint = belief.pipe(
                maps.pushforward, self.observe, inputs="x", output="y"
            ).add_noise(y=self.noise_cov)
            log_evidence.append(joint.log_density(y=y))
            belief = joint.condition(y=y).compress()
            distributions.append(belief)
        return tuple(distributions), jnp.stack(log_evidence)

    def __repr__(self) -> str:
        """As ``LinearStateSpace(state_dim=3, data_dim=2, n_times=20)``; never raises."""
        try:
            return (
                f"LinearStateSpace(state_dim={self.state_dim}, "
                f"data_dim={self.data_dim}, n_times={self.n_times})"
            )
        except Exception:
            return "<LinearStateSpace (unprintable fields)>"


# ---------------------------------------------------------------------------
# private: shared validation and per-particle models
# ---------------------------------------------------------------------------


def _check_array_field(cls_name: str, field_name: str, value, shape: tuple) -> None:
    """Require an array of exactly ``shape``, finite in debug mode.

    A field that cannot be inspected is rejected rather than waved through:
    a Python list passes a shape check and then fails inside the model, with
    an error naming a tracer rather than the field.
    """
    if getattr(value, "shape", None) is None:
        raise TypeError(
            f"{cls_name}.{field_name}: expected an array of shape {shape}, got "
            f"{type(value).__name__}, which has no shape to check. Pass a JAX "
            f"array — jnp.asarray() on a nested list."
        )
    if tuple(value.shape) != shape:
        raise ValueError(
            f"{cls_name}.{field_name}: expected an array of shape {shape}, got "
            f"shape {tuple(value.shape)}"
        )
    value_check(
        value,
        lambda arr: bool(jnp.all(jnp.isfinite(arr))),
        f"{cls_name}.{field_name}: must be finite",
    )


def _check_not_family(cls_name: str, field_name: str, value) -> None:
    """Reject a vmapped family, naming the object being built.

    A family field is accepted by every size check here and diagnosed much
    later, by the operator, in a message about the operator rather than about
    the problem.
    """
    if value.batch_shape != ():
        raise ValueError(
            f"{cls_name}.{field_name}: {value!r} is a vmapped family with batch "
            f"shape {value.batch_shape}; build a family of problems with "
            f"jax.vmap over a function that constructs one, not from a family "
            f"field."
        )


def _check_problem(
    cls_name: str,
    *,
    parameter_dim: int,
    data_dim: int,
    prior: Gaussian,
    noise_cov: PSDLinOp,
    y: Array,
    u_true: Array,
) -> None:
    """Validate the four fields every problem carries against its sizes."""
    if not isinstance(prior, Gaussian):
        raise TypeError(
            f"{cls_name}.prior: must be an enskit.distribution.Gaussian, got "
            f"{type(prior).__name__}. Build one as "
            f"Gaussian({{'u': mean}}, block_covs={{'u': PSDDiagonal(variances)}})."
        )
    if not isinstance(noise_cov, PSDLinOp):
        raise TypeError(
            f"{cls_name}.noise_cov: must be an enskit.linalg.PSDLinOp, got "
            f"{type(noise_cov).__name__}. Wrap a dense matrix as "
            f"enskit.linalg.DensePSD(matrix)."
        )
    _check_not_family(cls_name, "prior", prior)
    _check_not_family(cls_name, "noise_cov", noise_cov)
    if prior.names != ("u",):
        raise ValueError(
            f"{cls_name}: the prior must have exactly one block, 'u', got blocks "
            f"{prior.names}"
        )
    if prior.dims["u"] != parameter_dim:
        raise ValueError(
            f"{cls_name}: the prior has dimension {prior.dims['u']}, but the model "
            f"takes {parameter_dim} parameters"
        )
    if noise_cov.shape[0] != data_dim:
        raise ValueError(
            f"{cls_name}: {noise_cov!r} has side {noise_cov.shape[0]}, but the "
            f"model returns {data_dim} predictions"
        )
    _check_array_field(cls_name, "y", y, (data_dim,))
    _check_array_field(cls_name, "u_true", u_true, (parameter_dim,))


def _check_dim(where: str, name: str, value) -> None:
    """Require a positive Python ``int``, excluding ``bool``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{where}: {name} must be an int, got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"{where}: {name} must be at least 1, got {value}")


def _check_scale(where: str, name: str, value) -> None:
    """Require a positive finite scalar: ``nan`` and ``inf`` both fail.

    ``not (value > 0)`` alone catches ``nan`` and lets ``inf`` through, which
    would build a problem whose every array is non-finite.
    """
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{where}: {name} must be a real scalar, got "
            f"{type(value).__name__}"
        ) from exc
    if not (scale > 0.0) or not math.isfinite(scale):
        raise ValueError(
            f"{where}: {name} must be positive and finite, got {value}"
        )


def _check_ensemble(
    cls_name: str, ensemble, parameter_dim: int, *, method: str = "forward"
):
    """Require exactly ``(J, parameter_dim)``: every particle's parameters, one per row.

    The generalized-ufunc convention would carry any leading batch rank
    through, so a single parameter vector returns a plausible ``(N,)`` and a
    stack of ensembles a plausible ``(B, J, N)``. Neither is what a run passes
    — a run binds one ensemble — and the first is the mistake the forward-model
    guide calls the most common one, so the models say so rather than
    answering. ``jax.vmap`` over the method still works: it passes each slice
    as a two-dimensional ensemble.
    """
    shape = getattr(ensemble, "shape", None)
    if shape is None:
        raise TypeError(
            f"{cls_name}.{method}: expected a (J, {parameter_dim}) array, got "
            f"{type(ensemble).__name__}, which has no shape"
        )
    if len(shape) != 2 or shape[1] != parameter_dim:
        raise ValueError(
            f"{cls_name}.{method}: expected a (J, {parameter_dim}) ensemble, got shape "
            f"{tuple(shape)}. The model is called once with the whole "
            f"ensemble, one particle per row — not with a single parameter "
            f"vector, and never with a further leading axis."
        )
    return ensemble


def _decay(particle: Array, times: Array) -> Array:
    """One particle of the decay model: ``(2,) -> (N,)``.

    .. math::

        v_i = u_0 \\, e^{-u_1 t_i} .
    """
    return particle[0] * jnp.exp(-particle[1] * times)


def _restricted_decay(particle: Array, times: Array, rate_floor: float) -> Array:
    """One particle of the decay model, outside its domain returning ``nan``.

    The rate is clamped inside the valid branch as well as selected outside
    it. ``jnp.where`` evaluates both branches, and for a very negative rate
    the discarded one overflows to ``inf``; the derivative then multiplies
    that by a zero cotangent and returns ``nan``. Clamping first keeps the
    discarded branch finite, so the model differentiates.
    """
    valid = particle[1] > rate_floor
    safe = jnp.where(valid, particle[1], rate_floor + 1.0)
    prediction = particle[0] * jnp.exp(-safe * times)
    return jnp.where(valid, prediction, jnp.nan)


def _check_times(cls_name: str, times) -> None:
    """Require a rank-1 array of at least one point, finite in debug mode.

    Checked through the object's own ``shape`` rather than through
    :func:`jax.numpy.shape`, which accepts anything shape-like — a Python
    list among them — and is deprecated for non-arrays.
    """
    shape = getattr(times, "shape", None)
    if shape is None:
        raise TypeError(
            f"{cls_name}.times: expected a rank-1 array, got "
            f"{type(times).__name__}, which has no shape to check. Pass a JAX "
            f"array — jnp.asarray() on a nested list."
        )
    if len(shape) != 1 or shape[0] < 1:
        raise ValueError(
            f"{cls_name}.times: must be a rank-1 array of at least one point, "
            f"got shape {tuple(shape)}"
        )
    value_check(
        times,
        lambda arr: bool(jnp.all(jnp.isfinite(arr))),
        f"{cls_name}.times: must be finite",
    )


def _decay_problem(
    *,
    n_times: int,
    t_max: float,
    noise_std: float,
    seed: int,
    where: str,
) -> tuple[Array, Gaussian, PSDLinOp, Array, Array]:
    """The pieces both decay factories share, so the two problems agree."""
    _check_dim(where, "n_times", n_times)
    _check_scale(where, "t_max", t_max)
    _check_scale(where, "noise_std", noise_std)
    times = jnp.linspace(t_max / n_times, t_max, n_times)
    u_true = jnp.array([2.0, 1.5])
    prior = Gaussian(
        {"u": jnp.array([1.0, 1.0])}, block_covs={"u": PSDDiagonal(jnp.array([1.0, 1.0]))}
    )
    noise_cov = PSDDiagonal(jnp.full(n_times, float(noise_std) ** 2))
    error = noise_std * jax.random.normal(jax.random.key(seed), (n_times,))
    return times, prior, noise_cov, _decay(u_true, times) + error, u_true


def _check_finite(where: str, name: str, value) -> float:
    """Require a finite real scalar, returned as a Python float."""
    if isinstance(value, bool):
        raise TypeError(f"{where}: {name} must be a real scalar, got the bool {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{where}: {name} must be a real scalar, got {type(value).__name__}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"{where}: {name} must be finite, got {value}")
    return number


def _check_linear_map(cls_name: str, field_name: str, value, d_out: int, d_in: int):
    """Require a single-input :class:`~enskit.maps.Linear` of shape ``(d_out, d_in)``."""
    if not isinstance(value, maps.Linear) or value.input_names is not None:
        raise TypeError(
            f"{cls_name}.{field_name}: must be an enskit.maps.Linear of one operator, "
            f"got {type(value).__name__}. Wrap a matrix as "
            f"maps.Linear(enskit.linalg.Dense(matrix))."
        )
    shape = tuple(value.op.shape)
    if shape != (d_out, d_in):
        raise ValueError(
            f"{cls_name}.{field_name}: the operator has shape {shape}, expected "
            f"({d_out}, {d_in})"
        )


def _check_state_space(
    cls_name: str, *, initial, observe, noise_cov, observations, truth
) -> tuple[int, int, int]:
    """Validate the fields both state-space problems carry; return ``(d, N, T)``."""
    if not isinstance(initial, Gaussian):
        raise TypeError(
            f"{cls_name}.initial: must be an enskit.distribution.Gaussian, got "
            f"{type(initial).__name__}"
        )
    if not isinstance(noise_cov, PSDLinOp):
        raise TypeError(
            f"{cls_name}.noise_cov: must be an enskit.linalg.PSDLinOp, got "
            f"{type(noise_cov).__name__}"
        )
    _check_not_family(cls_name, "initial", initial)
    _check_not_family(cls_name, "noise_cov", noise_cov)
    if initial.names != ("x",):
        raise ValueError(
            f"{cls_name}: the initial distribution must have exactly one block, 'x', "
            f"got blocks {initial.names}"
        )
    state_dim = initial.dims["x"]
    data_dim = noise_cov.shape[0]
    _check_linear_map(cls_name, "observe", observe, data_dim, state_dim)
    shape = getattr(observations, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] < 1:
        raise ValueError(
            f"{cls_name}.observations: expected a (T, {data_dim}) array with T >= 1, "
            f"got {shape if shape is not None else type(observations).__name__}"
        )
    n_times = int(shape[0])
    _check_array_field(cls_name, "observations", observations, (n_times, data_dim))
    _check_array_field(cls_name, "truth", truth, (n_times, state_dim))
    return state_dim, data_dim, n_times


def _toy_keys(seed: int, n: int) -> tuple:
    """``n`` keys for a state-space problem, from a stream callers do not reach."""
    stream = jax.random.fold_in(jax.random.key(seed), _TOY_STREAM)
    return tuple(jax.random.split(stream, n))


def _lorenz96_tendency(x: Array, forcing: Array) -> Array:
    """``f(x)`` of the Lorenz-96 equations, on the trailing axis."""
    return (
        (jnp.roll(x, -1, axis=-1) - jnp.roll(x, 2, axis=-1)) * jnp.roll(x, 1, axis=-1)
        - x
        + forcing
    )


@jax.jit
def _lorenz96_step(x: Array, forcing: Array, dt: Array) -> Array:
    """One fourth-order Runge-Kutta step, compiled once per shape."""
    k1 = _lorenz96_tendency(x, forcing)
    k2 = _lorenz96_tendency(x + 0.5 * dt * k1, forcing)
    k3 = _lorenz96_tendency(x + 0.5 * dt * k2, forcing)
    k4 = _lorenz96_tendency(x + dt * k3, forcing)
    return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _lorenz96_trajectory(
    start: Array, forcing: float, dt: float, n_times: int
) -> tuple[Array, Array]:
    """Spin up onto the attractor, then record ``n_times`` steps.

    Returns the state after the spin-up, :math:`x_0^\\dagger`, and the
    ``(T, d)`` states after each further step.
    """

    def step(x, _):
        x = _lorenz96_step(x, jnp.asarray(forcing), jnp.asarray(dt))
        return x, x

    x0, _ = jax.lax.scan(step, start, None, length=_LORENZ96_SPIN_UP)
    _, truth = jax.lax.scan(step, x0, None, length=n_times)
    return x0, truth


def _linear_trajectory(A: Array, x0: Array, shocks: Array) -> Array:
    """``x_t = A x_{t-1} + w_t`` for each row ``w_t`` of ``shocks``; ``(T, d)``."""

    def step(x, w):
        x = A @ x + w
        return x, x

    _, truth = jax.lax.scan(step, x0, shocks)
    return truth
