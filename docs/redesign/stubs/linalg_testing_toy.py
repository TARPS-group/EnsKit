# ---- enskit.linalg: additions and changes (everything else is kept as is) ----

class Zero(LinOp):
    """The ``(n_out, n_in)`` zero operator; stores no array.

    Used by :meth:`enskit.distribution.Gaussian.absorb` to pad factor rows when the
    latent space grows, so padding never allocates. ``matvec``, ``rmatvec``,
    ``matmat`` and ``to_dense`` return zeros of the right shape; composites
    containing it (``hstack``, ``product``) skip it where they can.

    Parameters
    ----------
    n_out, n_in : int
        Static sizes, each at least 0.
    dtype : dtype
        Keyword-only. The dtype of the zeros it returns.
    """


class LowRankUpdate(PSDLinOp):
    r"""A PSD operator plus a low-rank term: :math:`C = D + F F^\top`.

    The covariance of a block of a :class:`~enskit.distribution.Gaussian` that
    has both a factor row and an independent term. With :math:`W_D` a whitener
    of :math:`D` and :math:`S = (W_D F)^\top = U \Sigma V^\top`:

    .. math::

        W_C &= \big(I - V \operatorname{diag}(1 - (1+\sigma_i^2)^{-1/2}) V^\top\big) W_D, \\
        C^{-1} &= W_D^\top \big(I - V \operatorname{diag}\big(\tfrac{\sigma_i^2}{1+\sigma_i^2}\big) V^\top\big) W_D, \\
        \log\det C &= \log\det D + \textstyle\sum_i \log(1 + \sigma_i^2),

    and ``factor`` is :math:`[L_D,\ F]` when :math:`D = L_D L_D^\top` has one.

    Built on :class:`IdentityPlusGram` of :math:`S`, computed once at
    construction (stored, never cached lazily). ``solve`` and ``logdet`` use
    its methods, so their derivatives are its custom ones and finite at
    degenerate spectra; ``whiten``'s
    derivative is that of the SVD, finite away from exactly repeated or zero
    singular values.

    Parameters
    ----------
    base : PSDLinOp
        ``D``; must support ``whiten``.
    factor : LinOp
        ``F``, of shape ``(n, k)``.
    """


class Kronecker(LinOp):
    r"""The Kronecker product :math:`A \otimes B` of two operators (issue #15).

    .. math::

        (A \otimes B)\, \operatorname{vec}(X) = \operatorname{vec}(A X B^\top),
        \qquad X \in \mathbb R^{n_A \times n_B} \text{ (row-major vec).}


    ``matvec`` reshapes the operand to ``(..., n_A, n_B)`` and applies
    ``A`` and ``B`` along the two axes, never forming the product. The first
    factor's index is the slow one, matching ``np.kron``. Square, PSD and
    rectangular variants are selected by the operands' levels, as the scaled
    composites are: for PSD operands ``factor``, ``whiten``, ``solve`` and
    ``logdet`` are the Kronecker products (or sums, for ``logdet``) of the
    operands'.
    """

# Renamed: SquareLinOp.n -> SquareLinOp.dim (issue #20).


# ---- enskit.testing: conformance checks for user-written pieces ----

def check_operator(op, key=None) -> None:
    """The operator conformance suite (moved here from linalg.testing)."""


def check_simulator(f, ensemble: Ensemble, *, inputs, n_trials: int = 2) -> None:
    """Check a simulator from outside: shape at two ensemble sizes, dtype,
    determinism, and row independence (a bit-exact permutation and a subset
    re-evaluation to a tolerance; two comparisons, because a symmetric
    coupling survives a permutation)."""


def check_update_rule(rule, *, key=None) -> None:
    """Check a user :class:`~enskit.kalman.UpdateRule`: ``build`` returns a
    callable whose output has the right blocks, particle count and dtype;
    finiteness; that the build does not depend on the given values (two calls
    with different values agree with two fresh builds); exactness of the mean
    on an exact-moment linear-Gaussian fixture (and of the covariance for
    deterministic rules, or in expectation for stochastic ones); vmap and jit
    compatibility."""


def check_conditional_map(cmap, joint: Gaussian, given) -> None:
    """Check a :class:`~enskit.distribution.ConditionalMap` against
    :meth:`~enskit.distribution.Gaussian.condition` on a Gaussian joint (Matheron)."""


def check_schedule(schedule, evaluation=None) -> None:
    """Check a schedule: returns a finite, strictly positive 0-d increment or
    ``None``; is pure (two calls agree bitwise); respects its own
    ``beta_target``; propagates ``nan`` misfits as ``nan``, never as a step."""


def check_inflation(inflation, key=None, ensemble=None) -> None:
    """Check an inflation or relaxation policy: same blocks, particles, shapes and
    dtype; deterministic for a fixed key; accepts and ignores unknown context
    keywords."""


def check_stopping_rule(stop, evaluation=None) -> None:
    """Check a stopping rule: returns a Python ``bool`` and is pure."""


# ---- enskit.toy: small problems for tests and documentation ----

def linear_gaussian(parameter_dim: int = 4, data_dim: int = 6, *, seed: int = 0,
                    noise_sd: float = 0.3) -> InverseProblem:
    """A linear forward model from ``parameter_dim`` parameters (block ``"u"``) to
    ``data_dim`` data; ``posterior(beta)`` is closed form."""


def exponential_decay(*, seed: int = 0, n_times: int = 10,
                      noise_sd: float = 0.05) -> InverseProblem:
    """Two parameters ``(amplitude, rate)`` of ``a exp(-r t)`` at ``n_times``
    times; mildly nonlinear."""


def restricted_decay(*, seed: int = 0) -> InverseProblem:
    """The same model with a valid domain: a particle with a non-positive rate
    returns a non-finite row, to exercise failure handling."""


def lorenz96(dim: int = 40, n_steps: int = 300, *, obs_every: int = 2,
             noise_sd: float = 1.0, forcing: float = 8.0, seed: int = 0
             ) -> StateSpaceProblem:
    """The Lorenz-96 system on a ring, observed at every ``obs_every``-th site;
    carries site coordinates for localization."""


def linear_state_space(dim: int = 3, obs_dim: int = 2, n_steps: int = 20, *,
                       seed: int = 0) -> StateSpaceProblem:
    """Linear dynamics and observations; the Kalman filter is exact."""
