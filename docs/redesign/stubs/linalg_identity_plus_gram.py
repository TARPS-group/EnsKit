class IdentityPlusGram(PSDLinOp):
    r"""The operator :math:`I + S S^\top` for a whitened factor :math:`S`.

    .. math::

        A = I_k + S S^\top, \qquad S \in \mathbb R^{k \times N},
        \qquad S = U \Sigma V^\top \ \text{(thin SVD)}.

    Every Gaussian conditioning in the package reduces to this operator, with
    :math:`S = (W F_y)^\top` the given blocks' factor rows whitened by their
    noise. It is built from one thin SVD at construction (stored, never cached
    lazily) and never forms :math:`SS^\top` or :math:`S^\top S`: forming either
    rounds away every :math:`\sigma_i < \sqrt{\varepsilon}\,\sigma_{\max}`.

    Parameters
    ----------
    S : Array
        ``(k, N)``, exactly 2-D.

    Notes
    -----
    Every method below carries a custom derivative rule, finite and correct at
    exactly repeated and exactly zero singular values, where the derivative of
    a plain SVD is ``nan``. Those spectra are routine here: a localization
    mask's zero-padded columns, or a predicted coordinate that is constant
    across particles.

    References
    ----------
    .. [1] Higham, N. J. (2008). Functions of Matrices: Theory and
       Computation. SIAM.
    """

    def solve_factor(self, b: Array) -> Array:
        r"""Apply :math:`A^{-1} S`: the latent coefficients of a Gaussian conditional.

        .. math::

            w = (I + S S^\top)^{-1} S\, b
              = U \operatorname{diag}\!\Big(\frac{\sigma_i}{1+\sigma_i^2}\Big) V^\top b .

        The multipliers are at most 1/2, so the result is bounded however
        collapsed :math:`S` is. ``b`` is ``(..., N)``; the result ``(..., k)``.

        Notes
        -----
        Custom JVP, rational in :math:`S` with no division by singular-value
        gaps:

        .. math::

            dw = M (dS\, b + S\, db) - M (dS\, S^\top + S\, dS^\top)\, w,
            \qquad M = A^{-1} = I + U\big((I+\Sigma^2)^{-1} - I\big)U^\top .
        """

    def inverse_sqrt(self) -> SquareLinOp:
        r"""The symmetric inverse square root, in thin form.

        .. math::

            A^{-1/2} = I + U_r\big((I + \Sigma^2)^{-1/2} - I\big) U_r^\top,
            \qquad r = \min(k, N),

        exact at every rank (the identity completion). Applying it costs
        :math:`O(kr)` and nothing of size ``(k, k)`` is stored.

        Notes
        -----
        Custom JVP by the Daleckii-Krein formula for
        :math:`f(\lambda) = (1+\lambda)^{-1/2}`, in the thin basis with
        :math:`P = I - U_r U_r^\top` and :math:`s_i = \sqrt{1+\sigma_i^2}`:

        .. math::

            d(A^{-1/2}) = U_r \big(G \circ (U_r^\top dA\, U_r)\big) U_r^\top
               + U_r \operatorname{diag}(g)\, U_r^\top dA\, P
               + P\, dA\, U_r \operatorname{diag}(g)\, U_r^\top,

        .. math::

            dA = dS\, S^\top + S\, dS^\top, \qquad
            G_{ij} = \frac{-1}{s_i s_j (s_i + s_j)}, \qquad
            g_i = \frac{-1}{s_i (s_i + 1)} .

        The complement-complement block vanishes because :math:`PS = 0`.
        """

    def logdet(self) -> Array:
        r"""The log-determinant, a 0-d array.

        .. math::

            \log\det(I + S S^\top) = \sum_i \log(1 + \sigma_i^2),
            \qquad d(\log\det A) = 2 \langle A^{-1} S,\ dS \rangle .
        """

    # matvec, solve and whiten follow from the same SVD; factor is
    # [I, S] (k x (k + N)), which is exact but rarely the cheap choice.


def dense_fallback(*, max_n: int = 2048):
    """Context manager: let missing capabilities fall back to dense algebra.

    Off by default. Inside ``with linalg.dense_fallback(max_n=...):`` an
    operation an operator does not support (``solve``, ``whiten``,
    ``factor``, ``logdet``) is computed on :func:`densify`'s result instead
    of raising :class:`UnsupportedOpError`, with one warning per operator
    type and operation. Operators larger than ``max_n`` still raise, before
    allocating. Meant for prototyping on small problems; library code never
    relies on it.
    """
