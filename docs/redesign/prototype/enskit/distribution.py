"""Prototype distribution layer: Gaussian and Ensemble over named blocks.

Everything here is deliberately compact; the design document specifies the
real validation and docstrings.
"""
from __future__ import annotations

import math
from typing import Mapping

import jax
import jax.numpy as jnp
from jax import Array

from .linalg import Dense, IdentityPlusGram, LinOp, PSDLinOp, PSDLowRank
from pyeki.linalg import Product

# ---------------------------------------------------------------------------
# pytree helper
# ---------------------------------------------------------------------------


def _blocks(where, mapping, kwargs):
    """Merge a positional mapping of block values with keyword block values."""
    out = dict(mapping or {})
    twice = set(out) & set(kwargs)
    if twice:
        raise TypeError(f"{where}: block(s) {sorted(twice)} given twice")
    out.update(kwargs)
    return out


def _pytree(data: tuple[str, ...], meta: tuple[str, ...]):
    def deco(cls):
        def flatten(o):
            return [getattr(o, n) for n in data], tuple(getattr(o, n) for n in meta)

        def unflatten(aux, children):
            o = object.__new__(cls)
            for n, v in zip(data, children):
                object.__setattr__(o, n, v)
            for n, v in zip(meta, aux):
                object.__setattr__(o, n, v)
            return o

        jax.tree_util.register_pytree_node(cls, flatten, unflatten)
        return cls

    return deco


# ---------------------------------------------------------------------------
# Gaussian over named blocks
# ---------------------------------------------------------------------------


@_pytree(data=("means", "factors", "block_covs"), meta=("names", "_latent"))
class Gaussian:
    """mean + F xi + independent per-block terms; cov = F F^T + blockdiag(D_b)."""

    def __init__(self, means: Mapping, factors: Mapping | None = None,
                 block_covs: Mapping | None = None, *, latent_dim: int | None = None):
        factors = dict(factors or {})
        block_covs = dict(block_covs or {})
        names = tuple(means)
        unknown = (set(factors) | set(block_covs)) - set(names)
        if unknown:
            raise ValueError(f"Gaussian: factors/block_covs for unknown blocks {unknown}")
        widths = {f.shape[1] for f in factors.values() if f is not None}
        if len(widths) > 1:
            raise ValueError(f"Gaussian: factors disagree on latent width {widths}")
        if latent_dim is None:
            latent_dim = widths.pop() if widths else 0
        elif widths and widths != {latent_dim}:
            raise ValueError(f"Gaussian: factors have width {widths}, latent_dim {latent_dim}")
        object.__setattr__(self, "_latent", latent_dim)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "means", tuple(jnp.asarray(means[n]) for n in names))
        object.__setattr__(self, "factors", tuple(_as_op(factors.get(n)) for n in names))
        object.__setattr__(self, "block_covs", tuple(block_covs.get(n) for n in names))
        for n, m, f, d in zip(names, self.means, self.factors, self.block_covs):
            if m.ndim != 1:
                raise ValueError(f"Gaussian: mean of block {n!r} must be 1-D, got {m.shape}")
            if f is not None and f.shape[0] != m.shape[0]:
                raise ValueError(f"Gaussian: factor of {n!r} has {f.shape[0]} rows, mean {m.shape[0]}")
            if d is not None and d.shape[0] != m.shape[0]:
                raise ValueError(f"Gaussian: block_cov of {n!r} has side {d.shape[0]}")

    # -- constructors -----------------------------------------------------------
    @classmethod
    def independent(cls, blocks: Mapping[str, tuple] | None = None, /, **kw) -> Gaussian:
        """Independent blocks, each given as (mean, cov)."""
        blocks = _blocks("Gaussian.independent", blocks, kw)
        return cls({n: m for n, (m, _) in blocks.items()},
                   block_covs={n: c for n, (_, c) in blocks.items()})

    # -- introspection ------------------------------------------------------------
    def _i(self, name):
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"no block {name!r}; blocks are {self.names}") from None

    @property
    def dims(self) -> dict[str, int]:
        return {n: m.shape[0] for n, m in zip(self.names, self.means)}

    @property
    def latent_dim(self) -> int:
        return self._latent

    def mean(self, name):
        return self.means[self._i(name)]

    def factor(self, name):
        """The block's rows of the shared factor (a LinOp), or None."""
        return self.factors[self._i(name)]

    def block_cov(self, name):
        """The block's independent covariance term (a PSDLinOp), or None."""
        return self.block_covs[self._i(name)]

    def _dense_factor(self, name):
        i = self._i(name)
        f = self.factors[i]
        d = self.means[i].shape[0]
        return jnp.zeros((d, self.latent_dim)) if f is None else f.to_dense()

    def cov(self, a, b=None) -> LinOp:
        """Covariance block as an operator (prototype: dense)."""
        b = a if b is None else b
        C = self._dense_factor(a) @ self._dense_factor(b).T
        if a == b and self.block_covs[self._i(a)] is not None:
            C = C + self.block_covs[self._i(a)].to_dense()
        return Dense(C)

    def __repr__(self):
        return (f"{type(self).__name__}(blocks={self.dims}, latent_dim={self.latent_dim}, "
                f"block_covs={[n for n, d in zip(self.names, self.block_covs) if d is not None]})")

    def _like(self, means, factors, covs):
        """A Gaussian of this one's kind on the same latent space."""
        return Gaussian(means, factors, covs, latent_dim=self.latent_dim)

    def _parts(self):
        return ({n: m for n, m in zip(self.names, self.means)},
                {n: f for n, f in zip(self.names, self.factors) if f is not None},
                {n: d for n, d in zip(self.names, self.block_covs) if d is not None})


    def pipe(self, f, *args, **kwargs):
        """``f(self, *args, **kwargs)``: lets operations from other modules chain."""
        return f(self, *args, **kwargs)

    # -- structural operations --------------------------------------------------------
    def marginal(self, *names) -> Gaussian:
        for n in names:
            self._i(n)
        means, factors, covs = self._parts()
        keep = lambda d: {n: d[n] for n in names if n in d}  # noqa: E731
        return self._like(keep(means), keep(factors), keep(covs))

    def drop(self, *names) -> Gaussian:
        return self.marginal(*[n for n in self.names if n not in names])

    def with_block(self, name, mean, factor=None, block_cov=None) -> Gaussian:
        means, factors, covs = self._parts()
        means[name] = mean
        factors.pop(name, None)
        covs.pop(name, None)
        if factor is not None:
            factors[name] = factor
        if block_cov is not None:
            covs[name] = block_cov
        return self._like(means, factors, covs)

    def absorb(self, *names) -> Gaussian:
        """Move the independent terms of ``names`` into the shared factor."""
        names = [n for n in names if self.block_covs[self._i(n)] is not None]
        if not names:
            return self
        means, factors, covs = self._parts()
        k = self.latent_dim
        new = {n: [self._dense_factor(n)] if (n in factors or n in names) else None
               for n in self.names}
        for a in names:
            L = covs.pop(a).factor().to_dense()
            for n in self.names:
                if new[n] is None:
                    continue
                new[n].append(L if n == a else jnp.zeros((self.dims[n], L.shape[1])))
        factors = {n: Dense(jnp.concatenate(cols, axis=1)) for n, cols in new.items()
                   if cols is not None}
        del k
        return Gaussian(means, factors, covs)

    def add_noise(self, covs: Mapping[str, PSDLinOp] | None = None, /, **kw) -> Gaussian:
        """Block b becomes b + e_b, e_b ~ N(0, covs[b]) independent of everything."""
        covs = _blocks("add_noise", covs, kw)
        g = self.absorb(*[n for n in covs if self.block_covs[self._i(n)] is not None])
        means, factors, old = g._parts()
        old.update(covs)
        return g._like(means, factors, old)

    # -- conditioning -------------------------------------------------------------------
    def _whitened_factor(self, given):
        rows = []
        for c in given:
            D = self.block_covs[self._i(c)]
            if D is None:
                raise ValueError(
                    f"cannot condition on block {c!r}: it has no independent covariance "
                    f"term to whiten (noise-free conditioning is not supported)")
            rows.append(D.whiten_mat(self._dense_factor(c)))
        return jnp.concatenate(rows, axis=0).T  # (k, N)

    def _whitened_residual(self, values, at=None):
        parts = []
        for c, y in values.items():
            i = self._i(c)
            base = self.means[i] if at is None else at[c]
            parts.append(self.block_covs[i].whiten(jnp.asarray(y) - base))
        return jnp.concatenate(parts, axis=-1)

    def _noise_kind(self, given):
        has = [self.block_covs[self._i(c)] is not None for c in given]
        if all(has):
            return "whitened"
        if not any(has):
            return "noise_free"
        raise ValueError("cannot condition on a mix of blocks with and without "
                         "independent covariance terms")

    def _noise_free_qr(self, given):
        """QR of the stacked conditioned factor's transpose: F_y^T = Q R."""
        Fy = jnp.concatenate([self._dense_factor(c) for c in given], axis=0)  # (N, k)
        N, k = Fy.shape
        rank_bound = k - 1 if isinstance(self, EnsembleGaussian) else k   # centered columns
        if rank_bound < N:
            raise ValueError(f"noise-free conditioning on {N} coordinates needs a factor of "
                             f"rank >= {N}; this one has rank at most {rank_bound}")
        Q, Rt = jnp.linalg.qr(Fy.T)  # (k, N), (N, N)
        d = jnp.abs(jnp.diag(Rt))
        if not isinstance(d, jax.core.Tracer) and bool(jnp.min(d) <= 1e-12 * jnp.max(d)):
            raise ValueError("noise-free conditioning: the conditioned blocks' "
                             "covariance is singular")
        return Q, Rt

    def compress(self) -> Gaussian:
        """Re-factor the shared factor to width at most the total dimension."""
        D = sum(self.dims.values())
        if self.latent_dim <= D:
            return self
        F = jnp.concatenate([self._dense_factor(n) for n in self.names], axis=0)
        _, R = jnp.linalg.qr(F.T)          # F F^T = R^T R
        G = R.T
        means, _, covs = self._parts()
        factors, start = {}, 0
        for n in self.names:
            d = self.dims[n]
            if self.factors[self._i(n)] is not None:
                factors[n] = Dense(G[start:start + d])
            start += d
        return Gaussian(means, factors, covs)

    def condition(self, values: Mapping[str, Array] | None = None, /, **kw) -> Gaussian:
        values = _blocks("condition", values, kw)
        given = tuple(values)
        targets = [n for n in self.names if n not in given]
        if self.latent_dim == 0:
            return self.marginal(*targets)
        if self._noise_kind(given) == "noise_free":
            Q, Rt = self._noise_free_qr(given)
            r = jnp.concatenate([jnp.asarray(values[c]) - self.mean(c) for c in given])
            w = Q @ jax.scipy.linalg.solve_triangular(Rt.T, r, lower=True)
            P = jnp.eye(self.latent_dim) - Q @ Q.T
            means, factors, covs = self.marginal(*targets)._parts()
            for n in targets:
                f = self.factors[self._i(n)]
                if f is not None:
                    means[n] = means[n] + f.matvec(w)
                    factors[n] = Dense(f.matmat(P))
            return self._like(means, factors, covs)
        S = self._whitened_factor(given)
        G = IdentityPlusGram(S)
        w = G.solve_factor(self._whitened_residual(values))
        T = G.inverse_sqrt().to_dense()
        means, factors, covs = self.marginal(*targets)._parts()
        for n in targets:
            f = self.factors[self._i(n)]
            if f is not None:
                means[n] = means[n] + f.matvec(w)
                factors[n] = Dense(f.matmat(T))
        return self._like(means, factors, covs)

    def conditional_map(self, given) -> MatheronMap:
        given = (given,) if isinstance(given, str) else tuple(given)
        targets = tuple(n for n in self.names if n not in given)
        return MatheronMap(
            given=given, targets=targets, S=self._whitened_factor(given),
            target_factors=tuple(self.factors[self._i(n)] for n in targets),
            whiteners=tuple(self.block_covs[self._i(c)] for c in given))

    def log_density(self, values: Mapping[str, Array] | None = None, /, **kw) -> Array:
        """Log density of the marginal over exactly the blocks in ``values``."""
        values = _blocks("log_density", values, kw)
        given = tuple(values)
        if self._noise_kind(given) == "noise_free":
            Q, Rt = self._noise_free_qr(given)
            r = jnp.concatenate([jnp.asarray(values[c]) - self.mean(c) for c in given], axis=-1)
            z = jax.scipy.linalg.solve_triangular(Rt.T, r.T, lower=True).T
            n = r.shape[-1]
            logdet = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(Rt))))
            return -0.5 * (jnp.sum(z * z, axis=-1) + logdet + n * math.log(2 * math.pi))
        b = self._whitened_residual(values)
        n = b.shape[-1]
        logdet = sum(self.block_covs[self._i(c)].logdet() for c in given)
        quad = jnp.sum(b * b, axis=-1)
        if self.latent_dim:
            S = self._whitened_factor(given)
            G = IdentityPlusGram(S)
            w = G.solve_factor(b)
            quad = quad - jnp.sum(jnp.einsum("kn,...n->...k", S, b) * w, axis=-1)
            logdet = logdet + G.logdet()
        return -0.5 * (quad + logdet + n * math.log(2 * math.pi))

    # -- sampling ---------------------------------------------------------------------
    def sample(self, key, n_particles: int) -> Ensemble:
        k = self.latent_dim
        key, sub = jax.random.split(key)
        xi = jax.random.normal(sub, (n_particles, k)) if k else None
        out = {}
        for name, m, f, d in zip(self.names, self.means, self.factors, self.block_covs):
            x = jnp.broadcast_to(m, (n_particles, m.shape[0]))
            if f is not None:
                x = x + f.matvec(xi)
            if d is not None:
                key, sub = jax.random.split(key)
                L = d.factor()
                x = x + L.matvec(jax.random.normal(sub, (n_particles, L.shape[1])))
            out[name] = x
        return Ensemble(out)


@_pytree(data=("means", "factors", "block_covs"), meta=("names", "_latent", "n_particles"))
class EnsembleGaussian(Gaussian):
    """A Gaussian whose latent coordinates are an ensemble's particles."""

    def __init__(self, means, factors=None, block_covs=None, *, n_particles: int):
        super().__init__(means, factors, block_covs, latent_dim=n_particles)
        object.__setattr__(self, "n_particles", n_particles)

    def _like(self, means, factors, covs):
        return EnsembleGaussian(means, factors, covs, n_particles=self.n_particles)

    def __repr__(self):
        return super().__repr__()[:-1] + f", n_particles={self.n_particles})"

    def realize_particles(self, *, key=None, exclude_block_covs=()) -> Ensemble:
        """The particles this Gaussian is aligned with: x_j = m + sqrt(J-1) F e_j."""
        J = self.n_particles
        out = {}
        for name, m, f, d in zip(self.names, self.means, self.factors, self.block_covs):
            x = jnp.broadcast_to(m, (J, m.shape[0]))
            if f is not None:
                x = x + math.sqrt(J - 1) * f.to_dense().T
            if d is not None and name not in exclude_block_covs:
                if key is None:
                    raise ValueError(f"realize: block {name!r} has an independent "
                                     f"covariance term; pass a key to sample it")
                key, sub = jax.random.split(key)
                L = d.factor()
                x = x + L.matvec(jax.random.normal(sub, (J, L.shape[1])))
            out[name] = x
        return Ensemble(out)

    def square_root_map(self, given) -> SquareRootMap:
        """The square-root update of these particles, as a map from given values."""
        return SquareRootMap(self, given)


class SquareRootMap:
    """Moves the particle set an EnsembleGaussian was built from; values at call time.

    Prototype: composes condition and realize_particles at call time; the design
    builds S and T once at construction.
    """

    def __init__(self, gaussian: EnsembleGaussian, given):
        self.gaussian = gaussian
        self.given = (given,) if isinstance(given, str) else tuple(given)
        self.targets = tuple(n for n in gaussian.names if n not in self.given)

    def __call__(self, values=None, /, *, key=None, **kw) -> Ensemble:
        values = _blocks("square-root map", values, kw)
        if set(values) != set(self.given):
            raise ValueError(f"square-root map: expected values for {self.given}")
        return self.gaussian.condition(values).realize_particles(key=key)


@_pytree(data=("S", "target_factors", "whiteners"), meta=("given", "targets"))
class MatheronMap:
    """x -> x + K (y* - y): the exact conditional of a Gaussian as a transport map."""

    def __init__(self, *, given, targets, S, target_factors, whiteners):
        for k, v in dict(given=given, targets=targets, S=S,
                         target_factors=target_factors, whiteners=whiteners).items():
            object.__setattr__(self, k, v)

    def coefficients(self, samples: Ensemble, values=None, /, *, key=None, **kw) -> Array:
        values = _blocks("conditional map", values, kw)
        b = jnp.concatenate([W.whiten(jnp.asarray(values[c]) - samples[c])
                             for c, W in zip(self.given, self.whiteners)], axis=-1)
        if key is not None:
            # the given blocks' independent terms, drawn in whitened coordinates:
            # W(y* - (g + e)) = W(y* - g) - eps, eps ~ N(0, I); needs only `whiten`
            b = b - jax.random.normal(key, b.shape, b.dtype)
        return IdentityPlusGram(self.S).solve_factor(b)  # (n, k)

    def __call__(self, samples: Ensemble, values=None, /, *, key=None, **kw) -> Ensemble:
        w = self.coefficients(samples, values, key=key, **kw)
        new = {n: samples[n] + f.matvec(w) for n, f in zip(self.targets, self.target_factors)
               if f is not None and n in samples.names}
        return samples.assign(new).drop(*self.given)


def _as_op(f):
    if f is None or isinstance(f, LinOp):
        return f
    return Dense(jnp.asarray(f))


# ---------------------------------------------------------------------------
# Ensemble: (optionally weighted) empirical distribution over named blocks
# ---------------------------------------------------------------------------


@_pytree(data=("blocks", "log_weights"), meta=("names",))
class Ensemble:
    def __init__(self, blocks: Mapping[str, Array] | None = None, /, *, log_weights=None, **kw):
        blocks = _blocks("Ensemble", blocks, kw)
        names = tuple(blocks)
        arrays = tuple(jnp.asarray(blocks[n]) for n in names)
        sizes = {a.shape[0] for a in arrays}
        if len(sizes) != 1 or any(a.ndim != 2 for a in arrays):
            raise ValueError(f"Ensemble: blocks must be (n_particles, dim) with one "
                             f"n_particles; got { {n: a.shape for n, a in zip(names, arrays)} }")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "blocks", arrays)
        object.__setattr__(self, "log_weights",
                           None if log_weights is None else jnp.asarray(log_weights))

    def __getitem__(self, name) -> Array:
        try:
            return self.blocks[self.names.index(name)]
        except ValueError:
            raise KeyError(f"no block {name!r}; blocks are {self.names}") from None

    def __repr__(self):
        return f"Ensemble(n_particles={self.n_particles}, blocks={self.dims}, weighted={self.is_weighted})"

    @property
    def n_particles(self) -> int:
        return self.blocks[0].shape[0]

    @property
    def dims(self):
        return {n: a.shape[1] for n, a in zip(self.names, self.blocks)}

    @property
    def is_weighted(self) -> bool:
        return self.log_weights is not None

    @property
    def weights(self) -> Array:
        if self.log_weights is None:
            return jnp.full(self.n_particles, 1.0 / self.n_particles)
        return jax.nn.softmax(self.log_weights)

    @property
    def all_finite(self) -> Array:
        return jnp.all(jnp.stack([jnp.all(jnp.isfinite(a), axis=1) for a in self.blocks]), axis=0)

    def pipe(self, f, *args, **kwargs):
        """``f(self, *args, **kwargs)``: lets operations from other modules chain."""
        return f(self, *args, **kwargs)

    def as_dict(self):
        return dict(zip(self.names, self.blocks))

    def marginal(self, *names) -> Ensemble:
        return Ensemble({n: self[n] for n in names}, log_weights=self.log_weights)

    def drop(self, *names) -> Ensemble:
        return self.marginal(*[n for n in self.names if n not in names])

    def assign(self, blocks: Mapping[str, Array] | None = None, /, **kw) -> Ensemble:
        d = self.as_dict()
        d.update(_blocks("assign", blocks, kw))
        return Ensemble(d, log_weights=self.log_weights)

    def mean(self, name) -> Array:
        return self.weights @ self[name]

    def anomalies(self, name) -> Array:
        return self[name] - self.mean(name)

    def cov(self, a, b=None) -> LinOp:
        """Sample (cross-)covariance as an operator; never forms a d x d matrix itself."""
        g = self.project()
        if b is None or a == b:
            return PSDLowRank(g._dense_factor(a))
        return Product((g.factors[g._i(a)], g.factors[g._i(b)].T))

    def project(self) -> Gaussian:
        """Moment-matching Gaussian; member-aligned when unweighted."""
        J = self.n_particles
        if not self.is_weighted:
            means = {n: jnp.mean(a, axis=0) for n, a in zip(self.names, self.blocks)}
            factors = {n: Dense((a - means[n]).T / math.sqrt(J - 1))
                       for n, a in zip(self.names, self.blocks)}
            return EnsembleGaussian(means, factors, n_particles=J)
        w = self.weights
        lw = self.log_weights
        lse = jax.scipy.special.logsumexp
        one_minus_sum_w2 = -jnp.expm1(lse(2 * lw) - 2 * lse(lw))   # 1 - sum w^2, stably
        if not isinstance(one_minus_sum_w2, jax.core.Tracer) and not bool(one_minus_sum_w2 > 0):
            raise ValueError("project: the weights are concentrated on one member "
                             "(effective sample size 1 to working precision); the "
                             "weighted covariance is undefined")
        scale = 1.0 / jnp.sqrt(one_minus_sum_w2)
        means = {n: w @ a for n, a in zip(self.names, self.blocks)}
        factors = {n: Dense(((a - means[n]) * jnp.sqrt(w)[:, None]).T * scale)
                   for n, a in zip(self.names, self.blocks)}
        return Gaussian(means, factors)


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


def reweight(ensemble: Ensemble, log_weight_increments: Array) -> Ensemble:
    base = jnp.zeros(ensemble.n_particles) if ensemble.log_weights is None else ensemble.log_weights
    return Ensemble(ensemble.as_dict(), log_weights=base + log_weight_increments)


def effective_sample_size(ensemble: Ensemble) -> Array:
    w = ensemble.weights
    return 1.0 / jnp.sum(w * w)


def resample(key, ensemble: Ensemble, n_particles: int | None = None) -> Ensemble:
    """Systematic resampling; returns an unweighted ensemble."""
    n = ensemble.n_particles if n_particles is None else n_particles
    u = (jax.random.uniform(key) + jnp.arange(n)) / n
    idx = jnp.searchsorted(jnp.cumsum(ensemble.weights), u)
    idx = jnp.minimum(idx, ensemble.n_particles - 1)
    return Ensemble({k: v[idx] for k, v in ensemble.as_dict().items()})


def exact_moment_ensemble(key, gaussian: Gaussian, n_particles: int) -> Ensemble:
    """Members whose sample mean and covariance equal the Gaussian's exactly, jointly.

    Prototype: colours with a dense joint Cholesky of the (small) joint covariance.
    """
    names = gaussian.names
    dims = [gaussian.dims[n] for n in names]
    D = sum(dims)
    if n_particles <= D:
        raise ValueError(f"exact_moment_ensemble: need n_particles > {D}")
    C = jnp.block([[gaussian.cov(a, b).to_dense() for b in names] for a in names])
    m = jnp.concatenate([gaussian.mean(n) for n in names])
    Z = jax.random.normal(key, (n_particles, D))
    Z = Z - Z.mean(0)
    Q, _ = jnp.linalg.qr(Z)                   # orthonormal, centred columns
    X = m + jnp.sqrt(n_particles - 1.0) * Q @ jnp.linalg.cholesky(C).T
    out, start = {}, 0
    for n, d in zip(names, dims):
        out[n] = X[:, start:start + d]
        start += d
    return Ensemble(out)
