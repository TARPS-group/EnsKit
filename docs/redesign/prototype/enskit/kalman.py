"""Prototype Kalman layer: update rules build particle updates; values arrive at call time."""
from __future__ import annotations

import math
from typing import Callable, Mapping, Protocol

import jax
import jax.numpy as jnp
from jax import Array

from .distribution import Ensemble, EnsembleGaussian, Gaussian, _blocks, _pytree
from .linalg import IdentityPlusGram, PSDDiagonal, PSDLinOp, PSDScaled


class ParticleUpdate(Protocol):
    """A built update: given values in, updated particles (target blocks) out."""

    def __call__(self, values=None, /, *, key=None, **block_values) -> Ensemble: ...


class UpdateRule(Protocol):
    """Builds a ParticleUpdate from particles, their joint approximation and given names."""

    def build(self, particles: Ensemble, approximation: Gaussian,
              given: tuple[str, ...]) -> ParticleUpdate: ...


def _names(given):
    return (given,) if isinstance(given, str) else tuple(given)


def _aligned(particles, approximation):
    return (isinstance(approximation, EnsembleGaussian)
            and approximation.n_particles == particles.n_particles)


def _target_noise(key, particles, approximation, targets):
    """Draws from the targets' independent terms, one per particle (none if absent)."""
    out = {}
    for n in targets:
        D = approximation.block_cov(n)
        if D is not None:
            if key is None:
                raise ValueError(f"target block {n!r} has an independent term; pass a key")
            key, sub = jax.random.split(key)
            L = D.factor()
            out[n] = L.matvec(jax.random.normal(sub, (particles.n_particles, L.shape[1])))
    return out


# ---------------------------------------------------------------------------
# the two shipped rules
# ---------------------------------------------------------------------------


@_pytree(data=(), meta=())
class SymmetricSquareRoot:
    """Deterministic square-root update: the approximation's SquareRootMap."""

    def build(self, particles, approximation, given):
        if not _aligned(particles, approximation):
            raise ValueError("SymmetricSquareRoot needs an EnsembleGaussian aligned with "
                             "the particles (kalman.gaussian_approximation builds one)")
        return approximation.square_root_map(_names(given))


class _MatheronUpdate:
    """The approximation's MatheronMap applied to the particles it was built for."""

    def __init__(self, particles, approximation, given):
        self.particles, self.approximation = particles, approximation
        self.given = _names(given)
        self.targets = tuple(n for n in approximation.names if n not in self.given)
        self.map = approximation.conditional_map(self.given)  # pointwise; value-free
        self.aligned = _aligned(particles, approximation)

    def __call__(self, values=None, /, *, key=None, **kw) -> Ensemble:
        values = _blocks("Matheron update", values, kw)
        if key is None:
            raise ValueError("Matheron needs a key")
        k_targets, k_noise = jax.random.split(key)
        P, g = self.particles, self.approximation
        extra = _target_noise(k_targets, P, g, self.targets)
        if self.aligned:
            # W(y* - g_j) = W(y* - g_bar) - sqrt(J-1) S[j]: J + 1 whitenings in all
            J = P.n_particles
            S = self.map.S
            b = (g._whitened_residual(values) - math.sqrt(J - 1) * S
                 - jax.random.normal(k_noise, (J, S.shape[1]), S.dtype))
            w = IdentityPlusGram(S).solve_factor(b)
            out = {}
            for n in self.targets:
                f = g.factor(n)
                x = P[n] + extra.get(n, 0.0)
                out[n] = x if f is None else x + f.matvec(w)
            return Ensemble(out)
        moved = P.assign({n: P[n] + v for n, v in extra.items()})
        return self.map(moved, values, key=k_noise).marginal(*self.targets)


@_pytree(data=(), meta=())
class Matheron:
    """Stochastic update: transport the particles through the approximation's MatheronMap."""

    def build(self, particles, approximation, given):
        return _MatheronUpdate(particles, approximation, given)


# ---------------------------------------------------------------------------
# the one-call form
# ---------------------------------------------------------------------------


def gaussian_approximation(ensemble: Ensemble,
                           noise: Mapping[str, PSDLinOp] | None = None) -> Gaussian:
    """The joint Gaussian approximation an update conditions: moment match plus known noise."""
    g = ensemble.project()
    return g if not noise else g.add_noise(noise)


def update(ensemble: Ensemble, given: Mapping[str, Array] | None = None, /, *,
           update_rule: UpdateRule, noise: Mapping[str, PSDLinOp] | None = None,
           approximation: Callable | None = None, key=None, **kw) -> Ensemble:
    """One ensemble Kalman update: build the rule's update, then call it with the values."""
    given = _blocks("update", given, kw)
    if ensemble.is_weighted:
        raise ValueError("update: weighted ensembles are not supported; resample first")
    build_approximation = gaussian_approximation if approximation is None else approximation
    joint = build_approximation(ensemble, noise or {})
    return update_rule.build(ensemble, joint, tuple(given))(given, key=key)


# ---------------------------------------------------------------------------
# inflation and relaxation: plain functions on ensembles
# ---------------------------------------------------------------------------


def inflate_multiplicative(ensemble: Ensemble, anomaly_scale, names=None) -> Ensemble:
    names = ensemble.names if names is None else names
    return ensemble.assign({n: ensemble.mean(n) + anomaly_scale * ensemble.anomalies(n) for n in names})


def relax_to_prior_spread(prior: Ensemble, posterior: Ensemble, alpha, names=None) -> Ensemble:
    names = posterior.names if names is None else names
    out = {}
    for n in names:
        sp, sa = jnp.std(prior[n], axis=0, ddof=1), jnp.std(posterior[n], axis=0, ddof=1)
        out[n] = posterior.mean(n) + posterior.anomalies(n) * (alpha * (sp - sa) / sa + 1.0)
    return posterior.assign(out)


# ---------------------------------------------------------------------------
# domain localization
# ---------------------------------------------------------------------------


def gaspari_cohn(r: Array) -> Array:
    """Gaspari-Cohn taper of r = distance / radius; zero for r >= 1 (half-width 0.5)."""
    z = 2.0 * jnp.abs(r)
    inner = -0.25 * z**5 + 0.5 * z**4 + 0.625 * z**3 - 5 / 3 * z**2 + 1
    outer = z**5 / 12 - 0.5 * z**4 + 0.625 * z**3 + 5 / 3 * z**2 - 5 * z + 4 - 2 / (3 * jnp.maximum(z, 1e-12))
    return jnp.where(z < 1, inner, jnp.where(z < 2, outer, 0.0))


def euclidean(a, b):
    return jnp.sqrt(jnp.sum((b - a) ** 2, axis=-1))


@_pytree(data=("target_coords", "given_coords"), meta=("radius", "taper", "distance", "max_neighbors"))
class DomainLocalization:
    """Locations, taper and neighborhood size for domain-localized updates."""

    def __init__(self, target_coords: Mapping[str, Array | None], given_coords,
                 *, radius: float, max_neighbors: int, taper: Callable = gaspari_cohn,
                 distance: Callable = euclidean):
        object.__setattr__(self, "target_coords", dict(target_coords))
        object.__setattr__(self, "given_coords",
                           dict(given_coords) if isinstance(given_coords, Mapping) else given_coords)
        for k, v in dict(radius=radius, taper=taper, distance=distance,
                         max_neighbors=max_neighbors).items():
            object.__setattr__(self, k, v)


def _row_sd(where, name, D):
    """Per-row noise standard deviations of a row-local (diagonal) noise operator."""
    scale = 1.0
    while isinstance(D, PSDScaled):
        scale = scale * D.c
        D = D.op
    if not isinstance(D, PSDDiagonal):
        raise TypeError(f"{where}: noise on {name!r} must be row-local (a PSDDiagonal, "
                        f"possibly scaled); got {type(D).__name__}")
    return jnp.sqrt(scale * D.diag())


@_pytree(data=("localization",), meta=("update_rule",))
class LocalizedUpdateRule:
    """Domain localization around SymmetricSquareRoot or Matheron (row-local noise only)."""

    def __init__(self, update_rule, localization: DomainLocalization):
        if not isinstance(update_rule, (SymmetricSquareRoot, Matheron)):
            raise TypeError("LocalizedUpdateRule wraps SymmetricSquareRoot or Matheron")
        object.__setattr__(self, "update_rule", update_rule)
        object.__setattr__(self, "localization", localization)

    def build(self, particles, approximation, given):
        if not _aligned(particles, approximation):
            raise ValueError("LocalizedUpdateRule needs an approximation aligned with the particles")
        return _LocalizedUpdate(self, particles, approximation, _names(given))


class _LocalizedUpdate:
    def __init__(self, spec, particles, joint, gnames):
        self.spec, self.particles, self.joint, self.gnames = spec, particles, joint, gnames

    def __call__(self, values=None, /, *, key=None, **kw) -> Ensemble:
        given = _blocks("localized update", values, kw)
        loc, joint, particles, gnames = self.spec.localization, self.joint, self.particles, self.gnames
        J = particles.n_particles
        gc = loc.given_coords
        if not isinstance(gc, dict):  # one unnamed array: the single given block
            if len(gnames) != 1:
                raise ValueError("given_coords must be a mapping when conditioning on several blocks")
            gc = {gnames[0]: gc}
        coords = jnp.concatenate([gc[c] for c in gnames], axis=0)
        sd = jnp.concatenate([_row_sd("LocalizedUpdateRule", c, joint.block_cov(c)) for c in gnames])
        WF = jnp.concatenate([joint._dense_factor(c) for c in gnames], 0) / sd[:, None]  # (N, J)
        ystar = jnp.concatenate([jnp.asarray(given[c]) for c in gnames])
        mean_y = jnp.concatenate([joint.mean(c) for c in gnames])
        matheron = isinstance(self.spec.update_rule, Matheron)
        if matheron:
            if key is None:
                raise ValueError("LocalizedUpdateRule(Matheron) needs a key")
            # one whitened draw, shared by every local analysis and the global ones
            yi = jnp.concatenate([particles[c] for c in gnames], axis=1)
            b_all = (ystar - yi) / sd - jax.random.normal(key, yi.shape)  # (J, N)
        else:
            b_all = (ystar - mean_y) / sd  # (N,)
        K = loc.max_neighbors

        def analyze(xrow, s, b):
            G = IdentityPlusGram(s)
            if matheron:
                return xrow @ G.solve_factor(b).T  # (J,) increments
            return xrow @ G.solve_factor(b) + math.sqrt(J - 1) * (xrow @ G.inverse_sqrt().to_dense())

        def local(point, xrow):
            dist = loc.distance(point, coords)
            negd, idx = jax.lax.top_k(-dist, K)
            root = jnp.sqrt(loc.taper(-negd / loc.radius))
            s = (WF[idx] * root[:, None]).T  # (J, K)
            b = b_all[:, idx] * root if matheron else b_all[idx] * root
            return analyze(xrow, s, b)

        out = {}
        for n in joint.names:
            if n in gnames:
                continue
            F = joint._dense_factor(n)  # (d, J)
            pts = loc.target_coords.get(n)
            if pts is None:  # unlocated block: global, untapered, same draw
                res = jax.vmap(lambda xrow: analyze(xrow, WF.T, b_all))(F)
            else:
                res = jax.vmap(local)(pts, F)  # (d, J)
            out[n] = (particles[n] if matheron else joint.mean(n)) + res.T
        return Ensemble(out)
