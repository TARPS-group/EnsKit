"""Prototype toy problems."""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from .distribution import Gaussian
from .linalg import Dense, DensePSD, PSDDiagonal
from .maps import Linear


@dataclasses.dataclass(frozen=True)
class InverseProblem:
    prior: Gaussian
    forward: object
    noise_cov: object
    y: jnp.ndarray
    u_true: jnp.ndarray


def linear_gaussian(parameter_dim=4, data_dim=6, seed=0, noise_sd=0.3):
    k = jax.random.split(jax.random.key(seed), 4)
    A = jax.random.normal(k[0], (data_dim, parameter_dim)) / jnp.sqrt(parameter_dim)
    B = jax.random.normal(k[1], (parameter_dim, parameter_dim))
    C = B @ B.T / parameter_dim + 0.5 * jnp.eye(parameter_dim)
    prior = Gaussian.independent({"u": (jnp.zeros(parameter_dim), DensePSD(C))})
    R = PSDDiagonal(jnp.full(data_dim, noise_sd**2))
    u_true = jnp.linalg.cholesky(C) @ jax.random.normal(k[2], (parameter_dim,))
    y = A @ u_true + noise_sd * jax.random.normal(k[3], (data_dim,))
    return InverseProblem(prior, Linear(Dense(A)), R, y, u_true)


def _decay(u, t):
    return u[..., :1] * jnp.exp(-u[..., 1:2] * t)


def exponential_decay(seed=0, n_times=10, noise_sd=0.05):
    t = jnp.linspace(0.1, 2.0, n_times)
    u_true = jnp.array([1.5, 1.2])
    prior = Gaussian.independent({"u": (jnp.array([1.0, 0.5]), PSDDiagonal(jnp.array([0.5, 0.5])))})
    R = PSDDiagonal(jnp.full(n_times, noise_sd**2))
    y = _decay(u_true, t) + noise_sd * jax.random.normal(jax.random.key(seed), (n_times,))
    return InverseProblem(prior, lambda u: _decay(u, t), R, y, u_true)


def l96_tendency(x, forcing):
    return (jnp.roll(x, -1, -1) - jnp.roll(x, 2, -1)) * jnp.roll(x, 1, -1) - x + forcing


def l96_step(x, forcing=8.0, dt=0.05):
    f = lambda z: l96_tendency(z, forcing)  # noqa: E731
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


@dataclasses.dataclass(frozen=True)
class StateSpaceProblem:
    initial: Gaussian
    transition: object
    transition_noise: object
    observe: object
    noise_cov: object
    observations: jnp.ndarray
    truth: jnp.ndarray
    coords: jnp.ndarray = None
    obs_coords: jnp.ndarray = None


def lorenz96(dim=40, n_steps=200, obs_every=2, noise_sd=1.0, forcing=8.0, seed=0):
    key = jax.random.key(seed)
    x = forcing + 0.01 * jax.random.normal(key, (dim,))
    for _ in range(1000):  # spin up onto the attractor
        x = l96_step(x, forcing)
    truth, obs = [], []
    idx = jnp.arange(0, dim, obs_every)
    H = jnp.eye(dim)[idx]
    keys = jax.random.split(jax.random.key(seed + 1), n_steps)
    x0 = x
    for t in range(n_steps):
        x = l96_step(x, forcing)
        truth.append(x)
        obs.append(H @ x + noise_sd * jax.random.normal(keys[t], (idx.shape[0],)))
    initial = Gaussian.independent({"x": (x0, PSDDiagonal(jnp.ones(dim)))})
    return StateSpaceProblem(
        initial, lambda X: l96_step(X, forcing), None, Linear(Dense(H)),
        PSDDiagonal(jnp.full(idx.shape[0], noise_sd**2)), jnp.stack(obs), jnp.stack(truth),
        coords=jnp.arange(dim, dtype=float)[:, None], obs_coords=idx.astype(float)[:, None])


def linear_state_space(dim=3, obs_dim=2, n_steps=20, seed=0):
    k = jax.random.split(jax.random.key(seed), 5)
    Q_, _ = jnp.linalg.qr(jax.random.normal(k[0], (dim, dim)))
    A = 0.95 * Q_
    H = jax.random.normal(k[1], (obs_dim, dim))
    Q = PSDDiagonal(jnp.full(dim, 0.1))
    R = PSDDiagonal(jnp.full(obs_dim, 0.2))
    x = jnp.zeros(dim)
    truth, obs = [], []
    keys = jax.random.split(k[2], n_steps)
    for t in range(n_steps):
        a, b = jax.random.split(keys[t])
        x = A @ x + jnp.sqrt(0.1) * jax.random.normal(a, (dim,))
        truth.append(x)
        obs.append(H @ x + jnp.sqrt(0.2) * jax.random.normal(b, (obs_dim,)))
    initial = Gaussian.independent({"x": (jnp.zeros(dim), PSDDiagonal(jnp.ones(dim)))})
    return StateSpaceProblem(initial, Linear(Dense(A)), Q, Linear(Dense(H)), R,
                             jnp.stack(obs), jnp.stack(truth))
