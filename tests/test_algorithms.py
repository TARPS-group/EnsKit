"""Tests of the shared policies of ``enskit.algorithms``.

The four classes wrap the inflation and relaxation functions of
``enskit.kalman``, whose numerics ``tests/test_kalman.py`` checks against
hand-written formulas. What is tested here is the wrapping: each class equals
its function bit for bit, validates its fields, is a pytree whose families
refuse, and prints its fields.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import enskit  # noqa: F401  -- enables x64 before any array exists
from enskit import kalman
from enskit.algorithms import (
    AdditiveInflation,
    Inflation,
    MultiplicativeInflation,
    Relaxation,
    RelaxToPriorPerturbations,
    RelaxToPriorSpread,
)
from enskit.distribution import Ensemble
from enskit.linalg import DensePSD, PSDDiagonal, debug_checks

RNG = np.random.default_rng(3)


def _pair():
    prior = Ensemble(
        u=jnp.asarray(RNG.normal(size=(7, 3))),
        v=jnp.asarray(RNG.normal(size=(7, 2))),
        g=jnp.asarray(RNG.normal(size=(7, 4))),
    )
    posterior = Ensemble(
        u=jnp.asarray(0.2 * RNG.normal(size=(7, 3))),
        v=jnp.asarray(0.2 * RNG.normal(size=(7, 2))),
    )
    return prior, posterior


def _same(a: Ensemble, b: Ensemble) -> None:
    assert a.names == b.names
    for n in a.names:
        assert np.array_equal(np.asarray(a[n]), np.asarray(b[n])), n


def test_each_policy_is_its_kalman_function():
    prior, posterior = _pair()
    key = jax.random.key(5)
    cov = DensePSD(jnp.eye(3) * 0.1)
    _same(
        MultiplicativeInflation(1.1, names="u")(key, ensemble=prior, step=3, beta=0.5),
        kalman.inflate_multiplicative(prior, 1.1, "u"),
    )
    _same(
        AdditiveInflation({"u": cov})(key, ensemble=prior),
        kalman.inflate_additive(key, prior, u=cov),
    )
    _same(
        RelaxToPriorSpread(0.3)(prior=prior, posterior=posterior, step=1),
        kalman.relax_to_prior_spread(prior, posterior, 0.3),
    )
    _same(
        RelaxToPriorPerturbations(0.3, names=("v",))(prior=prior, posterior=posterior),
        kalman.relax_to_prior_perturbations(prior, posterior, 0.3, ("v",)),
    )


def test_the_protocols_are_structural():
    """Any callable of the right signature is a policy; the classes are too."""
    assert isinstance(MultiplicativeInflation(1.1), Inflation)
    assert isinstance(RelaxToPriorSpread(0.5), Relaxation)


def test_the_fields_are_validated():
    with pytest.raises(ValueError, match="scalar"):
        MultiplicativeInflation(jnp.ones(3))
    with pytest.raises(TypeError, match="real number"):
        MultiplicativeInflation(True)
    with pytest.raises(ValueError, match="scalar"):
        RelaxToPriorSpread(jnp.asarray([0.1, 0.2]))
    with pytest.raises(ValueError, match="repeated"):
        RelaxToPriorPerturbations(0.5, names=("u", "u"))
    with pytest.raises(TypeError, match="str"):
        MultiplicativeInflation(1.1, names=(1,))
    with pytest.raises(ValueError, match="at least one"):
        AdditiveInflation()
    with pytest.raises(TypeError, match="PSDLinOp"):
        AdditiveInflation(u=jnp.eye(3))
    with pytest.raises(TypeError, match="twice"):
        AdditiveInflation({"u": PSDDiagonal(jnp.ones(3))}, u=PSDDiagonal(jnp.ones(3)))
    # The value checks are the kalman functions', in debug mode, at the call.
    prior, posterior = _pair()
    with debug_checks(), pytest.raises(ValueError, match="alpha"):
        RelaxToPriorSpread(1.5)(prior=prior, posterior=posterior)
    with debug_checks(), pytest.raises(ValueError, match="anomaly_scale"):
        MultiplicativeInflation(-1.0)(jax.random.key(0), ensemble=prior)


def test_the_policies_are_pytrees_and_a_family_refuses():
    for policy in (
        MultiplicativeInflation(1.1, names="u"),
        AdditiveInflation(u=PSDDiagonal(jnp.ones(3))),
        RelaxToPriorSpread(0.5),
        RelaxToPriorPerturbations(0.5),
    ):
        leaves, treedef = jax.tree.flatten(policy)
        rebuilt = jax.tree.unflatten(treedef, leaves)
        assert type(rebuilt) is type(policy)
        assert rebuilt.names == policy.names
        assert policy.batch_shape == ()
        assert repr(rebuilt) == repr(policy)
        with pytest.raises(AttributeError):
            policy.names = ("w",)

    # A traced scale flows through, and jit sees the policy as an argument.
    prior, _ = _pair()

    @jax.jit
    def inflate(policy, ensemble):
        return policy(jax.random.key(0), ensemble=ensemble)

    got = inflate(MultiplicativeInflation(1.5), prior)
    assert np.allclose(np.asarray(got["u"]),
                       np.asarray(kalman.inflate_multiplicative(prior, 1.5)["u"]))

    family = jax.vmap(lambda a: RelaxToPriorPerturbations(a))(jnp.asarray([0.1, 0.2]))
    assert family.batch_shape == (2,)
    assert repr(family).startswith("vmapped(")


def test_repr_shows_the_fields_and_no_arrays():
    assert repr(MultiplicativeInflation(1.05)) == (
        "MultiplicativeInflation(anomaly_scale=1.05)"
    )
    assert repr(RelaxToPriorPerturbations(0.25, names="u")) == (
        "RelaxToPriorPerturbations(alpha=0.25, names=('u',))"
    )
    assert repr(AdditiveInflation(u=PSDDiagonal(jnp.ones(3)))) == (
        "AdditiveInflation(u=PSDDiagonal(3, 3))"
    )


def test_a_float32_ensemble_stays_float32():
    """A Python-float scale on a float32 ensemble does not promote it."""
    prior, posterior = _pair()
    as32 = Ensemble({n: prior[n].astype(jnp.float32) for n in prior.names})
    post32 = Ensemble({n: posterior[n].astype(jnp.float32) for n in posterior.names})
    out = MultiplicativeInflation(1.2)(jax.random.key(0), ensemble=as32)
    assert all(out[n].dtype == jnp.float32 for n in out.names)
    out = RelaxToPriorSpread(0.5)(prior=as32, posterior=post32)
    assert all(out[n].dtype == jnp.float32 for n in out.names)
