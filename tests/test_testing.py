"""Tests of ``enskit.testing``: the checks pass conforming pieces and fail broken ones.

A check that passes everything is worse than none, so each obligation is
exercised by a mutant that violates it and nothing else.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import enskit  # noqa: F401  -- enables x64 before any array exists
from enskit import kalman
from enskit.distribution import Ensemble, Gaussian
from enskit.linalg import DensePSD, PSDDiagonal
from enskit.testing import check_conditional_map, check_update_rule

# ---------------------------------------------------------------------------
# check_update_rule
# ---------------------------------------------------------------------------


class DEnKF:
    """Example 14's rule: full gain on the mean, half on the anomalies."""

    def build(self, particles, approximation, given):
        cmap = approximation.conditional_map(given)
        halfway = particles.assign(
            {c: particles.mean(c) + 0.5 * particles.anomalies(c) for c in given}
        )

        def call(values=None, /, *, key=None, **block_values):
            return cmap(halfway, values, **block_values).marginal(*cmap.targets)

        return call


class _Wrapped:
    """A shipped rule with its result changed by ``change``."""

    def __init__(self, rule, change):
        self.rule, self.change = rule, change

    def build(self, particles, approximation, given):
        inner = self.rule.build(particles, approximation, given)

        def call(values=None, /, *, key=None, **kw):
            return self.change(inner(values, key=key, **kw))

        return call


class _Remembers:
    """Keeps the first values it is called with: the build depends on values."""

    def build(self, particles, approximation, given):
        inner = kalman.SymmetricSquareRoot().build(particles, approximation, given)
        seen = []

        def call(values=None, /, *, key=None, **kw):
            inner(values, key=key)  # validates the values
            if not seen:
                seen.append(values)
            return inner(seen[0], key=key)

        return call


class _Unperturbed:
    """The transport without the noise draw: deterministic, too little spread."""

    def build(self, particles, approximation, given):
        cmap = approximation.conditional_map(given)

        def call(values=None, /, *, key=None, **kw):
            return cmap(particles, values, **kw).marginal(*cmap.targets)

        return call


class _Untraceable:
    """Reads a value as a Python float, which fails under jit."""

    def build(self, particles, approximation, given):
        inner = kalman.SymmetricSquareRoot().build(particles, approximation, given)

        def call(values=None, /, *, key=None, **kw):
            float(values["g"][0])
            return inner(values, key=key)

        return call


def test_check_update_rule_passes_the_shipped_rules_and_example_14():
    check_update_rule(kalman.SymmetricSquareRoot())
    check_update_rule(kalman.Matheron())
    check_update_rule(kalman.Matheron(), key=jax.random.key(99))
    check_update_rule(DEnKF(), exact_covariance=False)


def test_check_update_rule_fails_the_deterministic_enkf_claiming_exact_covariance():
    with pytest.raises(AssertionError, match="obligation 4.*covariance"):
        check_update_rule(DEnKF())


@pytest.mark.parametrize(
    ("rule", "obligation"),
    [
        pytest.param(
            _Wrapped(kalman.SymmetricSquareRoot(), lambda out: out.marginal("v", "u")),
            "1",
            id="targets-out-of-order",
        ),
        pytest.param(
            _Wrapped(
                kalman.SymmetricSquareRoot(),
                lambda out: Ensemble({n: out[n].astype(jnp.float32) for n in out.names}),
            ),
            "1",
            id="demotes-the-dtype",
        ),
        pytest.param(
            _Wrapped(kalman.SymmetricSquareRoot(), lambda out: out.marginal("u")),
            "1",
            id="drops-a-target",
        ),
        pytest.param(_Remembers(), "3", id="build-depends-on-values"),
        pytest.param(
            _Wrapped(
                kalman.SymmetricSquareRoot(), lambda out: out.assign(u=out["u"] + 0.1)
            ),
            "4",
            id="biased-deterministic",
        ),
        pytest.param(
            _Wrapped(kalman.Matheron(), lambda out: out.assign(u=out["u"] + 0.05)),
            "4",
            id="biased-stochastic",
        ),
        pytest.param(_Unperturbed(), "4", id="no-noise-draw"),
        pytest.param(_Untraceable(), "5", id="not-traceable"),
    ],
)
def test_check_update_rule_fails_each_mutant(rule, obligation):
    with pytest.raises(AssertionError, match=f"obligation {obligation}"):
        check_update_rule(rule)


# ---------------------------------------------------------------------------
# check_conditional_map
# ---------------------------------------------------------------------------


def _joint():
    return Gaussian(
        {"u": jnp.zeros(3), "v": jnp.ones(2), "g": jnp.asarray([0.5, -1.0, 0.0, 2.0])},
        factors={
            "u": jnp.eye(3) + 0.1,
            "g": jnp.asarray(
                [[1.0, 0.5, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.2, 0.0, 2.0]]
            ),
        },
        block_covs={"v": PSDDiagonal(jnp.full(2, 2.0)), "g": DensePSD(0.5 * jnp.eye(4))},
    )


class _MapWrapper:
    """A MatheronMap with its result changed by ``change(samples, out)``."""

    def __init__(self, cmap, change):
        self.cmap, self.change = cmap, change
        self.given, self.targets = cmap.given, cmap.targets

    def __call__(self, samples, values=None, /, *, key=None, **kw):
        return self.change(samples, self.cmap(samples, values, key=key, **kw))


def test_check_conditional_map_passes_the_matheron_map():
    joint = _joint()
    check_conditional_map(joint.conditional_map("g"), joint, "g")
    check_conditional_map(joint.conditional_map(("g", "v")), joint, ("v", "g"))


@pytest.mark.parametrize(
    ("change", "obligation"),
    [
        pytest.param(
            lambda s, out: out.assign(u=out["u"] + 1e-3 * s["u"][:1]),
            "4",
            id="couples-samples",
        ),
        pytest.param(
            lambda s, out: out.marginal(*[n for n in out.names if n in ("u", "v")]),
            "3",
            id="drops-an-extra-block",
        ),
        pytest.param(
            lambda s, out: Ensemble({n: out[n] for n in out.names}),
            "3",
            id="loses-the-weights",
        ),
        pytest.param(
            lambda s, out: out.assign(u=out["u"] * 1.01),
            "6",
            id="wrong-conditional",
        ),
    ],
)
def test_check_conditional_map_fails_each_mutant(change, obligation):
    joint = _joint()
    cmap = _MapWrapper(joint.conditional_map("g"), change)
    with pytest.raises(AssertionError, match=f"obligation {obligation}"):
        check_conditional_map(cmap, joint, "g")


def test_check_conditional_map_checks_the_names():
    joint = _joint()
    cmap = joint.conditional_map("g")
    with pytest.raises(AssertionError, match="obligation 1"):
        check_conditional_map(cmap, joint, ("g", "v"))


# ---------------------------------------------------------------------------
# mutants for the obligations the review found unexercised
# ---------------------------------------------------------------------------


class _Promotes:
    """Returns float64 from float32 particles (casts every result up)."""

    def build(self, particles, approximation, given):
        inner = kalman.SymmetricSquareRoot().build(particles, approximation, given)

        def call(values=None, /, *, key=None, **kw):
            out = inner(values, key=key, **kw)
            return Ensemble({n: out[n].astype(jnp.float64) for n in out.names})

        return call


class _Lenient:
    """Drops values for blocks that are not given and fills missing ones."""

    def build(self, particles, approximation, given):
        inner = kalman.SymmetricSquareRoot().build(particles, approximation, given)
        default = {c: approximation.mean(c) for c in given}

        def call(values=None, /, *, key=None, **kw):
            vals = {**default, **{c: v for c, v in (values or {}).items() if c in given}}
            vals = {
                c: (v if v.shape == default[c].shape else default[c])
                for c, v in vals.items()
            }
            return inner(vals, key=key)

        return call


class _Forgetful:
    """Stochastic, but ignores the key it is given after the first call."""

    def build(self, particles, approximation, given):
        inner = kalman.Matheron().build(particles, approximation, given)
        count = [0]

        def call(values=None, /, *, key=None, **kw):
            count[0] += 1
            return inner(values, key=jax.random.key(count[0]), **kw)

        return call


class _DefaultKey:
    """Stochastic, but supplies its own key when none is given."""

    def build(self, particles, approximation, given):
        inner = kalman.Matheron().build(particles, approximation, given)

        def call(values=None, /, *, key=None, **kw):
            return inner(values, key=jax.random.key(0) if key is None else key, **kw)

        return call


class _BreaksUnderVmap:
    """Correct, except when its values are batched by jax.vmap."""

    def build(self, particles, approximation, given):
        inner = kalman.SymmetricSquareRoot().build(particles, approximation, given)

        def call(values=None, /, *, key=None, **kw):
            out = inner(values, key=key, **kw)
            if type(values["g"]).__name__ == "BatchTracer":
                out = out.assign(u=out["u"] + 1.0)
            return out

        return call


@pytest.mark.parametrize(
    ("rule", "obligation"),
    [
        pytest.param(_Promotes(), "1 \\(float32\\)", id="promotes-float32"),
        pytest.param(_Lenient(), "1", id="accepts-bad-values"),
        pytest.param(_Forgetful(), "2", id="not-deterministic-given-the-key"),
        pytest.param(_DefaultKey(), "2", id="no-error-without-a-key"),
        pytest.param(_BreaksUnderVmap(), "5", id="wrong-under-vmap"),
    ],
)
def test_check_update_rule_fails_the_review_mutants(rule, obligation):
    with pytest.raises(AssertionError, match=f"obligation {obligation}"):
        check_update_rule(rule)


class _LenientMap(_MapWrapper):
    """Ignores values for blocks that are not given and fills missing ones."""

    def __init__(self, cmap, joint):
        super().__init__(cmap, lambda s, out: out)
        self.default = {c: joint.mean(c) for c in cmap.given}

    def __call__(self, samples, values=None, /, *, key=None, **kw):
        vals = {
            **self.default,
            **{c: v for c, v in (values or {}).items() if c in self.given},
        }
        vals = {
            c: (v if v.shape == self.default[c].shape else self.default[c])
            for c, v in vals.items()
        }
        return self.cmap(samples, vals, key=key)


def test_check_conditional_map_fails_a_map_accepting_bad_values():
    joint = _joint()
    with pytest.raises(AssertionError, match="obligation 2"):
        check_conditional_map(_LenientMap(joint.conditional_map("g"), joint), joint, "g")


def test_check_conditional_map_checks_the_keyed_path_when_asked():
    joint = _joint()
    check_conditional_map(joint.conditional_map("g"), joint, "g", keyed_noise_free=True)

    class _WrongWhenKeyed(_MapWrapper):
        def __call__(self, samples, values=None, /, *, key=None, **kw):
            out = self.cmap(samples, values, key=key, **kw)
            return out if key is None else out.assign(u=out["u"] + 0.5)

    cmap = _WrongWhenKeyed(joint.conditional_map("g"), lambda s, out: out)
    check_conditional_map(cmap, joint, "g")  # the keyed path is opt-in
    with pytest.raises(AssertionError, match="obligation 7"):
        check_conditional_map(cmap, joint, "g", keyed_noise_free=True)
