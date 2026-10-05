"""The user guide's structure, and the code of its reference pages.

Several user-guide pages have a test of their own, in the test module of the
layer they document, which runs every block and checks the page's claims:
``running-an-inversion.md`` and the landing page in ``test_eki.py``,
``filtering.md`` in ``test_enkf.py``, ``localization.md`` in
``test_localization.py``, ``maps.md`` in ``test_maps.py`` and
``toy-models.md`` in ``test_toy.py``. ``forecast-and-update.md`` has its test
here.

The other pages are written as fragments, each block assuming an ensemble or
an operator the prose describes rather than builds. Here each fragment runs
after a setup that builds what it assumes, so a fragment that names a class,
a method or an argument the package no longer has fails here, even though
the page never shows the setup. A block that begins indented, a method shown
out of its class, is skipped.
"""

from __future__ import annotations

import re
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest

import enskit  # noqa: F401  -- enables x64 before any array exists

DOCS = Path(__file__).parents[1] / "docs"
GUIDE = DOCS / "user-guide"


def _blocks(page: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", page.read_text(), re.S)


#: The blocks of each page that begin indented, methods shown out of their
#: class, and so are skipped. A count, so that an edit indenting a block of
#: runnable code, which would silently stop testing it, fails instead.
SKIPPED = {"writing-an-operator.md": 3}


def _run(page: Path, setups: dict[int, str]) -> dict:
    """Every block of ``page`` in one namespace, ``setups[i]`` before block ``i``.

    The namespace is a module's, registered in ``sys.modules``, because
    ``@linop`` resolves a class's annotations through its module, as it would
    for a page's code pasted into a file.
    """
    module = types.ModuleType(f"docs_page_{page.stem.replace('-', '_')}")
    sys.modules[module.__name__] = module
    ns = module.__dict__
    ran = skipped = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, block in enumerate(_blocks(page)):
            if i in setups:
                exec(compile(setups[i], f"{page.name} setup {i}", "exec"), ns)
            if block[:1].isspace():
                skipped += 1
                continue
            exec(compile(block, f"{page.name} block {i}", "exec"), ns)
            ran += 1
    assert ran >= 1
    assert skipped == SKIPPED.get(page.name, 0), (page.name, skipped)
    return ns


_UPDATE_ENSEMBLE = """
import jax, jax.numpy as jnp
from enskit import maps, toy
from enskit.distribution import Ensemble
from enskit.linalg import Dense, DensePSD, PSDDiagonal
problem = toy.linear_gaussian(parameter_dim=4, data_dim=6)
ens = problem.prior.sample(jax.random.key(0), n_particles=32)
ens = maps.pushforward(ens, problem.forward, inputs="u", output="g")
y, R = problem.y, problem.noise_cov
y_a, y_b = y, y + 0.1
k1, k2, key = jax.random.split(jax.random.key(1), 3)
"""

_HYBRID_ENSEMBLE = """
x = jax.random.normal(jax.random.key(2), (32, 5))
H = Dense(jnp.eye(3, 5))
ens = maps.pushforward(Ensemble(x=x), maps.Linear(H), inputs="x", output="y")
B = DensePSD(jnp.eye(5))
y, R = jnp.ones(3), PSDDiagonal(jnp.full(3, 0.1))
"""

_DISTRIBUTIONS = """
import jax, jax.numpy as jnp
from enskit.linalg import PSDDiagonal
J = 64
x = jax.random.normal(jax.random.key(0), (J, 40))
g = x[:, :6] + 0.1 * jax.random.normal(jax.random.key(1), (J, 6))
m0, C0 = jnp.zeros(40), jnp.eye(40)
R = PSDDiagonal(jnp.full(6, 0.25))
y_obs = jnp.ones(6)
key = jax.random.key(2)
log_likelihoods = -0.5 * jnp.sum((g - y_obs) ** 2, axis=-1)
"""

_OPERATORS = """
import jax, jax.numpy as jnp
from enskit.linalg import *
k = jax.random.split(jax.random.key(0), 8)
d = jnp.linspace(1.0, 2.0, 6)
F = jax.random.normal(k[0], (6, 2))
b = jnp.ones(6)
y = jnp.ones(6)
A = DensePSD(jnp.eye(3) + 0.2)
B = DensePSD(jnp.eye(2) + 0.1)
x = jnp.ones(6)
S = jax.random.normal(k[1], (5, 3))
r = jnp.ones(3)
"""

_OPERATOR_ARITHMETIC = """
cov = LowRankUpdate(PSDDiagonal(d), Dense(F))
A, B = DensePSD(jnp.eye(4) + 0.2), Dense(jax.random.normal(k[2], (4, 3)))
R, dbeta = PSDDiagonal(jnp.ones(4)), 0.25
n = 4
As = jnp.broadcast_to(jnp.eye(n) + 0.1, (100, n, n))
xs = jnp.ones((100, n))
"""

_QUICKSTART = """
from enskit.linalg import Dense, DensePSD
A = DensePSD(jnp.eye(5) + 0.2)
B = Dense(jnp.ones((5, 2)))
dbeta = 0.5
residual = jnp.ones(5)
op = noise
b = jnp.ones(5)
"""

FRAGMENT_PAGES = {
    "updates.md": {0: _UPDATE_ENSEMBLE, 2: _HYBRID_ENSEMBLE, 3: _UPDATE_ENSEMBLE},
    "distributions.md": {0: _DISTRIBUTIONS},
    "operators.md": {
        0: _OPERATORS, 3: _OPERATOR_ARITHMETIC, 6: "A, b = jnp.eye(4) + 0.2, jnp.ones(4)"
    },
    "quickstart.md": {4: _QUICKSTART},
    "writing-a-forward-model.md": {},
    "writing-an-operator.md": {},
}


@pytest.mark.parametrize("name", sorted(FRAGMENT_PAGES))
def test_every_fragment_of_a_reference_page_runs(name):
    _run(GUIDE / name, FRAGMENT_PAGES[name])


def test_every_page_of_the_user_guide_has_a_test():
    """A page with no test of its own here is one of the layer tests' pages."""
    elsewhere = {
        "running-an-inversion.md", "filtering.md", "localization.md", "maps.md",
        "toy-models.md", "forecast-and-update.md", "index.md",
    }
    pages = {p.name for p in GUIDE.glob("*.md")}
    assert pages == elsewhere | set(FRAGMENT_PAGES)


def test_the_user_guide_index_lists_every_page_once_in_level_order():
    index = (GUIDE / "index.md").read_text()
    listed = []
    for tree in re.findall(r"```\{toctree\}\n(.*?)```", index, re.S):
        listed += [line.strip() for line in tree.splitlines()
                   if line.strip() and not line.strip().startswith(":")]
    pages = {p.stem for p in GUIDE.glob("*.md")} - {"index"}
    assert sorted(listed) == sorted(pages) and len(listed) == len(set(listed))
    levels = re.findall(r"^## (.+)$", index, re.M)
    assert levels == [
        "Running an algorithm", "One update", "Forecast and update separately",
        "Probabilistic operations", "Operators",
    ]


def test_the_forecast_and_update_page_runs_and_its_claims_hold():
    """Every block of ``forecast-and-update.md``, and the claims of its prose.

    "is within a small fraction of a posterior standard deviation of the exact
    one", "``exact_filter`` is this loop", and the hand-written cycle equals
    ``enkf.forecast`` then ``enkf.analysis`` bit for bit.
    """
    import jax.numpy as jnp

    ns = _run(GUIDE / "forecast-and-update.md", {})
    assert ns["ens"].names == ("x",)
    assert float(ns["gap"] / ns["spread"]) < 0.15
    exact, _ = ns["problem"].exact_filter()
    assert np.allclose(exact[-1].mean("x"), ns["belief"].mean("x"), atol=1e-12)
    assert np.allclose(
        exact[-1].cov("x").to_dense(), ns["belief"].cov("x").to_dense(), atol=1e-12
    )
    assert bool(jnp.all(ns["mine"]["x"] == ns["theirs"]["x"]))
    assert jnp.isfinite(ns["log_evidence"])
