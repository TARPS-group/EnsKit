"""The release: its version, the pages that name it, and the README's code.

The version is written once, as ``enskit.__version__``; the package metadata
and the documentation read it from there. These tests check the places that
cannot read it: the changelog's entry, the install commands that name the
release's tag, and the README's example, which no other test runs.

Between releases ``__version__`` is the next release's ``X.Y.Z.dev0`` and the
changelog's newest entry is headed ``X.Y.Z (unreleased)``; at a release it is
``X.Y.Z`` and the heading loses the suffix. The install commands always name
the newest released tag.
"""

from __future__ import annotations

import re
import tomllib
import warnings
from pathlib import Path

import enskit

ROOT = Path(__file__).parents[1]
#: A release tag in an install command: a direct reference, with or without
#: ``.git``, or a uv source's ``rev`` or ``tag``.
TAG = re.compile(
    r"github\.com/TARPS-group/EnsKit(?:\.git)?@(v[^\s\"']+)"
    r"|\b(?:rev|tag) = \"(v[^\"]+)\""
)


def _changelog() -> list[str]:
    """The changelog's entry headings, newest first."""
    return re.findall(r"^## (.+)$", (ROOT / "CHANGELOG.md").read_text(), re.M)


def _released() -> str:
    """The newest released version the changelog records."""
    return next(h for h in _changelog() if not h.endswith(" (unreleased)"))


def _parse(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def test_the_version_is_written_once():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert "version" not in project
    assert project["dynamic"] == ["version"]
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev0)?", enskit.__version__)


def test_the_changelog_has_an_entry_for_the_version():
    newest = _changelog()[0]
    if enskit.__version__.endswith(".dev0"):
        upcoming = enskit.__version__.removesuffix(".dev0")
        assert newest == f"{upcoming} (unreleased)"
        assert _parse(upcoming) > _parse(_released())
    else:
        assert newest == enskit.__version__


def test_every_install_command_names_the_newest_released_tag():
    for page in (ROOT / "README.md", ROOT / "docs" / "installation.md"):
        tags = [a or b for a, b in TAG.findall(page.read_text())]
        assert tags, page.name
        assert set(tags) == {f"v{_released()}"}, page.name


def test_the_installation_page_states_the_released_version():
    text = (ROOT / "docs" / "installation.md").read_text()
    assert f"enskit.__version__     # '{_released()}'" in text


def test_the_installation_page_states_the_requirements():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    floors = dict(
        re.fullmatch(r"(\w+)>=([\d.]+)", d).groups() for d in project["dependencies"]
    )
    text = " ".join((ROOT / "docs" / "installation.md").read_text().split())
    assert f"Python {project['requires-python'].removeprefix('>=')} or later" in text
    assert f"JAX {floors['jax']} or later" in text
    assert f"NumPy {floors['numpy']} or later" in text


def test_the_readme_examples_run():
    """Every block of the README, in one namespace, as a reader runs them."""
    blocks = re.findall(r"```python\n(.*?)```", (ROOT / "README.md").read_text(), re.S)
    assert len(blocks) == 3
    ns: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for i, block in enumerate(blocks):
            exec(compile(block, f"README.md block {i}", "exec"), ns)
    assert ns["result"].mean("u").shape == (2,)
    assert ns["noise"].shape == (5, 5)
