"""The examples gallery: its stored notebooks, and the sources they come from.

Each example is a percent-format source in ``docs/examples/src``, and its
notebook in ``docs/examples`` is generated from it with outputs stored, which
is what the documentation renders. Two kinds of test:

1. **Fast**: every stored notebook's cells are exactly its source's cells, so
   a notebook cannot show code its source no longer holds; the gallery's
   index lists every example; and every citation resolves.
2. **Slow** (``pytest -m slow``): every source executes in a fresh
   interpreter. Each source ends with a ``# checks`` cell of assertions on the
   numbers its text states, so this is what keeps the text true. The
   Lorenz-96 examples take up to twenty seconds each.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[1] / "docs" / "examples"
sys.path.insert(0, str(EXAMPLES))

import build  # noqa: E402  -- docs/examples/build.py, on the path just above

SOURCES = build.sources()


def test_there_are_fifteen_examples():
    assert [p.stem[:4] for p in SOURCES] == [f"ex{i:02d}" for i in range(1, 16)]


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.stem)
def test_the_stored_notebook_is_its_source(source):
    """The cells match the source's, in order; every code cell was executed.

    Regenerate with ``uv run python docs/examples/build.py <name>`` when this
    fails.
    """
    import nbformat

    path = build.notebook_path(source)
    assert path.exists(), f"{path.name} is missing; run docs/examples/build.py"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    stored = [(cell.cell_type, cell.source) for cell in notebook.cells]
    assert stored == build.cells(source), f"{path.name} is stale; rebuild it"
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert "execution_count" in cell
            assert not [o for o in cell.outputs if o.output_type == "error"]


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.stem)
def test_the_example_has_the_gallery_structure(source):
    """A title, when to use it, its setup, what to notice, and final checks."""
    cells = build.cells(source)
    markdown = "\n".join(text for kind, text in cells if kind == "markdown")
    assert cells[0][0] == "markdown" and cells[0][1].startswith("# ")
    headings = ("## When to use this", "## Setup", "## What to notice", "## References")
    for heading in headings:
        assert heading in markdown, (source.stem, heading)
    code = [text for kind, text in cells if kind == "code"]
    assert code[-1].lstrip().startswith("# checks"), source.stem
    assert "assert " in code[-1], source.stem


def test_the_index_lists_every_example_once():
    index = (EXAMPLES / "index.md").read_text()
    toctree = index.split("```{toctree}", 1)[1].split("```", 1)[0]
    listed = [line.strip() for line in toctree.splitlines()]
    listed = [line for line in listed if line.startswith("ex")]
    assert listed == [p.stem for p in SOURCES]
    for source in SOURCES:
        assert index.count(f"{{doc}}`{source.stem}`") == 1, source.stem


def test_an_unknown_citation_raises():
    with pytest.raises(KeyError, match="nosuch2099"):
        build._resolve_citations("as shown [@nosuch2099]", [])


def test_citations_resolve_in_order_of_first_use():
    used: list[str] = []
    text = "[@iglesias2013; @evensen1994] and [@iglesias2013]"
    text = build._resolve_citations(text, used)
    assert used == ["iglesias2013", "evensen1994"]
    assert "@" not in text


@pytest.mark.slow
@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.stem)
def test_the_example_runs_and_its_checks_pass(source):
    """Execute the source in a fresh interpreter; its final cell asserts its claims."""
    done = subprocess.run(
        [sys.executable, str(source)], capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, done.stderr[-4000:]
