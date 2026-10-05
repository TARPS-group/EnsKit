"""Execute the example sources and write the gallery's notebooks.

A source is a percent-format Python file in ``src/``: ``# %%`` opens a code
cell, ``# %% [markdown]`` a Markdown cell, and a Markdown cell's lines are
prefixed with ``# ``. For each source this script

1. resolves the ``[@key]`` citations of its Markdown cells against
   ``references.REFERENCES`` and appends a References cell;
2. executes the code cells in order, in one namespace, in a fresh
   interpreter, capturing what each prints;
3. writes ``<name>.ipynb`` beside this file, validated with ``nbformat``,
   with those outputs stored.

Sphinx renders the stored notebooks and never executes them. The sources are
what is reviewed and edited; a notebook is regenerated from its source, never
edited by hand. ``tests/test_examples.py`` checks that every stored notebook's
cells are its source's, and, in a test marked ``slow``, executes every source.

Run from the repository root::

    uv run python docs/examples/build.py             # every example
    uv run python docs/examples/build.py ex02 ex12   # the ones named
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import re
import subprocess
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "src"
sys.path.insert(0, str(HERE))

from references import REFERENCES  # noqa: E402

CITE = re.compile(r"\[(@\w+(?:\s*;\s*@\w+)*)\]")


def sources() -> list[pathlib.Path]:
    """Every example source, in gallery order."""
    return sorted(SRC.glob("ex*.py"))


def notebook_path(source: pathlib.Path) -> pathlib.Path:
    """Where the notebook built from ``source`` is stored."""
    return HERE / f"{source.stem}.ipynb"


def cells(source: pathlib.Path) -> list[tuple[str, str]]:
    """The notebook's cells, as ``(kind, text)`` pairs, before execution.

    Citations are resolved and the References cell is appended, so these are
    exactly the cell sources the stored notebook must hold.
    """
    used: list[str] = []
    out = []
    for kind, body in _parse(source.read_text()):
        if kind == "markdown":
            body = _resolve_citations(body, used)
        out.append((kind, body))
    if used:
        entries = sorted(REFERENCES[k][1] for k in used)
        listed = "\n".join(f"- {e}" for e in entries)
        out.append(("markdown", "## References\n\n" + listed))
    return out


def build(source: pathlib.Path) -> bool:
    """Execute ``source`` and write its notebook; return whether it succeeded."""
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python"}
    namespace: dict = {"__name__": "__main__"}
    executed = 0
    for kind, body in cells(source):
        if kind == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(body))
            continue
        printed = io.StringIO()
        try:
            with contextlib.redirect_stdout(printed):
                exec(compile(body, f"{source.stem}.cell", "exec"), namespace)
        except Exception:
            print(f"FAIL {source.stem}:\n{body}\n{traceback.format_exc()}")
            return False
        executed += 1
        cell = nbformat.v4.new_code_cell(body, execution_count=executed)
        if printed.getvalue().strip():
            cell.outputs = [
                nbformat.v4.new_output("stream", name="stdout", text=printed.getvalue())
            ]
        nb.cells.append(cell)
    nbformat.validate(nb)
    nbformat.write(nb, notebook_path(source))
    print(f"PASS {source.stem}")
    return True


def _parse(text: str) -> list[tuple[str, str]]:
    found, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "markdown":
            body = "\n".join(
                line[2:] if line.startswith("# ") else line.lstrip("#")
                for line in body.splitlines()
            )
        if body.strip():
            found.append((kind, body))

    for line in text.splitlines():
        if line.startswith("# %% [markdown]"):
            flush()
            kind, buf = "markdown", []
        elif line.startswith("# %%"):
            flush()
            kind, buf = "code", []
        elif kind is not None:
            buf.append(line)
    flush()
    return found


def _resolve_citations(text: str, used: list[str]) -> str:
    def replace(match):
        keys = [k.strip()[1:] for k in match.group(1).split(";")]
        for k in keys:
            if k not in REFERENCES:
                raise KeyError(f"unknown citation key {k!r}")
            if k not in used:
                used.append(k)
        return "(" + "; ".join(REFERENCES[k][0] for k in keys) + ")"

    return CITE.sub(replace, text)


if __name__ == "__main__":
    wanted = sys.argv[1:]
    chosen = [p for p in sources() if not wanted or any(w in p.stem for w in wanted)]
    if len(chosen) == 1:
        sys.exit(0 if build(chosen[0]) else 1)
    # one fresh interpreter per example, so no state carries from one to the next
    failed = [
        p.stem
        for p in chosen
        if subprocess.run([sys.executable, __file__, p.stem]).returncode != 0
    ]
    sys.exit(1 if failed else 0)
