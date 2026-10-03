"""Execute example sources and write notebooks plus document fragments.

A source is a percent-format Python file (``# %%`` code cells,
``# %% [markdown]`` Markdown cells, Markdown lines prefixed with ``# ``).
For each source this script:

1. executes the code cells in order, in one namespace, capturing stdout;
2. resolves ``[@key]`` citations against ``references.REFERENCES`` and
   appends a References cell;
3. writes ``out/<name>.ipynb`` (validated with nbformat), with outputs;
4. writes ``out/<name>.html``, the same cells as an HTML fragment for the
   design document (Markdown via markdown-it, math passed through for MathJax).

Run from the pyEKI checkout with the prototype on PYTHONPATH:
    uv run python build_notebooks.py [names...]
"""
from __future__ import annotations

import contextlib
import html
import io
import pathlib
import re
import sys
import traceback

import nbformat
from markdown_it import MarkdownIt

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from references import REFERENCES  # noqa: E402

SRC, OUT = HERE / "src", HERE / "out"
OUT.mkdir(exist_ok=True)
MD = MarkdownIt("commonmark").enable("table")


def parse(text: str):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "markdown":
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        if body.strip():
            cells.append((kind, body))

    for line in text.splitlines():
        if line.startswith("# %% [markdown]"):
            flush(); kind, buf = "markdown", []
        elif line.startswith("# %%"):
            flush(); kind, buf = "code", []
        elif kind is not None:
            buf.append(line)
    flush()
    return cells


CITE = re.compile(r"\[(@[\w]+(?:\s*;\s*@[\w]+)*)\]")


def resolve_citations(text: str, used: list[str]) -> str:
    def repl(m):
        keys = [k.strip()[1:] for k in m.group(1).split(";")]
        labels = []
        for k in keys:
            if k not in REFERENCES:
                raise KeyError(f"unknown citation key {k!r}")
            if k not in used:
                used.append(k)
            labels.append(REFERENCES[k][0])
        return "(" + "; ".join(labels) + ")"
    return CITE.sub(repl, text)


def references_cell(used: list[str]) -> str:
    entries = sorted((REFERENCES[k][1] for k in used))
    return "## References\n\n" + "\n".join(f"- {e}" for e in entries)


MATH = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)


def md_to_html(text: str, heading_offset: int, anchor: str | None) -> str:
    """Markdown to HTML; $$..$$ and $..$ become MathJax \\[..\\] and \\(..\\)."""
    stash = []

    def protect(m):
        disp, inline = m.group(1), m.group(2)
        stash.append(("\\[" + disp + "\\]") if disp is not None else ("\\(" + inline + "\\)"))
        return f"MATHSTASH{len(stash) - 1}X"
    body = MD.render(MATH.sub(protect, text))
    body = re.sub(r"MATHSTASH(\d+)X", lambda m: html.escape(stash[int(m.group(1))], quote=False), body)

    def demote(m):
        level = min(int(m.group(1)) + heading_offset, 6)
        return f"<h{level}" + (f' id="{anchor}"' if (anchor and int(m.group(1)) == 1) else "")
    body = re.sub(r"<h([1-6])", demote, body)
    body = re.sub(r"</h([1-6])>", lambda m: f"</h{min(int(m.group(1)) + heading_offset, 6)}>", body)
    return body


def build(path: pathlib.Path) -> bool:
    name = path.stem
    cells = parse(path.read_text())
    used: list[str] = []
    ns: dict = {"__name__": "__main__"}
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    frag = []
    for kind, body in cells:
        if kind == "markdown":
            body = resolve_citations(body, used)
            nb.cells.append(nbformat.v4.new_markdown_cell(body))
            frag.append(md_to_html(body, 2, name.split("_")[0]))
            continue
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(body, f"{name}.cell", "exec"), ns)
        except Exception:
            print(f"FAIL {name}:\n{body}\n{traceback.format_exc()}")
            return False
        out = "\n".join(l for l in buf.getvalue().splitlines() if "Warning" not in l)
        cell = nbformat.v4.new_code_cell(body)
        if out.strip():
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=out + "\n")]
        nb.cells.append(cell)
        frag.append(f'<pre><code class="language-python">{html.escape(body)}</code></pre>')
        if out.strip():
            frag.append(f'<pre class="output"><code class="language-plaintext">{html.escape(out)}</code></pre>')
    if used:
        ref = references_cell(used)
        nb.cells.append(nbformat.v4.new_markdown_cell(ref))
        frag.append('<div class="refs">' + md_to_html(ref, 2, None) + "</div>")
    nbformat.validate(nb)
    nbformat.write(nb, OUT / f"{name}.ipynb")
    (OUT / f"{name}.html").write_text("\n".join(frag))
    (OUT / f"{name}.citations").write_text("\n".join(used))
    print(f"PASS {name}: {sum(k == 'code' for k, _ in cells)} code cells, {len(used)} citations")
    return True


if __name__ == "__main__":
    wanted = sys.argv[1:]
    paths = sorted(p for p in SRC.glob("ex*.py") if not wanted or any(w in p.stem for w in wanted))
    ok = all([build(p) for p in paths])
    sys.exit(0 if ok else 1)
