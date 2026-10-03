"""Assemble the EnsKit design document (``docs/redesign/design.html``).

Inputs, all under ``docs/redesign/``: the page sections in ``document/sections``,
the API stubs in ``stubs``, and the executed notebook fragments in
``notebooks/out`` (rebuild those first with ``notebooks/build_notebooks.py``).
Citations written ``[@key]`` resolve against ``notebooks/references.py``.

Run with any Python 3: ``python docs/redesign/document/build.py``.
"""
import html
import pathlib
import re

ROOT = pathlib.Path(__file__).parent
REDESIGN = ROOT.parent

sections = sorted((ROOT / "sections").glob("*.html"))
head = sections[0].read_text()
body = "\n".join(p.read_text() for p in sections[1:])


def code_block(label, text, lang="python", cls=""):
    return (f'<div class="codehead"><strong>{html.escape(label)}</strong></div>'
            f'<pre class="{cls}"><code class="language-{lang}">{html.escape(text.rstrip())}</code></pre>')


def stub(m):
    name = m.group(1)
    return code_block(name.replace(".py", "").replace("_", " "), (REDESIGN / "stubs" / name).read_text())


# citations in the prose: [@key] or [@a; @b] -> (Author, Year); collected for References
import sys as _sys
_sys.path.insert(0, str(REDESIGN / "notebooks"))
from references import REFERENCES  # noqa: E402
cited = []


def cite(m):
    keys = [k.strip()[1:] for k in m.group(1).split(";")]
    for k in keys:
        if k not in REFERENCES:
            raise KeyError(k)
        if k not in cited:
            cited.append(k)
    return "(" + "; ".join(html.escape(REFERENCES[k][0]) for k in keys) + ")"


body = re.sub(r"\[(@\w+(?:\s*;\s*@\w+)*)\]", cite, body)

NB = REDESIGN / "notebooks" / "out"
nb_count = [0]


def notebook(m):
    name = m.group(1)
    nb_count[0] += 1
    frag = (NB / f"{name}.html").read_text()
    for k in (NB / f"{name}.citations").read_text().split():
        if k not in cited:
            cited.append(k)
    tag = name.split("_")[0]
    frag = frag.replace(f'<h3 id="{tag}">', f'<h3 id="{tag}"><span class="num">7.{nb_count[0]}</span>', 1)
    return f'<div class="notebook">{frag}</div>'


body = re.sub(r"\{\{NB:([\w.]+)\}\}", notebook, body)
body = re.sub(r"\{\{STUB:([\w.]+)\}\}", stub, body)
assert "{{" not in body, re.findall(r"\{\{[^}]*\}\}", body)

def _md_inline(t):
    t = html.escape(t)
    return re.sub(r"\*(.+?)\*", r"<em>\1</em>", t)


stub_text = "\n".join(q.read_text() for q in (REDESIGN / "stubs").glob("*.py"))
for k, (_, full) in REFERENCES.items():
    if full.replace("*", "")[:60] in stub_text.replace("\n", " ").replace("     ", " ") and k not in cited:
        cited.append(k)
refs = sorted(REFERENCES[k][1] for k in cited)
body += ('\n<h2 id="references">References</h2>\n<p>Every work cited in the prose, '
         'the docstrings and the examples, checked against the publisher or arXiv record.</p>\n'
         '<ul class="biblio">' + "".join(f"<li>{_md_inline(r)}</li>" for r in refs) + "</ul>\n")


# contents rail from h2/h3 ids
items = []
for level, ident, inner in re.findall(r'<h([23]) id="([^"]+)">(.*?)</h\1>', body, flags=re.S):
    label = re.sub(r"<span class=\"tag[^\"]*\">.*?</span>", "", inner)
    label = re.sub(r"<[^>]+>", "", label)
    label = re.sub(r"^\s*[\d.]+", lambda m: m.group(0).strip() + " ", label).strip()
    items.append((int(level), ident, " ".join(label.split())))

toc, open_sub = [], False
for level, ident, label in items:
    if level == 2:
        if open_sub:
            toc.append("</ol></li>")
            open_sub = False
        elif toc:
            toc.append("</li>")
        toc.append(f'<li><a href="#{ident}">{html.escape(label)}</a>')
    else:
        if not open_sub:
            toc.append("<ol>")
            open_sub = True
        toc.append(f'<li><a href="#{ident}">{html.escape(label)}</a></li>')
toc.append("</ol></li>" if open_sub else "</li>")

page = f"""{head}
<div class="shell">
<nav class="toc" aria-label="Contents"><details id="toc-details" open><summary>Contents</summary>
<ol>{''.join(toc)}</ol></details></nav>
<main>
{body}
</main>
</div>
<script>
(function () {{
  try {{ if (window.matchMedia('(max-width: 1119px)').matches) document.getElementById('toc-details').open = false; }} catch (e) {{}}
  if (window.hljs) {{ document.querySelectorAll('pre code').forEach(function (el) {{
    if (!el.classList.contains('language-plaintext')) window.hljs.highlightElement(el); }}); }}
}})();
</script>
"""
out = REDESIGN / "design.html"
out.write_text(page)
print(out, len(page) // 1024, "KiB,", len(items), "headings")
