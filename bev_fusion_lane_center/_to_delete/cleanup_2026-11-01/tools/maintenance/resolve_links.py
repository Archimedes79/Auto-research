"""Resolve [[ID]] tokens in markdown files of literature/ to links of the paper notes:  [[GR01]] -> [GR01](papers/GR01_<slug>.md)   (relative to the file's folder).
Usage: python resolve_links.py literature/README.md [more files]      Unknown ids are reported and left as they are."""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PAPERS = os.path.join(ROOT, "literature", "papers")
files = {}
for f in os.listdir(PAPERS):
    if f.endswith(".md") and "_" in f:
        files[f.split("_", 1)[0]] = f
for path in sys.argv[1:]:
    p = os.path.abspath(path)
    src = open(p, encoding="utf-8").read()
    rel = os.path.relpath(PAPERS, os.path.dirname(p)).replace(os.sep, "/")
    unknown = []

    def sub(m):
        i = m.group(1)
        if i in files:
            return f"[{i}]({rel}/{files[i]})"
        unknown.append(i)
        return m.group(0)

    out = re.sub(r"\[\[([A-Z]+[0-9]+)\]\]", sub, src)
    open(p, "w", encoding="utf-8").write(out)
    print(path, "resolved; unknown ids:", sorted(set(unknown)))
