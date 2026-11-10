"""backticked file names of my READMEs that resolve nowhere (scratch). python deadpaths.py README [README ...]"""
import os
import re
import sys

ROOT = os.getcwd()
BASES = ["", "bev_fusion", "autoresearch", "autoresearch/evalsuite", "bev_fusion/evaluation", "bev_fusion/tests", "bev_fusion/training", "bev_fusion/runner", "bev_fusion/records",
         "bev_fusion/results", "bev_fusion/data"]
for readme in sys.argv[1:]:
    here = os.path.dirname(readme)
    text = open(readme, encoding="utf-8").read()
    bad = []
    for tok in sorted(set(re.findall(r"`([A-Za-z0-9_./~-]+\.(?:py|md|sh|json|csv|txt))`", text))):
        if tok.startswith("~") or tok.startswith("/") or "<" in tok:
            continue
        t = tok.lstrip("./")
        if any(os.path.exists(os.path.join(ROOT, b, t)) for b in [here] + BASES):
            continue
        bad.append(tok)
    print(readme, "->", "all resolve" if not bad else "UNRESOLVED: " + ", ".join(bad))
