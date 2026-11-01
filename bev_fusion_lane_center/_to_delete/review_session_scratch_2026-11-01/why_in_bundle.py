"""one-off: which module of the bundle pulls in train.py / nuscenes_dataset.py / augment.py / feature_cache.py"""
import os
import sys

sys.path.insert(0, "autoresearch/tools")
import provenance as V  # noqa: E402

code = sys.argv[1]
want = {"train.py", "nuscenes_dataset.py", "augment.py", "feature_cache.py", "fallbacks.py"}
for rel in V.import_closure(code, ("model.py", "losses.py")):
    sub = V.import_closure(code, (rel,))
    others = [x for x in sub if x != rel]
    hit = sorted(want & set(others))
    # direct importers: closure of a single file minus closures of its dependencies is too fancy; print the direct import lines instead
    src = open(os.path.join(code, rel), encoding="utf-8").read()
    lines = [ln.strip() for ln in src.splitlines() if (ln.startswith(("import ", "from ")) or ln.strip().startswith(("import ", "from "))) and any(w[:-3] in ln for w in want)]
    if lines:
        print(rel, "->", lines[:4])
