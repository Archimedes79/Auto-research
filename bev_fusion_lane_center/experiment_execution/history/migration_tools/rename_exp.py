"""Folder scheme change: E<first id><letter>_<slug> (one id, lower-case letter kept) instead of ranges that looked like E19-26. Renames the existing experiment folders."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mig_common as M

GO = "--go" in sys.argv


def new_name(row):
    m = re.search(r"E(\d+)([a-z]?)", row["id"])
    tag = f"E{int(m.group(1)):03d}{m.group(2)}" if m else (re.sub(r"[^A-Za-z0-9]+", "", row["id"]) or "X")
    return f"{tag}_{M.slug(row['what'])}"


tab = json.load(open(f"{M.R}/experiments/EXPERIMENT_TABLE.json", encoding="utf-8"))
rows = list(tab["rows"])
# the extra rows of build_experiments.py (loop 17, E123, E34-36) were named by the old function too: rebuild them by name pattern
old_dirs = sorted(d for d in os.listdir(f"{M.R}/experiments") if os.path.isdir(f"{M.R}/experiments/{d}"))
pairs = {}
for r in rows:
    pairs[M.folder_name(r)] = new_name(r)
EXTRA = [("E193 / E194", "Champion D7 and the control on the failure series (seven failure types), plus four milder types"), ("E195", "Memory on the input: spatial hold, EMA + monitor hold"),
         ("E196", "Builder recursion switched off"), ("E197 / E198", "Failure-trained dense head and failure-frame fine-tune of the builder (two folds)"), ("E199", "Learned gated memory (12.5 k parameters, two folds)"),
         ("E200 / E201", "Route success on the failure series (map assignment under failure)"), ("E202", "Joint end-to-end fine-tune, three arms (builder + route head + consumer)"),
         ("E203", "Time per frame (latency of the chain under the loop-17 additions)"), ("E204 / E205 / E206", "Map-conditioned builder (SD map raster as extra input channels), diagnosis, dense head trained on failures"),
         ("E207", "Learned gate behind the map-conditioned builder (frozen builder B, two folds)"), ("E208", "Matched clean reference: failure-half frames of the series without perturbation"),
         ("E209", "Unseen failure types: builder trained WITHOUT night and blur rows, evaluated on night and blur (rerun pending, the first run crashed)"), ("E210", "Map-only baseline: the SD map alone drawn as a lane graph"),
         ("E212", "Planner-side consumer under sensor failure"), ("E123", "Differentiable ladder: eight builder designs D1-D8 by successive halving (rungs 0-3), control arms B0-B3"),
         ("E34 / E35 / E36", "Loop 3: map clean / stale / oracle conditions and joint runs (no catalogue row)")]
for i, w in EXTRA:
    r = dict(id=i, what=w)
    pairs[M.folder_name(r)] = new_name(r)
n = 0
for d in old_dirs:
    if d in pairs and pairs[d] != d:
        src, dst = f"{M.R}/experiments/{d}", f"{M.R}/experiments/{pairs[d]}"
        if os.path.exists(dst):
            print("COLLISION", d, "->", pairs[d])
            continue
        n += 1
        if GO:
            os.rename(src, dst)
            open(f"{M.RS}/MIGRATION_LOG.csv", "a").write(f"x,expname,{src},{dst}\n")
    elif d not in pairs:
        print("unmapped dir:", d)
print(n, "folders renamed" if GO else "folders would be renamed", "| new names unique:", len(set(pairs.values())) == len(pairs))
