"""Candidates for deletion in the lane project (scratch): files that the live head and the full test list do not import, with every place that still names them, sorted by kind of place."""
import collections
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.dirname(__file__))
import reach2  # noqa: E402,F401  (prints its tables; we reuse its sets)

orph = sorted(reach2.allp - reach2.alltests - reach2.active)
KEEP_DIRS = ("training/data/prep/", "tools/literature/")               # reproduction recipes and the literature tools stay by policy


def kind(path):
    rel = os.path.relpath(path, REPO).replace("\\", "/")
    r = rel.split("bev_fusion_lane_center/", 1)[-1] if "bev_fusion_lane_center/" in rel else rel
    if rel.startswith("autoresearch/") or rel.startswith("GtGenerator/") or rel.startswith("bev_fusion/"):
        return "shared/other"
    if r.startswith("temp/"):
        return "scratch"
    if r.startswith(("experiments/", "results/")):
        return "card/result"
    if r.startswith(("experiment_execution/", "runs/", "reports/", "literature/")):
        return "record"
    if os.path.basename(r) in ("README.md", "HANDOVER.md", "PROBLEM.md", "CLAUDE.md", "STRUCTURE.md", "architecture.json", "project.json"):
        return "living doc"
    if r.endswith((".py", ".sh")):
        return "code"
    return "other"


rows = []
for p in orph:
    rel = os.path.relpath(p, ROOT).replace("\\", "/")
    if rel.startswith(KEEP_DIRS):
        continue
    stem = os.path.basename(p)[:-3]
    out = subprocess.run(["grep", "-rIlw", "--exclude-dir=__pycache__", "--exclude-dir=.git", "--exclude-dir=_to_delete", stem, ROOT, os.path.join(REPO, "autoresearch"), os.path.join(REPO, "GtGenerator")],
                         capture_output=True, text=True).stdout.split()
    refs = collections.defaultdict(list)
    for q in out:
        if os.path.abspath(q) == os.path.abspath(p):
            continue
        refs[kind(q)].append(os.path.relpath(q, REPO).replace("\\", "/"))
    n = sum(1 for _ in open(p, encoding="utf-8", errors="replace"))
    rows.append((rel, n, refs))

print("\n=== candidates (not imported, not in a test; recipes and literature tools excluded)")
for rel, n, refs in sorted(rows, key=lambda r: (bool(r[2].get("code") or r[2].get("scratch") or r[2].get("shared/other")), -r[1])):
    summary = ", ".join(f"{k} {len(v)}" for k, v in sorted(refs.items())) or "no reference anywhere"
    print(f"{n:5d}  {rel}  |  {summary}")
    for k in ("code", "scratch", "shared/other", "living doc"):
        for q in refs.get(k, [])[:3]:
            print(f"           {k}: {q}")
