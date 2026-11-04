"""scratch: which bev_fusion paper notes (bev_fusion/literature/papers/*.md) does nothing live cite?
A note counts as cited when its file stem or its arXiv id appears in a tracked text file OUTSIDE bev_fusion/literature/, not counting plan/plan.md and plan/radar_encoding.md (the two documents that cite almost everything and are to be trimmed),
the documentation-review records of today and temp/ scratch.
    wsl python3 bev_fusion/temp/t16/literature_delete_list.py [--write-list]     (from the repository root)"""
import os
import re
import subprocess
import sys

files = [f for f in subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split("\n") if f]
notes = [f for f in files if f.startswith("bev_fusion/literature/papers/") and f.endswith(".md")]
SKIP = ("bev_fusion/literature/", "bev_fusion/plan/plan.md", "bev_fusion/plan/radar_encoding.md", "bev_fusion/records/doc_review_2026-11-03", "bev_fusion/temp/", "bev_fusion_lane_center/")
citing = []
for f in files:
    if f.startswith(SKIP) or not f.endswith((".md", ".py", ".json", ".csv", ".sh", ".txt", ".ps1", ".yml")):
        continue
    if os.path.exists(f) and os.path.getsize(f) < 3_000_000:
        citing.append((f, open(f, encoding="utf-8", errors="replace").read()))
print("notes", len(notes), "citing files", len(citing))
keep, delete = [], []
for n in notes:
    stem = os.path.basename(n)[:-3]
    ids = re.findall(r"\d{4}\.\d{4,5}", stem)
    needles = [stem] + ids
    hit = [f for f, t in citing if any(x in t for x in needles)]
    (keep if hit else delete).append((n, hit[:3]))
print("cited by live files (keep):", len(keep), "| cited by nothing live (delete):", len(delete))
for n, h in keep:
    print("  keep", os.path.basename(n), "<-", [os.path.basename(x) for x in h])
if "--write-list" in sys.argv:
    open("bev_fusion/temp/t16/literature_delete_list.txt", "w").write("\n".join(n for n, _ in delete) + "\n")
    print("list written:", len(delete))
