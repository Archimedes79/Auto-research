"""One-off (2026-10-25): move everything of the front-camera + radar BEV question out of bev_fusion_lane_center into bev_fusion.

Moves: the paper notes and papers.csv rows of the F4 / polar scans (ids from the scout tables; only files git does not track),
the code-source survey, the plan draft, the scout tables and the one-off scripts of this question. Nothing is deleted.
Run from the repository root: wsl python3 bev_fusion/temp/setup/move_from_lcr.py
"""
import csv
import glob
import os
import shutil
import subprocess

LCR, BF = "bev_fusion_lane_center", "bev_fusion"
tables = sorted(glob.glob(f"{LCR}/temp/loop19/f4_scout_tables/*.csv"))
ids = {"NUSC_RADAR"}
for t in tables:
    with open(t, newline="", encoding="utf-8") as f:
        ids |= {r["id"].strip() for r in csv.DictReader(f) if r.get("id")}
tracked = set(subprocess.run(["git", "ls-files", f"{LCR}/literature"], capture_output=True, text=True).stdout.split())

for d in ("literature/papers", "literature/code", "experiment_execution/plans", "temp/scout_tables", "temp/setup"):
    os.makedirs(f"{BF}/{d}", exist_ok=True)

moved = 0
for p in sorted(glob.glob(f"{LCR}/literature/papers/*.md")):
    name = os.path.basename(p)
    if name.split("_")[0] in ids and p.replace(os.sep, "/") not in tracked:
        shutil.move(p, f"{BF}/literature/papers/{name}")
        moved += 1
print("notes moved:", moved)

src = f"{LCR}/literature/papers.csv"
with open(src, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fields, rows = reader.fieldnames, list(reader)
keep = [r for r in rows if r["id"] not in ids]
out = [r for r in rows if r["id"] in ids]
with open(src, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, lineterminator="\r\n")
    w.writeheader()
    w.writerows(keep)
with open(f"{BF}/literature/papers.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(out)
print("rows moved:", len(out), "rows kept:", len(keep))

for a, b in [
    (f"{LCR}/literature/code/front_bev_code_sources.md", f"{BF}/literature/code/front_bev_code_sources.md"),
    (f"{LCR}/experiment_execution/plans/front_bev_plan.md", f"{BF}/experiment_execution/plans/plan_draft2.md"),
]:
    shutil.move(a, b)
for t in tables:
    shutil.move(t, f"{BF}/temp/scout_tables/{os.path.basename(t)}")
for s in ("cam_front_geometry.py", "merge_f4_papers.py", "fix_f4_axes.py", "fix_extract_prefix.py", "mark_parts_elsewhere.sh"):
    shutil.move(f"{LCR}/temp/loop19/{s}", f"{BF}/temp/setup/{s}")
print("done")
