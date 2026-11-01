"""scratch: the footprint-share helper of autogt/coverage.py on the stored v5 GT against the study's per-object csv (BEV chat, 18700 objects).
Run: wsl bash autogt/run_nice.sh <this file> [store folder of the signature, default data/autogt/15efbde2d1]"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
from autogt import coverage as C
from autogt.cli import load_keyframe
from autogt.grid import GridSpec

store = sys.argv[1] if len(sys.argv) > 1 else "data/autogt/15efbde2d1"
g = GridSpec(res=0.5)
rows = defaultdict(list)
for r in csv.DictReader(open("bev_fusion/temp/coverage/footprint_share.csv")):
    if r["val"] == "1":
        rows[r["sample"]].append(r)
toks = {f[:8]: f[:-4] for f in os.listdir(store) if f.endswith(".npz") and ".tmp" not in f}
print("frames in the csv (val):", len(rows), "of them in the store:", sum(k in toks for k in rows))
diffs, n_obj, miss = [], 0, 0
f_mine, f_csv, grp = [], [], []
for k, rs in rows.items():
    if k not in toks:
        continue
    gt = load_keyframe(store, toks[k])
    fus = (gt["camera"] > 0) & (gt["occluded"] == 0) & (gt["radar"] > 0)
    for r in rs:
        cx, cy = float(r["cx"]), float(r["cy"])
        cand = [b for b in gt["meta"]["boxes"] if b["group"] == r["group"] and abs(b["center"][0] - cx) < 0.02 and abs(b["center"][1] - cy) < 0.02]
        if len(cand) != 1:
            miss += 1
            continue
        f = C.footprint_share(cand[0]["corners"], fus, g)
        f_mine.append(f)
        f_csv.append(float(r["f_fus"]))
        grp.append(r["group"])
f_mine, f_csv, grp = np.array(f_mine), np.array(f_csv), np.array(grp)
d = np.abs(f_mine - f_csv)
print(f"objects matched {len(f_mine)}, not matched {miss}; |mine - study|: mean {d.mean():.4f}, max {d.max():.3f}, > 0.05: {(d > 0.05).sum()}, > 0.1: {(d > 0.1).sum()}")
for name, f in (("study", f_csv), ("mine", f_mine)):
    print(f"{name}: all {len(f)}, f = 0: {(f == 0).sum()}, f >= 0.5: {(f >= 0.5).sum()}, 0.1 <= f < 0.5: {((f >= 0.1) & (f < 0.5)).sum()}, "
          f"0 < f < 0.1: {((f > 0) & (f < 0.1)).sum()}, f = 1: {(f == 1).sum()}")
for gname in ("vehicle", "pedestrian", "two_wheeler"):
    m = grp == gname
    print(f"  {gname}: {m.sum()}, max diff {d[m].max():.3f}, mean {d[m].mean():.4f}")
