"""How the complete GT set splits by what the sensors can see, on the frames of one stored evaluation (default: the 1000 val frames of t10_dual_s0).
Classes per annotated box inside the scored window (x 0 - 60 m, +-32.3 deg):
  complete      every annotated box
  observed      the current evaluation's GT (>= 1 lidar OR radar point in the box)
  camera        the centre cell is a camera cell and not hidden behind another box (the 'centre cell' primitive of D38; the footprint share comes later)
  camera+radar  camera AND >= 1 radar point in the box (nuScenes annotation field num_radar_pts, read from sample_annotation.json by the box token)
  radar only    not camera, >= 1 radar point;   neither   not camera, no radar point
Run from bev_fusion (WSL, gcar python): python temp/pitch/split_counts.py [run_dir]"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, ".")
import numpy as np  # noqa: E402

import evaluation.bev_gtcheck as G  # noqa: E402  (import paths)
from evaluation.metrics import in_front  # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "runs/t10_dual_s0"
frames = [json.loads(line)["key"] for line in open(glob.glob(os.path.join(run, "eval/suite/quant/eval_n1000/frames.jsonl"))[0])]
print(len(frames), "frames of", run)
sig = "15efbde2d1"
boxes = []                                   # (annotation token, group, centre cell, camera cell, hidden, observed)
for tok in frames:
    f = glob.glob(os.path.join(G.cache_dir(), f"{tok}_*_{sig}.npz"))
    if not f:
        continue
    g = G.load(f[0]).get("cur")
    if not g:
        continue
    cam, occ = np.asarray(g["camera"], np.float32) > 0, np.asarray(g["occluded"], np.float32) > 0
    for b in g["meta"]["boxes"]:
        cx, cy = float(b["center"][0]), float(b["center"][1])
        if not in_front({"cx": cx, "cy": cy}):
            continue
        i, j = int(cx // 0.5), int((cy + 26.0) // 0.5)
        ok = 0 <= i < cam.shape[0] and 0 <= j < cam.shape[1]
        boxes.append((b["token"], b["group"], bool(ok and cam[i, j]), bool(ok and occ[i, j]), bool(b.get("observed", True))))
print(len(boxes), "annotated boxes inside the window")
want = {b[0] for b in boxes}
radar = {}
pat = re.compile(r'"token": "([0-9a-f]{32})".{0,700}?"num_lidar_pts": (\d+),\s*"num_radar_pts": (\d+)', re.S)
path = os.path.expanduser("~/data/nuscenes_full/v1.0-trainval/sample_annotation.json")
with open(path, "r", encoding="utf-8") as fh:
    tail = ""
    while True:
        chunk = fh.read(1 << 25)
        if not chunk:
            break
        buf = tail + chunk
        for m in pat.finditer(buf):
            if m.group(1) in want:
                radar[m.group(1)] = (int(m.group(2)), int(m.group(3)))
        tail = buf[-2000:]
print(len(radar), "of", len(want), "annotation tokens found")
rows = {}
for tok, grp, cam, hid, obs in boxes:
    lid, rad = radar.get(tok, (None, None))
    c = rows.setdefault(grp, dict(complete=0, observed=0, camera=0, cam_radar=0, radar_only=0, neither=0, unknown=0, lidar_only_observed=0))
    c["complete"] += 1
    c["observed"] += obs
    if rad is None:
        c["unknown"] += 1
        continue
    seen = cam and not hid
    c["camera"] += seen
    c["cam_radar"] += seen and rad > 0
    c["radar_only"] += (not seen) and rad > 0
    c["neither"] += (not seen) and rad == 0
    c["lidar_only_observed"] += obs and rad == 0 and lid > 0
print(f"{'class':14s}" + "".join(f"{k:>14s}" for k in next(iter(rows.values()))))
for grp, c in rows.items():
    print(f"{grp:14s}" + "".join(f"{v:14d}" for v in c.values()))
for grp, c in rows.items():
    n = c["complete"]
    print(f"{grp}: camera can see {100 * c['camera'] / n:.0f} % of the complete set, camera AND radar {100 * c['cam_radar'] / n:.0f} % "
          f"({100 * c['cam_radar'] / max(c['camera'], 1):.0f} % of what the camera sees); in the current GT only because of a lidar point: {c['lidar_only_observed']}")
