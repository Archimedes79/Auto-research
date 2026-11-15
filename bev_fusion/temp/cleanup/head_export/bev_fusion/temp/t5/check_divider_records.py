"""t5 finding check: does the GT drop dividers that cross the patch without a node inside it?

For val frames of the probe: the divider records whose bounding box meets the GT patch, split into (a) records the devkit's
is_record_in_patch(..., 'intersect') keeps and (b) records whose LINE GEOMETRY intersects the patch (shapely) but the devkit drops.
Prints per frame and in total the length of divider inside the patch for both. Run from the bev_fusion folder.
"""
import json
import os
import sys

import numpy as np
from shapely.geometry import LineString, box as sbox

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "training"))
from nuscenes.map_expansion.map_api import NuScenesMap   # noqa: E402

DATA = os.path.expanduser("~/data/nuscenes_full")
meta_dir = os.path.join(DATA, "v1.0-trainval")
scenes = {s["name"]: s for s in json.load(open(os.path.join(meta_dir, "scene.json")))}
logs = {l["token"]: l for l in json.load(open(os.path.join(meta_dir, "log.json")))}
frames = json.load(open(os.path.join(ROOT, "temp", "t5", "frames.json")))["val"]
n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
pick = [frames[i] for i in np.linspace(0, len(frames) - 1, n).round().astype(int)]
maps, tot_keep, tot_drop, frames_hit = {}, 0.0, 0.0, 0
for sc, tok, f in pick:
    meta = np.load(f, allow_pickle=True)["gt"].item()["cur"]["meta"]
    T = np.asarray(meta["T_e2g"])
    corners = np.array([[0, -26], [0, 26], [60, -26], [60, 26]], float)         # the GT grid (lift_pd.Grid defaults)
    cg = (np.c_[corners, np.zeros(4), np.ones(4)] @ T.T)[:, :2]
    patch = (cg[:, 0].min() - 5, cg[:, 1].min() - 5, cg[:, 0].max() + 5, cg[:, 1].max() + 5)
    loc = logs[scenes[sc]["log_token"]]["location"]
    nm = maps.setdefault(loc, NuScenesMap(DATA, loc))
    # the grid window itself in world coordinates (the patch is its bounding box)
    win = LineString(np.r_[cg[[0, 2, 3, 1]], cg[[0]]]).convex_hull
    keep, drop = 0.0, 0.0
    for layer in ("lane_divider", "road_divider"):
        for r in getattr(nm, layer):
            line = nm.extract_line(r["line_token"])
            if line.is_empty or not line.intersects(sbox(*patch)):
                continue
            inside = line.intersection(win).length
            if inside <= 0:
                continue
            if nm.explorer.is_record_in_patch(layer, r["token"], patch, "intersect"):
                keep += inside
            else:
                drop += inside
    tot_keep += keep
    tot_drop += drop
    frames_hit += drop > 1.0
    print(f"{sc} {tok[:8]}: divider length in the grid window kept {keep:6.1f} m, dropped {drop:6.1f} m")
print(f"TOTAL over {len(pick)} val frames: kept {tot_keep:.0f} m, dropped {tot_drop:.0f} m "
      f"({100 * tot_drop / max(tot_keep + tot_drop, 1e-9):.0f} % of the divider length lost); frames losing > 1 m: {frames_hit}")
