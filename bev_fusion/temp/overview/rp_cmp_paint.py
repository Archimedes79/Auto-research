"""scratch: compare my stage-1 points with the stored paint check (per-frame n in paint_result.json)"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
res = json.load(open(os.path.join(ROOT, "temp", "lane_tests", "paint_out", "paint_result.json")))
fr = json.load(open(os.path.join(HERE, "rp_frames.json")))
z = np.load(os.path.join(HERE, "rp_points.npz"))
for layer in ("lane_divider", "road_divider"):
    old = {r["frame"]: r["n"] for r in res["per_frame_rows"][layer]}
    fi = z[f"{layer}/fi"]
    new = {fr[i]["file"]: int((fi == i).sum()) for i in range(len(fr))}
    print(layer, "stored frames with n>=8:", len(old), "sum n", sum(old.values()), "| mine frames >=8:", sum(1 for v in new.values() if v >= 8), "sum n over those", sum(v for v in new.values() if v >= 8))
    diffs = [(k, old[k], new.get(k)) for k in old if new.get(k) != old[k]]
    print("  frames with different n:", len(diffs), diffs[:10])
    extra = [(k, v) for k, v in new.items() if v >= 8 and k not in old]
    print("  mine >=8 but not stored:", len(extra), extra[:5])
print("paint table bins:", {k: v["points"] for k, v in res["layers"]["lane_divider"]["bins"].items()})
