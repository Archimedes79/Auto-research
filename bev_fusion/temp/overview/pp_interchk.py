"""scratch: is the intersection raster aligned? (a) share of intersection cells (fov, 5-30 m, image) that are in the GT drivable area, (b) share of GT lane_divider / road_divider / ped_crossing /
stop_line cells inside an intersection polygon, (c) the same for the GT drivable cells. If the raster were shifted or rotated, (b) would sit near the base rate."""
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pp_precision as PP  # noqa: E402

frames = json.load(open(os.path.join(HERE, "rp_frames.json")))
maps = PP.Maps()
acc = {}
for k, f in enumerate(frames):
    p = os.path.join(PP.ROOT, "data", "cache", "gt", f["file"])
    if not os.path.exists(os.path.join(HERE, "rp_out", f"{f['i']}.npz")):
        continue
    g = np.load(p, allow_pickle=True)["gt"].item()["cur"]
    meta = g["meta"]
    inter = maps.raster(meta["scene"], meta["T_e2g"])
    im = np.asarray(Image.open(os.path.join(PP.FROOT, meta["cam_file"])).convert("L"))
    sel = (np.asarray(g["camera"], np.float32) > 0) & (PP.XC >= 5) & (PP.XC < 30) & PP.centre_valid(meta, im.shape)
    lay = lambda n: (np.asarray(g[n], np.float32) > 0.5) & sel      # noqa: E731
    for name, m in (("all", sel), ("drivable", lay("drivable_area")), ("lane_divider", lay("lane_divider")), ("road_divider", lay("road_divider")),
                    ("ped_crossing", lay("ped_crossing")), ("stop_line", lay("stop_line")), ("walkway", lay("walkway")), ("centreline", lay("centreline"))):
        a = acc.setdefault(name, [0, 0])
        a[0] += int((m & inter).sum())
        a[1] += int(m.sum())
    a = acc.setdefault("intersection cells that are drivable", [0, 0])
    a[0] += int((inter & sel & (np.asarray(g["drivable_area"], np.float32) > 0.5)).sum())
    a[1] += int((inter & sel).sum())
print("share inside an intersection polygon (fov, 5-30 m, image), 246 frames:")
for k, (a, b) in acc.items():
    print(f"  {k:40s} {a / max(b, 1):.3f}  ({b} cells)")
