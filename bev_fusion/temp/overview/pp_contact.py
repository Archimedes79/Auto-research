"""scratch: contact sheet of the 6 frames (distinct scenes) with the most FALSE predicted lane_divider cells (5-30 m, fov, image): the front image with the predicted cells projected on the flat
ground (TRUE = a GT lane_divider cell within 0.5 m: green, FALSE: red), and next to it the same cells in the bird's-eye grid (GT lane_divider cells grey, map intersections shaded blue).
-> temp/overview/precision_paint_contact.png"""
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pp_precision as PP  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d = np.load(os.path.join(HERE, "precision_paint_cells.npz"))
frames = {f["i"]: f for f in json.load(open(os.path.join(HERE, "rp_frames.json")))}
fi, dist = d["fi"].astype(int), d["dist"]
false_n = {i: int(((fi == i) & (dist > 0.5 + 1e-9)).sum()) for i in np.unique(fi)}
order = sorted(false_n, key=lambda i: -false_n[i])
pick, seen = [], set()
for i in order:
    sc = frames[i]["scene"]
    if sc not in seen:
        seen.add(sc)
        pick.append(i)
    if len(pick) == 6:
        break
maps = PP.Maps()
fig, ax = plt.subplots(len(pick), 2, figsize=(13, 3.6 * len(pick)), gridspec_kw={"width_ratios": [2.4, 1]})
for r, i in enumerate(pick):
    f = frames[i]
    g = np.load(os.path.join(PP.ROOT, "data", "cache", "gt", f["file"]), allow_pickle=True)["gt"].item()["cur"]
    meta = g["meta"]
    im = Image.open(os.path.join(PP.FROOT, meta["cam_file"])).convert("RGB")
    K, T = np.asarray(meta["K"], float), np.asarray(meta["T_c2e"], float)
    m = fi == i
    x, y, tr = d["x"][m], d["y"][m], dist[m] <= 0.5 + 1e-9
    Pc = np.c_[x, y, np.zeros(len(x)), np.ones(len(x))] @ np.linalg.inv(T).T
    u, v = K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2], K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2]
    a = ax[r, 0]
    a.imshow(im)
    a.scatter(u[~tr], v[~tr], s=7, c="red", label=f"FALSE ({int((~tr).sum())})", linewidths=0)
    a.scatter(u[tr], v[tr], s=7, c="lime", label=f"TRUE ({int(tr.sum())})", linewidths=0)
    a.set_ylim(min(im.size[1], v.max() + 50), max(0, v.min() - 60))
    a.set_xlim(0, im.size[0])
    a.set_title(f"{f['scene']}  frame {i}  precision of the shown cells {tr.mean():.2f}", fontsize=9)
    a.legend(loc="upper right", fontsize=8, markerscale=2)
    a.axis("off")
    b = ax[r, 1]
    inter = maps.raster(meta["scene"], meta["T_e2g"])
    Gl = np.asarray(g["lane_divider"], np.float32) > 0.5
    ext = (PP.Y0 + PP.NY * PP.RES, PP.Y0, 0.0, PP.NX * PP.RES)                       # left = +y: as seen from the car
    b.imshow(np.where(inter, 0.35, 1.0)[:, ::-1], cmap="Blues_r", vmin=0, vmax=1, extent=(PP.Y0 + PP.NY * PP.RES, PP.Y0, 0.0, PP.NX * PP.RES), aspect="equal", origin="lower")
    gx, gy = np.nonzero(Gl)
    b.scatter(PP.Y0 + (gy + 0.5) * PP.RES, (gx + 0.5) * PP.RES, s=5, c="gray", marker="s", linewidths=0, label="GT lane_divider")
    b.scatter(y[~tr], x[~tr], s=8, c="red", linewidths=0)
    b.scatter(y[tr], x[tr], s=8, c="lime", linewidths=0)
    b.set_xlim(14, -14)
    b.set_ylim(0, 34)
    b.set_title("bird's-eye (left = +y), blue = map intersection", fontsize=9)
    b.legend(loc="upper right", fontsize=7)
fig.tight_layout()
out = os.path.join(HERE, "precision_paint_contact.png")
fig.savefig(out, dpi=75)
print("saved", out, "frames", pick, [false_n[i] for i in pick])
