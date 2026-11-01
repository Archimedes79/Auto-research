"""Threshold study for D38 (b): the share f of every GT object's footprint that lies inside the fusion area (camera cells x radar fan x not hidden behind another box),
from the GT cache of the signature 15efbde2d1 (GT version v5). CPU only, read only. Run from bev_fusion (WSL, gcar python):
    python temp/coverage/footprint_share.py [--val-only] [--limit N]
Writes temp/coverage/footprint_share.csv (one row per object) and prints the histograms and the sensitivity tables.

f is measured on a sample grid of 0.125 m inside the footprint polygon (the corners of the GT box, ego frame at the camera time) looked up in the 0.5 m masks; samples outside the 60 x 52 m grid count as
not observed. Three variants: f_cam (camera cells only), f_fusnoocc (camera x radar), f_fus (camera x radar x not hidden). The old rules are kept next to them: the centre cell in the camera / fusion area and
the `observed` flag (a lidar or radar point in the box) of the evaluation today."""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

SIG = "15efbde2d1"
RES, X0, Y0, NX, NY = 0.5, 0.0, -26.0, 120, 104
STEP = 0.125
NUSC = os.path.expanduser("~/data/nuscenes_full/v1.0-trainval")


def val_scene_tokens():
    from nuscenes.utils.splits import val as val_names
    sc = json.load(open(os.path.join(NUSC, "scene.json")))
    names = {s["token"]: s["name"] for s in sc}
    return {t for t, n in names.items() if n in set(val_names)}, names


def share(poly, mask):
    """mean of the mask over sample points inside the quadrilateral; points outside the grid count 0. poly (4, 2) consecutive corners."""
    c0, c1, c3 = poly[0], poly[1], poly[3]
    a, b = c1 - c0, c3 - c0
    n = max(1, int(np.ceil(np.linalg.norm(a) / STEP)))
    m = max(1, int(np.ceil(np.linalg.norm(b) / STEP)))
    u = (np.arange(n) + 0.5) / n
    v = (np.arange(m) + 0.5) / m
    pts = c0[None, None, :] + u[:, None, None] * a[None, None, :] + v[None, :, None] * b[None, None, :]
    pts = pts.reshape(-1, 2)
    i = np.floor((pts[:, 0] - X0) / RES).astype(int)
    j = np.floor((pts[:, 1] - Y0) / RES).astype(int)
    ok = (i >= 0) & (i < NX) & (j >= 0) & (j < NY)
    val = np.zeros(len(pts), np.float32)
    val[ok] = mask[i[ok], j[ok]]
    return float(val.mean()), float(ok.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-only", action="store_true", help="official val scenes only (default: all cached frames, split column kept)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    val_tok, names = val_scene_tokens()
    files = sorted(glob.glob(f"data/cache/gt/*_0.5_4_{SIG}.npz"))
    rows, nfiles = [], 0
    for fn in files:
        z = np.load(fn, allow_pickle=True)
        gt = z["gt"].item()["cur"]
        meta = gt["meta"]
        is_val = meta["scene"] in val_tok
        if a.val_only and not is_val:
            continue
        nfiles += 1
        cam = (gt["camera"].astype(np.float32) > 0.5).astype(np.float32)
        rad = (gt["radar"].astype(np.float32) > 0.5).astype(np.float32)
        occ = (gt["occluded"].astype(np.float32) > 0.5).astype(np.float32)
        fusnoocc = cam * rad
        fus = fusnoocc * (1.0 - occ)
        for b in meta["boxes"]:
            poly = np.asarray(b["corners"], float)[:, :2]
            f_cam, ing = share(poly, cam)
            f_fno, _ = share(poly, fusnoocc)
            f_fus, _ = share(poly, fus)
            cx, cy = float(b["center"][0]), float(b["center"][1])
            ci, cj = int(np.floor((cx - X0) / RES)), int(np.floor((cy - Y0) / RES))
            inside = 0 <= ci < NX and 0 <= cj < NY
            rows.append(dict(scene=names.get(meta["scene"], meta["scene"]), val=int(is_val), sample=meta["sample"][:8], group=b["group"], cx=round(cx, 2), cy=round(cy, 2),
                             rng=round(float(np.hypot(cx, cy)), 1), length=round(float(b["size"][1]), 2), width=round(float(b["size"][0]), 2),
                             visibility=b.get("visibility", ""), observed=int(bool(b.get("observed", True))), vel_known=int(bool(b.get("vel_known", True))),
                             in_grid=round(ing, 3), f_cam=round(f_cam, 3), f_fusnoocc=round(f_fno, 3), f_fus=round(f_fus, 3),
                             c_cam=int(inside and cam[ci, cj] > 0.5), c_fus=int(inside and fus[ci, cj] > 0.5)))
        if a.limit and nfiles >= a.limit:
            break
    os.makedirs("temp/coverage", exist_ok=True)
    out = "temp/coverage/footprint_share.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{nfiles} frames, {len(rows)} objects -> {out}")


if __name__ == "__main__":
    main()
