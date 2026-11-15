"""D91 scratch: what does ONE radar sweep carry against the 0.5 s of sweeps the network gets today?

python temp/d91/radar_one_sweep.py [--version v1.0-mini --root ~/data/nuscenes --age 0.09]

For every keyframe (CAM_FRONT time) it reads the radar the way the dataset does (FrontRadarDataset._radar: all sweeps of the five radars within 0.5 s,
moved into the ego frame of the camera time) and counts, per annotated vehicle / pedestrian in the grid (0 <= x < 60 m, |y| < 26 m, observed = lidar
or radar points in the box), the points inside its footprint (+ margin) for the newest sweep only (age <= --age) against all sweeps. CPU, no model.
"""
import argparse
import math
import os
import sys

import numpy as np
from pyquaternion import Quaternion

ROOT = "/mnt/c/Auto-research"
for p in (ROOT, ROOT + "/bev_fusion", ROOT + "/bev_fusion/training"):
    if p not in sys.path:
        sys.path.insert(0, p)
from nuscenes_dataset import FrontRadarDataset  # noqa: E402


def objects_in_ego(nusc, sample, cur):
    pose = nusc.get("ego_pose", cur["ego_pose_token"])
    q_inv = Quaternion(pose["rotation"]).inverse
    t = np.array(pose["translation"])
    out = []
    for tok in sample["anns"]:
        a = nusc.get("sample_annotation", tok)
        cat = a["category_name"]
        if not (cat.startswith("vehicle.") or cat.startswith("human.pedestrian")):
            continue
        if a["num_lidar_pts"] + a["num_radar_pts"] <= 0:
            continue
        c = q_inv.rotate(np.array(a["translation"]) - t)
        if not (0 <= c[0] < 60 and abs(c[1]) < 26):
            continue
        fwd = (q_inv * Quaternion(a["rotation"])).rotate(np.array([1.0, 0, 0]))
        out.append({"c": c[:2], "yaw": math.atan2(fwd[1], fwd[0]), "w": a["size"][0], "l": a["size"][1],
                    "ped": cat.startswith("human"), "rng": float(np.hypot(c[0], c[1]))})
    return out


def inside(o, x, y, margin):
    dx, dy = x - o["c"][0], y - o["c"][1]
    cs, sn = math.cos(o["yaw"]), math.sin(o["yaw"])
    u, v = dx * cs + dy * sn, -dx * sn + dy * cs
    return (np.abs(u) <= o["l"] / 2 + margin) & (np.abs(v) <= o["w"] / 2 + margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--root", default="~/data/nuscenes")
    ap.add_argument("--age", type=float, default=0.09, help="the newest sweep of each radar: age <= this (s)")
    ap.add_argument("--margin", type=float, default=1.0)
    a = ap.parse_args()
    ds = FrontRadarDataset(os.path.expanduser(a.root), a.version, with_gt=False, radar_window=0.5)
    wins = {"one sweep": a.age, "0.25 s": 0.25, "0.5 s": 0.5}
    pts = {k: [] for k in wins}
    rows = []                                                 # per object: range, ped, {window: (n points, n moving points)}
    for token in ds.samples:
        sample = ds.nusc.get("sample", token)
        cur = ds.gt.cam(token)
        r = ds._radar(sample, cur)
        x, y, dt, vr = (np.asarray(r[k], dtype=np.float64) for k in ("x", "y", "dt", "vr"))
        for k, w in wins.items():
            pts[k].append(int(((dt <= w) & (x >= 0) & (x < 60) & (np.abs(y) < 26)).sum()))
        for o in objects_in_ego(ds.nusc, sample, cur):
            row = {"rng": o["rng"], "ped": o["ped"]}
            for k, w in wins.items():
                m = inside(o, x, y, a.margin) & (dt <= w)
                row[k] = (int(m.sum()), int((m & (np.abs(vr) > 0.5)).sum()))
            rows.append(row)
    print(f"{a.version}: {len(ds.samples)} keyframes, {len(rows)} observed vehicles / pedestrians in the grid, margin {a.margin} m, one sweep = age <= {a.age} s")
    print("radar points in the grid per keyframe (mean): " + ", ".join(f"{k} {np.mean(v):.0f}" for k, v in pts.items()))

    def table(title, sel):
        s = [r for r in rows if sel(r)]
        if not s:
            return
        print(f"{title} (n = {len(s)}):  " + " | ".join(
            f"{k}: >=1 pt {np.mean([r[k][0] >= 1 for r in s]) * 100:4.1f} %, >=3 pts {np.mean([r[k][0] >= 3 for r in s]) * 100:4.1f} %, "
            f"mean {np.mean([r[k][0] for r in s]):4.1f} pts, >=1 moving-Doppler {np.mean([r[k][1] >= 1 for r in s]) * 100:4.1f} %" for k in wins))

    table("all", lambda r: True)
    table("vehicles", lambda r: not r["ped"])
    table("pedestrians", lambda r: r["ped"])
    for lo, hi in ((0, 20), (20, 40), (40, 70)):
        table(f"range {lo}-{hi} m", lambda r, lo=lo, hi=hi: lo <= r["rng"] < hi)


if __name__ == "__main__":
    main()
