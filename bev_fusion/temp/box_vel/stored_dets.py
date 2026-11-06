"""t11 box velocity, step 0: the velocities of the stored detections of every evaluated run (runs/<run>/eval/scene_records*.pkl).

Per run and dt: detections, max speed, the number of distinct (vx, vy) values (1 = the output is one constant, the branch's bias, at every
detected cell: a dead hidden layer), and the share of matched GT vehicles moving faster than 1 m/s.
Run (from bev_fusion/): ~/miniconda3/envs/gcar/bin/python temp/box_vel/stored_dets.py [runs ...] > temp/box_vel/out/stored_dets.txt
"""
import glob
import os
import pickle
import sys

import numpy as np


def scan(path, dts=(0.0, 0.5)):
    d = pickle.load(open(path, "rb"))
    out = {}
    for dt in dts:
        v, gv = [], []
        for sc, by_dt in d["det"].items():
            for dets, gts in by_dt.get(dt, []):
                v += [(b["vx"], b["vy"]) for b in dets if b["group"] == "vehicle"]
                gv += [np.hypot(b["vx"], b["vy"]) for b in gts if b["group"] == "vehicle"]
        if not v:
            continue
        v = np.array(v)
        sp = np.hypot(v[:, 0], v[:, 1])
        out[dt] = (len(v), float(sp.max()), len({(round(a, 9), round(b, 9)) for a, b in v}), float(np.mean(np.array(gv) > 1.0)))
    return out


if __name__ == "__main__":
    runs = sys.argv[1:] or sorted(glob.glob("runs/*"))
    print(f"{'run / pickle':58s} {'dt':>5s} {'n_det':>7s} {'max |v|':>9s} {'distinct v':>10s} {'GT >1 m/s':>9s}")
    for r in runs:
        for p in sorted(glob.glob(os.path.join(r, "eval", "scene_records*.pkl"))):
            base = os.path.basename(p)
            if "ctl_" in base or "legacy" in base or "_old" in base:
                continue
            try:
                res = scan(p)
            except Exception as e:                                # noqa: BLE001
                print(f"{p:58s} unreadable: {type(e).__name__}")
                continue
            for dt, (n, mx, nd, mov) in res.items():
                print(f"{os.path.relpath(p, 'runs'):58s} {dt:5.1f} {n:7d} {mx:9.4f} {nd:10d} {mov:9.2f}")
