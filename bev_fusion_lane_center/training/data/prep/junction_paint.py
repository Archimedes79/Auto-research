"""Painted junction evidence of a keyframe in the lane frame: the stop lines (with their type) and the pedestrian crossings of the nuScenes map expansion (E285; the lane GT of GtGenerator does not carry them, D82: lane-side derived data).

    wsl bash tools/env/_py14.sh junction_paint.py make --split val_clean [--limit N]      writes `<feature dir>/paint/<sample token>.pkl` for the frames of the split (VAL / TRAIN only, never TEST)
    wsl bash tools/env/_py14.sh junction_paint.py check                               a few stored frames: counts, the stop lines of the junction frames
    paint = load(token)                                                              -> dict(stop_line=[dict(poly (n, 2) float32, type int)], ped_crossing=[poly (n, 2)]), x right, y forward, |x|, |y| <= 50 m

The frame and the crop are the lane GT's own (`GtGenerator.lane_gt.LaneGT.pose`, `lane_lines.crop_areas`, frame "lane"), so a polygon lies where the lane GT lies. Stop line types: TURN_STOP 0, PED_CROSSING 1, TRAFFIC_LIGHT 2, STOP_SIGN 3, YIELD 4.
"""
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for sub in ("training/data", "training/grid", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))

TYPES = ("TURN_STOP", "PED_CROSSING", "TRAFFIC_LIGHT", "STOP_SIGN", "YIELD")
NUSC = os.path.expanduser("~/data/nuscenes_full")


def paint_dir():
    import p16_data as PD
    return os.path.join(PD.D, "paint")


def load(token):
    with open(os.path.join(paint_dir(), token + ".pkl"), "rb") as f:
        return pickle.load(f)


def _chunks(nmap, layer):
    from GtGenerator import lane_lines as LL
    items = [(rec["token"], nmap.extract_polygon(rec["polygon_token"])) for rec in getattr(nmap, layer)]
    return LL.polygon_chunks(items), {rec["token"]: rec for rec in getattr(nmap, layer)}


def make(split, limit=None):
    import lane_gt_data as LD
    from GtGenerator import lane_gt as LGT
    from GtGenerator import lane_lines as LL
    from nuscenes.map_expansion.map_api import NuScenesMap
    ds = LD.QData(split, pixels=False)
    toks = [ds.tokens[int(ds.bd.rows[k])] for k in ds.ks]
    if limit:
        toks = toks[:limit]
    out = paint_dir()
    os.makedirs(out, exist_ok=True)
    todo = [t for t in toks if not os.path.exists(os.path.join(out, t + ".pkl"))]
    print(f"{split}: {len(toks)} frames, {len(todo)} to make into {out}", flush=True)
    lg = LGT.LaneGT(NUSC, "v1.0-trainval")
    maps, t0 = {}, time.time()
    for n, tok in enumerate(todo):
        sample = lg.nusc.get("sample", tok)
        loc = lg.nusc.get("log", lg.nusc.get("scene", sample["scene_token"])["log_token"])["location"]
        if loc not in maps:
            nmap = NuScenesMap(dataroot=NUSC, map_name=loc)
            (sc, sr), (pc, pr) = _chunks(nmap, "stop_line"), _chunks(nmap, "ped_crossing")
            maps[loc] = (sc, sr, pc)
        sc, sr, pc = maps[loc]
        T = lg.pose(tok)
        stops = []
        for a in LL.crop_areas(sc, T, half=50.0, frame="lane", merge=False, min_area=0.0):
            stops.append(dict(poly=np.asarray(a["exterior"], np.float32), type=TYPES.index(sr[a["id"]]["stop_line_type"])))
        peds = [np.asarray(a["exterior"], np.float32) for a in LL.crop_areas(pc, T, half=50.0, frame="lane", merge=False, min_area=0.0)]
        tmp = os.path.join(out, tok + f".{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(dict(stop_line=stops, ped_crossing=peds), f, protocol=4)
        os.replace(tmp, os.path.join(out, tok + ".pkl"))
        if n % 200 == 0:
            print(n, round(time.time() - t0, 1), flush=True)
    print("done", len(todo), round(time.time() - t0, 1), "s")


def check():
    d = paint_dir()
    names = sorted(os.listdir(d))[:300]
    ns = np.array([len(load(n[:-4])["stop_line"]) for n in names])
    npd = np.array([len(load(n[:-4])["ped_crossing"]) for n in names])
    print(f"{len(os.listdir(d))} files; of {len(names)}: stop lines median {np.median(ns)} max {ns.max()}, crossings median {np.median(npd)} max {npd.max()}, frames without a stop line {(ns == 0).mean():.2f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("make", "check"))
    ap.add_argument("--split", default="val_clean", choices=("train", "val_clean"))
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    make(a.split, a.limit) if a.cmd == "make" else check()
