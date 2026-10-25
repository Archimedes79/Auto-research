"""E74: registration front end for the end-to-end head - precomputation.
For the TEST rows (and VAL rows) and the map-pose noise levels sigma in {0.5, 1, 2, 3} m the map of the sample is disturbed exactly as route_head2.EvalSet does it
(rng = RandomState(crc32('val<row>') + 17); translation d = randn(2) * sigma, yaw = randn() * 2 sigma degrees) and registered to the extracted lane graph of the loop-4 system
(out/loop2/e44_cons_p/graphs_<set>.pkl, no map input) with graph-ICP (assoc_lib.register): x_map = R x_ex + t.
Output: ~/exp_gcar_b123/reg_val.pkl = {sigma: {row: (R, t, score)}}   (rows index the val list: VAL + TEST rows)
Usage (gcar):  reg_precompute.py [--sets test,val] [--workers 6]"""
import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc_lib as AL  # noqa: E402

HOME = os.path.expanduser("~")
D = f"{HOME}/exp_gcar_b123"
OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
SIGMAS = (0.5, 1.0, 2.0, 3.0)
_G = {}


def one(args):
    k, row = args
    ex, g = _G["ex"][k], _G["gt"][k]
    res = {}
    EV, ET, _, _ = AL.vertices(ex["pts"])
    for s in SIGMAS:
        rng = np.random.RandomState((zlib.crc32(f"val{row}".encode()) + 17) % (2 ** 32))
        d = rng.randn(2) * s
        R = AL.rot2(rng.randn() * 2.0 * s)
        pieces = [np.asarray(p, np.float64) @ R.T + d for p in g["pts"] if len(p) >= 2]
        MV, MT, _, _ = AL.vertices(pieces)
        if len(EV) < 5 or len(MV) < 5:
            res[s] = (np.eye(2), np.zeros(2), 0.0)
            continue
        Rr, tr, sc = AL.register(MV, MT, EV, ET)
        res[s] = (Rr, tr, sc)
    return row, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="test,val")
    ap.add_argument("--graphs", default="e44_cons_p/graphs")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    GT = pickle.load(open(f"{D}/graph_gt_val.pkl", "rb"))
    RT = pickle.load(open(f"{D}/route_gt_val.pkl", "rb"))
    out = {s: {} for s in SIGMAS}
    for st in a.sets.split(","):
        t0 = time.time()
        d = pickle.load(open(f"{OUT}/{a.graphs}_{st}.pkl", "rb"))
        ids = [int(i) for i in d["ids"]]
        keep = [(k, i) for k, i in enumerate(ids) if RT[i] is not None]
        _G["ex"], _G["gt"] = d["graphs"], [GT[i] for i in ids]
        with mp.get_context("fork").Pool(a.workers) as pool:
            rs = pool.map(one, keep, chunksize=4)
        for row, res in rs:
            for s in SIGMAS:
                out[s][row] = res[s]
        print(f"[{st}] {len(rs)} rows registered in {time.time() - t0:.0f} s", flush=True)
    pickle.dump(out, open(f"{D}/reg_val.pkl", "wb"))
    for s in SIGMAS:
        sc = [v[2] for v in out[s].values()]
        print(f"sigma {s}: {len(sc)} rows, median registration score {np.median(sc):.3f}", flush=True)


if __name__ == "__main__":
    main()
