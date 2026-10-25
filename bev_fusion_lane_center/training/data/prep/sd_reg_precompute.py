"""E78: registration of the (degraded, pose-disturbed) SD map to the extracted lane graph (graph-ICP of assoc_lib), precomputed like reg_precompute.py for the HD map.
Same random draws as route_head3.EvalSet / build_maps_sd: rng = RandomState(crc32('val<row>') + 17); d = randn(2) sigma; yaw = randn() 2 sigma deg; degrade_sd(sd, rng, drop=0).
Output ~/exp_gcar_b123/reg_sd_val.pkl = {sigma: {row: (R, t, score)}}  (x_map = R x_ex + t)."""
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
import sd_lib as SL  # noqa: E402

HOME = os.path.expanduser("~")
D = f"{HOME}/exp_gcar_b123"
OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
SIGMAS = (0.5, 1.0, 2.0, 3.0)
_G = {}


def one(args):
    k, row = args
    ex, sd = _G["ex"][k], _G["sd"][k]
    res = {}
    EV, ET, _, _ = AL.vertices(ex["pts"])
    for s in SIGMAS:
        rng = np.random.RandomState((zlib.crc32(f"val{row}".encode()) + 17) % (2 ** 32))
        d = rng.randn(2) * s
        R = AL.rot2(rng.randn() * 2.0 * s)
        if sd is None:
            res[s] = (np.eye(2), np.zeros(2), 0.0)
            continue
        dg = SL.degrade_sd(sd, rng, 0.5, 2.0, 0.1, 0.0)
        mp_, _ = SL.as_graph(dg)
        MV, MT, _, _ = AL.vertices([p @ R.T + d for p in mp_["pieces"]])
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
    SD = pickle.load(open(f"{D}/sd_val.pkl", "rb"))
    out = {s: {} for s in SIGMAS}
    for st in a.sets.split(","):
        d = pickle.load(open(f"{OUT}/{a.graphs}_{st}.pkl", "rb"))
        ids = [int(i) for i in d["ids"]]
        _G["ex"], _G["sd"] = d["graphs"], [SD[i] for i in ids]
        t0 = time.time()
        with mp.get_context("fork").Pool(a.workers) as pool:
            rs = pool.map(one, list(zip(range(len(ids)), ids)), chunksize=4)
        for row, res in rs:
            for s in SIGMAS:
                out[s][row] = res[s]
        print(st, len(ids), f"{time.time() - t0:.0f} s", flush=True)
    pickle.dump(out, open(f"{D}/reg_sd_val.pkl", "wb"))
    print("saved", {s: len(v) for s, v in out.items()})


if __name__ == "__main__":
    main()
