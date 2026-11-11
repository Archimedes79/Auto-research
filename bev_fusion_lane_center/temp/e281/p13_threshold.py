"""E281 side check (card section 3b), CPU only: the delivered graph of baseline B when the long-flag links are read out at a plain threshold t < 0.5 (short links and nodes at 0.5), from the full dump
(`temp/e276/predfull_e273_p_s<seed>.pkl`); scored like the variants of p12_precheck.py (E273 scorer, pair-rule long precision, long_persist). Reuses the labels cached by p12 (`info_s<seed>.pkl`).
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 bash tools/env/_py14.sh temp/e281/p13_threshold.py <seed> [--rows N]
Output: temp/e281/p13_s<seed>.json. A diagnostic by hand, in-sample on VAL 873, no verdict."""
import json
import os
import pickle
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402

SEED = int(sys.argv[1])
N_ROWS = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 0
OUT = os.path.join(ROOT, "temp", "e281")
THR, TOL, BOXM = 0.5, 2.0, 45.0
LONG_THR = (0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1)
POSES = os.path.expanduser("~/exp_gcar_b123/poses_val.npz")

frames = QE.val_frames()
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{SEED}.pkl"), "rb"))
assert [d["row"] for d in dump] == [f.row for f in frames]
INFO = pickle.load(open(os.path.join(OUT, f"info_s{SEED}.pkl"), "rb"))
if N_ROWS:
    frames, dump, INFO = frames[:N_ROWS], dump[:N_ROWS], INFO[:N_ROWS]
rows = [f.row for f in frames]
by_row = {r: i for i, r in enumerate(rows)}
poses = np.load(POSES)
T, R, PREV = poses["t"], poses["R"], poses["prev"]
prev_of = {by_row[r]: by_row[int(PREV[r])] for r in rows if int(PREV[r]) >= 0 and int(PREV[r]) in by_row}


def warp(p, a, b):                                    # "the pose acts on the lane frame" (found from the GT nodes in E276 and again in p12: 0.49 m)
    return (np.asarray(p, float) @ R[rows[a]].T + T[rows[a]] - T[rows[b]]) @ R[rows[b]]


def persist_one(A, B, C, D):
    hit = np.zeros(len(A), bool)
    if not len(C):
        return hit
    tc = cKDTree(C)
    for i, cand in enumerate(tc.query_ball_point(A, TOL)):
        if cand and (np.linalg.norm(D[cand] - B[i], axis=1) <= TOL).any():
            hit[i] = True
    return hit


lgs = [np.asarray(d["link_long"], bool) for d in dump]


def delivered(d, lg, t):
    keep = d["e"] >= THR
    s, tt, p = d["link_src"], d["link_dst"], d["link_p"]
    return keep[s] & keep[tt] & (s != tt) & np.where(lg, p >= t, p >= THR)


ref_card = json.load(open(os.path.join(ROOT, "runs", "e275", f"e273_p_s{SEED}.plain_t0.5", "card.json")))["p"]["graph"]
rec = {t: [] for t in LONG_THR}
t0 = time.time()
for n, (f, d) in enumerate(zip(frames, dump)):
    cached = getattr(f, "_ref", None) is not None
    lg, p = lgs[n], d["link_p"]
    for t in LONG_THR:
        p2 = np.where(lg, np.where(p >= t, np.maximum(p, THR), np.minimum(p, 0.4999)), p)
        nd = dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"])
        if d.get("s") is not None:
            nd["s"] = d["s"]
        lk = dict(src=d["link_src"], dst=d["link_dst"], p=p2, long=lg, a=d["link_a"], b=d["link_b"])
        graph, _ = QE.readout(dict(nodes=nd, links=lk), THR, THR)
        rec[t].append(QE.graph_counts(graph, f))
    if not cached:
        f._lane = f._ref = None
    if (n + 1) % 100 == 0:
        print(f"  frame {n + 1}/{len(dump)}  {time.time() - t0:.0f} s", flush=True)
res = dict(seed=SEED, rows=N_ROWS, variants={})
for t in LONG_THR:
    g = QE.graph_numbers(rec[t])
    dl = [delivered(d, lgs[i], t) for i, d in enumerate(dump)]
    h = n_ = 0
    for b, a in prev_of.items():
        da, db = dump[a], dump[b]
        ma = dl[a] & lgs[a]
        A, B = warp(da["pos"][da["link_src"][ma]], a, b), warp(da["pos"][da["link_dst"][ma]], a, b)
        inb = (np.abs(A) < BOXM).all(1) & (np.abs(B) < BOXM).all(1)
        A, B = A[inb], B[inb]
        mb = dl[b]
        hit = persist_one(A, B, db["pos"][db["link_src"][mb]], db["pos"][db["link_dst"][mb]])
        h += int(hit.sum())
        n_ += len(hit)
    nd_ = sum(int((dl[i] & lgs[i]).sum()) for i in range(len(dump)))
    nm_ = sum(int((dl[i] & lgs[i] & INFO[i]["matched"]).sum()) for i in range(len(dump)))
    nr_ = sum(int((dl[i] & lgs[i] & INFO[i]["right"]).sum()) for i in range(len(dump)))
    res["variants"][f"{t:g}"] = dict({k: float(g[k]) for k in ("topo_F1", "topo_P", "topo_R", "junction_F1", "junction_P", "junction_R", "F1@1.0", "lanes_found", "vert100_dp")},
                                     long_persist=h / max(n_, 1), delivered_long=nd_, matched_long=nm_, right_long=nr_, long_precision_pair=nr_ / max(nm_, 1))
res["repro_0.5_vs_E275_card"] = {k: res["variants"]["0.5"][k] - ref_card[k] for k in ("topo_F1", "junction_F1", "F1@1.0", "lanes_found")}
json.dump(res, open(os.path.join(OUT, f"p13_s{SEED}" + (f"_rows{N_ROWS}" if N_ROWS else "") + ".json"), "w"), indent=1)
print("t     topo F1  junc F1  F1@1.0  lanes  prec_pair  delivered  persist")
for t, v in res["variants"].items():
    print("%-5s %.3f    %.3f    %.3f   %.3f  %.3f      %6d     %.3f" % (t, v["topo_F1"], v["junction_F1"], v["F1@1.0"], v["lanes_found"], v["long_precision_pair"], v["delivered_long"], v["long_persist"]))
print("repro:", res["repro_0.5_vs_E275_card"])
