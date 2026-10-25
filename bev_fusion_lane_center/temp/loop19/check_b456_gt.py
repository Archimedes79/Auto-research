"""Post-check of the b456 ground truth (D7 steps 5-6, run E235.gt): every artefact has one row per token, and its statistics look like b123's.

    wsl bash tools/env/_py14.sh temp/loop19/check_b456_gt.py

Rows: graph_gt, route_gt, route_pts, sd (pickles), poses, gtK, dir, route_heat, gt_dist_exact, gt_pts_exact must all have len(tokens) rows.
Statistics per part, b456 next to b123: lane pieces and edges per frame, share of frames with a route, mean route pieces, mean exact distance,
share of direction pixels set, share of route-heat pixels > 0.5, frames with a previous frame. Gate (the run's rule): row counts exact, the route
share within 5 points of b123, every other statistic within 25 % of b123 (a different city mix may move them; a larger gap is looked at).
"""
import json
import os
import pickle

import numpy as np

H = os.path.expanduser("~")
DIRS = {"b123": (f"{H}/exp_gcar_b123", f"{H}/exp_crn_b123/feats"), "b456": (f"{H}/exp_gcar_b456", f"{H}/exp_gcar_b456/feats")}


def stats(tag, part):
    d, f = DIRS[tag]
    n = len(json.load(open(f"{d}/feats/{part}_tokens.json")))
    G = pickle.load(open(f"{d}/graph_gt_{part}.pkl", "rb"))
    RT = pickle.load(open(f"{d}/route_gt_{part}.pkl", "rb"))
    RP = pickle.load(open(f"{d}/route_pts_{part}.pkl", "rb"))
    SD = pickle.load(open(f"{d}/sd_{part}.pkl", "rb"))
    P = np.load(f"{d}/poses_{part}.npz")
    K = np.load(f"{d}/gtK_{part}.npz")
    DIR = np.load(f"{d}/dir_{part}.npy", mmap_mode="r")
    HEAT = np.load(f"{d}/route_heat_{part}.npy", mmap_mode="r")
    DIST = np.load(f"{f}/{part}_gt_dist_exact.npy", mmap_mode="r")
    PTS = np.load(f"{f}/{part}_gt_pts_exact.npy", allow_pickle=True)
    rows = dict(tokens=n, graph_gt=len(G), route_gt=len(RT), route_pts=len(RP), sd=len(SD), poses=len(P["t"]), gtK=len(K["valid"]), dir=len(DIR),
                route_heat=len(HEAT), gt_dist_exact=len(DIST), gt_pts_exact=len(PTS))
    idx = np.linspace(0, n - 1, min(n, 400)).astype(int)  # a sample of frames for the raster statistics
    s = dict(
        pieces=np.mean([len(g["pts"]) for g in G]),
        edges=np.mean([len(g["edges"]) for g in G]),
        route_share=100.0 * np.mean([r is not None for r in RT]),
        route_pieces=np.mean([len(r["route"]) for r in RT if r]),
        sd_share=100.0 * np.mean([x is not None for x in SD]),
        prev_share=100.0 * np.mean(P["prev"] >= 0),
        dist_mean=float(np.asarray(DIST[idx], np.float32).mean()),
        dir_set=100.0 * float((np.abs(np.asarray(DIR[idx], np.float32)).sum(1) > 0).mean()),
        heat_set=100.0 * float((np.asarray(HEAT[idx]) > 127).mean()),
        gt_pts=np.mean([len(p) for p in PTS]),
    )
    return rows, s


ok = True
for part in ("train", "val"):
    r0, s0 = stats("b123", part)
    r1, s1 = stats("b456", part)
    bad = {k: v for k, v in r1.items() if v != r1["tokens"]}
    ok &= not bad
    print(f"{part}: b456 rows {r1['tokens']} in every artefact" if not bad else f"{part}: ROW COUNT MISMATCH {bad} (tokens {r1['tokens']})")
    for k in s0:
        a, b = s0[k], s1[k]
        if k.endswith("_share"):
            good = abs(a - b) <= 5.0 if k == "route_share" else True
            gap = f"{b - a:+.1f} points"
        else:
            good = abs(b - a) <= 0.25 * abs(a)
            gap = f"{100 * (b - a) / a:+.0f} %"
        ok &= good
        print(f"  {k:13} b123 {a:9.3f}   b456 {b:9.3f}   ({gap}) {'ok' if good else 'LOOK'}")
print("PASS" if ok else "FAIL (or LOOK: a statistic outside its band)")
