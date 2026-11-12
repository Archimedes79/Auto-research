"""E293 (scratch): how deep into the junction can points be placed? Evidence and interpolation error along the GT connectors by depth from the entry.
usage: depth_probe.py <seed> [max frames]    -> runs/e293/depth_probe_s<seed>.json

For every connector run (the connector part of a lane piece, `jgt._runs`, at least 3 points, 0.5 m apart) and every point of it at depth d (arclength from the entry, metres):
  paint        distance to the nearest painted evidence: a lane divider / road divider / boundary line of the lane GT, a stop line, a pedestrian crossing (sidecar) -> share within 1 m
  heat         B's 1 m lane-centre heat (pyramid centre1 channel 0, sigmoid, 3 x 3 maximum at the point's cell) -> share >= 0.3 and the mean
  hermite      the cubic Hermite curve from the TRUE entry to the TRUE exit of the run (tangent = chord, `qends.curve`): distance of the GT point to the curve -> share within 0.5 m / 1 m, median
Bins by depth: 0-2, 2-5, 5-10, 10-15, 15-25, 25+ m; and by the fraction of the run length."""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

R = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/data/prep", "evaluation", "tools/viz"):
    sys.path.insert(0, os.path.join(R, sub))
import jgt as J
import junction_paint as JP
import lane_gt_data as LD
import qends as QE
import qpredict as QP
from GtGenerator import lane_gt

seed = int(sys.argv[1])
maxn = int(sys.argv[2]) if len(sys.argv) > 2 else 0
m = QP.load_model(os.path.join(R, "runs", f"e273_p_s{seed}"), "cpu")
m.eval()
ds, dl = LD.loader("val_clean", bs=4, workers=2, shuffle=False, map_mode="exact", pixels=False)
BINS = [0, 2, 5, 10, 15, 25, 1e9]
FR = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
acc = {k: dict(n=0, paint=0, heat=0, heat_sum=0.0, h05=0, h1=0, herr=[]) for k in range(len(BINS) - 1)}
accf = {k: dict(n=0, paint=0, heat=0, h05=0, h1=0) for k in range(len(FR) - 1)}
runs_n = 0
ctrl = {k: dict(n=0, heat=0, paint=0, heat_sum=0.0) for k in range(len(BINS) - 1)}
t0, n = time.time(), 0
for b in dl:
    with torch.no_grad():
        P = m.pyr(b["x"].float())
    cen = P["centre1"][:, 0].float().sigmoid()
    cen = torch.nn.functional.max_pool2d(cen[:, None], 3, 1, 1)[:, 0].numpy()                      # (B, 100, 100) rows = x, columns = y, origin -50 m
    for i, tok in enumerate(b["token"]):
        g = lane_gt.load(tok)
        runs = J._runs(g)
        if not runs:
            continue
        pts = [np.asarray(l["pts"]) for k in ("lane_divider", "road_divider", "boundary") for l in g["lines"][k] if len(l["pts"]) > 1]
        paint = JP.load(tok)
        parts = []
        for a_ in pts:                                                                        # resample the lines to 0.5 m
            seg = np.linalg.norm(np.diff(a_, axis=0), axis=1)
            s = np.r_[0, np.cumsum(seg)]
            tt = np.arange(0, s[-1] + 1e-6, 0.5)
            parts.append(np.stack([np.interp(tt, s, a_[:, 0]), np.interp(tt, s, a_[:, 1])], 1))
        for sl in paint["stop_line"]:
            parts.append(np.asarray(sl["poly"]))
        for pc in paint["ped_crossing"]:
            parts.append(np.asarray(pc))
        tree = cKDTree(np.concatenate(parts)) if parts else None
        et = QE.end_table(g)                                                                    # control: the geometrically possible but FALSE pairs (curves between a true entry and a wrong exit): does the heat tell them from the true connectors?
        if et is not None:
            for (a_, b_), yy in zip(et["cand"], et["y"]):
                if yy:
                    continue
                cv = QE.curve(et["ent"][a_][:2], et["ent"][a_][2:], et["ext"][b_][:2], et["ext"][b_][2:])
                sg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cv, axis=0), axis=1))]
                ij = np.clip(np.floor((cv + 50.0) / 1.0).astype(int), 0, 99)
                hv = cen[i][ij[:, 0], ij[:, 1]]
                dpf = tree.query(cv)[0] if tree is not None else np.full(len(cv), 99.0)
                for d_, h_, p_ in zip(sg, hv, dpf):
                    k = int(np.searchsorted(BINS, d_, side="right") - 1)
                    c = ctrl[k]
                    c["n"] += 1; c["heat"] += int(h_ >= 0.3); c["heat_sum"] += float(h_); c["paint"] += int(p_ <= 1.0)
        for r in runs:
            q = r["pts"]
            seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
            s = np.r_[0, np.cumsum(seg)]
            L = s[-1]
            if L < 2.0:
                continue
            runs_n += 1
            curve = QE.curve(r["a"][0], r["a"][1], r["b"][0], r["b"][1])
            herr = cKDTree(curve).query(q)[0]
            dp = tree.query(q)[0] if tree is not None else np.full(len(q), 99.0)
            ij = np.clip(np.floor((q + 50.0) / 1.0).astype(int), 0, 99)
            hv = cen[i][ij[:, 0], ij[:, 1]]
            for d, e_, p_, h_ in zip(s, herr, dp, hv):
                k = int(np.searchsorted(BINS, d, side="right") - 1)
                a = acc[k]
                a["n"] += 1; a["paint"] += int(p_ <= 1.0); a["heat"] += int(h_ >= 0.3); a["heat_sum"] += float(h_); a["h05"] += int(e_ <= 0.5); a["h1"] += int(e_ <= 1.0); a["herr"].append(float(e_))
                kf = int(np.searchsorted(FR, d / L, side="right") - 1)
                f = accf[kf]
                f["n"] += 1; f["paint"] += int(p_ <= 1.0); f["heat"] += int(h_ >= 0.3); f["h05"] += int(e_ <= 0.5); f["h1"] += int(e_ <= 1.0)
    n += len(b["token"])
    if maxn and n >= maxn:
        break


def summ(a, herr=True):
    out = dict(points=a["n"], paint_within_1m=a["paint"] / max(a["n"], 1), heat_ge_0_3=a["heat"] / max(a["n"], 1), hermite_within_0_5m=a["h05"] / max(a["n"], 1), hermite_within_1m=a["h1"] / max(a["n"], 1))
    if herr and a["herr"]:
        out["hermite_error_median_m"] = float(np.median(a["herr"]))
        out["hermite_error_p90_m"] = float(np.percentile(a["herr"], 90))
        out["heat_mean"] = a["heat_sum"] / max(a["n"], 1)
    return out


res = dict(seed=seed, frames=n, runs=runs_n, seconds=round(time.time() - t0),
           false_pair_control_by_depth_m={f"{BINS[k]:g}-{BINS[k + 1] if BINS[k + 1] < 1e8 else 'inf'}": dict(points=ctrl[k]["n"], heat_ge_0_3=ctrl[k]["heat"] / max(ctrl[k]["n"], 1), heat_mean=ctrl[k]["heat_sum"] / max(ctrl[k]["n"], 1),
                                                                                                           paint_within_1m=ctrl[k]["paint"] / max(ctrl[k]["n"], 1)) for k in ctrl},
           by_depth_m={f"{BINS[k]:g}-{BINS[k + 1] if BINS[k + 1] < 1e8 else 'inf'}": summ(acc[k]) for k in acc}, by_fraction={f"{FR[k]:g}-{min(FR[k + 1], 1):g}": summ(accf[k], False) for k in accf})
print(json.dumps(res, indent=1))
os.makedirs(os.path.join(R, "runs", "e293"), exist_ok=True)
json.dump(res, open(os.path.join(R, "runs", "e293", f"depth_probe_s{seed}.json"), "w"), indent=1)
