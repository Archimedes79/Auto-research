"""E291 step 0 (CPU, no training; the ceiling of a lane-count prior): on the 873 VAL frames with the stored dumps of E312 (`runs/e312_p_s<S>/ends_val.pkl`) and E324 s, the shares of junction ARMS whose number of REAL keypoints
(peaks of the maps head with kp_p >= THR, within 6 m of the arm centre, heading within 60 degrees of the arm's direction) is below / equal to / above the arm's GT end count n_in (entries) / n_out (exits) of `jgt.junction_gt`, and the number of GT ends
behind a deficit and of surplus keypoints behind a surplus. A count can only help the pair head where the number of found keypoints and the count disagree; the pair head cannot add a missing end.
usage: wsl bash tools/env/_py14.sh step0.py [--runs e312_p_s0,e312_p_s1,e324_s_s0,e324_s_s1] [--thr 0.3]   -> temp/e291/step0.json, a table on stdout"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.getcwd()
for sub in ("evaluation", "training", "training/grid", "training/data", "tools/viz", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import jgt as J  # noqa: E402
import qgraph_view as QV  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="e312_p_s0,e312_p_s1,e324_s_s0,e324_s_s1")
ap.add_argument("--thr", type=float, default=0.3)
ap.add_argument("--radius", type=float, default=6.0)
ap.add_argument("--cos", type=float, default=0.5)
a = ap.parse_args()
runs = a.runs.split(",")
frames = {r: {f["token"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", r, "ends_val.pkl"), "rb"))} for r in runs}
stat = {r: dict(arms=0, arms_with_in=0, arms_with_out=0, def_in=0, eq_in=0, sur_in=0, def_out=0, eq_out=0, sur_out=0, gt_in=0, gt_out=0, miss_in=0, miss_out=0, extra_in=0, extra_out=0, found_in=0, found_out=0, frames=0, frames_with_mismatch=0) for r in runs}
n = 0
for sid, (name, toks) in QV.val_scenes().items():
    for t in toks:
        if not all(t in frames[r] for r in runs):
            continue
        jg = J.junction_gt(QV.load_gt(t))
        if jg is None:
            continue
        n += 1
        for r in runs:
            f = frames[r][t]
            K = f["logit"].shape[0]
            S = stat[r]
            S["frames"] += 1
            mism = False
            for arm in jg["arms"]:
                c, tt = np.asarray(arm["c"], float), np.asarray(arm["t"], float)
                S["arms"] += 1
                for kind, n_gt in (("in", arm["n_in"]), ("out", arm["n_out"])):
                    if n_gt == 0:
                        continue
                    sl = slice(0, K) if kind == "in" else slice(K, 2 * K)
                    v = f["valid"][0 if kind == "in" else 1] & (f["kp_p"][sl] >= a.thr)
                    pos, dr = f["kp_pos"][sl], f["kp_dir"][sl]
                    d = np.linalg.norm(pos - c[None], axis=1)
                    u = dr / (np.linalg.norm(dr, axis=1, keepdims=True) + 1e-9)
                    ref = tt if kind == "in" else -tt
                    ok = v & (d <= a.radius) & (u @ ref >= a.cos)
                    found = int(ok.sum())
                    S[f"gt_{kind}"] += n_gt
                    S[f"found_{kind}"] += found
                    S[f"arms_with_{kind}"] += 1
                    if found < n_gt:
                        S[f"def_{kind}"] += 1
                        S[f"miss_{kind}"] += n_gt - found
                        mism = True
                    elif found == n_gt:
                        S[f"eq_{kind}"] += 1
                    else:
                        S[f"sur_{kind}"] += 1
                        S[f"extra_{kind}"] += found - n_gt
                        mism = True
            S["frames_with_mismatch"] += int(mism)
print(f"{n} VAL frames with junction GT; keypoints with p >= {a.thr} within {a.radius} m of the arm centre, heading within {np.degrees(np.arccos(a.cos)):.0f} degrees")
rows = []
for r in runs:
    S = stat[r]
    for kind in ("in", "out"):
        t = S[f"arms_with_{kind}"]
        rows.append((r, kind, t, S[f"def_{kind}"] / t, S[f"eq_{kind}"] / t, S[f"sur_{kind}"] / t, S[f"miss_{kind}"] / max(S[f"gt_{kind}"], 1), S[f"extra_{kind}"] / max(S[f"found_{kind}"], 1)))
print(f"{'run':12s} kind arms   deficit  exact  surplus | GT ends missed (share) | surplus keypoints (share of found)")
for r, kind, t, d, e, s, m, x in rows:
    print(f"{r:12s} {kind:4s} {t:5d}  {d:6.3f} {e:6.3f} {s:6.3f} | {m:8.3f} | {x:8.3f}")
for r in runs:
    S = stat[r]
    print(f"{r}: frames with at least one arm whose count disagrees with the found keypoints: {S['frames_with_mismatch']} of {S['frames']} ({S['frames_with_mismatch'] / S['frames']:.3f})")
json.dump(dict(args=vars(a), frames=n, stats=stat), open(os.path.join(ROOT, "temp", "e291", f"step0_thr{a.thr:g}_r{a.radius:g}.json"), "w"), indent=1)
