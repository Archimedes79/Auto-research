"""Loop 18 / E231.a (one-off): the j1 family on the loop-17 failure series (VAL, 22 scenes, 291 rows per series, fork pruning 0.4, no time recursion): failure-phase and clean-phase Q, lanes found, J, F1@1, S per model, and paired scene bootstraps of the failure-phase frames for j1rc / j1rcs1 against the equal-budget controls j1eq / j1eqs1 (and j1eq / j1eqs1 against j1).
Usage (gcar): p18_e231_table.py   -> prints the tables and writes runs/e18/e231_table.json"""
import bflc_paths as BP  # project paths from the location of this checkout
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap as SB  # noqa: E402
import sparse_metrics as SM  # noqa: E402

R = BP.ROOT
D = f"{R}/runs/e17/e18_j1fam"
MODELS = ["j1", "j1eq", "j1eqs1", "j1rc", "j1rcs1"]
PERTS = [("dark", "night"), ("blur", "blur"), ("cam_front3_off", "three cameras off")]
VARS = [("single", "single frame"), ("maxdecay_0.95", "spatial hold")]
PAIRS = [("j1rc", "j1eq"), ("j1rcs1", "j1eq"), ("j1rc", "j1eqs1"), ("j1rcs1", "j1eqs1"), ("j1eq", "j1"), ("j1eqs1", "j1"), ("j1rc", "j1"), ("j1rcs1", "j1")]
KEYS = ("S", "Q", "J", "lanes_found", "F1@1.0")
out = {"cards": {}, "boot": {}}
for p, pn in PERTS:
    for v, vn in VARS:
        res = {}
        for m in MODELS:
            f = f"{D}/{p}_{v}_{m}.pkl"
            if os.path.exists(f):
                res[m] = pickle.load(open(f, "rb"))
        print(f"== {pn}, {vn}")
        for m, d in res.items():
            row = []
            for ph in ("cl", "fail"):
                c = d["cards"][ph]
                row.append(f"{ph}: Q {SM.quality(c):.3f} S {c['S']:.3f} J {c['J']:.3f} found {c['lanes_found']:.3f} F1@1 {c['F1@1.0']:.3f} v100 {c['vert100']:.1f}")
            out["cards"][f"{p}|{v}|{m}"] = {ph: dict(Q=float(SM.quality(d["cards"][ph])), **{k: float(d["cards"][ph][k]) for k in ("S", "J", "lanes_found", "F1@1.0", "vert100")}) for ph in ("cl", "fail")}
            print(f"  {m:7s} " + " | ".join(row))
        if "j1eq" not in res:
            continue
        ph0 = res["j1eq"]["phase"]
        mask = ph0 == 1
        recs = {}
        for m, d in res.items():
            idx = np.where(mask)[0]
            recs[m] = SB.frame_records([d["graphs"][i] for i in idx], [d["gts"][i] for i in idx])
        scene = res["j1eq"]["scene"][mask]
        for a, b in PAIRS:
            if a in recs and b in recs:
                q = SB.paired(recs[a], recs[b], scene, B=1000, keys=KEYS)
                out["boot"][f"{p}|{v}|{a} - {b}"] = {k: dict(diff=q[k]["diff"], lo=q[k]["lo"], hi=q[k]["hi"]) for k in KEYS if k in q}
                print(f"  failure frames {a} - {b}: " + " | ".join(f"{k} {q[k]['diff']:+.3f} [{q[k]['lo']:+.3f}, {q[k]['hi']:+.3f}]" for k in KEYS if k in q))
json.dump(out, open(f"{R}/runs/e18/e231_table.json", "w"), indent=1)
