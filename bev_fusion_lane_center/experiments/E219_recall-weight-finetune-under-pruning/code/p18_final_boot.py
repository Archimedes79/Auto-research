"""Loop 18: paired scene bootstrap (VAL, 873 frames, 22 scenes) of the final systems against the control E94 and against each other: S, Q, J, lanes found, F1@1, vertices per 100 m. Cached raw outputs; fork pruning 0.3 (f1, j1) / 0.4 (rc1, rc2, rc4, rc1s1, j1p4, j1rc).
Writes runs/e18/final_boot.json and prints a table. Usage (gcar): p18_final_boot.py"""
import bflc_paths as BP  # project paths from the location of this checkout
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap as SB  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_timing as TM  # noqa: E402

R = BP.ROOT
H = os.path.expanduser("~/exp_gcar_b123/p18")
sp = json.load(open(f"{R}/data/splits.json"))
scene = np.load(os.path.expanduser("~/exp_gcar_b123/poses_val.npz"))["scene"][np.array(sp["val"])]
ctrl = pickle.load(open(f"{R}/runs/e94_a/graphs_val_clean.pkl", "rb"))["preds"]
SYS = [("f1p", "e123_D7_r3_f1", 288, 144, "raw_val873.pkl", 0.3), ("j1p", "e123_D7_r3_j1", 192, 96, "raw_val873_e123_D7_r3_j1_192.pkl", 0.3), ("rc1p", "e123_D7_r3_rc1", 288, 144, "raw_val873_e123_D7_r3_rc1_288.pkl", 0.4),
       ("rc2p", "e18_D7_rc2", 288, 144, "raw_val873_e18_D7_rc2_288.pkl", 0.4), ("rc4p", "e18_D7_rc4", 288, 144, "raw_val873_e18_D7_rc4_288.pkl", 0.4),
       ("j1p4", "e123_D7_r3_j1", 192, 96, "raw_val873_e123_D7_r3_j1_192.pkl", 0.4), ("j1rcp", "e18_D7_j1rc", 192, 96, "raw_val873_e18_D7_j1rc_192.pkl", 0.4), ("rc1s1p", "e18_D7_rc1s1", 288, 144, "raw_val873_e18_D7_rc1s1_288.pkl", 0.4),
       ("j1rcs1p", "e18_D7_j1rcs1", 192, 96, "raw_val873_e18_D7_j1rcs1_192.pkl", 0.4),
       ("j1eqp", "e18_D7_j1eq", 192, 96, "raw_val873_e18_D7_j1eq_192.pkl", 0.4),
       ("j1eqs1p", "e18_D7_j1eqs1", 192, 96, "raw_val873_e18_D7_j1eqs1_192.pkl", 0.4), ("f1eqp", "e18_D7_f1eq", 288, 144, "raw_val873_e18_D7_f1eq_288.pkl", 0.4),
       ("f1p4", "e123_D7_r3_f1", 288, 144, "raw_val873.pkl", 0.4),
       ("j1rc6p", "e18_D7_j1rc6", 192, 96, "raw_val873_e18_D7_j1rc6_192.pkl", 0.4)]
recs = {}
gts = None
for nm, name, M, K0, cache, fp in SYS:
    if not os.path.exists(f"{H}/{cache}"):
        continue
    net, cfg = TM.load_builder(M, K0, name)
    raw, gts = pickle.load(open(f"{H}/{cache}", "rb"))
    gs = [PI.assemble(r, None, 0.7, cfg, fast=2, fork_prune=fp, **TM.ASM) for r in raw]
    recs[nm] = SB.frame_records(gs, gts)
recs["control"] = SB.frame_records(ctrl, gts)
out = {}
pairs = [(a, "control") for a in recs if a != "control"] + [("rc1p", "f1p"), ("rc2p", "rc1p"), ("rc4p", "rc1p"), ("j1rcp", "j1p4"), ("rc1s1p", "rc1p"), ("j1rcs1p", "j1p4"), ("j1rcs1p", "j1rcp"), ("j1eqp", "j1p4"), ("j1rcp", "j1eqp"), ("j1rcs1p", "j1eqp"), ("j1eqs1p", "j1p4"), ("j1rcp", "j1eqs1p"), ("j1rcs1p", "j1eqs1p"), ("f1eqp", "f1p4"), ("rc1p", "f1eqp"), ("rc1s1p", "f1eqp"), ("rc1p", "f1p4"), ("j1rc6p", "j1rcp"), ("j1rc6p", "j1eqp")]
keys = ("S", "Q", "J", "lanes_found", "F1@1.0", "vert100")
print("pair | " + " | ".join(keys))
for a, b in pairs:
    if a not in recs or b not in recs:
        continue
    p = SB.paired(recs[a], recs[b], scene, B=1000, keys=keys)
    out[f"{a} - {b}"] = {k: dict(diff=p[k]["diff"], lo=p[k]["lo"], hi=p[k]["hi"]) for k in keys if k in p}
    print(f"{a} - {b} | " + " | ".join(f"{p[k]['diff']:+.3f} [{p[k]['lo']:+.3f}, {p[k]['hi']:+.3f}]" for k in keys if k in p), flush=True)
json.dump(out, open(f"{R}/runs/e18/final_boot.json", "w"), indent=1)
