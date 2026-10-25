"""E246 step 0 (CPU, in-sample ceiling): do range-specific admission thresholds raise R@1 beyond 40 m at the same false length?

    wsl bash tools/env/_py14.sh temp/loop19/e246_step0.py [registry name, default fast_recall]

The assembly accepts a per-node threshold array (`thr = tau`, alive = pi >= thr), so the three bins near (< 25 m), mid (25 - 40 m), far (> 40 m) of the node
position get their own tau without any change of shared code. Coordinate descent on the FULL VAL 873 (in-sample: this is the optimistic ceiling the card
asks for, not a result): far bin first (others at the registered tau), then mid, then near; objective = R@1_far at false_len_frac <= base + 0.003
(card budget) and vert100 <= base + 0.3. Gate of the card: gain in R@1_far >= +0.015, else the out-of-fold version cannot reach the bar: kill.
Prints the base point, every grid point and the ceiling. Cache: the registry entry's raw VAL 873 outputs.
"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np

import evaluate as EV
import p16_builder as PB
import p16_infer as PI
import p16_train as PT
import sparse_metrics as SM

name = sys.argv[1] if len(sys.argv) > 1 else "fast_recall"
reg = EV.registry()
e = reg[name]
raw, gts, cache = EV.builder_raw(e, "val", None, False, True)
c = json.load(open(f"{EV.R}/runs/e123_ladder/cards/{e['run']}.json", encoding="utf-8-sig"))
extra = dict(c["cfg"])
extra.update(PT.parse_set([f"{k}={v}" for k, v in e["set"].items()]))
kw = dict(PT.DESIGNS[e["design"]])
kw.update(extra)
cfg = PB.DefaultCfg(**{k: v for k, v in kw.items() if k not in ("w",)})
asm = dict(e["asm"])
BINS = (0.0, 25.0, 40.0, 1e9)
GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
TAU0 = e["tau"]


def run(taus):
    g = []
    for r in raw:
        rr = np.linalg.norm(r["pos"], axis=1)
        thr = np.full(len(rr), taus[2])
        thr[rr < BINS[2]] = taus[1]
        thr[rr < BINS[1]] = taus[0]
        g.append(PI.assemble(r, None, thr, cfg, **asm))
    m = SM.card_any(g, gts, extra=True)
    return {k: float(m[k]) for k in ("Q", "S", "R@1_far", "lanes_found", "false_len_frac", "vert100", "topo_F1", "F1@1.0", "R@1_near", "R@1_mid", "J")}


def sweep(base, which):
    cands = []
    for t in GRID:
        x = list(base)
        x[which] = t
        cands.append(tuple(x))
    with mp.get_context("fork").Pool(4) as p:
        res = p.map(run, cands)
    return list(zip(cands, res))


b0 = run((TAU0, TAU0, TAU0))
print(f"{name}: base tau {TAU0} -> Q {b0['Q']:.4f} R@1_far {b0['R@1_far']:.4f} false_len {b0['false_len_frac']:.4f} vert100 {b0['vert100']:.2f} lanes_found {b0['lanes_found']:.3f}", flush=True)
cur, cur_m = (TAU0, TAU0, TAU0), b0
for which, label in ((2, "far"), (1, "mid"), (0, "near")):
    out = sweep(cur, which)
    for taus, m in out:
        print(f"  {label:4} tau {taus[which]:.1f}: R@1_far {m['R@1_far']:.4f} Q {m['Q']:.4f} false_len {m['false_len_frac']:.4f} vert100 {m['vert100']:.2f} R@1_mid {m['R@1_mid']:.4f} R@1_near {m['R@1_near']:.4f}", flush=True)
    ok = [(t, m) for t, m in out if m["false_len_frac"] <= b0["false_len_frac"] + 0.003 and m["vert100"] <= b0["vert100"] + 0.3]
    cur, cur_m = max(ok, key=lambda a: (a[1]["R@1_far"], a[1]["Q"]))
    print(f"  -> best {label}: taus {cur} R@1_far {cur_m['R@1_far']:.4f}", flush=True)
gain = cur_m["R@1_far"] - b0["R@1_far"]
print(f"\nCEILING (in-sample, VAL 873): taus near/mid/far {cur}: R@1_far {cur_m['R@1_far']:.4f} ({gain:+.4f}), Q {cur_m['Q']:.4f} ({cur_m['Q'] - b0['Q']:+.4f}), false_len {cur_m['false_len_frac']:.4f} "
      f"({cur_m['false_len_frac'] - b0['false_len_frac']:+.4f}), vert100 {cur_m['vert100']:.2f} ({cur_m['vert100'] - b0['vert100']:+.2f})")
print("GATE (+0.015 R@1_far):", "PASS" if gain >= 0.015 else "KILL")
