"""E259: is the spur / bubble clean-up (assemble(clean=True): simplify) needed? CPU, 1 thread.

    wsl bash tools/env/_py14.sh temp/loop21/e259_clean.py [registry name ...]
Default models: fast_recall champion recall e24_j1rc_s1 e24_j1rc_s2 e24_j1rc_s3 fast_recall_na.
Per model, VAL 873, same cached raw outputs: base (registry settings) against clean=False, each without and with the output stage dp_eps 0.2 (the registry's own dp_eps is ignored: the stage is
set here). Prints Q, J, F1@0.5, false length, vert100 and the guards of the card; assembly time by interleaved A/B (5 repetitions of 200 frames), load average per repetition.
Layer 1: the base without the output stage reproduces the stored card."""
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loop20"))
import e255_matched as E  # noqa: E402
import evaluate as EV  # noqa: E402
import p16_infer as PI  # noqa: E402
import sparse_metrics as SM  # noqa: E402

NAMES = ["fast_recall", "champion", "recall", "e24_j1rc_s1", "e24_j1rc_s2", "e24_j1rc_s3", "fast_recall_na"]
KEYS = ("Q", "J", "F1@0.5", "false_len_frac", "vert100", "lanes_found", "topo_F1")
GUARD = dict(Q=-0.003, J=-0.01, **{"F1@0.5": -0.005})
REPS = 5
rows = []
for name in sys.argv[1:] or NAMES:
    E.load(name)
    S = E.S
    gts = S["gts"]
    base_asm = {k: v for k, v in S["asm"].items() if k != "dp_eps"}
    stored = json.load(open(f"{EV.R}/runs/eval/{name}/val.json", encoding="utf-8-sig"))["metrics"]
    res = {}
    for clean in (True, False):
        for dp in (0.0, 0.2):
            asm = dict(base_asm)
            asm["clean"] = clean
            g = [PI.assemble(r, None, S["tau"], S["cfg"], dp_eps=dp, **asm) for r in S["raw"]]
            m = SM.card_any(g, gts, extra=True)
            res[(clean, dp)] = {k: float(m[k]) for k in KEYS}
    b0 = res[(True, 0.0)]
    ok1 = abs(b0["Q"] - stored["Q"]) <= 5e-4 and abs(b0["vert100"] - stored["vert100"]) <= 0.01
    print(f"== {name}: layer 1 {'PASS' if ok1 else 'FAIL'} (Q {b0['Q']:.4f} vs stored {stored['Q']:.4f}, vert100 {b0['vert100']:.2f} vs {stored['vert100']:.2f})")
    if not ok1:
        sys.exit("stored card not reproduced")
    for dp in (0.0, 0.2):
        a, c = res[(False, dp)], res[(True, dp)]
        d = {k: a[k] - c[k] for k in KEYS}
        guards = {"Q": d["Q"] >= GUARD["Q"], "J": d["J"] >= GUARD["J"], "F1@0.5": d["F1@0.5"] >= GUARD["F1@0.5"], "vert100": abs(d["vert100"]) <= 0.3, "FL": d["false_len_frac"] <= 0.01}
        print(f"  dp_eps {dp}: clean off minus base: " + " ".join(f"d{k} {v:+.4f}" for k, v in d.items() if k in ("Q", "J", "F1@0.5", "false_len_frac", "vert100", "lanes_found", "topo_F1")) +
              "  guards " + " ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in guards.items()))
        rows.append((name, dp, all(guards.values())))
    idx = np.linspace(0, len(S["raw"]) - 1, 200).astype(int)
    raw = [S["raw"][i] for i in idx]
    for r in raw[:20]:
        PI.assemble(r, None, S["tau"], S["cfg"], **base_asm)
    t = {True: [], False: []}
    loads = []
    for rep in range(REPS):
        loads.append(open("/proc/loadavg").read().split()[0])
        for clean in ((True, False) if rep % 2 == 0 else (False, True)):
            asm = dict(base_asm)
            asm["clean"] = clean
            ts = []
            for r in raw:
                t0 = time.perf_counter()
                PI.assemble(r, None, S["tau"], S["cfg"], **asm)
                ts.append((time.perf_counter() - t0) * 1e3)
            t[clean].append(float(np.median(ts)))
    sav = np.array(t[True]) - np.array(t[False])
    print(f"  assembly ms: base {np.median(t[True]):.2f}, clean off {np.median(t[False]):.2f}; paired saving per repetition " + " ".join(f"{x:+.2f}" for x in sav) + f" -> median {np.median(sav):+.2f} ms (load {' '.join(loads)})", flush=True)
print("ALL GUARDS HOLD on every model and both output settings:", all(r[2] for r in rows))
