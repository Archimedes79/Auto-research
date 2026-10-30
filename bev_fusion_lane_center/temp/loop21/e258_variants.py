"""E258 step 1b: drop the fork relocation and / or the junction anchoring from the assembly? CPU, 1 thread.

    wsl bash tools/env/_py14.sh temp/loop21/e258_variants.py [registry name ...]      (default: fast_recall champion)

Variants (asm keys of the registry entry): base, reloc off (reloc_thr 0), anchoring off (junc_thr 0), both off. VAL 873: Q, J, split6 / merge6 F1, topo F1, lanes found, vert100;
assembly time per frame by interleaved A/B (repetition r: every variant on the same 200 frames in turn), median of the five repetition medians; the load average at every repetition.
Layer 1: base reproduces the stored card."""
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

VARIANTS = (("base", {}), ("reloc off", {"reloc_thr": 0.0}), ("anchor off", {"junc_thr": 0.0}), ("both off", {"reloc_thr": 0.0, "junc_thr": 0.0}))
KEYS = ("Q", "J", "split6_F1", "merge6_F1", "topo_F1", "lanes_found", "R@1_far", "vert100")
REPS = 5

for name in sys.argv[1:] or ["fast_recall", "champion"]:
    E.load(name)
    S = E.S
    gts = S["gts"]
    stored = json.load(open(f"{EV.R}/runs/eval/{name}/val.json", encoding="utf-8-sig"))["metrics"]
    print(f"== {name}")
    print(f"  {'variant':11} " + "  ".join(f"{k:>11}" for k in KEYS))
    ms = {}
    for lab, over in VARIANTS:
        asm = dict(S["asm"])
        asm.update(over)
        g = [PI.assemble(r, None, S["tau"], S["cfg"], **asm) for r in S["raw"]]
        m = SM.card_any(g, gts, extra=True)
        ms[lab] = m
        print(f"  {lab:11} " + "  ".join(f"{m[k]:11.4f}" for k in KEYS) + (f"   (layer 1: stored Q {stored['Q']:.4f}, vert100 {stored['vert100']:.2f})" if lab == "base" else f"   dQ {m['Q'] - ms['base']['Q']:+.4f} dJ {m['J'] - ms['base']['J']:+.4f} dvert100 {m['vert100'] - ms['base']['vert100']:+.2f}"))
    assert abs(ms["base"]["Q"] - stored["Q"]) <= 5e-4 and abs(ms["base"]["vert100"] - stored["vert100"]) <= 0.01, "stored card not reproduced"
    idx = np.linspace(0, len(S["raw"]) - 1, 200).astype(int)
    raw = [S["raw"][i] for i in idx]
    for r in raw[:20]:  # warm-up
        PI.assemble(r, None, S["tau"], S["cfg"], **S["asm"])
    t = {lab: [] for lab, _ in VARIANTS}
    loads = []
    for rep in range(REPS):
        loads.append(open("/proc/loadavg").read().split()[0])
        for lab, over in VARIANTS:
            asm = dict(S["asm"])
            asm.update(over)
            ts = []
            for r in raw:
                t0 = time.perf_counter()
                PI.assemble(r, None, S["tau"], S["cfg"], **asm)
                ts.append((time.perf_counter() - t0) * 1e3)
            t[lab].append(float(np.median(ts)))
    print("  assembly ms per frame (median of 5 repetition medians; range of the repetitions); load average at the repetitions: " + " ".join(loads))
    for lab, _ in VARIANTS:
        d = np.array(t[lab]) - np.array(t["base"])
        print(f"   {lab:11} {np.median(t[lab]):6.2f} ms  [{min(t[lab]):.2f} - {max(t[lab]):.2f}]  paired saving vs base per repetition: " + " ".join(f"{-x:+.2f}" for x in d) + f"  median {-np.median(d):+.2f} ms")
