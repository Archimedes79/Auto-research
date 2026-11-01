"""E265 step 0e (card author's pre-measurement, CPU only, assembly options that exist: spur_len, bubble_r, clean):
which half of the clean-up (spur removal < 5 m / bubble removal < 2.5 m) costs the far coverage, judged on the per-piece DP vertex basis (D20).
Usage: tools/env/_py14.sh <this file> <registry name> [...]"""
import sys
import time

import numpy as np

import evaluate as EV
import sparse_metrics as SM

reg = EV.registry()
arms = [("registry", {}), ("spurs kept (spur_len 0)", dict(spur_len=0.0)), ("bubbles kept (bubble_r 0)", dict(bubble_r=0.0))]
for name in sys.argv[1:]:
    for an, over in arms:
        t0 = time.time()
        if over:
            import p16_infer as PI
            orig = PI.assemble

            def patched(*a, _o=over, _f=orig, **k):
                k.update(_o)
                return _f(*a, **k)
            PI.assemble = patched
        g, gts, info = EV.graphs_of(name, reg[name], "val", require_cache=True)
        if over:
            PI.assemble = orig
        m = SM.card_any(g, gts, extra=True)
        print(f"{name:12s} {an:26s} Q {m['Q']:.4f} lanes_found {m['lanes_found']:.4f} R@1_far {m['R@1_far']:.4f} R@1_mid {m['R@1_mid']:.4f} P@1_far {m['P@1_far']:.4f} topo_F1 {m['topo_F1']:.4f} "
              f"J {m['J']:.4f} F1@0.5 {m['F1@0.5']:.4f} FL {m['false_len_frac']:.4f} dup {m['dup_frac']:.4f} vert100 {m['vert100']:.2f} vert100_dp {SM.vert100_dp(g):.2f} asm_ms {info['asm_ms']:.1f}  [{time.time() - t0:.0f} s]", flush=True)
