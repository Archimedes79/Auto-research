"""Loop 18 / WP4 test 2b (one-off): the missing lanes sit beyond 40 m. The path validity is a cumulative product over the steps of a path, so deep nodes fall below the global threshold tau = 0.7 by construction.
Test: admit a node at tau * exp(-d * level) (p16_infer.assemble tau_decay), with fork pruning 0.3. Prints S, Q, J, F1@1, R@1, lanes found, false length, vertices per 100 m. Usage (gcar): p18_tau_decay.py"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_infer as PI  # noqa: E402
import p16_timing as TM  # noqa: E402
import sparse_metrics as SM  # noqa: E402

raw, gts = pickle.load(open(os.path.expanduser("~/exp_gcar_b123/p18/raw_val873.pkl"), "rb"))
net, cfg = TM.load_builder(288, 144)
lv = np.concatenate([r["level"][(r["valid"] > 0) & (r["pi"] >= 0.7)] for r in raw[:200]])
print("levels of admitted nodes: max", int(lv.max()), "mean", float(lv.mean()), "| all valid nodes max level", int(max(r["level"][r["valid"] > 0].max() for r in raw[:200])))
for d in (0.0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3):
    for tau in (0.7,):
        gs = [PI.assemble(r, None, tau, cfg, fast=2, fork_prune=0.3, tau_decay=d, **TM.ASM) for r in raw]
        m = SM.card_any(gs, gts)
        print(f"tau {tau} decay {d:.2f}: S {m['S']:.3f} Q {SM.quality(m):.3f} J {m['J']:.3f} F1@1 {m['F1@1.0']:.3f} R@1 {m['R@1.0']:.3f} P@1 {m['P@1.0']:.3f} found {m['lanes_found']:.3f} FL {m['false_len_frac']:.3f} v100 {m['vert100']:.1f}", flush=True)
