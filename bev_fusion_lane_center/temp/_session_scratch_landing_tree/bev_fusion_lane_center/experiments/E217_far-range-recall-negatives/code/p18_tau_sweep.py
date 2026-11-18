"""Loop 18 / WP4 test 3 (one-off): is the recall gap to the control a different operating point? Validity threshold tau of the champion (fork pruning 0.3) swept on the 873 VAL frames; the control E94 is one point
(R@1 0.894, P@1 0.872, false length 0.062, lanes found 0.80, vertices 21.8 per 100 m). Prints recall, precision, false length, lanes found, J, S, Q and vertices per 100 m for every tau.
Usage (gcar): p18_tau_sweep.py"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_infer as PI  # noqa: E402
import p16_timing as TM  # noqa: E402
import sparse_metrics as SM  # noqa: E402

raw, gts = pickle.load(open(os.path.expanduser("~/exp_gcar_b123/p18/raw_val873.pkl"), "rb"))
net, cfg = TM.load_builder(288, 144)
for tau in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    gs = [PI.assemble(r, None, tau, cfg, fast=2, fork_prune=0.3, **TM.ASM) for r in raw]
    m = SM.card_any(gs, gts)
    print(f"tau {tau:.1f}: R@1 {m['R@1.0']:.3f} P@1 {m['P@1.0']:.3f} F1@1 {m['F1@1.0']:.3f} FL {m['false_len_frac']:.3f} found {m['lanes_found']:.3f} | J {m['J']:.3f} S {m['S']:.3f} Q {SM.quality(m):.3f} | v100 {m['vert100']:.1f}", flush=True)
