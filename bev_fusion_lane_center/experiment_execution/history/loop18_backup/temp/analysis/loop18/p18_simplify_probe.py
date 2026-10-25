"""Loop 18 / WP4 test 5 (one-off): the assembled graph loses 3 points of far-range recall against the node segments (0.709 vs 0.743 beyond 40 m). Does the clean-up (spur / bubble removal in simplify) cost recall? Variants of the clean-up on the cached raw outputs
(fork pruning 0.3). Usage (gcar): p18_simplify_probe.py"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_infer as PI  # noqa: E402
import p16_timing as TM  # noqa: E402
import sparse_metrics as SM  # noqa: E402

raw, gts = pickle.load(open(os.path.expanduser("~/exp_gcar_b123/p18/raw_val873.pkl"), "rb"))
net, cfg = TM.load_builder(288, 144)
for name, kw in (("default (spur 5 m, bubble 2.5 m)", {}), ("no clean-up", dict(clean=False)), ("spur 2.5 m", dict(spur_len=2.5)), ("spur 0 m", dict(spur_len=0.0)), ("bubble 1.5 m", dict(bubble_r=1.5)), ("spur 2.5 m, bubble 1.5 m", dict(spur_len=2.5, bubble_r=1.5))):
    asm = dict(TM.ASM)
    asm.update(kw)
    gs = [PI.assemble(r, None, 0.7, cfg, fast=2, fork_prune=0.3, **asm) for r in raw]
    m = SM.card_any(gs, gts)
    print(f"{name:34s} S {m['S']:.3f} Q {SM.quality(m):.3f} J {m['J']:.3f} F1@1 {m['F1@1.0']:.3f} R@1 {m['R@1.0']:.3f} P@1 {m['P@1.0']:.3f} found {m['lanes_found']:.3f} FL {m['false_len_frac']:.3f} v100 {m['vert100']:.1f}", flush=True)
