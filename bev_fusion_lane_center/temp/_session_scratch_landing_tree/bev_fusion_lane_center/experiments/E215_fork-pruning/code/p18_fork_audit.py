"""Loop 18 / WP4 test 1 (one-off): the champion's J deficit is a PRECISION deficit (split6_P 0.18 vs 0.28 of the control, recall 0.42 vs 0.48): where do the false forks come from, and does an assembly setting cut them?
Assembles the cached raw node outputs of the 873 VAL frames (champion f1, M 288 / K0 144, tau 0.7) with a list of settings and prints J, split / merge precision and recall (6 m), forks per frame, S, Q, vertices per 100 m, lanes found.
Usage (gcar): p18_fork_audit.py [--n 873]"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_common as C  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_timing as TM  # noqa: E402
import graph_eval as GE  # noqa: E402
import sparse_metrics as SM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=873)
a = ap.parse_args()
raw, gts = pickle.load(open(os.path.expanduser(f"~/exp_gcar_b123/p18/raw_val{a.n}.pkl"), "rb"))
net, cfg = TM.load_builder(288, 144)
BASE = dict(TM.ASM)
VARIANTS = [
    ("champion (junc 0.5, reloc 0.4)", {}),
    ("fork prune 0.05", dict(fork_prune=0.05)),
    ("fork prune 0.1", dict(fork_prune=0.1)),
    ("fork prune 0.2", dict(fork_prune=0.2)),
    ("fork prune 0.3", dict(fork_prune=0.3)),
    ("fork prune 0.5", dict(fork_prune=0.5)),
    ("prune split 0.2 / merge 0.1", dict(fork_prune=0.2, fork_prune_m=0.1)),
]
rows = []
for name, kw in VARIANTS:
    asm = dict(BASE)
    asm.update(kw)
    t0 = time.time()
    gs = [PI.assemble(r, None, 0.7, cfg, fast=2, **asm) for r in raw]
    ms = (time.time() - t0) / len(gs) * 1e3
    m = SM.card_any(gs, gts)
    nf = np.mean([sum(1 for _ in range(1)) and (len(GE.junctions(g)[0]) + len(GE.junctions(g)[1])) for g in gs])
    ng = np.mean([(len(GE.junctions(g)[0]) + len(GE.junctions(g)[1])) for g in gts])
    rows.append((name, m, nf, ng, ms))
    print(f"{name:34s} S {m['S']:.3f} Q {m.get('Q', float('nan')):.3f} J {m['J']:.3f} | split P {m['split6_P']:.3f} R {m['split6_R']:.3f} | merge P {m['merge6_P']:.3f} R {m['merge6_R']:.3f} | forks/frame {nf:.2f} (GT {ng:.2f}) | v100 {m['vert100']:.1f} found {m['lanes_found']:.3f} F1@1 {m['F1@1.0']:.3f} | {ms:.1f} ms", flush=True)
