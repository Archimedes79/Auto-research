"""Loop 18 / WP4 test 4 (one-off): with fork pruning in place, do the earlier assembly settings (junction anchoring threshold junc_thr, fork relocation threshold reloc_thr) move? The pruning removes false forks, so a lower anchoring threshold may now add recall of true forks
without paying precision. Cached raw outputs of the 873 VAL frames; S, Q, J, split / merge P and R, forks per frame, vertices per 100 m. Usage (gcar): p18_asm_grid.py"""
import itertools
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_timing as TM  # noqa: E402
import sparse_metrics as SM  # noqa: E402

raw, gts = pickle.load(open(os.path.expanduser("~/exp_gcar_b123/p18/raw_val873.pkl"), "rb"))
net, cfg = TM.load_builder(288, 144)
for jt, rt, fp in itertools.product((0.2, 0.3, 0.5, 0.7), (0.2, 0.4, 0.6), (0.3, 0.4)):
    asm = dict(TM.ASM)
    asm.update(junc_thr=jt, reloc_thr=rt, fork_prune=fp)
    gs = [PI.assemble(r, None, 0.7, cfg, fast=2, **asm) for r in raw]
    m = SM.card_any(gs, gts)
    nf = np.mean([len(GE.junctions(g)[0]) + len(GE.junctions(g)[1]) for g in gs])
    print(f"junc {jt} reloc {rt} prune {fp}: S {m['S']:.3f} Q {SM.quality(m):.3f} J {m['J']:.3f} | split P {m['split6_P']:.2f} R {m['split6_R']:.2f} merge P {m['merge6_P']:.2f} R {m['merge6_R']:.2f} | forks {nf:.1f} | found {m['lanes_found']:.3f} F1@1 {m['F1@1.0']:.3f} v100 {m['vert100']:.1f}", flush=True)
