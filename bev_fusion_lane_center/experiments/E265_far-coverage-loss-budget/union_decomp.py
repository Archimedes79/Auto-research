"""E265 step 0c (card author's pre-measurement, CPU only): which sub-step of the path union loses the far points that the admitted tree covers?
Toggles on a monkey-patched copy of the stage settings (no change to training/): has_path skip (the duplicate-path test with limit 1.5 x chord + 3 m),
vertex reuse radius vtx_r, edge snap radius cov_r, the straightening of snap vertices, and the clean-up; junction steps and pruning on (registry) or off.
Usage: tools/env/_py14.sh <this file> <registry name> [n_frames]"""
import math
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

import evaluate as EV
import graph_eval as GE
import p16_infer as PI
import p18_union_nb as UN
import sparse_metrics as SM

name = sys.argv[1] if len(sys.argv) > 1 else "recall"
nf = int(sys.argv[2]) if len(sys.argv) > 2 else 0
reg = EV.registry()
e = reg[name]
raw, gts, src = EV.builder_raw(e, "val", require_cache=True)
if nf:
    raw, gts = raw[:nf], gts[:nf]
import json
import p16_builder as PB
import p16_train as PT

kw = dict(PT.DESIGNS[e["design"]])
kw.update(json.load(open(f"{EV.R}/runs/e123_ladder/cards/{e['run']}.json", encoding="utf-8-sig"))["cfg"])
kw.update(PT.parse_set([f"{k}={v}" for k, v in e["set"].items()]))
cfg = PB.DefaultCfg(**{k: v for k, v in kw.items() if k != "w"})
asm0, tau0 = dict(e["asm"]), e["tau"]
G_all = []
for g in gts:
    G, _, I = GE.dense_points(g)
    G_all.append((G, I, np.linalg.norm(G, axis=1) >= 40.0))

orig_has_path = {c: c.has_path for c in (PI.Union, PI.FastUnion, UN.NumbaUnion) if hasattr(c, "has_path")}


def set_has_path(mode):
    """mode 'orig' | 'off' (never skip an edge as duplicate) | 'tight' (limit 1.1 x chord + 1 m)"""
    for c, f in orig_has_path.items():
        if mode == "orig":
            c.has_path = f
        elif mode == "off":
            c.has_path = lambda self, a, b, max_hops=12, slack=3.0: False
        else:
            def tight(self, a, b, max_hops=12, slack=3.0, _f=f):
                Va, Vb = self.V[a], self.V[b]
                ch = math.hypot(Va[0] - Vb[0], Va[1] - Vb[1])
                # the original rule accepts a path up to 1.5 x chord + slack; here up to 1.1 x chord + 1 m (an existing path must follow the new edge)
                return _f(self, a, b, max_hops, slack=1.0 - 0.4 * ch)
            c.has_path = tight


def score(graphs):
    hs, lf, lt = [], 0, 0
    for g, (G, I, far) in zip(graphs, G_all):
        P, _, _ = GE.dense_points(g)
        d = cKDTree(P).query(G)[0] if len(P) else np.full(len(G), np.inf)
        hs.append(d[far] <= 1.0)
        for lane in np.unique(I[far]):
            lt += 1
            lf += int((d[I == lane] <= 1.0).mean() >= 0.8)
    return float(np.concatenate(hs).mean()), lf / max(lt, 1)


nojunc = dict(junc_thr=0.0, reloc_thr=0.0, fork_prune=0.0)
variants = [
    ("union as is (no clean-up / junction)", "orig", dict(nojunc), False),
    ("has_path skip off", "off", dict(nojunc), False),
    ("has_path tight (1.1 x chord + 1 m)", "tight", dict(nojunc), False),
    ("vtx_r 0.75", "orig", dict(nojunc, vtx_r=0.75), False),
    ("vtx_r 1.0", "orig", dict(nojunc, vtx_r=1.0), False),
    ("has_path off + vtx_r 1.0", "off", dict(nojunc, vtx_r=1.0), False),
    ("cov_r 0.05 (no edge snap)", "orig", dict(nojunc, cov_r=0.05), False),
    ("minimal union: vtx_r / cov_r 0.05, has_path off", "off", dict(nojunc, vtx_r=0.05, cov_r=0.05), False),
    ("registry output", "orig", dict(), True),
    ("registry + has_path off", "off", dict(), True),
    ("registry + vtx_r 1.0", "orig", dict(vtx_r=1.0), True),
    ("registry + cov_r 0.25", "orig", dict(cov_r=0.25), True),
    ("registry + cov_r 0.05", "orig", dict(cov_r=0.05), True),
    ("registry + minimal union", "off", dict(vtx_r=0.05, cov_r=0.05), True),
]
print(f"{name}: {len(raw)} frames, {src}")
for vname, hp, over, clean in variants:
    set_has_path(hp)
    t0 = time.time()
    a = dict(asm0, **over)
    graphs = [PI.assemble(r, None, tau0, cfg, clean=clean, **a) for r in raw]
    rf, lf = score(graphs)
    msg = f"{vname:42s} R@1_far {rf:.4f} far-lanes found {lf:.4f}"
    if clean:  # whole card for the full-chain variants (Q, lanes found, vertex counts on both bases)
        m = SM.card_any(graphs, gts, extra=True)
        msg += f" | Q {m['Q']:.4f} lanes_found {m['lanes_found']:.4f} R@1_far(card) {m['R@1_far']:.4f} topo_F1 {m['topo_F1']:.4f} FL {m['false_len_frac']:.4f} vert100 {m['vert100']:.2f} vert100_dp {SM.vert100_dp(graphs):.2f}"
    print(msg + f"  [{time.time() - t0:.0f} s]", flush=True)
set_has_path("orig")
