"""E265 step 0 (card author's pre-measurement, CPU only, cached raw outputs): where along the chain tree -> admission -> union -> clean-up ->
junction steps -> fork pruning are the far ground-truth points (>= 40 m, |x|,|y| < 48, dense 0.5 m) lost?  Usage:
    tools/env/_py14.sh <this file> <registry name> [n_frames]
Prints one line per stage (R@1_far, lanes_found of far-touching lanes, far points first lost here) and the miss-distance histogram."""
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

import evaluate as EV
import graph_eval as GE
import p16_builder as PB
import p16_infer as PI
import p16_train as PT
import topo_metrics as TM

name = sys.argv[1] if len(sys.argv) > 1 else "recall"
nf = int(sys.argv[2]) if len(sys.argv) > 2 else 0
reg = EV.registry()
e = reg[name]
raw, gts, src = EV.builder_raw(e, "val", require_cache=True)
if nf:
    raw, gts = raw[:nf], gts[:nf]
kw = dict(PT.DESIGNS[e["design"]])
kw.update(dict(__import__("json").load(open(f"{EV.R}/runs/e123_ladder/cards/{e['run']}.json", encoding="utf-8-sig"))["cfg"]))
kw.update(PT.parse_set([f"{k}={v}" for k, v in e["set"].items()]))
cfg = PB.DefaultCfg(**{k: v for k, v in kw.items() if k != "w"})
asm0, tau0 = dict(e["asm"]), e["tau"]
print(f"{name}: {len(raw)} frames, cache {src}, asm {asm0}, tau {tau0}")


def chords(r, tau):
    """the raw tree as polylines: one 2-point piece per admitted parent -> child (no union, no clean-up)"""
    pos = r["pos_rev"] if "pos_rev" in r else r["pos"]
    ok = (r["valid"] > 0) & (r["pi"] >= tau) & (r["parent"] >= 0)
    return dict(pts=[np.stack([pos[r["parent"][i]], pos[i]]) for i in np.nonzero(ok)[0]], edges=[])


def far_gt(g):
    G, _, I = GE.dense_points(g)
    return G, I, np.linalg.norm(G, axis=1) >= 40.0


stages = [
    ("tree, all valid nodes (tau 0)", lambda r: chords(r, 0.0)),
    ("tree, tau 0.3", lambda r: chords(r, 0.3)),
    ("tree, admitted (tau 0.7)", lambda r: chords(r, tau0)),
    ("union, no clean-up / junction / pruning", lambda r: PI.assemble(r, None, tau0, cfg, **dict(asm0, junc_thr=0.0, reloc_thr=0.0, fork_prune=0.0), clean=False)),
    ("+ clean-up (spur / bubble)", lambda r: PI.assemble(r, None, tau0, cfg, **dict(asm0, junc_thr=0.0, reloc_thr=0.0, fork_prune=0.0))),
    ("+ junction anchoring", lambda r: PI.assemble(r, None, tau0, cfg, **dict(asm0, reloc_thr=0.0, fork_prune=0.0))),
    ("+ fork relocation", lambda r: PI.assemble(r, None, tau0, cfg, **dict(asm0, fork_prune=0.0))),
    ("+ fork pruning = registry output", lambda r: PI.assemble(r, None, tau0, cfg, **asm0)),
]
G_all = [far_gt(g) for g in gts]
nfar = sum(int(f.sum()) for _, _, f in G_all)
hit = {}
for sname, fn in stages:
    t0 = time.time()
    hs, lf, lt, vtx, length = [], 0, 0, 0, 0.0
    for r, (G, I, far) in zip(raw, G_all):
        g = fn(r)
        P, _, _ = GE.dense_points(g)
        d = cKDTree(P).query(G)[0] if len(P) else np.full(len(G), np.inf)
        hs.append(d[far] <= 1.0)
        for lane in np.unique(I[far]):  # lanes found among the lanes that reach >= 40 m (whole-lane coverage, as graph_eval)
            lt += 1
            lf += int((d[I == lane] <= 1.0).mean() >= 0.8)
    hit[sname] = np.concatenate(hs) if hs else np.zeros(0, bool)
    print(f"{sname:42s} R@1_far {hit[sname].mean():.4f}  lanes_found(far lanes) {lf / max(lt, 1):.4f} ({lt} lanes)  [{time.time() - t0:.0f} s]")
print(f"far GT points: {nfar}")
names = [s for s, _ in stages]
print("first lost at (share of all far points):")
for a, b in zip(names[:-1], names[1:]):
    print(f"  {a:42s} -> {b:42s} lost {np.mean(hit[a] & ~hit[b]):.4f} gained {np.mean(~hit[a] & hit[b]):.4f}")
print(f"  never in the tree at tau 0: {np.mean(~hit[names[0]]):.4f}")
# distance of the final misses to the all-valid tree and to the admitted union; Chebyshev ring of the misses
dist0, distA, ring, ringall = [], [], [], []
for r, (G, I, far) in zip(raw, G_all):
    Gf = G[far]
    if not len(Gf):
        continue
    gF = stages[-1][1](r)
    P, _, _ = GE.dense_points(gF)
    dF = cKDTree(P).query(Gf)[0] if len(P) else np.full(len(Gf), np.inf)
    P0, _, _ = GE.dense_points(chords(r, 0.0))
    d0 = cKDTree(P0).query(Gf)[0] if len(P0) else np.full(len(Gf), np.inf)
    m = dF > 1.0
    dist0.append(d0[m])
    distA.append(dF[m])
    ring.append(np.abs(Gf[m]).max(1))
    ringall.append(np.abs(Gf).max(1))
dist0, distA, ring, ringall = map(np.concatenate, (dist0, distA, ring, ringall))
edges = [1.0, 2.0, 3.0, 5.0, 10.0, np.inf]
print(f"final misses: {len(distA)} of {nfar} far points ({len(distA) / nfar:.4f})")
for nm, dd in (("to the all-valid tree", dist0), ("to the output graph", distA)):
    lo = 0.0
    parts = []
    for hi in edges:
        parts.append(f"[{lo:g},{hi:g}) {np.mean((dd >= lo) & (dd < hi)):.3f}")
        lo = hi
    print(f"  distance {nm}: " + "  ".join(parts))
print(f"  misses with max(|x|,|y|) >= 45 m (outside the seed lattice): {np.mean(ring >= 45):.3f} (all far points: {np.mean(ringall >= 45):.3f})")
