"""E265 step 0d (card author's pre-measurement, CPU only, scratch copy of assemble, nothing in training/ changes):
(a) range-dependent edge snap: a node at >= R_FAR m from the ego snaps onto an existing edge only within cov_far (near nodes keep cov_r 0.5);
(b) coverage restore: after the full assembly, every admitted tree chord whose samples (0.5 m) lie > gap m from the output over a run of >= run_m
    is added back as an unlinked piece (its two end vertices only).
Usage: tools/env/_py14.sh <this file> <registry name> [n_frames]"""
import inspect
import json
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

import evaluate as EV
import graph_eval as GE
import p16_builder as PB
import p16_infer as PI
import p16_train as PT
import sparse_metrics as SM

name = sys.argv[1] if len(sys.argv) > 1 else "recall"
nf = int(sys.argv[2]) if len(sys.argv) > 2 else 0
reg = EV.registry()
e = reg[name]
raw, gts, src = EV.builder_raw(e, "val", require_cache=True)
if nf:
    raw, gts = raw[:nf], gts[:nf]
kw = dict(PT.DESIGNS[e["design"]])
kw.update(json.load(open(f"{EV.R}/runs/e123_ladder/cards/{e['run']}.json", encoding="utf-8-sig"))["cfg"])
kw.update(PT.parse_set([f"{k}={v}" for k, v in e["set"].items()]))
cfg = PB.DefaultCfg(**{k: v for k, v in kw.items() if k != "w"})
asm0, tau0 = dict(e["asm"]), e["tau"]

# (a) scratch copy of assemble with a per-node snap radius
src_asm = inspect.getsource(PI.assemble)
old = "if e is not None and de < cov_r:"
assert src_asm.count(old) == 1, "assemble changed: the patch point is gone"
src_asm = src_asm.replace("def assemble(", "def assemble_far(").replace(old, "if e is not None and de < (cov_r if float(np.hypot(p[0], p[1])) < R_FAR else COV_FAR):")
ns = dict(vars(PI))
ns.update(R_FAR=40.0, COV_FAR=0.5)
exec(compile(src_asm, "assemble_far", "exec"), ns)


def assemble_far(r, cov_far, r_far=40.0):
    ns["COV_FAR"], ns["R_FAR"] = cov_far, r_far
    return ns["assemble_far"](r, None, tau0, cfg, **asm0)


def restore(g, r, gap=2.0, run_m=4.0):
    """(b): add the uncovered runs of admitted chords as unlinked pieces"""
    pos = r["pos_rev"] if "pos_rev" in r else r["pos"]
    ok = np.nonzero((r["valid"] > 0) & (r["pi"] >= tau0) & (r["parent"] >= 0))[0]
    P, _, _ = GE.dense_points(g, clip=1e9)
    tree = cKDTree(P) if len(P) else None
    extra = []
    for i in ok[np.argsort(-r["pi"][ok], kind="stable")]:
        q = GE.resample(np.stack([pos[r["parent"][i]], pos[i]]).astype(np.float64), 0.5)
        if len(q) < 2:
            continue
        d = tree.query(q)[0] if tree is not None else np.full(len(q), np.inf)
        un = d > gap
        if un.sum() * 0.5 < run_m:
            continue
        k = np.nonzero(un)[0]
        piece = q[[k[0], k[-1]]]
        if np.linalg.norm(piece[1] - piece[0]) < run_m:
            continue
        extra.append(piece)
        tree = cKDTree(np.concatenate([tree.data if tree is not None else np.zeros((0, 2)), GE.resample(piece, 0.5)]))
    if not extra:
        return g
    n0 = 1 + max((max(nd) for nd in g.get("nodes") or [] if len(nd)), default=-1)
    out = dict(pts=list(g["pts"]) + extra, edges=list(g["edges"]))
    if g.get("nodes") is not None:
        out["nodes"] = list(g["nodes"]) + [[n0 + 2 * k, n0 + 2 * k + 1] for k in range(len(extra))]
    return out


def show(vname, graphs, t0):
    m = SM.card_any(graphs, gts, extra=True)
    print(f"{vname:34s} Q {m['Q']:.4f} lanes_found {m['lanes_found']:.4f} R@1_far {m['R@1_far']:.4f} topo_F1 {m['topo_F1']:.4f} J {m['J']:.4f} F1@1 {m['F1@1.0']:.4f} "
          f"FL {m['false_len_frac']:.4f} dup {m['dup_frac']:.4f} vert100 {m['vert100']:.2f} vert100_dp {SM.vert100_dp(graphs):.2f}  [{time.time() - t0:.0f} s]", flush=True)


print(f"{name}: {len(raw)} frames, {src}")
base = [PI.assemble(r, None, tau0, cfg, **asm0) for r in raw]
show("registry output", base, time.time())
t0 = time.time()
g = [assemble_far(r, 0.5) for r in raw]
assert all(len(a["pts"]) == len(b["pts"]) and all(np.array_equal(x, y) for x, y in zip(a["pts"], b["pts"])) for a, b in zip(g, base)), "scratch copy != assemble"
print("scratch copy with cov_far 0.5 reproduces the registry graphs: yes")
for cf in (0.25, 0.05):
    t0 = time.time()
    show(f"(a) cov_r far (>= 40 m) {cf}", [assemble_far(r, cf) for r in raw], t0)
for gap, run_m in ((2.0, 4.0), (1.0, 2.0)):
    t0 = time.time()
    show(f"(b) restore gap {gap} m run {run_m} m", [restore(gg, r, gap, run_m) for gg, r in zip(base, raw)], t0)
