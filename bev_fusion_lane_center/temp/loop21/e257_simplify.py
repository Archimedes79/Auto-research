"""E257: Douglas-Peucker simplification of the assembled graph. CPU, 2 processes.

    wsl bash tools/env/_py14.sh temp/loop21/e257_simplify.py [registry name ...]      (default: fast_recall champion)

Layer 1: assembling the cached raw outputs at the registry settings reproduces runs/eval/<name>/val.json (Q within 5e-4, vert100 within 0.01).
Layer 0: simplify_dp keeps every polyline end and every shared node id, and no original vertex is farther than eps from the simplified polyline (asserted on all frames).
Then eps in EPS: Q, F1@0.5, F1@1.0, J, topo F1, lanes found, R@1 far, vert100 on VAL 873; paired scene bootstrap of the Q change at eps 0.3; the cards runs/eval/<name>/val_dp03.json;
secondary: Q at matched vertex counts 8 / 10 / 12 (vtx_r sweep, interpolation) with and without the simplification.
"""
import json
import multiprocessing as mp
import os
import sys
from collections import Counter

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loop20"))
import e255_matched as E  # noqa: E402
import evaluate as EV  # noqa: E402
import bootstrap as SB  # noqa: E402
import p16_infer as PI  # noqa: E402
import sparse_metrics as SM  # noqa: E402

EPS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
DECLARED = 0.3
KEYS = ("Q", "F1@0.5", "F1@1.0", "J", "topo_F1", "lanes_found", "R@1_far", "vert100", "false_len_frac")
VTX = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
TARGETS = (8.0, 10.0, 12.0)


def simplify_dp(g, eps):
    """polylines keep their ends, the nodes shared with another polyline and the Douglas-Peucker vertices (tolerance eps, m) between those anchors; edges (polyline pairs) are unchanged"""
    cnt = Counter(i for nd in g["nodes"] for i in nd)
    pts, nodes = [], []
    for q, nd in zip(g["pts"], g["nodes"]):
        q = np.asarray(q, np.float64)
        n = len(q)
        if n < 3:
            pts.append(q)
            nodes.append(list(nd))
            continue
        anchors = sorted({0, n - 1} | {j for j in range(n) if cnt[nd[j]] > 1})
        keep = {0, n - 1}
        for a, b in zip(anchors, anchors[1:]):
            keep.update(a + k for k in SM._dp_keep(q[a:b + 1], eps))
        keep.update(anchors)
        idx = sorted(keep)
        pts.append(q[idx])
        nodes.append([nd[j] for j in idx])
    return dict(pts=pts, edges=list(g["edges"]), nodes=nodes)


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def check_geometry(g0, g1, eps):
    """layer 0: ends and shared nodes kept; every original vertex within eps of the simplified polyline"""
    worst = 0.0
    c0 = Counter(i for nd in g0["nodes"] for i in nd)
    for q0, n0, q1, n1 in zip(g0["pts"], g0["nodes"], g1["pts"], g1["nodes"]):
        q0, q1 = np.asarray(q0, np.float64), np.asarray(q1, np.float64)
        assert n1[0] == n0[0] and n1[-1] == n0[-1], "end vertex lost"
        assert all(i in n1 for i in n0 if c0[i] > 1), "shared node lost"
        if len(q1) < 2:
            continue
        d = np.min(np.stack([seg_dist(q0, a, b) for a, b in zip(q1[:-1], q1[1:])]), axis=0)
        worst = max(worst, float(d.max()))
    assert worst <= eps + 1e-6, f"geometry bound violated: {worst:.4f} > {eps}"
    return worst


def metrics_of(graphs, gts):
    m = SM.card_any(graphs, gts, extra=True)
    return {k: float(m[k]) for k in KEYS if k in m}


def curve_point(args):
    vtx, eps = args
    asm = dict(E.S["asm"])
    asm["vtx_r"] = vtx
    g = [PI.assemble(r, None, E.S["tau"], E.S["cfg"], **asm) for r in E.S["raw"]]
    if eps:
        g = [simplify_dp(x, eps) for x in g]
    m = metrics_of(g, E.S["gts"])
    return vtx, eps, m["vert100"], m["Q"]


def interp(rows, target):
    rows = sorted(rows, key=lambda r: r[0])
    for (n0, q0), (n1, q1) in zip(rows, rows[1:]):
        if (n0 - target) * (n1 - target) <= 0 and n0 != n1:
            return q0 + (n0 - target) / (n0 - n1) * (q1 - q0)
    return None


def scene_bootstrap(g0, g1, gts, nboot=2000):
    sp = json.load(open(f"{EV.R}/data/splits.json"))
    rows = np.array(sp["val"])
    scene = np.load(f"{EV.HOME}/exp_gcar_b123/poses_val.npz")["scene"][rows]
    r0, r1 = SB.frame_records(g0, gts), SB.frame_records(g1, gts)
    ids = np.unique(scene)
    rng = np.random.RandomState(0)
    diffs = []
    for _ in range(nboot):
        pick = rng.choice(ids, len(ids), replace=True)
        idx = np.concatenate([np.nonzero(scene == s)[0] for s in pick])
        diffs.append(SB.pooled(r1, idx)["Q"] - SB.pooled(r0, idx)["Q"])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


if __name__ == "__main__":
    names = sys.argv[1:] or ["fast_recall", "champion"]
    for name in names:
        E.load(name)
        S = E.S
        gts = S["gts"]
        base = [PI.assemble(r, None, S["tau"], S["cfg"], **S["asm"]) for r in S["raw"]]
        m0 = metrics_of(base, gts)
        stored = json.load(open(f"{EV.R}/runs/eval/{name}/val.json", encoding="utf-8-sig"))
        ok = abs(m0["Q"] - stored["metrics"]["Q"]) <= 5e-4 and abs(m0["vert100"] - stored["metrics"]["vert100"]) <= 0.01
        print(f"== {name}: layer 1 (registry settings reproduce the stored card): Q {m0['Q']:.4f} vs {stored['metrics']['Q']:.4f}, vert100 {m0['vert100']:.2f} vs {stored['metrics']['vert100']:.2f}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit("stored card not reproduced")
        print("  eps      " + "  ".join(f"{k:>9}" for k in KEYS) + "   max dev (m)")
        print("  none     " + "  ".join(f"{m0[k]:9.4f}" for k in KEYS))
        kept = {}
        for eps in EPS:
            gs = [simplify_dp(g, eps) for g in base]
            worst = max(check_geometry(a, b, eps) for a, b in zip(base, gs))
            m = metrics_of(gs, gts)
            kept[eps] = (gs, m)
            print(f"  {eps:<8} " + "  ".join(f"{m[k]:9.4f}" for k in KEYS) + f"   {worst:.3f}   (dQ {m['Q'] - m0['Q']:+.4f}, dvert100 {m['vert100'] - m0['vert100']:+.2f})")
        gs, m = kept[DECLARED]
        lo, hi = scene_bootstrap(base, gs, gts)
        print(f"  eps {DECLARED}: paired scene bootstrap of the Q change: [{lo:+.4f}, {hi:+.4f}] (2000 draws over the 22 scenes)")
        card = dict(stored)
        card["metrics"] = dict(stored["metrics"])
        card["metrics"].update({k: v for k, v in m.items()})
        card["note"] = f"E257: assembled graphs simplified with simplify_dp(eps={DECLARED}) (temp/loop21/e257_simplify.py); not an evaluate.py card"
        card["dp_eps"] = DECLARED
        json.dump(card, open(f"{EV.R}/runs/eval/{name}/val_dp03.json", "w"), indent=1)
        print(f"  card runs/eval/{name}/val_dp03.json written")
        with mp.get_context("fork").Pool(2) as p:
            res = p.map(curve_point, [(v, e) for v in VTX for e in (0.0, DECLARED)])
        print("  secondary: Q at matched vertex counts (vtx_r sweep on VAL 873, interpolation; vtx_r is not tuned, nothing is chosen)")
        for e in (0.0, DECLARED):
            rows = [(n, q) for v, ee, n, q in res if ee == e]
            print(f"   {'no simplification' if not e else 'DP ' + str(e) + ' m':18}: " + "  ".join(f"vtx_r {v:.1f}: {n:5.2f} / {q:.4f}" for v, ee, n, q in sorted(res) if ee == e))
            print("                       Q at " + ", ".join(f"{t:g} vertices: {interp(rows, t):.4f}" if interp(rows, t) is not None else f"{t:g}: outside grid" for t in TARGETS))
