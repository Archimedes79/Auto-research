"""Connectivity, duplicate and range metrics of lane graphs (added 2026-10-25), next to graph_eval / sparse_metrics.

The metric-response test showed three blind spots of S, Q and point F1: a lane that is broken into unconnected pieces, a parallel duplicate,
and where along the range the misses are. These metrics close them; graph format and clip (|x|, |y| < 48 m) as in graph_eval.

  topo      directed reachability (TOPO-style, Biagioni and Eriksson 2012): seeds every 5 m on the ground truth; from each seed the ground-truth
            points reachable along the travel direction within 30 m ('holes') against the predicted points reachable from the matched predicted
            point ('marbles', snap 2 m, direction within 60 deg); 1:1 matching within 1 m.  topo_P = matched / marbles, topo_R = matched / holes
            (a seed without a match adds its holes to the recall denominator).  Responds to breaks, cut links, false branches, reversed direction.
  dup_frac  share of the predicted length that runs parallel (|cos| > 0.9) within 0.75 m of another predicted piece that is neither its
            neighbour nor its sibling in the graph, not counting the first and last 4 m of a piece (fork zones); parallel duplicates are
            invisible to point F1, FL and S.
  range     recall / precision at 1 m by distance from the ego: near < 20 m, mid 20 - 40 m, far >= 40 m (R@1_far is the open gap of loop 18; a frame without
            ground truth adds its predicted points, all false, to the precision bins);
            the far bin is also split at max(|x|,|y|) = 45 m, the edge of the inference seed lattice (E266): far_inner (< 45) and far_ring (45 - 48).

    extra(preds, gts) -> dict(topo_P, topo_R, topo_F1, dup_frac, R@1_near / mid / far / far_inner / far_ring, P@1_ the same)
Self-test: python topo_metrics.py
"""
import os
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

import bflc_paths as BP  # project paths from the location of this checkout
import graph_eval as GE

if os.path.join(BP.REPO, "autoresearch") not in sys.path:
    sys.path.append(os.path.join(BP.REPO, "autoresearch"))
from Evaluation import stats as ST  # noqa: E402  (the shared scene bootstrap)

CLIP = 48.0
BINS = (("near", 0.0, 20.0), ("mid", 20.0, 40.0), ("far", 40.0, 1e9))
RING = 45.0  # E266: the far bin split by max(|x|,|y|) at the edge of the inference seed lattice: far_inner < 45 m <= far_ring (< CLIP)
NAMES = tuple(b[0] for b in BINS) + ("far_inner", "far_ring")
STEP, REACH, SNAP, TOL, COS_SNAP = 5, 30.0, 2.0, 1.0, 0.5


def _pgraph(g, ds=1.0):
    """directed point graph (nodes every ds m along each polyline, edges along it and end -> start for every link) + unit tangent per node
    + alias mask: the first node of a piece with an incoming link sits on its predecessor's end node (same place); it is skipped as seed, snap target, hole and marble"""
    xy, tan, edges, first, last, off = [], [], [], {}, {}, 0
    for i, p in enumerate(g["pts"]):
        q = GE.resample(p, ds)
        if len(q) < 2:
            continue
        t = np.gradient(q, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        xy.append(q)
        tan.append(t)
        n = len(q)
        edges += [(off + k, off + k + 1) for k in range(n - 1)]
        first[i], last[i] = off, off + n - 1
        off += n
    if not xy:
        return np.zeros((0, 2)), np.zeros((0, 2)), csr_matrix((0, 0)), np.zeros(0, bool)
    xy, tan = np.concatenate(xy), np.concatenate(tan)
    links = [(last[i], first[j]) for i, j in g["edges"] if i in last and j in first]
    edges += links
    alias = np.zeros(len(xy), bool)
    alias[[b for _, b in links]] = True
    r, c = np.array([e[0] for e in edges]), np.array([e[1] for e in edges])
    w = np.linalg.norm(xy[c] - xy[r], axis=1) + 1e-3
    return xy, tan, csr_matrix((w, (r, c)), shape=(len(xy), len(xy))), alias


def _inside(xy):
    return (np.abs(xy) < CLIP).all(1) if len(xy) else np.zeros(0, bool)


def topo_counts(pred, gt):
    """(matched, marbles, holes) of one frame"""
    xg, tg, Ag, alg = _pgraph(gt)
    if not len(xg):
        return 0, 0, 0
    ing = _inside(xg) & ~alg
    seeds = [k for k in range(0, len(xg), STEP) if ing[k]]
    if not seeds:
        return 0, 0, 0
    Dg = dijkstra(Ag, directed=True, indices=seeds, limit=REACH)
    xp, tp, Ap, alp = _pgraph(pred)
    m_tot = n_mar = n_hol = 0
    if len(xp):
        inp = _inside(xp) & ~alp
        tree = cKDTree(xp)
        d, idx = tree.query(xg[seeds], k=min(6, len(xp)), distance_upper_bound=SNAP)
        d, idx = d.reshape(len(seeds), -1), idx.reshape(len(seeds), -1)
        match = []
        for s, k in enumerate(seeds):
            j = -1
            for dd, ii in zip(d[s], idx[s]):
                if np.isfinite(dd) and not alp[ii] and float(tp[ii] @ tg[k]) >= COS_SNAP:
                    j = int(ii)
                    break
            match.append(j)
        uniq = sorted({j for j in match if j >= 0})
        Dp = dijkstra(Ap, directed=True, indices=uniq, limit=REACH) if uniq else None
        row = {j: r for r, j in enumerate(uniq)}
    else:
        match = [-1] * len(seeds)
    for s in range(len(seeds)):
        holes = np.nonzero(np.isfinite(Dg[s]) & ing)[0]
        n_hol += len(holes)
        if match[s] < 0 or not len(holes):
            continue
        mar = np.nonzero(np.isfinite(Dp[row[match[s]]]) & inp)[0]
        n_mar += len(mar)
        if len(mar):
            m_tot += GE.match_1to1(xp[mar], xg[holes], cKDTree(xg[holes]), TOL)
    return m_tot, n_mar, n_hol


def _arc_margin(g, ds=0.5, clip=CLIP):
    """per dense point (same order as GE.dense_points): distance along its piece to the nearer piece end"""
    out = []
    for p in g["pts"]:
        q = GE.resample(p, ds)
        if len(q) < 2:
            continue
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
        out.append(np.minimum(s, s[-1] - s)[(np.abs(q) < clip).all(1)])
    return np.concatenate(out) if out else np.zeros(0)


def dup_counts(pred, r=0.75, cos_min=0.9, end_zone=4.0):
    """(duplicate points, points) of one frame, dense points every 0.5 m; points within end_zone m of a piece end are never counted
    (branches of a fork, also an un-linked pruned one, overlap their neighbour there by construction)"""
    P, D, I = GE.dense_points(pred)
    if len(P) < 2:
        return 0, len(P)
    margin = _arc_margin(pred)
    near = set()
    for i, j in pred["edges"]:
        near.add((i, j))
        near.add((j, i))
    succ, pred_of = {}, {}
    for i, j in pred["edges"]:
        succ.setdefault(i, set()).add(j)
        pred_of.setdefault(j, set()).add(i)
    for grp in list(succ.values()) + list(pred_of.values()):  # siblings: branches of one split / one merge
        for a in grp:
            for b in grp:
                if a != b:
                    near.add((a, b))
    tree = cKDTree(P)
    dup = 0
    for k, nb in enumerate(tree.query_ball_point(P, r)):
        if margin[k] < end_zone:
            continue
        for j in nb:
            if I[j] != I[k] and (int(I[k]), int(I[j])) not in near and abs(float(D[k] @ D[j])) > cos_min:
                dup += 1
                break
    return dup, len(P)


def range_counts(pred, gt, tau=1.0):
    """per row of NAMES (the range bins, then the far bin split at RING): (gt points within tau, gt points, pred points within tau, pred points)"""
    P, _, _ = GE.dense_points(pred)
    G, _, _ = GE.dense_points(gt)
    out = np.zeros((len(NAMES), 4))
    dg = cKDTree(P).query(G)[0] if len(P) and len(G) else np.full(len(G), np.inf)  # without ground truth the predicted points are all false (they count in the precision bins)
    dp = cKDTree(G).query(P)[0] if len(P) and len(G) else np.full(len(P), np.inf)
    P, G = P.reshape(-1, 2), G.reshape(-1, 2)
    rg, rp = np.linalg.norm(G, axis=1), np.linalg.norm(P, axis=1)
    cg, cp = np.abs(G).max(1), np.abs(P).max(1)
    far_g, far_p = rg >= BINS[-1][1], rp >= BINS[-1][1]
    sel = [((rg >= lo) & (rg < hi), (rp >= lo) & (rp < hi)) for _, lo, hi in BINS] + [(far_g & (cg < RING), far_p & (cp < RING)), (far_g & (cg >= RING), far_p & (cp >= RING))]
    for b, (kg, kp) in enumerate(sel):
        out[b] = (np.sum(dg[kg] <= tau), kg.sum(), np.sum(dp[kp] <= tau), kp.sum())
    return out


def frame_record(pred, gt):
    return dict(topo=np.array(topo_counts(pred, gt), float), dup=np.array(dup_counts(pred), float), rng=range_counts(pred, gt))


def pooled(recs):
    t = sum(r["topo"] for r in recs)
    u = sum(r["dup"] for r in recs)
    g = sum(r["rng"] for r in recs)
    P, R = t[0] / max(t[1], 1), t[0] / max(t[2], 1)
    out = {"topo_P": float(P), "topo_R": float(R), "topo_F1": float(2 * P * R / max(P + R, 1e-9)), "dup_frac": float(u[0] / max(u[1], 1))}
    for nm, row in zip(NAMES, g):  # NaN when a bin holds no points
        out[f"R@1_{nm}"] = float(row[0] / row[1]) if row[1] else float("nan")
        out[f"P@1_{nm}"] = float(row[2] / row[3]) if row[3] else float("nan")
    return out


def extra(preds, gts):
    return pooled([frame_record(p, g) for p, g in zip(preds, gts)])


def paired(recs_a, recs_b, scenes, B=1000, seed=0, keys=("topo_F1", "topo_R", "topo_P", "dup_frac", "R@1_far", "R@1_mid", "R@1_near", "R@1_far_inner", "R@1_far_ring")):
    """a - b (a = variant, b = control) with a 95 % interval over resampled scenes (same draw for both systems), like bootstrap.paired for the
    card metrics and with the same draws (the shared Evaluation.stats bootstrap). Draws without a number (a range bin with no point: NaN) are
    dropped; then the key carries n_valid. p_a_better = share of the valid draws with a - b > 0 (for dup_frac, lower is better)."""
    res = ST.paired_draws(lambda idx: pooled(recs_a if idx is None else [recs_a[i] for i in idx]),
                          lambda idx: pooled(recs_b if idx is None else [recs_b[i] for i in idx]), ST.scene_draws(scenes, B, seed), keys)
    out = {}
    for k, v in res.items():
        out[k] = dict(a=v["a"], b=v["b"], diff=v["diff"], lo=v["lo"], hi=v["hi"], p_a_better=v["p_gt0"])
        if v["n_valid"] < v["n_draws"]:
            out[k]["n_valid"] = v["n_valid"]
    return out


def _selftest():
    line = lambda y, x0, x1: np.stack([np.linspace(x0, x1, 95), np.full(95, y)], 1)  # noqa: E731
    gt = dict(pts=[line(0.0, -47, 0), line(0.0, 0, 47), np.stack([np.linspace(0, 30, 31), np.linspace(0, 12, 31)], 1)], edges=[(0, 1), (0, 2)])
    m = extra([gt], [gt])
    assert m["topo_F1"] > 0.999 and m["dup_frac"] < 1e-9, m
    broken = dict(pts=gt["pts"], edges=[])                      # all links cut: geometry identical, connectivity gone
    assert extra([broken], [gt])["topo_R"] < 0.8, extra([broken], [gt])
    rev = dict(pts=[p[::-1].copy() for p in gt["pts"]], edges=[(j, i) for i, j in gt["edges"]])
    assert extra([rev], [gt])["topo_F1"] < 0.05
    dup = dict(pts=gt["pts"] + [gt["pts"][1] + np.array([0.0, 0.3])], edges=gt["edges"])
    assert extra([dup], [gt])["dup_frac"] > 0.2
    far = dict(pts=[p[np.linalg.norm(p, axis=1) < 40] for p in gt["pts"]], edges=gt["edges"])
    e = extra([far], [gt])
    assert e["R@1_far"] < 0.2 and e["R@1_near"] > 0.99, e  # far points 40 - 41 m are still within 1 m of the cut lane
    lane = dict(pts=[np.stack([np.linspace(40, 47.5, 16), np.zeros(16)], 1)], edges=[])  # 40 - 47.5 m ahead: 10 inner points (< 45), 6 ring points
    cut = dict(pts=[np.stack([np.linspace(40, 43.8, 20), np.zeros(20)], 1), np.stack([np.linspace(45, 47.5, 6), np.full(6, 10.0)], 1)], edges=[])
    r = range_counts(cut, lane)
    assert r[3, 1] == 10 and r[4, 1] == 6 and r[3, 1] + r[4, 1] == r[2, 1], r  # inner + ring = far
    e = extra([cut], [lane])
    assert e["R@1_far_inner"] == 1.0 and e["R@1_far_ring"] == 0.0 and e["P@1_far_inner"] == 1.0 and e["P@1_far_ring"] == 0.0 and abs(e["R@1_far"] - 10 / 16) < 1e-12, e
    nothing = dict(pts=[], edges=[])  # a frame without ground truth: its predicted points are false in their range bin (they once entered no P@1_<bin>)
    junk = dict(pts=[np.stack([np.linspace(2, 10, 17), np.zeros(17)], 1)], edges=[])  # 17 points, all near
    nn = range_counts(gt, gt)[0, 3]
    assert list(range_counts(junk, nothing)[0]) == [0, 0, 0, 17] and range_counts(nothing, nothing).sum() == 0
    e = extra([gt, junk], [gt, nothing])
    assert abs(e["P@1_near"] - nn / (nn + 17)) < 1e-12 and e["P@1_far"] == 1.0 and e["R@1_near"] == 1.0, e
    print("topo_metrics selftest passed")


if __name__ == "__main__":
    _selftest()
