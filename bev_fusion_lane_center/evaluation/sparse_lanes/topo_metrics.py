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
  range     recall / precision at 1 m by distance from the ego: near < 20 m, mid 20 - 40 m, far >= 40 m (R@1_far is the open gap of loop 18).

    extra(preds, gts) -> dict(topo_P, topo_R, topo_F1, dup_frac, R@1_near, R@1_mid, R@1_far, P@1_near, P@1_mid, P@1_far)
Self-test: python topo_metrics.py
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

import graph_eval as GE

CLIP = 48.0
BINS = (("near", 0.0, 20.0), ("mid", 20.0, 40.0), ("far", 40.0, 1e9))
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
    """per bin: (gt points within tau, gt points, pred points within tau, pred points)"""
    P, _, _ = GE.dense_points(pred)
    G, _, _ = GE.dense_points(gt)
    out = np.zeros((len(BINS), 4))
    if not len(G):
        return out
    dg = cKDTree(P).query(G)[0] if len(P) else np.full(len(G), np.inf)
    dp = cKDTree(G).query(P)[0] if len(P) else np.zeros(0)
    rg, rp = np.linalg.norm(G, axis=1), np.linalg.norm(P, axis=1) if len(P) else np.zeros(0)
    for b, (_, lo, hi) in enumerate(BINS):
        kg, kp = (rg >= lo) & (rg < hi), (rp >= lo) & (rp < hi)
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
    for b, (nm, _, _) in enumerate(BINS):  # NaN when a bin holds no points
        out[f"R@1_{nm}"] = float(g[b, 0] / g[b, 1]) if g[b, 1] else float("nan")
        out[f"P@1_{nm}"] = float(g[b, 2] / g[b, 3]) if g[b, 3] else float("nan")
    return out


def extra(preds, gts):
    return pooled([frame_record(p, g) for p, g in zip(preds, gts)])


def paired(recs_a, recs_b, scenes, B=1000, seed=0, keys=("topo_F1", "topo_R", "topo_P", "dup_frac", "R@1_far", "R@1_mid", "R@1_near")):
    """a - b with a 95 % interval over resampled scenes (same draw for both systems), like bootstrap.paired for the card metrics"""
    scenes = np.asarray(scenes)
    us = np.unique(scenes)
    idx = {s: np.nonzero(scenes == s)[0] for s in us}
    rng = np.random.RandomState(seed)
    base_a, base_b = pooled(recs_a), pooled(recs_b)
    draws = {k: [] for k in keys}
    for _ in range(B):
        fr = np.concatenate([idx[s] for s in rng.choice(us, len(us), replace=True)])
        pa, pb = pooled([recs_a[i] for i in fr]), pooled([recs_b[i] for i in fr])
        for k in keys:
            draws[k].append(pa[k] - pb[k])
    out = {}
    for k in keys:
        d = np.asarray(draws[k], float)
        d = d[np.isfinite(d)]
        out[k] = dict(a=base_a[k], b=base_b[k], diff=base_a[k] - base_b[k], lo=float(np.percentile(d, 2.5)) if len(d) else float("nan"),
                      hi=float(np.percentile(d, 97.5)) if len(d) else float("nan"), p_a_better=float((d > 0).mean()) if len(d) else float("nan"))
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
    print("topo_metrics selftest passed")


if __name__ == "__main__":
    _selftest()
