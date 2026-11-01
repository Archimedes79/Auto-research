"""The common lane-centre-line measures as ADDITIVE counts of one frame (numpy + scipy), and the formulas that turn summed counts
into numbers. Every dataset-level number is a function of summed counts, so it can be recomputed for any set of frames.

image level (the picture of a dense head, `image_counts` / `image_numbers`)
  strict pixel      IoU, Dice, precision, recall of the thresholded map against the one-cell-wide ground truth
  tolerant pixel    precision / recall / F1 where a cell counts when it lies within tau of the other set (the relaxed
                    correctness / completeness of road extraction), for every tau
  skeleton          the same on the thinned prediction (what a graph read from the picture would be)
  soft Dice         continuous, on the probabilities
  sweep             tolerant precision / recall for a list of thresholds -> average precision (area under the tolerant PR curve),
                    best F1 and its threshold (threshold free)
sparse level (polylines, `geometry_counts` / `geometry_numbers`)
  P / R / F1 at tau  dense points every ds metres, a point is right when the nearest point of the other graph is within tau
  chamfer            mean clipped distance pred -> gt (accuracy), gt -> pred (completeness) and their mean
  lanes found        ground-truth polylines of which >= found_frac of the points are within tau
  1:1 F1, dup_frac   one-to-one matched points (a parallel duplicate does not count twice); dup_frac = share of matched-by-nearest
                     prediction points that have no 1:1 partner
  direction          share of matched points whose tangents agree (directed graphs only)
  sparsity           vertices per 100 m, length ratio
topology (`topo_counts` / `topo_numbers`)
  TOPO               marbles: from seeds every 5 m on the ground truth, everything reachable along the graph within 30 m, compared
                     with what the prediction reaches from the snapped seed (precision / recall / F1 at 2 m)  [Biagioni & Eriksson]
  junction F1        forks / merges matched one-to-one within 3 m
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from .geometry import LaneGraph
from .raster import Grid, edt, thin

Counts = Dict[str, np.ndarray]
_E = 1e-12


def tkey(t: float) -> str:
    """tolerance as it appears in a key: 0.5 -> '0.5', 1 -> '1.0', 1.25 -> '1.25'"""
    return f"{t:.1f}" if abs(t * 10 - round(t * 10)) < 1e-9 else f"{t:g}"


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def prf(ok_p: float, n_p: float, ok_g: float, n_g: float) -> Tuple[float, float, float]:
    """precision, recall, F1 with the vacuous conventions: nothing predicted -> precision 1 (nothing false), nothing to find ->
    recall 1; a frame with neither is perfect"""
    p = ok_p / n_p if n_p > 0 else 1.0
    r = ok_g / n_g if n_g > 0 else 1.0
    return p, r, f1(p, r) if (n_p > 0 or n_g > 0) else 1.0


def _arr(*x) -> np.ndarray:
    return np.asarray(x, np.float64)


def nn_dist(P: np.ndarray, G: np.ndarray) -> np.ndarray:
    """distance of every point of P to the nearest point of G (inf when G is empty)"""
    if len(P) == 0:
        return np.zeros(0)
    if len(G) == 0:
        return np.full(len(P), np.inf)
    from scipy.spatial import cKDTree
    return cKDTree(G).query(P)[0]


def nn_index(P: np.ndarray, G: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(P) == 0 or len(G) == 0:
        return np.full(len(P), np.inf), np.full(len(P), -1)
    from scipy.spatial import cKDTree
    return cKDTree(G).query(P)


def match_1to1(P: np.ndarray, G: np.ndarray, tau: float, K: int = 8) -> np.ndarray:
    """greedy one-to-one matching of points within tau, nearest pairs first -> (n_pairs, 2) indices (P, G)"""
    if len(P) == 0 or len(G) == 0:
        return np.zeros((0, 2), int)
    from scipy.spatial import cKDTree
    k = min(K, len(G))
    d, i = cKDTree(G).query(P, k=k, distance_upper_bound=tau)
    d, i = d.reshape(len(P), k), i.reshape(len(P), k)
    pi, kj = np.nonzero(np.isfinite(d))
    if not len(pi):
        return np.zeros((0, 2), int)
    order = np.argsort(d[pi, kj], kind="stable")
    usedP, usedG, out = set(), set(), []
    for a in order:
        p, g = int(pi[a]), int(i[pi[a], kj[a]])
        if p in usedP or g in usedG:
            continue
        usedP.add(p); usedG.add(g); out.append((p, g))
    return np.asarray(out, int).reshape(-1, 2)


# ----------------------------------------------------------------------------------------------------------------- sparse
def geometry_counts(pred: LaneGraph, gt: LaneGraph, ds: float, taus: Sequence[float], clip: float, found_frac: float,
                    box=None) -> Counts:
    """counts of one frame for the sparse geometry measures; `box` = (u0, u1, v0, v1): only dense points inside are scored"""
    dp, dg = pred.dense(ds, box), gt.dense(ds, box)
    P, G = dp.xy, dg.xy
    dpg, dgp = nn_dist(P, G), nn_dist(G, P)
    ft = taus[min(1, len(taus) - 1)]
    c = {"n": _arr(len(P), len(G)),
         "p_ok": np.array([(dpg <= t).sum() for t in taus], np.float64),
         "g_ok": np.array([(dgp <= t).sum() for t in taus], np.float64),
         "cham": _arr(np.minimum(dpg, clip).sum() if len(P) else 0.0, np.minimum(dgp, clip).sum() if len(G) else 0.0)}
    found = tot = 0
    for ln in np.unique(dg.line):
        k = dg.line == ln
        tot += 1
        found += int((dgp[k] <= ft).mean() >= found_frac)
    c["lanes"] = _arr(found, tot)
    pairs = match_1to1(P, G, ft)
    c["m11"] = _arr(len(pairs), int((dpg <= ft).sum()))                  # one-to-one matches, nearest-matched prediction points
    if pred.directed and gt.directed and len(pairs):
        cos = (dp.tan[pairs[:, 0]] * dg.tan[pairs[:, 1]]).sum(1)
        c["dir"] = _arr((cos > 0).sum(), len(pairs))
    else:
        c["dir"] = _arr(0, 0)
    c["size"] = _arr(pred.length(), gt.length(), pred.vertices(), len(pred), len(gt))
    return c


def geometry_numbers(c: Counts, taus: Sequence[float], clip: float) -> Dict[str, float]:
    n_p, n_g = c["n"]
    out: Dict[str, float] = {}
    for k, t in enumerate(taus):
        p, r, f = prf(c["p_ok"][k], n_p, c["g_ok"][k], n_g)
        out[f"P@{tkey(t)}"], out[f"R@{tkey(t)}"], out[f"F1@{tkey(t)}"] = p, r, f
    acc = c["cham"][0] / n_p if n_p > 0 else (0.0 if n_g == 0 else clip)
    comp = c["cham"][1] / n_g if n_g > 0 else (0.0 if n_p == 0 else clip)
    out.update({"chamfer": 0.5 * (acc + comp), "chamfer_acc": acc, "chamfer_comp": comp})
    out["lanes_found"] = c["lanes"][0] / c["lanes"][1] if c["lanes"][1] > 0 else 1.0
    m, near = c["m11"]
    p11 = m / n_p if n_p > 0 else 1.0
    r11 = m / n_g if n_g > 0 else 1.0
    out["F1_1to1"] = f1(p11, r11) if (n_p or n_g) else 1.0
    out["dup_frac"] = 1.0 - m / near if near > 0 else 0.0
    out["dir_acc"] = c["dir"][0] / c["dir"][1] if c["dir"][1] > 0 else float("nan")
    pl, gl, nv = c["size"][:3]
    out["vert100"] = 100.0 * nv / pl if pl > 0 else 0.0
    out["len_ratio"] = pl / gl if gl > 0 else (0.0 if pl == 0 else float("inf"))
    return out


# ------------------------------------------------------------------------------------------------------------------ topology
def _reach(xy: np.ndarray, ed: np.ndarray, seeds: np.ndarray, L: float) -> np.ndarray:
    """boolean (S, N): nodes reachable from every seed within path length L along directed edges"""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    n = len(xy)
    ed = np.unique(ed, axis=0) if len(ed) else ed                       # csr_matrix would add the weights of duplicate edges
    w = np.linalg.norm(xy[ed[:, 0]] - xy[ed[:, 1]], axis=1) + 1e-6 if len(ed) else np.zeros(0)
    W = csr_matrix((w, (ed[:, 0], ed[:, 1])), shape=(n, n)) if len(ed) else csr_matrix((n, n))
    return np.isfinite(dijkstra(W, directed=True, indices=seeds, limit=L))


def _symmetric(ed: np.ndarray) -> np.ndarray:
    return np.vstack([ed, ed[:, ::-1]]) if len(ed) else ed


def _junctions(xy: np.ndarray, ed: np.ndarray) -> np.ndarray:
    if not len(ed):
        return np.zeros((0, 2))
    e = np.unique(_symmetric(ed), axis=0)
    deg = np.bincount(e[:, 0], minlength=len(xy))
    return xy[deg >= 3]


def topo_counts(pred: LaneGraph, gt: LaneGraph, ds: float = 1.0, seed_step: float = 5.0, L: float = 30.0, snap: float = 2.0,
                r: float = 2.0, j_tol: float = 3.0, box=None) -> Counts:
    """TOPO marbles and junctions of one frame. Directed when both graphs are (travel direction respected), else undirected"""
    gx, ge = gt.point_graph(ds)
    px, pe = pred.point_graph(ds)
    undirected = not (gt.directed and pred.directed)
    if undirected:
        ge, pe = _symmetric(ge), _symmetric(pe)
    out = {"marb": np.zeros(4), "junc": np.zeros(3)}
    sd = gt.dense(seed_step, box)
    if len(sd) and len(gx):
        seeds = np.unique(nn_index(sd.xy, gx)[1])
        Rg = _reach(gx, ge, seeds, L)
        d, pj = nn_index(gx[seeds], px) if len(px) else (np.full(len(seeds), np.inf), np.full(len(seeds), -1))
        ok_seed = d <= snap
        Rp = _reach(px, pe, pj[ok_seed], L) if ok_seed.any() else np.zeros((0, len(px)), bool)
        k = 0
        for s in range(len(seeds)):
            G = gx[Rg[s]]
            if not ok_seed[s]:
                out["marb"][0] += len(G)
                continue
            P = px[Rp[k]]
            k += 1
            out["marb"] += _arr(len(G), (nn_dist(G, P) <= r).sum(), len(P), (nn_dist(P, G) <= r).sum())
    jp, jg = _junctions(px, pe), _junctions(gx, ge)
    if box is not None:
        u0, u1, v0, v1 = box
        inb = lambda a: a[(a[:, 0] >= u0) & (a[:, 0] <= u1) & (a[:, 1] >= v0) & (a[:, 1] <= v1)] if len(a) else a
        jp, jg = inb(jp), inb(jg)
    out["junc"] = _arr(len(jp), len(jg), len(match_1to1(jp, jg, j_tol)))
    return out


def topo_numbers(c: Counts) -> Dict[str, float]:
    g_m, g_ok, p_m, p_ok = c["marb"]
    # recall: reachable ground-truth marbles that the prediction also reaches; precision: predicted marbles that exist in the ground truth
    p, r, f = prf(p_ok, p_m, g_ok, g_m)
    n_p, n_g, tp = c["junc"]
    jp, jr, jf = prf(tp, n_p, tp, n_g)
    return {"topo_P": p, "topo_R": r, "topo_F1": f, "junction_P": jp, "junction_R": jr, "junction_F1": jf}


# ------------------------------------------------------------------------------------------------------------------- image
def image_counts(prob: np.ndarray, gt: np.ndarray, grid: Grid, thr: float, taus: Sequence[float], sweep: Sequence[float],
                 valid=None, d_gt: np.ndarray = None) -> Counts:
    """counts of one frame for the image-level measures. `prob` (probabilities, or a 0/1 mask), `gt` bool one-cell-wide lines,
    `valid` bool evaluated cells (default all)"""
    prob = np.asarray(prob, np.float32)
    gt = np.asarray(gt, bool)
    v = np.ones(gt.shape, bool) if valid is None else np.asarray(valid, bool)
    pred = prob >= thr
    dgt = edt(gt, grid) if d_gt is None else d_gt
    dpr = edt(pred, grid)
    sk = thin(pred)
    dsk = edt(sk, grid)
    c = {"px": _arr((pred & v).sum(), (gt & v).sum(), (pred & gt & v).sum()),
         "rel_p": np.array([(pred & v & (dgt <= t)).sum() for t in taus], np.float64),
         "rel_g": np.array([(gt & v & (dpr <= t)).sum() for t in taus], np.float64),
         "sk_n": _arr((sk & v).sum()),
         "sk_p": np.array([(sk & v & (dgt <= t)).sum() for t in taus], np.float64),
         "sk_g": np.array([(gt & v & (dsk <= t)).sum() for t in taus], np.float64),
         "soft": _arr((prob * gt * v).sum(), (prob * v).sum(), (gt * v).sum())}
    t1 = taus[min(1, len(taus) - 1)]
    sp, sn, sg = [], [], []
    for th in sweep:
        m = prob >= th
        d = edt(m, grid)
        sp.append((m & v & (dgt <= t1)).sum()); sn.append((m & v).sum()); sg.append((gt & v & (d <= t1)).sum())
    c["sw_p"], c["sw_n"], c["sw_g"] = (np.asarray(x, np.float64) for x in (sp, sn, sg))
    return c


def image_numbers(c: Counts, taus: Sequence[float], sweep: Sequence[float]) -> Dict[str, float]:
    n_p, n_g, tp = c["px"]
    out = {"iou": tp / (n_p + n_g - tp) if (n_p + n_g - tp) > 0 else 1.0,
           "dice": 2 * tp / (n_p + n_g) if (n_p + n_g) > 0 else 1.0,
           "px_P": tp / n_p if n_p > 0 else 1.0, "px_R": tp / n_g if n_g > 0 else 1.0}
    sg, sp, sgt = c["soft"]
    out["soft_dice"] = 2 * sg / (sp + sgt) if (sp + sgt) > 0 else 1.0
    for k, t in enumerate(taus):
        tk = tkey(t)
        p, r, f = prf(c["rel_p"][k], n_p, c["rel_g"][k], n_g)
        out[f"P_rel@{tk}"], out[f"R_rel@{tk}"], out[f"F1_rel@{tk}"] = p, r, f
        den = n_g + n_p - c["rel_p"][k]                         # quality (Wiedemann): matched gt / (gt + false prediction)
        out[f"quality_rel@{tk}"] = c["rel_g"][k] / den if den > 0 else 1.0
        p, r, f = prf(c["sk_p"][k], c["sk_n"][0], c["sk_g"][k], n_g)
        out[f"P_skel@{tk}"], out[f"R_skel@{tk}"], out[f"F1_skel@{tk}"] = p, r, f
    P = np.array([c["sw_p"][i] / c["sw_n"][i] if c["sw_n"][i] > 0 else 1.0 for i in range(len(sweep))])
    R = np.array([c["sw_g"][i] / n_g if n_g > 0 else 1.0 for i in range(len(sweep))])
    F = np.array([f1(P[i], R[i]) for i in range(len(sweep))])
    # thresholds ascending -> recall descending; precision envelope = best precision at this recall or higher (index <= i)
    env = np.maximum.accumulate(P) if len(P) else P
    out["AP_rel"] = float(np.sum((R - np.r_[R[1:], 0.0]) * env)) if len(P) else 0.0
    k = int(np.argmax(F)) if len(F) else 0
    out["best_F1_rel"], out["best_thr"] = (float(F[k]), float(sweep[k])) if len(F) else (0.0, float("nan"))
    return out
