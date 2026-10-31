"""Metrics of SPARSE lane graphs, on top of graph_eval (one definition for every run, control and designs).

Input contract (same as graph_eval): a graph is dict(pts=[(n,2) float polylines in metres, ego frame, travel direction],
edges=[(i, j): end of piece i -> start of piece j], nodes=[vertex ids per polyline]  # optional, needed for the vertex count).
  graph_eval.evaluate(preds, gts)   geometry P/R/F1 at 0.5/1.0/1.5 m, 1:1 F1, lanes found, false length, split/merge P/R/F1, route, direction
  card(preds, gts)                  the metrics of KEEP + vertices per 100 m + J, Sp and the composite S (what the loop-16 tables report)
  composite(m)                      S = 0.30 F1@1 + 0.20 F1@0.5 + 0.20 J + 0.15 Sp + 0.15 (1 - FL)    J = mean of split/merge F1 at 6 m, Sp = min(1, 14.3 / vert100)
  frontier_scalars(curve)           F1@Sp17 and FL@R85 over a validity sweep
  graph_from_vertices(xy, edges)    vertex list + directed edges -> graph pieces (maximal chains between vertices with in/out degree != 1)
  vert100_dp(preds, eps)            vertices per 100 m after Douglas-Peucker: the sparsity of graphs WITHOUT vertex ids (rule graphs from dense heads), comparable with vert100 of vertex graphs
Moved here from scripts/p16_common.py (which re-exports these names); functions are unchanged.
"""
import numpy as np

import graph_eval as GE

DP_ORACLE = 14.3  # vertices per 100 m, Douglas-Peucker at 0.3 m on the ground truth (Imai-Iri floor about 13)


KEEP = ("P@0.5", "R@0.5", "F1@0.5", "P@1.0", "R@1.0", "F1@1.0", "P@1.5", "R@1.5", "F1@1.5", "F1o2o@1.0", "lanes_found", "false_len_frac", "split6_P", "split6_R", "split6_F1", "merge6_P", "merge6_R", "merge6_F1",
        "split_F1", "merge_F1", "dir_acc", "n_samples")


def vert100(preds):
    """unique vertices per 100 m of polyline length (E94 on TEST: 22.0)"""
    nv, tot = 0, 0.0
    for g in preds:
        ids = set()
        for k, p in enumerate(g["pts"]):
            q = np.asarray(p, np.float64)
            if len(q) < 2:
                continue
            tot += float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
            nd = g["nodes"][k] if g.get("nodes") is not None and k < len(g["nodes"]) else None
            if nd is None:
                ids.update((k, j) for j in range(len(q)))
            else:
                ids.update(int(i) for i in nd)
        nv += len(ids)
    return 100.0 * nv / max(tot, 1e-9)


def composite(m):
    """S = 0.30 F1@1 + 0.20 F1@0.5 + 0.20 J + 0.15 Sp + 0.15 (1 - FL)   (plan section 5)"""
    J = 0.5 * (m.get("split6_F1", 0.0) + m.get("merge6_F1", 0.0))
    Sp = min(1.0, DP_ORACLE / max(m.get("vert100", 1e9), 1e-9))
    return 0.30 * m["F1@1.0"] + 0.20 * m["F1@0.5"] + 0.20 * J + 0.15 * Sp + 0.15 * (1.0 - m["false_len_frac"]), J, Sp


def quality(m):
    """Q = (0.30 F1@1 + 0.20 F1@0.5 + 0.20 J + 0.15 (1 - FL)) / 0.85: S without the sparsity term, rescaled to 0..1.
    Sp is capped at 1 (vert100 <= 14.3), so a design at the cap can only gain through Q: compare designs at equal sparsity with Q and the constraint vert100 <= 14.3.
    Added 2026-10-24 (not part of the loop-16 plan; S itself is unchanged)."""
    J = 0.5 * (m.get("split6_F1", 0.0) + m.get("merge6_F1", 0.0))
    return (0.30 * m["F1@1.0"] + 0.20 * m["F1@0.5"] + 0.20 * J + 0.15 * (1.0 - m["false_len_frac"])) / 0.85


def card(preds, gts, with_route=False, extra=False):
    """all metrics of one run on the same frames: graph_eval + vertices per 100 m + J, Sp, S, Q.
    extra=True adds topo_metrics.extra (topo_P / R / F1, dup_frac, recall / precision by range); used by evaluation/evaluate.py"""
    r = GE.evaluate(preds, gts, with_route=with_route)
    m = {k: float(r[k]) for k in KEEP if k in r}
    m["vert100"] = vert100(preds)
    m["verts_per_frame"] = float(np.mean([len({int(i) for nd in (g.get("nodes") or []) for i in nd}) for g in preds]))
    m["S"], m["J"], m["Sp"] = composite(m)
    m["Q"] = quality(m)
    if extra:
        import topo_metrics as TM
        m.update(TM.extra(preds, gts))
    return m


def frontier_scalars(curve):
    """curve: list of dicts (thr, vert100, F1@1.0, R@1.0, false_len_frac) over the validity sweep.
    F1@Sp17 = best F1@1 among points with at most 17 vertices per 100 m (NaN if none); FL@R85 = false length fraction at point recall 0.85 (linear interpolation between sweep points, NaN if never reached)."""
    out = {}
    ok = [c for c in curve if c["vert100"] <= 17.0]
    out["F1@Sp17"] = max((c["F1@1.0"] for c in ok), default=float("nan"))
    pts = sorted([(c["R@1.0"], c["false_len_frac"]) for c in curve])
    fl = float("nan")
    for (r0, f0), (r1, f1) in zip(pts[:-1], pts[1:]):
        if r0 <= 0.85 <= r1 and r1 > r0:
            fl = f0 + (f1 - f0) * (0.85 - r0) / (r1 - r0)
    if np.isnan(fl) and pts and pts[0][0] >= 0.85:
        fl = pts[0][1]
    out["FL@R85"] = fl
    return out


def graph_from_vertices(xy, edges, order=None):
    """vertex list + directed edges (a -> b) -> graph_eval pieces: maximal chains between vertices with in-degree != 1 or out-degree != 1.
    A vertex with in = out = 1 is interior; a chain starts at every out-edge of a non-interior vertex and runs while its end is interior,
    so a fork 0-1-2 -> 2-3-4 / 2-5-6 gives [0,1,2], [2,3,4], [2,5,6] and a merge cuts the same way. A cycle of interior vertices only
    becomes one closed piece [a, ..., a] that starts at its smallest edge. Self-loops and repeated edges are dropped.
    Fixed 2026-10-31 (D24, review bug L1): before, a split vertex with one predecessor never started a chain, so each split gave stubs
    [3,4], [5,6] plus the two-vertex leftovers [2,3], [2,5], and a pure cycle gave one 2-vertex piece per edge.
    xy [n,2]; edges list of (a, b); `order` is unused (kept for old callers). Returns dict(pts, edges, nodes)."""
    n = len(xy)
    outs = [[] for _ in range(n)]
    ins = [[] for _ in range(n)]
    seen = set()
    for a, b in edges:
        a, b = int(a), int(b)
        if a == b or (a, b) in seen:
            continue
        seen.add((a, b))
        outs[a].append(b)
        ins[b].append(a)
    pieces = []
    used = set()

    def walk(s, nxt):
        chain = [s, nxt]
        used.add((s, nxt))
        while len(outs[chain[-1]]) == 1 and len(ins[chain[-1]]) == 1 and (chain[-1], outs[chain[-1]][0]) not in used:
            a, b = chain[-1], outs[chain[-1]][0]
            used.add((a, b))
            chain.append(b)
        pieces.append(chain)

    starts = [v for v in range(n) if outs[v] and (len(ins[v]) != 1 or len(outs[v]) != 1)]
    for s in starts:
        for nxt in outs[s]:
            if (s, nxt) not in used:
                walk(s, nxt)
    for a, b in sorted(seen):  # only cycles of interior vertices are left: one closed piece each
        if (a, b) not in used:
            walk(a, b)
    pts = [np.asarray([xy[v] for v in c], np.float64) for c in pieces]
    first = {}
    for i, c in enumerate(pieces):
        first.setdefault(c[0], []).append(i)
    pe = [(i, j) for i, c in enumerate(pieces) for j in first.get(c[-1], []) if j != i]
    return dict(pts=pts, edges=pe, nodes=[list(map(int, c)) for c in pieces])


def _dp_keep(q, eps):
    """indices kept by Douglas-Peucker (iterative) on the polyline q [n,2] with tolerance eps (m)"""
    n = len(q)
    if n <= 2:
        return list(range(n))
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        ab = q[b] - q[a]
        t = np.clip(((q[a + 1 : b] - q[a]) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        d = np.linalg.norm(q[a + 1 : b] - (q[a] + t[:, None] * ab), axis=1)
        k = int(np.argmax(d))
        if d[k] > eps:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return list(np.nonzero(keep)[0])


def vert100_dp(preds, eps=0.3, ds=0.5, clip=48.0, unique_ends=False):
    """vertices per 100 m of polyline length when every piece is simplified with Douglas-Peucker (tolerance eps m, samples every ds m, |x|,|y| < clip, pieces with < 3 samples skipped).
    Default = the definition of scripts/oracle_vertices.py that gave the oracle value 14.3 per 100 m on the ground truth: kept vertices of every piece, end points of neighbouring pieces counted per piece.
    unique_ends=True counts coinciding piece end points once (closer to vert100 of vertex graphs). For graphs WITHOUT vertex ids (rule graphs from dense heads), where vert100 would count every polyline point."""
    nv, tot = 0, 0.0
    for g in preds:
        ends = set()
        for p in g["pts"]:
            q = GE.resample(np.asarray(p, np.float64), ds)
            if len(q) < 3:
                continue
            q = q[(np.abs(q) < clip).all(1)]
            if len(q) < 3:
                continue
            tot += float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
            k = _dp_keep(q, eps)
            if unique_ends:
                nv += len(k) - 2
                ends.add((round(float(q[0, 0]), 1), round(float(q[0, 1]), 1)))
                ends.add((round(float(q[-1, 0]), 1), round(float(q[-1, 1]), 1)))
            else:
                nv += len(k)
        nv += len(ends)
    return 100.0 * nv / max(tot, 1e-9)


def card_any(preds, gts, with_route=False, extra=False):
    """card() for graphs of any producer: the vertex count comes from the node ids when every graph has them (learned builders), otherwise from vert100_dp
    (rule graphs of a dense head carry no vertex ids; card() would count every polyline point as a vertex and S would not be comparable)."""
    m = card(preds, gts, with_route, extra)
    if not all(g.get("nodes") is not None for g in preds):
        m["vert100"] = vert100_dp(preds)
        m["verts_per_frame"] = float("nan")
        m["S"], m["J"], m["Sp"] = composite(m)
    return m
