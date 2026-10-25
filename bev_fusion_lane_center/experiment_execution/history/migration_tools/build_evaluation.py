"""Create evaluation/sparse_lanes/sparse_metrics.py (from p16_common.py) and evaluation/lane_grid/grid_eval.py (from dense_head.py + field_tools.py)
by cutting the function sources out verbatim (ast line ranges), so no formula is retyped. The originals are NOT changed here (patching is a separate, verified step).
Run: _run.sh build_evaluation.py"""
import ast
import os

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
SC = f"{R}/scripts"


def seg(path, names):
    src = open(path, encoding="utf-8").read()
    lines = src.splitlines()
    tree = ast.parse(src)
    out = []
    for nd in tree.body:
        if isinstance(nd, (ast.FunctionDef, ast.ClassDef)) and nd.name in names:
            start = (nd.decorator_list[0].lineno if nd.decorator_list else nd.lineno) - 1
            out.append((nd.name, "\n".join(lines[start : nd.end_lineno])))
    got = {n for n, _ in out}
    missing = set(names) - got
    assert not missing, (path, missing)
    order = {n: i for i, n in enumerate(names)}
    return [s for n, s in sorted(out, key=lambda t: order[t[0]])]


# ------------------------------------------------------------------ sparse_metrics
p16 = f"{SC}/p16_common.py"
funcs = seg(p16, ["vert100", "composite", "card", "frontier_scalars", "graph_from_vertices"])
const = [l for l in open(p16, encoding="utf-8").read().splitlines() if l.startswith("DP_ORACLE =")]
keep_start = next(i for i, l in enumerate(open(p16, encoding="utf-8").read().splitlines()) if l.startswith("KEEP = ("))
lines = open(p16, encoding="utf-8").read().splitlines()
keep_block = "\n".join(lines[keep_start : keep_start + 2])
head = '''"""Metrics of SPARSE lane graphs, on top of graph_eval (one definition for every run, control and designs).

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

'''
extra = '''

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


def card_any(preds, gts, with_route=False):
    """card() for graphs of any producer: the vertex count comes from the node ids when every graph has them (learned builders), otherwise from vert100_dp
    (rule graphs of a dense head carry no vertex ids; card() would count every polyline point as a vertex and S would not be comparable)."""
    m = card(preds, gts, with_route)
    if not all(g.get("nodes") is not None for g in preds):
        m["vert100"] = vert100_dp(preds)
        m["verts_per_frame"] = float("nan")
        m["S"], m["J"], m["Sp"] = composite(m)
    return m
'''
body = head + "\n".join(const) + "\n\n\n" + keep_block + "\n\n\n" + "\n\n\n".join(funcs) + "\n" + extra
open(f"{R}/evaluation/sparse_lanes/sparse_metrics.py", "w", encoding="utf-8").write(body)
print("sparse_metrics.py written:", len(body.splitlines()), "lines")

# ------------------------------------------------------------------ grid_eval
dh = f"{SC}/dense_head.py"
ft = f"{SC}/field_tools.py"
g1 = seg(dh, ["thin_batch", "skeleton_points", "geom_counts", "eval_probs", "load_gt_points"])
g2 = seg(ft, ["skeletons", "est_eval"])
gh = seg(f"{SC}/heat_align.py", ["gauss_blur"])
prf = seg(f"{SC}/centerline_estimator.py", ["prf"]) if False else None
ce_src = open(f"{SC}/centerline_estimator.py", encoding="utf-8").read().splitlines()
tree = ast.parse("\n".join(ce_src))
prf_src = None
for nd in ast.walk(tree):
    if isinstance(nd, ast.FunctionDef) and nd.name == "prf":
        prf_src = "\n".join(ce_src[nd.lineno - 1 : nd.end_lineno])
assert prf_src
# est_eval calls CE.prf and skeletons calls H.gauss_blur / uses dev, HALF globals: bind them in this module
g2 = [s.replace("CE.prf(", "prf(").replace("H.gauss_blur(", "gauss_blur(") for s in g2]
ghead = '''"""Metrics for GRID heads (dense lane-centre-line output on the 200 x 200 BEV raster, 0.5 m cells): prediction map -> thin skeleton -> points in metres -> point P/R/F1.

Pipeline (one definition for every dense head of the project):
  prob map [N,200,200]   (heat map in 0..1)         -> skeleton_points(prob, tau)         threshold, Zhang-Suen thinning, drop components < 30 px
  distance field [S,200,200] in metres (E19 ff.)   -> skeletons(D, thr, blur, low, ...)   threshold on the distance, optional blur / hysteresis / range-dependent threshold
  points P vs dense ground-truth points G (m)      -> geom_counts(P, G)                  counts for P/R/F1 at 0.5/1.0/1.5 m + greedy 1:1 matches (graph_eval.match_1to1)
  eval_probs(prob, gts, taus)  = threshold sweep on a probability map;  est_eval(D, gts, thr, ...) = the same on distance fields (prf counts, no 1:1)
The cut-out functions are unchanged copies of scripts/dense_head.py, scripts/field_tools.py, scripts/heat_align.py and scripts/centerline_estimator.py
(the originals import them from here after the verified patch). Coordinates: row = x, column = y, x = -50 + (row + 0.5) * 0.5 m.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

import graph_eval as GE

HOME = __import__("os").path.expanduser("~")
dev = "cuda"
HALF = 50.0
'''
gbody = ghead + "\n\n" + "\n\n\n".join(gh + [prf_src] + g1 + g2) + "\n"
open(f"{R}/evaluation/lane_grid/grid_eval.py", "w", encoding="utf-8").write(gbody)
print("grid_eval.py written:", len(gbody.splitlines()), "lines")
