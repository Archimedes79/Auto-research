"""L1 impact, isolated: the same cached raw outputs assembled with the OLD graph_from_vertices (HEAD) and the NEW one (working tree);
per registry name: pieces, an invariant check (same vertex-level edges and vertex ids per frame), and every metric of card_any(extra=True) that differs. Writes no card."""
import os, sys, json, subprocess, types
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import numpy as np
R = os.environ["BFLC_ROOT"]
import evaluate as EV
import p16_common as C
import sparse_metrics as SM
src = subprocess.run(["git", "-c", "safe.directory=*", "-C", R, "show", "HEAD:bev_fusion_lane_center/evaluation/sparse_lanes/sparse_metrics.py"], capture_output=True, text=True).stdout
OLD = types.ModuleType("sm_old"); exec(compile(src, "sm_old", "exec"), OLD.__dict__)
NEW = SM.graph_from_vertices
reg = EV.registry()
names = sys.argv[1:] or [n for n, e in reg.items() if e.get("kind") != "graphs"]
def edges_of(g):
    return sorted({(a, b) for nd in g["nodes"] for a, b in zip(nd[:-1], nd[1:])}), sorted({i for nd in g["nodes"] for i in nd})
out = {}
for name in names:
    e = reg[name]
    try:
        C.graph_from_vertices = OLD.graph_from_vertices
        g0, gts, _ = EV.graphs_of(name, e, "val", require_cache=True)
        C.graph_from_vertices = NEW
        g1, _, _ = EV.graphs_of(name, e, "val", require_cache=True)
    except RuntimeError as ex:
        print(f"{name}: skipped ({ex})"); continue
    finally:
        C.graph_from_vertices = NEW
    same = sum(edges_of(a) == edges_of(b) for a, b in zip(g0, g1))
    p0, p1 = sum(len(g["pts"]) for g in g0), sum(len(g["pts"]) for g in g1)
    two0 = sum(len(nd) == 2 for g in g0 for nd in g["nodes"]); two1 = sum(len(nd) == 2 for g in g1 for nd in g["nodes"])
    m0, m1 = SM.card_any(g0, gts, extra=True), SM.card_any(g1, gts, extra=True)
    diff = {k: (m0[k], m1[k]) for k in m0 if isinstance(m0[k], float) and k in m1 and not (m0[k] != m0[k] and m1[k] != m1[k]) and abs(m0[k] - m1[k]) > 1e-9}
    print(f"{name}: {len(g0)} frames, same vertex-level edges + ids in {same}/{len(g0)} | pieces {p0} -> {p1} (2-vertex pieces {two0} -> {two1}) | {len(diff)} of {len(m0)} metrics differ")
    for k, (x, y) in sorted(diff.items()):
        print(f"    {k:22s} {x:.4f} -> {y:.4f}  ({y - x:+.4f})")
    out[name] = dict(frames=len(g0), same=same, pieces=[p0, p1], two=[two0, two1], diff=diff)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "l1_impact.json"), "w"), indent=1)
