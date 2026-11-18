import argparse, sys, time, os
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import jtoken_eval as JE
QE = JE.QE; MS = QE.MS; LaneGraph = QE.LaneGraph
N = 60
a = argparse.Namespace(run="e367_b25_s0", model="model.pt", seed=0, split="train", every=3, drop=0.0, drop_seed=0, live=False, max=0)
r = JE.read_dump(a, JE.RUNS, 0)
cell = (0.3, 0.2, 0.6, False)
fr, tk, va, frames = r["fr"][:N], r["toks"][:N], r["valid"][:N], r["frames"][:N]
dec = [JE.decode_variant(f, v, cell[0], cell[2], cell[3], cell[1]) for f, v in zip(fr, va)]
preds = [JE.frame_prediction(t, f, "rec", cell[2]) for t, f in zip(tk, dec)]
def junc_only(pred, gt, ds=1.0, j_tol=3.0, box=None):
    gx, ge = gt.point_graph(ds); px, pe = pred.point_graph(ds)
    jp, jg = MS._junctions(px, pe), MS._junctions(gx, ge)
    if box is not None:
        u0, u1, v0, v1 = box
        inb = lambda a: a[(a[:, 0] >= u0) & (a[:, 0] <= u1) & (a[:, 1] >= v0) & (a[:, 1] <= v1)] if len(a) else a
        jp, jg = inb(jp), inb(jg)
    return np.array([len(jp), len(jg), len(MS.match_1to1(jp, jg, j_tol))], float)
graphs = [QE.readout(p, 0.5, 0.5)[0] for p in preds]
raws = [QE.to_raw(g) for g in graphs]
t_full = t_junc = 0.0; same = True
for raw, f in zip(raws, frames):
    gt = f.gt
    P, G = LaneGraph.of(raw), LaneGraph.of({"pts": gt["pts"], "edges": gt.get("edges")})
    t0 = time.perf_counter(); c = MS.topo_counts(P, G, box=QE.CFG.box, **QE.CFG.topo); t_full += time.perf_counter() - t0
    t0 = time.perf_counter(); j = junc_only(P, G, box=QE.CFG.box, **{k: v for k, v in QE.CFG.topo.items() if k in ("ds", "j_tol")}); t_junc += time.perf_counter() - t0
    same &= bool(np.array_equal(c["junc"], j))
print("CFG.topo", QE.CFG.topo)
print(f"MS.topo_counts full {1000*t_full/N:.1f} ms/frame, junction-only {1000*t_junc/N:.1f} ms/frame, junc identical on all {N}: {same}")
