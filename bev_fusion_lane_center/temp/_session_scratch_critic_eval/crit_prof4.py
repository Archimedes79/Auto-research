import argparse, sys, time, os
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import jtoken_eval as JE
QE = JE.QE; TM = QE.TM; SB = QE.SB; MS = QE.MS; LaneGraph = QE.LaneGraph
N = 30
a = argparse.Namespace(run="e367_b25_s0", model="model.pt", seed=0, split="train", every=3, drop=0.0, drop_seed=0, live=False, max=0)
r = JE.read_dump(a, JE.RUNS, 0)
cell = (0.3, 0.2, 0.6, False)
fr, tk, va, frames = r["fr"][:N], r["toks"][:N], r["valid"][:N], r["frames"][:N]
dec = [JE.decode_variant(f, v, cell[0], cell[2], cell[3], cell[1]) for f, v in zip(fr, va)]
preds = [JE.frame_prediction(t, f, "rec", cell[2]) for t, f in zip(tk, dec)]
T = {}
def tm(name, fn):
    t0 = time.perf_counter(); out = fn(); T[name] = T.get(name, 0) + time.perf_counter() - t0; return out
with JE.JT.lattice(JE.LATTICE):
    for f in frames: f.ref()
    for rep in range(2):
        for p, f in zip(preds, frames):
            g, bk = tm("readout", lambda: QE.readout(p, 0.5, 0.5))
            raw = QE.to_raw(g); gt = f.gt
            tm("SB.frame_records", lambda: SB.frame_records([raw], [gt]))
            tm("SB._vstats(dp)", lambda: SB._vstats(raw, False))
            tm("TM.topo_counts", lambda: TM.topo_counts(raw, gt))
            tm("TM.dup_counts", lambda: TM.dup_counts(raw))
            tm("TM.range_counts", lambda: TM.range_counts(raw, gt))
            tm("MS.topo_counts(junction)", lambda: MS.topo_counts(LaneGraph.of(raw), LaneGraph.of({"pts": gt["pts"], "edges": gt.get("edges")}), box=QE.CFG.box, **QE.CFG.topo))
            tm("section_counts", lambda: QE.section_counts(g.meta, f.ref()))
            tm("tier_counts", lambda: QE.tier_counts(raw, f.ref()))
            tm("node_counts", lambda: QE.node_counts(p, f.ref()))
            tm("assign_counts", lambda: QE.assign_counts(p, f.ref()))
            tm("frame_record_gt_side_pgraph(gt)", lambda: TM._pgraph(gt))
tot = sum(T.values())
for k, v in sorted(T.items(), key=lambda x: -x[1]):
    print(f"{k:<36} {1000*v/(2*N):7.1f} ms/frame  {100*v/tot:4.0f}%")
print("sum", round(1000 * tot / (2 * N), 1), "ms/frame")
# GT-side only cost: dense_points(gt)+cKDTree etc.
import graph_eval as GE
t0 = time.perf_counter()
for f in frames:
    GE.dense_points(f.gt)
print("dense_points(gt) ms/frame", round(1000 * (time.perf_counter() - t0) / N, 1))
