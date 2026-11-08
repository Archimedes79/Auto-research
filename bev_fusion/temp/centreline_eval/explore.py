"""exploration: oracle / shuffled / null numbers of the centre-line layer on real cached GT frames"""
import os, sys, time
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")]
from evaluation import bev_adapter as A, bev_gtcheck as GC
from evaluation.bev_task import BevSegMetric
from evalsuite import Prediction, Sample
from evalsuite.contracts import FrameRef
import losses as L
from lift_pd import Grid

torch.set_num_threads(2)
SPEC, SPEC_1M = L.layer_spec(False, True), L.layer_spec_1m(True)
grid = Grid()


def head(g, mode):
    T = L.build_layer_targets([g], L.LossConfig(centreline_1m=True))
    ch = []
    for name, kind, key in SPEC_1M:
        t = T[key]
        if kind in ("line", "centreline"):
            ch.append(20.0 * (2 * t - 1) if mode != "null" else torch.full_like(t, -20.0))
        elif kind == "dist":
            ch.append(t)
        else:
            ch.append(t[:, 0:1] if name.endswith("_cos") else t[:, 1:2])
    sem_1m = torch.cat(ch, 1)
    return {"sem": torch.zeros(1, len(SPEC), 120, 104), "sem_1m": sem_1m}


def out_of(p):
    prob, dist = A.predictions(p, 0, SPEC)
    p1, d1 = A.predictions_1m(p, 0, SPEC_1M)
    prob.update(p1)
    return {"prob": prob, "dist": dist, "mass": None, "dets": [], "detr": None, "vhat": None, "dir": d1}


n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
sig = GC.resolve_sig(None)
samples = GC.pick(n, sig, 0)
spec = A.task_spec(types_ns := type("M", (), {"spec": SPEC, "spec_1m": SPEC_1M})())
print(spec[-2:])
metric = BevSegMetric(grid, spec)
t0 = time.time()
res = {}
for mode in ("oracle", "shuffled", "null"):
    counts = []
    for i, (name, s) in enumerate(samples):
        sample = Sample(ref=FrameRef("gt", i, name), inputs=None, reference=s)
        src = samples[(i + 1) % len(samples)][1] if mode == "shuffled" else s
        outs = {dt: out_of(head(src[dt], mode)) for dt in s}
        counts.append(metric.frame(sample, Prediction(outputs={"dt": outs})))
    r = metric.reduce(counts)
    res[mode] = r
    print(mode, f"{time.time() - t0:.1f}s")
    for k in ("iou@0", "iou@0.5", "iou@1", "n_gt"):
        print("   ", k, r[f"fov/dt+0.00/centreline/all/all/{k}"])
    print("    angle", r["fov/dt+0.00/centreline:dir/all/all/angle_mae"], r["fov/dt+0.00/centreline:dir/all/all/n_gt"],
          "dt+0.5", r.get("fov/dt+0.50/centreline/all/all/iou@1"))
    print("    bins", [r[f"fov/dt+0.00/centreline/all/{b}/iou@1"] for b in ("0-20", "20-40", "40-60")],
          [r[f"fov/dt+0.00/centreline:dir/all/{b}/angle_mae"] for b in ("0-20", "20-40", "40-60")])
    print("    seen/out", r["seen/dt+0.00/centreline/all/all/iou@1"], r["out/dt+0.00/centreline/all/all/iou@1"])
