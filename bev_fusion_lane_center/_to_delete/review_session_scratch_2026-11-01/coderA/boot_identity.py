"""throw-away: the lane bootstrap numbers on a fixed synthetic input, written exactly (repr of every float) -> argv[1]"""
import json
import sys

import numpy as np

import bootstrap as SB
import topo_metrics as TM

rng = np.random.RandomState(123)


def line(y, x0, x1, n=60):
    return np.stack([np.linspace(x0, x1, n), np.full(n, y)], 1)


def gt_graph(k):
    pts = [line(-3.5 + 3.5 * j + 0.1 * k, -45, 45) for j in range(3)]
    pts.append(np.stack([np.linspace(0, 30, 31), np.linspace(0, 12, 31)], 1))
    return dict(pts=pts, edges=[(1, 3)])


def pred_graph(g, quality):
    pts = []
    for p in g["pts"]:
        if rng.rand() < 0.15 * (1 - quality):
            continue                                       # dropped lane
        q = p + rng.normal(0, 0.6 * (1.2 - quality), p.shape)
        if rng.rand() < 0.3:
            q = q[: int(len(q) * rng.uniform(0.4, 1.0))]   # truncated
        pts.append(q)
    if rng.rand() < 0.3 * (1 - quality):
        pts.append(g["pts"][0] + np.array([0.0, 0.4]))     # duplicate
    return dict(pts=pts, edges=[])


scene_names = ["scene-0103", "scene-0007", "scene-0550", "scene-0042", "scene-0911", "scene-0300", "scene-0001"]
scenes, gts, pa, pb = [], [], [], []
for i, s in enumerate(scene_names):
    for t in range(3 + i % 3):
        g = gt_graph(i + t)
        scenes.append(s)
        gts.append(g)
        pa.append(pred_graph(g, 0.8))
        pb.append(pred_graph(g, 0.6))
rec_a, rec_b = SB.frame_records(pa, gts), SB.frame_records(pb, gts)
ta = [TM.frame_record(p, g) for p, g in zip(pa, gts)]
tb = [TM.frame_record(p, g) for p, g in zip(pb, gts)]
out = {"n_frames": len(scenes),
       "interval_B200": SB.interval(rec_a, scenes, B=200),
       "interval_B60_seed5": SB.interval(rec_b, np.asarray(scenes), B=60, seed=5),
       "paired_B200": SB.paired(rec_a, rec_b, scenes, B=200),
       "paired_B50_seed3": SB.paired(rec_b, rec_a, scenes, B=50, seed=3),
       "topo_paired_B200": TM.paired(ta, tb, scenes, B=200),
       "topo_paired_B40_seed7": TM.paired(tb, ta, np.asarray(scenes), B=40, seed=7)}


def exact(o):
    if isinstance(o, dict):
        return {k: exact(v) for k, v in o.items()}
    if isinstance(o, (float, np.floating)):
        return repr(float(o))
    return o


with open(sys.argv[1], "w") as f:
    json.dump(exact(out), f, indent=1, sort_keys=False)
print("written", sys.argv[1], "frames", len(scenes), "S diff", out["paired_B200"]["S"]["diff"], out["paired_B200"]["S"]["lo"])
