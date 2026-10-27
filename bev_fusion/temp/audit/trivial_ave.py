"""One-off audit: the velocity error of the trivial prediction 'every object stands still' (perfect boxes, velocity 0) on the
val samples the run is evaluated on -> the AVE a velocity head has to beat. Usage: python temp/audit/trivial_ave.py runs/t1_r0 --n 600"""
import argparse
import os
import sys

sys.argv, _argv = [sys.argv[0], next(a for a in sys.argv[1:] if not a.startswith("-"))], sys.argv
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--n", type=int, default=600)
a = ap.parse_args(_argv[1:])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evaluation"))
sys.argv = [sys.argv[0], a.run]
import evaluate_run as E  # noqa: E402
from evaluation.metrics import DetMeter, report  # noqa: E402

os.chdir(E.ROOT)
cfg = E.TrainConfig(**__import__("json").load(open(os.path.join(a.run, "config.json"))))
cfg.seq_len, cfg.workers = 1, 0
if os.path.isdir("data/cache/radar"):
    cfg.radar_cache = "data/cache/radar"
_, val_ds = E.make_datasets(cfg)
ds = E.val_subset(val_ds, a.n)
det = DetMeter(classes=[g for g, _ in E.NUSC_GROUPS])
for i in range(len(ds)):
    g = ds[i]["gt"]["cur"]
    gb = E.gt_boxes(g)
    dets = [dict(b, vx=0.0, vy=0.0, score=1.0) for b in gb]
    det.add(dets, gb, dt=0.0)
f = report(seg=None, det=det)
for c in ("vehicle", "pedestrian", "two_wheeler"):
    print(c, "n_gt", f.get(f"det/dt+0.00/{c}/all/n_gt"), "AVE of 'standing still' (m/s):", f.get(f"det/dt+0.00/{c}/all/ave"))
