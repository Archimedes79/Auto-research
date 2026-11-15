"""scratch: how much of the lane-line IoU is capped by label scatter? A prediction that sits exactly on the true paint, scored against the labels
under the measured label-to-paint offsets (diag.paint_val, per-frame best lateral offsets): GT as prediction, rolled by k cells (0.5 m) across the road,
through the real metrics (evaluation/bev_gtcheck.py functions), then averaged over the measured |offset| distribution. CPU only."""
import json
import os
import sys
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = "/mnt/c/Auto-research/bev_fusion"
sys.path[:0] = [ROOT, f"{ROOT}/training", "/mnt/c/Auto-research", "/mnt/c/Auto-research/autoresearch"]
from evaluation import bev_gtcheck as G

paint = json.load(open(f"{ROOT}/temp/lane_tests/paint_out/paint_result.json"))
pf = paint["per_frame_rows"]
print("per_frame_rows type:", type(pf).__name__, list(pf)[:3] if isinstance(pf, dict) else len(pf))
offs = {}
if isinstance(pf, dict):
    for layer, rows in pf.items():
        offs[layer] = np.array([abs(r["d_best"]) for r in rows if r.get("d_best") is not None])
else:
    for r in pf:
        offs.setdefault(r.get("layer", "?"), []).append(abs(r["d_best"]))
    offs = {k: np.array(v) for k, v in offs.items()}
for k, v in offs.items():
    print(f"  {k}: {len(v)} frames, mean |d| {v.mean():.2f} m, median {np.median(v):.2f}, share > 0.5 m {np.mean(v > 0.5):.2f}, share > 1 m {np.mean(v > 1.0):.2f}")

task = G.build_task()
samples = G.pick(48, "6a7aa7864a", 0)
print("frames:", len(samples))
orc = [{dt: G.oracle_output(g, task.spec, task.groups) for dt, g in s.items()} for _, s in samples]
KEYS = {"lane IoU@0.5": "seg/fov/dt+0.00/lane_divider/all/all/iou@0.5", "lane IoU@1": "seg/fov/dt+0.00/lane_divider/all/all/iou@1",
        "road IoU@1": "seg/fov/dt+0.00/road_divider/all/all/iou@1", "road IoU@0.5": "seg/fov/dt+0.00/road_divider/all/all/iou@0.5"}
tab = {}
for k in range(0, 6):
    outs = orc if k == 0 else [{dt: G.shifted(o, k) for dt, o in s.items()} for s in orc]
    res = G.run(samples, outs, task)
    tab[k] = {n: res.get(key, float("nan")) for n, key in KEYS.items()}
    print(f"shift {k} cells = {0.5 * k:.1f} m:", {n: round(v, 3) for n, v in tab[k].items()})

print("\nexpected ceiling under the measured offsets (a prediction exactly on the true paint, scored against the labels):")
for layer, v in offs.items():
    ks = np.clip(np.rint(v / 0.5).astype(int), 0, 5)
    for n in KEYS:
        if layer.startswith("lane") and n.startswith("lane") or layer.startswith("road") and n.startswith("road"):
            e = np.mean([tab[k][n] for k in ks])
            print(f"  {layer:14s} {n:13s} ceiling {e:.3f}   (shift 0: {tab[0][n]:.3f})")
