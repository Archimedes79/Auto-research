"""How much radial velocity does the radar Doppler carry for the moving GT vehicles? (data only, no model; review)
Radar detections of the sample's 0.5 s window inside / within 1 m of an observed GT vehicle box (front field of view, speed > 1 m/s):
median vr against the GT velocity projected on the line of sight."""
import math
import sys

import numpy as np

sys.path.insert(0, "evaluation")
import observe_run as ob  # noqa: E402
from evaluate_run import gt_boxes  # noqa: E402
from evaluation.metrics import in_front  # noqa: E402
from train import collate_cached  # noqa: E402

cfg, model, loss, step, ds, device = ob.load_all("runs/t1_r4_soa", "cpu")
rows, n_mov, n_with = [], 0, 0
step_i = max(1, len(ds) // 240)
for i in range(0, len(ds), step_i):
    try:
        item = ds[i]
    except KeyError:
        continue
    rad = item["radar"]
    x, y, vr = rad["x"].numpy(), rad["y"].numpy(), rad["vr"].numpy()
    for b in gt_boxes(item["gt"]["cur"]["meta"] and {"meta": item["gt"]["cur"]["meta"]}):
        pass
    for b in __import__("evaluation.metrics", fromlist=["x"]).gt_boxes_from_meta(item["gt"]["cur"]["meta"]["boxes"]):
        if b["group"] != "vehicle" or not in_front(b) or math.hypot(b["vx"], b["vy"]) < 1.0:
            continue
        n_mov += 1
        c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
        dx, dy = x - b["cx"], y - b["cy"]
        u, w = c * dx + s * dy, -s * dx + c * dy
        m = (np.abs(u) <= b["length"] / 2 + 1.0) & (np.abs(w) <= b["width"] / 2 + 1.0)
        if m.sum() == 0:
            continue
        n_with += 1
        r = math.hypot(b["cx"], b["cy"])
        gr = (b["vx"] * b["cx"] + b["vy"] * b["cy"]) / r
        rows.append((gr, float(np.median(vr[m])), int(m.sum()), math.hypot(b["vx"], b["vy"]), r))
a = np.array(rows)
print(f"moving observed vehicles in front view (speed > 1 m/s): {n_mov}; with radar detections in the window: {n_with} ({100 * n_with / max(n_mov, 1):.0f} %)")
print(f"detections per such vehicle: median {np.median(a[:, 2]):.0f}")
print("GT radial speed vs median Doppler vr: mean |GT radial| %.2f, mean |error| %.2f, correlation %.3f" %
      (np.abs(a[:, 0]).mean(), np.abs(a[:, 0] - a[:, 1]).mean(), np.corrcoef(a[:, 0], a[:, 1])[0, 1]))
rad_only = np.abs(a[:, 0]) > 1.0
print("vehicles with |GT radial| > 1 m/s: %d, Doppler error there %.2f m/s (relative %.0f %%)" %
      (rad_only.sum(), np.abs(a[rad_only, 0] - a[rad_only, 1]).mean(), 100 * np.abs(a[rad_only, 0] - a[rad_only, 1]).mean() / np.abs(a[rad_only, 0]).mean()))
