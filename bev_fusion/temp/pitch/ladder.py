"""The maps of every stepping-stone run in one table, from the stored eval files (cold VAL, `seg_fov`, dt 0). Nothing typed by hand.
Usage (bev_fusion folder, gcar python): python temp/pitch/ladder.py [out.md]"""
import json
import os
import sys

RUNS = [("t2_ref_s0", "start: DualBEV heads + radar, 3000 steps"), ("t2_ref_s1", "seed twin of the start"), ("t4_fpn_order", "FPN order fixed (tie)"),
        ("t3_ref_long", "start, 6000 steps"), ("t1_r4_soa", "start, 7000 x 2 steps"), ("t7_res4_s0", "+ stride-4 branch (frozen ResNet)"),
        ("t7_rgb_s0", "+ stride-4 branch with RGB"), ("t8_ctl_s0", "control for t8 / t10 (same code)"), ("t8_ctl_s1", "control, seed 1"),
        ("t8_bevfeat_s0", "BEVFusion map-trained features, seed 0"), ("t8_bevfeat_s1", "BEVFusion map-trained features, seed 1"),
        ("t10_dual_s0", "both feature sets, seed 0"), ("t10_dual_s1", "both feature sets, seed 1")]
LAYERS = [("drivable", "drivable_area", "iou@0"), ("walkway", "walkway", "iou@0"), ("carpark", "carpark_area", "iou@0"),
          ("crossing", "ped_crossing", "iou@0"), ("lane div. @0.5 m", "lane_divider", "iou@0.5"), ("road div. @0.5 m", "road_divider", "iou@0.5")]


def get(m, key):
    d = m.get("metrics", m)
    return d.get(key)


rows, out = [], []
for run, what in RUNS:
    p = f"runs/{run}/eval/metrics.json"
    if not os.path.exists(p):
        continue
    m = json.load(open(p))
    vals = [get(m, f"seg_fov/seg/dt+0.00/{layer}/all/all/{tol}") for _, layer, tol in LAYERS]
    step = get(m, "info/step")
    n = get(m, "info/n_samples")
    rows.append((run, what, step, n, m.get("gt_signature", ""), vals))
head = "| run | what | step | frames | " + " | ".join(n for n, _, _ in LAYERS) + " |"
out.append(head)
out.append("|" + "---|" * (4 + len(LAYERS)))
for run, what, step, n, sig, vals in rows:
    out.append(f"| {run} | {what} | {step} | {n} | " + " | ".join("-" if v is None else f"{v:.3f}" for v in vals) + " |")
text = "\n".join(out)
print(text)
if len(sys.argv) > 1:
    open(sys.argv[1], "w", encoding="utf-8").write(text + "\n")
