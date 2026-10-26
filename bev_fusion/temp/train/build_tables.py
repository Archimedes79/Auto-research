"""Result tables of experiment t1 from the metric files (nothing typed by hand): baseline run vs combined run (VAL, EMA weights).
Usage (from the bev_fusion folder): python temp/train/build_tables.py runs/t1_r0 runs/t1_r3_combined > experiments/t1_full_chain/results/tables.md
Inputs: <run>/eval/metrics.json (evaluation/evaluate_run.py), <final>/eval/memory.json (eval_memory.py, optional), the paired
scene bootstrap (evaluate_run.compare) and the logs. All numbers: VAL (82 scenes, 3109 keyframes), camera cells, 0.5 m grid."""
import json
import math
import os
import sys

sys.path.insert(0, ".")
from evaluation import evaluate_run as ev  # noqa: E402

base, final = sys.argv[1], sys.argv[2]
B = json.load(open(os.path.join(base, "eval", "metrics.json")))
F = json.load(open(os.path.join(final, "eval", "metrics.json")))
bm, fm = B["metrics"], F["metrics"]
has_ref = any(k.startswith("detr/") for k in fm)
mem = os.path.join(final, "eval", "memory.json")
M = json.load(open(mem))["metrics"] if os.path.exists(mem) else None


def g(m, key):
    v = m.get(key)
    return float("nan") if v is None else v


def f(v, d=3):
    return "-" if v != v else f"{v:.{d}f}"


def row(label, key, d=3, prior=None, lower_better=False):
    b, c = g(bm, key), g(fm, key)
    p = "" if prior is None else f" {f(g(bm, prior), d)} |"
    delta = "-" if (b != b or c != c) else f"{c - b:+.{d}f}"
    return f"| {label} |{p} {f(b, d)} | {f(c, d)} | {delta} |"


print(f"# t1 results: baseline `{base}` (step {int(bm['info/step'])}, window-only, plain heading head) vs combined `{final}` "
      f"(step {int(fm['info/step'])})\n")
print(f"VAL split, {int(fm['info/n_samples'])} keyframes of {int(fm['info/n_scenes'])} scenes, EMA weights, camera cells (maps), front field of "
      f"view <= 60 m (boxes), 0.5 m grid; every number from `runs/<run>/eval/metrics.json`.\n")

print("## 1 Maps at the current time (dt = 0)\n")
print("| metric | prior map | baseline | combined | combined - baseline |\n|---|---|---|---|---|")
for label, key in (("drivable area IoU@0", "drivable_area/all/all/iou@0"), ("drivable area IoU@0.5 m", "drivable_area/all/all/iou@0.5"),
                   ("walkway IoU@0", "walkway/all/all/iou@0"), ("carpark IoU@0", "carpark_area/all/all/iou@0"),
                   ("pedestrian crossing IoU@0", "ped_crossing/all/all/iou@0"),
                   ("lane-divider mask IoU@0.5 m", "lane_divider/all/all/iou@0.5"), ("lane-divider mask IoU@1 m", "lane_divider/all/all/iou@1"),
                   ("road-divider mask IoU@1 m", "road_divider/all/all/iou@1"),
                   ("lane-divider zero-crossing F1@0.5 m", "lane_divider:zc/all/all/f1@0.5"),
                   ("road-divider zero-crossing F1@0.5 m", "road_divider:zc/all/all/f1@0.5"),
                   ("lane-divider distance-field error near the line (m, lower is better)", "lane_divider:zc/all/all/field_mae")):
    k = "seg_fov/seg/dt+0.00/" + key
    print(row(label, k, prior="prior_fov/seg/dt+0.00/" + key))

print("\n## 2 Time offsets: the same maps decoded at t - 0.5 s, t, t + 0.5 s (drivable area IoU@0, vehicle AP@2 m)\n")
print("| quantity | dt | baseline | combined | combined - baseline |\n|---|---|---|---|---|")


def rowdt(label, dt, kb, kc, d=3):
    b, c = g(bm, kb), g(fm, kc)
    return f"| {label} | {dt} | {f(b, d)} | {f(c, d)} | {'-' if b != b or c != c else format(c - b, '+.%df' % d)} |"


for dt in ("dt-0.50", "dt+0.00", "dt+0.50"):
    k = f"seg_fov/seg/{dt}/drivable_area/all/all/iou@0"
    print(rowdt("drivable area IoU@0", dt, k, k))
for dt in ("dt-0.50", "dt+0.00", "dt+0.50"):
    k = f"det/{dt}/vehicle/all/ap@2"
    print(rowdt("vehicle AP@2 m", dt, k, k))
    if has_ref:
        print(rowdt("vehicle AP@2 m, combined with refined boxes", dt, k, f"detr/{dt}/vehicle/all/ap@2"))
for dt in ("dt-0.50", "dt+0.00", "dt+0.50"):
    k = f"det/{dt}/mean/all/mAP"
    print(rowdt("mAP (3 classes)", dt, k, k))

print("\n## 3 Objects at dt = 0 (nuScenes-style, centre-distance AP; TP errors at 2 m)\n")
print("| metric | baseline | combined | combined, refined boxes | combined - baseline (plain) |\n|---|---|---|---|---|")
for label, key, d in (("mAP (3 classes, 4 thresholds)", "mean/all/mAP", 3), ("vehicle AP@2 m", "vehicle/all/ap@2", 3),
                      ("vehicle AP@4 m", "vehicle/all/ap@4", 3), ("vehicle centre error ATE (m)", "vehicle/all/ate", 2),
                      ("vehicle size error ASE (1 - IoU)", "vehicle/all/ase", 3), ("vehicle heading error AOE (rad)", "vehicle/all/aoe", 2),
                      ("vehicle velocity error AVE (m/s)", "vehicle/all/ave", 2), ("pedestrian AP@2 m", "pedestrian/all/ap@2", 3),
                      ("pedestrian centre error ATE (m)", "pedestrian/all/ate", 2), ("two-wheeler AP@2 m", "two_wheeler/all/ap@2", 3)):
    k = "det/dt+0.00/" + key
    r = f(g(fm, "detr/dt+0.00/" + key), d) if has_ref else "-"
    b, c = g(bm, k), g(fm, k)
    print(f"| {label} | {f(b, d)} | {f(c, d)} | {r} | {'-' if b != b or c != c else format(c - b, '+.%df' % d)} |")
print("\n| coverage IoU@0 (dense object layer) | baseline | combined | combined - baseline |\n|---|---|---|---|")
for label, cls in (("vehicle", "vehicle"), ("pedestrian", "pedestrian"), ("two-wheeler", "two_wheeler")):
    print(row(label, f"seg_obj/seg/dt+0.00/{cls}/all/all/iou@0"))

print("\n## 4 Paired scene bootstrap (VAL scenes resampled with replacement, 1000 draws; 95 % interval of combined - baseline)\n")
print("Maps at dt 0; boxes at dt 0 (plain boxes of both) and the refined boxes of the combined run against the baseline's.\n")
print("```")
for kw in (dict(det_a="det", det_b="det"),) + ((dict(det_a="det", det_b="detr"),) if has_ref else ()):
    _, txt = ev.compare(base, final, n_boot=1000, **kw)
print("```")

if M:
    print("\n## 5 Does the carried world memory help? (last keyframe of val clips, warm = memory carried, cold = same frame from an empty memory)\n")
    print("```\n" + open(os.path.join(final, "eval", "memory.txt")).read() + "```")

print("\n## 6 Training and cost\n")
for name, run, m in (("baseline", base, bm), ("combined", final, fm)):
    cfgj = json.load(open(os.path.join(run, "config.json")))
    steps = int(m["info/step"])
    per_step = cfgj["batch_size"] * cfgj["accum"] * cfgj.get("seq_len", 1)
    print(f"- {name}: {steps} optimiser steps x {per_step} samples ({cfgj['batch_size']} x {cfgj['accum']}"
          f"{' clips of ' + str(cfgj['seq_len']) + ' keyframes' if cfgj.get('seq_len', 1) > 1 else ''}), lr {cfgj['lr']}, "
          f"{cfgj['balancing']} task weights, model options {cfgj['model']}; validation losses (VAL, all samples, dt-weighted): "
          + ", ".join(f"{k[len('group.'):]} {f(m['val_loss/' + k], 3)}" for k in ("group.areas", "group.lines", "group.objects", "group.motion"))
          + f", total {f(m['val_loss/total'], 3)}")
for name, m in (("baseline", bm), ("combined", fm)):
    ms = {k[3:]: v for k, v in m.items() if k.startswith("ms/")}
    if ms:
        print(f"- latency {name} (batch 1, idle GPU: {ms.get('idle')}): " + ", ".join(f"{k} {v:.1f} ms" for k, v in ms.items()
                                                                              if isinstance(v, float) and k.endswith("median")))
