"""What are the false positives? (2026-10-30, CPU, lean: no NuScenes devkit object, no dataset, about 1 GB.)
The detection AP of t1_r4_soa loses more to false positives than to misses (det_breakdown.py). Each detection with score >= 0.3 in the
front field of view is classified against ALL GT boxes of the frame, including the ones the metric ignores because they carry no lidar /
radar point ("unobserved"), and the AP is recomputed with the unobserved boxes counted as GT.
The stored records have no tokens: frame k of a scene = the k-th sample of that scene in the order of the nuScenes sample table with a
previous and a next keyframe (the dataset's order, which the evaluation loop followed); verified by comparing the observed GT boxes of
every record with the GT cache (must be 0 differing frames).
Usage (bev_fusion folder): python temp/soa_gap/fp_breakdown.py runs/t1_r4_soa [dataroot]"""
import glob
import json
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, "evaluation")
from metrics import DetMeter, RANGE_BINS, gt_boxes_from_meta, in_front  # noqa: E402

run = sys.argv[1]
root = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/data/nuscenes_full")
GT_DIR = "data/cache/gt"
GROUPS = ("vehicle", "two_wheeler", "pedestrian")
tab = os.path.join(root, "v1.0-trainval")
scene_name = {s["token"]: s["name"] for s in json.load(open(os.path.join(tab, "scene.json")))}
by_scene = {}
for s in json.load(open(os.path.join(tab, "sample.json"))):
    if s["prev"] and s["next"]:
        by_scene.setdefault(scene_name[s["scene_token"]], []).append(s["token"])
rec = pickle.load(open(f"{run}/eval/scene_records.pkl", "rb"))["det"]


def gt_boxes(tok, observed_only):
    files = glob.glob(f"{GT_DIR}/{tok}_0.5_4_*.npz")
    g = np.load(files[0], allow_pickle=True)["gt"].item()["cur"]
    return [b for b in gt_boxes_from_meta(g["meta"]["boxes"], observed_only=observed_only) if b["group"] in GROUPS]


frames, bad = [], 0
for sc, fr in rec.items():
    toks = by_scene[sc]
    assert len(toks) == len(fr[0.0]), (sc, len(toks), len(fr[0.0]))
    for t, (dets, gts) in zip(toks, fr[0.0]):
        obs = gt_boxes(t, True)
        if len(obs) != len(gts) or any(abs(a["cx"] - b["cx"]) > 1e-3 for a, b in zip(obs, gts)):
            bad += 1
        frames.append((dets, gts, gt_boxes(t, False)))
print(f"{len(frames)} frames, {bad} frames whose observed GT differs from the record (must be 0)", flush=True)

m_obs, m_all = DetMeter(classes=list(GROUPS)), DetMeter(classes=list(GROUPS))
for dets, gts, allg in frames:
    m_obs.add(dets, gts)
    m_all.add(dets, allg)
for name, m in (("observed GT only (the evaluation as run)", m_obs), ("all GT incl. unobserved boxes", m_all)):
    r = m.result()[0.0]
    print("%-45s" % name, " ".join("%s mean AP %.3f" % (c, np.mean([r[c]["all"]["ap@%g" % t] for t in (0.5, 1.0, 2.0, 4.0)])) for c in GROUPS),
          " mAP %.3f" % r["mean"]["all"]["mAP"], flush=True)

cats = {}
for dets, gts, allg in frames:
    unobs = [b for b in allg if b not in gts]
    for d in dets:
        if d["score"] < 0.3 or not in_front(d):
            continue
        b = next(i for i, (lo, hi) in enumerate(RANGE_BINS) if lo <= d["cx"] < hi)

        def near(lst, same):
            ds = [math.hypot(g["cx"] - d["cx"], g["cy"] - d["cy"]) for g in lst if (g["group"] == d["group"]) == same and in_front(g)]
            return min(ds) if ds else 1e9
        if near(gts, True) < 2.0:
            c = "true positive-ish (same-group observed GT within 2 m; includes duplicates of it)"
        elif near(unobs, True) < 2.0:
            c = "on an UNOBSERVED GT box of the same group (within 2 m)"
        elif near(gts, False) < 2.0:
            c = "on an observed GT box of ANOTHER group (class confusion between vehicle / pedestrian / two-wheeler)"
        elif near(gts, True) < 4.0:
            c = "2-4 m from a same-group GT (localisation / duplicate)"
        else:
            c = "background (no GT within 4 m)"
        cats.setdefault(c, [0, 0, 0])[b] += 1
tot = sum(sum(v) for v in cats.values())
print("\nclassification of all detections with score >= 0.3 in the front field of view (%d):" % tot)
for c, v in sorted(cats.items(), key=lambda kv: -sum(kv[1])):
    print("  %-100s %5d (%4.1f %%)   by range 0-20 / 20-40 / 40-60 m: %d / %d / %d" % (c, sum(v), 100 * sum(v) / tot, *v))
