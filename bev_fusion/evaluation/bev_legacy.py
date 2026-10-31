"""Reader of the evaluation records of BEFORE the evaluation suite: `runs/<name>/eval/scene_records.pkl`, written by the former
evaluate_run.py (per scene: SegMeter counts of the headline layers, detection frames (dets, gt boxes) per dt).

New evaluations are made by `bev_suite.py` and stored per frame (`bev_suite.py compare` is their paired bootstrap, any number of
metrics, no pickle). This module exists only so that the numbers of the runs evaluated before can still be compared with each
other (the t1 and t2 / t3 studies): the paired scene bootstrap of the headline numbers, b - a, and the helpers the
report builders used (per-scene true-positive flags, AP from flags). Functions are the former ones, unchanged.
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from evaluation.metrics import ap_nuscenes, in_front, match

HEADLINE_SEG = (("drivable_area", "fov"), ("walkway", "fov"), ("lane_divider", "fov"), ("lane_divider:zc", "fov"),
                ("vehicle", "fov"))
SEG_METRIC = {"drivable_area": "iou@0", "walkway": "iou@0", "lane_divider": "iou@0.5", "lane_divider:zc": "f1@0.5",
              "vehicle": "iou@0"}


LEGACY_SEG_KEYS = {"drivable_area": "fov", "walkway": "fov", "lane_divider": "fov", "lane_divider:zc": "fov", "vehicle": "obj"}


def write_scene_records(path, seg_metric, store, det_rec, detr_rec=None, meta=None):
    """Write `scene_records.pkl` in the former format from a suite run (`bev_suite.py eval` does it, as scene_records<tag>_n<N>.pkl, so
    that the study tools that read it keep working across runs evaluated before and after the suite): per scene the SegMeter counts of
    the headline layers at dt 0 (tolerances 0 and 0.5 m) summed from the stored per-frame counts, and per scene and dt the detection
    frames (dets, gt boxes) collected while the frames were run (det_rec[scene][dt] = [(dets, gts), ...]; detr_rec the same for refined
    boxes); meta (optional): {"frames": the frame keys, "gt_signature": ...}, which `bev_suite.py compare` checks before pairing."""
    seg = {}
    for scene, counts in store.by_scene(seg_metric.name).items():
        want = {f"{m}|+0.00|{layer}|all|all" for layer, m in LEGACY_SEG_KEYS.items()}
        acc = seg_metric._accumulate(counts, want)
        rec = {}
        for layer, m in LEGACY_SEG_KEYS.items():
            a = acc.get(f"{m}|+0.00|{layer}|all|all")
            if a is not None:
                rec[layer] = {(0.0, layer, "all", "all"): {"cnt": a[0][:2], "n_gt": a[1], "err": a[2], "n_err": a[3]}}
        seg[scene] = rec
    with open(path, "wb") as f:
        pickle.dump({"seg": seg, "det": det_rec, "detr": detr_rec or {s: {} for s in det_rec}, "meta": meta or {}}, f)


def _seg_value(counts, metric):
    """metric from summed SegMeter counts of one layer (subset all, bin all)"""
    tol = {"iou@0": 0, "iou@0.5": 1, "f1@0": 0, "f1@0.5": 1}[metric]
    tp_p, fp, tp_g, fn = counts[tol]
    if metric.startswith("iou"):
        tp = (tp_p + tp_g) / 2
        return tp / max(tp + fp + fn, 1e-9)
    p, r = tp_p / max(tp_p + fp, 1e-9), tp_g / max(tp_g + fn, 1e-9)
    return 2 * p * r / max(p + r, 1e-9)


def scene_counts(rec, layer):
    """scene -> (2, 4) counts of the layer (dt 0, subset all, bin all)"""
    out = {}
    for s, layers in rec["seg"].items():
        c = layers.get(layer, {})
        key = (0.0, layer, "all", "all")
        if key in c:
            out[s] = c[key]["cnt"]
    return out


def scene_flags(frames_by_scene, scenes, cls, thr=2.0):
    """scene -> (scores, true-positive flags, n_gt) of one class (front field of view, matching at thr metres)"""
    out = {}
    for s in scenes:
        frames = [([d for d in ds if d["group"] == cls and in_front(d)], [g for g in gs if g["group"] == cls and in_front(g)])
                  for ds, gs in frames_by_scene[s]]
        m, n_gt = match(frames, thr)
        out[s] = (np.array([x[0] for x in m], float), np.array([x[1] for x in m], float), n_gt)
    return out


def ap_from_flags(flags, scenes):
    """nuScenes AP of the detections of the listed scenes (repeats allowed): one sort of the concatenated flags"""
    sc = np.concatenate([flags[s][0] for s in scenes])
    fl = np.concatenate([flags[s][1] for s in scenes])
    n_gt = sum(flags[s][2] for s in scenes)
    return ap_nuscenes(fl[np.argsort(-sc, kind="stable")], n_gt)


def det_ap(frames_by_scene, scenes, cls, thr=2.0):
    frames = [([d for d in ds if d["group"] == cls], [g for g in gs if g["group"] == cls])
              for s in scenes for ds, gs in frames_by_scene[s]]
    frames = [([d for d in ds if in_front(d)], [g for g in gs if in_front(g)]) for ds, gs in frames]
    m, n_gt = match(frames, thr)
    return ap_nuscenes([x[1] for x in m], n_gt)


def compare(run_a, run_b, n_boot=1000, seed=0, det_a="det", det_b="det", dt=0.0, tag=""):
    """paired scene bootstrap of the headline numbers: b - a, mean and 95 % interval (draws: evalsuite.stats.scene_picks, the same stream for
    every number; since 2026-10-31, D24 - intervals of earlier compares used numpy default_rng draws and differ in the random part only). det_a / det_b: the box set of each run
    ('det' = plain decode, 'detr' = refined by S5c); dt: the target time of the detection numbers (the map numbers are at dt 0).
    -> ({name: (b - a, lo, hi, a, b)}, text)"""
    ra = pickle.load(open(os.path.join(run_a, "eval", f"scene_records{tag}.pkl"), "rb"))    # tag: "" cold, "_warm" ...
    rb = pickle.load(open(os.path.join(run_b, "eval", f"scene_records{tag}.pkl"), "rb"))
    from evalsuite.stats import percentile_interval, scene_picks    # the one scene bootstrap of the research system (D24)
    lines = [f"paired scene bootstrap ({n_boot} resamples), b - a; a = {run_a} ({det_a}), b = {run_b} ({det_b}); val; "
             f"maps at dt 0, boxes at dt {dt:+.1f}"]
    res = {}
    for layer, _ in HEADLINE_SEG:
        metric = SEG_METRIC[layer]
        ca, cb = scene_counts(ra, layer), scene_counts(rb, layer)
        sc = sorted(set(ca) & set(cb))
        A, Bc = np.stack([ca[s] for s in sc]), np.stack([cb[s] for s in sc])
        point = _seg_value(Bc.sum(0), metric) - _seg_value(A.sum(0), metric)
        ds = []
        for ix in scene_picks(len(sc), n_boot, seed):
            ds.append(_seg_value(Bc[ix].sum(0), metric) - _seg_value(A[ix].sum(0), metric))
        lo, hi, nv = percentile_interval(ds)
        if nv < n_boot:
            lines.append(f"  note: {layer} {metric}: {n_boot - nv} of {n_boot} resamples had nothing to score; interval over the other {nv}")
        res[f"{layer} {metric}"] = (point, lo, hi, _seg_value(A.sum(0), metric), _seg_value(Bc.sum(0), metric))
    fa = {s: r[dt] for s, r in ra[det_a].items() if dt in r}
    fb = {s: r[dt] for s, r in rb[det_b].items() if dt in r}
    sc = sorted(set(fa) & set(fb))
    for cls in ("vehicle", "pedestrian"):
        for thr in (1.0, 2.0):            # AP@1 added 2026-10-29: AP@2 has a high no-sensor floor (t3_improve_loop try 1)
            # the greedy matching of a detection only involves its own frame, so the true-positive flags of every scene are fixed:
            # compute them once, a resample only re-sorts the concatenated flags (equal to re-running det_ap, but fast)
            fl_a, fl_b = scene_flags(fa, sc, cls, thr), scene_flags(fb, sc, cls, thr)
            a0, b0 = ap_from_flags(fl_a, sc), ap_from_flags(fl_b, sc)
            ds = []
            for ix in scene_picks(len(sc), n_boot, seed):
                s_ix = [sc[i] for i in ix]
                ds.append(ap_from_flags(fl_b, s_ix) - ap_from_flags(fl_a, s_ix))
            lo, hi, nv = percentile_interval(ds)
            if nv < n_boot:
                lines.append(f"  note: {cls} AP@{thr:g}: {n_boot - nv} of {n_boot} resamples had nothing to score; interval over the other {nv}")
            res[f"{cls} AP@{thr:g}"] = (b0 - a0, lo, hi, a0, b0)
    for k, (d, lo, hi, a, b) in res.items():
        lines.append(f"{k:28s} a {a:.4f}  b {b:.4f}  b-a {d:+.4f}  95% [{lo:+.4f}, {hi:+.4f}]"
                     + ("  (excludes 0)" if lo > 0 or hi < 0 else ""))
    txt = "\n".join(lines)
    print(txt)
    return res, txt
