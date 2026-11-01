"""Response test of the evaluation: worsen the GT by KNOWN amounts and check that every number moves by exactly that amount.

    python evaluation/bev_response.py [--n 50] [--sig 15efbde2d1] [--seed 0]        the table (intervention, number, measured, expected, difference)
    python evaluation/bev_response.py --selftest                                      the same with every row asserted

Real cached GT, no model. Each row of the table is one deliberate damage and the number that must show it; the expected value is derived from
the damage by a formula (stated in the row) or by an independent numpy computation, never from the metric under test.
Boxes (vehicles; the metrics of `bev_suite.py eval`: all-point AP `ap_all@t`, nuScenes AP `ap@t`, TP errors ate / ase / aoe / ave at 2 m):
    remove k of N boxes            n_tp = N - k, ap_all = (N - k) / N, the TP errors stay 0
    k false boxes, LOW score       ap_all = 1 (they rank after every true box), n_det = N + k
    k false boxes, HIGH score      ap_all = N / (N + k) (the precision envelope of a list that starts with k false boxes)
    move every 3rd box by d = 0.3  ate = d x moved / N (all still true positives at 0.5 m)
    move by d = 0.8 m, lower score the moved boxes are false at 0.5 m: ap_all@0.5 = (N - moved) / N, ap_all@1 = 1; ate = d x moved / N
    scale length and width by s    ase = 1 - s^2          (1 - IoU after aligning centre and heading)
    turn by 0.2 rad                aoe = 0.2
    velocity + (0.3, 0.4) m/s        ave = ave_moving = ave_static = 0.5; ave_skill_moving = 1 - 0.5 / ave0_moving
    all of it together             the numbers add up: nothing hides another
Maps and curves (lane dividers and the drivable area):
    positives dropped to p = 0     IoU@0 = 1 - r, AP = (1 - r) + r P / (P + N), ROC-AUC = 1 - r / 2, ECE = Brier = r P / (P + N)
    negatives raised to p = 1      IoU@0 = P / (P + q N), AP = P / (P + q N), ROC-AUC = 1 - q / 2, ECE = Brier = q N / (P + N)
    constant p = 0.5               AUC = 0.5, AP = prevalence, Brier = 0.25, ECE = |prevalence - 0.5|
Velocity field: GT + (0.3, 0.4) and 0.8 x GT: EPE, RMSE, bias and R2 against numpy computed straight from the arrays.
ROLES SWAPPED (--swap rows are part of the same table): the damaged set is the ground truth and the original GT is the prediction. Symmetric numbers (ate, ase, aoe,
ave, IoU) must not change; precision and recall trade places, so the AP of 'k boxes removed' (recall limited) must equal the AP of 'k boxes ranked first and false'
(precision limited), the AP of 'm false boxes added' equals the AP of 'm boxes missing'; for the maps the PR / ROC / calibration formulas swap their roles.
A row that is off means a number that does not measure what its name says."""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")]
from evaluation import bev_gtcheck as G  # noqa: E402
from evaluation.bev_views import gt_boxes, masks  # noqa: E402
from evaluation.metrics import in_front  # noqa: E402

CLASS = "vehicle"


def det_numbers(samples, dets, task):
    from evalsuite import Prediction, Sample
    from evalsuite.contracts import FrameRef
    metric = next(m for m in task.metrics() if m.name == "det")
    counts = []
    for i, ((name, s), d) in enumerate(zip(samples, dets)):
        sample = Sample(ref=FrameRef("gt", i, name), inputs=None, reference={0.0: s[0.0]})
        counts.append(metric.frame(sample, Prediction(outputs={"dt": {0.0: {"dets": d}}})))
    return {k[len("dt+0.00/"):]: v for k, v in metric.reduce(counts).items() if k.startswith("dt+0.00/")}


def scored_gt(samples, groups):
    """per sample the GT boxes the meter scores (observed, inside the front window), all classes"""
    return [[b for b in gt_boxes(s[0.0], groups) if in_front(b)] for _, s in samples]


def make_false(boxes_per_sample, rng, score):
    """one false vehicle per frame, 6 - 50 m ahead and well inside the window, at least 6 m from every GT box (no threshold of 0.5 - 4 m can match it)"""
    out = []
    for gts in boxes_per_sample:
        for _ in range(200):
            x = rng.uniform(8.0, 50.0)
            y = rng.uniform(-0.5, 0.5) * x * math.tan(0.56)
            if all(math.hypot(x - b["cx"], y - b["cy"]) > 6.0 for b in gts):
                out.append([dict(group=CLASS, cx=x, cy=y, vx=0.0, vy=0.0, length=4.5, width=1.9, yaw=float(rng.uniform(-3, 3)), score=score)])
                break
        else:
            out.append([])
    return out


def boxes_rows(samples, task):
    rng = np.random.RandomState(1)
    gts = scored_gt(samples, task.groups)
    base = [[dict(b, score=1.0) for b in g] for g in gts]
    veh_idx = [[i for i, b in enumerate(g) if b["group"] == CLASS] for g in gts]
    N = sum(len(v) for v in veh_idx)
    ref = det_numbers(samples, base, task)
    rows = [("oracle", "n_gt", ref[f"{CLASS}/all/n_gt"], N, 0.0), ("oracle", "ap_all@2", ref[f"{CLASS}/all/ap_all@2"], 1.0, 1e-9)]

    def run(dets):
        r = det_numbers(samples, dets, task)
        return lambda k: r[f"{CLASS}/all/{k}"]

    def edit(fn, score_changed=1.0):
        """copy of base with fn(box) -> box or None applied to every vehicle (None = removed); the edited boxes get `score_changed`"""
        out, n_changed = [], 0
        for g in base:
            o = []
            for b in g:
                if b["group"] != CLASS:
                    o.append(b)
                    continue
                nb = fn(dict(b))
                if nb is None:
                    continue
                if nb is not b and any(nb.get(k) != b.get(k) for k in nb):
                    n_changed += 1
                    nb["score"] = score_changed
                o.append(nb)
            out.append(o)
        return out, n_changed

    counter = {"i": 0}

    def drop_every_5th(b):
        counter["i"] += 1
        return None if counter["i"] % 5 == 0 else b
    dets, _ = edit(drop_every_5th)
    k = N // 5
    m = run(dets)
    rows += [("remove %d of %d" % (k, N), "n_tp", m("n_tp"), N - k, 0.0), ("remove %d of %d" % (k, N), "ap_all@2", m("ap_all@2"), (N - k) / N, 1e-9),
             ("remove %d of %d" % (k, N), "ap_all@0.5", m("ap_all@0.5"), (N - k) / N, 1e-9), ("remove %d of %d" % (k, N), "ate", m("ate"), 0.0, 1e-9),
             ("remove %d of %d" % (k, N), "ap@2 = (floor(100 R) - 10) / 90", m("ap@2"), max(0, math.floor(100 * (N - k) / N + 1e-9) - 10) / 90, 1e-9)]
    for label, score, exp_ap in (("false boxes, LOW score", 0.1, None), ("false boxes, HIGH score", 2.0, "high")):
        fp = make_false([g for g in gts], rng, score)
        dets = [b + f for b, f in zip(base, fp)]
        mm = sum(len(f) for f in fp)
        m = run(dets)
        exp = 1.0 if exp_ap is None else N / (N + mm)
        rows += [(f"add {mm} {label}", "n_det", m("n_det"), N + mm, 0.0), (f"add {mm} {label}", "n_tp", m("n_tp"), N, 0.0),
                 (f"add {mm} {label}", "ap_all@2", m("ap_all@2"), exp, 1e-9), (f"add {mm} {label}", "ap_all@0.5", m("ap_all@0.5"), exp, 1e-9)]

    def toward_axis(d, every=3):
        """moves every third vehicle by d towards the axis (the others stay: a wrong denominator or a wrong rank of moved against unmoved boxes shows)"""
        cnt = {"i": 0}

        def f(b):
            cnt["i"] += 1
            if cnt["i"] % every:
                return b
            nb = dict(b, cy=b["cy"] - math.copysign(d, b["cy"]) if abs(b["cy"]) > 1e-6 else b["cy"] + d)
            return nb if in_front(nb) else b
        return f
    dets, n_moved = edit(toward_axis(0.3))
    m = run(dets)
    rows += [("move %d boxes by 0.3 m" % n_moved, "ate", m("ate"), 0.3 * n_moved / N, 1e-6), ("move %d boxes by 0.3 m" % n_moved, "ap_all@0.5", m("ap_all@0.5"), 1.0, 1e-9)]
    dets, n_moved = edit(toward_axis(0.8), score_changed=0.9)
    m = run(dets)
    rows += [("move %d boxes by 0.8 m (score 0.9)" % n_moved, "ap_all@0.5", m("ap_all@0.5"), (N - n_moved) / N, 1e-9),
             ("move %d boxes by 0.8 m (score 0.9)" % n_moved, "ap_all@1", m("ap_all@1"), 1.0, 1e-9),
             ("move %d boxes by 0.8 m (score 0.9)" % n_moved, "ate", m("ate"), 0.8 * n_moved / N, 1e-6)]
    dets, n_moved = edit(toward_axis(1.5), score_changed=0.9)
    m = run(dets)
    lab = "move %d boxes by 1.5 m (score 0.9)" % n_moved
    rows += [(lab, "n_tp (TP at 2 m)", m("n_tp"), N, 0.0), (lab, "ap_all@1", m("ap_all@1"), (N - n_moved) / N, 1e-9), (lab, "ap_all@2", m("ap_all@2"), 1.0, 1e-9),
             (lab, "ate", m("ate"), 1.5 * n_moved / N, 1e-6)]
    s = 0.8
    dets, _ = edit(lambda b: dict(b, length=b["length"] * s, width=b["width"] * s))
    m = run(dets)
    rows += [("scale length and width by 0.8", "ase", m("ase"), 1 - s * s, 1e-9), ("scale length and width by 0.8", "ate", m("ate"), 0.0, 1e-9),
             ("scale length and width by 0.8", "aoe", m("aoe"), 0.0, 1e-9)]
    dets, _ = edit(lambda b: dict(b, yaw=b["yaw"] - 0.2 + 2 * math.pi))                 # the same heading as yaw - 0.2, written one turn further: aoe must wrap
    m = run(dets)
    rows += [("turn by -0.2 rad, written as + 2 pi - 0.2", "aoe", m("aoe"), 0.2, 1e-9)]
    dets, _ = edit(lambda b: dict(b, yaw=b["yaw"] + 0.2))
    m = run(dets)
    rows += [("turn by 0.2 rad", "aoe", m("aoe"), 0.2, 1e-9), ("turn by 0.2 rad", "ase", m("ase"), 0.0, 1e-9), ("turn by 0.2 rad", "ap_all@0.5", m("ap_all@0.5"), 1.0, 1e-9)]
    dets, _ = edit(lambda b: dict(b, vx=b["vx"] + 0.3, vy=b["vy"] + 0.4))
    m = run(dets)
    rows += [("velocity + (0.3, 0.4) m/s", "ave", m("ave"), 0.5, 1e-6), ("velocity + (0.3, 0.4) m/s", "ave_moving", m("ave_moving"), 0.5, 1e-6),
             ("velocity + (0.3, 0.4) m/s", "ave_static", m("ave_static"), 0.5, 1e-6),
             ("velocity + (0.3, 0.4) m/s", "ave_skill_moving", m("ave_skill_moving"), 1 - 0.5 / m("ave0_moving"), 1e-6), ("velocity + (0.3, 0.4) m/s", "ate", m("ate"), 0.0, 1e-9)]
    counter["i"] = 0
    mv = []
    mover = toward_axis(0.3)

    def combined(b):
        counter["i"] += 1
        if counter["i"] % 5 == 0:
            return None
        nb = mover(dict(b))
        mv.append(nb["cy"] != b["cy"])
        return dict(nb, yaw=nb["yaw"] + 0.1, length=nb["length"] * 0.9, width=nb["width"] * 0.9, vx=nb["vx"] + 0.18, vy=nb["vy"] + 0.24)
    dets, _ = edit(combined)
    m = run(dets)
    moved = sum(mv)
    tp = N - k
    lab = "removed 20 %, moved 0.3 m, yaw +0.1, scale 0.9, v + 0.3"
    rows += [(lab, "n_tp", m("n_tp"), tp, 0.0), (lab, "ap_all@2", m("ap_all@2"), tp / N, 1e-9), (lab, "aoe", m("aoe"), 0.1, 1e-9), (lab, "ase", m("ase"), 1 - 0.81, 1e-9),
             (lab, "ave", m("ave"), 0.3, 1e-6), (lab, "ate", m("ate"), 0.3 * moved / tp, 0.02)]
    ped = [[dict(b, radius=b["radius"] * 0.8) if b["group"] == "pedestrian" else b for b in g] for g in base]       # circles: ase = 1 - (r_min / r_max)^2, no heading
    mp = det_numbers(samples, ped, task)
    rows += [("pedestrian radius x 0.8", "ase", mp["pedestrian/all/ase"], 1 - 0.64, 1e-9), ("pedestrian radius x 0.8", "aoe (circle: nan)", mp["pedestrian/all/aoe"], float("nan"), 0.0),
             ("pedestrian radius x 0.8", "ap_all@0.5", mp["pedestrian/all/ap_all@0.5"], 1.0, 1e-9)]
    outside = [[dict(group=CLASS, cx=30.0, cy=25.0, vx=0.0, vy=0.0, length=4.5, width=1.9, yaw=0.0, score=2.0),                  # 39.8 deg off the axis: outside the front window
                dict(group=CLASS, cx=65.0, cy=0.0, vx=0.0, vy=0.0, length=4.5, width=1.9, yaw=0.0, score=2.0)] + g for g in base]  # beyond 60 m
    m = run(outside)
    rows += [("2 false boxes per frame outside the window (40 deg, 65 m)", "n_det", m("n_det"), N, 0.0), ("2 false boxes per frame outside the window (40 deg, 65 m)", "ap_all@2", m("ap_all@2"), 1.0, 1e-9)]
    return rows


def gap_rows(samples, task):
    """what the evaluation does NOT do yet (decision D38: outside the fusion area everything is don't care on both sides): the damage is real but the number should not see it"""
    from evaluation.metrics import gt_boxes_from_meta
    base, hidden = [], 0
    for _, s in samples:
        g = s[0.0]
        allb = gt_boxes_from_meta(g["meta"]["boxes"], observed_only=False)
        seen = {(round(b["cx"], 3), round(b["cy"], 3)) for b in gt_boxes_from_meta(g["meta"]["boxes"])}
        dets = [dict(b, score=1.0) for b in gt_boxes(g, task.groups) if in_front(b)]
        extra = [dict(b, score=1.0) for b in allb if b["group"] == CLASS and in_front(b) and (round(b["cx"], 3), round(b["cy"], 3)) not in seen]
        hidden += len(extra)
        base.append(dets + extra)
    m = det_numbers(samples, base, task)
    n_gt = m[f"{CLASS}/all/n_gt"]
    return [(f"a perfect detection on each of {hidden} UNOBSERVED vehicle boxes (no lidar / radar point)", "n_det (D38: ignored, = n_gt)", m[f"{CLASS}/all/n_det"], n_gt, 0.0),
            (f"a perfect detection on each of {hidden} UNOBSERVED vehicle boxes (no lidar / radar point)", "ap_all@2 (D38: 1)", m[f"{CLASS}/all/ap_all@2"], 1.0, 1e-9)]


def maps_rows(samples, task):
    from evalsuite import Prediction, Sample
    from evalsuite.contracts import FrameRef
    rng = np.random.RandomState(2)
    seg = next(m for m in task.metrics() if m.name == "seg")
    cur = next(m for m in task.metrics() if m.name == "curve")
    rows = []
    for layer, kind in (("lane_divider", "line"), ("drivable_area", "area")):
        key = {"lane_divider": "lane_divider", "drivable_area": "drivable_area"}[layer]

        def measure(make):
            cs, cc, n_pos, n_neg = [], [], 0.0, 0.0
            for i, (name, s) in enumerate(samples):
                g = s[0.0]
                m = masks(g)["fov"]
                y = np.asarray(g[key], np.float32) > 0.5
                p = make(y.copy(), m)
                n_pos += float((y & m).sum())
                n_neg += float((~y & m).sum())
                o = G.oracle_output(g, task.spec, task.groups)
                o["prob"] = dict(o["prob"], **{layer: p.astype(np.float32)})
                sample = Sample(ref=FrameRef("gt", i, name), inputs=None, reference={0.0: g})
                pred = Prediction(outputs={"dt": {0.0: o}})
                cs.append(seg.frame(sample, pred))
                cc.append(cur.frame(sample, pred))
            a, b = seg.reduce(cs), cur.reduce(cc)
            return (lambda k: a[f"fov/dt+0.00/{layer}/all/all/{k}"]), (lambda k: b[f"dt+0.00/{layer}/{k}"]), n_pos, n_neg
        # dropped positives
        stats = {}

        def drop(y, m, r=0.3):
            hit = y & m & (rng.rand(*y.shape) < r)
            stats["dropped"] = stats.get("dropped", 0) + int(hit.sum())
            return np.where(y & ~hit, 1.0, 0.0)
        iou, c, P, Nn = measure(drop)
        r = stats["dropped"] / P
        lab = f"{layer}: {100 * r:.1f} % of the positive cells set to p = 0"
        rows += [(lab, "IoU@0", iou("iou@0"), 1 - r, 1e-9), (lab, "AP", c("ap"), (1 - r) + r * P / (P + Nn), 1e-9), (lab, "ROC-AUC", c("roc_auc"), 1 - r / 2, 1e-9),
                 (lab, "ECE", c("ece"), r * P / (P + Nn), 1e-9), (lab, "Brier", c("brier"), r * P / (P + Nn), 1e-9)]
        stats.clear()

        def raise_neg(y, m, q=0.2):
            hit = ~y & m & (rng.rand(*y.shape) < q)
            stats["raised"] = stats.get("raised", 0) + int(hit.sum())
            return np.where(y | hit, 1.0, 0.0)
        iou, c, P, Nn = measure(raise_neg)
        q = stats["raised"] / Nn
        lab = f"{layer}: {100 * q:.1f} % of the negative cells set to p = 1"
        rows += [(lab, "IoU@0", iou("iou@0"), P / (P + q * Nn), 1e-9), (lab, "AP", c("ap"), P / (P + q * Nn), 1e-9), (lab, "ROC-AUC", c("roc_auc"), 1 - q / 2, 1e-9),
                 (lab, "ECE", c("ece"), q * Nn / (P + Nn), 1e-9), (lab, "Brier", c("brier"), q * Nn / (P + Nn), 1e-9)]
        iou, c, P, Nn = measure(lambda y, m: np.full(y.shape, 0.5))
        prev = P / (P + Nn)
        lab = f"{layer}: p = 0.5 everywhere"
        rows += [(lab, "ROC-AUC", c("roc_auc"), 0.5, 1e-9), (lab, "AP", c("ap"), prev, 1e-9), (lab, "Brier", c("brier"), 0.25, 1e-9), (lab, "ECE", c("ece"), abs(prev - 0.5), 1e-9)]
    return rows


def velocity_rows(samples, task):
    from evalsuite import Prediction, Sample
    from evalsuite.contracts import FrameRef
    cur = next(m for m in task.metrics() if m.name == "curve")
    rows = []
    for label, fn in (("velocity field + (0.3, 0.4) m/s", lambda v: v + np.array([0.3, 0.4], np.float32)[:, None, None]), ("velocity field x 0.8", lambda v: 0.8 * v)):
        cs, E, GV, DX = [], [], [], []
        for i, (name, s) in enumerate(samples):
            g = s[0.0]
            m = masks(g)["obj_fov"] & np.any([np.asarray(g[c], np.float32) > 0.5 for c in task.groups if c in g], axis=0)
            gv = np.asarray(g["velocity"], np.float32)
            v = fn(gv)
            o = G.oracle_output(g, task.spec, task.groups)
            o["vhat"] = v.astype(np.float32)
            cs.append(cur.frame(Sample(ref=FrameRef("gt", i, name), inputs=None, reference={0.0: g}), Prediction(outputs={"dt": {0.0: o}})))
            for sel, store in ((m, "obj"), (m & (np.hypot(gv[0], gv[1]) > 0.5), "moving")):
                GV.append((store, gv[:, sel].astype(np.float64)))
                DX.append((store, (v[:, sel] - gv[:, sel]).astype(np.float64)))
        res = cur.reduce(cs)
        for sub in ("obj", "moving"):
            gv = np.concatenate([a for s_, a in GV if s_ == sub], axis=1)
            dx = np.concatenate([a for s_, a in DX if s_ == sub], axis=1)
            epe = np.hypot(dx[0], dx[1])
            sst = float(((gv - gv.mean(axis=1, keepdims=True)) ** 2).sum())
            lab = f"{label} ({sub} cells)"
            rows += [(lab, "EPE", res[f"dt+0.00/velocity/{sub}/epe"], float(epe.mean()), 1e-5), (lab, "RMSE", res[f"dt+0.00/velocity/{sub}/rmse"], float(np.sqrt((epe ** 2).mean())), 1e-5),
                     (lab, "R2", res[f"dt+0.00/velocity/{sub}/r2"], 1 - float((epe ** 2).sum()) / sst, 1e-5),
                     (lab, "bias", res[f"dt+0.00/velocity/{sub}/bias"], float(np.hypot(*dx.mean(axis=1))), 1e-5)]
    return rows


def to_meta_box(b):
    """a box dict of the metrics -> the box format of the GT meta (what gt_boxes_from_meta reads), so a degraded box list can serve as the GT"""
    if "radius" in b:
        size = [2 * b["radius"], 2 * b["radius"], 1.7]
    else:
        size = [b["width"], b["length"], 1.7]
    return {"group": b["group"], "center": [b["cx"], b["cy"], 0.0], "size": size, "vel": [b["vx"], b["vy"]], "yaw": b.get("yaw", 0.0), "observed": True, "token": ""}


def with_boxes(samples, boxes_per_sample):
    out = []
    for (name, s), boxes in zip(samples, boxes_per_sample):
        g = dict(s[0.0], meta=dict(s[0.0]["meta"], boxes=[to_meta_box(b) for b in boxes]))
        out.append((name, {0.0: g}))
    return out


def swap_boxes_rows(samples, task):
    """ROLES SWAPPED: the damaged boxes are the ground truth, the original GT boxes are the 'detections'. A removed box is now a false alarm, an added false box
    a missed object, a moved / scaled / turned box is the same error seen from the other side. Symmetric numbers (ate, ase, aoe, ave, IoU) stay, precision and
    recall trade places: AP of 'k removed' (recall limited) equals AP of 'k false boxes ranked first' (precision limited)."""
    rng = np.random.RandomState(3)
    gts = scored_gt(samples, task.groups)
    base = [[dict(b, score=1.0) for b in g] for g in gts]
    N = sum(1 for g in gts for b in g if b["group"] == CLASS)
    rows = []

    def run(ref_boxes, dets):
        r = det_numbers(with_boxes(samples, ref_boxes), dets, task)
        return lambda k: r[f"{CLASS}/all/{k}"]

    def plain(g):
        return [{k: v for k, v in b.items() if k != "score"} for b in g]
    # a vehicle removed from the GT = a false alarm among the detections; every 5th vehicle
    i = 0
    ref, k = [], 0
    for g in base:
        keep = []
        for b in g:
            if b["group"] == CLASS:
                i += 1
                if i % 5 == 0:
                    k += 1
                    continue
            keep.append(b)
        ref.append(plain(keep))
    for label, sc, exp in (("removed from the GT, ranked LAST", 0.5, 1.0), ("removed from the GT, ranked FIRST", 2.0, (N - k) / N)):
        i = 0
        dets = []
        for g in base:
            d = []
            for b in g:
                if b["group"] == CLASS:
                    i += 1
                    d.append(dict(b, score=sc) if i % 5 == 0 else b)
                else:
                    d.append(b)
            dets.append(d)
        m = run(ref, dets)
        lab = f"SWAPPED: {k} boxes {label}"
        rows += [(lab, "n_gt", m("n_gt"), N - k, 0.0), (lab, "n_det", m("n_det"), N, 0.0), (lab, "n_tp", m("n_tp"), N - k, 0.0), (lab, "ap_all@2", m("ap_all@2"), exp, 1e-9)]
    # false boxes added to the GT = objects the detector missed
    fp = make_false(gts, rng, 1.0)
    ref = [plain(g) + [{kk: v for kk, v in f.items() if kk != "score"} for f in ff] for g, ff in zip(base, fp)]
    mm = sum(len(f) for f in fp)
    m = run(ref, base)
    lab = f"SWAPPED: {mm} boxes added to the GT (missed by the detector)"
    rows += [(lab, "n_gt", m("n_gt"), N + mm, 0.0), (lab, "n_tp", m("n_tp"), N, 0.0), (lab, "ap_all@2 = N / (N + m)", m("ap_all@2"), N / (N + mm), 1e-9)]
    # moved / scaled / turned / faster boxes in the GT: the same errors, same size

    def damaged(fn, every=3):
        out, n_changed, idx = [], 0, 0
        for g in base:
            o = []
            for b in g:
                if b["group"] == CLASS:
                    idx += 1
                    if idx % every == 0:
                        nb = fn(dict(b))
                        if nb is not None and any(nb.get(kk) != b.get(kk) for kk in nb if kk != "score"):
                            n_changed += 1
                            o.append(nb)
                            continue
                o.append(b)
            out.append(o)
        return out, n_changed

    def toward(d):
        return lambda b: (lambda nb: nb if in_front(nb) else b)(dict(b, cy=b["cy"] - math.copysign(d, b["cy"]) if abs(b["cy"]) > 1e-6 else b["cy"] + d))
    for d, rank in ((0.3, 1.0), (0.8, 0.9), (1.5, 0.9)):
        mod, n_moved = damaged(toward(d))
        dets = [[dict(b, score=rank) if (b["group"] == CLASS and any(abs(b["cx"] - c["cx"]) < 1e-9 and abs(b["cy"] - c["cy"]) > 1e-9 for c in mg)) else b for b in g]
                for g, mg in zip(base, mod)]                      # the detections that sit at the old place of a moved GT box rank lower
        m = run([plain(g) for g in mod], dets)
        lab = f"SWAPPED: {n_moved} GT boxes moved by {d} m"
        rows += [(lab, "ate", m("ate"), d * n_moved / N, 1e-6)]
        if d == 0.8:
            rows += [(lab, "ap_all@0.5", m("ap_all@0.5"), (N - n_moved) / N, 1e-9), (lab, "ap_all@1", m("ap_all@1"), 1.0, 1e-9)]
        if d == 1.5:
            rows += [(lab, "n_tp (TP at 2 m)", m("n_tp"), N, 0.0), (lab, "ap_all@1", m("ap_all@1"), (N - n_moved) / N, 1e-9)]
    for label, fn, key, exp in (("scaled by 0.8", lambda b: dict(b, length=b["length"] * 0.8, width=b["width"] * 0.8), "ase", 1 - 0.64),
                                ("turned by 0.2 rad", lambda b: dict(b, yaw=b["yaw"] + 0.2), "aoe", 0.2),
                                ("given (0.3, 0.4) m/s", lambda b: dict(b, vx=b["vx"] + 0.3, vy=b["vy"] + 0.4), "ave", 0.5)):
        mod, n_ch = damaged(fn, every=1)
        m = run([plain(g) for g in mod], base)
        rows += [(f"SWAPPED: GT boxes {label}", key, m(key), exp, 1e-9)]
    return rows


def swap_maps_rows(samples, task):
    """ROLES SWAPPED for the maps: the damaged binary map is the GT, the original GT is the prediction (p in {0, 1})"""
    from evalsuite import Prediction, Sample
    from evalsuite.contracts import FrameRef
    rng = np.random.RandomState(4)
    seg = next(m for m in task.metrics() if m.name == "seg")
    cur = next(m for m in task.metrics() if m.name == "curve")
    rows = []
    for layer in ("lane_divider", "drivable_area"):
        for kind, rate in (("drop", 0.3), ("raise", 0.2)):
            cs, cc, stat = [], [], {"P": 0.0, "N": 0.0, "hit": 0.0}
            for i, (name, s) in enumerate(samples):
                g = s[0.0]
                m = masks(g)["fov"]
                y = np.asarray(g[layer], np.float32) > 0.5
                hit = (y & m & (rng.rand(*y.shape) < rate)) if kind == "drop" else (~y & m & (rng.rand(*y.shape) < rate))
                y2 = (y & ~hit) if kind == "drop" else (y | hit)
                stat["P"] += float((y & m).sum())
                stat["N"] += float((~y & m).sum())
                stat["hit"] += float(hit.sum())
                o = G.oracle_output(g, task.spec, task.groups)                      # the prediction: the ORIGINAL GT, binarised (the soft coverage values of the
                o["prob"] = dict(o["prob"], **{layer: y.astype(np.float32)})         # area edges would put cells between the bins the formulas assume)
                g2 = dict(g, **{layer: y2.astype(np.float32)})                       # the reference: the damaged map
                sample = Sample(ref=FrameRef("gt", i, name), inputs=None, reference={0.0: g2})
                pred = Prediction(outputs={"dt": {0.0: o}})
                cs.append(seg.frame(sample, pred))
                cc.append(cur.frame(sample, pred))
            a, b = seg.reduce(cs), cur.reduce(cc)
            P, N, hit = stat["P"], stat["N"], stat["hit"]
            if kind == "drop":                       # predicted positives P, GT positives P - hit: recall 1, precision (P - hit) / P
                r = hit / P
                lab = f"SWAPPED {layer}: GT lost {100 * r:.1f} % of its positives"
                fpr = hit / (N + hit)
                rows += [(lab, "IoU@0", a[f"fov/dt+0.00/{layer}/all/all/iou@0"], 1 - r, 1e-9), (lab, "AP", b[f"dt+0.00/{layer}/ap"], 1 - r, 1e-9),
                         (lab, "ROC-AUC", b[f"dt+0.00/{layer}/roc_auc"], 1 - fpr / 2, 1e-9), (lab, "ECE", b[f"dt+0.00/{layer}/ece"], hit / (P + N), 1e-9),
                         (lab, "Brier", b[f"dt+0.00/{layer}/brier"], hit / (P + N), 1e-9)]
            else:                                    # GT positives P + hit, predicted P: precision 1, recall P / (P + hit)
                tpr = P / (P + hit)
                lab = f"SWAPPED {layer}: GT gained {100 * hit / N:.1f} % of its negatives as positives"
                rows += [(lab, "IoU@0", a[f"fov/dt+0.00/{layer}/all/all/iou@0"], tpr, 1e-9), (lab, "AP = tpr + (1 - tpr) x prevalence of the GT", b[f"dt+0.00/{layer}/ap"], tpr + (1 - tpr) * (P + hit) / (P + N), 1e-9),
                         (lab, "ROC-AUC", b[f"dt+0.00/{layer}/roc_auc"], (1 + tpr) / 2, 1e-9), (lab, "ECE", b[f"dt+0.00/{layer}/ece"], hit / (P + N), 1e-9),
                         (lab, "Brier", b[f"dt+0.00/{layer}/brier"], hit / (P + N), 1e-9)]
    return rows


def swap_velocity_rows(samples, task):
    """ROLES SWAPPED for the velocity field: the damaged field is the GT, the original GT velocity is the prediction; numpy straight from the arrays"""
    from evalsuite import Prediction, Sample
    from evalsuite.contracts import FrameRef
    cur = next(m for m in task.metrics() if m.name == "curve")
    rows = []
    for label, fn in (("GT velocity + (0.36, 0.48) m/s", lambda v: v + np.array([0.36, 0.48], np.float32)[:, None, None]), ("GT velocity x 0.8", lambda v: 0.8 * v)):
        cs, GV, DX = [], [], []                          # (0.36, 0.48) has length 0.6: a clear margin to the moving threshold 0.5 (an offset of exactly 0.5 would sit on it)
        for i, (name, s) in enumerate(samples):
            g = s[0.0]
            m = masks(g)["obj_fov"] & np.any([np.asarray(g[c], np.float32) > 0.5 for c in task.groups if c in g], axis=0)
            gv0 = np.asarray(g["velocity"], np.float32)
            gv = fn(gv0).astype(np.float32)                                         # the damaged field is the reference
            o = G.oracle_output(g, task.spec, task.groups)                          # the prediction is the original
            g2 = dict(g, velocity=gv)
            cs.append(cur.frame(Sample(ref=FrameRef("gt", i, name), inputs=None, reference={0.0: g2}), Prediction(outputs={"dt": {0.0: o}})))
            for sel, store in ((m, "obj"), (m & (np.hypot(gv[0], gv[1]) > 0.5), "moving")):
                GV.append((store, gv[:, sel].astype(np.float64)))
                DX.append((store, (gv0[:, sel] - gv[:, sel]).astype(np.float64)))
        res = cur.reduce(cs)
        for sub in ("obj", "moving"):
            gv = np.concatenate([a for s_, a in GV if s_ == sub], axis=1)
            dx = np.concatenate([a for s_, a in DX if s_ == sub], axis=1)
            epe = np.hypot(dx[0], dx[1])
            sst = float(((gv - gv.mean(axis=1, keepdims=True)) ** 2).sum())
            lab = f"SWAPPED: {label} as the GT ({sub} cells)"
            rows += [(lab, "EPE", res[f"dt+0.00/velocity/{sub}/epe"], float(epe.mean()), 1e-5), (lab, "RMSE", res[f"dt+0.00/velocity/{sub}/rmse"], float(np.sqrt((epe ** 2).mean())), 1e-5),
                     (lab, "R2", res[f"dt+0.00/velocity/{sub}/r2"], 1 - float((epe ** 2).sum()) / sst, 1e-5)]
    return rows


def all_rows(n, sig, seed):
    task = G.build_task()
    samples = G.pick(n, sig, seed)
    return samples, (boxes_rows(samples, task) + maps_rows(samples, task) + velocity_rows(samples, task)
                     + swap_boxes_rows(samples, task) + swap_maps_rows(samples, task) + swap_velocity_rows(samples, task))


def print_rows(rows):
    print(f"{'intervention':58s}{'number':30s}{'measured':>10s}{'expected':>10s}{'diff':>10s}  ok")
    last = None
    for lab, key, got, exp, tol in rows:
        diff = got - exp
        ok = abs(diff) <= tol or (math.isnan(got) and math.isnan(exp))
        print(f"{lab if lab != last else '':58s}{key:30s}{got:10.4f}{exp:10.4f}{diff:+10.5f}  {'ok' if ok else 'OFF'}")
        last = lab


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--sig", default="15efbde2d1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    samples, rows = all_rows(a.n, a.sig, a.seed)
    print(f"{len(samples)} real GT frames, signature {a.sig}")
    print_rows(rows)
    gaps = gap_rows(samples, G.build_task())
    print("\nKNOWN GAP (not asserted; becomes ok when D38 is implemented, then move the rows up):")
    print_rows(gaps)
    off = [r for r in rows if not (abs(r[2] - r[3]) <= r[4] or (math.isnan(r[2]) and math.isnan(r[3])))]
    if a.selftest:
        assert not off, f"{len(off)} rows off: {[(r[0], r[1], round(r[2], 4), round(r[3], 4)) for r in off]}"
        print(f"bev_response selftest ok: {len(rows)} numbers reflect exactly what was worsened")
    elif off:
        print(f"{len(off)} of {len(rows)} rows are OFF")


if __name__ == "__main__":
    main()
