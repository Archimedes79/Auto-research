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
Velocity split (`vel`, bev_velocity.py): EPE and n per cell set (static, per class, moving part) and at the GT box centres (field / object head / zero, box
velocities painted into the centre cells), numpy from the arrays and the box list; swapped: the damaged velocities are the reference.
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
from evaluation.metrics import gt_boxes_from_meta, in_front  # noqa: E402

CLASS = "vehicle"


def det_numbers(samples, dets, task, name="det"):
    from Evaluation import Prediction, Sample
    from Evaluation.contracts import FrameRef
    metric = next(m for m in task.metrics() if m.name == name)
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


def annotated(samples, task):
    """per sample: every annotated box of the scored classes inside the window (observed or not), with its fusion-area class (footprint share from the corners)
    and the sample's area"""
    from evaluation.bev_fusion_area import fusion_area, gt_class, gt_with_shares
    out = []
    for _, s in samples:
        g = s[0.0]
        boxes, shares = gt_with_shares(g, task.groups)
        out.append(([dict(b, cls=gt_class(f)) for b, f in zip(boxes, shares) if in_front(b)], fusion_area(g)))
    return out


def geometry_rows(samples, task):
    """footprint geometry against an INDEPENDENT area: the cached `vehicle` layer is the area, so the footprint of a vehicle box must lie in it (mean share >= 0.95);
    a y flip, swapped length / width or a negated yaw must fall below 0.9. Then the GtGenerator reference implementation (GtGenerator/coverage.py, the GT owner's): the same fusion
    cells, the same corner shares, and the same classes for the dict-based footprint that detections use."""
    from GtGenerator import coverage as AC
    from GtGenerator.grid import GridSpec
    from evaluation.bev_fusion_area import corner_share, footprint_share, fusion_area, gt_class, gt_with_shares
    rows, grid = [], GridSpec(res=0.5)
    shares = {"as is": [], "y flipped": [], "length / width swapped": [], "yaw negated": []}
    differ, maxdiff, n_cls, n_box = 0, 0.0, 0, 0
    for _, s in samples:
        g = s[0.0]
        layer = np.asarray(g["vehicle"], np.float32) > 0                       # the layer holds the coverage of each cell by a box: any coverage counts
        boxes, f_corner = gt_with_shares(g, task.groups)
        for b, fc, rb in zip(boxes, f_corner, [r for r in g["meta"]["boxes"] if r["group"] in task.groups]):
            if b["group"] == CLASS and "length" in b and in_front(b) and rb.get("observed", True):             # (an unobserved box may have no pixels in the layer)
                for k, fn in (("as is", lambda b: b), ("y flipped", lambda b: dict(b, cy=-b["cy"])), ("length / width swapped", lambda b: dict(b, length=b["width"], width=b["length"])),
                              ("yaw negated", lambda b: dict(b, yaw=-b["yaw"]))):
                    shares[k].append(footprint_share(fn(b), layer))
            ref = AC.footprint_share(rb["corners"], AC.fusion_cells(g), grid)
            maxdiff = max(maxdiff, abs(fc - ref))
            if "length" in b:
                n_box += 1
                n_cls += gt_class(footprint_share(b, fusion_area(g))) != gt_class(fc)
        differ += int((fusion_area(g) != AC.fusion_cells(g)).sum())
    m = {k: float(np.mean(v)) for k, v in shares.items()}
    lab = f"geometry: {len(shares['as is'])} observed vehicle footprints against the cached vehicle layer as the area"
    rows += [(lab, "mean share as is >= 0.95 (1 = yes)", float(m["as is"] >= 0.95), 1.0, 0.0)]
    rows += [(lab, f"mean share {k} < 0.9 (1 = yes)", float(m[k] < 0.9), 1.0, 0.0) for k in ("y flipped", "length / width swapped", "yaw negated")]
    lab = "GtGenerator reference (GtGenerator/coverage.py)"
    rows += [(lab, "cells where fusion_area differs from fusion_cells", float(differ), 0.0, 0.0), (lab, "max |corner share - GtGenerator footprint_share|", maxdiff, 0.0, 0.02),
             (lab, "share of box footprints (dict form, as for detections) with another class than the corner form", n_cls / max(n_box, 1), 0.0, 0.01)]
    return rows


def false_in_area(ann, rng, score, inside):
    """one false vehicle per frame, 8 - 50 m ahead, at least 6 m from EVERY annotated box (neutral ones too): inside the fusion area (share 1) or well outside (share 0)"""
    from evaluation.bev_fusion_area import footprint_share
    out = []
    for bs, area in ann:
        for _ in range(400):
            x = rng.uniform(8.0, 50.0)
            y = rng.uniform(-0.5, 0.5) * x * math.tan(0.56)
            b = dict(group=CLASS, cx=x, cy=y, vx=0.0, vy=0.0, length=4.5, width=1.9, yaw=float(rng.uniform(-3, 3)), score=score)
            f = footprint_share(b, area)
            if (f == 1.0 if inside else f == 0.0) and all(math.hypot(x - a["cx"], y - a["cy"]) > 6.0 for a in bs):
                out.append([b])
                break
        else:
            out.append([])
    return out


def fusion_rows(samples, task):
    """the fusion-area numbers (D37 / D38, evaluation/bev_fusion_area.py): `detf` scores the care boxes (footprint share >= 0.5 in camera x radar x not hidden), `detfp`
    care + partial; boxes outside are neutral on BOTH sides. Expected values come from counting the annotated boxes by their class, never from the metric."""
    rng = np.random.RandomState(5)
    ann = annotated(samples, task)
    veh = [[b for b in bs if b["group"] == CLASS] for bs, _ in ann]
    cnt = {c: sum(1 for v in veh for b in v if b["cls"] == c) for c in ("care", "partial", "dont_care")}
    Nc, Np = cnt["care"], cnt["partial"]
    rows = geometry_rows(samples, task)
    assert Nc > 0, f"too few frames for the fusion rows (no care box): use --n 24 or more"
    rows += [("sanity: the fusion rows rest on all three classes being present", f"{c} boxes > 0 (1 = yes)", float(cnt[c] > 0), 1.0, 0.0) for c in ("care", "partial", "dont_care")]

    def strip(bs):
        return [dict({k: v for k, v in b.items() if k != "cls"}, score=1.0) for b in bs]

    def run(dets, metric):
        r = det_numbers(samples, dets, task, metric)
        return lambda k: r[f"{CLASS}/all/{k}"]
    everything = [strip(bs) for bs, _ in ann]
    m = run(everything, "detf")
    lab = f"detf: a perfect detection of EVERY annotated box ({Nc} care, {Np} partial, {cnt['dont_care']} don't care)"
    rows += [(lab, "n_gt = care boxes", m("n_gt"), Nc, 0.0), (lab, "n_det (neutral ones ignored)", m("n_det"), Nc, 0.0), (lab, "n_tp", m("n_tp"), Nc, 0.0), (lab, "ap_all@2", m("ap_all@2"), 1.0, 1e-9)]
    mp = run(everything, "detfp")
    lab = "detfp: the same detections"
    rows += [(lab, "n_gt = care + partial", mp("n_gt"), Nc + Np, 0.0), (lab, "n_det", mp("n_det"), Nc + Np, 0.0), (lab, "ap_all@2", mp("ap_all@2"), 1.0, 1e-9)]
    care_only = [strip([b for b in bs if b["cls"] == "care"]) for bs, _ in ann]
    m = run(care_only, "detf")
    lab = "detf: detections of the care boxes ONLY (the neutral boxes are not 'missed')"
    rows += [(lab, "n_gt", m("n_gt"), Nc, 0.0), (lab, "n_tp", m("n_tp"), Nc, 0.0), (lab, "ap_all@2", m("ap_all@2"), 1.0, 1e-9)]
    from evaluation.bev_fusion_area import footprint_share
    shifted, n_sh = [], 0                                              # the detection of a neutral box nudged (<= 1.5 m) until at least half of it lies in the area: still on a neutral object
    for bs, area in ann:
        o = [dict(b, cls=None) for b in bs if b["cls"] == "care"]
        for b in bs:
            if b["cls"] == "care":
                continue
            cand = [dict(b, cx=b["cx"] + dx, cy=b["cy"] + dy) for dx in (-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5) for dy in (-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5) if math.hypot(dx, dy) <= 1.5]
            hit = [c for c in cand if footprint_share(c, area) >= 0.5 and in_front(c)]
            if hit:
                o.append(hit[0])
                n_sh += 1
        shifted.append(strip(o))
    assert n_sh > 0, "no neutral detection could be nudged into the area"
    m = run(shifted, "detf")
    lab = f"detf: the care boxes + {n_sh} detections of NEUTRAL boxes nudged up to 1.5 m into the area (still on a neutral object)"
    rows += [(lab, "n_det (neutral ones ignored)", m("n_det"), Nc, 0.0), (lab, "ap_all@2", m("ap_all@2"), 1.0, 1e-9)]
    i, kept, k = 0, [], 0
    for bs, _ in ann:
        o = []
        for b in bs:
            if b["group"] == CLASS and b["cls"] == "care":
                i += 1
                if i % 5 == 0:
                    k += 1
                    continue
            o.append(b)
        kept.append(strip(o))
    m = run(kept, "detf")
    lab = f"detf: {k} of {Nc} care boxes removed"
    rows += [(lab, "n_tp", m("n_tp"), Nc - k, 0.0), (lab, "ap_all@2", m("ap_all@2"), (Nc - k) / Nc, 1e-9)]
    fi = false_in_area(ann, rng, 2.0, inside=True)
    mi = sum(len(f) for f in fi)
    assert mi > 0, "no false box could be placed inside the area"
    m = run([e + f for e, f in zip(everything, fi)], "detf")
    lab = f"detf: {mi} false vehicles INSIDE the fusion area, high score"
    rows += [(lab, "n_det", m("n_det"), Nc + mi, 0.0), (lab, "ap_all@2 = N / (N + m)", m("ap_all@2"), Nc / (Nc + mi), 1e-9)]
    fo = false_in_area(ann, rng, 2.0, inside=False)
    mo = sum(len(f) for f in fo)
    assert mo > 0, "no false box could be placed outside the area"
    m = run([e + f for e, f in zip(everything, fo)], "detf")
    lab = f"detf: {mo} false vehicles well OUTSIDE the fusion area, high score (don't care on both sides)"
    rows += [(lab, "n_det (ignored)", m("n_det"), Nc, 0.0), (lab, "ap_all@2", m("ap_all@2"), 1.0, 1e-9)]
    m = det_numbers(samples, everything, task, "det")
    n_all = sum(1 for bs, _ in ann for b in bs if b["group"] == CLASS)
    lab = "det (old) on the same detections: nothing is ignored, the detections on unobserved boxes are false alarms (the known D38 gap)"
    rows += [(lab, "n_det = every detection in the window", m[f"{CLASS}/all/n_det"], n_all, 0.0), (lab, "n_gt < every annotated box (lidar-based `observed`)", float(m[f"{CLASS}/all/n_gt"] <= n_all), 1.0, 0.0)]
    # role swap: the annotated boxes are the detections, the GT holds m extra care boxes the detector missed
    ref = [strip(bs) + [{k2: v for k2, v in f.items() if k2 != "score"} for f in ff] for (bs, _), ff in zip(ann, fi)]
    r = det_numbers(with_boxes(samples, ref), everything, task, "detf")
    lab = f"SWAPPED detf: {mi} care boxes added to the GT (missed by the detector)"
    rows += [(lab, "n_gt", r[f"{CLASS}/all/n_gt"], Nc + mi, 0.0), (lab, "n_tp", r[f"{CLASS}/all/n_tp"], Nc, 0.0), (lab, "ap_all@2 = N / (N + m)", r[f"{CLASS}/all/ap_all@2"], Nc / (Nc + mi), 1e-9)]
    rows += coverage_rows(samples, task, ann, everything, kept)
    return rows


def coverage_rows(samples, task, ann, everything, kept):
    """the `cov` split of the complete GT set: classes are counted straight from the annotated boxes, the recall ceiling follows from the detections removed"""
    from Evaluation import Prediction, Sample
    from Evaluation.contracts import FrameRef
    metric = next(m for m in task.metrics() if m.name == "cov")

    def run(dets):
        counts = [metric.frame(Sample(ref=FrameRef("gt", i, n), inputs=None, reference={0.0: s[0.0]}), Prediction(outputs={"dt": {0.0: {"dets": d}}})) for i, ((n, s), d) in enumerate(zip(samples, dets))]
        return metric.reduce(counts)
    rows = []
    r = run(everything)
    g = lambda res, cls, c, k: res[f"dt+0.00/{cls}/{c}/{k}"]   # noqa: E731
    n_all = sum(1 for bs, _ in ann for b in bs if b["group"] == CLASS)
    n_cls = {c: sum(1 for bs, _ in ann for b in bs if b["group"] == CLASS and b["cls"] == c) for c in ("care", "partial", "dont_care")}
    lab = "cov: a perfect detection of every annotated box"
    rows += [(lab, "n_gt all = every annotated vehicle", g(r, CLASS, "all", "n_gt"), n_all, 0.0), (lab, "recall2m all", g(r, CLASS, "all", "recall2m"), 1.0, 1e-9)]
    for c in n_cls:
        rows += [(lab, f"n_gt {c}", g(r, CLASS, c, "n_gt"), n_cls[c], 0.0)]
    rows += [(lab, "care + partial + dont_care = all", sum(g(r, CLASS, c, "n_gt") for c in n_cls), n_all, 0.0)]
    lab = "cov: nothing detected"
    r0 = run([[] for _ in samples])
    rows += [(lab, "recall2m all", g(r0, CLASS, "all", "recall2m"), 0.0, 0.0), (lab, "n_gt all", g(r0, CLASS, "all", "n_gt"), n_all, 0.0)]
    i, k_by = 0, {c: 0 for c in n_cls}
    removed = []
    for bs, _ in ann:
        o = []
        for b in bs:
            if b["group"] == CLASS:
                i += 1
                if i % 4 == 0:
                    k_by[b["cls"]] += 1
                    continue
            o.append({kk: v for kk, v in b.items() if kk != "cls"})
        removed.append(o)
    r1 = run(removed)
    lab = f"cov: every 4th vehicle not detected ({sum(k_by.values())} of {n_all})"
    rows += [(lab, "recall2m all", g(r1, CLASS, "all", "recall2m"), (n_all - sum(k_by.values())) / n_all, 1e-9)]
    for c in n_cls:
        if n_cls[c]:
            rows += [(lab, f"recall2m {c}", g(r1, CLASS, c, "recall2m"), (n_cls[c] - k_by[c]) / n_cls[c], 1e-9)]
    if metric.pts is not None:                                                    # the radar split needs the annotation table
        lab = "cov (radar returns of the annotations): camera = cam_radar + cam_only; the rest = radar_only + neither"
        cam = g(r, CLASS, "camera", "n_gt")
        rows += [(lab, "n_gt cam_radar + cam_only = camera", g(r, CLASS, "cam_radar", "n_gt") + g(r, CLASS, "cam_only", "n_gt"), cam, 0.0),
                 (lab, "n_gt radar_only + neither = all - camera", g(r, CLASS, "radar_only", "n_gt") + g(r, CLASS, "neither", "n_gt"), n_all - cam, 0.0),
                 (lab, "recall2m cam_radar (all detected)", g(r, CLASS, "cam_radar", "recall2m"), 1.0, 1e-9)]
    return rows


def maps_rows(samples, task):
    from Evaluation import Prediction, Sample
    from Evaluation.contracts import FrameRef
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


def _object_cells(g, task):
    """the cells whose GT velocity is scored (the curve numbers' obj / moving cells and bev_velocity's sets), from the masks and the GT arrays: camera cells without ignored
    boxes x any object layer > 0.5 x known velocity (`flow_weight` > 0 where the GT has it: GtGenerator writes 0 m/s with flow_weight 0 for a box of unknown velocity)"""
    obj = masks(g)["obj_fov"] & np.any([np.asarray(g[c], np.float32) > 0.5 for c in task.groups if c in g], axis=0)
    return obj & (np.asarray(g["flow_weight"], np.float32) > 0) if "flow_weight" in g else obj


def velocity_rows(samples, task):
    from Evaluation import Prediction, Sample
    from Evaluation.contracts import FrameRef
    cur = next(m for m in task.metrics() if m.name == "curve")
    rows = []
    for label, fn in (("velocity field + (0.3, 0.4) m/s", lambda v: v + np.array([0.3, 0.4], np.float32)[:, None, None]), ("velocity field x 0.8", lambda v: 0.8 * v)):
        cs, E, GV, DX = [], [], [], []
        for i, (name, s) in enumerate(samples):
            g = s[0.0]
            m = _object_cells(g, task)
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
    from Evaluation import Prediction, Sample
    from Evaluation.contracts import FrameRef
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
    from Evaluation import Prediction, Sample
    from Evaluation.contracts import FrameRef
    cur = next(m for m in task.metrics() if m.name == "curve")
    rows = []
    for label, fn in (("GT velocity + (0.36, 0.48) m/s", lambda v: v + np.array([0.36, 0.48], np.float32)[:, None, None]), ("GT velocity x 0.8", lambda v: 0.8 * v)):
        cs, GV, DX = [], [], []                          # (0.36, 0.48) has length 0.6: a clear margin to the moving threshold 0.5 (an offset of exactly 0.5 would sit on it)
        for i, (name, s) in enumerate(samples):
            g = s[0.0]
            m = _object_cells(g, task)
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


def _vel_numbers(task, samples, outs):
    from Evaluation import Prediction, Sample
    from Evaluation.contracts import FrameRef
    vm = next(m for m in task.metrics() if m.name == "vel")
    cs = [vm.frame(Sample(ref=FrameRef("gt", i, name), inputs=None, reference={0.0: s[0.0]}), Prediction(outputs={"dt": {0.0: o}}))
          for i, ((name, s), o) in enumerate(zip(samples, outs))]
    return vm.reduce(cs)


def _centres(g, task):
    """the scored GT box centres, computed here from the rule (not by bev_velocity): observed, velocity known, in the front window, centre cell
    floor((c - min) / res) inside the grid and inside obj_fov -> (ix, iy, vx, vy) arrays. Shared with the code under test (not independent): masks(),
    gt_boxes_from_meta, in_front - a bug inside them hits both sides (as for masks() in bev_mutants.py); independent: the floor rule, the indexing, the
    moving split and the known-velocity filter"""
    fov = masks(g)["obj_fov"]
    rows = []
    raw = [b for b in g["meta"]["boxes"] if b.get("observed", True)]
    for b, rb in zip(gt_boxes_from_meta(raw), raw):
        if b["group"] not in task.groups or not rb.get("vel_known", True):
            continue
        i, j = int(math.floor((b["cx"] - task.grid.x_min) / task.grid.res)), int(math.floor((b["cy"] - task.grid.y_min) / task.grid.res))
        if in_front(b) and 0 <= i < fov.shape[0] and 0 <= j < fov.shape[1] and fov[i, j]:
            rows.append((i, j, b["vx"], b["vy"]))
    a = np.array(rows, np.float64).reshape(-1, 4)
    return a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2:].T


def _cell_sets(g, gv, task):
    """the cell sets of bev_velocity, from the masks and the GT arrays: object cells; static = speed <= 0.5 (of gv, the reference); per class"""
    obj = _object_cells(g, task)
    mov = np.hypot(gv[0], gv[1]) > 0.5
    out = {"static": obj & ~mov}
    for c in task.groups:
        if c in g:
            out[c] = obj & (np.asarray(g[c], np.float32) > 0.5)
            out[c + "_moving"] = out[c] & mov
    return out


def velocity_split_rows(samples, task, swap=False):
    """`vel` (bev_velocity.py), every expectation computed here with numpy from the arrays and the box list (cell sets and centre cells by their stated rule).
    Cells: the field = GT + (0.3, 0.4) and 0.8 x GT -> EPE and n per cell set (static, per class, the class's moving part).
    Centres: field = the GT box velocity + (0.3, 0.4) painted into the centre cells, object head = 0.8 x the box velocity painted, zero -> EPE and n, all and
    moving, read back from the painted arrays (two boxes in one cell: the later one wins, the expectation reads the same cell; without collisions the field
    EPE is exactly 0.5 and the object head's 0.2 x the mean speed).
    SWAPPED: the reference is damaged (velocity layer and box velocities + (0.36, 0.48)), the prediction is the original (layer; box velocities painted);
    the static / moving sets follow the damaged reference."""
    off = np.array([0.36, 0.48] if swap else [0.3, 0.4], np.float64)
    fields = ((f"SWAPPED: GT velocity + (0.36, 0.48) as the GT", None),) if swap else \
        (("velocity field + (0.3, 0.4) m/s", lambda v: v + off[:, None, None]), ("velocity field x 0.8", lambda v: 0.8 * v))
    rows = []
    for li, (label, fn) in enumerate(fields):
        samp, cell_outs, cent_outs, acc = [], [], [], {}
        for name, s in samples:
            g = s[0.0]
            gv0 = np.asarray(g["velocity"], np.float64)
            if swap:
                meta = dict(g["meta"], boxes=[dict(b, vel=[b["vel"][0] + off[0], b["vel"][1] + off[1]]) for b in g["meta"]["boxes"]])
                g = dict(g, velocity=(gv0 + off[:, None, None]).astype(np.float32), meta=meta)
                ref, v = gv0 + off[:, None, None], gv0
            else:
                ref, v = gv0, fn(gv0)
            samp.append((name, {0.0: g}))
            cell_outs.append({"vhat": v.astype(np.float32), "objvel": None})
            for k, sel in _cell_sets(g, ref, task).items():
                acc.setdefault(("cells", k), []).append(np.hypot(*(v[:, sel] - ref[:, sel])))
            if li:                                                             # the centre rows once per table
                continue
            ix, iy, bv = _centres(g, task)                                     # bv: the reference box velocity (damaged when swapped)
            pf, po = np.zeros_like(gv0), np.zeros_like(gv0)
            pf[:, ix, iy] = (bv - off[:, None]) if swap else (bv + off[:, None])
            po[:, ix, iy] = (bv - off[:, None]) if swap else 0.8 * bv
            cent_outs.append({"vhat": pf.astype(np.float32), "objvel": po.astype(np.float32)})
            mov = np.hypot(bv[0], bv[1]) > 0.5
            for src, arr in (("field", pf.astype(np.float32)), ("objhead", po.astype(np.float32)), ("zero", np.zeros_like(gv0))):
                e = np.hypot(*(np.asarray(arr, np.float64)[:, ix, iy] - bv))
                acc.setdefault(("centre", src, "all"), []).append(e)
                acc.setdefault(("centre", src, "moving"), []).append(e[mov])
        res = {"cells": _vel_numbers(task, samp, cell_outs)}
        if cent_outs:
            res["centre"] = _vel_numbers(task, samp, cent_outs)
        for key, es in acc.items():
            if key == ("cells", "static_moving"):
                continue
            e = np.concatenate(es)
            k = f"dt+0.00/cells/{key[1]}" if key[0] == "cells" else f"dt+0.00/centre/{key[1]}/{key[2]}"
            lab = f"{label} ({key[0]} {'/'.join(key[1:])})"
            rows += [(lab, "EPE", res[key[0]].get(k + "/epe", float("nan")), float(e.mean()) if len(e) else float("nan"), 1e-5),
                     (lab, "n", res[key[0]].get(k + "/n", 0.0), float(len(e)), 0)]
    return rows


def all_rows(n, sig, seed):
    task = G.build_task()
    samples = G.pick(n, sig, seed)
    return samples, (boxes_rows(samples, task) + maps_rows(samples, task) + velocity_rows(samples, task) + fusion_rows(samples, task)
                     + velocity_split_rows(samples, task)
                     + swap_boxes_rows(samples, task) + swap_maps_rows(samples, task) + swap_velocity_rows(samples, task)
                     + velocity_split_rows(samples, task, swap=True))


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
    ap.add_argument("--sig", default=None, help="GT signature (default: 15efbde2d1 = v5, else the most frequent one in the cache)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    a.sig = G.resolve_sig(a.sig)
    samples, rows = all_rows(a.n, a.sig, a.seed)
    print(f"{len(samples)} real GT frames, signature {a.sig}")
    print_rows(rows)
    gaps = gap_rows(samples, G.build_task())
    print("\nKNOWN GAP OF THE OLD `det` (not asserted: old numbers keep their label, D37; `detf` above closes it, those rows are asserted):")
    print_rows(gaps)
    off = [r for r in rows if not (abs(r[2] - r[3]) <= r[4] or (math.isnan(r[2]) and math.isnan(r[3])))]
    if a.selftest:
        assert not off, f"{len(off)} rows off: {[(r[0], r[1], round(r[2], 4), round(r[3], 4)) for r in off]}"
        print(f"bev_response selftest ok: {len(rows)} numbers reflect exactly what was worsened")
    elif off:
        print(f"{len(off)} of {len(rows)} rows are OFF")


if __name__ == "__main__":
    main()
