"""The kind of every detection error of one frame: each false positive and each missed box with the reason it is an error, as the pictures (bev_views) label it.

    from bev_fusion.evaluation.bev_errors import classify_frame;  r = classify_frame(dets, gts, groups, thr=2.0, min_score=0.3)     # -> {"tp", "fp", "fn"}
Tests: tests/test_bev_errors.py

The matching is the metric's own (`Evaluation.comparison.boxes.match`: best score first, nearest free box of the class within `thr` metres), applied to the
detections with score >= `min-score` (what the pictures draw), so an error here is a box the picture shows in the error colour. (v1 also had a command line that
tabled the errors of a whole evaluation from stores no v2 entry point writes; it is in the tag v1.)

Kinds (the reason an error is what it is; the first that applies wins, in the order listed):
  FP  duplicate       a ground-truth box of its class lies within `thr` m but another detection has it
      wrong_class     a ground-truth box of ANOTHER class within `thr` m that no detection of its own class found (the class confusion: that box is a miss of
      its class) isolated nothing of the ground truth within `thr` m (a ghost; the nearest box is given)
  FN  taken           a detection of its class lies within `thr` m but went to another box (two objects, one detection)
      below_score     a detection of its class within `thr` m exists but its score is under `min-score` (a threshold effect, not a miss of the network)
      wrong_class     a detection of another class sits on it and is itself a false positive (the class confusion)
      missed          nothing of the kinds above (the nearest detection within 2 x `thr`, if any, is given)"""
from __future__ import annotations

import math
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from Evaluation.comparison import boxes as BX


def _dist(a, b):
    return math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])


def classify_frame(dets, gts, groups, thr=2.0, min_score=0.3):
    """One frame -> {"tp": [(det, gt)], "fp": [(det, kind, other)], "fn": [(gt, kind, other)]}; `other` is the box that explains the kind (or None). dets / gts:
    box dicts (group, cx, cy, ...; score for dets). The detections with score < min_score are not matched (they only explain a 'below_score' miss)."""
    shown = [d for d in dets if d.get("score", 1.0) >= min_score]
    low = [d for d in dets if d.get("score", 1.0) < min_score]
    per = {}
    for cls in groups:                                                   # pass 1: the matching of every class (the metric's own)
        ds, gs = [d for d in shown if d["group"] == cls], [g for g in gts if g["group"] == cls]
        match = BX.match(ds, gs, thr)
        per[cls] = (ds, gs, match, {j for j in match if j >= 0})
    free_gt = [g for cls in groups for j, g in enumerate(per[cls][1]) if j not in per[cls][3]]            # ground-truth boxes nobody found
    false_det = [d for cls in groups for d, j in zip(per[cls][0], per[cls][2]) if j < 0]                  # detections that found nothing
    out = {"tp": [], "fp": [], "fn": []}
    for cls in groups:
        ds, gs, match, got = per[cls]
        taken_by = {j: i for i, j in enumerate(match) if j >= 0}
        for i, (d, j) in enumerate(zip(ds, match)):
            if j >= 0:
                out["tp"].append((d, gs[j]))
                continue
            twin = min((g_ for g_ in gs if _dist(d, g_) <= thr), key=lambda g_: _dist(d, g_), default=None)
            other = [g for g in free_gt if g["group"] != cls and _dist(d, g) <= thr]
            if twin is not None:
                out["fp"].append((d, "duplicate", twin))                                      # a box of its class is near, but another detection has it
            elif other:
                out["fp"].append((d, "wrong_class", min(other, key=lambda g: _dist(d, g))))
            else:
                near = min(gts, key=lambda g: _dist(d, g), default=None)
                out["fp"].append((d, "isolated", near))
        for j, g in enumerate(gs):
            if j in taken_by:
                continue
            same = [d for d in shown if d["group"] == cls and _dist(d, g) <= thr]            # (all matched elsewhere, or this one would have taken g)
            lows = [d for d in low if d["group"] == cls and _dist(d, g) <= thr]
            wrong = [d for d in false_det if d["group"] != cls and _dist(d, g) <= thr]
            anyd = [d for d in dets if _dist(d, g) <= 2 * thr]
            if same:
                out["fn"].append((g, "taken", same[0]))
            elif lows:
                out["fn"].append((g, "below_score", max(lows, key=lambda d: d["score"])))
            elif wrong:
                out["fn"].append((g, "wrong_class", wrong[0]))
            else:
                out["fn"].append((g, "missed", None if not anyd else min(anyd, key=lambda d: _dist(d, g))))
    return out
