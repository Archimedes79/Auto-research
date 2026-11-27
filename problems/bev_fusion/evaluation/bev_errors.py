"""The kind of every detection error of one frame: each false positive and each missed box with the reason it is an error, as the pictures (bev_views) label it.

    from evaluation.bev_errors import classify_frame;  r = classify_frame(dets, gts, groups, thr=2.0, min_score=0.3)     # -> {"tp", "fp", "fn"}
    python evaluation/bev_errors.py --selftest

The matching is the metric's own (`Evaluation.comparison.boxes.match`: best score first, nearest free box of the class within `thr` metres), applied to the detections
with score >= `min-score` (what the pictures draw), so an error here is a box the picture shows in the error colour. (v1 also had a command line that tabled the errors of
a whole evaluation from stores no v2 entry point writes; it is in the tag v1.)

Kinds (the reason an error is what it is; the first that applies wins, in the order listed):
  FP  duplicate       a ground-truth box of its class lies within `thr` m but another detection has it
      wrong_class     a ground-truth box of ANOTHER class within `thr` m that no detection of its own class found (the class confusion: that box is a miss of its class)
      isolated        nothing of the ground truth within `thr` m (a ghost; the nearest box is given)
  FN  taken           a detection of its class lies within `thr` m but went to another box (two objects, one detection)
      below_score     a detection of its class within `thr` m exists but its score is under `min-score` (a threshold effect, not a miss of the network)
      wrong_class     a detection of another class sits on it and is itself a false positive (the class confusion)
      missed          nothing of the kinds above (the nearest detection within 2 x `thr`, if any, is given)"""
from __future__ import annotations

import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "..", ".."), os.path.join(ROOT, "..", "..", "autoresearch")]
from Evaluation.comparison import boxes as BX  # noqa: E402


def _dist(a, b):
    return math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])


def classify_frame(dets, gts, groups, thr=2.0, min_score=0.3):
    """One frame -> {"tp": [(det, gt)], "fp": [(det, kind, other)], "fn": [(gt, kind, other)]}; `other` is the box that explains the kind (or None). dets / gts: box dicts
    (group, cx, cy, ...; score for dets). The detections with score < min_score are not matched (they only explain a 'below_score' miss)."""
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


# ------------------------------------------------------------------------------------------------ self-test
def selftest():
    import random

    def box(g, x, y, s=None):
        b = {"group": g, "cx": float(x), "cy": float(y), "length": 4.0, "width": 2.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0}
        if s is not None:
            b["score"] = s
        return b
    gts = [box("vehicle", 10, 0), box("vehicle", 20, 5), box("vehicle", 30, 0), box("vehicle", 40, 0), box("vehicle", 50, 3), box("pedestrian", 15, 8)]
    dets = [box("vehicle", 10.3, 0, 0.9),                      # TP of gts[0]
            box("vehicle", 10.9, 0.4, 0.8),                    # duplicate on gts[0]
            box("vehicle", 20.2, 5.1, 0.7),                    # TP of gts[1]
            box("pedestrian", 30.1, 0.0, 0.8),                 # wrong class on a vehicle: FP wrong_class, and gts[2] a FN wrong_class
            box("vehicle", 40.4, 0.2, 0.2),                    # under min_score: gts[3] is a FN below_score
            box("vehicle", 55.0, 20.0, 0.95),                  # a ghost: FP isolated
            box("pedestrian", 15.2, 8.1, 0.6)]                 # TP of the pedestrian
    r = classify_frame(dets, gts, ("vehicle", "pedestrian"), thr=2.0, min_score=0.3)
    kinds = lambda lst: sorted(k for _, k, _ in lst)           # noqa: E731
    assert len(r["tp"]) == 3 and kinds(r["fp"]) == ["duplicate", "isolated", "wrong_class"], (len(r["tp"]), kinds(r["fp"]))
    assert kinds(r["fn"]) == ["below_score", "missed", "wrong_class"], kinds(r["fn"])
    fn = {k: g for g, k, _ in r["fn"]}
    assert fn["below_score"]["cx"] == 40 and fn["wrong_class"]["cx"] == 30 and fn["missed"]["cx"] == 50
    # 'taken': two boxes 1 m apart, one detection between them
    r2 = classify_frame([box("vehicle", 10.4, 0, 0.9)], [box("vehicle", 10, 0), box("vehicle", 11, 0)], ("vehicle",))
    assert len(r2["tp"]) == 1 and kinds(r2["fn"]) == ["taken"], r2
    # a ghost pedestrian next to a car that WAS found is a ghost, not a class confusion; next to a car nobody found it is the confusion; a duplicate wins over the confusion
    car = box("vehicle", 20, 0)
    r3 = classify_frame([box("vehicle", 20.1, 0, 0.9), box("pedestrian", 20.8, 0.3, 0.8)], [car], ("vehicle", "pedestrian"))
    assert kinds(r3["fp"]) == ["isolated"] and not r3["fn"], r3
    r4 = classify_frame([box("pedestrian", 20.8, 0.3, 0.8)], [car], ("vehicle", "pedestrian"))
    assert kinds(r4["fp"]) == ["wrong_class"] and kinds(r4["fn"]) == ["wrong_class"], r4
    r5 = classify_frame([box("vehicle", 20.1, 0, 0.9), box("vehicle", 20.6, 0.2, 0.7), box("pedestrian", 20.8, 0.3, 0.8)], [car], ("vehicle", "pedestrian"))
    assert sorted(kinds(r5["fp"])) == ["duplicate", "isolated"], r5                  # the pedestrian: the car is found, so a ghost; the second car: a duplicate
    # the counts equal the metric's matching on random worlds (same function, applied per class and frame)
    rng = random.Random(3)
    for _ in range(40):
        g = [box(rng.choice(("vehicle", "pedestrian")), rng.uniform(0, 60), rng.uniform(-20, 20)) for _ in range(rng.randint(0, 9))]
        d = [dict(box(rng.choice(("vehicle", "pedestrian")), rng.uniform(0, 60), rng.uniform(-20, 20), round(rng.uniform(0, 1), 1))) for _ in range(rng.randint(0, 9))]
        res = classify_frame(d, g, ("vehicle", "pedestrian"), 2.0, 0.3)
        want_tp = sum(sum(j >= 0 for j in BX.match([x for x in d if x["group"] == c and x["score"] >= 0.3], [x for x in g if x["group"] == c], 2.0)) for c in ("vehicle", "pedestrian"))
        n_det = sum(1 for x in d if x["score"] >= 0.3)
        assert len(res["tp"]) == want_tp and len(res["tp"]) + len(res["fp"]) == n_det and len(res["tp"]) + len(res["fn"]) == len(g), (res, d, g)
    print("bev_errors selftest ok: FP duplicate / wrong class / isolated, FN below score / wrong class / missed / taken, counts equal the metric's matching")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
