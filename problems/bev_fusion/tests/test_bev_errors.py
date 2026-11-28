"""Tests of evaluation/bev_errors.py (its former self-test).

    python tests/test_bev_errors.py      (from the problem folder, with its module path)
"""
from __future__ import annotations

from Evaluation.comparison import boxes as BX
from evaluation.bev_errors import classify_frame


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
    # a ghost pedestrian next to a car that WAS found is a ghost, not a class confusion; next to a car nobody found it is the confusion; a duplicate wins over
    # the confusion
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
    selftest()
