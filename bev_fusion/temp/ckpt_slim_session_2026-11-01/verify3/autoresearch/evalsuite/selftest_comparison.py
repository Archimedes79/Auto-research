"""Known-answer tests of the comparison core (`comparison/maps.py`, `boxes.py`, `metrics.py`). Registered into the core self-test
(`python -m evalsuite selftest comparison`).

Every function is checked three ways: by a hand calculation, against an independent brute-force implementation written here in plain
loops (no code shared with the core), and - for what the BEV system used before the core existed - against `_REF`, a frozen copy of
that code (bev_fusion/evaluation/metrics.py as of 2026-10-31). The random data have ties on purpose (integer coordinates, scores
rounded to fifths): ties are where matching rules differ."""
from __future__ import annotations

import math
from typing import Callable, List, Tuple

import numpy as np

from . import EvalSuite
from .comparison import boxes as B
from .comparison import maps as M
from .comparison.metrics import BoxMetric, MapMetric
from .contracts import FrameRef, Page, Panel, Prediction, Sample, Source, Subject, Task, View

CHECKS: List[Tuple[str, Callable]] = []


def check(name: str):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def same(a, b, tol=1e-12) -> bool:
    """equal numbers, nan equal to nan"""
    return (a != a and b != b) or abs(a - b) <= tol * max(1.0, abs(a))


# ------------------------------------------------------------------------------------------------ the reference: the former BEV code, frozen
class _REF:
    """bev_fusion/evaluation/metrics.py (match, ap_nuscenes, ap_all_points, tp_errors, dilate) as of 2026-10-31, BEFORE the move into
    comparison/. Do not edit: it is what the core must keep reproducing."""

    @staticmethod
    def match(frames, thr):
        cand = sorted(((d["score"], f, d) for f, (ds, _) in enumerate(frames) for d in ds), key=lambda c: -c[0])
        used, out = set(), []
        for sc, f, d in cand:
            best, bd = None, thr
            for j, g in enumerate(frames[f][1]):
                if (f, j) in used:
                    continue
                dist = math.hypot(d["cx"] - g["cx"], d["cy"] - g["cy"])
                if dist <= bd:
                    best, bd = j, dist
            if best is not None:
                used.add((f, best))
            out.append((sc, best is not None, d, None if best is None else frames[f][1][best]))
        return out, sum(len(gs) for _, gs in frames)

    @staticmethod
    def ap_nuscenes(tp_flags, n_gt, min_recall=0.1, min_precision=0.1):
        if n_gt == 0:
            return float("nan")
        if len(tp_flags) == 0:
            return 0.0
        f = np.asarray(tp_flags, float)
        tp, fp = np.cumsum(f), np.cumsum(1 - f)
        prec, rec = tp / (tp + fp), tp / n_gt
        p = np.interp(np.linspace(0, 1, 101), rec, prec, right=0)
        p = p[int(round(100 * min_recall)) + 1:] - min_precision
        p[p < 0] = 0
        return float(np.mean(p)) / (1.0 - min_precision)

    @staticmethod
    def ap_all_points(tp_flags, n_gt):
        if n_gt == 0:
            return float("nan")
        if len(tp_flags) == 0:
            return 0.0
        f = np.asarray(tp_flags, float)
        tp, fp = np.cumsum(f), np.cumsum(1 - f)
        prec, rec = tp / (tp + fp), tp / n_gt
        prec = np.maximum.accumulate(prec[::-1])[::-1]
        return float(np.sum(prec * np.diff(np.r_[0.0, rec])))

    @staticmethod
    def tp_errors(d, g):
        from shapely.geometry import Point, Polygon

        def shape(b):
            if "radius" in b:
                return Point(b["cx"], b["cy"]).buffer(b["radius"], 32)
            c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
            hl, hw = b["length"] / 2, b["width"] / 2
            return Polygon([(b["cx"] + c * a - s * w, b["cy"] + s * a + c * w) for a, w in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))])

        def circle(b):
            return b if "radius" in b else dict(b, radius=max(b["length"], b["width"]) / 2)
        if "radius" in d or "radius" in g:
            d, g = circle(d), circle(g)
            rmin, rmax = sorted((d["radius"], g["radius"]))
            ase, aoe = 1.0 - (rmin / rmax) ** 2, float("nan")
        else:
            inter = min(d["length"], g["length"]) * min(d["width"], g["width"])
            ase = 1.0 - inter / (d["length"] * d["width"] + g["length"] * g["width"] - inter)
            aoe = abs(math.remainder(d["yaw"] - g["yaw"], 2 * math.pi))
        sd, sg = shape(d), shape(g)
        u = sd.union(sg).area
        return {"ate": math.hypot(d["cx"] - g["cx"], d["cy"] - g["cy"]), "ase": ase, "aoe": aoe,
                "ave": math.hypot(d["vx"] - g["vx"], d["vy"] - g["vy"]), "biou": sd.intersection(sg).area / u if u else 0.0}

    @staticmethod
    def dilate(m, radius_cells):
        m = np.asarray(m, bool)
        r = int(math.floor(radius_cells + 1e-9))
        if r <= 0:
            return m.copy()
        X, Y = m.shape
        p = np.pad(m, r)
        out = np.zeros_like(m)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= radius_cells * radius_cells + 1e-9:
                    out |= p[r + dx:r + dx + X, r + dy:r + dy + Y]
        return out


# ------------------------------------------------------------------------------------------------ random worlds
def random_frames(rng, n_frames=30, classes=("car", "ped"), tie_scores=True, tie_positions=True):
    """[(dets, gts)] with boxes of both shapes; integer coordinates and scores in fifths make exact ties likely"""
    frames = []
    for _ in range(n_frames):
        def box(cls):
            x, y = (float(rng.integers(0, 12)), float(rng.integers(0, 12))) if tie_positions else (float(rng.uniform(0, 12)), float(rng.uniform(0, 12)))
            b = {"group": cls, "cx": x, "cy": y, "vx": float(rng.normal()), "vy": float(rng.normal())}
            if cls == "ped":
                b["radius"] = float(rng.uniform(0.3, 0.6))
            else:
                b.update(length=float(rng.uniform(3, 5)), width=float(rng.uniform(1.5, 2.2)), yaw=float(rng.uniform(-4, 4)))
            return b
        gts = [box(classes[int(rng.integers(0, len(classes)))]) for _ in range(int(rng.integers(0, 7)))]
        dets = [dict(box(classes[int(rng.integers(0, len(classes)))]), score=(int(rng.integers(1, 6)) / 5.0 if tie_scores else float(rng.uniform())))
                for _ in range(int(rng.integers(0, 9)))]
        frames.append((dets, gts))
    return frames


def of_class(frames, cls):
    return [([d for d in ds if d["group"] == cls], [g for g in gs if g["group"] == cls]) for ds, gs in frames]


# ================================================================================================ maps
@check("comparison.maps: confusion and scores by hand; valid cells; nothing to find is nan; float maps and shape mismatches are refused; counts add up")
def _():
    gt = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]], bool)
    pr = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]], bool)
    c = M.confusion(pr, gt)
    assert c.tolist() == [2, 2, 1, 4]                                  # tp (0,0) (1,1); fp (1,0) (2,2); fn (0,1); tn the other four
    s = M.scores(c)
    assert same(s["iou"], 2 / 5) and same(s["precision"], 1 / 2) and same(s["recall"], 2 / 3) and same(s["f1"], 2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    valid = np.ones((3, 3), bool)
    valid[2, :] = False                                                # the last row is not evaluated: the false positive at (2, 2) goes
    assert M.confusion(pr, gt, valid).tolist() == [2, 1, 1, 2]
    assert M.confusion(pr.astype(np.uint8), gt.astype(int)).tolist() == c.tolist()           # 0 / 1 integer maps are fine
    empty = M.scores(M.confusion(np.zeros((3, 3), bool), np.zeros((3, 3), bool)))
    assert all(v != v for v in empty.values()), "nothing to find and nothing predicted: every ratio is nan"
    nothing = M.scores(M.confusion(np.zeros((3, 3), bool), gt))        # something to find, nothing predicted: recall 0, precision undefined
    assert nothing["recall"] == 0.0 and nothing["precision"] != nothing["precision"] and nothing["f1"] != nothing["f1"] and nothing["iou"] == 0.0
    assert M.f1(0.0, 0.0) == 0.0 and M.f1(float("nan"), 0.5) != M.f1(float("nan"), 0.5)
    for bad, exc in ((lambda: M.confusion(pr.astype(float), gt), TypeError), (lambda: M.confusion(pr, gt[:2]), ValueError),
                     (lambda: M.confusion(pr, gt[:1]), ValueError),                   # numpy would broadcast this one silently
                     (lambda: M.tolerant_confusion(pr, gt[:1], 1), ValueError),
                     (lambda: M.confusion(pr, gt, valid[:1]), ValueError),             # a `valid` that numpy would broadcast
                     (lambda: M.confusion(pr, gt, valid[:, :1]), ValueError),
                     (lambda: M.tolerant_confusion(pr, gt, 1, valid[:, :1]), ValueError),
                     (lambda: M.confusion(pr, gt, valid[:, :2]), ValueError),
                     (lambda: M.confusion(pr.astype(np.uint8) * 255, gt), ValueError),   # an integer map that is not 0 / 1 (255 = "ignore") is refused
                     (lambda: M.tolerant_confusion(pr, gt, -1), ValueError),
                     (lambda: M.tolerant_confusion(np.zeros((2, 3, 3), bool), np.zeros((2, 3, 3), bool), 1), ValueError)):   # a tolerance needs 2-D maps
        try:
            bad()
        except exc:
            pass
        else:
            raise AssertionError("a float map, an integer map other than 0 / 1 or a shape mismatch must be refused")
    cube = np.random.default_rng(9).random((3, 4, 5)) > 0.5                       # tolerance 0 works for maps of any rank
    assert M.tolerant_confusion(cube, ~cube, 0).tolist() == [0, int(cube.sum()), 0, int((~cube).sum())] and M.confusion(cube, cube).tolist() == [int(cube.sum()), 0, 0, int((~cube).sum())]
    rng = np.random.default_rng(0)
    a, b = rng.random((20, 30)) > 0.5, rng.random((20, 30)) > 0.4
    assert (M.confusion(a, b) == M.confusion(a[:7], b[:7]) + M.confusion(a[7:], b[7:])).all(), "counts of the parts add up to the counts of the whole"
    assert M.confusion(a, b).sum() == a.size


@check("comparison.maps: dilate and the tolerant comparison against a brute-force distance search; tolerance 0 is the plain comparison; the former BEV dilate")
def _():
    rng = np.random.default_rng(1)

    def brute_near(src, dst, tol):
        """for every cell of src: is a set cell of dst within tol (centre distance in cells)? plain loops"""
        pts = [(i, j) for i in range(dst.shape[0]) for j in range(dst.shape[1]) if dst[i, j]]
        return np.array([[any(math.hypot(i - a, j - b) <= tol + 1e-9 for a, b in pts) for j in range(src.shape[1])] for i in range(src.shape[0])], bool)
    for tol in (0, 1, 1.5, 2, 2.5):
        p, g = rng.random((14, 17)) > 0.85, rng.random((14, 17)) > 0.85
        valid = rng.random((14, 17)) > 0.2
        assert (M.dilate(g, tol) == brute_near(np.ones_like(g), g, tol)).all(), tol
        assert (M.dilate(g, tol) == _REF.dilate(g, tol)).all(), "the same dilation as the BEV system used before"
        near_g, near_p = brute_near(np.ones_like(g), g, tol), brute_near(np.ones_like(p), p, tol)
        want = [(p & near_g & valid).sum(), (p & ~near_g & valid).sum(), (g & near_p & valid).sum(), (g & ~near_p & valid).sum()]
        assert M.tolerant_confusion(p, g, tol, valid).tolist() == want, tol
        assert M.tolerant_counts(p, g, brute_near(np.ones_like(g), g, tol), near_p, valid).tolist() == want, "the counting step with dilations given"
        for bad_near in (near_p[:1], near_p[:, :1]):                             # a dilation of another shape (numpy would broadcast it) is refused
            try:
                M.tolerant_counts(p, g, bad_near, near_p, valid)
            except ValueError:
                pass
            else:
                raise AssertionError("a dilation of the wrong shape must be refused")
    one = np.zeros((9, 9), bool)
    one[4, 4] = True
    assert (M.dilate(one, 0.3 / 0.1) == M.dilate(one, 3)).all() and M.dilate(one, 3).sum() == 29, "a radius that is 3 up to float noise (0.3 / 0.1) is 3"
    p, g = rng.random((14, 17)) > 0.7, rng.random((14, 17)) > 0.7
    tp, fp, fn, _tn = M.confusion(p, g)
    assert M.tolerant_confusion(p, g, 0).tolist() == [tp, fp, tp, fn], "tolerance 0 is the plain comparison"
    assert all(same(M.tolerant_scores(M.tolerant_confusion(p, g, 0))[k], M.scores(M.confusion(p, g))[k]) for k in ("iou", "precision", "recall", "f1"))
    # a 10 x 10 block shifted by one cell: iou 90 / 110 plain, 1 at a tolerance of one cell (the BEV system's own check)
    block = np.zeros((40, 40), bool)
    block[10:20, 10:20] = True
    shifted = np.roll(block, 1, axis=0)
    assert same(M.tolerant_scores(M.tolerant_confusion(shifted, block, 0))["iou"], 90 / 110)
    assert same(M.tolerant_scores(M.tolerant_confusion(shifted, block, 1))["iou"], 1.0)
    # an asymmetric case by hand (1 x 16 cells, tolerance 1): predictions at 0, 2 and 14, ground truth at 1 and 8. Both predictions at 0 and 2
    # lie within 1 of the ground truth at 1, the one at 14 is far from everything; the ground truth at 8 is far from every prediction.
    # p_hit 2, p_miss 1, g_hit 1, g_miss 1: precision 2/3, recall 1/2, tp = 1.5 -> iou 1.5 / 3.5 = 3/7, f1 4/7
    pred, truth = np.zeros((1, 16), bool), np.zeros((1, 16), bool)
    pred[0, [0, 2, 14]], truth[0, [1, 8]] = True, True
    c = M.tolerant_confusion(pred, truth, 1)
    s = M.tolerant_scores(c)
    assert c.tolist() == [2, 1, 1, 1] and same(s["precision"], 2 / 3) and same(s["recall"], 1 / 2) and same(s["iou"], 3 / 7) and same(s["f1"], 4 / 7), (c, s)
    # the dilation uses the whole maps: a hit just outside `valid` still protects a cell inside it
    one_p, one_g = np.zeros((5, 5), bool), np.zeros((5, 5), bool)
    one_p[2, 2], one_g[2, 3] = True, True
    only_p = np.zeros((5, 5), bool)
    only_p[2, 2] = True
    assert M.tolerant_confusion(one_p, one_g, 1, valid=only_p).tolist() == [1, 0, 0, 0]


# ================================================================================================ boxes
@check("comparison.boxes.match: score order, one-to-one, a distance of exactly thr counts, equal distances take the later box, equal scores keep list order, nan score is refused")
def _():
    def box(x, y=0.0, score=None):
        return dict({"cx": x, "cy": y}, **({} if score is None else {"score": score}))
    gts = [box(0.0), box(10.0)]
    assert B.match([box(0.5, 0, 0.9), box(9.0, 0, 0.8)], gts, 2.0) == [0, 1]
    assert B.match([box(0.5, 0, 0.5), box(0.2, 0, 0.9)], gts, 2.0) == [-1, 0], "the better score takes the only box in reach, the other is a false positive"
    assert B.match([box(2.0, 0, 0.9)], gts, 2.0) == [0] and B.match([box(2.0001, 0, 0.9)], gts, 2.0) == [-1], "exactly thr is a hit, beyond is not"
    assert B.match([box(2.0 + 1e-12, 0, 0.9)], gts, 2.0) == [-1], "the threshold has no tolerance of its own"
    assert B.match([box(float("nan"), 0, 0.9)], gts, 2.0) == [-1], "a nan centre is never within reach"
    # EQUAL scores are served in list order, and that order decides which detection becomes the true positive (documented, pinned): ground truth
    # at 0.0 and 1.9, threshold 2.0; detection A at 0.4, B at -0.3, both with score 0.5. [A, B]: A takes 0.0, B finds nothing else in reach
    # (1.9 is 2.2 away). [B, A]: B takes 0.0, A (1.5 from 1.9) takes 1.9.
    two_gt = [box(0.0), box(1.9)]
    assert B.match([box(0.4, 0, 0.5), box(-0.3, 0, 0.5)], two_gt, 2.0) == [0, -1]
    assert B.match([box(-0.3, 0, 0.5), box(0.4, 0, 0.5)], two_gt, 2.0) == [0, 1]
    nan_err = np.full((2, 5), np.nan)                          # the AP of the two orders (same scores, so the default tie rule puts the false positive first)
    ap_ab = B.detection_numbers(np.array([0.5, 0.5]), np.array([[1.0], [0.0]]), nan_err, 2, (2.0,))["ap@2"]
    ap_ba = B.detection_numbers(np.array([0.5, 0.5]), np.array([[1.0], [1.0]]), nan_err, 2, (2.0,))["ap@2"]
    assert same(ap_ab, 0.1012, 1e-3) and same(ap_ba, 1.0), (ap_ab, ap_ba)
    assert same(B.detection_numbers(np.array([0.5, 0.5]), np.array([[1.0], [0.0]]), nan_err, 2, (2.0,), None, "list")["ap@2"], 0.4383, 1e-3)
    assert B.match([box(5.0, 0, 0.9)], [box(0.0), box(10.0)], 6.0) == [1], "two free boxes equally near: the later one"
    assert B.match([box(0.0, 0, 0.5), box(0.0, 0, 0.5)], [box(0.0)], 1.0) == [0, -1], "equal scores: the first in the list is served first"
    assert B.match([], gts, 1.0) == [] and B.match([box(0.0, 0, 1.0)], [], 1.0) == [-1]
    assert B.match([box(0.0, 0, 0.9), box(0.0, 0, 0.8)], [box(0.0), box(0.5)], 1.0) == [0, 1], "the second detection takes the next box, no box twice"
    try:
        B.match([box(0.0, 0, float("nan"))], gts, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("nan score")
    dets = [box(1.0, 0, 0.9)]
    keep = [dict(d) for d in dets]
    B.match(dets, gts, 2.0)
    assert dets == keep, "the inputs are not changed"


@check("comparison.boxes.match_frames: the former BEV matching on random frames with ties (same rows, same objects, same order); frames do not compete")
def _():
    rng = np.random.default_rng(2)
    for thr in (0.5, 1.0, 2.0, 4.0):
        frames = of_class(random_frames(rng, 40), "car")
        new, n_new = B.match_frames(frames, thr)
        old, n_old = _REF.match(frames, thr)
        assert n_new == n_old and len(new) == len(old)
        for a, b in zip(new, old):
            assert a[0] == b[0] and a[1] == b[1] and a[2] is b[2] and a[3] is b[3], (thr, a, b)
    some = of_class(random_frames(rng, 8), "car")
    from_list, from_iter = B.match_frames(some, 2.0), B.match_frames(iter(some), 2.0)           # a one-shot iterator gives the same n_gt
    assert from_iter[1] == from_list[1] > 0 and len(from_iter[0]) == len(from_list[0])
    frames = of_class(random_frames(rng, 12), "car")
    order = list(rng.permutation(len(frames)))
    flags_a = sorted((d["score"], id(d), ok) for _, ok, d, _ in B.match_frames(frames, 2.0)[0])
    flags_b = sorted((d["score"], id(d), ok) for _, ok, d, _ in B.match_frames([frames[i] for i in order], 2.0)[0])
    assert flags_a == flags_b, "which detections are true positives does not depend on the order of the frames"


@check("comparison.boxes.tp_errors: hand cases, yaw wrap, circles, oriented IoU, missing velocity, zero size; the former BEV numbers on random pairs")
def _():
    g0 = {"cx": 10.0, "cy": 1.0, "length": 4.5, "width": 1.8, "yaw": 0.3, "vx": 5.0, "vy": 0.0}
    e = B.tp_errors(dict(g0, yaw=g0["yaw"] + math.pi / 2, vx=g0["vx"] + 1.0), g0)
    assert same(e["aoe"], math.pi / 2) and same(e["ase"], 0.0) and same(e["biou"], 0.25, 1e-6) and same(e["ave"], 1.0) and same(e["ate"], 0.0)
    assert same(B.tp_errors(dict(g0, length=4.0, width=2.0), g0)["ase"], 1 - 4.0 * 1.8 / (4.0 * 2.0 + 4.5 * 1.8 - 4.0 * 1.8))
    ped = {"cx": 0.0, "cy": 0.0, "radius": 0.4, "vx": 0.0, "vy": 0.0}
    c = B.tp_errors(dict(ped, radius=0.5, cx=0.3, cy=0.4), ped)
    assert same(c["ase"], 1 - 0.64) and c["aoe"] != c["aoe"] and same(c["ate"], 0.5)
    assert same(B.tp_errors(dict(g0, yaw=3.1), dict(g0, yaw=-3.1))["aoe"], 2 * math.pi - 6.2), "the smallest angle across the wrap at +-pi"
    shifted = B.tp_errors(dict(g0, yaw=0.0, cx=1.0, cy=0.0, length=4.0, width=2.0), dict(g0, yaw=0.0, cx=0.0, cy=0.0, length=4.0, width=2.0))
    assert same(shifted["biou"], 3 * 2 / (8 + 8 - 6), 1e-9) and same(shifted["ate"], 1.0)
    no_v = {k: v for k, v in g0.items() if k not in ("vx", "vy")}
    assert B.tp_errors(no_v, no_v)["ave"] != B.tp_errors(no_v, no_v)["ave"], "no velocity: the velocity error is nan"
    with_v = dict(g0, vx=1.0, vy=1.0)
    assert B.tp_errors(no_v, with_v)["ave"] != B.tp_errors(no_v, with_v)["ave"] and B.tp_errors(with_v, no_v)["ave"] != B.tp_errors(with_v, no_v)["ave"], \
        "a velocity missing on one side: nan, not 0 and not an error"
    half_v = dict(no_v, vx=1.0)                                                   # vx there, vy missing: still nan, on either side
    assert B.tp_errors(half_v, with_v)["ave"] != B.tp_errors(half_v, with_v)["ave"] and B.tp_errors(with_v, half_v)["ave"] != B.tp_errors(with_v, half_v)["ave"]
    flat = dict(g0, length=0.0, width=0.0)
    assert B.tp_errors(flat, flat)["ase"] != B.tp_errors(flat, flat)["ase"] and B.tp_errors(flat, flat)["biou"] == 0.0, \
        "a box of size 0: ase nan (0 / 0), biou 0, not a ZeroDivisionError"
    lying = B.tp_errors(dict(g0, yaw=float("nan")), g0)
    assert lying["biou"] != lying["biou"] and lying["aoe"] != lying["aoe"] and same(lying["ate"], 0.0), "a nan yaw gives nan, not a crash in shapely"
    wide = {"cx": 0.0, "cy": 0.0, "length": 2.0, "width": 4.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0}          # width > length: the circle is max(l, w) / 2 = 2
    assert same(B.tp_errors(wide, dict(ped, radius=1.0))["ase"], 1 - (1.0 / 2.0) ** 2), "a rectangle against a circle: radius max(length, width) / 2"
    assert same(B.tp_errors(wide, dict(ped, radius=1.0))["biou"], math.pi / (math.pi * 4), 1e-3), "... and biou is that of two circles"
    rng = np.random.default_rng(3)
    pairs = 0
    for dets, gts in random_frames(rng, 40, tie_scores=False, tie_positions=False):
        for d, g in zip(dets, gts):
            new, old = B.tp_errors(d, g), _REF.tp_errors(d, g)
            assert all(same(new[k], old[k]) for k in old), (new, old)
            pairs += 1
    assert pairs > 50


@check("comparison.boxes.ap_*: hand ranking, perfect and empty cases, an independent implementation of both definitions, the former BEV numbers")
def _():
    # 2 ground-truth boxes, a false positive (0.9) above a true positive (0.8): all-point AP 0.5 * 0.5; nuScenes sum_{k=11}^{50}(k/100 - 0.1) / 90 / 0.9
    assert same(B.ap_all_points([0, 1], 2), 0.25) and same(B.ap_nuscenes([0, 1], 2), 8.2 / 81)
    assert same(B.ap_all_points([1, 1], 2), 1.0) and same(B.ap_nuscenes([1, 1], 2), 1.0)
    assert B.ap_nuscenes([], 3) == 0.0 and B.ap_all_points([], 3) == 0.0
    assert B.ap_nuscenes([0, 1], 0) != B.ap_nuscenes([0, 1], 0) and B.ap_all_points([], 0) != B.ap_all_points([], 0), "no ground truth: nan"
    assert B.ap_nuscenes([0, 0], 2) == 0.0 and B.ap_all_points([0, 0], 2) == 0.0
    for fn_ in (B.ap_nuscenes, B.ap_all_points):
        try:
            fn_([1, 1, 1], 2)
        except ValueError:
            pass
        else:
            raise AssertionError("more true positives than ground-truth boxes must be refused")

    def pr_points(flags, n_gt):
        tp = fp = 0
        out = []
        for f in flags:
            tp, fp = tp + f, fp + (1 - f)
            out.append((tp / n_gt, tp / (tp + fp)))                          # (recall, precision) after each detection
        return out

    def voc_all_point(flags, n_gt):
        """the textbook algorithm: envelope from the right, then the area over the recall steps"""
        pts = pr_points(flags, n_gt)
        mrec, mpre = [0.0] + [r for r, _ in pts] + [1.0], [0.0] + [p for _, p in pts] + [0.0]
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        return sum((mrec[i + 1] - mrec[i]) * mpre[i + 1] for i in range(len(mrec) - 1))

    def devkit(flags, n_gt, min_recall=0.1, min_precision=0.1):
        """nuScenes calc_ap with the interpolation written out: linear between the points, 0 beyond the last recall, the first precision before the first"""
        pts = pr_points(flags, n_gt)
        rec, prec = [r for r, _ in pts], [p for _, p in pts]
        vals = []
        for k in range(101):
            r = k / 100
            if r > rec[-1]:
                vals.append(0.0)
            elif r < rec[0]:
                vals.append(prec[0])
            else:
                i = max(j for j in range(len(rec)) if rec[j] <= r)
                vals.append(prec[i] if rec[i] == r or i == len(rec) - 1 else prec[i] + (prec[i + 1] - prec[i]) * (r - rec[i]) / (rec[i + 1] - rec[i]))
        vals = [max(v - min_precision, 0.0) for v in vals[int(round(100 * min_recall)) + 1:]]
        return sum(vals) / len(vals) / (1.0 - min_precision)
    # a recall plateau on a grid value: [1, 0, 1, 1] with 5 ground-truth boxes has recall 0.2, 0.2, 0.4, 0.6 and precision 1, 1/2, 2/3, 3/4;
    # at recall 0.2 the interpolation takes the precision after the LAST detection of the plateau (1/2). The real devkit does the same
    # (np.interp); 0.37603 is its value, computed with accumulate + calc_ap on such a ranking.
    plateau = [1, 0, 1, 1]
    assert same(B.ap_nuscenes(plateau, 5), 0.37603, 1e-4) and same(B.ap_nuscenes(plateau, 5), devkit(plateau, 5), 1e-9)
    rng = np.random.default_rng(4)
    for _ in range(60):
        n = int(rng.integers(1, 40))
        flags = (rng.random(n) < rng.uniform(0.1, 0.9)).astype(float)
        n_gt = int(flags.sum() + rng.integers(0, 15)) or 1
        assert same(B.ap_all_points(flags, n_gt), voc_all_point(flags, n_gt), 1e-9), (flags, n_gt)
        assert same(B.ap_nuscenes(flags, n_gt), devkit(flags, n_gt), 1e-9), (flags, n_gt)
        assert same(B.ap_all_points(flags, n_gt), _REF.ap_all_points(flags, n_gt)) and same(B.ap_nuscenes(flags, n_gt), _REF.ap_nuscenes(flags, n_gt))


@check("comparison.boxes.match_flags + detection_numbers: the rows of single frames give the numbers of one global matching (any order of frames, ties, thresholds without tp_threshold)")
def _():
    rng = np.random.default_rng(5)
    thresholds = (0.5, 1.0, 2.0, 4.0)
    frames = of_class(random_frames(rng, 50), "car")

    def oracle(fr, tp_thr=2.0):
        row = {}
        for t in thresholds:
            mm, n_gt = _REF.match(fr, t)
            fl = [x[1] for x in mm]
            row["ap@%g" % t], row["ap_all@%g" % t] = _REF.ap_nuscenes(fl, n_gt), _REF.ap_all_points(fl, n_gt)
            if tp_thr is not None and t == tp_thr:
                errs = [_REF.tp_errors(d, g) for _, ok, d, g in mm if ok]
                for k in B.TP_ERROR_KEYS:
                    v = [e[k] for e in errs if e[k] == e[k]]
                    row[k] = float(np.mean(v)) if v else float("nan")
                row["n_tp"] = len(errs)
        row["n_gt"], row["n_det"] = sum(len(g) for _, g in fr), sum(len(d) for d, _ in fr)
        return row

    def numbers(fr, tp_thr=2.0, ties="list"):
        scores, flags, errors, n_gt = [], [], [], 0
        for dets, gts in fr:
            f, e = B.match_flags(dets, gts, thresholds, tp_thr)
            assert f.shape == (len(dets), len(thresholds)) and e.shape == (len(dets), 5)
            scores += [d["score"] for d in dets]
            flags.append(f)
            errors.append(e)
            n_gt += len(gts)
        return B.detection_numbers(np.array(scores), np.concatenate(flags), np.concatenate(errors), n_gt, thresholds, tp_thr, ties)
    want, got = oracle(frames), numbers(frames)                              # ties="list": the former BEV order of equal scores
    assert list(want) == list(got) and all(same(want[k], got[k]) for k in want), (want, got)
    # the order of the frames matters only where scores are EQUAL (rows with equal scores keep their order, so AP depends on it);
    # with distinct scores the numbers do not depend on it
    distinct = of_class(random_frames(rng, 50, tie_scores=False), "car")
    a, b = numbers(distinct), numbers([distinct[i] for i in rng.permutation(len(distinct))])
    assert all(same(a[k], b[k]) for k in a), "distinct scores: the numbers do not depend on the order of the frames"
    assert all(same(a[k], oracle(distinct)[k]) for k in a)
    # equal scores (the random frames have many): "list" depends on the order of the frames, "pessimistic" (the default) does not
    shuffled = [frames[i] for i in rng.permutation(len(frames))]
    p_a, p_b = numbers(frames, ties="pessimistic"), numbers(shuffled, ties="pessimistic")
    assert all(same(p_a[k], p_b[k]) for k in p_a), "ties pessimistic: the numbers are a function of the set of rows"
    assert any(not same(got[k], numbers(shuffled)[k]) for k in got if k.startswith("ap")), "ties 'list': the order matters (else this test has no ties)"
    for t in thresholds:                                                    # independent: sort the oracle's rows by (-score, false positive first)
        mm, n_gt = _REF.match(frames, t)
        fl = [x[1] for x in sorted(mm, key=lambda x: (-x[0], x[1]))]
        assert same(p_a["ap@%g" % t], _REF.ap_nuscenes(fl, n_gt)) and same(p_a["ap_all@%g" % t], _REF.ap_all_points(fl, n_gt)), t
    one_tie = B.detection_numbers(np.array([0.9, 0.9]), np.array([[1.0], [0.0]]), np.full((2, 5), np.nan), 2, (1.0,), None, "pessimistic")
    assert same(one_tie["ap_all@1"], 0.25) and same(B.detection_numbers(np.array([0.9, 0.9]), np.array([[1.0], [0.0]]), np.full((2, 5), np.nan), 2, (1.0,), None, "list")["ap_all@1"], 0.5)
    try:
        B.detection_numbers(np.zeros(0), np.zeros((0, 1)), np.zeros((0, 5)), 1, (1.0,), None, "luck")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown tie rule must be refused")
    # a mis-shaped array is an error, not a silent reshape; nan errors are skipped in the mean (one velocity missing among the true positives)
    for flags_, errors_ in ((np.zeros((4, 3)), np.zeros((3, 5))), (np.zeros((4, 3)).T, np.zeros((3, 5)))):      # (T, n) instead of (n, T)
        try:
            B.detection_numbers(np.zeros(3), flags_, errors_, 3, (1.0, 2.0, 4.0), 2.0)
        except ValueError:
            pass
        else:
            raise AssertionError("flags of the wrong shape must be refused")
    err = np.array([[1.0, 0.1, 0.2, np.nan, 0.5], [3.0, 0.3, 0.4, 2.0, 0.7]])          # the first true positive has no velocity (ave nan)
    mixed = B.detection_numbers(np.array([0.9, 0.8]), np.ones((2, 1)), err, 2, (2.0,), 2.0)
    assert same(mixed["ate"], 2.0) and same(mixed["ave"], 2.0) and same(mixed["ase"], 0.2) and mixed["n_tp"] == 2, mixed
    # the mean of an error is summed in rank order (best score first), so the BEV numbers stay bit for bit what they always were:
    # ranks 0.2, 0.3, 0.1 give 0.19999999999999998, the row order 0.1, 0.2, 0.3 would give 0.20000000000000004 (exact equality on purpose)
    ranked = B.detection_numbers(np.array([0.5, 0.9, 0.7]), np.ones((3, 1)), np.tile([[0.1], [0.2], [0.3]], (1, 5)), 3, (2.0,), 2.0)["ate"]
    assert ranked == (0.2 + 0.3 + 0.1) / 3 and ranked != (0.1 + 0.2 + 0.3) / 3, ranked
    no_tp = numbers(frames, tp_thr=None)
    assert "ate" not in no_tp and "n_tp" not in no_tp and all(same(no_tp[k], got[k]) for k in no_tp)
    f, e = B.match_flags([], [{"cx": 0.0, "cy": 0.0}], thresholds, 2.0)
    assert f.shape == (0, 4) and e.shape == (0, 5)
    assert B.detection_numbers(np.zeros(0), np.zeros((0, 4)), np.zeros((0, 5)), 3, thresholds, 2.0)["ap@2"] == 0.0


@check("comparison.boxes.velocity_*: hand case (moving / static / missing velocity / predicted standing, the speed boundaries), counts add up, the mean of both groups = the `ave` of detection_numbers, brute force on random worlds")
def _():
    def box(x, v=None, score=None):
        b = {"cx": float(x), "cy": 0.0, "length": 4.0, "width": 2.0, "yaw": 0.0}
        if v is not None:
            b["vx"], b["vy"] = v
        if score is not None:
            b["score"] = score
        return b
    gts = [box(0, (4.0, 0.0)), box(10, (0.2, 0.0)), box(20, (0.0, 2.0)), box(30), box(40, (0.0, -3.0))]
    dets = [box(0.5, (3.0, 0.0), 0.9), box(10, (0.0, 0.0), 0.8), box(100, (1.0, 0.0), 0.7), box(30, (1.0, 1.0), 0.6), box(40, (0.1, 0.0), 0.5)]
    c = B.velocity_counts(dets, gts, 2.0)
    want = dict(zip(B.VELOCITY_COUNT_KEYS, [3, 1, 2, 1, 1.0 + math.hypot(0.1, 3.0), 0.2, 4.0 + 3.0, 0.2, 1]))      # box(30) has no velocity, box(100) is a false positive
    assert c.shape == (9,) and all(same(c[i], want[k]) for i, k in enumerate(B.VELOCITY_COUNT_KEYS)), (c, want)
    n = B.velocity_numbers(c)
    assert same(n["ave_moving"], (1.0 + math.hypot(0.1, 3.0)) / 2) and same(n["ave0_moving"], 3.5) and same(n["ave_skill_moving"], 1 - (1.0 + math.hypot(0.1, 3.0)) / 7.0)
    assert same(n["ave_static"], 0.2) and same(n["ave0_static"], 0.2) and same(n["ave_skill_static"], 0.0) and same(n["ave_slow_moving"], 0.5)
    assert (n["n_tp_moving"], n["n_tp_static"], n["n_gt_moving"], n["n_gt_static"]) == (2, 1, 3, 1)
    # the boundaries: a speed of exactly `moving` is static; a detection of exactly `slow` is not slow; the detection's speed alone decides "slow"
    edge = B.velocity_counts([box(0, (0.5, 0.0), 0.9), box(10, (0.5, 0.0), 0.9)], [box(0, (0.5, 0.0)), box(10, (0.5001, 0.0))], 2.0)
    assert (edge[0], edge[1], edge[2], edge[3], edge[8]) == (1, 1, 1, 1, 0), edge
    d55, g55 = [box(0, (0.55, 0.0), 0.9)], [box(0, (0.55, 0.0))]                  # the arguments are used (no hidden constants)
    base = B.velocity_counts(d55, g55, 2.0)
    assert (base[2], base[3], base[8]) == (1, 0, 0)
    assert tuple(B.velocity_counts(d55, g55, 2.0, moving=0.6)[[2, 3, 8]]) == (0, 1, 0)
    assert tuple(B.velocity_counts(d55, g55, 2.0, slow=0.6)[[2, 3, 8]]) == (1, 0, 1)
    # nothing to judge is nan; a detector without velocities has no pair; a zero error of the zero predictor has no skill
    empty = B.velocity_numbers(B.velocity_counts([], [], 2.0))
    assert all(math.isnan(empty[k]) for k in ("ave_moving", "ave_static", "ave0_moving", "ave_skill_moving", "ave_skill_static", "ave_slow_moving")) and empty["n_tp_moving"] == 0
    none = B.velocity_numbers(B.velocity_counts([box(0, None, 0.9)], [box(0, (4.0, 0.0))], 2.0))
    assert none["n_tp_moving"] == 0 and none["n_gt_moving"] == 1 and math.isnan(none["ave_moving"])
    for part in ({"vx": 1.0}, {"vy": 1.0}):                                         # half a velocity is no velocity, on either side
        half_det = B.velocity_counts([dict(box(0, None, 0.9), **part)], [box(0, (4.0, 0.0))], 2.0)
        half_gt = B.velocity_counts([box(0, (1.0, 0.0), 0.9)], [dict(box(0, None), **part)], 2.0)
        assert half_det[2] == 0 and half_det[0] == 1 and not half_gt[:4].any(), (part, half_det, half_gt)
    assert math.isnan(B.velocity_numbers(B.velocity_counts([box(0, (0.0, 0.0), 0.9)], [box(0, (0.0, 0.0))], 2.0))["ave_skill_static"])
    for bad in (np.zeros(8), np.zeros((2, 3, 9))):
        try:
            B.velocity_numbers(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("counts of the wrong shape must be refused")
    # random worlds with ties: counts of single frames add up; the two groups together are the `ave` of detection_numbers (same pairs, same
    # mean); every number against a plain-loop version that uses the frozen former matching and tp_errors
    rng = np.random.default_rng(11)
    frames = of_class(random_frames(rng, 250), "car")
    frames = [([dict(b, vx=b["vx"] * f, vy=b["vy"] * f) for b, f in zip(d, rng.choice([0.15, 1.0], len(d)))],
               [dict(b, vx=b["vx"] * f, vy=b["vy"] * f) for b, f in zip(g, rng.choice([0.15, 1.0], len(g)))]) for d, g in frames]    # half of the boxes nearly stand
    per_frame = np.array([B.velocity_counts(d, g, 2.0) for d, g in frames])
    tot = B.velocity_numbers(per_frame)
    scores, flags, errors, n_gt = [], [], [], 0
    for d, g in frames:
        f, e = B.match_flags(d, g, (2.0,), 2.0)
        scores += [x["score"] for x in d]
        flags.append(f)
        errors.append(e)
        n_gt += len(g)
    ave = B.detection_numbers(np.array(scores), np.concatenate(flags), np.concatenate(errors), n_gt, (2.0,), 2.0, "list")["ave"]
    pooled = (per_frame[:, 4].sum() + per_frame[:, 5].sum()) / (per_frame[:, 2].sum() + per_frame[:, 3].sum())
    assert same(pooled, ave, 1e-9), (pooled, ave)
    assert same(B.velocity_numbers(per_frame[:20])["n_tp_moving"] + B.velocity_numbers(per_frame[20:])["n_tp_moving"], tot["n_tp_moving"])
    sums = {k: 0.0 for k in ("e0", "e1", "z0", "z1", "n0", "n1", "slow", "g0", "g1")}
    for d, g in frames:
        mm, _ = _REF.match([(d, g)], 2.0)
        for dd in g:
            sums["g0" if math.hypot(dd["vx"], dd["vy"]) > 0.5 else "g1"] += 1
        for _, ok, det, gt in mm:
            if ok:
                k = 0 if math.hypot(gt["vx"], gt["vy"]) > 0.5 else 1
                sums[f"e{k}"] += _REF.tp_errors(det, gt)["ave"]
                sums[f"z{k}"] += math.hypot(gt["vx"], gt["vy"])
                sums[f"n{k}"] += 1
                sums["slow"] += float(k == 0 and math.hypot(det["vx"], det["vy"]) < 0.5)
    assert sums["n0"] > 3 and sums["n1"] > 3 and sums["slow"] > 0, "the random world must have moving and static pairs, and slow ones"
    for k, v in (("ave_moving", sums["e0"] / sums["n0"]), ("ave_static", sums["e1"] / sums["n1"]), ("ave0_moving", sums["z0"] / sums["n0"]),
                 ("ave0_static", sums["z1"] / sums["n1"]), ("ave_skill_moving", 1 - sums["e0"] / sums["z0"]), ("ave_slow_moving", sums["slow"] / sums["n0"])):
        assert same(tot[k], v, 1e-9), (k, tot[k], v)
    assert (tot["n_gt_moving"], tot["n_gt_static"]) == (sums["g0"], sums["g1"])


@check("comparison.boxes against the real nuScenes devkit (accumulate + calc_ap): the same average precision on random frames; the two places where the rules differ are pinned")
def _():
    try:
        from nuscenes.eval.common.data_classes import EvalBoxes
        from nuscenes.eval.common.utils import center_distance
        from nuscenes.eval.detection.algo import accumulate, calc_ap
        from nuscenes.eval.detection.data_classes import DetectionBox
    except ImportError:
        print("      (nuscenes-devkit not installed: skipped)")
        return

    def eval_boxes(frames, detections):
        eb = EvalBoxes()
        for f, (dets, gts) in enumerate(frames):
            eb.add_boxes(str(f), [DetectionBox(sample_token=str(f), translation=(b["cx"], b["cy"], 0.0), size=(b["width"], b["length"], 1.5),
                                               rotation=(1.0, 0.0, 0.0, 0.0), velocity=(b["vx"], b["vy"]), detection_name="car",
                                               detection_score=b["score"] if detections else -1.0) for b in (dets if detections else gts)])
        return eb

    def devkit_ap(frames, thr):
        md = accumulate(eval_boxes(frames, False), eval_boxes(frames, True), "car", center_distance, thr)
        return calc_ap(md, 0.1, 0.1)
    rng = np.random.default_rng(6)
    checked = 0
    for _ in range(4):
        frames = of_class(random_frames(rng, 40, tie_scores=False, tie_positions=False), "car")
        for thr in (0.5, 1.0, 2.0, 4.0):
            rows, n_gt = B.match_frames(frames, thr)
            ours = B.ap_nuscenes([r[1] for r in rows], n_gt)
            assert same(ours, devkit_ap(frames, thr), 1e-9), (thr, ours, devkit_ap(frames, thr))
            checked += 1
    assert checked == 16
    # where the rules differ, and only there: a distance of EXACTLY the threshold (ours is a hit, the devkit's `<` is not), and two free boxes
    # at exactly the same distance (ours takes the later one, the devkit the first); the BEV system has always had our rule
    one = [([{"group": "car", "cx": 2.0, "cy": 0.0, "length": 4.0, "width": 2.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0, "score": 0.9}],
            [{"group": "car", "cx": 0.0, "cy": 0.0, "length": 4.0, "width": 2.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0}])]
    assert B.match(one[0][0], one[0][1], 2.0) == [0] and devkit_ap(one, 2.0) == 0.0
    box = {"group": "car", "length": 4.0, "width": 2.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0}
    assert B.match([dict(box, cx=5.0, cy=0.0, score=0.9)], [dict(box, cx=0.0, cy=0.0), dict(box, cx=10.0, cy=0.0)], 6.0) == [1], \
        "ours: the later of two equally near boxes (the devkit would take the first)"


# ================================================================================================ the Metrics through the suite
class _MapSource(Source):
    def scenes(self):
        return ["m-0", "m-1"]

    def frames(self, scene):
        return [FrameRef(scene, t) for t in range(4)]

    def load(self, ref):
        truth = np.zeros((16, 16), bool)
        truth[4 + ref.index:9 + ref.index, 3:9] = True
        return Sample(ref, inputs=truth, reference={"gt": truth, "valid": np.arange(16)[:, None] < 14 + 0 * truth})


class _MapSubject(Subject):
    def predict(self, sample, taps=()):
        return Prediction(outputs=np.roll(sample.inputs, 2, axis=1))            # the prediction is the truth shifted by two cells


class _TextView(View):
    def render(self, sample, pred, ctx):
        return Page("comparison", [Panel("text", "frame", sample.ref.label)], ncols=1)


class _Task(Task):
    def __init__(self, metric):
        self.metric = metric

    def metrics(self):
        return [self.metric]

    def views(self):
        return {"overview": _TextView()}


@check("comparison.metrics.MapMetric: through quant.run the suite's numbers equal the core applied to the same frames; contract checker; the default score works")
def _():
    import tempfile
    tols = (0, 1.5, 2)
    metric = MapMetric(lambda s, p: (p.outputs, s.reference["gt"], s.reference["valid"]), name="map", tolerances=tols)
    src, sub = _MapSource(), _MapSubject()
    with tempfile.TemporaryDirectory() as d:
        suite = EvalSuite(src, sub, _Task(metric), out=d)
        assert all(r["ok"] for r in suite.check()), suite.check()
        st = suite.quant.run(name="a", progress=0)
        tab = suite.quant.table(st)
    totals = {t: np.zeros(4, np.int64) for t in tols}
    for sc in src.scenes():
        for ref in src.frames(sc):
            s = src.load(ref)
            p = sub.predict(s).outputs
            for t in tols:
                totals[t] += M.tolerant_confusion(p, s.reference["gt"], t, s.reference["valid"])
    for t, suffix in ((0, ""), (1.5, "@1.5"), (2, "@2")):                     # three different keys, each the core's number for its tolerance
        for k, v in M.tolerant_scores(totals[t]).items():
            assert same(tab[f"map.{k}{suffix}"], v), (k, suffix, tab)
    assert tab["map.iou@2"] == 1.0 > tab["map.iou@1.5"] > tab["map.iou"], "two cells forgive the two-cell shift, 1.5 cells forgive part of it"
    fr = [f for f in st.frames if f["label"] == "m-1/002"][0]
    s22 = src.load(src.find("m-1/002"))
    p22, g22, v22 = sub.predict(s22).outputs, s22.reference["gt"], s22.reference["valid"]
    assert same(fr["scores"]["map.iou"], M.tolerant_scores(M.tolerant_confusion(p22, g22, 0, v22))["iou"])
    assert fr["support"]["map.iou"] == (p22 & ~g22 & v22).sum() + (g22 & v22).sum() > 0, "support = ground-truth cells + false positives, inside `valid`"
    # a frame with nothing to find but a hallucinated map has something to judge (support > 0, iou 0); an empty frame has nothing to judge
    nothing = Sample(FrameRef("m-0", 0), inputs=None, reference={"gt": np.zeros((16, 16), bool), "valid": np.ones((16, 16), bool)})
    halluc = metric.frame(nothing, Prediction(outputs=np.ones((16, 16), bool)))
    assert metric.support(halluc)["iou"] == 256 and metric.score(halluc)["iou"] == 0.0
    assert metric.support(metric.frame(nothing, Prediction(outputs=np.zeros((16, 16), bool))))["iou"] == 0
    # maps of any rank work at tolerance 0 (a tolerance needs 2-D); extract must hand over two or three things
    cube_metric = MapMetric(lambda s, p: (p.outputs, s.reference), tolerances=(0,))
    cube = np.random.default_rng(1).random((2, 6, 6)) > 0.5
    assert cube_metric.reduce([cube_metric.frame(Sample(FrameRef("m-0", 0), None, cube), Prediction(outputs=cube))])["iou"] == 1.0
    for bad in (MapMetric(lambda s, p: (p.outputs, s.reference), tolerances=(1,)),
                MapMetric(lambda s, p: (p.outputs, s.reference, np.ones(s.reference.shape, bool), None))):                    # four values
        try:
            bad.frame(Sample(FrameRef("m-0", 0), None, cube), Prediction(outputs=cube))
        except ValueError:
            pass
        else:
            raise AssertionError("a 3-D map with a tolerance, and an extract of four values, must be refused")


class _BoxSource(Source):
    def scenes(self):
        return ["b-0", "b-1", "b-2"]

    def frames(self, scene):
        return [FrameRef(scene, t) for t in range(5)]

    def load(self, ref):
        """ground-truth boxes and detections that mostly sit near them (shifted by up to 2 cells), plus a few false positives"""
        rng = np.random.default_rng(100 * int(ref.scene[-1]) + ref.index)
        gts, fakes = random_frames(rng, 1)[0]
        dets = [dict(g, cx=g["cx"] + int(rng.integers(-2, 3)), cy=g["cy"] + int(rng.integers(-2, 3)), score=int(rng.integers(1, 6)) / 5.0)
                for g in gts if rng.random() < 0.8]
        return Sample(ref, inputs=dets + [dict(f, score=0.2) for f in fakes[:2]], reference=gts)


class _BoxSubject(Subject):
    def predict(self, sample, taps=()):
        return Prediction(outputs=sample.inputs)


@check("comparison.metrics.BoxMetric: through quant.run the suite's per-class and mean numbers equal one global matching (the former BEV code); contract checker; frames add up")
def _():
    import tempfile
    metric = BoxMetric(lambda s, p: (p.outputs, s.reference), classes=("car", "ped", "bike"), name="det")       # "bike": no box at all
    src, sub = _BoxSource(), _BoxSubject()
    with tempfile.TemporaryDirectory() as d:
        suite = EvalSuite(src, sub, _Task(metric), out=d)
        assert all(r["ok"] for r in suite.check()), suite.check()
        st = suite.quant.run(name="a", progress=0)
        tab = suite.quant.table(st)
        part = suite.quant.table(st, labels=[f["label"] for f in st.frames if f["scene"] == "b-1"])
    frames = [(src.load(r).inputs, src.load(r).reference) for sc in src.scenes() for r in src.frames(sc)]
    for cls in ("car", "ped"):
        fr = of_class(frames, cls)
        for t in (0.5, 1.0, 2.0, 4.0):
            mm, n_gt = _REF.match(fr, t)
            fl = [x[1] for x in sorted(mm, key=lambda x: (-x[0], x[1]))]            # equal scores: false positives first (the metric's rule)
            assert same(tab[f"det.{cls}/ap@%g" % t], _REF.ap_nuscenes(fl, n_gt)) and same(tab[f"det.{cls}/ap_all@%g" % t], _REF.ap_all_points(fl, n_gt)), (cls, t)
            if t == 2.0:
                errs = [_REF.tp_errors(d, g) for _, ok, d, g in mm if ok]
                assert tab[f"det.{cls}/n_tp"] == len(errs) and len(errs) > 0
                assert same(tab[f"det.{cls}/ate"], float(np.mean([e["ate"] for e in errs])))
    aps = [tab[f"det.{c}/ap@%g" % t] for c in ("car", "ped") for t in (0.5, 1.0, 2.0, 4.0) if tab[f"det.{c}/n_gt"] and tab[f"det.{c}/ap@%g" % t] == tab[f"det.{c}/ap@%g" % t]]
    assert same(tab["det.mAP"], float(np.mean(aps)))
    assert tab["det.bike/n_gt"] == 0 and tab["det.bike/ap@2"] != tab["det.bike/ap@2"], "a class without ground truth is nan ..."
    assert all(tab["det.m" + k] == tab["det.m" + k] for k in ("AP", "AP_all", "ATE", "ASE", "AOE", "AVE")), "... and stays out of every mean (no mean is nan)"
    try:
        BoxMetric(lambda s, p: (p.outputs, s.reference), classes=("car",), thresholds=(1.0, 2.0), tp_threshold=3.0)
    except ValueError:
        pass
    else:
        raise AssertionError("a tp_threshold that is no threshold must be refused")
    fr = st.frames[3]
    n_boxes = len(src.load(src.find(fr["label"])).reference)
    assert fr["support"]["det.mAP"] == n_boxes, "support = the ground-truth boxes of the frame"
    one = [(src.load(r).inputs, src.load(r).reference) for r in src.frames("b-1")]
    assert part["det.car/n_gt"] == sum(1 for _, gs in one for g in gs if g["group"] == "car"), "the numbers of a subset of scenes come from its own counts"
    # the aggregated numbers against the oracle: plain means over the true positives at 2 m per class, then over the classes
    per_cls = {k: {} for k in B.TP_ERROR_KEYS}
    for cls in ("car", "ped"):
        mm, _ = _REF.match(of_class(frames, cls), 2.0)
        errs = [_REF.tp_errors(d, g) for _, ok, d, g in mm if ok]
        for k in per_cls:
            v = [e[k] for e in errs if e[k] == e[k]]
            per_cls[k][cls] = float(np.mean(v)) if v else float("nan")
            assert same(tab[f"det.{cls}/{k}"], per_cls[k][cls]), (cls, k)
    for name, k in (("mATE", "ate"), ("mASE", "ase"), ("mAOE", "aoe"), ("mAVE", "ave")):
        assert same(tab["det." + name], float(np.mean([x for x in per_cls[k].values() if x == x]))), name
    aps_all = [tab[f"det.{c}/ap_all@%g" % t] for c in ("car", "ped") for t in (0.5, 1.0, 2.0, 4.0) if tab[f"det.{c}/n_gt"]]
    assert same(tab["det.mAP_all"], float(np.mean(aps_all))) and tab["det.mAP_all"] != tab["det.mAP"]
    assert metric.headline == ("mAP", "mATE") and metric.direction("mATE") == "low" and metric.direction("mAVE") == "low" and metric.direction("mAP") == "high"
    # a class with ground truth that is not found at all counts as the worst error (1.0): missing a class must not improve mATE
    def boxes(*specs):
        return [{"group": g, "cx": x, "cy": 0.0, "length": 4.0, "width": 2.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0, **({"score": s} if s else {})} for g, x, s in specs]
    two = BoxMetric(lambda s, p: (p.outputs, s.reference), classes=("car", "truck"), name="det")
    truth = boxes(("car", 10.0, None), ("truck", 30.0, None))
    found = two.reduce([two.frame(Sample(FrameRef("x-0", 0), None, truth), Prediction(outputs=boxes(("car", 10.0, 0.9), ("truck", 31.0, 0.8))))])
    missed = two.reduce([two.frame(Sample(FrameRef("x-0", 0), None, truth), Prediction(outputs=boxes(("car", 10.0, 0.9))))])
    assert same(found["mATE"], 0.5) and same(missed["mATE"], 0.5) and missed["truck/ate"] != missed["truck/ate"], (found["mATE"], missed["mATE"])
    assert same(found["mAP"], 7 / 8) and same(missed["mAP"], 4 / 8), "(the truck is found within 1 m: AP 0 at 0.5 m, 1 above; missed: 0 everywhere)"
    # the tie rule is passed on: two equal scores, a false positive and a true positive, ground truth of one box
    tied = [boxes(("car", 10.0, 0.5), ("car", 14.0, 0.5))]               # list order: the first (near) detection is the true positive, the second a false positive
    truth1 = boxes(("car", 10.0, None))
    for ties, expect_pessimistic in (("pessimistic", True), ("list", False)):
        mt = BoxMetric(lambda s, p: (p.outputs, s.reference), classes=("car",), name="det", ties=ties)
        ap = mt.reduce([mt.frame(Sample(FrameRef("x-0", 0), None, truth1), Prediction(outputs=tied[0]))])["car/ap_all@2"]
        assert same(ap, 0.5 if expect_pessimistic else 1.0), (ties, ap)
