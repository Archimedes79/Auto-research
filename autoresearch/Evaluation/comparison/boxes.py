"""Compare two lists of boxes: detections against ground truth, in the way of the nuScenes detection benchmark, in bird's-eye view.

A box is a plain dict (the caller turns whatever its dataset delivers into this; nothing else is read):
    "cx", "cy"                          centre (m)
    "length", "width", "yaw"            a rotated rectangle (yaw in rad, length along the heading), or
    "radius"                            a circle (pedestrians): matched by its centre like any other box
    "vx", "vy"                          velocity (m/s), optional (the velocity error is then nan)
    "score"                             detections only: the confidence the detector gives
    "group"                             the class, used by the callers that split by class (the functions here see ONE class)

The comparison of one frame in four steps, each a function below:
    match         every detection, best score first, takes the nearest still unmatched ground-truth box within a distance threshold
    match_flags   the same at several thresholds: one 0 / 1 flag per detection and threshold, and the errors of the true positives
    tp_errors     how far a matched pair is off: ATE, ASE, AOE, AVE, BIoU
    ap_*          precision and recall over a score-ordered list of flags, from the rows of ANY set of frames: average precision
    velocity_*    the velocity error of the matched pairs split into moving and static ground truth, against the predictor "0 m/s"
The rows of one frame (score, flags, errors) and the number of ground-truth boxes are additive over frames: put the rows of several
frames together, sort by score and the average precision of all of them follows (`detection_numbers`), without matching again.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

NAN = float("nan")
TP_ERROR_KEYS = ("ate", "ase", "aoe", "ave", "biou")          # translation, scale, orientation, velocity, oriented BEV IoU


# ------------------------------------------------------------------------------------------------ matching
def center_distance(a: dict, b: dict) -> float:
    return math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])


def match(dets: Sequence[dict], gts: Sequence[dict], thr: float) -> List[int]:
    """Greedy one-to-one matching of the detections of ONE frame and ONE class to its ground-truth boxes.
    Detections are taken best score first; each takes the nearest ground-truth box that no better detection has taken, if it lies within
    `thr` metres (centre distance, a distance of exactly thr still counts). If two free boxes are exactly equally near, the later one in
    the list is taken. A box with a nan centre is never within reach (a false positive).
    EQUAL SCORES are served in list order, and the list order can decide which of two detections becomes the true positive: a detector
    that gives many detections the same score (1.0, a quantised confidence) has an average precision that depends on the order of its
    output. The nuScenes devkit has the same dependence with the opposite order. -> for every detection, in the order of `dets`: the
    index of its ground-truth box, or -1 (false positive)."""
    scores = [d["score"] for d in dets]
    if any(s != s for s in scores):
        raise ValueError("a detection without a score (nan) cannot be ranked")
    taken, out = set(), [-1] * len(dets)
    for i in sorted(range(len(dets)), key=lambda i: -scores[i]):          # sorted() is stable: equal scores keep list order
        best, best_dist = -1, thr
        for j, g in enumerate(gts):
            if j in taken:
                continue
            dist = center_distance(dets[i], g)
            if dist <= best_dist:
                best, best_dist = j, dist
        if best >= 0:
            taken.add(best)
            out[i] = best
    return out


def match_frames(frames: Sequence[Tuple[Sequence[dict], Sequence[dict]]], thr: float):
    """`match` over many frames of one class: frames = [(dets, gts), ...]. -> (rows, n_gt) with rows = [(score, is_tp, det, gt or None)]
    over all frames in score order (equal scores: frame order, then list order) and n_gt the ground-truth boxes of all frames. Frames do
    not compete with each other, so this equals one global score-ordered matching."""
    frames = list(frames)                                     # read twice below (a generator would leave n_gt = 0)
    rows = []
    for dets, gts in frames:
        for d, j in zip(dets, match(dets, gts, thr)):
            rows.append((d["score"], j >= 0, d, gts[j] if j >= 0 else None))
    rows.sort(key=lambda r: -r[0])
    return rows, sum(len(gts) for _, gts in frames)


def match_flags(dets: Sequence[dict], gts: Sequence[dict], thresholds: Sequence[float], tp_threshold: float = None):
    """The rows of ONE frame and ONE class. -> (flags, errors): flags (n_det, T) float, 1 where the detection is a true positive at that
    threshold; errors (n_det, 5) float in the order TP_ERROR_KEYS, the `tp_errors` of the match at `tp_threshold` (nan where the
    detection is no true positive there, or tp_threshold is None). Both in the order of `dets`."""
    wanted = set(thresholds) | ({tp_threshold} if tp_threshold is not None else set())
    matched = {t: match(dets, gts, t) for t in wanted}
    flags = np.zeros((len(dets), len(thresholds)))
    for k, t in enumerate(thresholds):
        flags[:, k] = [float(j >= 0) for j in matched[t]]
    errors = np.full((len(dets), len(TP_ERROR_KEYS)), NAN)
    if tp_threshold is not None:
        for i, j in enumerate(matched[tp_threshold]):
            if j >= 0:
                e = tp_errors(dets[i], gts[j])
                errors[i] = [e[k] for k in TP_ERROR_KEYS]
    return flags, errors


# ------------------------------------------------------------------------------------------------ errors of a matched pair
def _as_circle(b: dict) -> dict:
    return b if "radius" in b else dict(b, radius=max(b["length"], b["width"]) / 2)


def _finite_geometry(b: dict) -> bool:
    keys = ("cx", "cy", "radius") if "radius" in b else ("cx", "cy", "length", "width", "yaw")
    return all(math.isfinite(b[k]) for k in keys)


def _shape(b: dict):
    from shapely.geometry import Point, Polygon                     # only the oriented IoU needs it
    if "radius" in b:
        return Point(b["cx"], b["cy"]).buffer(b["radius"], 32)
    c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
    hl, hw = b["length"] / 2, b["width"] / 2
    return Polygon([(b["cx"] + c * a - s * w, b["cy"] + s * a + c * w) for a, w in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))])


def tp_errors(d: dict, g: dict) -> Dict[str, float]:
    """The errors of one matched pair (detection d, ground truth g).
    ate   translation: distance of the centres (m)
    ase   scale: 1 - IoU of the two boxes after aligning centre and heading, 1 - min(l) min(w) / (l w + l' w' - min(l) min(w)); for a
          circle against anything: 1 - (r_min / r_max)^2 (a rectangle counts as the circle of radius max(l, w) / 2)
    aoe   orientation: the smallest yaw difference, in [0, pi] rad (period 2 pi); nan for circles
    ave   velocity: length of the velocity difference (m/s); nan if a velocity is missing
    biou  the oriented BEV IoU of the two boxes as they are, without aligning anything (needs shapely)
    A pair with a circle is compared as two circles in every measure (the rectangle becomes the circle of radius max(l, w) / 2), biou
    included. A box of size 0 gives ase nan (a ratio of 0 / 0) and biou 0; a nan in a box's geometry gives nan in the measures that
    use it (biou nan), never an exception."""
    if "radius" in d or "radius" in g:
        d, g = _as_circle(d), _as_circle(g)
        r_min, r_max = sorted((d["radius"], g["radius"]))
        ase, aoe = (1.0 - (r_min / r_max) ** 2 if r_max > 0 else NAN), NAN
    else:
        inter = min(d["length"], g["length"]) * min(d["width"], g["width"])
        union = d["length"] * d["width"] + g["length"] * g["width"] - inter
        ase = 1.0 - inter / union if union > 0 else NAN
        aoe = abs(math.remainder(d["yaw"] - g["yaw"], 2 * math.pi))
    biou = NAN                                                      # a nan in the geometry gives nan, not a crash in shapely
    if _finite_geometry(d) and _finite_geometry(g):
        shape_d, shape_g = _shape(d), _shape(g)
        union_area = shape_d.union(shape_g).area
        biou = shape_d.intersection(shape_g).area / union_area if union_area else 0.0
    return {"ate": center_distance(d, g), "ase": ase, "aoe": aoe,
            "ave": math.hypot(d.get("vx", NAN) - g.get("vx", NAN), d.get("vy", NAN) - g.get("vy", NAN)), "biou": biou}


# ------------------------------------------------------------------------------------------------ average precision
def _precision_recall(flags: Sequence[float], n_gt: int):
    f = np.asarray(flags, float)
    tp, fp = np.cumsum(f), np.cumsum(1 - f)
    if tp[-1] > n_gt:
        raise ValueError(f"{int(tp[-1])} true positives for {n_gt} ground-truth boxes: n_gt must count every box the flags were matched against")
    return tp / (tp + fp), tp / n_gt


def ap_nuscenes(tp_flags: Sequence[float], n_gt: int, min_recall: float = 0.1, min_precision: float = 0.1) -> float:
    """The average precision as the nuScenes devkit computes it. tp_flags: 1 / 0 per detection in DESCENDING score order, n_gt: the
    ground-truth boxes. Precision is interpolated at the 101 recall values 0, 0.01 ... 1 (0 beyond the last recall reached; no
    monotone envelope), the recall values up to min_recall are dropped, min_precision is subtracted (below it counts as 0) and the
    mean is divided by 1 - min_precision. nan if there is no ground truth, 0 if there is ground truth but no detection."""
    if n_gt == 0:
        return NAN
    if len(tp_flags) == 0:
        return 0.0
    prec, rec = _precision_recall(tp_flags, n_gt)
    p = np.interp(np.linspace(0, 1, 101), rec, prec, right=0)
    p = p[int(round(100 * min_recall)) + 1:] - min_precision
    p[p < 0] = 0
    return float(np.mean(p)) / (1.0 - min_precision)


def ap_all_points(tp_flags: Sequence[float], n_gt: int) -> float:
    """The all-point average precision: the area under the precision-recall curve with the monotone (envelope) precision. Same
    arguments and nan / 0 rules as `ap_nuscenes`."""
    if n_gt == 0:
        return NAN
    if len(tp_flags) == 0:
        return 0.0
    prec, rec = _precision_recall(tp_flags, n_gt)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.sum(prec * np.diff(np.r_[0.0, rec])))


def detection_numbers(scores: np.ndarray, flags: np.ndarray, errors: np.ndarray, n_gt: int, thresholds: Sequence[float],
                      tp_threshold: float = None, ties: str = "pessimistic") -> Dict[str, float]:
    """The numbers of ONE class from the rows of any set of frames put together: scores (n,), flags (n, T), errors (n, 5), n_gt.
    -> "ap@t" and "ap_all@t" per threshold (`ap_nuscenes`, `ap_all_points`); if tp_threshold is one of the thresholds also "ate", "ase",
    "aoe", "ave", "biou" (plain mean over the true positives at tp_threshold, nan values skipped, nan if none are left) and "n_tp";
    always "n_gt" and "n_det".
    Rows with EQUAL scores have no natural order, and the average precision depends on it. ties="pessimistic" (default) puts the false
    positives of a score before its true positives, at every threshold: the numbers are then a function of the set of ROWS, whatever the
    order of the frames. (Which detection of a frame is a true positive was already decided by `match`, which serves equal scores in list
    order: see there.) ties="list" keeps the order the rows are given in (what the BEV system's DetMeter has always done: frame order,
    then detection order)."""
    scores = np.asarray(scores, float)
    flags, errors = np.asarray(flags, float), np.asarray(errors, float)
    if len(scores) == 0:                                              # no detection at all: empty arrays of any second dimension are fine
        flags, errors = flags.reshape(0, len(thresholds)), errors.reshape(0, len(TP_ERROR_KEYS))
    if flags.shape != (len(scores), len(thresholds)) or errors.shape != (len(scores), len(TP_ERROR_KEYS)):
        raise ValueError(f"for {len(scores)} detections and {len(thresholds)} thresholds flags must be {(len(scores), len(thresholds))} and "
                         f"errors {(len(scores), len(TP_ERROR_KEYS))}, got {flags.shape} and {errors.shape}")
    if ties not in ("pessimistic", "list"):
        raise ValueError("ties must be 'pessimistic' or 'list'")
    row: Dict[str, float] = {}
    for k, t in enumerate(thresholds):
        order = np.argsort(-scores, kind="stable") if ties == "list" else np.lexsort((flags[:, k], -scores))   # lexsort: score first, then flag 0 before 1
        ranked = flags[order, k]
        row["ap@%g" % t], row["ap_all@%g" % t] = ap_nuscenes(ranked, n_gt), ap_all_points(ranked, n_gt)
        if tp_threshold is not None and t == tp_threshold:
            tp = ranked > 0
            for j, name in enumerate(TP_ERROR_KEYS):
                v = errors[order, j][tp]                                  # summed in rank order, as the BEV numbers always were
                v = v[v == v]
                row[name] = float(v.mean()) if len(v) else NAN
            row["n_tp"] = int(tp.sum())
    row["n_gt"], row["n_det"] = int(n_gt), len(scores)
    return row


# ------------------------------------------------------------------------------------------------ velocity: moving against static
VELOCITY_COUNT_KEYS = ("n_gt_moving", "n_gt_static", "n_moving", "n_static", "ave_moving", "ave_static", "zero_moving", "zero_static",
                       "n_slow_moving")


def velocity_counts(dets: Sequence[dict], gts: Sequence[dict], thr: float, moving: float = 0.5, slow: float = 0.5) -> np.ndarray:
    """The additive velocity counts of ONE frame and ONE class (order VELOCITY_COUNT_KEYS), the plain mean `ave` of `tp_errors` split by
    what the object does. A ground-truth box is MOVING if its speed is above `moving` m/s, else STATIC (a box without a velocity is
    neither). The true positives are the matches `match` makes at `thr` (the same as the ones whose `ave` goes into `detection_numbers`),
    only the pairs with both velocities finite count. Per group: n_gt_* ground-truth boxes (matched or not), n_* matched pairs, ave_* the
    sum of the velocity errors (m/s), zero_* the sum of the speeds of the ground-truth boxes of the same pairs (= the velocity error of a
    predictor that always says 0), n_slow_moving the moving pairs whose detection is slower than `slow` m/s (a moving object
    predicted to stand). `velocity_numbers` turns the counts of any set of frames into numbers."""
    out = np.zeros(len(VELOCITY_COUNT_KEYS))
    for g in gts:
        s = math.hypot(g.get("vx", NAN), g.get("vy", NAN))
        if math.isfinite(s):
            out[0 if s > moving else 1] += 1
    for d, j in zip(dets, match(dets, gts, thr)):
        if j < 0:
            continue
        g = gts[j]
        e = math.hypot(d.get("vx", NAN) - g.get("vx", NAN), d.get("vy", NAN) - g.get("vy", NAN))
        s = math.hypot(g.get("vx", NAN), g.get("vy", NAN))
        if not (math.isfinite(e) and math.isfinite(s)):
            continue
        k = 0 if s > moving else 1
        out[2 + k] += 1
        out[4 + k] += e
        out[6 + k] += s
        if k == 0 and math.hypot(d["vx"], d["vy"]) < slow:
            out[8] += 1
    return out


def velocity_numbers(counts: np.ndarray) -> Dict[str, float]:
    """The numbers of the velocity counts of any set of frames (the counts of one frame, or an (n_frames, K) array that is summed):
    per group g in moving, static: "ave_g" the mean velocity error (m/s) of the matched pairs, "ave0_g" the same for the predictor that
    always says 0 (on the same pairs), "ave_skill_g" = 1 - ave_g / ave0_g (above 0: better than saying 0; 1: perfect), "n_tp_g" the
    pairs, "n_gt_g" the ground-truth boxes; and "ave_slow_moving", the share of the moving pairs predicted slower than `slow`. nan where
    there is nothing to judge (no pair; a zero predictor with no error)."""
    c = np.asarray(counts, float)
    if c.shape[-1:] != (len(VELOCITY_COUNT_KEYS),) or c.ndim > 2:
        raise ValueError(f"velocity counts must have {len(VELOCITY_COUNT_KEYS)} columns, got shape {c.shape}")
    c = c.sum(0) if c.ndim == 2 else c
    row: Dict[str, float] = {}
    for k, g in enumerate(("moving", "static")):
        n, e, z = c[2 + k], c[4 + k], c[6 + k]
        row[f"ave_{g}"] = float(e / n) if n else NAN
        row[f"ave0_{g}"] = float(z / n) if n else NAN
        row[f"ave_skill_{g}"] = float(1.0 - e / z) if z > 0 else NAN
        row[f"n_tp_{g}"], row[f"n_gt_{g}"] = int(n), int(c[k])
    row["ave_slow_moving"] = float(c[8] / c[2]) if c[2] else NAN
    return row
