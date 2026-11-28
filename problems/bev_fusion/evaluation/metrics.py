"""Evaluation metrics of the BEV system (plan.md 8.3 "Metrics", 8.4 "Metrics reported next to the losses", 10).

Not used for training. Everything accumulates counts over frames (dataset-level numbers, not means of per-frame
numbers) and is keyed by the prediction time offset dt (seconds; dt = 0 is the current frame).
- segmentation (SegMeter): IoU per layer per range bin (0-20, 20-40, 40-60 m by the cell's x distance) at tolerances
  0 / 0.5 / 1.0 m, only on defined cells (mask), separately for known / unknown cells when a splat mass map is given
  (unknown = mass < 0.5, plan 8.4). Tolerance t (boundary-tolerant IoU via dilation): a predicted cell is a true
  positive if a GT cell centre lies within t of its centre, a GT cell is found if a predicted cell lies within t;
  IoU_t = TP / (TP + FP + FN) with TP the mean of the two true-positive counts (t = 0 is the plain IoU).
- lines (SegMeter.add_lines): the line is decoded from the unsigned distance field (metres) as its minimum across the
  line ("zero crossing" of the field's slope): cells that are a local minimum along x or y and <= res / 2 (the GT
  mask rule of net/prep/gt_bev.py); precision / recall / F1 of those cells at the same tolerances, plus the mean
  absolute field error on cells with GT distance < 1 m (the sub-cell accuracy the loss trains, plan 8.4).
- directions (SegMeter.add_directions, the 1 m centre-line head): the mean absolute signed angle (rad, 0 .. pi) between the predicted (cos, sin) and the GT
  direction over the GT cells of the layer, `<layer>:dir` / angle_mae.
- detection (DetMeter), nuScenes style in the front field of view up to 60 m: greedy matching by BEV centre distance
  in score order over all frames (the logic of tests/test_object_head.py evaluate()), AP per class at 0.5 / 1 / 2 /
  4 m both as nuScenes computes it (precision interpolated at 101 recall points, recall <= 0.1 and precision <= 0.1
  clipped, from memory of the devkit's detection/algo.py: no precision envelope) and as the plain all-point AP (with
  the monotone precision envelope); true-positive errors of the matches at 2 m, plain means over the matches (the
  devkit additionally averages the cumulative error curve over recall; not done here): ATE (centre, m), ASE (1 - IoU
  after aligning centre and yaw: BEV, there is no height), AOE (smallest yaw difference, rad, period 2 pi; circles
  excluded), AVE (velocity, m/s), and BIoU (oriented BEV IoU without alignment, shapely); per class, per range bin
  (GT and detections both filtered to the bin by their centre x) and per dt. Circles (pedestrians): matched by centre,
  ASE = 1 - (r_min / r_max)^2; a GT box of a circle class is turned into a circle of radius max(l, w) / 2.
  The velocity error of the same matches split into moving and static GT boxes (MOVING_SPEED), next to the error of always saying 0 m/s:
  DetMeter.frame_velocity / reduce_velocity (scope: all boxes of the class, no range bins), because most vehicles stand and the plain AVE hides it.
- latency: ms_per_frame (warm-up, CUDA synchronize, median over N, batch 1); only valid on an idle GPU.
- report: one flat dict "task/dt+0.00/name/.../metric" (flatten / unflatten round trip) and a printable table.
Box dicts as net/model/object_head.py decode(): group, score, cx, cy (m, ego frame), length, width, yaw or radius, vx,
vy; gt_boxes_from_meta() converts net/prep/gt_bev.py meta["boxes"].
The comparison itself (cell counts and tolerant counts, matching, the errors of a matched pair, average precision) is not here but in
autoresearch/Evaluation/comparison/ (maps.py, boxes.py; read its README.md): this file adds what is specific to the BEV system (range bins,
field of view, circles for pedestrians, dt keys, the known / unknown split, the lines) and calls that core. `match`, `ap_nuscenes`,
`ap_all_points`, `tp_errors`, `dilate` and TP_ERR_KEYS stay importable from here under their old names (`match` is the core's
`match_frames`). DetMeter reduces with ties="list": equal scores keep frame order, then detection order, as it always did.
Two differences from the code before the core: a detection whose score is nan (a diverged model) raises a ValueError at its frame (it used to
rank last and give garbage numbers); and importing this file imports the Evaluation package (numpy only), so a broken Evaluation breaks this file.
The mean errors (mATE / mASE / mAOE / mAVE) count a class with ground truth and no true positive as WORST_ERROR (1.0, the devkit's rule and the
core's BoxMetric); until 2026-11-02 such a class was left out of the mean (a missed class improved it). AOE of the circle classes stays undefined
(left out, as the devkit leaves out the orientation of a traffic cone); cards made before this change have mean errors that are not comparable.
Tests: tests/test_metrics.py
"""
import math
import re
import time

import numpy as np

from Evaluation.comparison import boxes as _boxes, maps as _maps
from Evaluation.comparison.metrics import WORST_ERROR

RANGE_BINS = ((0.0, 20.0), (20.0, 40.0), (40.0, 60.0))
TOLERANCES = (0.0, 0.5, 1.0)
DIST_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
TP_THRESHOLD = 2.0                                       # nuScenes: TP errors of the matches at 2 m
MOVING_SPEED = 0.5                                       # m/s: a GT box faster than this is moving, a moving box predicted slower than this "stands"
FRONT_HALF_FOV = math.atan(800.0 / 1266.4)               # CAM_FRONT, 1600 px wide, f about 1266 px (memory): 32.3 deg
CIRCLE_GROUPS = ("pedestrian",)


def _np(a):
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a)


def bin_label(b):
    return "%g-%g" % RANGE_BINS[b]


def range_bin(x, bins=RANGE_BINS):
    """index of the range bin of a distance x (m), [lo, hi); None outside every bin."""
    for i, (lo, hi) in enumerate(bins):
        if lo <= x < hi:
            return i
    return None


def cell_bins(shape, grid, bins=RANGE_BINS):
    """(X, Y) int array: range bin of each cell by its centre x (x_min + (i + 0.5) res); -1 outside every bin."""
    x = grid.x_min + (np.arange(shape[0]) + 0.5) * grid.res
    b = np.full(shape[0], -1)
    for i, (lo, hi) in enumerate(bins):
        b[(x >= lo) & (x < hi)] = i
    return np.repeat(b[:, None], shape[1], 1)


def dilate(m, radius_cells):
    """boolean dilation of m (X, Y) by a disc: a cell is set if a set cell's centre lies within radius_cells (comparison.maps.dilate:
    2-D maps only; any array is read as boolean here)."""
    return _maps.dilate(np.asarray(m, bool), radius_cells)


def angle_error(pred_dir, gt_dir):
    """(X, Y) absolute signed angle (rad, 0 .. pi) between two direction fields (2, X, Y) of (cos, sin); the prediction need not be a unit vector. A reversed
    line
    is pi (not 0: centre-lines are directed); a zero prediction vector has no direction: pi."""
    p, g = _np(pred_dir).astype(np.float64), _np(gt_dir).astype(np.float64)
    return np.where(np.hypot(p[0], p[1]) > 1e-6, np.abs(np.arctan2(g[0] * p[1] - g[1] * p[0], g[0] * p[0] + g[1] * p[1])), np.pi)


def line_cells(dist, res):
    """cells of the line decoded from an unsigned distance field (metres): local minimum along x or along y (<= both
    neighbours) and value <= res / 2."""
    d = _np(dist).astype(np.float64)
    p = np.pad(d, 1, constant_values=np.inf)
    c = p[1:-1, 1:-1]
    min_x = (c <= p[:-2, 1:-1]) & (c <= p[2:, 1:-1])
    min_y = (c <= p[1:-1, :-2]) & (c <= p[1:-1, 2:])
    return (min_x | min_y) & (d <= res / 2 + 1e-9)


def dt_key(dt):
    return "dt%+.2f" % dt


_DT = re.compile(r"^dt[+-]\d+\.\d+$")


# ------------------------------------------------------------------------------------------------ segmentation
class SegMeter:
    """Accumulates tolerant IoU counts (areas, objects coverage, line masks) and zero-crossing F1 counts (lines).
    grid: anything with x_min and res (training/lift_pd.Grid). Inputs are (X, Y) arrays or tensors per layer."""

    def __init__(self, grid, tolerances=TOLERANCES, bins=RANGE_BINS, known_threshold=0.5):
        self.grid, self.tol, self.bins, self.known_threshold = grid, tolerances, bins, known_threshold
        self.c = {}                                       # (dt, name, subset, bin) -> counts
        self._bins = {}                                   # map shape -> range bin of every cell

    def _subsets(self, shape, mask, mass):
        m = np.ones(shape, bool) if mask is None else _np(mask).astype(bool)
        out = {"all": m}
        if mass is not None:
            known = _np(mass) >= self.known_threshold
            out["known"], out["unknown"] = m & known, m & ~known
        return out

    def _count(self, key_name, P, G, subsets, dt, extra=None):
        if P.shape not in self._bins:
            self._bins[P.shape] = cell_bins(P.shape, self.grid, self.bins)       # the same for every layer and frame of a grid
        bins = self._bins[P.shape]
        dil = [(dilate(G, t / self.grid.res), dilate(P, t / self.grid.res)) for t in self.tol]
        keys, cms = [], []                                                      # every selection of cells: (subset, range bin)
        for sname, sm in subsets.items():
            for b in list(range(len(self.bins))) + ["all"]:
                cms.append(sm & ((bins >= 0) if b == "all" else (bins == b)))
                keys.append((round(float(dt), 3), key_name, sname, "all" if b == "all" else bin_label(b)))
        # (tolerances, selections, 4): [p_hit, p_miss, g_hit, g_miss], one matrix product
        cnt = _maps.tolerant_counts_many(P, G, dil, cms)
        for j, (key, cm) in enumerate(zip(keys, cms)):
            acc = self.c.setdefault(key, {"cnt": np.zeros((len(self.tol), 4)), "n_gt": 0, "err": 0.0, "n_err": 0})
            for i in range(len(self.tol)):
                acc["cnt"][i] += cnt[i, j]
            acc["n_gt"] += int((G & cm).sum())
            if extra is not None:
                err, valid = extra
                acc["err"] += float(err[valid & cm].sum())
                acc["n_err"] += int((valid & cm).sum())

    def add(self, pred, gt, mask=None, mass=None, dt=0.0, threshold=0.5):
        """pred, gt: dict layer -> (X, Y) probability / coverage (cell set if > threshold); mask: defined cells (True =
        evaluated), e.g. gt['undefined'] == 0; mass: splat mass at the target time (known = mass >= 0.5)."""
        for name in pred:
            P, G = _np(pred[name]) > threshold, _np(gt[name]) > threshold
            self._count(name, P, G, self._subsets(P.shape, mask, mass), dt)

    def add_lines(self, pred_dist, gt_dist, mask=None, mass=None, dt=0.0):
        """pred_dist, gt_dist: dict line layer -> (X, Y) unsigned distance field (m, truncated, e.g. 2 m); counted
        under '<layer>:zc'."""
        for name in pred_dist:
            dp, dg = _np(pred_dist[name]).astype(np.float64), _np(gt_dist[name]).astype(np.float64)
            P, G = line_cells(dp, self.grid.res), line_cells(dg, self.grid.res)
            self._count(name + ":zc", P, G, self._subsets(P.shape, mask, mass), dt, (np.abs(dp - dg), dg < 1.0))

    def add_directions(self, pred_dir, gt_dir, gt_mask, mask=None, mass=None, dt=0.0):
        """pred_dir, gt_dir: dict layer -> (2, X, Y) direction (cos, sin; the prediction raw, not normalised); gt_mask: dict layer -> (X, Y) the layer's GT
        cells.
        Counted under '<layer>:dir': the mean absolute SIGNED angle between the two vectors (rad, 0 .. pi: a reversed line is pi, not 0) over the GT cells of
        the layer
        (the prediction is read at every one of them, whether or not it says 'line' there). A zero prediction vector has no direction: pi."""
        for name in pred_dir:
            G = (_np(gt_mask[name]) > 0.5) & (np.hypot(*_np(gt_dir[name])) > 0.5)
            self._count(name + ":dir", G, G, self._subsets(G.shape, mask, mass), dt, (angle_error(pred_dir[name], gt_dir[name]), G))

    def result(self):
        """{dt: {name: {subset: {bin: {metric: value}}}}}; areas: iou@t, lines (':zc'): f1@t, p@t, r@t, field_mae, directions (':dir'): angle_mae."""
        res = {}
        for (dt, name, sname, b), acc in sorted(self.c.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
            row = {}
            # a direction layer has no overlap numbers, its one number is the angle error
            for i, t in ([] if name.endswith(":dir") else enumerate(self.tol)):
                tp_p, fp, tp_g, fn = acc["cnt"][i]
                tp = (tp_p + tp_g) / 2
                if name.endswith(":zc"):
                    p = tp_p / (tp_p + fp) if tp_p + fp else float("nan")
                    r = tp_g / (tp_g + fn) if tp_g + fn else float("nan")
                    row["f1@%g" % t] = 2 * p * r / (p + r) if p + r > 0 else (0.0 if p == p and r == r else float("nan"))
                    row["p@%g" % t], row["r@%g" % t] = p, r
                else:
                    row["iou@%g" % t] = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
            if name.endswith(":zc"):
                row["field_mae"] = acc["err"] / acc["n_err"] if acc["n_err"] else float("nan")
            if name.endswith(":dir"):
                row["angle_mae"] = acc["err"] / acc["n_err"] if acc["n_err"] else float("nan")
            row["n_gt"] = acc["n_gt"]
            res.setdefault(dt, {}).setdefault(name, {}).setdefault(sname, {})[b] = row
        return res


# ------------------------------------------------------------------------------------------------ detection
def gt_boxes_from_meta(boxes, circle_groups=CIRCLE_GROUPS, observed_only=True):
    """net/prep/gt_bev.py meta['boxes'] -> box dicts (size = [w, l, h]); circle groups get radius max(l, w) / 2.
    observed_only: drop boxes without any lidar / radar point (meta 'observed' == False), as the nuScenes detection
    evaluation does; the training labels ignore them too (review 2026-10-26)."""
    out = []
    for b in boxes:
        if observed_only and not b.get("observed", True):
            continue
        w, l = float(b["size"][0]), float(b["size"][1])
        d = {"group": b["group"], "cx": float(b["center"][0]), "cy": float(b["center"][1]),
             "vx": float(b["vel"][0]), "vy": float(b["vel"][1])}
        if b["group"] in circle_groups:
            d["radius"] = max(l, w) / 2
        else:
            d.update(length=l, width=w, yaw=float(b["yaw"]))
        out.append(d)
    return out


def in_front(b, x_max=60.0, half_fov=FRONT_HALF_FOV):
    """centre in 0 <= x < x_max and within +-half_fov of the x axis (apex at the ego origin; None: no angle limit)."""
    if not (0.0 <= b["cx"] < x_max):
        return False
    return half_fov is None or abs(math.atan2(b["cy"], b["cx"])) <= half_fov


# the comparison of a detection with the ground truth is Evaluation/comparison/boxes.py (read its README.md); the old names stay
tp_errors = _boxes.tp_errors              # ATE, ASE, AOE, AVE, BIoU of one matched pair (AOE nan for circles, AVE nan if a velocity is missing)
match = _boxes.match_frames               # match(frames, thr): frames = [(dets, gts)] of ONE class -> (rows (score, is_tp, det, gt) in score order, n_gt)
ap_nuscenes = _boxes.ap_nuscenes          # nuScenes AP: precision at 101 recall points, recall <= 0.1 and precision <= 0.1 clipped
ap_all_points = _boxes.ap_all_points      # all-point AP with the monotone precision envelope
TP_ERR_KEYS = _boxes.TP_ERROR_KEYS


class DetMeter:
    """Detection numbers per class, range bin and dt. Accumulates frames of detections and GT boxes per dt (`add`) and gives
    AP and TP errors (`result`). Underneath, everything is additive PER FRAME so any set of frames can be reduced without
    re-matching: `frame_rows` turns one frame into the true-positive flags of each of its detections (the greedy matching only
    involves the detection's own frame) plus the GT counts; `reduce_rows` sorts the rows of any set of frames by score and
    computes the numbers. The evaluation suite stores the rows per frame (Evaluation adapter evaluation/bev_task.py)."""

    def __init__(self, classes=None, thresholds=DIST_THRESHOLDS, x_max=60.0, half_fov=FRONT_HALF_FOV, bins=RANGE_BINS):
        self.classes, self.thr, self.x_max, self.half_fov, self.bins = classes, thresholds, x_max, half_fov, bins
        self.frames = {}                                  # dt -> list of (dets, gts)

    def keep(self, bs):
        """the boxes of the scored field of view"""
        return [b for b in bs if in_front(b, self.x_max, self.half_fov)]

    def add(self, dets, gts, dt=0.0):
        """one frame: dets (box dicts with score), gts (box dicts) at the target time t + dt, ego frame of t + dt."""
        self.frames.setdefault(round(float(dt), 3), []).append((self.keep(dets), self.keep(gts)))

    # ---- one frame -> rows (additive)
    @property
    def n_cols(self):
        T = len(self.thr)
        return 3 + 2 * T + 2 * len(TP_ERR_KEYS)

    def frame_rows(self, dets, gts, classes):
        """dets, gts: the boxes of one frame already inside the scored field of view. -> {'rows': (n_det, n_cols) float64,
        'n_gt': (n_classes, 1 + n_bins) int}. Row = [class index, range bin of the detection (-1 none), score, true-positive
        flag per threshold when matched among ALL boxes of the class (T), the same when matched among the boxes of its own
        range bin (T), TP errors of the match at TP_THRESHOLD among all (5), among its bin (5)]; errors are nan where the
        detection is no true positive at TP_THRESHOLD. n_gt column 0 = all, then one per bin."""
        T, nb = len(self.thr), len(self.bins)
        tp_thr = TP_THRESHOLD if TP_THRESHOLD in self.thr else None             # (reduce_rows' 'mean' needs the errors: thresholds include TP_THRESHOLD)
        rows, n_gt = [], np.zeros((len(classes), 1 + nb), int)
        for ci, cls in enumerate(classes):
            ds = [d for d in dets if d["group"] == cls]
            gs = [g for g in gts if g["group"] == cls]
            bins = [range_bin(d["cx"], self.bins) for d in ds]
            gt_bins = [range_bin(g["cx"], self.bins) for g in gs]
            n_gt[ci] = [len(gs)] + [gt_bins.count(i) for i in range(nb)]
            flags_all, errs_all = _boxes.match_flags(ds, gs, self.thr, tp_thr)            # matched among ALL boxes of the class
            flags_bin, errs_bin = np.zeros((len(ds), T)), np.full((len(ds), len(TP_ERR_KEYS)), np.nan)
            for i in range(nb):                                                           # matched among the boxes of its own range bin
                idx = [j for j, b in enumerate(bins) if b == i]
                if idx:
                    flags_bin[idx], errs_bin[idx] = _boxes.match_flags([ds[j] for j in idx], [g for g, b in zip(gs, gt_bins) if b == i],
                                                                       self.thr, tp_thr)
            for j, d in enumerate(ds):
                rows.append([ci, -1 if bins[j] is None else bins[j], d["score"], *flags_all[j], *flags_bin[j], *errs_all[j], *errs_bin[j]])
        return {"rows": np.asarray(rows, float).reshape(-1, self.n_cols), "n_gt": n_gt}

    # ---- any set of frames -> numbers
    def _scope_row(self, rr, n_gt, si):
        """the numbers of one (class, scope) from its rows rr (all rows of the scope) and its GT count: comparison.boxes.detection_numbers
        with ties="list" (rows with equal scores keep frame order, then detection order)"""
        T, E = len(self.thr), len(TP_ERR_KEYS)
        c0 = 3 if si == 0 else 3 + T
        e0 = 3 + 2 * T + (0 if si == 0 else E)
        return _boxes.detection_numbers(rr[:, 2], rr[:, c0:c0 + T], rr[:, e0:e0 + E], n_gt, self.thr,
                                        TP_THRESHOLD if TP_THRESHOLD in self.thr else None, ties="list")

    def reduce_rows(self, frames, classes, only=None):
        """frames: list of frame_rows() results. -> {class: {bin: {...}}, 'mean': {'all': {...}}} as result()[dt];
        only: optional set of (class, scope name) to compute (the rest is left out, 'mean' is then not computed)."""
        nb = len(self.bins)
        rows = np.concatenate([f["rows"] for f in frames]) if frames else np.zeros((0, self.n_cols))
        n_gt = sum((f["n_gt"] for f in frames), np.zeros((len(classes), 1 + nb), int))
        out = {}
        names = ["all"] + [bin_label(i) for i in range(nb)]
        for ci, cls in enumerate(classes):
            rc = rows[rows[:, 0] == ci]
            for si, name in enumerate(names):
                if only is not None and (cls, name) not in only:
                    continue
                rr = rc if si == 0 else rc[rc[:, 1] == si - 1]
                out.setdefault(cls, {})[name] = self._scope_row(rr, n_gt[ci, si], si)
        if only is None:
            mean = {}
            for name, keys in (("mAP", ["ap@%g" % t for t in self.thr]), ("mAP_all", ["ap_all@%g" % t for t in self.thr]),
                               ("mATE", ["ate"]), ("mASE", ["ase"]), ("mAOE", ["aoe"]), ("mAVE", ["ave"])):
                v = []
                for c in out:
                    row = out[c]["all"]
                    for k in keys:
                        if not row["n_gt"]:
                            continue                                                    # a class without ground truth is not in the mean
                        if row[k] == row[k]:
                            v.append(row[k])
                        elif k in TP_ERR_KEYS and row["n_tp"] == 0 and not (k == "aoe" and c in CIRCLE_GROUPS):
                            # ground truth that was not found at all: the devkit's 1.0 (not "skipped")
                            v.append(WORST_ERROR)
                mean[name] = float(np.mean(v)) if v else float("nan")
            out["mean"] = {"all": mean}
        return out

    # ---- the velocity error of the true positives, moving against static ground truth (additive per frame, scope: all boxes of the class)
    def frame_velocity(self, dets, gts, classes):
        """dets, gts: the boxes of one frame already inside the scored field of view. -> (n_classes, K) float64, the counts of
        comparison.boxes.velocity_counts per class (K = len(VELOCITY_COUNT_KEYS)), matched at TP_THRESHOLD among all boxes of the class:
        the same pairs whose ave goes into the class's 'all' row."""
        return np.array([_boxes.velocity_counts([d for d in dets if d["group"] == cls], [g for g in gts if g["group"] == cls],
                                                TP_THRESHOLD, MOVING_SPEED, MOVING_SPEED) for cls in classes],
                        float).reshape(len(classes), len(_boxes.VELOCITY_COUNT_KEYS))

    def reduce_velocity(self, frames, classes):
        """frames: list of frame_velocity() results. -> {class: {ave_moving, ave_static, ave0_moving, ave0_static, ave_skill_moving,
        ave_skill_static, ave_slow_moving, n_tp_*, n_gt_*}} (comparison.boxes.velocity_numbers); ave0 = the error of always saying 0 m/s
        on the same pairs, ave_skill = 1 - ave / ave0."""
        tot = sum(frames, np.zeros((len(classes), len(_boxes.VELOCITY_COUNT_KEYS))))
        return {cls: _boxes.velocity_numbers(tot[ci]) for ci, cls in enumerate(classes)}

    def result(self):
        """{dt: {class: {bin: {ap@t, ap_all@t, ate, ase, aoe, ave, biou, n_tp, n_gt, n_det}}, 'mean': {'all': {mAP,
        mAP_all, mATE, mASE, mAOE, mAVE}}}}; mAP = mean over classes and distance thresholds (classes with GT)."""
        res = {}
        for dt in sorted(self.frames):
            frames = self.frames[dt]
            classes = self.classes or sorted({b["group"] for ds, gs in frames for b in ds + gs})
            res[dt] = self.reduce_rows([self.frame_rows(ds, gs, classes) for ds, gs in frames], classes)
        return res


# ------------------------------------------------------------------------------------------------ latency
def ms_per_frame(fn, *args, n=50, warmup=10, batch1=True):
    """ms per call of fn(*args): warm-up calls, then n timed calls, each closed by torch.cuda.synchronize() when CUDA is
    available; returns median, p90, mean (ms) and n. Batch 1: every tensor argument must have first dimension 1.
    Valid only on an idle GPU (no other job: check nvidia-smi first; CLAUDE.md timing rule)."""
    import torch
    if batch1:
        for a in args:
            assert not torch.is_tensor(a) or a.dim() == 0 or a.shape[0] == 1, "batch 1 timing: first dimension must be 1"
    cuda = torch.cuda.is_available()
    ts = []
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args)
        if cuda:
            torch.cuda.synchronize()
        for _ in range(n):
            t0 = time.perf_counter()
            fn(*args)
            if cuda:
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.asarray(ts)
    return {"median": float(np.median(ts)), "p90": float(np.percentile(ts, 90)), "mean": float(ts.mean()), "n": n}


# ------------------------------------------------------------------------------------------------ report
def flatten(d, prefix=""):
    """nested dict -> flat dict 'a/b/c'; float keys (dt) become 'dt+0.50'."""
    out = {}
    for k, v in d.items():
        ks = dt_key(k) if isinstance(k, float) else str(k)
        key = prefix + "/" + ks if prefix else ks
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def unflatten(flat):
    """inverse of flatten: 'dt+0.50' parts become float keys again."""
    out = {}
    for key, v in flat.items():
        parts = [float(p[2:]) if _DT.match(p) else p for p in key.split("/")]
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return out


def report(seg=None, det=None, ms=None):
    """flat dict of everything: seg/<dt>/<layer>/<subset>/<bin>/<metric>, det/<dt>/<class>/<bin>/<metric>,
    ms/<name>/<median|p90|mean|n>. seg, det: SegMeter / DetMeter or their result(); ms: {name: ms_per_frame(...)}."""
    nested = {}
    if seg is not None:
        nested["seg"] = seg.result() if hasattr(seg, "result") else seg
    if det is not None:
        nested["det"] = det.result() if hasattr(det, "result") else det
    if ms is not None:
        nested["ms"] = ms
    return flatten(nested)


def format_table(flat, columns=None, digits=3):
    """printable table: one row per key prefix (everything but the metric), one column per metric; a new header
    whenever the metric set changes. columns: optional list of metric names to show."""
    rows = {}
    for key, v in flat.items():
        pre, metric = key.rsplit("/", 1)
        if columns is None or metric in columns:
            rows.setdefault(pre, {})[metric] = v
    width = max([len(p) for p in rows] + [4])
    lines, header = [], None

    def fmt(v):
        if isinstance(v, (int, np.integer)):
            return "%9d" % v
        return "%9s" % "-" if v != v else "%9.*f" % (digits, v)
    for pre, r in rows.items():
        cols = list(r)
        if cols != header:
            header = cols
            lines.append(" " * width + " | " + " ".join("%9s" % c[:9] for c in cols))
        lines.append(pre.ljust(width) + " | " + " ".join(fmt(r[c]) for c in cols))
    return "\n".join(lines)
