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
  mask rule of training/gt_bev.py); precision / recall / F1 of those cells at the same tolerances, plus the mean
  absolute field error on cells with GT distance < 1 m (the sub-cell accuracy the loss trains, plan 8.4).
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
Box dicts as training/object_head.py decode(): group, score, cx, cy (m, ego frame), length, width, yaw or radius, vx,
vy; gt_boxes_from_meta() converts training/gt_bev.py meta["boxes"].
The comparison itself (cell counts and tolerant counts, matching, the errors of a matched pair, average precision) is not here but in
autoresearch/evalsuite/comparison/ (maps.py, boxes.py; read its README.md): this file adds what is specific to the BEV system (range bins,
field of view, circles for pedestrians, dt keys, the known / unknown split, the lines) and calls that core. `match`, `ap_nuscenes`,
`ap_all_points`, `tp_errors`, `dilate` and TP_ERR_KEYS stay importable from here under their old names (`match` is the core's
`match_frames`). DetMeter reduces with ties="list": equal scores keep frame order, then detection order, as it always did.
Two differences from the code before the core: a detection whose score is nan (a diverged model) raises a ValueError at its frame (it used to
rank last and give garbage numbers); and importing this file imports the evalsuite package (numpy only), so a broken evalsuite breaks this file.
Run the self-test: wsl ~/miniconda3/envs/gcar/bin/python evaluation/metrics.py
"""
import math
import os
import re
import sys
import time

import numpy as np

_AUTORESEARCH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "autoresearch"))
if _AUTORESEARCH not in sys.path:
    sys.path.append(_AUTORESEARCH)                       # the comparison core is in the autoresearch folder next to this problem
from evalsuite.comparison import boxes as _boxes, maps as _maps  # noqa: E402

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
        cnt = _maps.tolerant_counts_many(P, G, dil, cms)                        # (tolerances, selections, 4): [p_hit, p_miss, g_hit, g_miss], one matrix product
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

    def result(self):
        """{dt: {name: {subset: {bin: {metric: value}}}}}; areas: iou@t, lines (':zc'): f1@t, p@t, r@t, field_mae."""
        res = {}
        for (dt, name, sname, b), acc in sorted(self.c.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
            row = {}
            for i, t in enumerate(self.tol):
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
            row["n_gt"] = acc["n_gt"]
            res.setdefault(dt, {}).setdefault(name, {}).setdefault(sname, {})[b] = row
        return res


# ------------------------------------------------------------------------------------------------ detection
def gt_boxes_from_meta(boxes, circle_groups=CIRCLE_GROUPS, observed_only=True):
    """training/gt_bev.py meta['boxes'] -> box dicts (size = [w, l, h]); circle groups get radius max(l, w) / 2.
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


# the comparison of a detection with the ground truth is evalsuite/comparison/boxes.py (read its README.md); the old names stay
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
    computes the numbers. The evaluation suite stores the rows per frame (evalsuite adapter evaluation/bev_task.py)."""

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
                v = [out[c]["all"][k] for c in out for k in keys if out[c]["all"]["n_gt"] and out[c]["all"][k] == out[c]["all"][k]]
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


# ----------------------------------------------------------------------------------------------- self-test
def _selftest():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
    from lift_pd import Grid
    g = Grid()                                            # x 0..60, y -26..26, 0.5 m: 120 x 104 cells
    X, Y = g.nx, g.ny
    eq = lambda a, b, e=1e-9: abs(a - b) <= e                                                 # noqa: E731

    # 1 range bins of cells (by centre x) and boxes
    cb = cell_bins((X, Y), g)
    assert cb[0, 0] == 0 and cb[39, 5] == 0 and cb[40, 5] == 1 and cb[79, 0] == 1 and cb[80, 0] == 2 and cb[119, 0] == 2
    assert [(cb[:, 0] == i).sum() for i in range(3)] == [40, 40, 40]
    assert range_bin(19.99) == 0 and range_bin(20.0) == 1 and range_bin(45.0) == 2
    assert range_bin(-1.0) is None and range_bin(60.0) is None

    # 2 segmentation: perfect -> 1 at every tolerance; a 1-cell shift (0.5 m) -> 90 / 110 at 0, 1 at 0.5 and 1.0 m
    gt = np.zeros((X, Y))
    gt[10:20, 30:40] = 1.0                                # a 10 x 10 cell block in 0-20 m
    gt[60:64, 50:60] = 1.0                                # a 4 x 10 block in 20-40 m
    m = SegMeter(g)
    m.add({"vehicle": gt}, {"vehicle": gt})
    r = m.result()[0.0]["vehicle"]["all"]
    for b in ("all", "0-20", "20-40"):
        assert all(eq(r[b]["iou@%g" % t], 1.0) for t in TOLERANCES), r[b]
    assert math.isnan(r["40-60"]["iou@0"]) and r["all"]["n_gt"] == 140
    shifted = np.roll(gt, 1, axis=0)
    m = SegMeter(g)
    m.add({"vehicle": shifted}, {"vehicle": gt})
    r = m.result()[0.0]["vehicle"]["all"]
    assert eq(r["0-20"]["iou@0"], 90 / 110) and eq(r["0-20"]["iou@0.5"], 1.0) and eq(r["0-20"]["iou@1"], 1.0)
    iou_shift = r["0-20"]["iou@0"]
    # mask: false positives on undefined cells do not count
    fp = gt.copy()
    fp[100:110, 0:10] = 1.0
    mask = np.ones((X, Y), bool)
    mask[95:, :20] = False
    m = SegMeter(g)
    m.add({"vehicle": fp}, {"vehicle": gt}, mask=mask)
    assert eq(m.result()[0.0]["vehicle"]["all"]["all"]["iou@0"], 1.0)
    # known / unknown: the counts split exactly (mass 0 in the far half)
    mass = np.ones((X, Y))
    mass[:, :52] = 0.0
    m = SegMeter(g)
    m.add({"vehicle": shifted}, {"vehicle": gt}, mass=mass, dt=0.5)
    c = {s: m.c[(0.5, "vehicle", s, "all")] for s in ("all", "known", "unknown")}
    assert np.array_equal(c["all"]["cnt"], c["known"]["cnt"] + c["unknown"]["cnt"])
    assert c["known"]["n_gt"] == 32 and c["unknown"]["n_gt"] == 108

    # 3 lines: distance field of a line along x at y = 0.1 m; perfect -> F1 1 and field error 0; a lateral 1-cell
    #   shift -> F1 0 at 0 m, 1 at 0.5 m; field error on the 4 cells with GT distance < 1 m (y = -0.75 .. 0.75):
    #   |1.35 - 0.85|, |0.85 - 0.35|, |0.35 - 0.15|, |0.15 - 0.65| -> 1.7 / 4 = 0.425 m
    yc = g.y_min + (np.arange(Y) + 0.5) * g.res
    field = np.minimum(np.abs(yc - 0.1), 2.0)[None].repeat(X, 0)
    m = SegMeter(g)
    m.add_lines({"lane_divider": field}, {"lane_divider": field})
    r = m.result()[0.0]["lane_divider:zc"]["all"]["all"]
    assert all(eq(r["f1@%g" % t], 1.0) for t in TOLERANCES) and r["field_mae"] == 0 and r["n_gt"] == X
    m = SegMeter(g)
    m.add_lines({"lane_divider": np.minimum(np.abs(yc - 0.6), 2.0)[None].repeat(X, 0)}, {"lane_divider": field})
    r = m.result()[0.0]["lane_divider:zc"]["all"]["all"]
    assert r["f1@0"] == 0 and eq(r["f1@0.5"], 1.0) and eq(r["field_mae"], 0.425), r

    # 4 detection: perfect -> AP 1 (both), TP errors 0
    gts = [[{"group": "car", "cx": 10.0, "cy": 1.0, "length": 4.5, "width": 1.8, "yaw": 0.3, "vx": 5.0, "vy": 0.0},
            {"group": "car", "cx": 30.0, "cy": -2.0, "length": 4.5, "width": 1.8, "yaw": -1.0, "vx": 0.0, "vy": 0.0},
            {"group": "pedestrian", "cx": 15.0, "cy": 3.0, "radius": 0.4, "vx": 0.0, "vy": 1.2}],
           [{"group": "car", "cx": 50.0, "cy": 4.0, "length": 4.5, "width": 1.8, "yaw": 3.0, "vx": 1.0, "vy": 1.0},
            {"group": "car", "cx": 10.0, "cy": 20.0, "length": 4.5, "width": 1.8, "yaw": 0.0, "vx": 0.0, "vy": 0.0}]]
    assert not in_front(gts[1][1])                        # 63 deg off the axis: outside the front field of view
    dm = DetMeter()
    for gs in gts:
        dm.add([dict(b, score=0.9) for b in gs], gs)
    r = dm.result()[0.0]
    for cls in ("car", "pedestrian"):
        row = r[cls]["all"]
        assert all(eq(row["ap@%g" % t], 1.0) and eq(row["ap_all@%g" % t], 1.0) for t in DIST_THRESHOLDS), row
        assert all(eq(row[k], 0.0, 1e-6) for k in ("ate", "ase", "ave")) and eq(row["biou"], 1.0, 1e-6)
    assert r["car"]["all"]["n_gt"] == 3 and r["car"]["0-20"]["n_gt"] == 1 and r["car"]["40-60"]["n_gt"] == 1
    assert eq(r["car"]["all"]["aoe"], 0.0) and math.isnan(r["pedestrian"]["all"]["aoe"]) and eq(r["mean"]["all"]["mAP"], 1.0)
    # empty prediction -> AP 0
    dm = DetMeter()
    dm.add([], gts[0])
    assert all(dm.result()[0.0][c]["all"]["ap@%g" % t] == 0.0 for c in ("car", "pedestrian") for t in DIST_THRESHOLDS)
    # a box rotated by 90 deg: AOE pi / 2; ASE 0 (sizes aligned); BIoU by hand 1.8^2 / (2 * 4.5 * 1.8 - 1.8^2) = 0.25
    g0 = gts[0][0]
    e = tp_errors(dict(g0, yaw=g0["yaw"] + math.pi / 2, vx=g0["vx"] + 1.0), g0)
    assert eq(e["aoe"], math.pi / 2) and eq(e["ase"], 0.0) and eq(e["biou"], 0.25, 1e-6) and eq(e["ave"], 1.0)
    # size 4.0 x 2.0 against 4.5 x 1.8: ASE = 1 - 4.0 * 1.8 / (8.1 + 8.0 - 7.2); circle 0.5 against 0.4: 1 - 0.64
    assert eq(tp_errors(dict(g0, length=4.0, width=2.0), g0)["ase"], 1 - 7.2 / 8.9)
    assert eq(tp_errors(dict(gts[0][2], radius=0.5), gts[0][2])["ase"], 0.36)
    # known two-detection ranking: 2 GT, a false positive (0.9) above a true positive (0.8): precision (0, 0.5),
    #   recall (0, 0.5): all-point AP 0.5 * 0.5 = 0.25; nuScenes p(r) = r up to 0.5: sum_{k=11}^{50} (k / 100 - 0.1)
    #   = 8.2 over 90 bins / 0.9 = 8.2 / 81
    two = [{"group": "car", "cx": 10.0, "cy": 0.0, "length": 4.5, "width": 1.8, "yaw": 0.0, "vx": 0.0, "vy": 0.0},
           {"group": "car", "cx": 25.0, "cy": 0.0, "length": 4.5, "width": 1.8, "yaw": 0.0, "vx": 0.0, "vy": 0.0}]
    dm = DetMeter()
    dm.add([dict(two[0], cx=14.0, score=0.9), dict(two[1], cx=25.3, score=0.8)], two)
    row = dm.result()[0.0]["car"]["all"]
    assert eq(row["ap_all@1"], 0.25) and eq(row["ap@1"], 8.2 / 81) and eq(row["ate"], 0.3) and row["n_tp"] == 1, row
    assert eq(row["ap_all@4"], 1.0) and eq(row["ap@4"], 1.0)  # at 4 m the 4 m-off detection matches too
    ap_rank = row["ap@1"]

    # 5 dt-keyed results: three dt, flatten -> unflatten round trip, report and table
    dm, sm = DetMeter(), SegMeter(g)
    for dt in (0.0, 0.5, -0.5):
        dm.add([dict(b, score=0.9) for b in gts[0]], gts[0], dt=dt)
        sm.add({"vehicle": shifted}, {"vehicle": gt}, mass=mass if dt else None, dt=dt)
    nested = {"det": dm.result(), "seg": sm.result()}
    assert sorted(nested["det"]) == [-0.5, 0.0, 0.5] and "unknown" in nested["seg"][0.5]["vehicle"]
    back = unflatten(flatten(nested))
    fa, fb = flatten(nested), flatten(back)
    assert sorted(back["det"]) == [-0.5, 0.0, 0.5] and fa.keys() == fb.keys()
    assert all((fa[k] == fb[k]) or (fa[k] != fa[k] and fb[k] != fb[k]) for k in fa)
    assert "det/dt-0.50/car/0-20/ap@0.5" in fa and "seg/dt+0.50/vehicle/known/0-20/iou@0.5" in fa

    # 5b boxes without any lidar / radar point are not GT (as in the nuScenes detection evaluation); opt-out flag
    mb = [{"group": "vehicle", "size": [1.8, 4.5, 1.5], "center": [10.0, 0.0, 0.0], "vel": [0.0, 0.0], "yaw": 0.0, "observed": True},
          {"group": "vehicle", "size": [1.8, 4.5, 1.5], "center": [30.0, 0.0, 0.0], "vel": [0.0, 0.0], "yaw": 0.0, "observed": False},
          {"group": "vehicle", "size": [1.8, 4.5, 1.5], "center": [40.0, 0.0, 0.0], "vel": [0.0, 0.0], "yaw": 0.0}]
    assert len(gt_boxes_from_meta(mb)) == 2 and len(gt_boxes_from_meta(mb, observed_only=False)) == 3

    # 5c the per-frame rows (frame_rows / reduce_rows) give the numbers of the ORIGINAL global greedy matching (oracle below: the
    #    frozen copy of the former DetMeter code in evalsuite/selftest_comparison.py, independent of the core that DetMeter calls now)
    #    on random frames with ties, empty frames, circles and boxes outside the field of view; the reduction is order-independent;
    #    `only` computes just the asked scopes
    from evalsuite.selftest_comparison import _REF

    def oracle_eval(frames):
        row = {}
        for t in DIST_THRESHOLDS:
            mm, n_gt = _REF.match(frames, t)
            fl = [x[1] for x in mm]
            row["ap@%g" % t], row["ap_all@%g" % t] = _REF.ap_nuscenes(fl, n_gt), _REF.ap_all_points(fl, n_gt)
            if t == TP_THRESHOLD:
                errs = [_REF.tp_errors(d, g) for _, ok, d, g in mm if ok]
                for k in ("ate", "ase", "aoe", "ave", "biou"):
                    v = [e[k] for e in errs if e[k] == e[k]]
                    row[k] = float(np.mean(v)) if v else float("nan")
                row["n_tp"] = len(errs)
        row["n_gt"] = sum(len(gg) for _, gg in frames)
        row["n_det"] = sum(len(dd) for dd, _ in frames)
        return row
    rng = np.random.default_rng(3)
    classes = ["vehicle", "pedestrian"]
    dm = DetMeter(classes=classes)
    for dt in (0.0, 0.5):
        for f in range(30):
            gs = []
            for _ in range(int(rng.integers(0, 6))):
                cls = classes[int(rng.integers(0, 2))]
                b = {"group": cls, "cx": float(rng.uniform(0, 62)), "cy": float(rng.uniform(-30, 30)), "vx": float(rng.normal()),
                     "vy": float(rng.normal())}
                b.update({"radius": 0.4} if cls == "pedestrian" else {"length": 4.5, "width": 1.8, "yaw": float(rng.uniform(-3, 3))})
                gs.append(b)
            ds = [dict(b, cx=b["cx"] + float(rng.normal(0, 1.2)), cy=b["cy"] + float(rng.normal(0, 1.2)),
                       score=float(rng.choice([0.9, 0.9, 0.5, rng.uniform()]))) for b in gs if rng.uniform() < 0.8]
            ds += [dict(gs[0] if gs else {"group": "vehicle", "length": 4.5, "width": 1.8, "yaw": 0.0, "vx": 0.0, "vy": 0.0},
                        cx=float(rng.uniform(0, 60)), cy=float(rng.uniform(-20, 20)), score=float(rng.uniform())) for _ in range(3)]
            dm.add(ds, gs, dt=dt)
    new = dm.result()
    for dt, frames in dm.frames.items():
        for cls in classes:
            fc = [([d for d in ds if d["group"] == cls], [g for g in gs if g["group"] == cls]) for ds, gs in frames]
            want = {"all": oracle_eval(fc)}
            for i in range(3):
                inb = lambda bs: [b for b in bs if range_bin(b["cx"]) == i]                         # noqa: E731
                want[bin_label(i)] = oracle_eval([(inb(ds), inb(gs)) for ds, gs in fc])
            for sc, row in want.items():
                assert list(row) == list(new[dt][cls][sc]), (list(row), list(new[dt][cls][sc]))
                for k, v in row.items():
                    w = new[dt][cls][sc][k]
                    assert (v != v and w != w) or abs(v - w) < 1e-12, (dt, cls, sc, k, v, w)
        rows = [dm.frame_rows(ds, gs, classes) for ds, gs in frames]
        part = dm.reduce_rows(rows, classes, only={("vehicle", "all")})
        assert list(part) == ["vehicle"] and part["vehicle"]["all"] == new[dt]["vehicle"]["all"]
    # without tied scores the frame order does not matter (with ties the greedy order is 'frame order, then detection order')
    uni = DetMeter(classes=classes)
    for ds, gs in dm.frames[0.0]:
        uni.add([dict(d, score=float(rng.uniform())) for d in ds], gs)
    rows = [uni.frame_rows(ds, gs, classes) for ds, gs in uni.frames[0.0]]
    a, b = flatten({"x": uni.reduce_rows(rows, classes)}), flatten({"x": uni.reduce_rows(rows[::-1], classes)})
    assert a.keys() == b.keys() and all((a[k] == b[k]) or (a[k] != a[k] and b[k] != b[k]) or abs(a[k] - b[k]) < 1e-12 for k in a)
    # a detection without a score (nan: a diverged model) is refused at its frame, not ranked last
    nan_det = {"group": "vehicle", "cx": 5.0, "cy": 0.0, "length": 4.5, "width": 1.8, "yaw": 0.0, "vx": 0.0, "vy": 0.0, "score": float("nan")}
    try:
        DetMeter(classes=classes).frame_rows([nan_det], [dict(nan_det, score=1.0)], classes)
    except ValueError:
        pass
    else:
        raise AssertionError("a nan score must be refused")

    # 6 latency helper (CPU callable here; timing itself only on an idle GPU)
    import torch
    x = torch.randn(1, 8, 64, 64)
    conv = torch.nn.Conv2d(8, 8, 3, padding=1)
    lat = ms_per_frame(conv, x, n=10, warmup=2)
    assert lat["median"] > 0 and lat["n"] == 10
    flat = report(seg=sm, det=dm, ms={"conv": lat})
    table = format_table({k: v for k, v in flat.items() if "/dt+0.00/" in k and "/all/" in k or k.startswith("ms/")},
                         columns=["iou@0", "iou@0.5", "iou@1", "ap@1", "ap_all@1", "ate", "ase", "aoe", "ave",
                                  "median", "n_gt"])
    print(table)
    print(f"metrics selftest ok: range bins of cells and boxes, IoU perfect 1 / 1-cell shift {iou_shift:.3f} at 0 m "
          f"and 1 at 0.5 / 1 m, mask, known / unknown split, line zero-crossing F1 (perfect 1, shift 0 / 1, field "
          f"error 0.425 m), detection perfect AP 1 and errors 0, empty AP 0, 90 deg AOE pi/2 ASE 0 BIoU 0.25, ASE "
          f"by hand (box 0.191, circle 0.36), two-detection AP nuScenes {ap_rank:.4f} = 8.2/81 / all-point 0.25, "
          f"front field of view, dt-keyed round trip ({len(fa)} keys), per-frame rows = the original greedy matching on random "
          f"frames (ties, circles, empty frames), latency {lat['median']:.2f} ms (CPU)")


if __name__ == "__main__":
    _selftest()
