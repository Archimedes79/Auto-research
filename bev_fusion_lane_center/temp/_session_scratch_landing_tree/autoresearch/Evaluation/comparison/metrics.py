"""The join between the comparison core and the suite: two ready-made `Metric`s. The core (`maps.py`, `boxes.py`) knows nothing about
frames, samples or predictions; these classes only ask YOUR function for the two things to compare and store the core's counts.

    MapMetric(extract)   two pixel maps; extract(sample, pred) -> (pred_map, gt_map) or (pred_map, gt_map, valid_map): boolean maps
    BoxMetric(extract)   two lists of boxes; extract(sample, pred) -> (dets, gts): lists of box dicts (see boxes.py)

`extract` is the parsing: it reads the dataset's reference and the network's outputs and hands over plain arrays or dicts.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from ..contracts import Counts, Metric, Prediction, Sample
from . import boxes as B
from . import maps as M


WORST_ERROR = 1.0                           # the devkit's error of a class with ground truth and no true positive


def _suffix(tol: float) -> str:
    return "" if tol == 0 else "@%g" % tol


class MapMetric(Metric):
    """Two pixel maps compared cell by cell. Numbers: iou, precision, recall, f1 (maps.tolerant_scores); with a tolerance t > 0 (in cells)
    the same under the key suffix '@t': 'iou@2' counts a cell as right if the other map has a set cell within 2 cells. Counts per frame:
    'c' (n_tolerances, 4), additive. Frames without anything to find or predict give nan (see maps.py); the dataset number sums first.
    `extract(sample, pred)` returns (pred_map, gt_map) or (pred_map, gt_map, valid_map): boolean maps, any rank if every tolerance is 0."""

    def __init__(self, extract: Callable[[Sample, Prediction], Tuple], name: str = "map", tolerances: Sequence[float] = (0,),
                 headline: Optional[Sequence[str]] = None):
        self.extract, self.name, self.tolerances = extract, name, tuple(tolerances)
        self.headline = tuple(headline) if headline else tuple("iou" + _suffix(t) for t in self.tolerances)
        self.better = {}

    def frame(self, sample: Sample, pred: Prediction) -> Counts:
        maps = tuple(self.extract(sample, pred))
        if len(maps) not in (2, 3):
            raise ValueError(f"extract must return (pred_map, gt_map) or (pred_map, gt_map, valid_map), not {len(maps)} values")
        p, g, valid = (maps + (None,))[:3]
        return {"c": np.stack([M.tolerant_confusion(p, g, t, valid) for t in self.tolerances])}

    def reduce(self, counts: Sequence[Counts], keys: Optional[Sequence[str]] = None) -> Dict[str, float]:
        total = sum((c["c"] for c in counts), np.zeros((len(self.tolerances), 4), dtype=np.int64))
        out: Dict[str, float] = {}
        for t, c in zip(self.tolerances, total):
            out.update({k + _suffix(t): v for k, v in M.tolerant_scores(c).items()})
        return out

    def support(self, counts: Counts) -> Dict[str, float]:
        """the cells of the frame that have something to judge, ground-truth cells and false positives (what the iou is made of): a frame with
        nothing to find and nothing predicted is not ranked as an outlier, a frame with a hallucinated map is"""
        c = counts["c"][0]
        return {k: float(c[1] + c[2] + c[3]) for k in self.headline}


class BoxMetric(Metric):
    """Detections against ground-truth boxes, per class and over all classes, at several centre-distance thresholds (boxes.py).
    Numbers per class: 'car/ap@2', 'car/ap_all@2', 'car/ate' ... 'car/n_gt' (a class without ground truth is nan); over the classes:
    'mAP' (mean of ap over the classes that have ground truth and the thresholds), 'mAP_all', and the mean TP errors 'mATE', 'mASE',
    'mAOE', 'mAVE'. These are plain means over the true positives at `tp_threshold` and over the classes (NOT the devkit's cumulative
    curve over recall: not comparable with published numbers). A class with ground truth and no true positive counts 1.0, as in the
    devkit, so that a class that is not found at all does not simply drop out of the mean (which would improve it); a class whose
    error is nan by definition (aoe of a circle) is skipped.
    `ties` is the tie rule of boxes.detection_numbers. Counts per frame: 'rows' (n_det, 2 + T + 5) = [class index, score, flag per
    threshold, the five TP errors], ragged along axis 0, and 'n_gt' (n_classes,): additive. Boxes of a class that is not in `classes`
    are ignored."""

    def __init__(self, extract: Callable[[Sample, Prediction], Tuple[list, list]], classes: Sequence[str], name: str = "det",
                 thresholds: Sequence[float] = (0.5, 1.0, 2.0, 4.0), tp_threshold: float = 2.0, ties: str = "pessimistic"):
        if tp_threshold not in thresholds:
            raise ValueError(f"tp_threshold {tp_threshold} must be one of the thresholds {tuple(thresholds)}")
        self.extract, self.name, self.classes, self.ties = extract, name, tuple(classes), ties
        self.thr, self.tp_thr = tuple(thresholds), tp_threshold
        self.headline = ("mAP", "mATE")
        self.better = {"mATE": "low", "mASE": "low", "mAOE": "low", "mAVE": "low"}

    def frame(self, sample: Sample, pred: Prediction) -> Counts:
        dets, gts = self.extract(sample, pred)
        rows, n_gt = [], np.zeros(len(self.classes), dtype=np.int64)
        for ci, cls in enumerate(self.classes):
            ds, gs = [d for d in dets if d["group"] == cls], [g for g in gts if g["group"] == cls]
            n_gt[ci] = len(gs)
            flags, errors = B.match_flags(ds, gs, self.thr, self.tp_thr)
            rows += [[ci, d["score"], *flags[i], *errors[i]] for i, d in enumerate(ds)]
        return {"rows": np.array(rows, float).reshape(-1, 2 + len(self.thr) + len(B.TP_ERROR_KEYS)), "n_gt": n_gt}

    def reduce(self, counts: Sequence[Counts], keys: Optional[Sequence[str]] = None) -> Dict[str, float]:
        T = len(self.thr)
        rows = np.concatenate([c["rows"] for c in counts]) if counts else np.zeros((0, 2 + T + len(B.TP_ERROR_KEYS)))
        n_gt = sum((c["n_gt"] for c in counts), np.zeros(len(self.classes), dtype=np.int64))
        per_class = {}
        for ci, cls in enumerate(self.classes):
            r = rows[rows[:, 0] == ci]
            per_class[cls] = B.detection_numbers(r[:, 1], r[:, 2:2 + T], r[:, 2 + T:], int(n_gt[ci]), self.thr, self.tp_thr, self.ties)
        out = {f"{cls}/{k}": v for cls, row in per_class.items() for k, v in row.items()}
        for name, ks in (("mAP", ["ap@%g" % t for t in self.thr]), ("mAP_all", ["ap_all@%g" % t for t in self.thr]), ("mATE", ["ate"]),
                         ("mASE", ["ase"]), ("mAOE", ["aoe"]), ("mAVE", ["ave"])):
            v = []
            for row in per_class.values():
                for k in ks:
                    if k not in row:
                        continue
                    if row[k] == row[k]:
                        v.append(row[k])
                    elif k in B.TP_ERROR_KEYS and row["n_gt"] > 0 and row["n_tp"] == 0:
                        v.append(WORST_ERROR)                    # ground truth that was not found at all
            out[name] = float(np.mean(v)) if v else float("nan")
        return out

    def support(self, counts: Counts) -> Dict[str, float]:
        """the ground-truth boxes of the frame: a frame with nothing to find is not ranked as an outlier"""
        return {k: float(counts["n_gt"].sum()) for k in self.headline}
