"""Compare two pixel maps: a prediction against the ground truth, cell by cell.

Inputs are plain numpy arrays of the same shape. Everything that is not the comparison - loading, thresholding a probability map,
choosing which cells count - is the caller's job. Outputs are COUNTS, which add up over frames (the counts of two frames summed are
the counts of both frames together), and the numbers that follow from counts.

                         ground truth set    ground truth empty
        prediction set        tp                   fp
        prediction empty      fn                   tn

`valid` (optional boolean map) restricts the count to some cells: cells without ground truth, outside the sensor's view, one range
band. Cells outside `valid` are not counted at all.

A ratio whose denominator is 0 is nan ("nothing to find", "nothing predicted"), never 0 and never 1: the caller decides what
that means, and a dataset number sums the counts first, so the nan only shows where there is truly nothing to judge.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

NAN = float("nan")


def _as_mask(a, name: str) -> np.ndarray:
    """a boolean map from a bool or a 0 / 1 integer array. A float map is refused (a probability map has to be thresholded first: a
    nonzero probability is not a set cell), and so is an integer map with other values (a label like 255 = "ignore" belongs in `valid`)."""
    a = np.asarray(a)
    if a.dtype.kind == "f":
        raise TypeError(f"{name} is a float map: threshold it first (e.g. prob > 0.5), a nonzero probability is not a set cell")
    if a.dtype.kind in "iu" and a.size and (a.min() < 0 or a.max() > 1):
        raise ValueError(f"{name} is an integer map with values other than 0 and 1: make a boolean map of it (an 'ignore' label belongs in `valid`)")
    return a.astype(bool)


def _same_shape(*maps: np.ndarray) -> None:
    if len({m.shape for m in maps}) != 1:
        raise ValueError(f"the maps must have the same shape, got {[m.shape for m in maps]}")


def ratio(num: float, den: float) -> float:
    """num / den, nan when den is 0"""
    return float(num) / float(den) if den else NAN


def f1(precision: float, recall: float) -> float:
    """harmonic mean; nan when precision or recall is undefined, 0 when both are 0"""
    if math.isnan(precision) or math.isnan(recall):
        return NAN
    return 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


# ------------------------------------------------------------------------------------------------ plain comparison
def confusion(pred, gt, valid=None) -> np.ndarray:
    """[tp, fp, fn, tn] over the valid cells, int64 of shape (4,). pred, gt, valid: bool (or 0 / 1 integer) maps of one shape."""
    p, g = _as_mask(pred, "pred"), _as_mask(gt, "gt")
    v = np.ones_like(p) if valid is None else _as_mask(valid, "valid")
    _same_shape(p, g, v)
    return np.array([(p & g & v).sum(), (p & ~g & v).sum(), (~p & g & v).sum(), (~p & ~g & v).sum()], dtype=np.int64)


def scores(counts) -> Dict[str, float]:
    """iou, precision, recall and f1 from [tp, fp, fn, tn] (summed over any frames).
    iou = tp / (tp + fp + fn), precision = tp / (tp + fp), recall = tp / (tp + fn), f1 = their harmonic mean (nan if either is nan)."""
    tp, fp, fn = (float(x) for x in counts[:3])
    p, r = ratio(tp, tp + fp), ratio(tp, tp + fn)
    return {"iou": ratio(tp, tp + fp + fn), "precision": p, "recall": r, "f1": f1(p, r)}


# ------------------------------------------------------------------------------------------------ tolerant comparison
def dilate(mask, radius_cells: float) -> np.ndarray:
    """boolean dilation of a 2-D mask by a disc: a cell is set if a set cell's centre lies within radius_cells of its centre."""
    m = _as_mask(mask, "mask")
    if m.ndim != 2:
        raise ValueError("dilate works on 2-D maps")
    r = int(math.floor(radius_cells + 1e-9))
    if r <= 0:
        return m.copy()
    rows, cols = m.shape
    padded = np.pad(m, r)
    out = np.zeros_like(m)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx * dx + dy * dy <= radius_cells * radius_cells + 1e-9:
                out |= padded[r + dx:r + dx + rows, r + dy:r + dy + cols]
    return out


def tolerant_confusion(pred, gt, tol_cells: float, valid=None) -> np.ndarray:
    """[p_hit, p_miss, g_hit, g_miss] over the valid cells, int64 of shape (4,), for thin structures that cannot be hit cell by cell.
    A predicted cell is a hit if a ground-truth cell lies within tol_cells of it, a ground-truth cell is hit if a predicted cell lies
    within tol_cells of it. tol_cells = 0 is the plain comparison (p_hit = g_hit = tp, p_miss = fp, g_miss = fn) and works for maps of
    any rank; a tolerance above 0 needs 2-D maps. The dilation uses the whole maps, `valid` only chooses what is counted."""
    if tol_cells < 0:
        raise ValueError("tol_cells must not be negative")
    p, g = _as_mask(pred, "pred"), _as_mask(gt, "gt")
    near_gt, near_pred = (g, p) if tol_cells == 0 else (dilate(g, tol_cells), dilate(p, tol_cells))
    return tolerant_counts(p, g, near_gt, near_pred, valid)


def tolerant_counts(pred, gt, near_gt, near_pred, valid=None) -> np.ndarray:
    """The counting step of `tolerant_confusion`, for callers that count the same maps under several `valid` selections (range bands,
    known / unknown cells) and so dilate only once: near_gt = dilate(gt, tol), near_pred = dilate(pred, tol), of the whole maps.
    -> [p_hit, p_miss, g_hit, g_miss] over the valid cells (int64). The dilations are the caller's responsibility, only their shapes are checked."""
    p, g = _as_mask(pred, "pred"), _as_mask(gt, "gt")
    near_g, near_p = _as_mask(near_gt, "near_gt"), _as_mask(near_pred, "near_pred")
    v = np.ones_like(p) if valid is None else _as_mask(valid, "valid")
    _same_shape(p, g, near_g, near_p, v)
    return np.array([(p & near_g & v).sum(), (p & ~near_g & v).sum(), (g & near_p & v).sum(), (g & ~near_p & v).sum()], dtype=np.int64)


def tolerant_scores(counts) -> Dict[str, float]:
    """iou, precision, recall, f1 from tolerant counts [p_hit, p_miss, g_hit, g_miss].
    precision = p_hit / (p_hit + p_miss), recall = g_hit / (g_hit + g_miss), f1 from both; the boundary-tolerant iou counts a true
    positive as the mean of the two hit counts: iou = tp / (tp + p_miss + g_miss) with tp = (p_hit + g_hit) / 2."""
    p_hit, p_miss, g_hit, g_miss = (float(x) for x in counts[:4])
    tp = (p_hit + g_hit) / 2.0
    p, r = ratio(p_hit, p_hit + p_miss), ratio(g_hit, g_hit + g_miss)
    return {"iou": ratio(tp, tp + p_miss + g_miss), "precision": p, "recall": r, "f1": f1(p, r)}
