"""Curve and regression numbers of the BEV evaluation: classification (PR curve, ROC, calibration) and regression (velocity), added to the suite.

    BevCurveMetric   name "curve"; per layer and dt: the score histograms of the cells that are GT-positive / GT-negative (200 bins of the
                     predicted probability) -> average precision (area under the PR curve), ROC-AUC, best F1 and its threshold,
                     prevalence, expected calibration error (ECE), Brier score; the velocity field of the object cells (all / moving):
                     mean end-point error, RMSE, R^2 against the GT velocity, norm of the mean error (bias).

Why these numbers next to the IoU / AP of the suite: IoU and the box AP depend on ONE threshold (0.5) and one decode; the PR / ROC curves show the
ranking quality of the raw probabilities independent of it, the calibration says whether a probability can be read as a probability, the regression
numbers say how far the velocity is off in m/s, not only whether it is better than saying 0.
Counts are ADDITIVE (histograms and sums), so any set of frames (all, a scene, a bootstrap resample, train vs val) reduces to the same numbers; the
curves are exact up to the bin width 1 / 200 of the probability (ties inside a bin count as one threshold).
Cells: maps on the camera cells (`fov`), objects and velocity on the camera cells without ignored boxes (`obj_fov`), the same masks as the IoU.
Flat names: `curve/dt+0.00/<layer>/{ap, roc_auc, f1_max, thr_f1, prevalence, ece, brier, n_pos, n}`, `curve/dt+0.00/velocity/{obj, moving}/{epe, rmse, r2, bias, n}`.
Switched on with `bev_suite.py RUN --curves eval ...` (off by default: the standard card and its golden numbers stay unchanged)."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from evalsuite import Metric, Prediction, Sample
from evaluation.bev_views import masks
from evaluation.metrics import dt_key

BINS = 200
MOVING_SPEED = 0.5                                       # m/s, the evaluation's moving threshold (metrics.MOVING_SPEED)


def histogram(p: np.ndarray, y: np.ndarray, bins: int = BINS) -> np.ndarray:
    """(5, bins) float64: GT-positive count, GT-negative count, sum p, sum p^2, sum p over the positives, per probability bin"""
    p = np.clip(np.asarray(p, np.float64).ravel(), 0.0, 1.0)
    y = np.asarray(y, bool).ravel()
    idx = np.minimum((p * bins).astype(np.int64), bins - 1)
    f = lambda w: np.bincount(idx, weights=w, minlength=bins)  # noqa: E731
    return np.stack([f(y.astype(np.float64)), f((~y).astype(np.float64)), f(p), f(p * p), f(p * y)])


def classification_numbers(h: np.ndarray) -> Dict[str, float]:
    """AP, ROC-AUC, best F1 (+ its threshold), prevalence, ECE and Brier from the summed histogram of one layer"""
    pos, neg = h[0], h[1]
    bins = len(pos)
    P, N = float(pos.sum()), float(neg.sum())
    nan = float("nan")
    out = {"n_pos": P, "n": P + N, "prevalence": P / (P + N) if P + N else nan, "ap": nan, "roc_auc": nan, "f1_max": nan, "thr_f1": nan,
           "ece": nan, "brier": nan}
    if P + N == 0:
        return out
    n_bin = pos + neg
    conf = np.divide(h[2], n_bin, out=np.zeros(bins), where=n_bin > 0)
    acc = np.divide(pos, n_bin, out=np.zeros(bins), where=n_bin > 0)
    out["ece"] = float(np.sum(n_bin / (P + N) * np.abs(acc - conf)))
    out["brier"] = float((h[3].sum() - 2.0 * h[4].sum() + P) / (P + N))
    if P == 0 or N == 0:
        return out                                                      # no curve without both classes
    tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])                  # threshold from the top bin down
    rec, prec = tp / P, tp / np.maximum(tp + fp, 1.0)
    out["ap"] = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
    x, yv = np.concatenate([[0.0], fp / N]), np.concatenate([[0.0], rec])
    out["roc_auc"] = float(np.sum(np.diff(x) * (yv[1:] + yv[:-1]) / 2.0))
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    k = int(np.argmax(f1))
    out["f1_max"], out["thr_f1"] = float(f1[k]), float((bins - 1 - k) / bins)
    return out


def velocity_sums(v: np.ndarray, gv: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """(2, 8): rows = all object cells / moving cells; columns = n, sum epe, sum epe^2, sum dx, sum dy, sum gx, sum gy, sum |g|^2 (m/s)"""
    v, gv = np.asarray(v, np.float64), np.asarray(gv, np.float64)
    out = np.zeros((2, 8))
    speed = np.hypot(gv[0], gv[1])
    for r, sel in enumerate((cells, cells & (speed > MOVING_SPEED))):
        if not sel.any():
            continue
        dx, dy = v[0][sel] - gv[0][sel], v[1][sel] - gv[1][sel]
        epe = np.hypot(dx, dy)
        gx, gy = gv[0][sel], gv[1][sel]
        out[r] = [sel.sum(), epe.sum(), (epe * epe).sum(), dx.sum(), dy.sum(), gx.sum(), gy.sum(), (gx * gx + gy * gy).sum()]
    return out


def regression_numbers(s: np.ndarray) -> Dict[str, float]:
    n, epe, epe2, dx, dy, gx, gy, g2 = (float(x) for x in s)
    nan = float("nan")
    if n == 0:
        return {"n": 0.0, "epe": nan, "rmse": nan, "r2": nan, "bias": nan}
    sst = g2 - (gx * gx + gy * gy) / n                                    # variance of the GT velocity around its mean
    return {"n": n, "epe": epe / n, "rmse": float(np.sqrt(epe2 / n)), "r2": float(1.0 - epe2 / sst) if sst > 1e-9 else nan,
            "bias": float(np.hypot(dx, dy) / n)}


class BevCurveMetric(Metric):
    name = "curve"
    HEAD = {"drivable_ap": "dt+0.00/drivable_area/ap", "walkway_ap": "dt+0.00/walkway/ap", "lane_divider_ap": "dt+0.00/lane_divider/ap",
            "vehicle_ap": "dt+0.00/vehicle/ap", "velocity_moving_epe": "dt+0.00/velocity/moving/epe"}
    headline = tuple(HEAD.values())
    LOW = ("ece", "brier", "epe", "rmse", "bias")

    def __init__(self, spec, groups: Sequence[str]):
        self.spec, self.groups = list(spec), list(groups)

    def frame(self, sample: Sample, pred: Prediction):
        out = {}
        for dt, o in pred.outputs["dt"].items():
            g = sample.reference[dt]
            m = masks(g)
            dk = dt_key(dt)
            for name, kind, key in self.spec:
                if kind not in ("area", "line", "object") or key not in g or o["prob"].get(name) is None:
                    continue
                valid = m["obj_fov"] if kind == "object" else m["fov"]
                out[f"{dk}|{name}"] = histogram(np.asarray(o["prob"][name], np.float32)[valid], np.asarray(g[key], np.float32)[valid] > 0.5)
            if o.get("vhat") is not None and "velocity" in g:
                cells = m["obj_fov"] & np.any([np.asarray(g[c], np.float32) > 0.5 for c in self.groups if c in g], axis=0)
                out[f"{dk}|velocity"] = velocity_sums(o["vhat"], g["velocity"], cells)
        return out

    def reduce(self, counts, keys: Optional[Sequence[str]] = None):
        sums: Dict[str, np.ndarray] = {}
        for c in counts:
            for k, a in c.items():
                sums[k] = sums[k] + a if k in sums else np.array(a, np.float64)
        out: Dict[str, float] = {}
        for k, a in sums.items():
            dk, layer = k.split("|")
            if layer == "velocity":
                for r, sub in enumerate(("obj", "moving")):
                    for mk, v in regression_numbers(a[r]).items():
                        out[f"{dk}/velocity/{sub}/{mk}"] = float(v)
            else:
                for mk, v in classification_numbers(a).items():
                    out[f"{dk}/{layer}/{mk}"] = float(v)
        if keys is not None:
            return {k: out.get(k, float("nan")) for k in keys}
        return out

    def direction(self, key):
        return "low" if key.rsplit("/", 1)[-1] in self.LOW else "high"
