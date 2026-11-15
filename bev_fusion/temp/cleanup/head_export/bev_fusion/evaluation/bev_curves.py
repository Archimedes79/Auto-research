"""Curve and regression numbers of the BEV evaluation: classification (PR curve, ROC, calibration) and regression (velocity), added to the suite.

    BevCurveMetric   name "curve"; per layer and dt: the score histograms of the cells that are GT-positive / GT-negative (400 bins equally spaced in
                     logit(p) over [-12, 12]) -> average precision (area under the PR curve), ROC-AUC, best F1 and its threshold, prevalence, expected
                     calibration error (ECE, over the same bins), Brier score; the velocity field of the object cells (all / moving): mean end-point
                     error, RMSE, R^2 against the GT velocity, norm of the mean error (bias).

Why these numbers next to the IoU / AP of the suite: IoU and the box AP depend on ONE threshold (0.5) and one decode; the PR / ROC curves show the
ranking quality of the raw probabilities independent of it, the calibration says whether a probability can be read as a probability, the regression
numbers say how far the velocity is off in m/s, not only whether it is better than saying 0.
Counts are ADDITIVE (histograms and sums), so any set of frames (all, a scene, a bootstrap resample, train vs val) reduces to the same numbers. The curves
are exact up to ties inside a bin (they count as one threshold). The bins are equally spaced in logit(p), not in p: a network trained on a sparse layer puts
most negatives below p = 0.005, and equal-width bins in p lumped them into one bin and biased AP and AUC down by 0.015 - 0.02 (critic of 2026-11-01);
`--selftest` compares with the exact curves of sorted scores (within 0.004). A NaN probability counts as p = 0 (the IoU path does the same, NaN > 0.5 is false).
Cells: maps on the camera cells (`fov`), objects and velocity on the camera cells without ignored boxes (`obj_fov`), the same masks as the IoU.
Flat names: `curve/dt+0.00/<layer>/{ap, roc_auc, f1_max, thr_f1, prevalence, ece, brier, n_pos, n}`, `curve/dt+0.00/velocity/{obj, moving}/{epe, rmse, r2, bias, n}`.
Switched on with `bev_suite.py RUN --curves eval ...` (off by default: the standard card and its golden numbers stay unchanged); the evaluation of a run with
--curves goes into its own store (`eval_curves_n<N>`), so an evaluation made without them is never silently reused.
    python evaluation/bev_curves.py --selftest"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Sequence

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")]
from evalsuite import Metric, Prediction, Sample  # noqa: E402
from evaluation.bev_views import masks  # noqa: E402
from evaluation.metrics import dt_key  # noqa: E402

BINS = 400
LOGIT = 12.0                                             # the bins span logit(p) in [-LOGIT, LOGIT]; p outside lands in the end bins
MOVING_SPEED = 0.5                                       # m/s, the evaluation's moving threshold (metrics.MOVING_SPEED)


def histogram(p: np.ndarray, y: np.ndarray, bins: int = BINS) -> np.ndarray:
    """(5, bins) float64: GT-positive count, GT-negative count, sum p, sum p^2, sum p over the positives, per logit-spaced probability bin"""
    p = np.clip(np.nan_to_num(np.asarray(p, np.float64).ravel(), nan=0.0), 0.0, 1.0)
    y = np.asarray(y, bool).ravel()
    q = np.clip(p, 1e-9, 1 - 1e-9)
    idx = np.clip(((np.log(q / (1 - q)) + LOGIT) / (2 * LOGIT) * bins).astype(np.int64), 0, bins - 1)
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
    edge = (bins - 1 - k) / bins * 2 * LOGIT - LOGIT                       # lower edge of the bin of the best threshold, in logit
    out["f1_max"], out["thr_f1"] = float(f1[k]), float(1.0 / (1.0 + np.exp(-edge)))
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
    NO_DIRECTION = ("prevalence", "thr_f1", "n", "n_pos")

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
                from evaluation.bev_velocity import object_cells      # THE object-cell rule (obj_fov x object layer x known velocity); imported here because bev_velocity imports this module
                out[f"{dk}|velocity"] = velocity_sums(o["vhat"], g["velocity"], object_cells(g, self.groups))
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
        last = key.rsplit("/", 1)[-1]
        return "low" if last in self.LOW else "none" if last in self.NO_DIRECTION else "high"


def exact_ap_auc(score: np.ndarray, y: np.ndarray):
    """the exact curves of the sorted scores (no ties assumed), for the selftest"""
    order = np.argsort(-score, kind="stable")
    ys = y[order].astype(np.float64)
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    ap = float(np.sum(ys * tp / (tp + fp)) / ys.sum())
    ranks = np.empty(len(score))
    ranks[np.argsort(score, kind="stable")] = np.arange(1, len(score) + 1)
    P, N = ys.sum(), len(y) - ys.sum()
    return ap, float((ranks[y.astype(bool)].sum() - P * (P + 1) / 2) / (P * N))


def selftest():
    rng = np.random.RandomState(0)
    for prev, shift in ((0.30, 1.5), (0.01, 3.0), (0.003, 4.0)):          # a balanced layer and two sparse ones whose probabilities sit low (focal-like)
        n = 400000
        y = rng.rand(n) < prev
        logit = rng.normal(-5.0 if prev < 0.05 else 0.0, 1.5, n) + shift * (y - 0.5) * 2
        p = 1.0 / (1.0 + np.exp(-logit))
        h = histogram(p, y)
        num = classification_numbers(h)
        ap, auc = exact_ap_auc(p, y)
        assert abs(num["ap"] - ap) < 0.004 and abs(num["roc_auc"] - auc) < 0.004, (prev, num["ap"], ap, num["roc_auc"], auc)
        assert abs(num["brier"] - float(np.mean((p - y) ** 2))) < 1e-9
        thr = num["thr_f1"]
        pr = (p >= thr) & True
        f1 = 2 * (pr & y).sum() / (pr.sum() + y.sum())
        assert abs(f1 - num["f1_max"]) < 0.01, (f1, num["f1_max"])      # the threshold of the best bin reproduces the best F1 up to the bin width
        half = histogram(p[: n // 2], y[: n // 2]) + histogram(p[n // 2:], y[n // 2:])
        assert np.array_equal(half, h) or np.allclose(half, h, rtol=0, atol=1e-6), "histograms are additive"
    one = classification_numbers(histogram(np.array([0.2, 0.9]), np.array([True, True])))
    assert np.isnan(one["ap"]) and np.isnan(one["roc_auc"]) and np.isfinite(one["ece"]) and np.isfinite(one["brier"])
    nan_h = histogram(np.array([np.nan, 0.5, 1.0]), np.array([True, False, True]))
    assert nan_h[0][0] == 1 and nan_h.sum() > 0, "NaN counts as p = 0"
    h16 = histogram(np.array([0.25, 0.75], np.float16), np.array([False, True]))
    assert abs(classification_numbers(h16)["roc_auc"] - 1.0) < 1e-12
    v = np.array([[1.0, 2.0], [0.0, 0.0]])
    s = velocity_sums(v + np.array([[0.3], [0.4]]), v, np.array([True, True]))      # an error vector (0.3, 0.4): length 0.5 in L2, 0.7 in L1
    r = regression_numbers(s[0])
    assert abs(r["epe"] - 0.5) < 1e-12 and abs(r["bias"] - 0.5) < 1e-12 and abs(r["rmse"] - 0.5) < 1e-12
    # the cells of the velocity numbers (frame()): a car of unknown velocity (GT 0 m/s, flow_weight 0) is no static ground truth. 8 x 8 world: a moving car (4 cells, 3 m/s),
    # a parked one (1 cell), a car of unknown velocity (1 cell), the field = GT + (0.3, 0.4) m/s -> obj = 5 cells (not 6), moving = 4, EPE 0.5 on both
    from evalsuite.contracts import FrameRef
    z = np.zeros((8, 8), np.float32)
    veh, gv, fw = z.copy(), np.zeros((2, 8, 8), np.float32), np.ones((8, 8), np.float32)
    veh[2:4, 2:4], veh[6, 6], veh[1, 7] = 1, 1, 1
    gv[0, 2:4, 2:4], fw[1, 7] = 3.0, 0
    g = {"camera": np.ones_like(z), "undefined": z, "vehicle": veh, "velocity": gv, "flow_weight": fw}
    m = BevCurveMetric([], ["vehicle"])
    c = m.frame(Sample(ref=FrameRef("gt", 0, "s"), inputs=None, reference={0.0: g}),
                Prediction(outputs={"dt": {0.0: {"prob": {}, "vhat": gv + np.array([0.3, 0.4], np.float32)[:, None, None]}}}))
    r = m.reduce([c])
    assert r["dt+0.00/velocity/obj/n"] == 5 and r["dt+0.00/velocity/moving/n"] == 4 and abs(r["dt+0.00/velocity/obj/epe"] - 0.5) < 1e-6, r
    print("bev_curves selftest ok: AP / AUC within 0.004 of the exact curves (prevalence 30 %, 1 %, 0.3 %), Brier exact, additive, one-class / NaN / float16 handled, "
          "cars of unknown velocity left out of the velocity cells")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
