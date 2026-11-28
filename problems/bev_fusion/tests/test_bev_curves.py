"""Tests of evaluation/bev_curves.py (its former self-test).

    python -P -m bev_fusion.tests.test_bev_curves      (from the problem folder, with its module path)
"""
from __future__ import annotations

import numpy as np

from Evaluation import Prediction, Sample
from bev_fusion.evaluation.bev_curves import BevCurveMetric, classification_numbers, histogram, regression_numbers, velocity_sums


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
    # the cells of the velocity numbers (frame()): a car of unknown velocity (GT 0 m/s, flow_weight 0) is no static ground truth. 8 x 8 world: a moving car (4
    # cells, 3 m/s), a parked one (1 cell), a car of unknown velocity (1 cell), the field = GT + (0.3, 0.4) m/s -> obj = 5 cells (not 6), moving = 4, EPE 0.5 on
    # both
    from Evaluation.contracts import FrameRef
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
    selftest()
