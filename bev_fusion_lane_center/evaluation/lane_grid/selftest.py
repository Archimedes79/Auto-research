"""Regression test of grid_eval against the numbers that dense_head.py stored when the dense head e37_final was evaluated (runs/e37_final/metrics.json, validation rows):
the full validation prediction maps are re-scored and must reproduce the stored P/R/F1 (and 1:1 F1) at every tested threshold.
Run: scripts/_py14.sh ../evaluation/lane_grid/selftest.py"""
import bflc_paths as BP  # project paths from the location of this checkout
import json

import numpy as np
import torch

import grid_eval as GR

R = BP.ROOT


def spec_hysteresis_with_range():
    """skeletons with low and rg both set (review bug L3, D24): every seed pixel (D <= thr * (1 + rg r / 48)) survives the hysteresis, weak pixels
    (seed threshold < D <= low) survive only when connected to a seed. Synthetic distance field on the CPU."""
    D = np.full((1, 200, 200), 10.0, np.float32)
    D[0, 195:198, 180:199] = 1.0  # far (r > 60 m): a seed at rg = 1 (threshold > 1.1) but above low = 0.8
    D[0, 100:103, 100:111] = 0.3  # near seed
    D[0, 100:103, 111:121] = 0.7  # weak, connected to the near seed (threshold there < 0.61)
    D[0, 100:103, 80:91] = 0.7  # weak, isolated
    old, GR.dev = GR.dev, "cpu"
    try:
        sk = GR.skeletons(D, thr=0.5, low=0.8, rg=1.0)[0]
        sk0 = GR.skeletons(D, thr=0.5, low=0.8, rg=0.0)[0]
    finally:
        GR.dev = old
    assert sk[195:198, 180:199].any(), "far seed dropped by the hysteresis"
    assert sk[100:103, 111:121].any() and sk[100:103, 100:111].any(), "connected weak pixels not grown"
    assert not sk[100:103, 80:91].any(), "isolated weak pixels kept"
    assert not sk0[195:198, 180:199].any() and sk0[100:103, 111:121].any(), "rg = 0: far line is no seed, the weak run still grows"
    print("skeletons hysteresis + range threshold spec passed")


spec_hysteresis_with_range()
prob =torch.from_numpy(np.load(f"{R}/runs/e37_final/prob_val.npy"))
ids = np.load(f"{R}/runs/e37_final/ids_val.npy")
stored = json.load(open(f"{R}/runs/e37_final/metrics.json"))["val"]
taus = (0.3, 0.5, 0.7)
new = GR.eval_probs(prob, GR.load_gt_points("b123", "val", ids), taus=taus)
worst = 0.0
for t in taus:
    for k, v in new[str(t)].items():
        worst = max(worst, abs(v - stored[str(t)][k]))
print(f"{len(ids)} validation rows, thresholds {taus}: largest deviation from the stored metrics {worst:.2e}")
print("F1@1.0 at tau 0.5:", round(new["0.5"]["F1@1.0"], 4), "| 1:1 F1:", round(new["0.5"]["F1o2o@1.0"], 4))
assert worst < 1e-6
D = (2.0 * (1.0 - prob[:48].float())).numpy()  # synthetic distance field in metres
est = GR.est_eval(D, GR.load_gt_points("b123", "val", ids[:48]), thr=0.5)
print("est_eval on a synthetic distance field:", {k: tuple(round(x, 3) for x in v) for k, v in est.items()})
print("lane_grid selftest passed")
