"""Regression test of grid_eval against the numbers that dense_head.py stored when the dense head e37_final was evaluated (runs/e37_final/metrics.json, validation rows):
the full validation prediction maps are re-scored and must reproduce the stored P/R/F1 (and 1:1 F1) at every tested threshold.
Run: scripts/_py14.sh ../evaluation/lane_grid/selftest.py"""
import json

import numpy as np
import torch

import grid_eval as GR

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
prob = torch.from_numpy(np.load(f"{R}/runs/e37_final/prob_val.npy"))
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
