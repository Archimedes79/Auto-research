"""t23: the tolerance-aware BCE on the line presence (losses.line_bce_loss, LossConfig.line_bce); CPU, seconds.

  1 a hand-made row: the loss equals the analytic weighted BCE (positives weight pos, far negatives 1, the band between 0; a negative at exactly `ignore` metres counts)
  2 gradient: a cell in the ignore band gets none, a positive's gradient is pos times a negative's at the same logit, a cell with map weight 0 gets none
  3 map_terms: line_bce = 0 (the default) adds no term and leaves the lines group as before; line_bce = k adds k x mean of the per-channel terms and the terms line_bce.<name>
  4 a channel without GT line cells is punished for confident cells far from a line (the Dice is blind there) and not for the cells in the band
Run (from the bev_fusion folder): wsl ~/miniconda3/envs/gcar/bin/python tests/test_line_bce.py
"""
import math
import os
import sys
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for sub in ("tests/support", "tests", "training"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import losses as L  # noqa: E402

torch.set_num_threads(2)
fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


# ---- 1 the analytic value
logit = torch.tensor([[[[0.5, -1.0, 2.0, 0.3, -0.2, 1.5]]]], dtype=torch.float64)            # (B=1, K=1, X=1, Y=6)
target = torch.tensor([[[[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]]], dtype=torch.float64)
dist = torch.tensor([[[[3.0, 1.5, 0.0, 0.5, 1.0, 2.0]]]], dtype=torch.float64)                 # cell 3 is in the band (0.5 m), cell 4 is exactly at the limit (a negative)
w = torch.ones(1, 1, 1, 6, dtype=torch.float64)
got = L.line_bce_loss(logit, target, dist, w, pos=3.0, ignore=1.0)
wts = np.array([1, 1, 3, 0, 1, 1], dtype=np.float64)
tg_ = target.numpy().ravel()
lg = logit.numpy().ravel()
bce = -(tg_ * np.log([sig(x) for x in lg]) + (1 - tg_) * np.log([1 - sig(x) for x in lg]))
expect = float((wts * bce).sum() / wts.sum())
check(got.shape == (1,) and abs(float(got[0]) - expect) < 1e-12, f"hand-made row: loss {float(got[0]):.9f} = analytic {expect:.9f}")
got_band = L.line_bce_loss(logit, target, dist, w, pos=3.0, ignore=0.4)                       # a smaller band: cell 3 (0.5 m) becomes a negative
wts2 = np.array([1, 1, 3, 1, 1, 1], dtype=np.float64)
expect2 = float((wts2 * bce).sum() / wts2.sum())
check(abs(float(got_band[0]) - expect2) < 1e-12, f"ignore 0.4 m: the cell at 0.5 m is a negative ({float(got_band[0]):.9f} = {expect2:.9f})")

# ---- 2 gradients
lg_t = logit.clone().requires_grad_(True)
L.line_bce_loss(lg_t, target, dist, w, pos=3.0, ignore=1.0).sum().backward()
g = lg_t.grad.numpy().ravel()
check(g[3] == 0.0, "the cell in the ignore band gets no gradient")
check(g[0] != 0 and g[4] != 0 and g[5] != 0, "far negatives and the negative at exactly the limit get a gradient")
same = torch.zeros(1, 1, 1, 2, dtype=torch.float64, requires_grad=True)                       # one positive and one negative cell in ONE channel, same logit 0: gradient (p - t) x weight / sum of weights
tg2 = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
d2 = torch.tensor([[[[0.0, 5.0]]]], dtype=torch.float64)
L.line_bce_loss(same, tg2, d2, torch.ones(1, 1, 1, 2, dtype=torch.float64), pos=3.0, ignore=1.0).sum().backward()
check(abs(float(same.grad[0, 0, 0, 0]) + 0.375) < 1e-12 and abs(float(same.grad[0, 0, 0, 1]) - 0.125) < 1e-12, "gradients (p - t) x w / sum w: -3 x 0.5 / 4 for the positive, +0.5 / 4 for the negative")
w0 = torch.ones(1, 1, 1, 6, dtype=torch.float64)
w0[..., 5] = 0.0
lg_w = logit.clone().requires_grad_(True)
L.line_bce_loss(lg_w, target, dist, w0, pos=3.0, ignore=1.0).sum().backward()
check(float(lg_w.grad[0, 0, 0, 5]) == 0.0, "a cell with map weight 0 gets no gradient")

# ---- 3 map_terms
spec = [("lane_divider", "line", "lane_divider"), ("road_divider", "line", "road_divider"),
        ("lane_divider_dist", "dist", "lane_divider_dist"), ("road_divider_dist", "dist", "road_divider_dist")]
torch.manual_seed(0)
X, Y = 8, 10
logits = torch.randn(1, 4, X, Y)
mask = (torch.rand(1, 2, X, Y) > 0.9).float()
mask[:, 1] = 0.0                                                                                # the road divider channel has no GT line
dist_gt = torch.rand(1, 2, X, Y) * 3.0
dist_gt = torch.where(mask > 0.5, torch.zeros_like(dist_gt), dist_gt.clamp(min=0.3))
T = {"lane_divider": mask[:, :1], "road_divider": mask[:, 1:], "lane_divider_dist": dist_gt[:, :1], "road_divider_dist": dist_gt[:, 1:]}
weights = {"line": torch.ones(1, 1, X, Y)}
cfg0 = L.LossConfig()
t0, g0 = L.map_terms(logits, T, spec, weights, cfg0)
t1, g1 = L.map_terms(logits, T, spec, weights, replace(cfg0, line_bce=0.0))
check(not any(k.startswith("line_bce") for k in t0) and float(g0["lines"]) == float(g1["lines"]), "line_bce = 0 (default): no line_bce term, the lines group unchanged")
cfg1 = replace(cfg0, line_bce=2.0, line_bce_pos=3.0, line_bce_ignore=1.0)
t2, g2 = L.map_terms(logits, T, spec, weights, cfg1)
lb = L.line_bce_loss(logits[:, :2], mask, dist_gt, weights["line"], 3.0, 1.0)
check(abs(float(g2["lines"]) - (float(g0["lines"]) + 2.0 * float(lb.mean()))) < 1e-6, f"line_bce = 2: lines group = old + 2 x mean(line_bce) ({float(g2['lines']):.6f})")
check(set(t2) - set(t0) == {"line_bce.lane_divider", "line_bce.road_divider"}, "the new terms are line_bce.<line name>")
check(all(float(t2[k]) == float(t0[k]) for k in t0), "every old term is unchanged")

# ---- 4 a channel without GT lines
logit_all = torch.full((1, 1, 1, 4), 3.0)                                                       # confident everywhere
tg0 = torch.zeros(1, 1, 1, 4)
d4 = torch.tensor([[[[0.2, 0.9, 1.0, 4.0]]]])
w4 = torch.ones(1, 1, 1, 4)
v_far = float(L.line_bce_loss(logit_all, tg0, d4, w4, 3.0, 1.0)[0])
exp_far = float(F.softplus(torch.tensor(3.0)))                                                  # BCE of logit 3 against target 0
check(abs(v_far - exp_far) < 1e-6, f"confident cells far from a line cost the full BCE ({v_far:.5f}); the two cells in the band are not counted")
band_only = float(L.line_bce_loss(logit_all[..., :2], tg0[..., :2], d4[..., :2], w4[..., :2], 3.0, 1.0)[0])
check(band_only == 0.0, "a channel whose cells are all in the band costs nothing (and does not divide by zero)")

print("FAILED: " + "; ".join(fails) if fails else "all ok")
sys.exit(1 if fails else 0)
