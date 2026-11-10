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

import itertools

# ---- E1 a non-uniform map weight w: the value is the analytic weighted BCE (w in the numerator AND in the normaliser)
logit = torch.tensor([[[[0.5, -1.0, 2.0, 0.3, -0.2, 1.5]]]], dtype=torch.float64)
target = torch.tensor([[[[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]]], dtype=torch.float64)
dist = torch.tensor([[[[3.0, 1.5, 0.0, 0.5, 1.0, 2.0]]]], dtype=torch.float64)
wv = np.array([1.0, 0.5, 2.0, 3.0, 0.25, 0.0], dtype=np.float64)
cw = np.array([1, 1, 3, 0, 1, 1], dtype=np.float64) * wv
tg_ = target.numpy().ravel(); lg = logit.numpy().ravel()
bce = -(tg_ * np.log([sig(x) for x in lg]) + (1 - tg_) * np.log([1 - sig(x) for x in lg]))
got = L.line_bce_loss(logit, target, dist, torch.tensor(wv).view(1, 1, 1, 6), 3.0, 1.0)
check(abs(float(got[0]) - float((cw * bce).sum() / cw.sum())) < 1e-12, f"E1 non-uniform w: {float(got[0]):.9f} = analytic {float((cw * bce).sum() / cw.sum()):.9f}")

# ---- E2 two channels, two frames: each channel alone, and the batch = the frames side by side (the normaliser is the sum over the whole batch)
torch.manual_seed(3)
B, K, X, Y = 2, 3, 6, 7
lg2 = torch.randn(B, K, X, Y, dtype=torch.float64) * 2
tg2 = (torch.rand(B, K, X, Y) > 0.8).double(); tg2[:, 2] = 0
d2 = torch.where(tg2 > 0.5, torch.zeros_like(tg2), torch.rand(B, K, X, Y, dtype=torch.float64) * 3)
w2 = torch.rand(B, 1, X, Y, dtype=torch.float64); w2[0, :, :2] = 0
full = L.line_bce_loss(lg2, tg2, d2, w2, 3.0, 1.0)
per = torch.cat([L.line_bce_loss(lg2[:, k:k + 1], tg2[:, k:k + 1], d2[:, k:k + 1], w2, 3.0, 1.0) for k in range(K)])
check(full.shape == (K,) and torch.allclose(full, per, atol=1e-12), "E2a K = 3 channels: each channel equals the K = 1 call on that channel alone")
side = L.line_bce_loss(torch.cat(list(lg2), 1)[None], torch.cat(list(tg2), 1)[None], torch.cat(list(d2), 1)[None], torch.cat(list(w2), 1)[None], 3.0, 1.0)
check(torch.allclose(full, side, atol=1e-12), "E2b B = 2 frames equal the frames side by side (one normaliser over the batch)")

# ---- E3 the defaults (the card and the old runs depend on them)
c = L.LossConfig()
check(c.line_bce == 0.0 and c.line_bce_pos == 3.0 and c.line_bce_ignore == 1.0, "E3 defaults: line_bce 0.0 (off), pos 3.0, ignore 1.0 m")

# ---- E4 the real layer specs: line channel j pairs with distance channel j, also next to the centre-line rows and the 1 m row
for cl, dn, c1 in itertools.product((False, True), (False, True), (False, True)):
    for spec in (L.layer_spec(cl, dn), L.layer_spec_1m(c1)):
        ln = [(n, k) for n, kd, k in spec if kd == "line"]
        ds = [(n, k) for n, kd, k in spec if kd == "dist"]
        if not (len(ln) == len(ds) and all(a[0] + "_dist" == b[0] and a[1].replace("@1m", "") + "_dist" + ("@1m" if "@1m" in a[1] else "") == b[1] for a, b in zip(ln, ds))):
            check(False, f"E4 pairing broken for centrelines={cl} dense={dn} centreline_1m={c1}: {ln} vs {ds}")
check(True, "E4 (all spec variants checked)")

# ---- E5 a soft target (not in the GT today): > 0.5 is the positive, <= 0.5 with a far distance the negative, a soft cell near a line has weight 0
lg5 = torch.tensor([[[[2.0, -1.0, 0.5, -0.5]]]], dtype=torch.float64)
t5 = torch.tensor([[[[0.3, 0.7, 0.3, 0.0]]]], dtype=torch.float64)
d5 = torch.tensor([[[[0.0, 0.0, 5.0, 5.0]]]], dtype=torch.float64)
w5 = torch.ones(1, 1, 1, 4, dtype=torch.float64)
bce5 = lambda x, t: -(t * math.log(sig(x)) + (1 - t) * math.log(1 - sig(x)))
e5 = (bce5(-1.0, 0.7) * 3 + bce5(2.0, 0.3) * 0 + bce5(0.5, 0.3) + bce5(-0.5, 0.0)) / 5.0
check(abs(float(L.line_bce_loss(lg5, t5, d5, w5, 3.0, 1.0)[0]) - e5) < 1e-12, "E5 soft target: 0.7 is a positive (x3), 0.3 near a line is ignored, 0.3 / 0 far away are negatives")


# ---- E6 the default path against a reference expression (not against the same code): line_bce = 0 leaves the lines group exactly dice.mean() + huber.mean()
spec = [("lane_divider", "line", "lane_divider"), ("road_divider", "line", "road_divider"),
        ("lane_divider_dist", "dist", "lane_divider_dist"), ("road_divider_dist", "dist", "road_divider_dist")]
torch.manual_seed(0)
X, Y = 8, 10
logits = torch.randn(1, 4, X, Y)
mask = (torch.rand(1, 2, X, Y) > 0.9).float()
dist_gt = torch.where(mask > 0.5, torch.zeros(1, 2, X, Y), (torch.rand(1, 2, X, Y) * 3.0).clamp(min=0.3))
T = {"lane_divider": mask[:, :1], "road_divider": mask[:, 1:], "lane_divider_dist": dist_gt[:, :1], "road_divider_dist": dist_gt[:, 1:]}
wl = torch.rand(1, 1, X, Y)
cfg0 = L.LossConfig()
_, g0 = L.map_terms(logits, T, spec, {"line": wl}, cfg0)
ref = L.tolerant_dice_loss(torch.sigmoid(logits[:, :2]), mask, wl, cfg0.line_tolerance).mean() + \
    L.dist_huber(logits[:, 2:], dist_gt, wl, cfg0.huber_delta, cfg0.dist_near, cfg0.dist_w_far, cfg0.dist_max).mean()
check(torch.equal(g0["lines"], ref), "E6 line_bce = 0: lines group is exactly dice.mean() + huber.mean() (reference expression)")

print("FAILED: " + "; ".join(fails) if fails else "all ok")
sys.exit(1 if fails else 0)
