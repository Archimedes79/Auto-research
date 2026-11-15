"""t23: the tolerance-aware BCE on the line presence (losses.line_bce_loss, LossConfig.line_bce, map_terms(..., line_bce=True)); CPU, seconds.

  1 a hand-made row: the loss equals the analytic weighted BCE (positives weight pos, far negatives 1, the band between 0; a negative at exactly `ignore` metres counts)
  2 gradient: a cell in the ignore band gets none, a positive's gradient is pos times a negative's at the same logit, a cell with map weight 0 gets none
  3 map_terms: line_bce = 0 (the default) adds no term and leaves the lines group exactly dice.mean() + huber.mean() (a reference expression); line_bce = k adds k x mean of the
    per-channel terms and the terms line_bce.<name> ONLY when the caller asks (BEVLoss asks for the main head, not for the 1 m centre-line head or the camera aux head)
  4 a channel without GT line cells is punished for confident cells far from a line (the Dice is blind there) and not for the cells in the band
  5 (critic round) a non-uniform map weight in numerator AND normaliser, K channels = K single-channel calls, B frames = frames side by side, the defaults, a soft target,
    the pairing of line and distance channels in every real layer spec, ignore above the GT distance cap refused
Run (from the bev_fusion folder): wsl ~/miniconda3/envs/gcar/bin/python tests/test_line_bce.py   (PYTHONPATH = the repository root for GtGenerator)
"""
import itertools
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
wl = torch.rand(1, 1, X, Y)
weights = {"line": wl}
cfg0 = L.LossConfig()
t0, g0 = L.map_terms(logits, T, spec, weights, cfg0, line_bce=True)
ref = L.tolerant_dice_loss(torch.sigmoid(logits[:, :2]), mask, wl, cfg0.line_tolerance).mean() + \
    L.dist_huber(logits[:, 2:], dist_gt, wl, cfg0.huber_delta, cfg0.dist_near, cfg0.dist_w_far, cfg0.dist_max).mean()
check(not any(k.startswith("line_bce") for k in t0) and torch.equal(g0["lines"], ref), "line_bce = 0 (default): no line_bce term, the lines group is exactly dice.mean() + huber.mean() (reference expression)")
cfg1 = replace(cfg0, line_bce=2.0, line_bce_pos=3.0, line_bce_ignore=1.0)
t2, g2 = L.map_terms(logits, T, spec, weights, cfg1, line_bce=True)
lb = L.line_bce_loss(logits[:, :2], mask, dist_gt, wl, 3.0, 1.0)
check(abs(float(g2["lines"]) - (float(ref) + 2.0 * float(lb.mean()))) < 1e-6, f"line_bce = 2: lines group = dice + huber + 2 x mean(line_bce) ({float(g2['lines']):.6f})")
check(set(t2) - set(t0) == {"line_bce.lane_divider", "line_bce.road_divider"}, "the new terms are line_bce.<line name>")
check(all(torch.equal(t2[k], t0[k]) for k in t0), "every old term is unchanged")
t3, g3 = L.map_terms(logits, T, spec, weights, cfg1)                                            # the caller does not ask (the 1 m centre-line head, the camera aux head)
check(not any(k.startswith("line_bce") for k in t3) and torch.equal(g3["lines"], g0["lines"]), "line_bce > 0 in the config but not asked by the caller: no term, the group unchanged (head isolation)")

# ---- 3b through BEVLoss.terms_at on the synthetic frame, with the 1 m centre-line head and the camera aux head present: the term reaches the main head ONLY
import synthetic_world as sw  # noqa: E402
from lift_pd import Grid  # noqa: E402
grid = Grid()
gX, gY = grid.nx, grid.ny
scene = L._demo_scene(sw)
gts = {0.0: [L.synthetic_gt(scene, grid)]}


def heads(lm):
    gen = torch.Generator().manual_seed(3)
    return {"sem": torch.randn(1, len(lm.spec), gX, gY, generator=gen), "sem_1m": torch.randn(1, len(lm.spec_1m), gX // 2, gY // 2, generator=gen),
            "cam_aux": torch.randn(1, len(lm.cam_spec), gX, gY, generator=gen), "radar_aux": torch.randn(1, len(lm.radar_spec), gX, gY, generator=gen)}


lm_off, lm_on = L.BEVLoss(L.LossConfig(centreline_1m=True), grid), L.BEVLoss(L.LossConfig(centreline_1m=True, line_bce=2.0), grid)
p_ = heads(lm_off)
t_off, g_off, f_off = lm_off.terms_at(p_, gts[0.0], 0.0)
t_on, g_on, f_on = lm_on.terms_at(p_, gts[0.0], 0.0)
new = sorted(set(t_on) - set(t_off))
check(new == ["sem.line_bce.lane_divider", "sem.line_bce.road_divider"], f"BEVLoss: the only new terms are the main head's line_bce terms ({new})")
check(all(torch.equal(t_on[k], t_off[k]) for k in t_off), "BEVLoss: every other term (1 m centre-line head, camera aux head, radar aux, areas, objects) is bit-equal with the option on")
check(float(g_on["lines"]) > float(g_off["lines"]) and all(torch.equal(g_on[k], g_off[k]) for k in g_off if k != "lines"), "BEVLoss: only the lines group grows")
check(torch.equal(f_on["cam_aux"], f_off["cam_aux"]) and torch.equal(f_on["radar_aux"], f_off["radar_aux"]), "BEVLoss: the camera-aux and radar-aux totals (fixed weights) are bit-equal with the option on")

# ---- 4 a channel without GT lines
logit_all = torch.full((1, 1, 1, 4), 3.0)                                                       # confident everywhere
tg0 = torch.zeros(1, 1, 1, 4)
d4 = torch.tensor([[[[0.2, 0.9, 1.0, 4.0]]]])
w4 = torch.ones(1, 1, 1, 4)
v_far = float(L.line_bce_loss(logit_all, tg0, d4.clamp(max=2.0), w4, 3.0, 1.0)[0])
exp_far = float(F.softplus(torch.tensor(3.0)))                                                  # BCE of logit 3 against target 0
check(abs(v_far - exp_far) < 1e-6, f"confident cells far from a line cost the full BCE ({v_far:.5f}); the two cells in the band are not counted")
band_only = float(L.line_bce_loss(logit_all[..., :2], tg0[..., :2], d4[..., :2], w4[..., :2], 3.0, 1.0)[0])
check(band_only == 0.0, "a channel whose cells are all in the band costs nothing (and does not divide by zero)")

# ---- 5 critic round
# E1 a non-uniform map weight: the value is the analytic weighted BCE (w in the numerator AND in the normaliser)
wv = np.array([1.0, 0.5, 2.0, 3.0, 0.25, 0.0], dtype=np.float64)
cw = np.array([1, 1, 3, 0, 1, 1], dtype=np.float64) * wv
got = L.line_bce_loss(logit, target, dist, torch.tensor(wv).view(1, 1, 1, 6), 3.0, 1.0)
check(abs(float(got[0]) - float((cw * bce).sum() / cw.sum())) < 1e-12, f"E1 non-uniform w: {float(got[0]):.9f} = analytic {float((cw * bce).sum() / cw.sum()):.9f}")
# E2 two frames, three channels: each channel alone, and the batch = the frames side by side (one normaliser over the whole batch)
torch.manual_seed(3)
B, K, X5, Y5 = 2, 3, 6, 7
lg2 = torch.randn(B, K, X5, Y5, dtype=torch.float64) * 2
tg3 = (torch.rand(B, K, X5, Y5) > 0.8).double()
tg3[:, 2] = 0
d3 = torch.where(tg3 > 0.5, torch.zeros_like(tg3), torch.rand(B, K, X5, Y5, dtype=torch.float64) * 2.0)
w3 = torch.rand(B, 1, X5, Y5, dtype=torch.float64)
w3[0, :, :2] = 0
full = L.line_bce_loss(lg2, tg3, d3, w3, 3.0, 1.0)
per = torch.cat([L.line_bce_loss(lg2[:, k:k + 1], tg3[:, k:k + 1], d3[:, k:k + 1], w3, 3.0, 1.0) for k in range(K)])
check(full.shape == (K,) and torch.allclose(full, per, atol=1e-12), "E2a K = 3 channels: each channel equals the K = 1 call on that channel alone")
side = L.line_bce_loss(torch.cat(list(lg2), 1)[None], torch.cat(list(tg3), 1)[None], torch.cat(list(d3), 1)[None], torch.cat(list(w3), 1)[None], 3.0, 1.0)
check(torch.allclose(full, side, atol=1e-12), "E2b B = 2 frames equal the frames side by side (one normaliser over the batch)")
# E3 the defaults (the card and the old runs depend on them)
check(cfg0.line_bce == 0.0 and cfg0.line_bce_pos == 3.0 and cfg0.line_bce_ignore == 1.0, "E3 defaults: line_bce 0.0 (off), pos 3.0, ignore 1.0 m")
# E4 the real layer specs: line channel j pairs with distance channel j, also next to the centre-line rows and the 1 m row
bad = []
for cl, dn, c1 in itertools.product((False, True), (False, True), (False, True)):
    for sp in (L.layer_spec(cl, dn), L.layer_spec_1m(c1)):
        ln = [(n, k) for n, kd, k in sp if kd == "line"]
        ds = [(n, k) for n, kd, k in sp if kd == "dist"]
        if not (len(ln) == len(ds) and all(a[0] + "_dist" == b[0] and a[1].replace("@1m", "") + "_dist" + ("@1m" if "@1m" in a[1] else "") == b[1] for a, b in zip(ln, ds))):
            bad.append((cl, dn, c1, ln, ds))
check(not bad, f"E4 line / distance channel pairing in every real spec variant (8 x 2){': ' + str(bad[0]) if bad else ''}")
# E5 a soft target (not in the GT today): > 0.5 is the positive, <= 0.5 with a far distance the negative, a soft cell near a line has weight 0
lg5 = torch.tensor([[[[2.0, -1.0, 0.5, -0.5]]]], dtype=torch.float64)
t5 = torch.tensor([[[[0.3, 0.7, 0.3, 0.0]]]], dtype=torch.float64)
d5 = torch.tensor([[[[0.0, 0.0, 1.5, 1.5]]]], dtype=torch.float64)
w5 = torch.ones(1, 1, 1, 4, dtype=torch.float64)


def bce5(x, t):
    return -(t * math.log(sig(x)) + (1 - t) * math.log(1 - sig(x)))


e5 = (bce5(-1.0, 0.7) * 3 + bce5(0.5, 0.3) + bce5(-0.5, 0.0)) / 5.0
check(abs(float(L.line_bce_loss(lg5, t5, d5, w5, 3.0, 1.0)[0]) - e5) < 1e-12, "E5 soft target: 0.7 is a positive (x3), 0.3 near a line is ignored, 0.3 / 0 far away are negatives")
# E6 an ignore band beyond the truncated GT distance would leave no negative at all
try:
    L.line_bce_loss(logit, target, dist, w, 3.0, L.DIST_MAX + 0.5)
    check(False, "E6 ignore above the GT distance cap is refused")
except AssertionError:
    check(True, "E6 ignore above the GT distance cap is refused")

print("FAILED: " + "; ".join(fails) if fails else "all ok")
sys.exit(1 if fails else 0)
