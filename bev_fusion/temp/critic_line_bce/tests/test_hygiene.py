"""LossConfig.hygiene_2026_11_02 and the splat precision (records/open_findings_2026-11-02.md #1 - #4, #6; lead's order 2026-11-02); CPU, seconds, toy tensors.

  1. neutral inputs (every weight 1, every velocity known): the flag changes nothing, bit for bit (object_head_terms, MotionHead.loss, motion_term)
  2. #3 w_obj is a WEIGHT: the regression terms of the object head are weighted means over the objects (checked against single-object runs), the focal loss weights every cell (a cell of weight 0.5
     gets half the gradient, the others the same), the motion L1 is a weighted mean; the flag off keeps the plain means
  3. #2 an object of unknown velocity (vel_known False, velocity target 0) is out of the moving / static BCE and of the velocity L1, and its logit gets no gradient; off, it trains "static"
  4. #4 the dead `radar_slot` branch of BEVLoss is gone (an output of that name adds no term); a stored config with w_radar_slot still loads
  5. the wiring: BEVLoss(LossConfig(hygiene_2026_11_02=True)) reaches the object head, the motion loss and train.forward_loss (batch_ego); the flag is off by default
  6. #6 splat_positions / splat_bev / splat_window compute geometry in float32 for bf16 / fp16 inputs (the bf16 cell centres were 0.125 m off at 32 - 64 m); float32 input is unchanged
  7. #1 batch_ego raises for a dt other than +-0.5 s when strict, and not otherwise
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_hygiene.py   (PYTHONPATH = the repository root for autogt)
"""
import math
import sys

import torch

sys.path.insert(0, "training")
sys.path.insert(0, "tests/support")
import losses as L  # noqa: E402
import splat as S  # noqa: E402
import timed_head as TH  # noqa: E402
import train as T  # noqa: E402
from lift_pd import Grid  # noqa: E402

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


grid = Grid()
X, Y = grid.nx, grid.ny
G = L.NUSC_GROUPS


def cell(cx, cy):
    return int((cx - grid.x_min) // grid.res), int((cy - grid.y_min) // grid.res)


def box(group, cx, cy, vel=(0.0, 0.0), known=True, observed=True, yaw=0.3, size=(1.9, 4.5, 1.6)):
    return {"group": group, "center": (cx, cy, 0.0), "size": size if group != "pedestrian" else (0.7, 0.7, 1.7), "yaw": yaw, "vel": vel, "observed": observed, "vel_known": known}


def make_out(B=1, mov=False, seed=0, grad=True):
    g = torch.Generator().manual_seed(seed)
    out = {"hm": torch.randn(B, 3, X, Y, generator=g), "off": torch.randn(B, 2, X, Y, generator=g), "size": torch.randn(B, 2, X, Y, generator=g),
           "rot": torch.randn(B, 2, X, Y, generator=g), "rad": torch.randn(B, 1, X, Y, generator=g), "vel": torch.randn(B, 2, X, Y, generator=g)}
    if mov:
        out["mov"] = torch.randn(B, 1, X, Y, generator=g)
    if grad:
        out = {k: v.clone().requires_grad_(True) for k, v in out.items()}
    return out


def wmap(B=1, cells=None):
    w = torch.ones(B, 1, X, Y)
    for (b, ix, iy), v in (cells or {}).items():
        w[b, 0, ix, iy] = v
    return w


def terms(out, boxes, w_obj, hygiene, **kw):
    lo, _ = L.object_head_terms({k: v for k, v in out.items()}, boxes, w_obj, grid, G, hygiene=hygiene, **kw)
    return lo


# ---------------------------------------------------------------- 1 neutral inputs
boxes1 = [[box("vehicle", 20.0, -3.0, (3.0, 0.0)), box("vehicle", 30.0, 4.0), box("pedestrian", 12.0, 8.0, (0.6, 0.8))]]
for with_mov in (False, True):
    out = make_out(mov=with_mov)
    a, b = terms(out, boxes1, wmap(), False), terms(out, boxes1, wmap(), True)
    check(set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a), f"object_head_terms: flag on with neutral inputs = off, bit for bit (mov head: {with_mov}; terms {sorted(a)})")
torch.manual_seed(0)
v_hat, v_gt, objm = torch.randn(1, 2, 40, 1), torch.randn(1, 2, 40, 1) * 3, (torch.rand(1, 1, 40, 1) > 0.5).float()
plain = TH.MotionHead.loss(v_hat, v_gt, objm, 1.0, 0.1)
m_ = objm
vg = torch.where(v_gt.norm(dim=1, keepdim=True) < 1.0, torch.zeros_like(v_gt), v_gt)
by_hand = ((v_hat - vg).abs() * m_).sum() / (m_.sum() * 2).clamp(min=1) + 0.1 * (v_hat.abs() * (1 - m_)).sum() / ((1 - m_).sum() * 2).clamp(min=1)
check(torch.equal(plain, by_hand) and torch.equal(plain, TH.MotionHead.loss(v_hat, v_gt, objm, 1.0, 0.1, weight=None)), "MotionHead.loss without weight: the plain means (against the formula by hand)")
check(torch.equal(plain, TH.MotionHead.loss(v_hat, v_gt, objm, 1.0, 0.1, weight=torch.ones(1, 1, 40, 1))), "MotionHead.loss with weight 1 everywhere = without, bit for bit")

# ---------------------------------------------------------------- 2 weights
specs = [("vehicle", 20.0, -3.0, (3.0, 0.0)), ("vehicle", 30.0, 4.0, (0.0, 2.0)), ("vehicle", 40.0, -8.0, (1.0, 1.0))]
ws = [1.0, 0.5, 0.5]
bx3 = [[box(g_, cx, cy, v_) for g_, cx, cy, v_ in specs]]
cells3 = {(0, *cell(cx, cy)): w for (g_, cx, cy, v_), w in zip(specs, ws)}
out = make_out()
on = terms(out, bx3, wmap(cells=cells3), True)
singles = [terms(out, [[box(g_, cx, cy, v_)]], wmap(), False) for g_, cx, cy, v_ in specs]
ok = True
for k in ("off", "size", "rot", "vel"):
    hand = sum(w * s[k] for w, s in zip(ws, singles)) / sum(ws)
    ok &= bool(torch.allclose(on[k], hand, rtol=1e-5, atol=1e-7))
check(ok, "hygiene on: off / size / rot / vel are the weighted means of the single-object terms (weights 1, 0.5, 0.5)")
off_ = terms(out, bx3, wmap(cells=cells3), False)
check(all(torch.equal(off_[k], terms(out, bx3, wmap(), False)[k]) for k in off_), "hygiene off: the value of w_obj changes nothing (only its sign decides), as before")
check(not torch.allclose(on["off"], off_["off"]), "hygiene on differs from off when the weights differ")
# focal loss: a region of weight 0.5 gets half the gradient, the rest is unchanged
region = (slice(90, 110), slice(60, 90))                                   # away from all three boxes: no positive cell is down-weighted, the normaliser stays
W2 = wmap()
W2[0, 0][region] = 0.5
grads = []
for W in (wmap(), W2):
    o = make_out()
    terms(o, bx3, W, True)["hm"].backward()
    grads.append(o["hm"].grad.clone())
inside = grads[0][0, :, region[0], region[1]]
check(torch.allclose(grads[1][0, :, region[0], region[1]], 0.5 * inside, rtol=1e-4, atol=1e-9) and inside.abs().max() > 0, "focal loss: cells of weight 0.5 get half the gradient")
rest = torch.ones(X, Y, dtype=torch.bool)
rest[region] = False
check(torch.allclose(grads[1][0][:, rest], grads[0][0][:, rest], rtol=1e-5, atol=1e-9), "focal loss: the other cells get the same gradient")
# the weights are RELATIVE: uniform 0.5 over three objects leaves every term as it is (the normalisers are weighted sums); one object alone of weight 0.5 is down-weighted, because the
# normaliser clamps at 1 (a batch whose objects are all down-weighted equally is not down-weighted unless their total weight is below 1)
half = terms(out, bx3, wmap() * 0.5, True)
full = terms(out, bx3, wmap(), True)
check(all(torch.allclose(half[k], full[k], rtol=1e-4) for k in ("hm", "off", "size", "rot", "vel")), "uniform weight 0.5 over three objects: every term as with weight 1 (the weights are relative)")
one_full, one_half = terms(out, [[box(*specs[0][:3], specs[0][3])]], wmap(), True), terms(out, [[box(*specs[0][:3], specs[0][3])]], wmap() * 0.5, True)
check(torch.allclose(one_half["off"], 0.5 * one_full["off"], rtol=1e-5), "one object of weight 0.5 alone: its terms are halved (the normaliser clamps at 1; documented in LossConfig.hygiene_2026_11_02)")


def focal_ref(logits, target, w):
    """the weighted CenterNet focal loss, written out again (an independent reference for object_head.focal_loss)"""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = (target == 1).float()
    pos_l = -torch.log(p) * (1 - p) ** 2 * pos
    neg_l = -torch.log(1 - p) * p ** 2 * (1 - target) ** 4 * (1 - pos)
    return ((pos_l * w).sum() + (neg_l * w).sum()) / (pos * w).sum().clamp(min=1)


torch.manual_seed(1)
lg = torch.randn(2, 3, 20, 20)
tg = torch.rand(2, 3, 20, 20) * 0.9
tg[0, 0, 3, 3] = tg[0, 1, 10, 7] = tg[1, 2, 15, 15] = 1.0                    # three peaks
ww = torch.rand(2, 1, 20, 20) + 0.2
ww[0, 0, 3, 3] = 0.5                                                          # a down-weighted POSITIVE
import object_head as oh  # noqa: E402
check(torch.allclose(oh.focal_loss(lg, tg, ww), focal_ref(lg, tg, ww.expand_as(lg)), rtol=1e-6) and torch.equal(oh.focal_loss(lg, tg), oh.focal_loss(lg, tg, None)), "focal_loss(weight=): the weighted loss against its reference, positives included; no weight = the plain loss")

# every regression term, also dir and rad, is a weighted mean of the single-object terms
mixed = [("vehicle", 20.0, -3.0, (3.0, 0.0)), ("pedestrian", 12.0, 8.0, (0.6, 0.8)), ("vehicle", 40.0, -8.0, (1.0, 1.0)), ("pedestrian", 33.0, 2.0, (0.0, 1.0))]
wm = [1.0, 0.5, 0.5, 1.0]                                                    # vehicles 1.0 and 0.5, pedestrians 0.5 and 1.0: unequal within each class
o_dir = make_out()
o_dir["dir"] = torch.randn(1, 1, X, Y, requires_grad=True)
bxm = [[box(g_, cx, cy, v_) for g_, cx, cy, v_ in mixed]]
cellsm = {(0, *cell(cx, cy)): w for (g_, cx, cy, v_), w in zip(mixed, wm)}
onm = terms(o_dir, bxm, wmap(cells=cellsm), True)
sing = [terms(o_dir, [[box(g_, cx, cy, v_)]], wmap(), False) for g_, cx, cy, v_ in mixed]
veh, ped = [0, 2], [1, 3]
hand_dir = sum(wm[i] * sing[i]["dir"] for i in veh) / sum(wm[i] for i in veh)
hand_rad = sum(wm[i] * sing[i]["rad"] for i in ped) / sum(wm[i] for i in ped)
check(torch.allclose(onm["dir"], hand_dir, rtol=1e-5) and torch.allclose(onm["rad"], hand_rad, rtol=1e-5), "the direction (vehicles) and radius (pedestrians) terms are weighted means too")

# build_targets: the keys of the flag, and what a box without them means
tb = oh.build_targets([[{"group": "vehicle", "cx": 20.0, "cy": 0.0, "vx": 0.0, "vy": 0.0, "length": 4.5, "width": 1.9, "yaw": 0.1, "vel_known": False, "loss_w": 0.3},
                        {"group": "vehicle", "cx": 30.0, "cy": 0.0, "vx": 0.0, "vy": 0.0, "length": 4.5, "width": 1.9, "yaw": 0.1},
                        {"group": "vehicle", "cx": 40.0, "cy": 0.0, "vx": 0.0, "vy": 0.0, "length": 4.5, "width": 1.9, "yaw": 0.1, "vel_known": True, "loss_w": 0.7}]], grid, G, 8)
tp = oh.build_targets([[{"group": "vehicle", "cx": 20.0, "cy": 0.0, "vx": 0.0, "vy": 0.0, "length": 4.5, "width": 1.9, "yaw": 0.1}]], grid, G, 8)
check(tb["known"][0, :3].tolist() == [0.0, 1.0, 1.0] and [round(x, 6) for x in tb["mask"][0, :3].tolist()] == [0.3, 1.0, 0.7] and "known" not in tp and tp["mask"][0, 0].item() == 1.0,
      "build_targets: vel_known False -> 0, a box without the key -> 1, loss_w -> mask; no vel_known anywhere = no `known` key and mask 1")

# batch_ego strict: each side and each size
gs2 = [{"prev": [(0.48, 8.0, 0.1)], "next": [(0.52, 8.5, 0.0)]}]
bad_dts = []
for dts_ in ((-1.0,), (-0.1,), (0.1,), (1.0,)):
    try:
        T.batch_ego(gs2, dts_, "cpu", strict=True)
        bad_dts.append(dts_)
    except ValueError:
        pass
check(not bad_dts and set(T.batch_ego(gs2, (-0.5, 0.5), "cpu", strict=True)) == {-0.5, 0.5}, f"batch_ego strict: -1.0, -0.1, +0.1, +1.0 raise, -0.5 / +0.5 do not (not raised: {bad_dts})")

# the motion loss leaves the cells of an unknown-velocity box out (flow_weight 0)
gt_fw = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in L.synthetic_gt(L._demo_scene(__import__("synthetic_world")), grid).items()}
import numpy as np  # noqa: E402
unk = np.asarray(gt_fw["instance"]) == 0
fw = np.ones_like(np.asarray(gt_fw["instance"]), dtype=np.float32)
fw[unk] = 0.0
gt_fw["flow_weight"] = fw
cfg_off, cfg_on = L.LossConfig(), L.LossConfig(hygiene_2026_11_02=True)
T_off, T_on = L.build_layer_targets([gt_fw], cfg_off), L.build_layer_targets([gt_fw], cfg_on)
check("flow_weight" not in T_off and "flow_weight" in T_on, "build_layer_targets: the flow_weight layer is carried only with the flag")
gr = []
for cfg_, T_ in ((cfg_off, T_off), (cfg_on, T_on)):
    vh_ = torch.randn(1, 2, X, Y, generator=torch.Generator().manual_seed(3), requires_grad=True)
    L.motion_term(vh_, T_, cfg_).backward()
    gr.append(vh_.grad[0].abs().sum(0)[torch.as_tensor(unk)])
check(unk.any() and gr[0].min() > 0 and float(gr[1].abs().max()) == 0.0, f"motion_term: the cells of the unknown-velocity box ({int(unk.sum())}) get gradient without the flag and none with it")
# motion L1, weighted
torch.manual_seed(2)
wv = torch.rand(1, 1, 40, 1) + 0.2
wt = TH.MotionHead.loss(v_hat, v_gt, objm, 1.0, 0.1, weight=wv)
hand = ((v_hat - vg).abs() * m_ * wv).sum() / (m_ * wv * 2).sum().clamp(min=1) + 0.1 * (v_hat.abs() * (1 - m_) * wv).sum() / ((1 - m_) * wv * 2).sum().clamp(min=1)
check(torch.allclose(wt, hand, rtol=1e-6) and not torch.allclose(wt, plain), "MotionHead.loss with a weight: the weighted means by hand")
torch.manual_seed(3)
Tm = {"w_obj": wmap() * torch.rand(1, 1, X, Y).clamp(min=0.3), "velocity": torch.randn(1, 2, X, Y) * 3, "obj_cells": (torch.rand(1, 1, X, Y) > 0.7).float()}
vh = torch.randn(1, 2, X, Y)
off_m, on_m = L.motion_term(vh, Tm, L.LossConfig()), L.motion_term(vh, Tm, L.LossConfig(hygiene_2026_11_02=True))
Tn = dict(Tm, w_obj=torch.ones(1, 1, X, Y))
check(not torch.allclose(off_m, on_m) and torch.equal(L.motion_term(vh, Tn, L.LossConfig()), L.motion_term(vh, Tn, L.LossConfig(hygiene_2026_11_02=True))),
      "motion_term: the flag weights the cells by w_obj; with w_obj = 1 it changes nothing")

# ---------------------------------------------------------------- 3 unknown velocity
A, B_, C = box("vehicle", 20.0, -3.0, (3.0, 0.0)), box("vehicle", 30.0, 4.0, (0.0, 0.0), known=False), box("vehicle", 40.0, -8.0, (0.0, 0.0))
out = make_out(mov=True)
no_b = terms(out, [[A, C]], wmap(), False)
with_b_on, with_b_off = terms(out, [[A, B_, C]], wmap(), True), terms(out, [[A, B_, C]], wmap(), False)
check(torch.allclose(with_b_on["mov"], no_b["mov"], rtol=1e-6) and not torch.allclose(with_b_off["mov"], no_b["mov"]), "mov BCE: the unknown-velocity object is left out when the flag is on, a 'static' label when off")
check(torch.allclose(with_b_on["vel"], no_b["vel"], rtol=1e-6), "velocity L1 (hurdle): the unknown object is not in it")
cb = cell(30.0, 4.0)
grads = []
for flag in (True, False):
    o = make_out(mov=True)
    terms(o, [[A, B_, C]], wmap(), flag)["mov"].backward()
    grads.append(o["mov"].grad[0, 0, cb[0], cb[1]].item())
check(grads[0] == 0.0 and grads[1] != 0.0, f"the unknown object's mov logit: gradient {grads[0]} with the flag, {grads[1]:.3e} without")
out_nomov = make_out(mov=False)
v_on, v_off, v_ref = terms(out_nomov, [[A, B_, C]], wmap(), True)["vel"], terms(out_nomov, [[A, B_, C]], wmap(), False)["vel"], terms(out_nomov, [[A, C]], wmap(), False)["vel"]
check(torch.allclose(v_on, v_ref, rtol=1e-6) and not torch.allclose(v_off, v_ref), "no moving head: the velocity L1 leaves the unknown object out too (it would read its 0 as a target)")

# ---------------------------------------------------------------- 4 the dead radar_slot / old configs
import synthetic_world as sw  # noqa: E402
scene = L._demo_scene(sw)
gt = L.synthetic_gt(scene, grid)
check(L.LossConfig().hygiene_2026_11_02 is False and L.LossConfig(**{"w_radar_slot": 0.1, "w_obj_vel": 1.0}).w_radar_slot == 0.1, "the flag is off by default; a stored config with w_radar_slot still loads")
o = make_out(mov=False, grad=False)
for flag in (False, True):
    bl = L.BEVLoss(L.LossConfig(hygiene_2026_11_02=flag), grid)
    t_, g_, f_ = bl.terms_at({"obj": dict(o), "radar_slot": torch.tensor(1.0)}, [gt], 0.0)
    check("radar_slot" not in t_ and "radar_slot" not in f_, f"BEVLoss: an output called radar_slot adds no term (flag {flag})")

# ---------------------------------------------------------------- 5 the wiring through BEVLoss
gt2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in gt.items()}
bxs = [dict(b, vel_known=True) for b in gt2["meta"]["boxes"]]
bxs[0]["vel_known"] = False
gt2["meta"] = dict(gt2["meta"], boxes=bxs)
interp = torch.zeros(X, Y)
interp[: X // 2] = 1.0                                                       # the near half is an interpolated label: w_interpolated 0.5
gt2["interpolated"] = interp.numpy()
o = make_out(mov=True, grad=False)
res = {}
for name, g_, flag in (("neutral_off", gt, False), ("neutral_on", gt, True), ("mixed_off", gt2, False), ("mixed_on", gt2, True)):
    res[name] = L.BEVLoss(L.LossConfig(hygiene_2026_11_02=flag), grid).terms_at({"obj": dict(o)}, [g_], 0.0)[0]
same = lambda a, b: set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)      # noqa: E731
check(same(res["neutral_off"], res["neutral_on"]), "BEVLoss.terms_at: the flag with a neutral GT = off, bit for bit")
check(not torch.allclose(res["mixed_on"]["objhead.hm"], res["mixed_off"]["objhead.hm"]) and not torch.allclose(res["mixed_on"]["objhead.mov"], res["mixed_off"]["objhead.mov"]),
      "BEVLoss.terms_at: with interpolated cells and an unknown-velocity box the flag reaches the object head (hm, mov)")
res_flag = {}
for flag in (False, True):
    res_flag[flag] = L.BEVLoss(L.LossConfig(hygiene_2026_11_02=flag), grid).terms_at({"v_hat": torch.zeros(1, 2, X, Y), "obj": dict(o)}, [gt2], 0.0)
check(res_flag[True][0]["objhead.total"] is not None, "(terms_at runs with a v_hat entry)")


class Reached(Exception):
    pass


class FakeModel:
    def __call__(self, batch, dts=None, **kw):
        raise Reached()


class FakeLoss:
    def __init__(self, flag):
        self.cfg = L.LossConfig(hygiene_2026_11_02=flag)


steps = [(0.5, 8.0, 0.1)]
batch = {"gt": [{"cur": 1, "prev": 1, "next": 1}], "gap_steps": [{"prev": steps, "next": steps}]}
keep = {k: getattr(T, k) for k in ("sample_targets", "supports_ego", "to_device", "check_inputs")}
T.sample_targets = lambda gt_: {-0.5: [1], 0.0: [1], 1.0: [1]}                # a request for dt = +1.0 s, which the gap_steps cannot serve
T.supports_ego, T.to_device, T.check_inputs = (lambda m: True), (lambda b, d, non_blocking=False: b), (lambda *a, **k: None)
already = bool(T._CONTRACT_CHECKED)
T._CONTRACT_CHECKED.append(True)
outc = {}
try:
    for flag in (False, True):
        try:
            T.forward_loss(FakeModel(), FakeLoss(flag), batch, 0.5, "cpu", "off", T.TrainConfig(gap_ego=True))
            outc[flag] = "returned"
        except Reached:
            outc[flag] = "reached the model"
        except ValueError as e:
            outc[flag] = "ValueError: " + str(e)[:40]
finally:
    for k, v in keep.items():
        setattr(T, k, v)
    if not already:
        T._CONTRACT_CHECKED.clear()
check(outc[False] == "reached the model" and outc[True].startswith("ValueError: batch_ego"), f"forward_loss: dt +1.0 s reaches the model silently when the flag is off, raises with it on ({outc})")

# ---------------------------------------------------------------- 6 splat precision
v, om = torch.tensor([10.0]), torch.tensor([0.05])
bev = torch.randn(1, 6, X, Y)
o32, m32, _ = S.splat_bev(bev, grid, 0.5, v, om)
obf, mbf, _ = S.splat_bev(bev.bfloat16(), grid, 0.5, v, om)
err = (mbf.float() - m32).abs().max().item()
check(err < 2e-2, f"splat_bev: a bf16 BEV moves by the float32 positions (mass differs by {err:.1e}, only the weights' bf16 rounding)")
g4 = Grid(x_min=0.0, x_max=60.0, y_min=-26.0, y_max=26.0, res=0.4)             # a grid whose cell centres bf16 cannot hold (0.5 m cells have centres on a 0.25 m lattice, which it can)
bev4 = torch.randn(1, 4, g4.nx, g4.ny)
_, m4, _ = S.splat_bev(bev4, g4, 0.0, torch.zeros(1), torch.zeros(1))
_, m4b, _ = S.splat_bev(bev4.bfloat16(), g4, 0.0, torch.zeros(1), torch.zeros(1))
_, w4 = S.splat_window([(bev4, [])], g4)
_, w4b = S.splat_window([(bev4.bfloat16(), [])], g4)
check((m4b.float() - m4).abs().max().item() < 1e-2 and (w4b.float() - w4).abs().max().item() < 1e-2,
      f"splat_bev / splat_window, 0.4 m cells, no motion: a bf16 BEV maps every cell onto itself (mass error {(m4b.float() - m4).abs().max().item():.1e}, {(w4b.float() - w4).abs().max().item():.1e}); bf16 cell centres alone would be off by up to 0.1 m")
p32 = S.grid_points(grid)[None].expand(1, -1, -1)
q_bf = S.splat_positions(p32.bfloat16(), 0.5, v, om, v_obj=torch.randn(1, X * Y, 2).bfloat16())
check(q_bf.dtype == torch.float32, "splat_positions: half-precision input gives float32 positions")
q32 = S.splat_positions(p32, 0.5, v, om, v_obj=torch.randn(1, X * Y, 2))
check(q32.dtype == torch.float32 and torch.equal(S.splat_positions(p32, 0.5, v, om), S.splat_positions(p32.clone(), 0.5, v, om)), "float32 input: unchanged")
w32, wm32 = S.splat_window([(bev, [])], grid)
wbf, wmbf = S.splat_window([(bev.bfloat16(), [(0.5, 8.0, 0.1)])], grid)
w32b, wm32b = S.splat_window([(bev, [(0.5, 8.0, 0.1)])], grid)
check((wmbf.float() - wm32b).abs().max().item() < 2e-2, "splat_window: the same for a bf16 frame with a chain of ego steps")

# the chain of ego steps in half precision, float64 passing through, the correction head reading the BEV's own dtype
vo = torch.randn(1, X * Y, 2, generator=torch.Generator().manual_seed(5))
pc_bf = S.chain_positions(p32.bfloat16(), [(0.5, 8.0, 0.1), (0.5, 7.0, -0.1)], vo.bfloat16())
pc_ref = S.chain_positions(p32.bfloat16().float(), [(0.5, 8.0, 0.1), (0.5, 7.0, -0.1)], vo.bfloat16().float())
check(pc_bf.dtype == torch.float32 and torch.equal(pc_bf, pc_ref), "chain_positions: half-precision positions and object velocities are computed in float32, bit for bit the float32 path of the same values")
p64 = S.grid_points(grid, dtype=torch.float64)[None]
q64 = S.splat_positions(p64, 0.5, torch.tensor([10.0], dtype=torch.float64), torch.tensor([0.05], dtype=torch.float64))
check(q64.dtype == torch.float64 and S.chain_positions(p64, [(0.5, 8.0, 0.1)]).dtype == torch.float64, "float64 positions stay float64 (only half precision is widened)")
seen = []


def corr_spy(feat, pos, dt, v, omega):
    seen.append(pos.dtype)
    return torch.zeros(feat.shape[0], pos.shape[1], 2)


S.splat_bev(bev.bfloat16(), grid, 0.5, v, om, corr=corr_spy)
S.splat_window([(bev.bfloat16(), [(0.5, 8.0, 0.1)])], grid, corr=corr_spy)
check(seen == [torch.bfloat16, torch.bfloat16], f"the correction head still gets the cell centres in the BEV's dtype, as before ({seen})")

# ---------------------------------------------------------------- 7 batch_ego
gs = [{"prev": [(0.48, 8.0, 0.1)], "next": [(0.52, 8.5, 0.0)]}]
a = T.batch_ego(gs, (-0.5, 0.0, 0.5), "cpu")
b = T.batch_ego(gs, (-0.5, 0.0, 0.5), "cpu", strict=True)
check(a.keys() == b.keys() == {-0.5, 0.5} and all(torch.equal(a[k][i], b[k][i]) for k in a for i in range(3)), "batch_ego: dt -0.5 / 0 / +0.5 s works strict and not strict, the same")
loose = T.batch_ego(gs, (0.1, 0.5, 1.0), "cpu")
check(loose[1.0][0].item() == loose[0.5][0].item() and abs(loose[1.0][0].item() - 0.52) < 1e-6, "batch_ego not strict: dt 1.0 s silently gets the 0.5 s gap (the old behaviour)")
try:
    T.batch_ego(gs, (0.1, 0.5, 1.0), "cpu", strict=True)
    strict_err = False
except ValueError:
    strict_err = True
check(strict_err, "batch_ego strict: dt 0.1 s raises")

print("HYGIENE GATE: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
sys.exit(1 if fails else 0)
