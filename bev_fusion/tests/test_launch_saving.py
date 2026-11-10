"""The launch- and sync-saving variants (D53; training/launch_saving.py, records/speed_options_2026-11-03.md levers 1 - 3): each one does the arithmetic of its plain twin.
CPU, no data, no GPU, a minute. Every switch is off by default and a plain call must not know the switches exist.

  1. splat: the fused scatter (all four corners in one index_add) equals the loop bit for bit on the CPU (float64 and float32, sources outside the grid, importance, mass gate),
     its gradients equal to rounding; the plain call under no switch is the loop
  2. layers.truncated_distance: the cached discs give the same distance field bit for bit, and are built once per (resolution, cap, device)
  3. motion_block.radar_raster: the form without a boolean-mask index gives the same raster bit for bit (detections outside the grid, an empty batch row)
  4. temporal_fusion.recentre: the host-side shifts move the memory exactly as the per-sample int() does (zero, negative, whole-grid shifts)
  5. losses.motion_term: one nonzero for the four selections gives the same loss (with and without LossConfig.hygiene_2026_11_02)
  6. the model: ModelConfig.share_mover (one splat per dt for both heads, with and without the box-velocity carry of the motion block), splat_fused, sync_free and
     LossConfig.sync_free, each alone and all together: every output tensor, the loss terms and the gradients of every parameter equal the plain run (outputs bit for
     bit where the arithmetic is the same, the loss and the gradients to rounding: the two heads' gradients meet in one backward node)
  7. the switches are scoped: after a forward or a loss call every switch is off again, also after an error; the flags are off in a default config
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_launch_saving.py   (PYTHONPATH = the repository root for GtGenerator)"""
import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "training"), os.path.join(ROOT, "tests"), os.path.join(ROOT, "tests", "support"), os.path.join(ROOT, "..")]
import launch_saving as LS  # noqa: E402
import layers as LY  # noqa: E402
import losses as L  # noqa: E402
import motion_block as MB  # noqa: E402
import splat as S  # noqa: E402
import synthetic_world as sw  # noqa: E402
import temporal_fusion as TF  # noqa: E402
from feature_cache import CachedBEVFusionModel  # noqa: E402
from lift_pd import Grid  # noqa: E402
from model import ModelConfig  # noqa: E402
from test_slots import BASE, batch_for  # noqa: E402
from train import forward_loss  # noqa: E402

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


grid = Grid()
X, Y = grid.nx, grid.ny

# ---- 1 splat
for dtype in (torch.float64, torch.float32):
    g = torch.Generator().manual_seed(1)
    B, C, N = 2, 5, 400
    feat = torch.randn(B, C, N, generator=g, dtype=dtype)
    pos = torch.stack([torch.rand(B, N, generator=g, dtype=dtype) * 80 - 8, torch.rand(B, N, generator=g, dtype=dtype) * 60 - 30], -1)       # some outside the 0..60 x -26..26 map
    imp = torch.randn(B, N, generator=g, dtype=dtype)
    same = True
    for importance in (None, imp):
        for gate in (False, True):
            a = S.splat(feat, pos, grid, importance, mass_gate=gate, fused=False)
            b = S.splat(feat, pos, grid, importance, mass_gate=gate, fused=True)
            with LS.enabled(splat=True):
                c = S.splat(feat, pos, grid, importance, mass_gate=gate)
            same &= all(torch.equal(x, y) and torch.equal(x, z) for x, y, z in zip(a, b, c))
    check(same, f"splat {str(dtype)[6:]}: fused == loop bit for bit (outside sources, importance on / off, mass gate on / off), also through the process-wide switch")
f64 = dict(dtype=torch.float64)
feat = torch.randn(2, 4, 300, requires_grad=True, **f64)
pos = (torch.rand(2, 300, 2, **f64) * torch.tensor([60.0, 50.0], **f64) + torch.tensor([0.0, -25.0], **f64)).requires_grad_(True)
imp = torch.randn(2, 300, requires_grad=True, **f64)
w = torch.randn(2, 4, X, Y, **f64)
for gate in (False, True):                                     # the camera and radar memory splat with the mass gate: its gradient (through the clamped mass) is part of the claim
    grads = []
    for fu in (False, True):
        o, m = S.splat(feat, pos, grid, imp, mass_gate=gate, fused=fu)
        grads.append(torch.autograd.grad((o * w).sum() + m.sum() * 0.3, (feat, pos, imp)))
    check(all(torch.allclose(x, y, rtol=1e-10, atol=1e-12) for x, y in zip(*grads)), f"splat gradients (features, positions, importance), mass gate {gate}: fused == loop to 1e-10 (float64)")
for hd in (torch.bfloat16, torch.float16):                     # half precision: about one ulp apart on the CPU (the 2-D index_add rounds per call), never more
    g_ = torch.Generator().manual_seed(7)
    fh = torch.randn(2, 4, 300, generator=g_).to(hd)
    ph = torch.rand(2, 300, 2, generator=g_) * torch.tensor([60.0, 50.0]) + torch.tensor([0.0, -25.0])
    a = S.splat(fh, ph, grid, fused=False)
    b = S.splat(fh, ph, grid, fused=True)
    check(all(torch.allclose(x.float(), y.float(), rtol=2e-2, atol=2e-3) for x, y in zip(a, b)), f"splat {str(hd)[6:]}: fused within one ulp of the loop (the half-precision CPU scatter rounds per call)")
check(not LS.on("splat") and S.splat(feat, pos, grid, imp)[0].shape == (2, 4, X, Y), "splat without a switch is the plain call")

# ---- 2 truncated_distance
g = torch.Generator().manual_seed(2)
mask = (torch.rand(3, 1, X, Y, generator=g) > 0.995).float()
mask[0] = 0                                                    # an empty sample: everything at the cap
plain = LY.truncated_distance(mask)
with LS.enabled(sync=True):
    fast = LY.truncated_distance(mask)
    n_keys = len(LY._DISC_CACHE)
    again = LY.truncated_distance(mask)
check(torch.equal(plain, fast) and torch.equal(plain, again), "truncated_distance: the cached discs give the same field bit for bit")
check(len(LY._DISC_CACHE) == n_keys == 1 and not LY._DISC_CACHE.get(("x",)), "the discs are built once per (resolution, cap, device)")
p2, f2 = LY.pool_polyline(mask, torch.randn(3, 2, X, Y, generator=g), 2), None
with LS.enabled(sync=True):
    f2 = LY.pool_polyline(mask, torch.randn(3, 2, X, Y, generator=torch.Generator().manual_seed(3)), 2)
p2 = LY.pool_polyline(mask, torch.randn(3, 2, X, Y, generator=torch.Generator().manual_seed(3)), 2)
check(all(torch.equal(a, b) for a, b in zip(p2, f2)), "pool_polyline (mask, distance, direction) under the switch == plain")

# ---- 3 radar raster
g = torch.Generator().manual_seed(4)
n = 300
det = {"b": torch.randint(0, 3, (n,), generator=g), "x": torch.rand(n, generator=g) * 90 - 15, "y": torch.rand(n, generator=g) * 70 - 35,
       "psi": torch.rand(n, generator=g) * 6.28 - 3.14, "vr": torch.randn(n, generator=g) * 5, "dt": torch.rand(n, generator=g) * 0.5}
det["b"][det["b"] == 1] = 0                                    # sample 1 has no detection at all: its raster is zero
a = MB.radar_raster(det, 3, grid)
with LS.enabled(sync=True):
    b = MB.radar_raster(det, 3, grid)
check(torch.equal(a, b) and float(a[1].abs().sum()) == 0 and float(a[0, 0].sum()) > 20, "radar_raster: no boolean-mask index, same raster bit for bit (detections outside the grid, an empty sample)")

stray = {k: v.clone() for k, v in det.items()}
outside = (stray["x"] < grid.x_min) | (stray["x"] > grid.x_max)
stray["b"][outside] = 99                                       # a detection outside the grid with a stray sample index: the plain form drops it, the sync form must not index with it
a = MB.radar_raster(stray, 3, grid)
with LS.enabled(sync=True):
    b = MB.radar_raster(stray, 3, grid)
check(bool(outside.any()) and torch.equal(a, b), "radar_raster: a stray sample index on a detection outside the grid is dropped in both forms")

# ---- 4 recentre
g = torch.Generator().manual_seed(5)
M, W = torch.randn(4, 3, 20, 24, generator=g), torch.rand(4, 1, 20, 24, generator=g)
pose = torch.tensor([[0.2, -0.3, 0.1], [3.1, 0.0, 0.0], [-2.6, 5.4, 0.2], [60.0, 60.0, 0.0]])        # no shift, whole cells both ways, one beyond the memory
res = 0.5
a = TF.recentre(M, W, pose, res)
with LS.enabled(sync=True):
    b = TF.recentre(M, W, pose, res)
check(all(torch.equal(x, y) for x, y in zip(a[:2], b[:2])) and torch.equal(a[2], b[2]), "recentre: the host-side shifts move the memory exactly as int() per sample does")

# ---- 5 motion_term
g = torch.Generator().manual_seed(6)
T = {"w_obj": (torch.rand(2, 1, X, Y, generator=g) > 0.9).float() * torch.rand(2, 1, X, Y, generator=g), "velocity": torch.randn(2, 2, X, Y, generator=g) * 5,
     "obj_cells": (torch.rand(2, 1, X, Y, generator=g) > 0.5).float(), "flow_weight": (torch.rand(2, 1, X, Y, generator=g) > 0.2).float()}
vh = torch.randn(2, 2, X, Y, generator=g)
for hyg in (False, True):
    cfg = L.LossConfig(hygiene_2026_11_02=hyg)
    a = L.motion_term(vh, T, cfg)
    with LS.enabled(sync=True):
        b = L.motion_term(vh, T, cfg)
    check(torch.equal(a, b) and float(a) > 0, f"motion_term (hygiene {hyg}): one nonzero for the selections, the same loss bit for bit")

# ---- 6 the model
B_ = sw.Box


def with_gt(batch, grid_):
    scene = sw.Scene([B_(6.0, -1.75, 4.5, 1.8, 1.5, 0.0, 8.0, 0.0, "car", 10.0), B_(10.0, 4.0, 4.5, 1.8, 1.5, 0.0, 0.0, 0.0, "car", 10.0),
                      B_(13.0, -4.0, 8.0, 2.5, 3.0, 0.05, 0.0, 0.0, "truck", 20.0), B_(5.0, 5.5, 0.6, 0.6, 1.8, math.pi / 2, 0.0, 1.2, "pedestrian", -5.0)],
                     lanes=[-5.25, -1.75, 1.75, 5.25], v=8.0, omega=0.05)
    side = {"prev": L.synthetic_gt(scene.step(-0.5), grid_), "cur": L.synthetic_gt(scene, grid_), "next": L.synthetic_gt(scene.step(0.5), grid_)}
    return dict(batch, gt=[side] * batch["v"].shape[0])


PROBE = {"seen": [], "mover": 0}
_on = LS.on
LS.on = lambda name: (PROBE["seen"].append((name, _on(name))), _on(name))[1]       # which switches the code asked for, and what it was told


def run(m, loss, batch, flags):
    for k, v in flags.items():
        if k in ("share_mover", "splat_fused", "sync_free"):
            setattr(m.cfg, k, v)
    loss.cfg.sync_free = flags.get("loss_sync_free", False)
    m.zero_grad()
    torch.manual_seed(0)
    PROBE["seen"].clear()
    PROBE["mover"] = 0
    h = m.mover.register_forward_hook(lambda *a: PROBE.__setitem__("mover", PROBE["mover"] + 1))
    out, res, _ = forward_loss(m, loss, batch, 0.5, "cpu", "off", None)
    h.remove()
    out["total"].backward()
    grads = {n: q.grad.clone() for n, q in m.named_parameters() if q.grad is not None}
    m.cfg.share_mover = m.cfg.splat_fused = m.cfg.sync_free = False
    return out, res, grads, {"told": {n for n, v in PROBE["seen"] if v}, "mover": PROBE["mover"]}


def flat(res):
    d = {}
    for dt, p in res["out"].items():
        for k, v in p.items():
            for kk, t in (v.items() if isinstance(v, dict) else [("", v)]):
                if torch.is_tensor(t):
                    d[(dt, k, kk)] = t
    return d


ARMS = [("plain 1x1 motion head", {}), ("motion block + box read (the carry path)", dict(motion_block=True, box_vel_from_field=True))]
VARIANTS = [("share_mover", dict(share_mover=True)), ("splat_fused", dict(splat_fused=True)), ("sync_free (model only)", dict(sync_free=True)),
            ("sync_free (loss only)", dict(loss_sync_free=True)), ("sync_free (model + loss)", dict(sync_free=True, loss_sync_free=True)),
            ("all together", dict(share_mover=True, splat_fused=True, sync_free=True, loss_sync_free=True))]
for aname, extra in ARMS:
    cfg = ModelConfig(**dict(BASE, **extra))
    torch.manual_seed(0)
    m = CachedBEVFusionModel(cfg).eval()                      # eval: no modality dropout, the same random numbers in every run
    with torch.no_grad():
        for q in m.parameters():
            if q.requires_grad and q.abs().max() == 0:
                q.normal_(0, 1e-2)
    batch = with_gt(batch_for(cfg), m.grid)
    loss = L.BEVLoss(grid=m.grid)
    ref_out, ref_res, ref_g, ref_probe = run(m, loss, batch, {})
    ref_flat = flat(ref_res)
    check(not ref_probe["told"] and ref_probe["mover"] == 6, f"{aname}: the plain run is told no switch and calls the mover twice per dt ({ref_probe['mover']} calls for 3 dts)")
    for vname, flags in VARIANTS:
        out, res, g, probe = run(m, loss, batch, flags)
        want = {n for n, on in (("splat", flags.get("splat_fused")), ("sync", flags.get("sync_free") or flags.get("loss_sync_free"))) if on}
        check(probe["told"] == want and probe["mover"] == (3 if flags.get("share_mover") else 6),
              f"{aname} / {vname}: the code was told {sorted(probe['told'])} (wanted {sorted(want)}) and called the mover {probe['mover']} times ({'3 = once per dt' if flags.get('share_mover') else '6'})")
        f = flat(res)
        same_keys = set(f) == set(ref_flat)
        exact = [k for k in ref_flat if k in f and torch.equal(ref_flat[k], f[k])]
        worst = max((float((ref_flat[k] - f[k]).abs().max()) for k in ref_flat if k in f), default=0.0)
        scale = max(float(t.abs().max()) for t in ref_flat.values())
        terms_ok = set(out) == set(ref_out) and all(abs(float(out[k]) - float(ref_out[k])) <= 1e-5 * max(1.0, abs(float(ref_out[k]))) for k in ref_out if torch.is_tensor(ref_out[k]))
        grads_ok = set(g) == set(ref_g) and all(torch.allclose(g[n], ref_g[n], rtol=1e-4, atol=1e-5 * float(ref_g[n].abs().max()) + 1e-9) for n in ref_g)
        check(same_keys and worst <= 1e-5 * max(1.0, scale) and terms_ok and grads_ok,
              f"{aname} / {vname}: {len(exact)} of {len(ref_flat)} output tensors bit-equal (worst difference {worst:.1e}), {len(ref_out)} loss terms and {len(ref_g)} gradients equal to rounding")
        if flags in (dict(splat_fused=True), dict(sync_free=True), dict(loss_sync_free=True), dict(sync_free=True, loss_sync_free=True)):
            check(len(exact) == len(ref_flat), f"{aname} / {vname}: every output tensor is bit-equal to the plain run on the CPU")

# ---- 7 scoping and defaults
check(not any(LS.on(k) for k in ("splat", "sync")), "every switch is off after the model and loss calls above")
check(not (ModelConfig().share_mover or ModelConfig().splat_fused or ModelConfig().sync_free or L.LossConfig().sync_free), "the flags are off in a default config")
try:
    bad = ModelConfig(**dict(BASE))
    mm = CachedBEVFusionModel(bad).eval()
    mm.cfg.splat_fused = True
    bt = batch_for(bad)
    bt["steps"] = "not a list of steps"                                       # an error inside the forward: the switch must not stay on
    try:
        mm(bt, dts=(0.0,))
    except Exception:
        pass
    check(not LS.on("splat"), "an error inside a forward leaves the switches off")
except Exception as e:                                                         # noqa: BLE001
    check(False, f"scoping check crashed: {e!r}")
LS._selftest()
print(f"{len(fails)} failure(s)")
sys.exit(1 if fails else 0)
