"""Gate of the t11 velocity hurdle (experiments/t11_vel_hurdle/experiment.md): ObjectHead(moving=True) = a moving / static
logit `mov` + the velocity L1 on moving centres only + decode's gate; TrainConfig.init_from = warm start from another run.
(G1) defaults unchanged: no `mov` branch, key or term; known answers of the two velocity losses (all centres vs moving only);
(G2) toy: 80 % static, 20 % moving centres, the velocity readable from the features; Adam 300 steps of 64: gated error on moving
     centres < 0.3 m/s, on static < 0.1 m/s; gradients reach `mov` and `vel`;
(G3) decode zeroes the velocity where sigmoid(mov) < 0.5 and keeps it otherwise;
(G4) warm start on the real runs/t10_dual_s0/ckpt_00003000.pt (CPU) with both configs (arms A' / B': init_reset of the vel
     branch, rule C1 of experiments/t11_vel_cause): every shared tensor equals the checkpoint's EMA tensor bit for bit except
     obj_head.head.vel.*, which equals its fresh initialisation (the same in both arms); the loss takes its Kendall s; the fresh
     keys are exactly vel + mov (B') / vel (A'); a shape mismatch, a key the model lacks or a reset prefix that names nothing
     raises; in train(): the warm start happens only in a run folder without a checkpoint.
(G5, the existing self-tests of object_head.py, losses.py, train.py, are run separately.)
Run: wsl bash tools/env/_py14.sh tests/test_velocity_hurdle.py   (CPU, about 1 min; G4 needs the t10 checkpoint, ~1.5 GB RAM)"""
import dataclasses
import math
import os
import shutil
import sys
import tempfile

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "training")]
os.chdir(ROOT)
import train as tr  # noqa: E402
from lift_pd import Grid  # noqa: E402
from object_head import ObjectHead, _gather, build_targets, decode, head_loss, targets_as_output  # noqa: E402

torch.set_num_threads(2)
torch.manual_seed(0)
CKPT = "runs/t10_dual_s0/ckpt_00003000.pt"
grid = Grid()
boxes = [[{"group": "car", "cx": 20.3, "cy": 1.6, "length": 4.5, "width": 1.8, "yaw": 0.3, "vx": 8.0, "vy": 1.0},
          {"group": "truck", "cx": 41.1, "cy": -7.2, "length": 8.0, "width": 2.5, "yaw": -2.0, "vx": 0.2, "vy": 0.1},
          {"group": "pedestrian", "cx": 12.6, "cy": 9.9, "radius": 0.4, "vx": 0.0, "vy": 1.2}]]
t = build_targets(boxes, grid)
moving = torch.tensor([1.0, 0.0, 1.0])                         # |vel| > 0.5 m/s: car, pedestrian; the truck stands
logit = lambda p: torch.logit(p.clamp(1e-4, 1 - 1e-4))         # noqa: E731

# G1 defaults: no mov anywhere; velocity L1 over all centres. Known answers with the velocity off by (1, 1) at every centre:
#   default 'vel' = w_vel * 2 (all 3 centres); hurdle 'vel' = w_vel * 2 (only the 2 moving ones, mean over them); a static error
#   alone costs the default head and not the hurdle head
h0 = ObjectHead(cin=16)
o0 = h0(torch.randn(1, 16, grid.nx, grid.ny))
assert "mov" not in o0 and not hasattr(h0, "mov") and not any(k.startswith("mov.") for k in h0.state_dict())
po = dict(targets_as_output(t, grid), hm=logit(targets_as_output(t, grid)["hm"]))
assert set(head_loss(po, t)) == {"hm", "off", "size", "rot", "rad", "vel", "total"}
mov_map = torch.zeros(1, 1, grid.nx, grid.ny)
mov_map.view(1, 1, -1)[0, 0, t["ind"][0, :3]] = 10.0 * (2 * moving - 1)        # the right class, confidently
off_all = dict(po, vel=po["vel"] + 1.0)
off_static = dict(po, vel=po["vel"].clone())
off_static["vel"].view(1, 2, -1)[0, :, t["ind"][0, 1]] += 1.0                   # only the standing truck is off
for out, w_def, w_hur in ((off_all, 0.2 * 2, 0.2 * 2), (off_static, 0.2 * 2 / 3, 0.0)):
    ld, lh = head_loss(out, t), head_loss(dict(out, mov=mov_map), t)
    assert abs(float(ld["vel"]) - w_def) < 1e-5 and abs(float(lh["vel"]) - w_hur) < 1e-5, (float(ld["vel"]), float(lh["vel"]))
assert float(head_loss(dict(po, mov=mov_map), t)["mov"]) < 1e-3 and float(head_loss(dict(po, mov=-mov_map), t)["mov"]) > 5.0
print("G1 ok: default head without mov (branch, output, terms); velocity L1 over all centres vs moving only; mov BCE ordering")

# G2 toy: one box per sample on a 16 x 16 grid; features in a 3 x 3 patch at the centre: presence, moving flag, vx / 10, vy / 10;
#   20 % moving (2 - 10 m/s per axis, random signs), 80 % static (< 0.1 m/s)
g2 = Grid(x_min=0.0, x_max=8.0, y_min=-4.0, y_max=4.0, res=0.5)
gen = torch.Generator().manual_seed(0)


def toy(n):
    feat, bl = torch.zeros(n, 4, g2.nx, g2.ny), []
    for b in range(n):
        ix, iy = (int(v) for v in torch.randint(2, 14, (2,), generator=gen))
        mov = bool(torch.rand(1, generator=gen) < 0.2)
        if mov:
            v = (2 + 8 * torch.rand(2, generator=gen)) * torch.where(torch.rand(2, generator=gen) < 0.5, -1.0, 1.0)
        else:
            a = 2 * math.pi * float(torch.rand(1, generator=gen))
            v = 0.1 * float(torch.rand(1, generator=gen)) * torch.tensor([math.cos(a), math.sin(a)])
        feat[b, :, ix - 1:ix + 2, iy - 1:iy + 2] = torch.tensor([1.0, float(mov), v[0] / 10, v[1] / 10])[:, None, None]
        bl.append([{"group": "car", "cx": g2.x_min + (ix + 0.5) * g2.res, "cy": g2.y_min + (iy + 0.5) * g2.res, "length": 1.0,
                    "width": 1.0, "yaw": 0.0, "vx": float(v[0]), "vy": float(v[1])}])
    return feat, build_targets(bl, g2, max_obj=1)


head = ObjectHead(cin=4, hidden=32, moving=True)
opt = torch.optim.Adam(head.parameters(), 3e-3)
for it in range(300):
    if it == 200:                                              # the last third at lr / 10 (L1 converges slowly at a fixed lr)
        opt.param_groups[0]["lr"] = 3e-4
    feat, tt = toy(64)
    lo = head_loss(head(feat), tt, w_vel=1.0)
    opt.zero_grad()
    lo["total"].backward()
    if it == 0:
        assert head.mov[2].weight.grad.abs().sum() > 0 and head.vel[2].weight.grad.abs().sum() > 0, "gradients reach mov and vel"
    opt.step()
with torch.no_grad():
    feat, tt = toy(1000)
    out = head(feat)
v = _gather(out["vel"], tt["ind"])[:, 0]
v = torch.where(_gather(out["mov"], tt["ind"])[:, 0] < 0, torch.zeros_like(v), v)          # decode's gate
err, mv = (v - tt["vel"][:, 0]).norm(dim=-1), tt["vel"][:, 0].norm(dim=-1) > 0.5
e_mov, e_sta = float(err[mv].mean()), float(err[~mv].mean())
assert e_mov < 0.3 and e_sta < 0.1, (e_mov, e_sta)
print(f"G2 ok: toy hurdle after 300 steps: gated velocity error moving {e_mov:.3f} m/s ({int(mv.sum())} centres), static "
      f"{e_sta:.3f} m/s ({int((~mv).sum())}); gradients reach mov and vel")

# G3 decode: the perfect maps with mov = +-1: the boxes whose mov logit is negative get velocity 0, the others keep theirs
pr = targets_as_output(t, grid)
for sign in (1.0, -1.0):
    mm = torch.zeros(1, 1, grid.nx, grid.ny)
    mm.view(1, 1, -1)[0, 0, t["ind"][0, :3]] = sign * torch.tensor([1.0, -1.0, 1.0])
    dec = decode(dict(pr, mov=mm), grid, from_probabilities=True, threshold=0.99)[0]
    for gb, keep in zip(boxes[0], (sign > 0, sign < 0, sign > 0)):
        db = min(dec, key=lambda d: math.hypot(d["cx"] - gb["cx"], d["cy"] - gb["cy"]))
        want = (gb["vx"], gb["vy"]) if keep else (0.0, 0.0)
        assert abs(db["vx"] - want[0]) < 1e-5 and abs(db["vy"] - want[1]) < 1e-5, (sign, gb, db)
print("G3 ok: decode zeroes the velocity where sigmoid(mov) < 0.5 and keeps it otherwise")

# G4a the real checkpoint: the hurdle and the control config, CPU
ck = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)          # mmap: RAM of a shared box
loss_ck = {k[5:]: v for k, v in ck["state"].items() if k.startswith("loss.")}
branch = lambda b: [f"obj_head.head.{b}.{i}.{p}" for i in (0, 2) for p in ("bias", "weight")]      # noqa: E731
init = {}                                                       # the fresh vel branch: the same in both arms (built before mov)
for name, fresh_want in (("t11_vel_hurdle", sorted(branch("mov") + branch("vel"))), ("t11_vel_ctl", branch("vel"))):
    cfg = dataclasses.replace(tr.read_config(f"training/configs/{name}.json"), device="cpu")
    assert cfg.init_from == CKPT and cfg.steps == 1000 and cfg.model.get("object_moving", False) == (name == "t11_vel_hurdle")
    assert list(cfg.init_reset) == ["obj_head.head.vel."]
    torch.manual_seed(cfg.seed)
    model, loss, _ = tr.build(cfg)
    vel0 = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("obj_head.head.vel.")}
    fresh = tr.warm_start(cfg.init_from, model, loss, ck, reset=cfg.init_reset)
    assert fresh == fresh_want, (name, fresh)
    sd = model.state_dict()
    for k, v in ck["ema"].items():                              # vel: its fresh initialisation; everything else: the checkpoint
        assert torch.equal(sd[k], vel0[k] if k in vel0 else v), k
    assert all(not torch.equal(vel0[k], ck["ema"][k]) for k in vel0)
    for k, v in init.items():
        assert torch.equal(vel0[k], v), k
    init = vel0
    for k, v in ck["state"].items():
        if k.startswith("model.") and not v.dtype.is_floating_point:
            assert torch.equal(sd[k[6:]], v), k
    assert all(torch.equal(loss.state_dict()[k], v) for k, v in loss_ck.items()) and len(loss_ck) == len(loss.state_dict())
    n_ema = len(ck["ema"])
    if name == "t11_vel_ctl":                                   # a checkpoint that does not fit, a reset prefix that names nothing
        k0 = "obj_head.head.vel.2.weight"
        for bad, reset in (({**ck["ema"], k0: torch.zeros(1, 1, 1, 1)}, ()), ({**ck["ema"], "obj_head.head.nothing.weight": torch.zeros(1)}, ()),
                           (ck["ema"], ("obj_head.head.nothing.",))):
            try:
                tr.warm_start(CKPT, model, loss, dict(ck, ema=bad), reset=reset)
                raise AssertionError("a checkpoint that does not fit the model was accepted")
            except ValueError:
                pass
    del model, loss, sd
del ck
print(f"G4a ok: warm start from {CKPT}: {n_ema} EMA tensors and the integer buffers bit for bit except obj_head.head.vel.* "
      f"(init_reset: its fresh initialisation, the same in both arms), Kendall s taken; fresh keys exactly vel + mov (hurdle) / vel "
      f"(control); shape mismatch, unknown key and an empty reset prefix raise")

# G4b in train(): run A (steps 0) leaves a checkpoint; its EMA shifted by +1 is the start of run B (+ object_moving): B's weights are
#   the shifted EMA, the mov branch fresh; a second call of B resumes from B's own checkpoint and does not warm-start again
tmp = tempfile.mkdtemp()
try:
    base = tr.TrainConfig(name="a", out_root=tmp, data_root="~/data/nuscenes", version="v1.0-mini", gt_cache="", steps=0, workers=0,
                          eval_every=0, save_every=0, seq_len=1, device="cpu", model={"pretrained": "", "trunk_width": 0.25})
    data = (list(range(4)), list(range(4)))
    tr.train(base, datasets=data)
    cka = torch.load(os.path.join(tmp, "a", "ckpt_00000000.pt"), map_location="cpu", weights_only=False)
    cka["ema"] = {k: v + 1.0 for k, v in cka["ema"].items()}
    start = os.path.join(tmp, "start.pt")
    torch.save(cka, start)
    cfg_b = dataclasses.replace(base, name="b", init_from=start, model={**base.model, "object_moving": True})
    mb, _, emab, _ = tr.train(cfg_b, datasets=data)
    sdb = mb.state_dict()
    assert all(torch.equal(sdb[k], v) for k, v in cka["ema"].items()) and "obj_head.head.mov.2.weight" in sdb
    assert all(torch.equal(emab.shadow[k], sdb[k]) for k in emab.shadow), "the EMA starts at the warm-started weights"
    real = tr.warm_start

    def no_second(*a, **k):
        raise AssertionError("warm start in a run folder that has a checkpoint")
    tr.warm_start = no_second
    try:
        tr.train(cfg_b, datasets=data)
    finally:
        tr.warm_start = real
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print("G4b ok: train() warm-starts a new run folder (weights = the start's EMA, EMA reset to them) and resumes an existing one")
print("velocity hurdle gate PASSES (G1 - G4)")
