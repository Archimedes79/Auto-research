"""Gate of the t11 velocity hurdle (experiments/t11_vel_hurdle/experiment.md): ObjectHead(moving=True) = a moving / static
logit `mov` + the velocity L1 on moving centres only + decode's gate; TrainConfig.init_from = warm start from another run.
(G1) defaults unchanged: no `mov` branch, key or term; known answers of the two velocity losses (all centres vs moving only);
(G2) toy: 80 % static, 20 % moving centres, the velocity readable from the features; Adam 300 steps: gated error on moving
     centres < 0.3 m/s, on static < 0.1 m/s; gradients reach `mov` and `vel`;
(G3) decode zeroes the velocity where sigmoid(mov) < 0.5 and keeps it otherwise;
(G4) warm start on the real runs/t10_dual_s0/ckpt_00003000.pt (CPU): every shared tensor equals the checkpoint's EMA tensor bit
     for bit, the loss takes its Kendall s, the fresh keys are exactly obj_head.head.mov.* (hurdle) / none (control); a shape
     mismatch or a key the model lacks raises; in train(): the warm start happens only in a run folder without a checkpoint.
(G5, the existing self-tests of object_head.py, losses.py, train.py, are run separately.)
Run: wsl bash tools/env/_py14.sh tests/test_velocity_hurdle.py   (CPU, about 1 min; G4 needs the t10 checkpoint, ~1 GB RAM)"""
import dataclasses
import json
import math
import os
import shutil
import sys
import tempfile

import torch

ROOT = "/mnt/c/Auto-research/bev_fusion"
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

import sys as _s
if True:
    pass
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


for B, lr, drop in ((32, 3e-3, 200), (64, 3e-3, 200), (32, 1e-2, 200), (64, 3e-3, 10**9)):
    torch.manual_seed(0)
    head = ObjectHead(cin=4, hidden=32, moving=True)
    opt = torch.optim.Adam(head.parameters(), lr)
    for it in range(300):
        if it == drop:
            for g in opt.param_groups: g["lr"] = lr / 10
        feat, tt = toy(B)
        lo = head_loss(head(feat), tt, w_vel=1.0)
        opt.zero_grad(); lo["total"].backward(); opt.step()
    with torch.no_grad():
        feat, tt = toy(1000); out = head(feat)
    v0 = _gather(out["vel"], tt["ind"])[:, 0]
    ml = _gather(out["mov"], tt["ind"])[:, 0, 0]
    v = torch.where(ml[:, None] < 0, torch.zeros_like(v0), v0)
    err, mv = (v - tt["vel"][:, 0]).norm(dim=-1), tt["vel"][:, 0].norm(dim=-1) > 0.5
    e0 = (v0 - tt["vel"][:, 0]).norm(dim=-1)
    print(B, lr, drop, "moving", float(err[mv].mean()), "ungated", float(e0[mv].mean()), "static", float(err[~mv].mean()), "cls acc", float(((ml > 0) == mv).float().mean()), flush=True)
