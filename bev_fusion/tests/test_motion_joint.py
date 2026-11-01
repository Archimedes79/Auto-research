"""The motion block's joint form (t15: ModelConfig.motion_block, motion_strides, box_vel_from_field) through the trainer's path.

Synthetic cached batch (tests/test_slots.py) + synthetic GT at -0.5 / 0 / +0.5 s (losses.synthetic_gt: a moving car, a parked car, a
truck, a walking pedestrian), CPU, no data, no GPU:
  1. model -> train.forward_loss (the trainer's float cast, the output contract) -> BEVLoss -> backward, for the block alone, the block
     with the box read, the runtime arm (strides 4, 2) and an arm close to the real config (camera_features 'dual', head_channels,
     trunk_detail) with a mirrored sample; the ego turn on its own below
  2. the flags do what they say: the time mover has no head, v_hat IS the block's last level, v_levels holds the coarser ones
     (len(strides) - 1), the box velocity at dt 0 IS the field (only with box_vel_from_field), at dt +0.5 it is the field moved with
     the state and turned by R(-omega dt) (a constant field with the ego turning: the expected vector by hand), the motion group is the
     mean of the level terms (motion, motion_l0, ...), every tensor of the block gets a gradient, the object head's own vel branch
     is frozen when it is not read
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_motion_joint.py   (about 1 min)"""
import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "training"), os.path.join(ROOT, "tests"), os.path.join(ROOT, "..")]
import synthetic_world as sw  # noqa: E402
from feature_cache import CachedBEVFusionModel  # noqa: E402
from losses import BEVLoss, synthetic_gt  # noqa: E402
from model import ModelConfig  # noqa: E402
from test_slots import BASE, batch_for  # noqa: E402
from train import forward_loss  # noqa: E402

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def with_gt(batch, grid):
    B_ = sw.Box
    scene = sw.Scene([B_(6.0, -1.75, 4.5, 1.8, 1.5, 0.0, 8.0, 0.0, "car", 10.0), B_(10.0, 4.0, 4.5, 1.8, 1.5, 0.0, 0.0, 0.0, "car", 10.0),
                      B_(13.0, -4.0, 8.0, 2.5, 3.0, 0.05, 0.0, 0.0, "truck", 20.0),
                      B_(5.0, 5.5, 0.6, 0.6, 1.8, math.pi / 2, 0.0, 1.2, "pedestrian", -5.0)],
                     lanes=[-5.25, -1.75, 1.75, 5.25], v=8.0, omega=0.05)
    side = {"prev": synthetic_gt(scene.step(-0.5), grid), "cur": synthetic_gt(scene, grid), "next": synthetic_gt(scene.step(0.5), grid)}
    B = batch["v"].shape[0]
    return dict(batch, gt=[side] * B)


ARMS = [("block", dict(motion_block=True)),
        ("block + box read", dict(motion_block=True, box_vel_from_field=True)),
        ("runtime arm (4, 2) + box read", dict(motion_block=True, box_vel_from_field=True, motion_strides=(4, 2))),
        ("close to the real arm: dual, head_channels, detail, a mirrored sample", dict(motion_block=True, box_vel_from_field=True,
                                                                                    camera_features="dual", head_channels=16, trunk_detail=8))]

for name, kw in ARMS:
    cfg = ModelConfig(**dict(BASE, **kw))
    torch.manual_seed(0)
    m = CachedBEVFusionModel(cfg).train()
    with torch.no_grad():                                        # the zero-start last layers: small values, so the gradient paths are tested
        for q in m.parameters():
            if q.requires_grad and q.abs().max() == 0:
                q.normal_(0, 1e-2)
    batch = with_gt(batch_for(cfg), m.grid)
    if "dual" in name:
        batch["flip"] = torch.tensor([True, False])
    levels = {}
    m.motion_block.register_forward_hook(lambda mod, inp, out: levels.__setitem__("out", out))
    loss = BEVLoss(grid=m.grid)
    m.zero_grad()
    out, res, _ = forward_loss(m, loss, batch, 0.5, "cpu", "off", None)
    print(f"{name}: total {float(out['total']):.3f}")
    out["total"].backward()
    p0, p5 = res["out"][0.0], res["out"][0.5]
    lv = levels["out"]
    n_lv = len(cfg.motion_strides)
    check(m.mover.motion is None and len(lv) == n_lv, "the time mover has no head; the block returns one field per level")
    check(p0["v_hat"] is lv[-1], "v_hat is the block's finest level")
    check(isinstance(p0.get("v_levels"), dict) and len(p0["v_levels"]) == n_lv - 1
          and all(p0["v_levels"][f"l{i}"] is lv[i] for i in range(n_lv - 1)), f"v_levels = the {n_lv - 1} coarser levels (a dict)")
    terms = [float(out["motion"])] + [float(out[f"motion_l{i}"]) for i in range(n_lv - 1)]
    check(all(k in out for k in [f"motion_l{i}" for i in range(n_lv - 1)]) and abs(float(out["group.motion"]) - sum(terms) / len(terms)) < 1e-5,
          f"the motion group is the mean of {len(terms)} level terms")
    dead = [n for n, q in m.motion_block.named_parameters() if q.grad is None or q.grad.abs().max() == 0]
    check(not dead, f"every tensor of the block gets a gradient ({sum(1 for _ in m.motion_block.parameters())})")
    vel_frozen = all(not q.requires_grad for q in m.obj_head.head.vel.parameters())
    if cfg.box_vel_from_field:
        check(torch.equal(p0["obj"]["vel"], p0["v_hat"]) and vel_frozen, "box read: the box velocity at dt 0 IS the field; the own vel branch is frozen")
        check(p5["obj"]["vel"].shape == p0["v_hat"].shape and not torch.equal(p5["obj"]["vel"], p0["v_hat"]), "at dt +0.5 the box velocity is the moved field")
    else:
        check(not torch.equal(p0["obj"]["vel"], p0["v_hat"]) and not vel_frozen, "without the box read the object head keeps its own vel")

# the turn of the carried field: a constant field (4, 0) with the ego turning omega = 0.4 rad/s for dt = 0.5 s reads, in the target's
# ego axes, R(-0.2) (4, 0) = (3.92, -0.79) where it landed (the convention of splat.chain_positions: the world-fixed vector turns back)
cfg = ModelConfig(**dict(BASE, motion_block=True, box_vel_from_field=True))
torch.manual_seed(0)
m = CachedBEVFusionModel(cfg).eval()
batch = batch_for(cfg)
const = torch.zeros(2, 2, m.grid.nx, m.grid.ny)
const[:, 0] = 4.0
m.motion_block.forward = lambda *a: [const, const, const]
B = batch["v"].shape[0]
with torch.no_grad():
    r = m(batch, dts=(0.0, 0.5), ego={0.5: (torch.full((B,), 0.5), torch.full((B,), 0.0), torch.full((B,), 0.4))})
vel5, mass5 = r["out"][0.5]["obj"]["vel"], r["out"][0.5]["mass"]
inside = mass5[:, 0] > 0.99
exp = torch.tensor([4 * math.cos(-0.2), 4 * math.sin(-0.2)])
got = vel5.permute(0, 2, 3, 1)[inside]
check(inside.sum() > 100 and torch.allclose(got, exp.expand_as(got), atol=1e-3),
      f"the carried field is turned by R(-omega dt): {got.mean(0).tolist()} vs {exp.tolist()}")

print("MOTION JOINT GATE: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
sys.exit(1 if fails else 0)
