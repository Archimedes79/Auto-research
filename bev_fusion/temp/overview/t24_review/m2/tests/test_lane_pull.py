"""The lane branch's lift weighting, ModelConfig.lane_use_depth (card t24, experiments/t24_lane_pull/experiment.md); CPU, seconds, no data.

  A  default: lane_use_depth is True and the branch lifts with lift PD as before; False gives lift P (LiftPD.use_depth False) on the same table.
  B  nothing else changes: the tensors (keys and shapes) are the same either way and a state_dict of one loads strictly into the other (a checkpoint of the
     t20 network loads into the t24 network).
  C  the weighting: with lift PD the branch's contribution changes when the depth probabilities change; with lift P it does not, and it still changes
     with the image (the geometry, the calibration and the ida are those of lift PD: same table).
  D  gradient: with lift P a loss on the maps reaches every trainable tensor of the branch, the frozen ResNet gets none.
  E  mirror: a flipped sample gets the mirrored contribution also with lift P.
  F  configs/t24_lane_pull.json is configs/t21_ctl.json with exactly the change of the card (name, _about, model.lane_use_depth false) and its model passes check_config.
Run (from the bev_fusion folder): wsl ~/miniconda3/envs/gcar/bin/python tests/test_lane_pull.py"""
import json
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for sub in ("tests", "training"):
    sys.path.insert(0, os.path.join(ROOT, sub))
from feature_cache import CachedBEVFusionModel  # noqa: E402
from lift_pd import LiftPD  # noqa: E402
from model import ModelConfig, check_config  # noqa: E402
from nuscenes_dataset import DST_HW  # noqa: E402
from test_slots import batch_for  # noqa: E402

torch.set_num_threads(2)
fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


GRID = {"x_max": 16.0, "y_min": -8.0, "y_max": 8.0}                 # 32 x 32 cells
TINY = dict(pretrained="", trunk_pretrained="", lane_pretrained="", grid_kwargs=GRID, oversample=1, c_cam=16, c_rad=8, c_fuse=24,
            trunk_width=0.25, p_drop=0.0, camera_features="dual", lane_branch="res4")
T21 = os.path.join(ROOT, "training", "configs", "t21_ctl.json")
T24 = os.path.join(ROOT, "training", "configs", "t24_lane_pull.json")


def build(**kw):
    torch.manual_seed(0)
    return CachedBEVFusionModel(ModelConfig(**dict(TINY, **kw)))


def rows(m):
    return [f"{k} {tuple(v.shape)}" for k, v in m.state_dict().items()]


def wake(m):
    """the zero-initialised mix cuts the branch's contribution at init by design: give it small values"""
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        m.lane.mix.weight.copy_(torch.randn(m.lane.mix.weight.shape, generator=g) * 0.05)
        m.lane.mix.bias.copy_(torch.randn(m.lane.mix.bias.shape, generator=g) * 0.05)


t0 = time.time()
img = torch.randint(0, 256, (2, 3, *DST_HW), dtype=torch.uint8, generator=torch.Generator().manual_seed(1))


def batch(m, image=img):
    b = batch_for(m.cfg)
    b["lane_img"] = image
    return b


# ------------------------------------------------------------------------------------------------ A
print("A: the option")
check(ModelConfig().lane_use_depth is True, "ModelConfig.lane_use_depth defaults to True (lift PD as in t7 / t20)")
pd_, pull = build(), build(lane_use_depth=False)
check(pd_.lane.use_depth is True and pd_.lane.lift.use_depth is True, "default: the branch lifts with lift PD (LiftPD.use_depth True)")
check(pull.lane.use_depth is False and pull.lane.lift.use_depth is False, "lane_use_depth False: lift P (LiftPD.use_depth False)")
check(LiftPD(pd_.grid, [1.0, 2.0]).use_depth is True, "LiftPD's own default stays True")
check(build(lane_branch="").lane is None, "no lane branch: nothing is built, the option is not read")
for kind in ("res4", "rgb"):
    check(not check_config(ModelConfig(**dict(TINY, lane_branch=kind, lane_use_depth=False))), f"check_config accepts lane_use_depth False with lane_branch {kind!r}")

# ------------------------------------------------------------------------------------------------ B
print("B: nothing else changes")
check(rows(pd_) == rows(pull), f"the same tensors, keys and shapes with lift PD and lift P ({len(rows(pd_))} tensors)")
rep = pull.load_state_dict(pd_.state_dict(), strict=True)
check(not rep.missing_keys and not rep.unexpected_keys, "a state_dict made with lift PD loads strictly into the lift P model")

# ------------------------------------------------------------------------------------------------ C
print("C: the weighting")
pd_.load_state_dict(pull.state_dict())
wake(pd_), wake(pull)
pd_.eval(), pull.eval()
bt = batch(pull)
b_prob = dict(bt, prob=torch.rand_like(bt["prob"]).softmax(2))
b_img = dict(bt, lane_img=img.flip(0))
res = {}
for name, m in (("pd", pd_), ("pull", pull)):
    contrib = []
    h = m.lane.register_forward_hook(lambda mod, a, o, c=contrib: c.append(o.detach().clone()))
    with torch.no_grad():
        for b in (bt, b_prob, b_img):
            m(b, dts=(0.0,))
    h.remove()
    res[name] = contrib
a_ = res["pd"]
check(float(a_[0].abs().max()) > 0 and float((a_[0] - a_[1]).abs().max()) > 1e-4 * float(a_[0].abs().max()),
      "lift PD: the branch's contribution changes with the depth probabilities")
a_ = res["pull"]
check(float(a_[0].abs().max()) > 0 and torch.equal(a_[0], a_[1]), "lift P: the contribution does NOT change with the depth probabilities")
check(float((a_[0] - a_[2]).abs().max()) > 1e-4 * float(a_[0].abs().max()), "... and it does change with the image")
check(float((res["pd"][0] - res["pull"][0]).abs().max()) > 1e-4 * float(res["pull"][0].abs().max()), "the two lifts give different contributions for the same weights (the option has an effect)")

# ------------------------------------------------------------------------------------------------ D
print("D: gradient with lift P")
pull.train()
with torch.no_grad():
    for q in pull.parameters():
        if q.requires_grad and q.abs().max() == 0:
            q.normal_(0, 1e-2)
pull.zero_grad()
out = pull(bt, dts=(0.0,))
(out["out"][0.0]["sem"].float().pow(2).mean() + sum(v.float().pow(2).mean() for v in out["out"][0.0]["obj"].values())).backward()
trainable = [(n, q) for n, q in pull.named_parameters() if n.startswith("lane.") and q.requires_grad]
dead = [n for n, q in trainable if q.grad is None or not torch.isfinite(q.grad).all() or q.grad.abs().sum() == 0]
check(len(trainable) == 10 and not dead, f"a loss on the maps reaches all {len(trainable)} trainable tensors of the branch{' - none: ' + str(dead) if dead else ''}")
check(all(q.grad is None for q in pull.lane.features.parameters()), "... the frozen ResNet gets none")

# ------------------------------------------------------------------------------------------------ E
print("E: mirror")
pull.eval()
twin = build(lane_use_depth=False).eval()
twin.load_state_dict(pull.state_dict())
with torch.no_grad():
    twin.lane.mix.weight.zero_()
    twin.lane.mix.bias.zero_()
cams = []
ha = pull.fusion.register_forward_pre_hook(lambda m, a: cams.append(a[0].detach().clone()))
hb = twin.fusion.register_forward_pre_hook(lambda m, a: cams.append(a[0].detach().clone()))
delta = {}
with torch.no_grad():
    for fl in ((False, False), (True, False)):
        cams.clear()
        bf = dict(bt, flip=torch.tensor(fl))
        pull(bf, dts=(0.0,))
        twin(bf, dts=(0.0,))
        delta[fl] = cams[0] - cams[1]
ha.remove()
hb.remove()
a_, b_ = delta[(False, False)], delta[(True, False)]
check(float(a_.abs().max()) > 0 and torch.allclose(b_[0], a_[0].flip(-1), atol=1e-6) and torch.allclose(b_[1], a_[1], atol=1e-6),
      "a flipped sample gets the mirrored lane contribution with lift P, the other sample is unchanged")

# ------------------------------------------------------------------------------------------------ F
print("F: configs/t24_lane_pull.json")
check(os.path.exists(T24) and os.path.exists(T21), "configs/t24_lane_pull.json and configs/t21_ctl.json exist")
if os.path.exists(T24) and os.path.exists(T21):
    a, b = json.load(open(T21)), json.load(open(T24))
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    mdiff = {k for k in set(a["model"]) | set(b["model"]) if a["model"].get(k) != b["model"].get(k)}
    check(diff == {"_about", "name", "model"} and b["name"] == "t24_lane_pull", f"top level: only _about, name and model differ ({sorted(diff)})")
    check(mdiff == {"lane_use_depth"} and b["model"]["lane_use_depth"] is False and b["model"]["lane_branch"] == "res4",
          f"model: only lane_use_depth differs (False), lane_branch stays res4 ({sorted(mdiff)})")
    check(b["loss"] == a["loss"] and b["steps"] == 1500 and b["init_from"] == a["init_from"] == "runs/t20_base_s0/ckpt_00006000.pt" and b["seed"] == a["seed"] == 0,
          "loss, steps (1500), warm start (t20_base_s0 ckpt 6000) and seed 0 are the control's")
    check(not check_config(ModelConfig(**{k: v for k, v in b["model"].items() if k in ModelConfig.__dataclass_fields__})), "the model section passes check_config")

print(f"\n{'FAILED: ' + str(len(fails)) if fails else 'all ok'} ({time.time() - t0:.0f} s)")
sys.exit(1 if fails else 0)
