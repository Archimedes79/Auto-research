"""The stride-4 lane branch (L4, training/lane_branch.py) on the dual camera features (card t20, plan 2026-11-03 section 2); CPU, seconds, no data.

  A  the rule (model.check_config): the branch reads the current image, the depth and the calibration of the FIRST feature source, so that source
     must be the DualBEV family: 'dualbev' and 'dual' pass, 'bevfusion' alone is refused; camera_off and lift 's' stay refused.
  B  nothing else changes: the option off gives the state_dict keys and shapes of before (golden hash of the tiny 'dual' model, made from the code
     of commit 3d8edcb, before the rule changed); on, the keys are those plus `lane.*` and every shared tensor keeps its shape. With the real
     checkpoint runs/t16_full_s0 (skipped without it) the full t20 model differs from t16.full by exactly `lane.*`, `head_1m.*` and the radar
     aux head's last layer (radar_stationary off).
  C  zero start: the model with the branch, loaded with the weights of the model without it, gives the same outputs exactly (the t7 rule, now dual).
  D  gradient: with the zero start broken by small values, a loss on the maps reaches every trainable parameter of the branch and the trunk, the
     frozen ResNet gets none and stays in eval mode under .train().
  E  geometry: the branch's lift gets the first source's depth, the DualBEV `ida` and the image size, never prob2 / ida2; its contribution does
     not change when the second source's tensors change, and does when the image changes.
  F  mirror: a flipped sample gets the mirrored contribution (flip_aug 0.5 is on in the config), the other sample is unchanged.
  G  configs/t20_base.json is configs/t16_full.json with exactly the changes of the card (and its model builds).
Run (from the bev_fusion folder): wsl ~/miniconda3/envs/gcar/bin/python tests/test_lane_branch_dual.py [--real]
(--real: the full-size t20 model on the CPU with the weights of runs/t16_full_s0: outputs with and without the branch are equal; about 15 s)"""
import hashlib
import json
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for sub in ("tests", "training"):
    sys.path.insert(0, os.path.join(ROOT, sub))
from feature_cache import CachedBEVFusionModel  # noqa: E402
from model import ModelConfig, check_config  # noqa: E402
from nuscenes_dataset import DST_HW  # noqa: E402
from test_slots import batch_for  # noqa: E402

torch.set_num_threads(2)
fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


CKPT = os.path.join(ROOT, "runs", "t16_full_s0", "ckpt_00006000.pt")
T20 = os.path.join(ROOT, "training", "configs", "t20_base.json")
T16 = os.path.join(ROOT, "training", "configs", "t16_full.json")
GRID = {"x_max": 16.0, "y_min": -8.0, "y_max": 8.0}                 # 32 x 32 cells
TINY = dict(pretrained="", trunk_pretrained="", lane_pretrained="", grid_kwargs=GRID, oversample=1, c_cam=16, c_rad=8, c_fuse=24,
            trunk_width=0.25, p_drop=0.0, camera_features="dual")


def rows(m):
    return [f"{k} {tuple(v.shape)}" for k, v in m.state_dict().items()]


def build(**kw):
    torch.manual_seed(0)
    return CachedBEVFusionModel(ModelConfig(**dict(TINY, **kw)))


def checkpoint_weights():
    """the model's tensors of the stored checkpoint of runs/t16_full_s0 (its `state` holds them with the prefix `model.`, next to the loss's)"""
    st = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)["state"]
    return {k[len("model."):]: v for k, v in st.items() if k.startswith("model.")}


def refused(**kw):
    try:
        check_config(ModelConfig(**dict(TINY, **kw)))
        return False
    except ValueError:
        return True


t0 = time.time()
# ------------------------------------------------------------------------------------------------ A
print("A: the rule")
check(all(not refused(camera_features=cf, lane_branch=lb) for cf in ("dualbev", "dual") for lb in ("res4", "rgb")),
      "check_config accepts lane_branch res4 / rgb with camera_features 'dualbev' and 'dual'")
check(refused(camera_features="bevfusion", lane_branch="res4"), "refused with 'bevfusion' alone (its depth bins, stride and crop are not the lane image's)")
check(refused(lane_branch="res4", camera_off=True), "refused with camera_off (the branch would still see the image)")
check(refused(camera_features="dualbev", lane_branch="res4", lift="s"), "refused with lift 's' (the branch needs the depth probabilities)")
check(not refused(camera_features="dual") and not refused(camera_features="dual", lane_branch=""), "'dual' without the branch is accepted as before")

# ------------------------------------------------------------------------------------------------ B
print("B: nothing else changes")
off, on = build(), build(lane_branch="res4")
r_off = rows(off)
digest = hashlib.sha256("\n".join(r_off).encode()).hexdigest()[:16]
check(len(r_off) == 223 and digest == "0f169833b881bec8", f"option off: state_dict keys + shapes of the dual model = before ({len(r_off)} tensors, {digest})")
extra = set(on.state_dict()) - set(off.state_dict())
check(extra and all(k.startswith("lane.") for k in extra) and not set(off.state_dict()) - set(on.state_dict())
      and all(on.state_dict()[k].shape == v.shape for k, v in off.state_dict().items()),
      f"option on: the same keys and shapes plus {len(extra)} tensors, all `lane.*` (frozen ResNet stem + layer 1, reduce, collapse, mix)")
check(off.lane is None and all(not k.startswith("lane.") for k in off.state_dict()), "option off: no branch module")

if os.path.exists(CKPT):
    ref = {k: tuple(v.shape) for k, v in checkpoint_weights().items()}

    def full(path, **kw):                                             # the model of a config, no pretrained weights (shapes only)
        c = json.load(open(path))["model"]
        return CachedBEVFusionModel(ModelConfig(**dict(c, pretrained="", trunk_pretrained="", lane_pretrained="", **kw)))
    m16 = full(T16)
    s16 = {k: tuple(v.shape) for k, v in m16.state_dict().items()}
    check(s16 == ref, f"t16_full.json builds the model of runs/t16_full_s0 ({len(ref)} tensors, keys and shapes equal to its checkpoint)")
    if os.path.exists(T20):
        c20 = json.load(open(T20))
        m20 = CachedBEVFusionModel(ModelConfig(**dict(c20["model"], pretrained="", trunk_pretrained="", lane_pretrained="",
                                                      centreline_1m=c20["loss"]["centreline_1m"], radar_stationary=c20["loss"]["radar_stationary"])))
        s20 = {k: tuple(v.shape) for k, v in m20.state_dict().items()}
        new = sorted(set(s20) - set(ref))
        changed = sorted(k for k in ref if k in s20 and s20[k] != ref[k])
        check(not set(ref) - set(s20) and all(k.startswith(("lane.", "head_1m.")) for k in new) and changed == ["aux_rad.net.3.bias", "aux_rad.net.3.weight"],
              f"t20 model vs t16.full: {len(new)} new tensors (lane.*, head_1m.*), the radar aux head's last layer 4 -> 3 channels, nothing missing, nothing else changed")
else:
    print("  skip no runs/t16_full_s0/ckpt_00006000.pt: the comparison with the real model")

# ------------------------------------------------------------------------------------------------ C
print("C: zero start")
img = torch.randint(0, 256, (2, 3, *DST_HW), dtype=torch.uint8, generator=torch.Generator().manual_seed(1))


def batch(cfg_model, image=img):
    b = batch_for(cfg_model.cfg)
    b["lane_img"] = image
    return b


rep = on.load_state_dict(off.state_dict(), strict=False)
check(not rep.unexpected_keys and all(k.startswith("lane.") for k in rep.missing_keys), "the shared weights of the model without the branch load into the one with it")
off.eval(), on.eval()
bt = batch(on)
with torch.no_grad():
    o_off, o_on = off(bt, dts=(0.0, 0.5))["out"], on(bt, dts=(0.0, 0.5))["out"]
d = max(float((o_off[dt]["sem"] - o_on[dt]["sem"]).abs().max()) for dt in o_off)
d = max([d] + [float((o_off[dt]["obj"][k] - o_on[dt]["obj"][k]).abs().max()) for dt in o_off for k in o_off[dt]["obj"]])
check(d == 0.0 and float(o_on[0.0]["sem"].abs().max()) > 0, f"outputs with the branch at its zero start = without it, exactly (max diff {d}) at dt 0 and +0.5 s")

# ------------------------------------------------------------------------------------------------ D
print("D: gradient")
on.train()
check(not any(p.requires_grad for p in on.lane.features.parameters()) and not on.lane.features.bn1.training and not on.lane.features.layer1[0].bn1.training,
      "the ResNet stem + layer 1 are frozen and their BatchNorm stays in eval mode under .train()")
with torch.no_grad():
    for q in on.parameters():                                         # the zero-initialised mix cuts the gradient at init by design (as test_slots does)
        if q.requires_grad and q.abs().max() == 0:
            q.normal_(0, 1e-2)
on.zero_grad()
res = on(bt, dts=(0.0,))
(res["out"][0.0]["sem"].float().pow(2).mean() + sum(v.float().pow(2).mean() for v in res["out"][0.0]["obj"].values())).backward()
trainable = [(n, q) for n, q in on.named_parameters() if n.startswith("lane.") and q.requires_grad]
dead = [n for n, q in trainable if q.grad is None or not torch.isfinite(q.grad).all() or q.grad.abs().sum() == 0]
check(len(trainable) == 10 and not dead, f"a loss on the maps reaches all {len(trainable)} trainable tensors of the branch (reduce, collapse, mix){' - none: ' + str(dead) if dead else ''}")
check(all(q.grad is None for q in on.lane.features.parameters()), "... the frozen ResNet gets none")
check(any(q.grad is not None and q.grad.abs().sum() > 0 for q in on.trunk.parameters()) and
      any(q.grad is not None and q.grad.abs().sum() > 0 for q in on.cam_merge.parameters()), "... and the trunk and the camera merge train as before")

# ------------------------------------------------------------------------------------------------ E
print("E: geometry and inputs")
on.eval()
seen, contrib = [], []
h1 = on.lane.lift.register_forward_pre_hook(lambda m, a: seen.append(a))
h2 = on.lane.register_forward_hook(lambda m, a, o: contrib.append(o.detach().clone()))
with torch.no_grad():
    on(bt, dts=(0.0,))
h1.remove()
b0 = bt
check(len(seen) == 2 and all(torch.equal(a[1], b0["prob"][i:i + 1, 0].float()) and a[5] is b0["ida"] and a[5] is not b0["ida2"] and tuple(a[4]) == tuple(DST_HW)
                             for i, a in enumerate(seen)),
      f"the branch's lift gets, per sample, the depth of source 1 (DualBEV, {tuple(b0['prob'].shape[2:])}), the DualBEV ida and the image size {tuple(DST_HW)}")
b1 = dict(bt, prob2=torch.rand_like(bt["prob2"]).softmax(2), ctx2=torch.randn_like(bt["ctx2"]))
b2 = dict(bt, lane_img=img.flip(0))
with torch.no_grad():
    on(b1, dts=(0.0,))
    on(b2, dts=(0.0,))
h2.remove()
check(torch.equal(contrib[0], contrib[1]) and float(contrib[0].abs().max()) > 0, "the contribution does not change with the second source's tensors (prob2, ctx2)")
check(float((contrib[0] - contrib[2]).abs().max()) > 1e-4 * float(contrib[0].abs().max()), "... and it does with the image")

# ------------------------------------------------------------------------------------------------ F
print("F: mirror")
twin = build(lane_branch="res4").eval()
twin.load_state_dict(on.state_dict())
with torch.no_grad():
    twin.lane.mix.weight.zero_()
    twin.lane.mix.bias.zero_()
cams = []
ha = on.fusion.register_forward_pre_hook(lambda m, a: cams.append(a[0].detach().clone()))
hb = twin.fusion.register_forward_pre_hook(lambda m, a: cams.append(a[0].detach().clone()))
delta = {}
with torch.no_grad():
    for fl in ((False, False), (True, False)):
        cams.clear()
        bf = dict(bt, flip=torch.tensor(fl))
        on(bf, dts=(0.0,))
        twin(bf, dts=(0.0,))
        delta[fl] = cams[0] - cams[1]                                  # the branch's contribution alone, as the fusion gets it
ha.remove()
hb.remove()
a_, b_ = delta[(False, False)], delta[(True, False)]
check(float(a_.abs().max()) > 0 and torch.allclose(b_[0], a_[0].flip(-1), atol=1e-6) and torch.allclose(b_[1], a_[1], atol=1e-6),
      "a flipped sample gets the mirrored lane contribution, the other sample is unchanged")

# ------------------------------------------------------------------------------------------------ G
print("G: configs/t20_base.json")
check(os.path.exists(T20), "configs/t20_base.json exists")
if os.path.exists(T20):
    a, b = json.load(open(T16)), json.load(open(T20))
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    mdiff = {k for k in set(a["model"]) | set(b["model"]) if a["model"].get(k) != b["model"].get(k)}
    ldiff = {k for k in set(a["loss"]) | set(b["loss"]) if a["loss"].get(k) != b["loss"].get(k)}
    check(diff == {"_about", "name", "split", "lane_images", "model", "loss", "train_scenes", "val_scenes"} and "train_scenes" not in b and "val_scenes" not in b
          and b["split"] == "large" and len(a["train_scenes"]) == 428 and len(a["val_scenes"]) == 82,
          f"top level: only _about, name, split (large, instead of the 428 / 82 inline scene names of t16_full.json) and lane_images differ ({sorted(diff)})")
    check(mdiff == {"lane_branch", "centreline_1m", "radar_stationary"} and b["model"]["lane_branch"] == "res4" and "lane_pretrained" not in b["model"]
          and b["model"]["centreline_1m"] is True and b["model"]["radar_stationary"] is False,
          f"model: lane_branch res4 (lane_pretrained stays its default), centreline_1m on, radar_stationary off ({sorted(mdiff)})")
    check(ldiff == {"hygiene_2026_11_02", "centreline_1m", "radar_stationary"} and b["loss"]["hygiene_2026_11_02"] is True and b["loss"]["centreline_1m"] is True
          and b["loss"]["radar_stationary"] is False and b["loss"]["w_obj_vel"] == a["loss"]["w_obj_vel"],
          f"loss: hygiene on, centreline_1m on, radar_stationary off, w_obj_vel unchanged ({sorted(ldiff)})")
    mc = ModelConfig(**b["model"])
    check_config(mc)
    check(mc.camera_features == "dual" and mc.lane_branch == "res4" and mc.centreline_1m and not mc.radar_stationary,
          "the model fields of the config pass check_config (dual + res4, 1 m head, no stationary channel) and agree with the loss")

# ------------------------------------------------------------------------------------------------ --real
if "--real" in sys.argv and os.path.exists(CKPT) and os.path.exists(T20):
    print("--real: the full-size t20 model with the weights of runs/t16_full_s0 (CPU)")
    c20 = json.load(open(T20))
    kw = dict(c20["model"], pretrained="", trunk_pretrained="", lane_pretrained="", centreline_1m=False, radar_stationary=True, lane_branch="")
    ref_m = CachedBEVFusionModel(ModelConfig(**kw)).eval()
    lane_m = CachedBEVFusionModel(ModelConfig(**dict(kw, lane_branch="res4"))).eval()
    weights = checkpoint_weights()
    r1 = ref_m.load_state_dict(weights, strict=True)
    r2 = lane_m.load_state_dict(weights, strict=False)
    check(not r1.missing_keys and not r1.unexpected_keys and not r2.unexpected_keys and all(k.startswith("lane.") for k in r2.missing_keys),
          "the weights of runs/t16_full_s0 load strictly into the model without the branch, and into the one with it (only lane.* missing)")
    bb = batch_for(ref_m.cfg, B=1)
    bb["lane_img"] = img[:1]
    with torch.no_grad():
        oa, ob = ref_m(bb, dts=(0.0,))["out"][0.0], lane_m(bb, dts=(0.0,))["out"][0.0]
    dd = max(float((oa["sem"] - ob["sem"]).abs().max()), max(float((oa["obj"][k] - ob["obj"][k]).abs().max()) for k in oa["obj"]))
    check(dd == 0.0 and float(oa["sem"].abs().max()) > 0, f"real weights: outputs with the branch at zero start = without it, exactly (max diff {dd})")

print(f"test_lane_branch_dual " + ("FAILED: " + "; ".join(fails) if fails else f"passed ({time.time() - t0:.0f} s)"))
sys.exit(1 if fails else 0)
