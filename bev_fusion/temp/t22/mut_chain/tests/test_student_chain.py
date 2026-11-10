"""The camera student in the chain (t22 stage B): ModelConfig.camera_student / student_init / student_trainable / student_frames, the cached model's frame-0 splice, the
anchor loss of train.forward_loss. CPU, 2 threads, about 30 s, no data, no ImageNet weights (a random ResNet-18 student, and a stub registered in model.STUDENTS for the exact
oracles); the encoder itself is tests/test_student_encoder.py.

  A  OFF is the code before: the camera BEVs and the whole forward of the cached model equal a copy of the old camera_bevs bit for bit (dualbev, bevfusion, dual, cam_adapter,
     camera_off); no `student` module, no `student.*` key, no `student_side` in the result
  B  SPLICE: frame 0 of both sources comes from the student, the older frames keep their cached teacher outputs. Oracle: the BEV equals the BEV of the model WITHOUT the student on
     the batch whose frame 0 was replaced by the student's outputs (a stub that is a function of the image: wrong sample, wrong frame, swapped sources all differ), F = 1, 2, 3;
     a stub that returns the cached frame 0 gives the old BEV exactly; changing the stub's frame 0 changes frame 0 only; cached garbage in frame 0 changes nothing (replaced), in
     frame 1 only frames >= 1; a missing lane_img is a KeyError naming lane_images
  C  student_side: the student's tensors and the cached teacher's (detached, even when the cached batch requires grad) of frame 0; taken by _forward (res["student_side"], cleared)
  D  the real student (ResNet-18, random): student_init loads the NEWEST checkpoint's EMA tensors (and the integer buffers from its state), strictly; trainable: a loss on the camera
     BEV reaches every student tensor, BatchNorm statistics do not move under model.train() / ema.applied (a plain StudentEncoder's do); frozen: no gradient, no graph, no
     `student.` key in the checkpoint state; built last (every other tensor equal with the option on or off); lr_groups give the student its own rate; warm start keeps the
     student as loaded; a checkpoint of the chain without a student loads into the chain with a frozen one (the stand-in evaluation path)
  E  config: each refusal of model.check_config on its own, lane_branch stays allowed, the base model refuses the option, student_frames 'all' = NotImplementedError; train.build
     refusals (feature cache, lane_images, anchor without a trainable student, schedule range)
  F  anchor: zero for student = teacher and positive otherwise (by hand), the weight schedule at 0 / until/2 / until / 1, forward_loss adds weight x anchor to the total in
     training (with its gradient reaching the student) and only logs it in validation, run_clip logs it
  G  the camera-off class of bug: the evaluation's zero_cam and the shuffle control change the camera BEV of a model WITH the student, and zeroing / swapping only lane_img does
  H  the lane branch lifts with the STUDENT's DualBEV depth of frame 0 (not the cached one): its contribution changes when only the student's p16 changes and not when only the cached
     prob changes; the gradient of the lane branch reaches a trainable student; with the option off the branch still gets the cached depth (section A)
  I  grad_clip_split: 'student' names exactly the student's trainable tensors (refused for a frozen one); with it a huge student gradient leaves the others' gradient as it is, the
     global clip scales them down
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_student_chain.py"""
import copy
import dataclasses
import os
import sys
import tempfile
import time
import types

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "training"), os.path.join(ROOT, "tests"), os.path.join(ROOT, "tests", "support"), ROOT, os.path.join(ROOT, "..")]
import model as M  # noqa: E402
import student_encoder as S  # noqa: E402
import synthetic_world as sw  # noqa: E402
import train as T  # noqa: E402
from feature_cache import CachedBEVFusionModel  # noqa: E402
from losses import BEVLoss, synthetic_gt  # noqa: E402
from model import CAMERAS, BEVFusionModel, ModelConfig, check_config  # noqa: E402
from test_slots import BASE, batch_for  # noqa: E402
from trainkit import checkpoint as tk_ckpt  # noqa: E402
from trainkit.optim import clip_gradients, clip_groups, param_groups  # noqa: E402
from trainkit.schedule import lr_factor, scheduled_lr  # noqa: E402

torch.set_num_threads(2)
fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def raises(exc, fn, *needles):
    """True if fn() raises exc and the message holds every needle"""
    try:
        fn()
    except exc as e:
        return all(n in str(e) for n in needles)
    return False


# ------------------------------------------------------------------------------------------------ helpers
class Stub(nn.Module):
    """stands in for the student: every output is a fixed function of the IMAGE (its mean, per sample), so an output says which image it was made from; with `echo` set it returns that dict"""
    HEAD_OF = S.StudentEncoder.HEAD_OF

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.scale = nn.Parameter(torch.ones(()))
        self.base = {"logit16": torch.randn(118, 16, 44, generator=g), "ctx16": torch.randn(80, 16, 44, generator=g),
                     "logit8": torch.randn(118, 32, 88, generator=g), "ctx8": torch.randn(80, 32, 88, generator=g)}
        self.echo = None

    def keep_bn_eval(self, flag=True):
        return self

    def forward(self, img):
        if self.echo is not None:
            return self.echo
        g = (img.float().mean((1, 2, 3)) / 255).view(-1, 1, 1, 1)
        out = {}
        for k in ("16", "8"):
            out["logit" + k] = (self.base["logit" + k][None] + 6 * g) * self.scale
            out["ctx" + k] = (self.base["ctx" + k][None] + 3 * g) * self.scale
            out["prob" + k] = out["logit" + k].softmax(1)
        return out


M.STUDENTS["stub"] = Stub
DUAL = dict(BASE, camera_features="dual")


def images(B=2, seed=1):
    """uint8 key-frame images with a different brightness per sample (the stub reads the mean) and texture (the real student needs it)"""
    g = torch.Generator().manual_seed(seed)
    base = torch.linspace(50, 190, B).view(B, 1, 1, 1)
    return (base + torch.randint(0, 30, (B, 3, 256, 704), generator=g).float()).clamp(0, 255).to(torch.uint8)


def build(cls=CachedBEVFusionModel, seed=0, **kw):
    torch.manual_seed(seed)
    return cls(ModelConfig(**dict(DUAL, **kw)))


def data(F, B=2, **kw):
    b = batch_for(ModelConfig(**dict(DUAL, **kw)), B=B, Fn=F)
    b["lane_img"] = images(B)
    return b


def old_camera_bevs(self, batch):
    """the camera_bevs of the cached model before the student (commit ff8ba947), verbatim"""
    prob = batch["prob"].float()
    B, Fn = prob.shape[:2]
    if self.cfg.camera_off:
        return torch.zeros(B, Fn, self.cfg.c_cam, self.grid.nx, self.grid.ny, device=prob.device)
    bevs = []
    for i, src in enumerate(CAMERAS[self.cfg.camera_features]):
        s = src.suffix
        p, ctx = (prob if i == 0 else batch["prob" + s].float()), batch["ctx" + s].float()
        if i == 0 and self.cam_adapter is not None:
            ctx = self.cam_adapter(ctx.flatten(0, 1)).view_as(ctx)
        lift, collapse = getattr(self, "lift" + s), getattr(self, "cam_reduce" + s)
        keys = self._lift_keys(batch, batch["hw"], "ida" + s)
        bevs.append(torch.stack([collapse(lift(ctx[b], p[b], batch["K"][b, 0], batch["T_cam_car"][b, 0], batch["hw"], batch["ida" + s],
                                               key=None if keys is None else keys[b])) for b in range(B)]))
    if len(bevs) == 1:
        return bevs[0]
    return self.cam_merge(torch.cat(bevs, 2).flatten(0, 1)).view_as(bevs[0])


def frame0_replaced(b, out):
    """the batch with frame 0 of both sources replaced by the student's outputs `out` (the oracle of the splice)"""
    r = dict(b)
    for suffix, k in (("", "16"), ("2", "8")):
        r["prob" + suffix] = torch.cat([out["prob" + k].float()[:, None], b["prob" + suffix][:, 1:].float()], 1)
        r["ctx" + suffix] = torch.cat([out["ctx" + k].float()[:, None], b["ctx" + suffix][:, 1:].float()], 1)
    return r


def echo_of(b):
    return {"prob16": b["prob"][:, 0], "ctx16": b["ctx"][:, 0], "logit16": b["prob"][:, 0].log(), "prob8": b["prob2"][:, 0], "ctx8": b["ctx2"][:, 0], "logit8": b["prob2"][:, 0].log()}


def write_ckpt(path, student, scale=1.0):
    """a distillation run's checkpoint (distill_student / trainkit layout): `state` = the student's, `ema` = its float tensors x scale (so the EMA is told from the state)"""
    ema = {k: v * scale for k, v in student.state_dict().items() if v.dtype.is_floating_point}
    torch.save({"state": {"model." + k: v for k, v in student.state_dict().items()}, "ema": ema, "step": 7, "cfg": {"backbone": "resnet18"}}, path)
    return ema


def with_gt(batch, grid):
    """synthetic GT at -0.5 / 0 / +0.5 s (tests/test_motion_joint.py)"""
    B_ = sw.Box
    scene = sw.Scene([B_(6.0, -1.75, 4.5, 1.8, 1.5, 0.0, 8.0, 0.0, "car", 10.0), B_(10.0, 4.0, 4.5, 1.8, 1.5, 0.0, 0.0, 0.0, "car", 10.0)], lanes=[-5.25, -1.75, 1.75, 5.25], v=8.0, omega=0.05)
    side = {"prev": synthetic_gt(scene.step(-0.5), grid), "cur": synthetic_gt(scene, grid), "next": synthetic_gt(scene.step(0.5), grid)}
    return dict(batch, gt=[side] * batch["v"].shape[0])


t0 = time.time()
tmp = tempfile.mkdtemp(prefix="student_chain_", dir=os.path.join(ROOT, "temp") if os.path.isdir(os.path.join(ROOT, "temp")) else None)

# ------------------------------------------------------------------------------------------------ A
print("A: OFF is the code before")
for name, kw in (("dualbev", dict(camera_features="dualbev")), ("bevfusion", dict(camera_features="bevfusion")), ("dual", {}), ("dual + cam_adapter", dict(camera_features="dualbev", cam_adapter=True)),
                 ("camera_off", dict(camera_off=True))):
    m = build(**kw).eval()
    b = data(2, **kw)
    with torch.no_grad():
        new = m.camera_bevs(b)
        old = old_camera_bevs(m, b)
        o_new = m(b, dts=(0.0, 0.5))
        m.camera_bevs = types.MethodType(old_camera_bevs, m)
        o_old = m(b, dts=(0.0, 0.5))
    check(torch.equal(new, old), f"{name}: camera BEVs equal the old code's (max abs diff 0)")
    check(all(torch.equal(o_new["out"][dt]["sem"], o_old["out"][dt]["sem"]) and torch.equal(o_new["out"][dt]["obj"]["hm"], o_old["out"][dt]["obj"]["hm"]) for dt in (0.0, 0.5)),
          f"{name}: the whole forward at dt 0 / +0.5 equals the old code's")
    check(m.student is None and not any(k.startswith("student.") for k in m.state_dict()) and "student_side" not in o_new and m._student_side is None,
          f"{name}: no student module, no `student.*` key, no student_side in the result")
lm, bl, seen = build(lane_branch="rgb", lane_pretrained="").eval(), data(2), []
hk = lm.lane.lift.register_forward_pre_hook(lambda mod, a: seen.append(a[1].detach().clone()))
with torch.no_grad():
    lm(bl, dts=(0.0,))
hk.remove()
check(len(seen) == 2 and all(torch.equal(s, bl["prob"][i:i + 1, 0].float()) for i, s in enumerate(seen)), "lane branch, option off: it still lifts with the cached DualBEV depth of frame 0 (bit for bit)")

# ------------------------------------------------------------------------------------------------ B
print("B: splice")
for F in (1, 2, 3):
    off, on = build(), build(camera_student="stub", student_trainable=True)
    b = data(F)
    with torch.no_grad():
        want = off.eval().camera_bevs(frame0_replaced(b, on.student(b["lane_img"])))
        got = on.eval().camera_bevs(b)
        plain = off.camera_bevs(b)
    check(torch.equal(got, want), f"F = {F}: the BEV equals the student-free model's on the batch with frame 0 replaced by the student's outputs (images differ per sample)")
    check(not torch.allclose(got[:, 0], plain[:, 0]), f"F = {F}: the student changes frame 0")
    if F > 1:
        check(torch.equal(got[:, 1:], plain[:, 1:]), f"F = {F}: frames >= 1 keep their cached teacher outputs (bit for bit)")
    on.student.echo = echo_of(b)
    with torch.no_grad():
        echo = on.camera_bevs(b)
    check(torch.equal(echo, plain), f"F = {F}: a student that returns the cached frame 0 gives the old BEV exactly")
    on.student.echo = {k: (v + 0.5 if k.startswith("ctx") else v) for k, v in echo_of(b).items()}
    with torch.no_grad():
        moved = on.camera_bevs(b)
    check(not torch.equal(moved[:, 0], plain[:, 0]) and (F == 1 or torch.equal(moved[:, 1:], plain[:, 1:])), f"F = {F}: changing the student's frame 0 changes frame 0 only")
    on.student.echo = None
    g0 = dict(b, prob=torch.cat([torch.rand_like(b["prob"][:, :1]), b["prob"][:, 1:]], 1), ctx2=torch.cat([torch.randn_like(b["ctx2"][:, :1]), b["ctx2"][:, 1:]], 1))
    with torch.no_grad():
        again = on.camera_bevs(g0)
    check(torch.equal(again, got), f"F = {F}: garbage in the cached frame 0 changes nothing (it is replaced)")
    if F > 1:
        g1 = dict(b, ctx=torch.cat([b["ctx"][:, :1], torch.randn_like(b["ctx"][:, 1:])], 1))
        with torch.no_grad():
            other = on.camera_bevs(g1)
        check(torch.equal(other[:, 0], got[:, 0]) and not torch.equal(other[:, 1:], got[:, 1:]), f"F = {F}: garbage in the cached frame 1 changes frames >= 1 only")
nolane = {k: v for k, v in data(1).items() if k != "lane_img"}
check(raises(KeyError, lambda: on.camera_bevs(nolane), "lane_images", "lane_img"), "a batch without lane_img: KeyError that names TrainConfig.lane_images")

# ------------------------------------------------------------------------------------------------ C
print("C: student_side")
on = build(camera_student="stub", student_trainable=True).eval()
b = data(2)
b["prob"].requires_grad_()                                      # a cached batch that requires grad must not make the teacher side a graph
res = on(b, dts=(0.0,))
side = res["student_side"]
want = on.student(b["lane_img"])
check(sorted(side) == sorted(["p16", "c16", "l16", "p8", "c8", "l8", "t_p16", "t_c16", "t_p8", "t_c8"]), "student_side holds p/c/l 16 and 8 and the teacher's t_p/t_c 16 and 8")
check(all(torch.equal(side[k], want[n + k[1:]]) for k, n in (("p16", "prob"), ("c16", "ctx"), ("l16", "logit"), ("p8", "prob"), ("c8", "ctx"), ("l8", "logit"))),
      "the student's entries are its outputs on the sample's own image (16 = H16, 8 = H8)")
check(torch.equal(side["t_p16"], b["prob"][:, 0]) and torch.equal(side["t_c16"], b["ctx"][:, 0]) and torch.equal(side["t_p8"], b["prob2"][:, 0]) and torch.equal(side["t_c8"], b["ctx2"][:, 0]),
      "the teacher's entries are the cached frame 0 of the matching source (DualBEV 16, BEVFusion 8)")
check(not any(side[k].requires_grad for k in ("t_p16", "t_c16", "t_p8", "t_c8")) and side["l16"].requires_grad, "the teacher side is detached (the cached batch required grad), the student's is live")
check(on._student_side is None, "_forward took the side entry and cleared it")

# ------------------------------------------------------------------------------------------------ D
print("D: the real student")
real = S.StudentEncoder("resnet18", pretrained=False)
ck_a, ck_b = os.path.join(tmp, "run", "ckpt_00000007.pt"), os.path.join(tmp, "run", "ckpt_00000009.pt")
os.makedirs(os.path.dirname(ck_a))
ema_a = write_ckpt(ck_a, real, scale=1.05)
torch.manual_seed(5)
real2 = S.StudentEncoder("resnet18", pretrained=False)
ema_b = write_ckpt(ck_b, real2, scale=0.95)
frozen = build(camera_student="resnet18", student_init=os.path.dirname(ck_a))
st = frozen.student.state_dict()
check(all(torch.equal(st[k], v) for k, v in ema_b.items()) and all(torch.equal(st[k], v) for k, v in real2.state_dict().items() if not v.dtype.is_floating_point),
      "student_init = a run folder: the NEWEST checkpoint's EMA tensors (not its state, not the older one), the integer buffers from its state")
file_loaded = build(camera_student="resnet18", student_init=ck_a).student.state_dict()
check(all(torch.equal(file_loaded[k], v) for k, v in ema_a.items()), "student_init = a checkpoint file")
bad = dict(torch.load(ck_a, weights_only=False))
bad["ema"] = {k: v for k, v in bad["ema"].items() if k != "head8.out.bias"}
bad["state"] = {k: v for k, v in bad["state"].items() if k != "model.head8.out.bias"}
torch.save(bad, os.path.join(tmp, "lacking.pt"))
check(raises(ValueError, lambda: build(camera_student="resnet18", student_init=os.path.join(tmp, "lacking.pt")), "head8.out.bias"), "a checkpoint that lacks a student tensor is refused (strict)")
check(raises(FileNotFoundError, lambda: build(camera_student="resnet18", student_init=os.path.join(tmp, "nothing"))), "student_init without a checkpoint: FileNotFoundError")

# frozen student: no gradient, no graph, not in the checkpoint state
b = data(2)
fro = build(camera_student="resnet18", student_init=ck_a)
fro.train()
out = fro.camera_bevs(b)
out.sum().backward()
check(all(not p.requires_grad for p in fro.student.parameters()) and all(p.grad is None for p in fro.student.parameters()), "frozen: no student tensor requires grad or got a gradient")
check(not fro._student_side["l16"].requires_grad, "frozen: the student ran without a graph")
loss = BEVLoss(grid=fro.grid)
check(T._external(fro) == ("camera.", "student.") and not any(k.startswith("model.student.") for k in tk_ckpt.model_state(fro, loss, T._external(fro))), "frozen: `student.` is external, no student key in the checkpoint state")
check(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in fro.cam_reduce.parameters()), "frozen: the lifts and reductions behind it still train")

# trainable: every tensor gets a gradient, BatchNorm statistics fixed
tr = build(camera_student="resnet18", student_trainable=True)
bn = [mod for mod in tr.student.modules() if isinstance(mod, nn.modules.batchnorm._BatchNorm)]
stats = lambda mods: [(x.running_mean.clone(), x.running_var.clone(), x.num_batches_tracked.clone()) for x in mods]  # noqa: E731
before = stats(bn)
tr.train()
tr.camera_bevs(b).sum().backward()
dead = [n for n, p in tr.student.named_parameters() if p.grad is None or float(p.grad.abs().max()) == 0]
check(not dead, f"trainable: a loss on the camera BEV reaches every one of the {sum(1 for _ in tr.student.parameters())} student tensors" + (f" (no gradient: {dead[:3]})" if dead else ""))
check(all(not x.training for x in bn) and all(torch.equal(a[0], x.running_mean) and torch.equal(a[1], x.running_var) and torch.equal(a[2], x.num_batches_tracked) for a, x in zip(before, bn)),
      f"trainable: after model.train() and a forward + backward the {len(bn)} BatchNorm layers are in eval mode and their statistics did not move")
ema = T.EMA(tr, 0.99)
with ema.applied(tr):
    pass
check(all(not x.training for x in bn), "trainable: BatchNorm stays in eval mode after ema.applied (model.eval() / model.train() round trip)")
plain = S.StudentEncoder("resnet18", pretrained=False).train()
pb = [mod for mod in plain.modules() if isinstance(mod, nn.modules.batchnorm._BatchNorm)]
pre = stats(pb)
plain(images(2))
check(any(not torch.equal(a[0], x.running_mean) for a, x in zip(pre, pb)), "control: a plain student in train mode DOES move the statistics (the check above can fail)")
check(T._external(tr) == ("camera.",) and any(k.startswith("model.student.") for k in tk_ckpt.model_state(tr, loss, T._external(tr))), "trainable: the student is part of the checkpoint state")

# built last: the other tensors are the same with the option on or off
sd_on, sd_off = build(camera_student="resnet18", student_trainable=True).state_dict(), build().state_dict()      # fresh models: the forward above moved BatchNorm statistics of the lifts
check(set(sd_on) - set(sd_off) and all(k.startswith("student.") for k in set(sd_on) - set(sd_off)) and not set(sd_off) - set(sd_on) and all(torch.equal(sd_on[k], v) for k, v in sd_off.items()),
      "built last: the option adds `student.*` tensors only and every other tensor is equal at one seed")

# the student's own learning rate through the engine's groups (the function train() calls)
cfg = T.TrainConfig(device="cpu", lr=3e-4, warmup=100, steps=1500, lr_groups={"student": {"lr": 3e-5}})
groups = param_groups(tr, loss, cfg.lr, cfg.weight_decay, cfg.lr_groups)
f = lr_factor(700, cfg.warmup, cfg.steps, cfg.lr_min_frac)
student_ids = {id(p) for p in tr.student.parameters()}
mine = [g for g in groups if g.get("lr_group") == "student"]
rest = [g for g in groups if g.get("lr_group") != "student"]
check(mine and {id(p) for g in mine for p in g["params"]} == student_ids and all(abs(scheduled_lr(g, 700, cfg, f) - 3e-5 * f) < 1e-12 for g in mine)
      and all(abs(scheduled_lr(g, 700, cfg, f) - 3e-4 * f) < 1e-12 for g in rest) and not any(id(p) in student_ids for g in rest for p in g["params"]),
      "lr_groups {'student': 3e-5}: every student tensor in its own groups at 3e-5 x the schedule, nothing else touched")

# warm start from a chain without a student: the student stays as loaded from student_init, everything else is the checkpoint's
src = build(seed=3)
src_state = tk_ckpt.model_state(src, loss, ())
ck = {"state": src_state, "ema": {k[6:]: v for k, v in src_state.items() if k.startswith("model.") and v.dtype.is_floating_point}}
for label, kw in (("frozen", dict(student_init=ck_a)), ("trainable", dict(student_init=ck_a, student_trainable=True))):
    tgt = build(camera_student="resnet18", seed=9, **kw)
    kept = {k: v.clone() for k, v in tgt.student.state_dict().items()}
    fresh = tk_ckpt.warm_start("x.pt", tgt, copy.deepcopy(loss), ck=ck)
    check(fresh and all(k.startswith("student.") for k in fresh) and all(torch.equal(tgt.student.state_dict()[k], v) for k, v in kept.items())
          and all(torch.equal(tgt.state_dict()[k], v) for k, v in src.state_dict().items()), f"warm start ({label} student): the student keeps its loaded weights (its keys are the fresh ones), the rest is the checkpoint's")

# the stand-in evaluation path (bev_adapter.load_run): a chain checkpoint WITHOUT a student loads into the chain with a frozen one
cfgT = T.TrainConfig(device="cpu", lr=1e-4, steps=10)
ck_chain = os.path.join(tmp, "chain.pt")
adamw = lambda model, lo: torch.optim.AdamW(param_groups(model, lo, 1e-4, 0.01), lr=1e-4)  # noqa: E731
T.save_checkpoint(ck_chain, src, loss, adamw(src, loss), T.EMA(src, 0.99), 5, cfgT)
host, host_loss = build(camera_student="resnet18", student_init=ck_a, seed=11), copy.deepcopy(loss)
kept = {k: v.clone() for k, v in host.student.state_dict().items()}
host_ema = T.EMA(host, 0.99)
step, _ = T.load_checkpoint(ck_chain, host, host_loss, adamw(host, host_loss), host_ema)
host_ema.apply(host)
check(step == 5 and all(torch.equal(host.state_dict()[k], v) for k, v in src.state_dict().items()) and all(torch.equal(host.student.state_dict()[k], v) for k, v in kept.items()),
      "a chain checkpoint without a student loads into the chain with a frozen student (EMA applied): chain weights from the checkpoint, the student as loaded")
check(raises(RuntimeError, lambda: T.load_checkpoint(ck_chain, tr, copy.deepcopy(loss), adamw(tr, loss), T.EMA(tr, 0.99)), "student"),
      "...but not into one with a TRAINABLE student (its weights belong to the state: the checkpoint lacks them)")

# ------------------------------------------------------------------------------------------------ E
print("E: configuration")
refuse = lambda **kw: raises(ValueError, lambda: check_config(ModelConfig(**dict(DUAL, camera_student="stub", student_trainable=True, **kw))))  # noqa: E731
check(not raises(ValueError, lambda: check_config(ModelConfig(**dict(DUAL, camera_student="stub", student_trainable=True)))), "the plain option is accepted")
check(refuse(camera_features="dualbev") and refuse(camera_features="bevfusion"), "refused with camera_features 'dualbev' / 'bevfusion' (the student replaces both sources)")
check(raises(ValueError, lambda: check_config(ModelConfig(**dict(DUAL, camera_student="stub", student_trainable=True, lift="s"))), "lift 's'"), "refused with lift 's' (by the rule of 'dual', which the student needs anyway)")
check(raises(ValueError, lambda: check_config(ModelConfig(**dict(DUAL, camera_student="stub", student_trainable=True, cam_adapter=True))), "cam_adapter"), "refused with cam_adapter (the same rule)")
check(raises(ValueError, lambda: check_config(ModelConfig(**dict(DUAL, camera_student="stub", student_trainable=True, camera_off=True))), "camera_off"), "refused with camera_off")
check(raises(ValueError, lambda: check_config(ModelConfig(**dict(DUAL, camera_student="nope", student_trainable=True))), "camera_student='nope'", "stub"), "an unknown kind is refused and the kinds are named")
check(raises(ValueError, lambda: check_config(ModelConfig(**dict(DUAL, camera_student="stub", student_frames="oldest", student_trainable=True))), "student_frames"), "an unknown student_frames is refused")
check(raises(NotImplementedError, lambda: check_config(ModelConfig(**dict(DUAL, camera_student="stub", student_frames="all", student_trainable=True))), "sweep-image cache"),
      "student_frames 'all' = NotImplementedError naming the missing sweep-image cache")
check(raises(ValueError, lambda: check_config(ModelConfig(**dict(DUAL, camera_student="stub"))), "frozen", "student_init"), "a frozen student without student_init is refused")
check(raises(ValueError, lambda: check_config(ModelConfig(**dict(DUAL, student_trainable=True))), "without camera_student"), "student_init / student_trainable without camera_student: refused")
check(raises(ValueError, lambda: check_config(ModelConfig(**dict(DUAL, camera_student="stub", student_trainable=True, cam_adapter=True, camera_off=True))), "cam_adapter", "camera_off"),
      "several problems: one ValueError names them all")
check(not raises(ValueError, lambda: check_config(ModelConfig(**dict(DUAL, camera_student="stub", student_trainable=True, lane_branch="rgb")))),
      "lane_branch stays allowed (the t21 chain has it; section H: the branch then lifts with the student depth)")
check(raises(ValueError, lambda: BEVFusionModel(ModelConfig(**dict(DUAL, camera_student="stub", student_trainable=True))), "cached chain"), "the live-camera model refuses the option")


def tc(model=None, **kw):
    """a TrainConfig whose model has the stub student that trains (cache folders are only named: build() reads nothing)"""
    d = dict(name="x", device="cpu", feature_cache="c1", feature_cache2="c2", lane_images="img", model={**DUAL, "camera_student": "stub", "student_trainable": True, **(model or {})})
    return T.TrainConfig(**{**d, **kw})


check(isinstance(T.build(tc(student_anchor=1.0))[0].student, Stub), "train.build: the complete config builds the cached model with its student")
check(raises(ValueError, lambda: T.build(tc(feature_cache="")), "feature_cache") and raises(ValueError, lambda: T.build(tc(lane_images="")), "lane_images"), "train.build: no feature cache / no lane_images is refused")
check(raises(ValueError, lambda: T.build(T.TrainConfig(name="x", device="cpu", feature_cache="c", model=dict(BASE), student_anchor=1.0)), "student_anchor")
      and raises(ValueError, lambda: T.build(tc(model=dict(student_trainable=False, student_init=ck_a), student_anchor=1.0)), "student_anchor"), "train.build: an anchor without a trainable student is refused")
check(all(raises(ValueError, lambda kw=kw: T.build(tc(**kw)), "student_anchor") for kw in (dict(student_anchor=-1.0), dict(student_anchor_floor=1.5), dict(student_anchor_until=0.0), dict(student_anchor_until=1.5))),
      "train.build: a negative weight, a floor outside 0..1, an until outside (0, 1] are refused")

# ------------------------------------------------------------------------------------------------ F
print("F: the anchor")
start, floor, until = 2.0, 0.1, 0.5
w = lambda fr: S.student_anchor_weight(fr, start, floor, until)  # noqa: E731
check(abs(w(0.0) - 2.0) < 1e-12 and abs(w(0.25) - 1.1) < 1e-12 and abs(w(0.5) - 0.2) < 1e-12 and abs(w(1.0) - 0.2) < 1e-12 and abs(w(0.75) - 0.2) < 1e-12,
      "weight: start 2.0 at 0, 1.1 at until/2 (0.25), 0.2 = start x floor at until (0.5) and after (0.75, 1.0)")
side = {"t_p16": torch.softmax(torch.randn(2, 118, 16, 44), 1).half(), "t_c16": torch.randn(2, 80, 16, 44).half(),
        "t_p8": torch.softmax(torch.randn(2, 118, 32, 88), 1).half(), "t_c8": torch.randn(2, 80, 32, 88).half()}
same = dict(side, l16=side["t_p16"].float().log(), c16=side["t_c16"], l8=side["t_p8"].float().log(), c8=side["t_c8"])
check(abs(float(S.anchor_terms(same)["total"])) < 1e-4, f"anchor: student = teacher gives {float(S.anchor_terms(same)['total']):.1e} (fp16 teacher, within 1e-4)")
other = dict(same, c16=-side["t_c16"], l8=torch.zeros_like(side["t_p8"]).float())
parts = S.anchor_terms(other)
check(float(parts["total"]) > 1.0 and float(parts["cosl16"]) > 1.9 and float(parts["kl8"]) > 0.0 and abs(float(parts["cosl8"])) < 1e-3,
      f"anchor: opposite context on H16 (1 - cos = {float(parts['cosl16']):.2f}) and a uniform depth on H8 (KL {float(parts['kl8']):.2f}) give a positive total, H8 context untouched")

stub_model = build(camera_student="stub", student_trainable=True)
stub_model.train()
lossm = BEVLoss(grid=stub_model.grid)
batch = with_gt(data(3), stub_model.grid)
cfg0 = T.TrainConfig(device="cpu", gap_ego=False)
cfgA = dataclasses.replace(cfg0, student_anchor=2.0, student_anchor_floor=0.1, student_anchor_until=0.5)
o0, _, _ = T.forward_loss(stub_model, copy.deepcopy(lossm), batch, 0.0, "cpu", "off", cfg0)
oA, resA, _ = T.forward_loss(stub_model, copy.deepcopy(lossm), batch, 0.0, "cpu", "off", cfgA)
la = float(oA["student_anchor"].detach())
check("student_anchor" not in o0 and la > 0 and abs(float(oA["total"]) - float(o0["total"]) - 2.0 * la) < 1e-4 * max(1.0, float(oA["total"])) and torch.equal(oA["total_main"], o0["total_main"]),
      f"forward_loss, training, step 0: total = total without the anchor + 2.0 x anchor ({la:.3f}); the main total is untouched")
o0b, _, _ = T.forward_loss(stub_model, copy.deepcopy(lossm), batch, 0.5, "cpu", "off", cfg0)
oB, _, _ = T.forward_loss(stub_model, copy.deepcopy(lossm), batch, 0.5, "cpu", "off", cfgA)
check(abs(float(oB["total"]) - float(o0b["total"]) - 0.2 * float(oB["student_anchor"])) < 1e-4 * max(1.0, float(oB["total"])), "...at step_frac 0.5 the weight is 0.2 (start x floor)")
g, = torch.autograd.grad(oA["student_anchor"], stub_model.student.scale, retain_graph=True)
check(float(g.abs()) > 0, "the anchor's gradient reaches the student")
oV, _, _ = T.forward_loss(stub_model, copy.deepcopy(lossm), batch, 0.0, "cpu", "off", cfgA, train_terms=False)
check("student_anchor" in oV and not oV["student_anchor"].requires_grad and abs(float(oV["total"]) - float(o0["total"])) < 1e-6 and abs(float(oV["student_anchor"]) - la) < 1e-6,
      "validation: the anchor is logged (detached) and the total is the task loss only")
acc, ok, _, _ = T.run_clip(stub_model, copy.deepcopy(lossm), [batch], 0.0, "cpu", "off", cfgA)
check(ok and "student_anchor" in acc and "student_anchor" in T.LOGGED_TERMS, "run_clip logs student_anchor (LOGGED_TERMS)")
check("student_anchor" in T.RESUME_SWITCHES, "student_anchor is a resume switch")

# ------------------------------------------------------------------------------------------------ G
print("G: the camera-off class of bug")
from evaluation import bev_adapter as A  # noqa: E402
from evaluation import bev_modules as BM  # noqa: E402
m = build(camera_student="stub", student_trainable=True).eval()
b = data(2)
with torch.no_grad():
    ref = m.camera_bevs(b)
    zeroed = m.camera_bevs(BM.zero_cam(b))
    only_img = m.camera_bevs(dict(b, lane_img=b["lane_img"] * 0))
    swapped = m.camera_bevs(dict(b, lane_img=b["lane_img"].flip(0)))
check(not torch.equal(zeroed, ref) and not torch.equal(only_img, ref) and not torch.equal(swapped, ref),
      "zero_cam, zeroing only lane_img and swapping only lane_img each change the camera BEV: the evaluation's flags reach the student path")
frames = lambda i: [{k: b[k][i, f] for k in ("prob", "ctx", "prob2", "ctx2")} for f in range(2)]  # noqa: E731
item_a, item_b = {"frames": frames(0), "lane_img": b["lane_img"][0]}, {"frames": frames(1), "lane_img": b["lane_img"][1]}
A.perturb_camera(item_a, item_b, "shuffle_cam")
check(torch.equal(item_a["lane_img"], b["lane_img"][1]) and all(torch.equal(f["prob"], b["prob"][1, i]) for i, f in enumerate(item_a["frames"])), "the shuffle control swaps the image together with the cached tensors")

# ------------------------------------------------------------------------------------------------ H
print("H: the lane branch lifts with the student's depth")
ln = build(camera_student="stub", student_trainable=True, lane_branch="rgb", lane_pretrained="").eval()
with torch.no_grad():
    ln.lane.mix.weight.normal_(0, 0.1)                        # the branch starts at zero: give it a contribution to compare
    ln.lane.mix.bias.normal_(0, 0.1)


def lane_run(m, b, grad=False):
    """(the depth the branch's lift got per sample, the branch's contribution) of one forward"""
    seen, out = [], []
    h1 = m.lane.lift.register_forward_pre_hook(lambda mod, a: seen.append(a[1].detach().clone()))
    h2 = m.lane.register_forward_hook(lambda mod, a, o: out.append(o))
    try:
        with torch.set_grad_enabled(grad):
            m(b, dts=(0.0,))
    finally:
        h1.remove()
        h2.remove()
    return seen, out[0]


bh = data(2)
echo = echo_of(bh)
ln.student.echo = echo
seen0, c0 = lane_run(ln, bh)
check(len(seen0) == 2 and all(torch.equal(s, echo["prob16"][i:i + 1].float()) for i, s in enumerate(seen0)), "the branch's lift gets, per sample, the student's DualBEV depth of frame 0 (p16)")
torch.manual_seed(3)
junk = dict(bh, prob=torch.rand_like(bh["prob"]).softmax(2), prob2=torch.rand_like(bh["prob2"]).softmax(2))     # the cached depth of every frame replaced
_, c_junk = lane_run(ln, junk)
check(torch.equal(c0, c_junk) and float(c0.abs().max()) > 0, "its contribution does not change when only the cached depth changes (the student's replaces it)")
other = (4 * torch.randn_like(echo["prob16"])).softmax(1)                      # a sharply different depth distribution
ln.student.echo = dict(echo, prob16=other, logit16=other.log())
seen1, c1 = lane_run(ln, bh)
check(torch.equal(seen1[0], other[:1]) and float((c1 - c0).abs().max()) > 1e-3 * float(c0.abs().max()), "... and it changes when only the student's p16 changes (image, context, BEVFusion outputs as before)")
ln.student.echo = None
_, c2 = lane_run(ln, bh, grad=True)
gs, = torch.autograd.grad(c2.sum(), ln.student.scale)
check(float(gs.abs()) > 0, "a trainable student gets the gradient of the lane branch through its depth")

# ------------------------------------------------------------------------------------------------ I
print("I: grad_clip_split with the student")
student_ids = {id(p) for p in tr.student.parameters() if p.requires_grad}
split = clip_groups(tr, ["student"])
check(len(split) == 1 and {id(p) for p in split[0]} == student_ids, "the prefix 'student' names exactly the student's trainable tensors")
check(raises(ValueError, lambda: clip_groups(fro, ["student"]), "student"), "... and none of a frozen student: list it in grad_clip_split only when it trains")
params = [p for p in tr.parameters() if p.requires_grad]
others = [p for p in params if id(p) not in student_ids]
small = 1e-4


def set_grads():
    for p in params:
        p.grad = torch.full_like(p, 10.0 if id(p) in student_ids else small)


set_grads()
clip_gradients(params, 5.0, split)
st_norm = float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in tr.student.parameters() if p.requires_grad])))
check(all(torch.equal(p.grad, torch.full_like(p, small)) for p in others) and abs(st_norm - 5.0) < 0.05, f"with the split a huge student gradient is clipped on its own (norm {st_norm:.3f}) and leaves the other gradients as they were")
set_grads()
clip_gradients(params, 5.0)
check(all(float(p.grad.max()) < 0.01 * small for p in others), "control: the global clip scales every other gradient down by the student's size (what 'student' in the split prevents)")

print(f"\ntest_student_chain: {'ALL PASSED' if not fails else str(len(fails)) + ' FAILED'} ({time.time() - t0:.0f} s)")
sys.exit(1 if fails else 0)
