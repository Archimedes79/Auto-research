"""t26 track A (card experiments/t26_one_long_run, section 2, code 1 - 3): the camera student on EVERY frame of the recursive window (`ModelConfig.student_frames` 'all'), the image cache of the
older frames, and the photometric image jitter (`TrainConfig.img_jitter` / `img_jitter_from`). CPU, 2 threads, no GPU; the builder part reads nuScenes mini (`~/data/nuscenes`) and removes its
temporary folder. The stage B student on frame 0 is tests/test_student_chain.py; this file adds what t26 changed.

  A  OFF is the code before: student_frames 'current', no student, a frozen student, the real ResNet-18 student in training mode (recomputed backward), F = 1 and 3: the camera BEV, every
     gradient and every state tensor equal a verbatim copy of the camera_bevs / _run_student of commit 94a7539a bit for bit; without batch["img_jitter"] nothing changes
  B  'all': every frame of the window is the student's output of ITS OWN image (oracle made one frame at a time, F = 2, 3), also with a frozen student against a cached model whose cached outputs are
     the student's; frame 0 as with 'current'; 'current' ignores lane_img_old; garbage in the cached older frames changes nothing; a missing / miscounted lane_img_old is an error that names the
     builder; a one-frame (stream) batch needs only lane_img; the anchor side stays on frame 0
  C  gradient: from the newest frame only (evaluation mode: exactly none from the older frames, and unchanged when an older image is replaced; training mode with the recomputed lift, the real
     student: output, every gradient and every state tensor equal the oracle whose older outputs are constants)
  D  stream = batch with the student on all frames (one-frame steps against a 3-frame window, and across key frames); the gradient-free older forward is one batched call
  E  image jitter: known answers against an independent numpy reference, one draw per sample for all its frames, deterministic in (seed, step, micro-batch), a function of the step and not of a random
     stream, the key frames of a clip share the draw, off below img_jitter_from, none in evaluation mode, the anchor target stays the clean image's, the older frames' noise is its own stream (in the
     model), and the real train() (toy network) hands run_clip the step and the micro-batch
  F  window images on nuScenes mini: each cached older image is the JPEG of its frame's sample_data (independent walk along `prev`), the resumable builder and its finished-build markers (key frames and
     older frames: scenes and the builder's constants, merged by a second build under a lock, none after a crash half-way), the dataset item and collate, a config with a "split", the command line end
     to end (also on a student 'all' config with feature caches named: the t26 one); startable asks for both markers
  G  wiring: check_config, student_findings, make_datasets, interface contract, the evaluation's camera controls take lane_img_old away too
  H  planted bugs: each must break at least one check above
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_student_window.py"""
import atexit
import contextlib
import copy
import dataclasses
import json
import os
import shutil
import sys
import tempfile
import time
import fcntl  # noqa: F401 - the lock of the marker (section H plants a bug in it)
import types

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "training"), *[os.path.join(ROOT, "net", d) for d in ("prep", "model", "loss")], os.path.join(ROOT, "tests"), os.path.join(ROOT, "tests", "support"), ROOT, os.path.join(ROOT, "..")]
import augment as AU  # noqa: E402
import feature_cache as FC  # noqa: E402
import interface as IF  # noqa: E402
import image_cache as IC  # noqa: E402
import build_image_cache as BIC  # noqa: E402
import model as M  # noqa: E402
import startable as ST  # noqa: E402
import student_encoder as S  # noqa: E402
import train as T  # noqa: E402
from feature_cache import CachedBEVFusionModel  # noqa: E402
import batches as BT  # noqa: E402
import cached_model as CM  # noqa: E402
from model import CAMERAS, ModelConfig, check_config  # noqa: E402
from test_slots import BASE, batch_for  # noqa: E402

torch.set_num_threads(2)
fails = []
t0 = time.time()


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


class _Capture:
    """a stdout that keeps what was printed"""
    def __init__(self):
        self.text = ""

    def write(self, s):
        self.text += s

    def flush(self):
        pass


tmp = tempfile.mkdtemp(prefix="student_window_", dir=os.path.join(ROOT, "temp") if os.path.isdir(os.path.join(ROOT, "temp")) else None)
atexit.register(shutil.rmtree, tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ helpers
class Stub(nn.Module):
    """stands in for the student: every output is a fixed function of the IMAGE (its mean, per sample), so an output says which image it was made from; `seen` (a list) records its inputs"""
    HEAD_OF = S.StudentEncoder.HEAD_OF
    seen = None

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.scale = nn.Parameter(torch.ones(()))
        self.base = {"logit16": torch.randn(118, 16, 44, generator=g), "ctx16": torch.randn(80, 16, 44, generator=g),
                     "logit8": torch.randn(118, 32, 88, generator=g), "ctx8": torch.randn(80, 32, 88, generator=g)}

    def keep_bn_eval(self, flag=True):
        return self

    def forward(self, img):
        if self.seen is not None:
            self.seen.append(img.detach().clone())
        g = (img.float().mean((1, 2, 3)) / 255).view(-1, 1, 1, 1)
        out = {}
        for k in ("16", "8"):
            out["logit" + k] = (self.base["logit" + k][None] + 6 * g) * self.scale
            out["ctx" + k] = (self.base["ctx" + k][None] + 3 * g) * self.scale
            out["prob" + k] = out["logit" + k].softmax(1)
        return out


M.STUDENTS["stub"] = Stub
DUAL = dict(BASE, camera_features="dual")
STUB_CK = os.path.join(tmp, "stub.pt")
torch.save({"state": {"model." + k: v for k, v in Stub().state_dict().items()}, "ema": {k: v for k, v in Stub().state_dict().items() if v.dtype.is_floating_point}, "step": 7, "cfg": {}}, STUB_CK)
MUTATE = [None]                                                  # section H: a function applied to every model build() returns


def images(B=2, seed=1):
    """uint8 key-frame images with a different brightness per sample (the stub reads the mean) and texture (the real student needs it)"""
    g = torch.Generator().manual_seed(seed)
    base = torch.linspace(50, 190, B).view(B, 1, 1, 1)
    return (base + torch.randint(0, 30, (B, 3, 256, 704), generator=g).float()).clamp(0, 255).to(torch.uint8)


def older_images(B, K):
    """uint8 images of the older frames, (B, K, 3, H, W): another brightness than the current ones and than each other"""
    return torch.stack([(images(B, seed=20 + k).float() * 0.5 + 25 * (k + 1)).clamp(0, 255).to(torch.uint8) for k in range(K)], 1)


def build(cls=CachedBEVFusionModel, seed=0, **kw):
    torch.manual_seed(seed)
    m = cls(ModelConfig(**dict(DUAL, **kw)))
    if MUTATE[0] is not None:
        MUTATE[0](m)
    return m


def data(F, B=2, old=True, **kw):
    b = batch_for(ModelConfig(**dict(DUAL, **kw)), B=B, Fn=F)
    b["lane_img"] = images(B)
    if old and F > 1:
        b["lane_img_old"] = older_images(B, F - 1)
    return b


def own_outputs(m, b):
    """the student's outputs of every frame of the batch, made one frame at a time (the oracle's inputs)"""
    with torch.no_grad():
        return [m.student(b["lane_img"])] + [m.student(b["lane_img_old"][:, k]) for k in range(b["prob"].shape[1] - 1)]


def all_replaced(b, outs):
    """the batch with EVERY frame of both sources replaced by the student's outputs of that frame (outs: one output dict per frame)"""
    r = dict(b)
    for suffix, k in (("", "16"), ("2", "8")):
        r["prob" + suffix] = torch.stack([o["prob" + k].float() for o in outs], 1)
        r["ctx" + suffix] = torch.stack([o["ctx" + k].float() for o in outs], 1)
    return r


# verbatim copies of the camera_bevs / _run_student of the commit before t26 (94a7539a): the oracle of section A
def head_run_student(self, batch):
    if "lane_img" not in batch:
        raise KeyError("camera_student reads the current key-frame image batch['lane_img']: set TrainConfig.lane_images (the folder of the key-frame JPEGs, data/cache/img_front)")
    with torch.set_grad_enabled(torch.is_grad_enabled() and self.cfg.student_trainable):
        return self.student(batch["lane_img"])


def head_camera_bevs(self, batch):
    prob = batch["prob"].float()
    B, Fn = prob.shape[:2]
    if self.cfg.camera_off:
        return torch.zeros(B, Fn, self.cfg.c_cam, self.grid.nx, self.grid.ny, device=prob.device)
    stud, side = (self._run_student(batch) if self.student is not None else None), {}
    recompute_lift = stud is not None and self.cfg.student_trainable and self.training and torch.is_grad_enabled()
    bevs = []
    for i, src in enumerate(CAMERAS[self.cfg.camera_features]):
        s = src.suffix
        p, ctx = (prob if i == 0 else batch["prob" + s].float()), batch["ctx" + s].float()
        if stud is not None:
            h = self.student.HEAD_OF[src.features]
            side.update({"p" + h: stud["prob" + h], "c" + h: stud["ctx" + h], "l" + h: stud["logit" + h], "t_p" + h: p[:, 0].detach(), "t_c" + h: ctx[:, 0].detach()})
            p = torch.cat([stud["prob" + h].float()[:, None], p[:, 1:]], 1)
            ctx = torch.cat([stud["ctx" + h].float()[:, None], ctx[:, 1:]], 1)
        if i == 0 and self.cam_adapter is not None:
            ctx = self.cam_adapter(ctx.flatten(0, 1)).view_as(ctx)
        lift, collapse = getattr(self, "lift" + s), getattr(self, "cam_reduce" + s)
        keys = self._lift_keys(batch, batch["hw"], "ida" + s)
        vols = []
        for b in range(B):
            args = (ctx[b], p[b], batch["K"][b, 0], batch["T_cam_car"][b, 0], batch["hw"], batch["ida" + s])
            key = None if keys is None else keys[b]
            if recompute_lift:
                if key is None:
                    key = M.lift_key(*args[2:])
                vols.append(collapse(torch.utils.checkpoint.checkpoint(lift, *args, key=key, use_reentrant=False)))
            else:
                vols.append(collapse(lift(*args, key=key)))
        bevs.append(torch.stack(vols))
    self._student_side = side or None
    if len(bevs) == 1:
        return bevs[0]
    return self.cam_merge(torch.cat(bevs, 2).flatten(0, 1)).view_as(bevs[0])


def as_head(m):
    m._run_student = types.MethodType(head_run_student, m)
    m.camera_bevs = types.MethodType(head_camera_bevs, m)
    return m


def snap(m, b, train, w_seed=11):
    """(camera BEV, {param: grad}, state tensors) of one forward + backward of the camera BEV"""
    m.train(train)
    m.zero_grad(set_to_none=True)
    out = m.camera_bevs(b)
    if out.requires_grad:
        (out * torch.randn(out.shape, generator=torch.Generator().manual_seed(w_seed))).sum().backward()
    return out.detach().clone(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}, {k: v.clone() for k, v in m.state_dict().items()}


def same_snap(a, b):
    return torch.equal(a[0], b[0]) and set(a[1]) == set(b[1]) and all(torch.equal(a[1][n], b[1][n]) for n in b[1]) and all(torch.equal(a[2][k], v) for k, v in b[2].items())


# ------------------------------------------------------------------------------------------------ A
print("A: OFF is the code before")
cases = [("dualbev, no student", dict(camera_features="dualbev")), ("bevfusion, no student", dict(camera_features="bevfusion")), ("dual, no student", {}),
         ("dualbev + cam_adapter", dict(camera_features="dualbev", cam_adapter=True)), ("camera_off", dict(camera_off=True)),
         ("stub student trainable, 'current'", dict(camera_student="stub", student_trainable=True)), ("stub student frozen, 'current'", dict(camera_student="stub", student_init=STUB_CK)),
         ("ResNet-18 student trainable (the recomputed backward), 'current'", dict(camera_student="resnet18", student_trainable=True))]
for name, kw in cases:
    for F in (1, 3):
        b = data(F, old=False, **kw)
        res = {}
        for train in (False, True):
            new, old = build(**kw), as_head(build(**kw))
            res[train] = same_snap(snap(new, b, train), snap(old, b, train))
        check(all(res.values()), f"{name}, F = {F}: camera BEV, every gradient and every state tensor equal the old code's, bit for bit (evaluation and training mode)")
m_new, m_old = build(camera_student="stub", student_trainable=True, lane_branch="rgb", lane_pretrained="").eval(), as_head(build(camera_student="stub", student_trainable=True, lane_branch="rgb", lane_pretrained="")).eval()
b = data(2, old=False)
with torch.no_grad():
    o_new, o_old = m_new(dict(b), dts=(0.0, 0.5)), m_old(dict(b), dts=(0.0, 0.5))
check(all(torch.equal(o_new["out"][dt]["sem"], o_old["out"][dt]["sem"]) and torch.equal(o_new["out"][dt]["obj"]["hm"], o_old["out"][dt]["obj"]["hm"]) for dt in (0.0, 0.5))
      and all(torch.equal(o_new["student_side"][k], o_old["student_side"][k]) for k in o_new["student_side"]), "the whole forward with the student and the lane branch at dt 0 / +0.5, and student_side, equal the old code's")

# ------------------------------------------------------------------------------------------------ B, C
print("B: 'all' - every frame from its own image; C: the gradient comes from the newest frame only")


def model_checks():
    """B and the evaluation-mode half of C on the stub student (fast, a few seconds): name -> bool. Section H runs it under every planted bug"""
    ok = {}
    for F in (2, 3):
        off, on, cur = build(), build(camera_student="stub", student_trainable=True, student_frames="all"), build(camera_student="stub", student_trainable=True)
        fro = build(camera_student="stub", student_init=STUB_CK, student_frames="all")
        b = data(F)
        for label, m in (("trainable", on), ("frozen", fro)):
            want = off.eval().camera_bevs(all_replaced(b, own_outputs(m, b)))
            with torch.no_grad():
                got = m.eval().camera_bevs(b)
            ok[f"F = {F}, {label} student: the BEV equals the student-free model's on the batch whose every frame is the student's output of that frame's own image (oracle one frame at a time)"] = torch.equal(got, want)
        with torch.no_grad():
            got, got_cur, plain = on.camera_bevs(b), cur.eval().camera_bevs(b), off.camera_bevs(b)
            no_old = cur.camera_bevs({k: v for k, v in b.items() if k != "lane_img_old"})
            dim = on.camera_bevs(dict(b, lane_img_old=(b["lane_img_old"].float() * 0.5).to(torch.uint8)))
            junk = on.camera_bevs(dict(b, prob=torch.cat([b["prob"][:, :1], torch.rand_like(b["prob"][:, 1:])], 1), ctx2=torch.cat([b["ctx2"][:, :1], torch.randn_like(b["ctx2"][:, 1:])], 1)))
        ok[f"F = {F}: frame 0 is the same as with 'current'; the older frames differ from the cached teachers' and from 'current'"] = \
            torch.equal(got[:, 0], got_cur[:, 0]) and not torch.allclose(got[:, 1:], plain[:, 1:]) and not torch.allclose(got[:, 1:], got_cur[:, 1:])
        ok[f"F = {F}: 'current' ignores lane_img_old"] = torch.equal(got_cur, no_old)
        ok[f"F = {F}: other older images change the older frames only"] = torch.equal(dim[:, 0], got[:, 0]) and not torch.equal(dim[:, 1:], got[:, 1:])
        ok[f"F = {F}: garbage in the cached older frames changes nothing (they are not read)"] = torch.equal(junk, got)
        ok[f"F = {F}: no lane_img_old: KeyError that names it and the builder"] = raises(KeyError, lambda: on.camera_bevs({k: v for k, v in b.items() if k != "lane_img_old"}), "lane_img_old", "build-cache-older")
        ok[f"F = {F}: one image too many: ValueError"] = raises(ValueError, lambda: on.camera_bevs(dict(b, lane_img_old=torch.cat([b["lane_img_old"], b["lane_img_old"][:, :1]], 1))), "older frame")
    on, cur = build(camera_student="stub", student_trainable=True, student_frames="all"), build(camera_student="stub", student_trainable=True)
    b1 = data(1)
    with torch.no_grad():
        ok["'all' with a one-frame batch and no lane_img_old = 'current' (the stream step needs no older image)"] = torch.equal(on.eval().camera_bevs(b1), cur.eval().camera_bevs(b1))
    b3 = data(3)
    side = on(b3, dts=(0.0,))["student_side"]
    want = on.student(b3["lane_img"])
    ok["student_side (the anchor) is frame 0's: the student's outputs of lane_img against the cached teacher's frame 0"] = \
        all(torch.equal(side[k], want[n + k[1:]]) for k, n in (("p16", "prob"), ("c16", "ctx"), ("p8", "prob"), ("c8", "ctx"))) and torch.equal(side["t_p16"], b3["prob"][:, 0]) and torch.equal(side["t_c8"], b3["ctx2"][:, 0])
    # C, evaluation mode (BatchNorm fixed): exactly the newest frame
    sc = on.student.scale
    w = torch.randn(on.camera_bevs(b3).shape, generator=torch.Generator().manual_seed(5))
    g_cur = torch.autograd.grad(on.camera_bevs(b3)[:, 0].sum(), sc, allow_unused=True)[0]
    g_old = torch.autograd.grad(on.camera_bevs(b3)[:, 1:].sum(), sc, allow_unused=True)[0]
    ok["C evaluation mode: the newest frame's BEV reaches the student, the older frames' BEV does not (no graph)"] = g_cur is not None and float(g_cur.abs()) > 0 and (g_old is None or float(g_old.abs()) == 0.0)
    g_a = torch.autograd.grad((on.camera_bevs(b3) * w).sum(), sc)[0]
    g_b = torch.autograd.grad((on.camera_bevs(dict(b3, lane_img_old=older_images(2, 2).flip(1))) * w).sum(), sc)[0]
    ok["C evaluation mode: the student's gradient is unchanged when the older frames' images are replaced"] = torch.equal(g_a, g_b) and float(g_a.abs()) > 0
    old_out = on._run_student_older(b3["lane_img_old"])
    ok["C the older frames' student outputs carry no graph (gradients on), are (B, F - 1, ...) and only prob / ctx of the two heads"] = \
        all(not v.requires_grad for v in old_out.values()) and all(v.shape[:2] == (2, 2) for v in old_out.values()) and sorted(old_out) == ["ctx16", "ctx8", "prob16", "prob8"]
    cur_out = on._run_student(on._student_images(b3, 3)[0])
    ok["C the current frame's outputs carry the graph of a trainable student"] = all(v.requires_grad for k, v in cur_out.items() if k.startswith(("prob", "ctx", "logit")))
    return ok


for name, v in model_checks().items():
    check(v, name)
# the real encoder: the batch of older frames equals the frames made one at a time (flatten / view with real convolutions, BatchNorm in eval mode)
off_r, on_r = build(), build(camera_student="resnet18", student_trainable=True, student_frames="all")
b3 = data(3)
with torch.no_grad():
    want_r = off_r.eval().camera_bevs(all_replaced(b3, own_outputs(on_r, b3)))
    got_r = on_r.eval().camera_bevs(b3)
    swapped = on_r.camera_bevs(dict(b3, lane_img_old=b3["lane_img_old"].flip(0)))
check(torch.allclose(got_r, want_r, atol=1e-5, rtol=1e-4) and not torch.allclose(swapped[:, 1:], got_r[:, 1:]), f"real ResNet-18 student: the batch of older frames equals the frames made one at a time (max diff {float((got_r - want_r).abs().max()):.1e}); swapping the samples' images changes the older frames")
# C, training mode, the real student (checkpoint_encode, the lift recomputed in the backward): the older frames are constants - output, gradients and state equal the oracle's
for K in (1, 2):
    tr = build(camera_student="resnet18", student_trainable=True, student_frames="all")
    st = tr.student
    bt = data(K + 1)
    state_all = copy.deepcopy(tr.state_dict())
    w_all = torch.randn_like(tr.eval().camera_bevs(bt))
    runs = {}
    for oracle in (False, True):
        tr.load_state_dict(state_all)
        tr.train()
        tr.zero_grad(set_to_none=True)
        st.checkpoint_encode = not oracle
        out_ = as_head(tr).camera_bevs(all_replaced(bt, own_outputs(tr, bt))) if oracle else tr.camera_bevs(bt)       # the oracle: the code before, cached outputs = the student's, older frames constants
        (out_ * w_all).sum().backward()
        runs[oracle] = (out_.detach().clone(), {n: p.grad.clone() for n, p in tr.named_parameters() if p.grad is not None}, {k: v.clone() for k, v in tr.state_dict().items()})
    live, ref = runs[False], runs[True]
    if K == 1:
        ok_ = same_snap(live, ref)
    else:                                                                                 # one batch of 2 B frames against frames one at a time: real convolutions differ in the last bits
        ok_ = torch.allclose(live[0], ref[0], atol=1e-6, rtol=1e-5) and set(live[1]) == set(ref[1]) and all(torch.allclose(live[1][n], ref[1][n], atol=1e-5, rtol=1e-4) for n in ref[1]) \
            and all(torch.allclose(live[2][k].float(), ref[2][k].float(), atol=1e-6, rtol=1e-5) for k in ref[2])
    check(ok_, f"C training mode, F = {K + 1}, real student: output, all {len(ref[1])} gradients and all {len(ref[2])} state tensors equal the oracle whose older frames are constants ({'bit for bit' if K == 1 else 'to 1e-5'})")

# the documented caveat: in TRAINING mode the BatchNorm batch statistics behind the lift couple the frames' values (not a path through the student's older forward)
tr = build(camera_student="stub", student_trainable=True, student_frames="all").train()
bt = data(3)
gs = []
for old_img in (bt["lane_img_old"], older_images(2, 2).flip(1)):
    tr.zero_grad(set_to_none=True)
    (tr.camera_bevs(dict(bt, lane_img_old=old_img)) * torch.randn(2, 3, tr.cfg.c_cam, tr.grid.nx, tr.grid.ny, generator=torch.Generator().manual_seed(5))).sum().backward()
    gs.append(tr.student.scale.grad.clone())
print(f"  info training mode: the student's gradient changes by {float((gs[0] - gs[1]).abs() / gs[0].abs()):.1e} (relative) when the older frames' images are replaced - the BatchNorm statistics of the lift, not a gradient through the older frames")

# ------------------------------------------------------------------------------------------------ D
print("D: stream = batch with the student on all frames")
B, NF, TOL = 2, 6, 1e-5
CFG_D = ModelConfig(**dict(DUAL, cam_buffer="recurrent"))
BASE_D = batch_for(CFG_D, B=B, Fn=NF)
IMG_D = torch.stack([images(B, seed=40 + f) for f in range(NF)], 1)                      # (B, NF, 3, H, W): one image per frame
SEGS = [[(torch.full((B,), 1 / 12), torch.tensor([8.0 + 0.5 * i + 0.3 * j, 12.0 - 0.4 * i - 0.2 * j]), torch.tensor([0.30 - 0.05 * i + 0.02 * j, -0.20 + 0.04 * i - 0.03 * j])) for j in range(2)]
        for i in range(NF - 1)]                                                           # frame i -> i + 1, two sweeps
VG, WG = torch.tensor([9.0, 11.0]), torch.tensor([0.25, -0.15])


def mk(idx):
    """the batch of the frames `idx` (chronological), stored newest first as the collate does, with the images of those frames: lane_img = the newest, lane_img_old = the others"""
    order = idx[::-1]
    b = dict(BASE_D)
    for k in ("K", "T_cam_car", "prob", "ctx", "prob2", "ctx2"):
        b[k] = BASE_D[k][:, order]
    b["steps"] = [sum((SEGS[i] for i in range(o, idx[-1])), []) for o in order]
    b["lane_img"] = IMG_D[:, order[0]]
    if len(order) > 1:
        b["lane_img_old"] = IMG_D[:, order[1:]]
    return b


def build_rec(**kw):
    m = build(cam_buffer="recurrent", **kw).eval()
    with torch.no_grad():                                                                # zero-start layers get values: every path carries a signal
        for q in m.parameters():
            if q.requires_grad and q.abs().max() == 0:
                q.normal_(0, 0.05)
    return m


def run(m, batch, state=None, step=None):
    with torch.no_grad():
        return m(batch, dts=(0.0,), state=state, step=step)


def flat(res):
    out = {"features": res["features"]}
    for dt, p in res["out"].items():
        for k, v in p.items():
            for kk, t in (v.items() if isinstance(v, dict) else [("", v)]):
                if torch.is_tensor(t):
                    out[f"{dt}.{k}.{kk}"] = t
    for k, v in res["state"].items():
        for kk, t in (v or {}).items():
            out[f"state.{k}.{kk}"] = t
    return out


def gap(a, b):
    fa, fb = flat(a), flat(b)
    assert fa.keys() == fb.keys(), set(fa) ^ set(fb)
    return max(float((fa[k] - fb[k]).abs().max()) for k in fa)


def stream(m, idx):
    """one call per frame of `idx` (consecutive, one-frame batches with the image of that frame only), the state carried"""
    res, st = [], None
    for n, i in enumerate(idx):
        res.append(run(m, mk([i]), st, SEGS[i - 1] if n else None))
        st = res[-1]["state"]
    return res


def stream_batch_checks(kind="stub", tol=TOL):
    m = build_rec(camera_student=kind, student_trainable=True, student_frames="all")
    ok = {f"D1 stream of three frames = the 3-frame batch ({kind}, outputs and states)": gap(stream(m, [0, 1, 2])[2], run(m, mk([0, 1, 2]))) < tol}
    s = stream(m, [0, 1, 2]) + [None] * 3
    dt_o = torch.full((B,), 0.5) - sum((x[0] for x in SEGS[3] + SEGS[4]), torch.zeros(B))
    s[3] = run(m, mk([3]), s[2]["state"], [(dt_o, VG, WG)])
    s[4] = run(m, mk([4]), s[3]["state"], SEGS[3])
    s[5] = run(m, mk([5]), s[4]["state"], SEGS[4])
    k1 = run(m, mk([0, 1, 2]))
    k2 = run(m, mk([3, 4, 5]), k1["state"], (torch.full((B,), 0.5), VG, WG))
    ok[f"D2 across key frames: the batch with the previous state and the gap step = the stream of six frames ({kind})"] = gap(s[5], k2) < tol
    cold = run(m, mk([0, 1, 2]), None)
    ok[f"D3 the batch's older images matter ({kind})"] = gap(cold, run(m, dict(mk([0, 1, 2]), lane_img_old=mk([0, 1, 2])["lane_img_old"].flip(0)))) > 1e-4
    return ok


for kind, tol in (("stub", TOL), ("resnet18", 1e-4)):
    for name, v in stream_batch_checks(kind, tol).items():
        check(v, name)

# ------------------------------------------------------------------------------------------------ E
print("E: image jitter")


def ref_jitter(img, u, s, noise=0.0):
    """an independent numpy reference of augment.jitter_images for ONE frame (3, H, W) uint8, float64, no random noise"""
    x = img.astype(np.float64) / 255
    luma = lambda a: 0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2]  # noqa: E731
    x = np.clip(x * (1 + 0.4 * s * u[0]), 0, 1)
    x = np.clip((x - luma(x).mean()) * (1 + 0.4 * s * u[1]) + luma(x).mean(), 0, 1)
    y = luma(x)
    x = np.clip((x - y) * (1 + 0.4 * s * u[2]) + y, 0, 1)
    x = x ** (1.25 ** (s * u[3]))
    return np.clip(x, 0, 1) * 255


@contextlib.contextmanager
def no_noise():
    old = AU.JITTER_NOISE
    AU.JITTER_NOISE = 0.0
    try:
        yield
    finally:
        AU.JITTER_NOISE = old


def jitter_fn_checks():
    ok = {}
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (2, 3, 12, 20), dtype=np.uint8)
    u = np.array([[0.7, -0.4], [-0.9, 0.3], [0.5, 1.0], [-0.6, -1.0]])
    for s in (1.0, 0.5):
        j = {"u": torch.tensor(u), "strength": s, "noise_seed": 1}
        with no_noise():
            got = AU.jitter_images(torch.from_numpy(img), j).numpy()
        want = np.stack([ref_jitter(img[b], u[:, b], s) for b in range(2)])
        ok[f"known answer: brightness, contrast, saturation, gamma of 2 samples = the independent numpy reference, strength {s} (max diff {np.abs(got - want).max():.1e})"] = np.abs(got - want).max() < 0.02
    one = lambda a, **kw: AU.jitter_images(torch.from_numpy(np.asarray(a, np.uint8)), {"u": torch.tensor(kw["u"], dtype=torch.float32).view(4, 1), "strength": 1.0, "noise_seed": 0}).numpy()  # noqa: E731
    with no_noise():
        up = one(np.full((1, 3, 4, 4), 100), u=[1, 0, 0, 0])
        ok["by hand: brightness +1 at strength 1 = x 1.4 (100 -> 140)"] = np.allclose(up, 140, atol=0.01)
        c = np.zeros((1, 3, 2, 4), np.uint8)
        c[..., :2], c[..., 2:] = 50, 150
        ok["by hand: contrast +1 about the mean (50 / 150, mean 100 -> 30 / 170)"] = np.allclose(one(c, u=[0, 1, 0, 0])[0, 0, 0], [30, 30, 170, 170], atol=0.01)
        red = np.zeros((1, 3, 2, 2), np.uint8)
        red[:, 0] = 255
        sat = one(red, u=[0, 0, -1, 0])[0, :, 0, 0]
        ok["by hand: saturation -1 (x 0.6) of pure red: (255, 0, 0) -> (183.5, 30.5, 30.5)"] = np.allclose(sat, [183.5, 30.5, 30.5], atol=0.1)
        mid = np.full((1, 3, 2, 2), 128)
        ok["by hand: gamma +1 / -1 of 128 = 255 (0.502) ** 1.25 / 0.8 (107.7 / 146.4)"] = np.allclose(one(mid, u=[0, 0, 0, 1])[0, 0, 0, 0], 255 * (128 / 255) ** 1.25, atol=0.01) \
            and np.allclose(one(mid, u=[0, 0, 0, -1])[0, 0, 0, 0], 255 * (128 / 255) ** 0.8, atol=0.01)
    flat_img = torch.full((1, 3, 96, 96), 128, dtype=torch.uint8)
    j0 = {"u": torch.zeros(4, 1), "strength": 1.0, "noise_seed": 5}
    noise = AU.jitter_images(flat_img, j0) - 128.0
    ok[f"noise: sigma 0.02 x strength of the 0..1 range (measured {float(noise.std()) / 255:.4f}), no bias"] = abs(float(noise.std()) / 255 - 0.02) < 0.001 and abs(float(noise.mean())) < 0.3
    a, b2 = AU.jitter_images(flat_img, j0, 0), AU.jitter_images(flat_img, j0, 1)
    torch.manual_seed(0)
    torch.rand(7)
    ok["determinism: the same (draw, salt) gives the same pixels whatever the global random state did; another salt / another noise seed another noise"] = \
        torch.equal(a, AU.jitter_images(flat_img, j0, 0)) and not torch.equal(a, b2) and not torch.equal(a, AU.jitter_images(flat_img, dict(j0, noise_seed=6), 0))
    five = torch.randint(0, 256, (2, 1, 3, 8, 8), dtype=torch.uint8).expand(2, 3, 3, 8, 8).contiguous()      # three identical frames per sample, two different samples
    five[1] = (five[1].float() * 0.5).to(torch.uint8) + 20
    jd = AU.draw_jitter(2, 1.0, 77)
    with no_noise():
        out5 = AU.jitter_images(five, jd, 1)
    ok["one draw per sample for ALL its frames (identical frames stay identical), another sample another draw"] = all(torch.equal(out5[b][0], out5[b][k]) for b in range(2) for k in (1, 2)) \
        and not torch.equal(AU.draw_jitter(2, 1.0, 77)["u"][:, 0], AU.draw_jitter(2, 1.0, 77)["u"][:, 1])
    with_noise = AU.jitter_images(five, jd, 1)
    ok["...and the pixel noise is drawn per frame (the three identical frames differ by it alone)"] = not torch.equal(with_noise[0][0], with_noise[0][1]) and float((with_noise[0][0] - with_noise[0][1]).abs().max()) < 60
    d0, d1, dm, ds = AU.draw_jitter(3, 1.0, 5, 0), AU.draw_jitter(3, 1.0, 5, 1), AU.draw_jitter(3, 1.0, 6, 0), AU.draw_jitter(3, 0.5, 5, 0)
    ok["draw_jitter: the key frames of a clip share u and differ in the noise seed; another seed another u; the strength is carried; u in [-1, 1]"] = \
        torch.equal(d0["u"], d1["u"]) and d0["noise_seed"] != d1["noise_seed"] and not torch.equal(d0["u"], dm["u"]) and ds["strength"] == 0.5 and float(d0["u"].abs().max()) <= 1
    ok["jitter_seed: a function of (seed, step, micro-batch), distinct for neighbours"] = len({AU.jitter_seed(s_, st_, mi) for s_ in (0, 1) for st_ in (0, 1, 2, 35999) for mi in (0, 1, 2)}) == 24 \
        and AU.jitter_seed(1, 2, 3) == AU.jitter_seed(1, 2, 3)
    ok["the image is not changed in place and the result is float32 0..255 of the input's shape"] = torch.equal(flat_img, torch.full((1, 3, 96, 96), 128, dtype=torch.uint8)) \
        and a.dtype == torch.float32 and a.shape == flat_img.shape and float(a.min()) >= 0 and float(a.max()) <= 255
    return ok


for name, v in jitter_fn_checks().items():
    check(v, name)


def train_bn_eval(m):
    """the model in training mode with every BatchNorm in eval mode (the student's already are): a frame's BEV is then independent of the other frames"""
    m.train()
    for mod in m.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.eval()


def jitter_model_checks():
    ok = {}
    for frames in ("all", "current"):
        m = build(camera_student="stub", student_trainable=True, student_frames=frames)
        b = data(3, old=frames == "all")
        jb = dict(b, img_jitter=AU.draw_jitter(2, 1.0, 123))
        m.student.seen = []
        with torch.no_grad():
            m.eval()
            ev, ev_clean = m.camera_bevs(jb), m.camera_bevs(b)
            seen_eval = [t.dtype for t in m.student.seen[:1]]
            train_bn_eval(m)                                                                # training mode (the jitter is on), BatchNorm of the lift fixed: the frames do not couple through their statistics
            tr, tr2 = m.camera_bevs(jb), m.camera_bevs(jb)
            tr_clean = m.camera_bevs(b)
            m.student.seen = None
        ok[f"model ({frames}): in evaluation mode batch['img_jitter'] is ignored (output equal, the student gets the uint8 images)"] = torch.equal(ev, ev_clean) and seen_eval == [torch.uint8]
        ok[f"model ({frames}): in training mode the jitter changes frame 0 {'and the older frames' if frames == 'all' else 'only (the cached older frames are the teachers\'s)'}"] = \
            not torch.allclose(tr[:, 0], tr_clean[:, 0]) and (not torch.allclose(tr[:, 1:], tr_clean[:, 1:]) if frames == "all" else torch.equal(tr[:, 1:], tr_clean[:, 1:]))
        ok[f"model ({frames}): the same draw twice gives the same BEV"] = torch.equal(tr, tr2)
    m = build(camera_student="stub", student_trainable=True, student_frames="all").train()
    same_img = images(2, seed=9)                                                    # identical images in all three frames of a sample
    b = dict(data(3), lane_img=same_img, lane_img_old=same_img[:, None].expand(2, 2, 3, 256, 704).contiguous())
    m.student.seen = []
    with no_noise(), torch.no_grad():
        m.camera_bevs(dict(b, img_jitter=AU.draw_jitter(2, 1.0, 5)))
    cur_in, old_in = m.student.seen[0], m.student.seen[1].view(2, 2, 3, 256, 704)
    ok["model: the student's input is the jittered float image, the same factors for frame 0 and the older frames of a sample (noise off), different for the two samples"] = \
        cur_in.dtype == torch.float32 and not torch.equal(cur_in.float(), same_img.float()) and torch.equal(old_in[:, 0], cur_in) and torch.equal(old_in[:, 1], cur_in) \
        and not torch.equal(cur_in[0] / cur_in[0].mean(), cur_in[1] / cur_in[1].mean()) and not torch.equal(cur_in, AU.jitter_images(same_img, AU.draw_jitter(2, 1.0, 6)))
    m.student.seen = None
    on = build(camera_student="stub", student_trainable=True, student_frames="all").train()
    b = data(3)
    res_c = on(b, dts=(0.0,))["student_side"]
    res_j = on(dict(b, img_jitter=AU.draw_jitter(2, 1.0, 9)), dts=(0.0,))["student_side"]
    ok["the anchor keeps comparing with the clean image's teacher outputs: t_p / t_c equal the cached frame 0 under jitter, the student's side differs"] = \
        torch.equal(res_j["t_p16"], b["prob"][:, 0]) and torch.equal(res_j["t_c8"], b["ctx2"][:, 0]) and not torch.allclose(res_j["c16"], res_c["c16"])
    g = torch.autograd.grad(on.camera_bevs(dict(b, img_jitter=AU.draw_jitter(2, 1.0, 9)))[:, 0].sum(), on.student.scale)[0]
    ok["the gradient reaches the student through the jittered newest frame"] = float(g.abs()) > 0
    m = build(camera_student="stub", student_trainable=True, student_frames="all").train()
    same_img = images(2, seed=9)
    b = dict(data(3), lane_img=same_img, lane_img_old=same_img[:, None].expand(2, 2, 3, 256, 704).contiguous())
    m.student.seen = []
    with torch.no_grad():
        m.camera_bevs(dict(b, img_jitter=AU.draw_jitter(2, 1.0, 5)))                   # the noise ON: identical frames differ by the noise alone
    cur_n, old_n = m.student.seen[0], m.student.seen[1].view(2, 2, 3, 256, 704)
    m.student.seen = None
    d0, d1 = float((cur_n[0] - old_n[0, 0]).abs().mean()), float((old_n[0, 0] - old_n[0, 1]).abs().mean())
    ok["model: the pixel noise of the older frames is not frame 0's (the sample's first frame has its own stream) and not each other's, and it is noise only (mean |difference| about 5.7 of 255)"] = \
        2 < d0 < 10 and 2 < d1 < 10 and not torch.equal(cur_n[0], old_n[0, 0])
    return ok


for name, v in jitter_model_checks().items():
    check(v, name)


def keyframe(batch):
    """the fields run_clip reads from a key frame: the ego steps to the neighbouring key frames, the tokens (the GT is not needed: the loss is stubbed)"""
    gaps = {"prev": [(-1 / 12, 8.0, 0.05)] * 6, "next": [(1 / 12, 8.0, 0.05)] * 6}
    return dict(batch, gap_steps=[gaps] * B, tokens=["a", "b"])


def clip_checks():
    """train.run_clip: when the jitter is drawn and from what (the model is a one-line recorder; no GT)"""
    ok = {}
    seen = []

    def stub_forward_loss(model, loss, batch, step_frac, device, amp, cfg=None, diag=False, train_terms=True, state=None, step=None):
        seen.append(batch.get("img_jitter"))
        res = model(batch, dts=(0.0,), state=state, step=step)
        return {"total": res["features"].square().mean()}, res, None

    m = build_rec(camera_student="stub", student_trainable=True, student_frames="all").train()
    orig = T.forward_loss

    def clip(frac, cfg, step=3, micro=1, augment=True):
        del seen[:]
        batches = [keyframe(mk(i)) for i in ([0, 1, 2], [3, 4, 5])]
        m.zero_grad(set_to_none=True)
        T.forward_loss = stub_forward_loss
        try:
            acc, finite, _, totals = T.run_clip(m, None, batches, frac, "cpu", "off", cfg, backward=True, augment=augment, step=step, micro=micro)
        finally:
            T.forward_loss = orig
        return list(seen), acc, finite, totals, {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}

    mk_cfg = lambda **kw: T.TrainConfig(seq_bptt=True, accum=1, flip_aug=0.0, **kw)  # noqa: E731
    off, on_ = mk_cfg(), mk_cfg(img_jitter=0.5, img_jitter_from=0.25, seed=4)
    s_off = clip(0.5, off)
    s_below, s_at, s_above = clip(0.24, on_), clip(0.25, on_), clip(0.9, on_)
    ok["run_clip: off (img_jitter 0) puts no draw into any key frame"] = all(j is None for j in s_off[0])
    ok["run_clip: below img_jitter_from the key frames carry no draw and the terms and every gradient equal the run without jitter; from it on both key frames carry one"] = \
        all(j is None for j in s_below[0]) and s_below[3] == clip(0.24, off)[3] and all(torch.equal(s_below[4][n], clip(0.24, off)[4][n]) for n in s_below[4]) \
        and all(j is not None for j in s_at[0] + s_above[0]) and len(s_at[0]) == 2
    ok["run_clip: above the threshold the clip's output differs from the run without jitter (the jitter reaches the loss)"] = s_above[3] != clip(0.9, off)[3]
    ok["run_clip: evaluation (augment False) never draws, whatever the config says"] = all(j is None for j in clip(0.9, on_, augment=False)[0])
    j_a, j_b = clip(0.9, on_, step=7, micro=2)[0], clip(0.9, on_, step=7, micro=2)[0]
    torch.manual_seed(1)
    torch.rand(11)                                                                  # the global random state moved on: a resume is not in the state the stopped run was
    j_c = clip(0.9, on_, step=7, micro=2)[0]
    ok["run_clip: the draw is a function of (seed, step, micro-batch): the same three give the same draw, whatever the global random state did (a resume repeats it)"] = \
        all(torch.equal(x["u"], y["u"]) and torch.equal(x["u"], z["u"]) and x["noise_seed"] == y["noise_seed"] == z["noise_seed"] for x, y, z in zip(j_a, j_b, j_c))
    j_step, j_micro, j_seed = clip(0.9, on_, step=8, micro=2)[0], clip(0.9, on_, step=7, micro=3)[0], clip(0.9, dataclasses.replace(on_, seed=5), step=7, micro=2)[0]
    ok["run_clip: another step, micro-batch or seed gives another draw"] = all(not torch.equal(j_a[0]["u"], o[0]["u"]) for o in (j_step, j_micro, j_seed))
    ok["run_clip: the two key frames of a clip share the draw u and differ in the noise seed; the draw has one column per clip of the micro-batch"] = \
        torch.equal(j_a[0]["u"], j_a[1]["u"]) and j_a[0]["noise_seed"] != j_a[1]["noise_seed"] and j_a[0]["u"].shape == (4, 2) and j_a[0]["strength"] == 0.5
    return ok


for name, v in clip_checks().items():
    check(v, name)


def train_jitter_checks():
    """the REAL train() (the toy of tests/support/toy_train.py: a tiny network, the data path stubbed) with img_jitter: step_fn hands run_clip the optimiser step and the micro-batch, so every micro-batch
    of every step has its own draw, and the same run draws the same again"""
    import toy_train as TT
    ok, out_root = {}, tempfile.mkdtemp(prefix="tj_", dir=tmp)                    # a fresh folder per call: a finished run of the same name would be resumed to its end and draw nothing

    def run(name, **kw):
        seen, saved = [], {k: getattr(T, k) for k in ("build", "forward_loss", "make_loader", "val_subset")}

        def forward(model, loss, batch, *a, **k):
            j = batch.get("img_jitter")
            seen.append(None if j is None else (j["u"].flatten().tolist(), j["noise_seed"]))
            return TT.toy_forward(model, loss, batch, *a, **k)
        T.build, T.forward_loss, T.make_loader, T.val_subset = (lambda cfg: (TT.ToyNet(), TT.ToyLoss(), True)), forward, (lambda ds, cfg, train, workers=None, generator=None: ds), (lambda ds, n: ds)
        base = dict(name=name, out_root=out_root, steps=8, accum=2, warmup=2, log_every=1, eval_every=0, save_every=4, keep=3, device="cpu", workers=0, flip_aug=0.0, seq_len=1, seed=0, lr=1e-2, grad_clip=0.5,
                    img_jitter=0.5, img_jitter_from=0.25)
        data = TT.toy_batches(20)
        try:
            with contextlib.redirect_stdout(_Capture()):
                T.train(T.TrainConfig(**dict(base, **kw)), datasets=(data, data))
        finally:
            for k, v in saved.items():
                setattr(T, k, v)
        return seen

    a, b, c = run("tj_a"), run("tj_b"), run("tj_c", seed=1)
    ok["train(): 8 steps x 2 micro-batches; no draw before step_frac 0.25 (steps 0 and 1), a draw in each of the 12 micro-batches after it"] = len(a) == 16 and all(x is None for x in a[:4]) and all(x is not None for x in a[4:])
    ok["train(): the two micro-batches of one step have different draws, and so have consecutive steps"] = all(a[i][0] != a[i + 1][0] for i in range(4, 15, 2)) and len({(tuple(x[0]), x[1]) for x in a[4:]}) == 12
    ok["train(): the same run again draws the same in every micro-batch (a function of (seed, step, micro-batch)); another seed draws other ones"] = a == b and a[4:] != c[4:] and all(x[0] != y[0] for x, y in zip(a[4:], c[4:]))
    return ok


for name, v in train_jitter_checks().items():
    check(v, name)


# ------------------------------------------------------------------------------------------------ F
print("F: window images on nuScenes mini")
MINI = os.path.expanduser("~/data/nuscenes")
HAVE_MINI = os.path.isdir(os.path.join(MINI, "v1.0-mini"))
SWEEPS = "sweeps"                                                    # written out here, not IC.OLDER: a changed folder name must fail the test


def data_checks():
    """the window images on the mini set: builder, marker, dataset item, collate; name -> bool (the temporary folder is removed by the caller)"""
    ok = {}
    root = tempfile.mkdtemp(prefix="imgs_", dir=tmp)
    IC._DS = mini
    IC._fill((root, toks))
    IC._fill_older((root, toks))
    want = {sd["token"] for t in toks for sd in walk(t, 2, 2)}
    got = {f[:-4] for d, _, fs in os.walk(os.path.join(root, SWEEPS)) for f in fs if f.endswith(".jpg")}
    ok["builder: _fill_older writes one JPEG per older frame (-2 and -4 sweeps of each sample) under <root>/sweeps, named by the frame's sample_data token (independent walk along prev)"] = got == want and len(want) == 6
    keys = {mini.gt.cam(t)["token"] for t in toks}
    ok["builder: the key frames are in <root>, not in sweeps, and the two sets are disjoint (the next key frame's oldest frame is at -6: no gap, no overlap)"] = \
        all(os.path.exists(IC.image_path(root, k)) for k in keys) and not (keys & got) and not any(os.path.exists(IC.image_path(os.path.join(root, SWEEPS), k)) for k in keys)
    errs = [float((decode(os.path.join(root, SWEEPS), sd["token"]).float() - torch.tensor(mini._image_uint8(sd)).permute(2, 0, 1).float()).abs().mean()) for t in toks for sd in walk(t, 2, 2)]
    ok[f"builder: every older JPEG is the dataset's own image of that frame (resize + crop 256 x 704) within JPEG error (max {max(errs):.2f} of 255)"] = max(errs) < 3.0
    before = {p: os.path.getmtime(p) for p in (IC.image_path(os.path.join(root, SWEEPS), sd["token"]) for t in toks for sd in walk(t, 2, 2))}
    IC._fill_older((root, toks))
    ok["builder: resumable (a second run rewrites nothing, no temporary file is left)"] = all(os.path.getmtime(p) == t_ for p, t_ in before.items()) and not any(".tmp" in f for _, _, fs in os.walk(root) for f in fs)
    # the dataset item
    cds = FC.CachedFrontRadarDataset(MINI, "v1.0-mini", store=StubStore(), n_cam=3, cam_stride=2, with_gt=False, lane_images=root, lane_images_older=True, nusc=mini.nusc)
    plain = FC.CachedFrontRadarDataset(MINI, "v1.0-mini", store=StubStore(), n_cam=3, cam_stride=2, with_gt=False, lane_images=root, nusc=mini.nusc)
    items = [cds[cds.samples.index(t)] for t in toks[:2]]
    ok["dataset item: lane_img = the JPEG of frames[0], lane_img_old = the JPEGs of frames[1:] in frame order (independent decode of the files the walk names), uint8 (F - 1, 3, 256, 704)"] = \
        all(torch.equal(it["lane_img"], decode(root, it["frames"][0]["token"])) and it["lane_img_old"].dtype == torch.uint8 and tuple(it["lane_img_old"].shape) == (2, 3, 256, 704)
            and [f["token"] for f in it["frames"][1:]] == [sd["token"] for sd in walk(t, 2, 2)]
            and all(torch.equal(it["lane_img_old"][k], decode(os.path.join(root, SWEEPS), walk(t, 2, 2)[k]["token"])) for k in range(2)) for it, t in zip(items, toks))
    ok["dataset item: the three images of a window differ (key frame, -2, -4)"] = all(len({it["lane_img"].float().mean().item(), it["lane_img_old"][0].float().mean().item(), it["lane_img_old"][1].float().mean().item()}) == 3 for it in items)
    ok["dataset item: without lane_images_older no lane_img_old"] = "lane_img_old" not in plain[plain.samples.index(toks[0])]
    batch = FC.collate_cached(items)
    ok["collate_cached: lane_img_old is (B, F - 1, 3, H, W) uint8 in item order, lane_img (B, 3, H, W)"] = tuple(batch["lane_img_old"].shape) == (2, 2, 3, 256, 704) and batch["lane_img_old"].dtype == torch.uint8 \
        and all(torch.equal(batch["lane_img_old"][i], items[i]["lane_img_old"]) for i in range(2)) and tuple(batch["lane_img"].shape) == (2, 3, 256, 704)
    miss = tempfile.mkdtemp(prefix="imgs_missing_", dir=tmp)
    IC._fill((miss, toks[:1]))                                                      # the key frame, no sweeps
    d_miss = FC.CachedFrontRadarDataset(MINI, "v1.0-mini", store=StubStore(), n_cam=3, cam_stride=2, with_gt=False, lane_images=miss, lane_images_older=True, nusc=mini.nusc)
    ok["dataset: a missing older image is a FileNotFoundError that names build-cache-older"] = raises(FileNotFoundError, lambda: d_miss[d_miss.samples.index(toks[0])], "build-cache-older")
    ok["dataset: lane_images_older without lane_images, and with n_cam 1, is refused"] = \
        raises(ValueError, lambda: FC.CachedFrontRadarDataset(MINI, "v1.0-mini", store=StubStore(), lane_images_older=True, nusc=mini.nusc), "lane_images") \
        and raises(ValueError, lambda: FC.CachedFrontRadarDataset(MINI, "v1.0-mini", store=StubStore(), lane_images=root, lane_images_older=True, n_cam=1, nusc=mini.nusc), "no older frame")
    return ok


def marker_checks():
    """the finished-build marker (BUILT.json) of build_cache(older=True): scenes, merged by a second build; startable.older_cache_findings reads it"""
    ok = {}
    root, key = tempfile.mkdtemp(prefix="marker_", dir=tmp), IC.older_key(3, 2)
    s1, s2 = {scene_of(t) for t in toks[:2]}, {scene_of(toks[2])}
    with contextlib.redirect_stdout(_Capture()):
        IC.build_cache([_subset(toks[:2])], root, 1, older=True)
    bt = IC.older_built(root)
    ok["marker: build_cache(older=True) writes BUILT.json with the window and the scenes of the datasets, after every sample"] = set(bt) == {key} and bt[key]["scenes"] == sorted(s1) and "when" in bt[key]
    with contextlib.redirect_stdout(_Capture()):
        IC.build_cache([_subset(toks[2:])], root, 1, older=True)
    ok["marker: a second build of the same window is merged (the scenes of both)"] = IC.older_built(root)[key]["scenes"] == sorted(s1 | s2)
    ok["marker: older_missing_scenes names the scenes no finished build of this window covers (an unknown scene, another stride, all of them without a marker)"] = \
        IC.older_missing_scenes(root, 3, 2, s1 | s2) == [] and IC.older_missing_scenes(root, 3, 2, s1 | {"scene-9999"}) == ["scene-9999"] \
        and IC.older_missing_scenes(root, 3, 4, s1) == sorted(s1) and IC.older_missing_scenes(os.path.join(tmp, "nothing"), 3, 2, s1) == sorted(s1)
    kroot = tempfile.mkdtemp(prefix="marker_keys_", dir=tmp)
    with contextlib.redirect_stdout(_Capture()):
        IC.build_cache([_subset(toks[:1])], kroot, 1)
    ok["marker: the key-frame build writes <root>/BUILT.json (key key_frames, its scenes), not a sweeps folder or an older-frame marker"] = \
        IC.older_built(kroot) == {} and not os.path.exists(os.path.join(kroot, SWEEPS)) and set(IC.key_built(kroot)) == {IC.KEY} and IC.key_built(kroot)[IC.KEY]["scenes"] == [scene_of(toks[0])]
    ok["marker: it records the builder's constants (JPEG quality, crop): the ones of the code now"] = IC.key_built(kroot)[IC.KEY]["settings"] == IC.build_settings() and bt[key]["settings"] == IC.build_settings() \
        and IC.build_settings()["jpeg_quality"] == IC.JPEG_QUALITY and IC.build_settings()["hw"] == [256, 704]
    cfg = T.TrainConfig(name="x", device="cpu", lane_images="x", n_cam=3, cam_stride=2, train_scenes=tuple(s1), val_scenes=tuple(s2))
    ok["startable: a finished build that covers the config's scenes is no problem; a scene it lacks, another stride, no sweeps folder and no scene list are named"] = \
        ST.older_cache_findings(root, cfg) == [] \
        and any(f"lacks 1 of the config's {len(s1 | s2) + 1} scenes" in x and "scene-9999" in x for x in ST.older_cache_findings(root, dataclasses.replace(cfg, val_scenes=cfg.val_scenes + ("scene-9999",)))) \
        and any("no finished build" in x for x in ST.older_cache_findings(root, dataclasses.replace(cfg, cam_stride=4))) \
        and any("build-cache-older" in x for x in ST.older_cache_findings(kroot, cfg)) and any("names no scenes" in x for x in ST.older_cache_findings(root, dataclasses.replace(cfg, train_scenes=(), val_scenes=())))
    kcfg = dataclasses.replace(cfg, train_scenes=(scene_of(toks[0]),), val_scenes=(scene_of(toks[0]),))
    ok["startable: the key-frame marker is asked for too: one that covers the scenes is no problem, a scene it lacks, no marker and a cache made with other constants are named"] = \
        ST.key_cache_findings(kroot, kcfg) == [] and any("lacks 1 of the config's" in x and "scene-9999" in x for x in ST.key_cache_findings(kroot, dataclasses.replace(kcfg, val_scenes=("scene-9999",)))) \
        and any("no finished key-frame build" in x for x in ST.key_cache_findings(root, kcfg)) \
        and any("were made with" in x for x in ST.key_cache_findings(_marker_with(kroot, settings=dict(IC.build_settings(), jpeg_quality=50)), kcfg)) \
        and any("was made with" in x for x in ST.older_cache_findings(_marker_with(root, older=True, settings=dict(IC.build_settings(), hw=[1, 1])), cfg))
    return ok


def _marker_with(src, older=False, **extra):
    """a copy of a cache folder's BUILT.json (key frames or the older frames) with `extra` added to its entry: the folder of the copy"""
    folder = tempfile.mkdtemp(prefix="marker_copy_", dir=tmp)
    built = IC.older_built(src) if older else IC.key_built(src)
    for k in built:
        built[k] = dict(built[k], **extra)
    target = os.path.join(folder, SWEEPS) if older else folder
    os.makedirs(target, exist_ok=True)
    json.dump({"built": built}, open(os.path.join(target, IC.BUILT), "w"))
    return folder


def crash_checks():
    """a build that dies half-way leaves no marker (it is written after the last sample), keeps the files it made, and the rerun resumes and then marks"""
    ok = {}
    for older in (True, False):
        root = tempfile.mkdtemp(prefix="crash_", dir=tmp)
        d = _subset(toks[:2])
        real, calls, n_ok = d._image_uint8, [0], (2 if older else 1)                       # two older images per sample, one key frame

        def flaky(sd):
            calls[0] += 1
            if calls[0] == n_ok + 1:
                raise RuntimeError("disk full")
            return real(sd)
        d._image_uint8 = flaky
        with contextlib.redirect_stdout(_Capture()):
            died = raises(RuntimeError, lambda: IC.build_cache([d], root, 1, older=older), "disk full")
        folder = os.path.join(root, SWEEPS) if older else root
        made = {os.path.join(dp, f): os.path.getmtime(os.path.join(dp, f)) for dp, _, fs in os.walk(folder) for f in fs if f.endswith(".jpg") and (older or SWEEPS not in dp)}
        marker = (IC.older_built if older else IC.key_built)(root)
        d._image_uint8 = real
        with contextlib.redirect_stdout(_Capture()):
            IC.build_cache([d], root, 1, older=older)
        marker2 = (IC.older_built if older else IC.key_built)(root)
        ok[f"{'an older-frame' if older else 'a key-frame'} build that crashes on its image after the first sample leaves no marker and keeps what it wrote; the rerun keeps those files and writes the marker"] = \
            died and marker == {} and len(made) == n_ok and bool(marker2) and all(os.path.getmtime(k) == t_ for k, t_ in made.items())
    return ok


def lock_checks():
    """two builds marking the same cache folder at the same time keep both updates (read-modify-write under a lock). The read is made slow so that both reads happen before either write unless the lock holds"""
    import multiprocessing as mp
    ok = {}
    folder = tempfile.mkdtemp(prefix="lock_", dir=tmp)
    nusc = types.SimpleNamespace(get=lambda table, tok: {"scene_token": tok} if table == "sample" else {"name": tok})
    ds = lambda scene: types.SimpleNamespace(n_cam=3, cam_stride=2, samples=[scene], nusc=nusc)       # noqa: E731
    real_read = IC._built

    def slow_read(f):
        r = real_read(f)
        time.sleep(0.5)
        return r
    IC._built = slow_read
    try:
        procs = [mp.get_context("fork").Process(target=IC._mark_built, args=(folder, IC.KEY, [ds(sc)])) for sc in ("scene-A", "scene-B")]
        for q in procs:
            q.start()
        for q in procs:
            q.join()
    finally:
        IC._built = real_read
    ok["two builds marking one folder at once keep both scenes (lock file)"] = all(q.exitcode == 0 for q in procs) and real_read(folder)[IC.KEY]["scenes"] == ["scene-A", "scene-B"]
    return ok



def _subset(ts):
    """the mini dataset restricted to the samples `ts` (a shallow copy)"""
    d = copy.copy(mini)
    d.samples = list(ts)
    return d


def cli_checks():
    """python net/model/lane_branch.py build-cache[-older] / check-cache[-older] end to end on one train and one val scene of mini, with the scene lists in the config"""
    ok = {}
    from nuscenes.utils.splits import create_splits_scenes
    sp = create_splits_scenes()
    cfg_path, root = os.path.join(tmp, "cli_config.json"), tempfile.mkdtemp(prefix="cli_", dir=tmp)
    json.dump({"_about": "t26 cli test", "name": "x", "data_root": MINI, "version": "v1.0-mini", "n_cam": 3, "cam_stride": 2, "lane_images": root, "gt_cache": "", "train_scenes": [sp["mini_train"][0]],
               "val_scenes": [sp["mini_val"][0]], "device": "cpu"}, open(cfg_path, "w"))
    say = {}
    for cmd in ("build-cache", "build-cache-older", "check-cache", "check-cache-older"):
        with contextlib.redirect_stdout(_Capture()) as cap:
            BIC._cli([cmd, cfg_path, "--n", "3"])
        say[cmd] = cap.text
    n_key = len(os.listdir(root)) - 1 if os.path.isdir(os.path.join(root, SWEEPS)) else len(os.listdir(root))
    sweeps = [f for d, _, fs in os.walk(os.path.join(root, SWEEPS)) for f in fs if f.endswith(".jpg")]
    keys = [f for d, _, fs in os.walk(root) for f in fs if f.endswith(".jpg") and SWEEPS not in d]
    ok[f"cli: build-cache-older fills sweeps/ with two JPEGs per sample ({len(sweeps)} older images for {len(keys)} key frames) and BUILT.json covers both scenes of the config"] = \
        len(sweeps) == 2 * len(keys) and set(IC.older_built(root)[IC.older_key(3, 2)]["scenes"]) == {sp["mini_train"][0], sp["mini_val"][0]} and "older frames of" in say["build-cache-older"]
    ok["cli: check-cache-older compares the val frames with the dataset's own image and the marker with the config's scenes; check-cache the key frames"] = \
        "older-frame cache vs dataset image" in say["check-cache-older"] and "BUILT.json covers all 2 scenes" in say["check-cache-older"] and "cache vs dataset image" in say["check-cache"] and "older" not in say["check-cache"]
    cfg2_path = os.path.join(tmp, "cli_config_student.json")                              # the t26 kind: a student 'all', feature caches named (that do not exist here), n_cam 3
    json.dump({**json.load(open(cfg_path)), "feature_cache": os.path.join(tmp, "no_cam_feats"), "feature_cache2": os.path.join(tmp, "no_cam_feats_bevfusion"),
               "model": {"camera_features": "dual", "camera_student": "evit_b1", "student_frames": "all", "student_trainable": True, "cam_buffer": "recurrent"}}, open(cfg2_path, "w"))
    try:
        for cmd in ("build-cache", "build-cache-older", "check-cache", "check-cache-older"):
            with contextlib.redirect_stdout(_Capture()):
                BIC._cli([cmd, cfg2_path, "--n", "2"])
        student_ok = True
    except Exception as e:                                                                # noqa: BLE001 - the finding is the message
        student_ok = f"{type(e).__name__}: {e}"
    ok[f"cli: all four commands also run on a config with a student 'all' and feature caches named (the t26 config; the builders read plain datasets) {'' if student_ok is True else student_ok}"] = student_ok is True
    ok["cli: build-cache wrote <root>/BUILT.json with both scenes of the config, check-cache compared it"] = set(IC.key_built(root)[IC.KEY]["scenes"]) == {sp["mini_train"][0], sp["mini_val"][0]} and "BUILT.json covers all 2 scenes" in say["check-cache"]
    kb = IC.key_built(root)
    kb[IC.KEY]["scenes"] = kb[IC.KEY]["scenes"][:1]
    json.dump({"built": kb}, open(os.path.join(root, IC.BUILT), "w"))
    ok["cli: check-cache fails when the key-frame marker lacks a scene of the config"] = raises(AssertionError, lambda: BIC._cli(["check-cache", cfg_path, "--n", "2"]), "no finished build of key_frames")
    bt = IC.older_built(root)
    bt[IC.older_key(3, 2)]["scenes"] = bt[IC.older_key(3, 2)]["scenes"][:1]
    json.dump({"built": bt}, open(os.path.join(root, SWEEPS, IC.BUILT), "w"))
    ok["cli: check-cache-older fails when the marker lacks a scene of the config"] = raises(AssertionError, lambda: BIC._cli(["check-cache-older", cfg_path, "--n", "2"]), "no finished build")
    split_cfg = os.path.join(tmp, "split_config.json")
    json.dump({"_about": "x", "name": "x", "split": "small", "lane_images": root, "device": "cpu"}, open(split_cfg, "w"))
    c = BIC.load_config(split_cfg)
    ok["cli: a config with a \"split\" is resolved to its scene lists (small: 60 train, 10 val), the comment key is ignored"] = len(c.train_scenes) == 60 and len(c.val_scenes) == 10
    return ok


if not HAVE_MINI:
    print("  SKIPPED: no nuScenes mini at ~/data/nuscenes (the window-image checks need it)")
else:
    from nuscenes_dataset import FrontRadarDataset  # noqa: E402
    from torchvision.io import decode_jpeg, read_file  # noqa: E402
    mini = FrontRadarDataset(MINI, "v1.0-mini", n_cam=3, cam_stride=2, with_gt=False)
    toks = [mini.samples[10], mini.samples[200], mini.samples[300]]
    scene_of = lambda t: mini.nusc.get("scene", mini.nusc.get("sample", t)["scene_token"])["name"]  # noqa: E731

    def walk(sample_token, n, stride):
        """independent of the builder: the sample_data of the key frame, then `stride` steps along `prev` per older frame (clamped at the start of the scene)"""
        sd = mini.gt.cam(sample_token)
        out = []
        for _ in range(n):
            for _ in range(stride):
                sd = mini.nusc.get("sample_data", sd["prev"]) if sd["prev"] else sd
            out.append(sd)
        return out

    def decode(folder, tok):
        return decode_jpeg(read_file(IC.image_path(folder, tok)))

    class StubStore:
        def feature_hw(self):
            return (16, 44)

        def get(self, tok):
            return torch.zeros(118, 16, 44, dtype=torch.float16), torch.zeros(80, 16, 44, dtype=torch.float16)

    for group in (data_checks, marker_checks, crash_checks, lock_checks, cli_checks):
        for name, v in group().items():
            check(v, name)

# ------------------------------------------------------------------------------------------------ G
print("G: wiring")
stu = dict(camera_student="stub", student_trainable=True)
check(not raises(ValueError, lambda: check_config(ModelConfig(**dict(DUAL, student_frames="all", **stu)))), "check_config: student_frames 'all' with a student is accepted")
check(raises(ValueError, lambda: check_config(ModelConfig(**dict(DUAL, student_frames="all"))), "student_frames", "without camera_student"), "check_config: student_frames 'all' without camera_student is refused")
check(raises(ValueError, lambda: check_config(ModelConfig(**dict(DUAL, student_frames="oldest", **stu))), "student_frames", "'all'"), "check_config: an unknown student_frames is refused and the kinds are named")
findings = lambda mc, **kw: T.student_findings(T.TrainConfig(name="x", device="cpu", feature_cache="c", lane_images="i", **kw), ModelConfig(**dict(DUAL, **mc)))  # noqa: E731
check(any("n_cam >= 2" in m for m in findings(dict(stu, student_frames="all"), n_cam=1)) and not any("n_cam" in m for m in findings(dict(stu, student_frames="all"), n_cam=3)),
      "student_findings: student_frames 'all' with n_cam 1 is reported, n_cam 3 is fine")
check(findings(stu, img_jitter=0.5, img_jitter_from=0.25) == [] and any("needs model.camera_student" in m for m in findings({}, img_jitter=0.5)) and findings({}, img_jitter=0.0) == [],
      "student_findings: image jitter with a student is fine, without one it is reported, off needs nothing")
check(all(any("img_jitter" in m for m in findings(stu, **kw)) for kw in (dict(img_jitter=-0.1), dict(img_jitter=2.5), dict(img_jitter_from=1.0), dict(img_jitter_from=-0.1))),
      "student_findings: img_jitter outside 0 .. 2 and img_jitter_from outside [0, 1) are reported")
check(raises(ValueError, lambda: T.build(T.TrainConfig(name="x", device="cpu", feature_cache="c1", feature_cache2="c2", lane_images="img", n_cam=1, model=dict(DUAL, student_frames="all", **stu))), "n_cam >= 2")
      and isinstance(T.build(T.TrainConfig(name="x", device="cpu", feature_cache="c1", feature_cache2="c2", lane_images="img", img_jitter=0.5, model=dict(DUAL, student_frames="all", **stu)))[0].student, Stub),
      "train.build: 'all' with n_cam 1 is refused, the complete config with image jitter builds")
d = T.TrainConfig()
check(d.img_jitter == 0.0 and d.img_jitter_from == 0.0 and "img_jitter" in T.RESUME_SWITCHES and "img_jitter" not in T.RESUME_FREE and "img_jitter_from" not in T.RESUME_FREE,
      "TrainConfig: both default off, img_jitter is a resume switch, neither may change on a resume")
seen_kw = []


class RecDS:
    def __init__(self, *a, **k):
        seen_kw.append(k.get("lane_images_older"))


fake_nusc = types.SimpleNamespace(scene=[], get=lambda *a: {"name": "x"})
real = (BT.CachedFrontRadarDataset, BT.FeatureStore, BT.available_scenes)
prev_mods = {k: sys.modules.get(k) for k in ("nuscenes", "nuscenes.nuscenes")}
sys.modules["nuscenes"], sys.modules["nuscenes.nuscenes"] = types.ModuleType("nuscenes"), types.SimpleNamespace(NuScenes=lambda *a, **k: fake_nusc)
BT.CachedFrontRadarDataset, BT.FeatureStore, BT.available_scenes = RecDS, lambda p: None, lambda n, s: []
try:
    for frames in ("all", "current"):
        T.make_datasets(T.TrainConfig(name="x", device="cpu", feature_cache="c", feature_cache2="c2", lane_images="img", model={**DUAL, "camera_student": "stub", "student_frames": frames}))
    T.make_datasets(T.TrainConfig(name="x", device="cpu", feature_cache="c", model=dict(BASE)))
finally:
    BT.CachedFrontRadarDataset, BT.FeatureStore, BT.available_scenes = real
    for k, v in prev_mods.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
check(seen_kw == [True, True, False, False, False, False], f"train.make_datasets asks the datasets for the older images exactly with a student and student_frames 'all' (train, val: {seen_kw})")
zeros = torch.zeros
contract = {"K": zeros(2, 3, 3, 3), "T_cam_car": zeros(2, 3, 4, 4), "ida": None, "v": zeros(2), "omega": zeros(2), "steps": [[] for _ in range(3)], "tokens": ["a", "b"],
            "radar": dict({f: zeros(5) for f in IF.RADAR_FIELDS}, b=torch.tensor([0, 0, 1, 1, 1])), "prob": zeros(2, 3, 4, 2, 3), "ctx": zeros(2, 3, 80, 2, 3), "hw": (16, 24)}
check(IF.check_inputs(contract) == [] and IF.check_inputs(dict(contract, lane_img_old=zeros(2, 2, 3, 4, 4), img_jitter=AU.draw_jitter(2, 1.0, 0))) == []
      and any("lane_img_old" in e for e in IF.check_inputs(dict(contract, lane_img_old=zeros(2, 3, 3, 4, 4)))) and any("lane_img_old" in e for e in IF.check_inputs(dict(contract, lane_img_old=zeros(3, 2, 3, 4, 4)))),
      "interface: lane_img_old (B, F - 1, 3, H, W) and img_jitter are known inputs; a wrong number of older frames or of samples is named")
from evaluation import bev_adapter as A_  # noqa: E402
from evaluation import bev_modules as BM_  # noqa: E402
from evaluation import bev_probe as BP_  # noqa: E402
on_c = build(camera_student="stub", student_trainable=True, student_frames="all").eval()
bc = data(3)
with torch.no_grad():
    base_c, zero_c = on_c.camera_bevs(bc), on_c.camera_bevs(BM_.zero_cam(bc))
    zero_cache_only = on_c.camera_bevs({**bc, **{k: bc[k] * 0 for k in ("prob", "ctx", "prob2", "ctx2", "lane_img")}})
check(all(not torch.equal(zero_c[:, k], base_c[:, k]) for k in range(3)) and not torch.equal(zero_c[:, 1:], zero_cache_only[:, 1:]),
      "the evaluation's zero_cam changes the BEV of EVERY frame of an 'all' model (a zeroing that left lane_img_old in would not)")
pb = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in bc.items()}
pb["tokens"] = ["a", "b"]
BP_.perturb(pb, "zero_cam")
pb2 = {"tokens": ["a", "b"], "lane_img": bc["lane_img"].clone(), "lane_img_old": bc["lane_img_old"].clone()}
BP_.perturb(pb2, "shuffle_cam")
check(int(pb["lane_img_old"].sum()) == 0 and int(pb["lane_img"].sum()) == 0 and torch.equal(pb2["lane_img_old"], torch.roll(bc["lane_img_old"], 1, 0)), "bev_probe.perturb: zero_cam zeroes lane_img_old, shuffle_cam rolls it over the samples")
item_c = lambda v: {"frames": [{k: torch.full((2,), float(v)) for k in A_.CAMERA_FRAME_KEYS} | {"token": "t"} for _ in range(3)], "lane_img": torch.full((3, 2), v, dtype=torch.uint8),  # noqa: E731
                    "lane_img_old": torch.full((2, 3, 2), v, dtype=torch.uint8)}
it_, other_ = item_c(5), item_c(9)
A_.perturb_camera(it_, other_, "zero_cam")
it2 = item_c(5)
A_.perturb_camera(it2, other_, "shuffle_cam")
check(int(it_["lane_img"].sum()) == 0 and int(it_["lane_img_old"].sum()) == 0 and int(it2["lane_img_old"][0, 0, 0]) == 9, "bev_adapter.perturb_camera: zero_cam zeroes lane_img_old of a dataset item, shuffle_cam replaces it with the other item's")

# ------------------------------------------------------------------------------------------------ H
print("H: planted bugs")


def inst(name, fn):
    """a hook for section H: every model build() returns gets its method `name` replaced by fn(original bound method, self, *args)"""
    def hook(m):
        orig = getattr(m, name)
        setattr(m, name, types.MethodType(lambda self, *a, **k: fn(orig, self, *a, **k), m))
    return hook


def older_with_graph(self, img):
    B, K = img.shape[:2]
    out = self.student(img.flatten(0, 1))                                          # the bug: no torch.no_grad()
    return {k: v.reshape(B, K, *v.shape[1:]) for k, v in out.items() if k.startswith(("prob", "ctx"))}


def current_without_graph(self, img):
    with torch.no_grad():
        return self.student(img)


def jit_in_eval(orig, self, batch, Fn):
    was, self.training = self.training, True                                       # the bug: the jitter does not look at the mode
    try:
        return orig(batch, Fn)
    finally:
        self.training = was


def student_on_frame_one(orig, self, batch, Fn):
    cur, old = orig(batch, Fn)
    return (cur, old) if old is None else (old[:, 0], old)


real_jitter = AU.jitter_images


def jitter_per_frame(img, j, salt=0):                                              # the bug: another draw for every frame of a sample
    if img.dim() != 5:
        return real_jitter(img, j, salt)
    return torch.stack([real_jitter(img[:, k], dict(j, u=j["u"].roll(k + 1, 1)), salt) for k in range(img.shape[1])], 1)


def jitter_unseeded_noise(img, j, salt=0):                                         # the bug: the pixel noise from the global random state
    old, AU.JITTER_NOISE = AU.JITTER_NOISE, 0.0
    try:
        x = real_jitter(img, j, salt)
    finally:
        AU.JITTER_NOISE = old
    return (x + 255 * AU.JITTER_NOISE * j["strength"] * torch.randn_like(x)).clamp(0, 255)


real_clip, real_collate = T.run_clip, FC.collate_cached
real_build = IC.build_cache


def marks_first(datasets, root, workers=1, older=False):                          # the bug: the marker written before the work
    if datasets:
        IC._mark_built(IC.older_root(root), IC.older_key(datasets[0].n_cam, datasets[0].cam_stride), datasets) if older else IC._mark_built(root, IC.KEY, datasets)
    return real_build(datasets, root, workers, older)


def clip_without_step(*a, **k):                                                    # the bug: train.step_fn does not pass step / micro
    return real_clip(*a, **{kk: vv for kk, vv in k.items() if kk not in ("step", "micro")})


def collate_flipped(items):
    b = real_collate(items)
    return dict(b, lane_img_old=b["lane_img_old"].flip(1)) if "lane_img_old" in b else b


def always_from_start(m_, l_, c_, f_, d_, a_, cfg_, **k):                          # the bug: img_jitter_from ignored
    return real_clip(m_, l_, c_, f_, d_, a_, dataclasses.replace(cfg_, img_jitter_from=0.0), **k)


# name: (patches [(object, attribute, new)], hook for every model, groups to try in this order)
MODEL_G, JIT_G = ("model", "stream"), ("jfn", "jmodel", "clip")
MUTANTS = {
    "the older frames' student forward keeps its graph (no torch.no_grad())": ([], inst("_run_student_older", lambda orig, self, img: older_with_graph(self, img)), MODEL_G),
    "the newest frame's student forward has no graph": ([], inst("_run_student", lambda orig, self, img: current_without_graph(self, img)), MODEL_G),
    "the older frames in the wrong order": ([], inst("_run_student_older", lambda orig, self, img: orig(img.flip(1))), MODEL_G),
    "the older images of the wrong sample": ([], inst("_run_student_older", lambda orig, self, img: orig(img.roll(1, 0))), MODEL_G),
    "the student reads frame 1's image for frame 0 (and so does the anchor)": ([], inst("_student_images", student_on_frame_one), MODEL_G),
    "the older images are ignored (frames >= 1 stay the cached teachers')": ([], inst("_student_images", lambda orig, self, batch, Fn: (orig(batch, Fn)[0], None)), MODEL_G),
    "the jitter does not look at the training mode": ([], inst("_student_images", jit_in_eval), JIT_G),
    "the jitter is drawn per frame instead of per sample": ([(AU, "jitter_images", jitter_per_frame), (CM, "jitter_images", jitter_per_frame)], None, JIT_G),
    "the pixel noise comes from the global random state": ([(AU, "jitter_images", jitter_unseeded_noise), (CM, "jitter_images", jitter_unseeded_noise)], None, JIT_G),
    "the draw comes from the global random state, not from (seed, step, micro)": ([(T, "draw_jitter", lambda B, s, seed, key=0: {"u": torch.rand(4, B) * 2 - 1, "strength": float(s), "noise_seed": int(seed) * 64 + key})], None, JIT_G),
    "the draw ignores the step": ([(T, "jitter_seed", lambda seed, step, micro: int(seed) * 1009 + int(micro))], None, JIT_G),
    "img_jitter_from is ignored (the jitter is on from step 0)": ([(T, "run_clip", always_from_start)], None, JIT_G),
    "the brightness / contrast / saturation range is 0.2, not 0.4": ([(AU, "JITTER_RANGE", 0.2)], None, JIT_G),
    "the gamma range is 1.5, not 1.25": ([(AU, "JITTER_GAMMA", 1.5)], None, JIT_G),
    "the older frames' noise uses frame 0's stream (salt 0)": ([(CM, "jitter_images", lambda img, j, salt=0: real_jitter(img, j, 0))], None, ("jmodel",)),
    "train() does not pass the step and the micro-batch to run_clip": ([(T, "run_clip", clip_without_step)], None, ("train",)),
}
if HAVE_MINI:
    MUTANTS.update({
        "collate_cached drops lane_img_old": ([(FC, "collate_cached", lambda items: {k: v for k, v in real_collate(items).items() if k != "lane_img_old"})], None, ("data",)),
        "collate_cached stacks the older images in the wrong order": ([(FC, "collate_cached", collate_flipped)], None, ("data",)),
        "the builder caches the key frame again instead of the older frames": ([(IC, "older_records", lambda ds, t: [ds.gt.cam(t)] * (ds.n_cam - 1))], None, ("data",)),
        "the older frames go into the key-frame folder (no sweeps/)": ([(IC, "older_root", lambda root: root)], None, ("data",)),
        "the marker covers every scene": ([(IC, "older_missing_scenes", lambda *a: []), (ST, "older_missing_scenes", lambda *a: [])], None, ("marker",)),
        "the key-frame marker covers every scene": ([(ST, "key_missing_scenes", lambda *a: []), (IC, "key_missing_scenes", lambda *a: [])], None, ("marker",)),
        "the marker is written before the work (a crash leaves one)": ([(IC, "build_cache", marks_first)], None, ("crash",)),
        "the marker is written without the lock": ([(sys.modules["fcntl"], "flock", lambda *a: None)], None, ("lock",)),
    })
GROUPS = {"model": model_checks, "stream": stream_batch_checks, "jfn": jitter_fn_checks, "jmodel": jitter_model_checks, "clip": clip_checks, "train": train_jitter_checks}
if HAVE_MINI:
    GROUPS.update({"data": data_checks, "marker": marker_checks, "crash": crash_checks, "lock": lock_checks})


@contextlib.contextmanager
def planted(patches, hook):
    saved = [(o, a, getattr(o, a)) for o, a, _ in patches]
    for o, a, new in patches:
        setattr(o, a, new)
    MUTATE[0] = hook
    try:
        yield
    finally:
        MUTATE[0] = None
        for o, a, old in saved:
            setattr(o, a, old)


survived, killed = [], 0
for name, (patches, hook, groups) in MUTANTS.items():
    why = None
    with planted(patches, hook):
        for g in groups:
            try:
                bad = [k for k, v in GROUPS[g]().items() if not v]
            except Exception as e:                                                  # noqa: BLE001 - a planted bug that crashes a check is caught as well
                bad = [f"raised {type(e).__name__}: {e}"]
            if bad:
                why = f"{g}: {bad[0][:110]}" + (f" (+{len(bad) - 1})" if len(bad) > 1 else "")
                break
    check(why is not None, f"mutant killed: {name} -> {why or 'SURVIVED'}")
    killed += why is not None
    if why is None:
        survived.append(name)
print(f"  planted bugs killed: {killed} of {len(MUTANTS)}" + (f"; survived: {survived}" if survived else ""))
check(all(v for g in ("model", "jfn") for v in GROUPS[g]().values()), "after the planted bugs the unmutated code passes again")


print(f"\ntest_student_window: {'ALL PASSED' if not fails else str(len(fails)) + ' FAILED'} ({time.time() - t0:.0f} s)")
sys.exit(1 if fails else 0)
