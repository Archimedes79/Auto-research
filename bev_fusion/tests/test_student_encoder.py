"""The student image encoder (training/student_encoder.py) and its offline distillation (training/distill_student.py), t22 stage A. CPU, 2 threads, no GPU, no real data: known
answers and a synthetic key-frame tree. It does read the ImageNet weight files under models/pretrained/imagenet/ of the working directory (sections 2 and 6 build the student with them;
section 2 skips the file it does not find, the CLI runs of section 6 need them).

  1. contract: exact shapes of the six outputs at 256 x 704 (H16 16 x 44, H8 32 x 88), the probabilities are the softmax of the logits and sum to 1, uint8 = float input, a wrong
     window is refused; the checks are run against five mutants (wrong stride of either head, a channel split of 80 / 118, unnormalised probabilities) and each is caught
  2. weights: the EfficientViT-B1 and ResNet-18 files load strictly (when present); resnet18 gives the same output shapes
  3. H8 geometry: the fixed resampling puts a ramp of native pixel positions where the closed forms x_d = 0.44 / 0.48 (x_b + 32), y_d = 0.44 / 0.48 (y_b + 176) - 140 say (BEVFusion's
     frustum cell -> the JPEG's window), through the model's own decode; a missing warp, align_corners=False and swapped axes are caught (> 1e-3 px)
  4. training: with the backbone frozen the gradient reaches every FPN / head parameter and no backbone parameter and its BatchNorm statistics stay put; unfrozen it reaches every
     parameter; a teacher with perturbed FPN / heads is fitted by the real loss (> 50 % down in 30 steps); `lr_groups` give the backbone 0.1 x through the engine; the state_dict
     round trip is exact and does not hold the fixed buffers
  5. loss, validation, bands: known answers of the cell terms (identical = 0, one-hot against uniform = log 118, cosine -1, an unnormalised teacher), the flat-ground range bands of both
     heads' rows, `score` against planted errors (a context error in one row, a KL error in one row: the per-row and per-column curves), the null baseline (mean of a fake store with
     its floor, the first n frames only) and the skill (student = teacher: 1, student = null: 0), the samples/s and data-wait arithmetic on a fake clock, the resource gate
  6. data and CLI on a synthetic tree (shards + JPEGs): the key frames are the shard tokens with a JPEG, scene restrictions and a missing token are errors, the H16 and H8 targets of one
     item are those of its token, the loaders (drop_last on train only, shuffle on train only); the CLI trains 4 steps with a stop at step 2 (checkpoint, STOPPED, exit 75), refuses to
     go on without --resume (exit 2), resumes to the end and writes metrics.json with the EMA weights (the weights the validation saw are the checkpoint's EMA, not the live ones), the
     null baseline and the skill; the eval_every rule; the gate (exit 3) and the refusal of a missing CUDA (exit 3); --eval-only reproduces the run's validation from its checkpoint,
     writes eval_metrics.json and leaves metrics.json alone
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_student_encoder.py
"""
import copy
import gc
import json
import os
import re
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

torch.set_num_threads(2)
sys.path.insert(0, "training")
import distill_student as D  # noqa: E402
import student_encoder as S  # noqa: E402
from feature_cache import write_shard  # noqa: E402
from trainkit.optim import param_groups  # noqa: E402

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def make(backbone="efficientvit_b1", seed=0, **kw):
    torch.manual_seed(seed)
    return S.StudentEncoder(backbone, **kw)


def problems(out, B):
    """what is wrong with an output dict (empty list = nothing)"""
    want = {"prob16": (B, 118, 16, 44), "logit16": (B, 118, 16, 44), "ctx16": (B, 80, 16, 44), "prob8": (B, 118, 32, 88), "logit8": (B, 118, 32, 88), "ctx8": (B, 80, 32, 88)}
    bad = [f"{k} {tuple(out[k].shape) if k in out else None} != {s}" for k, s in want.items() if k not in out or tuple(out[k].shape) != s]
    if set(out) != set(want):
        bad.append(f"keys {sorted(out)}")
    if not bad:
        for k in ("16", "8"):
            p = out["prob" + k]
            if not torch.allclose(p.sum(1), torch.ones_like(p[:, 0]), atol=1e-5) or (p < 0).any() or not torch.allclose(p, out["logit" + k].softmax(1), atol=1e-6):
                bad.append(f"prob{k} is not the softmax of logit{k} / does not sum to 1")
    return bad


def ramp_error(grid, align_corners=True):
    """max abs error (pixels) of the ramp of native pixel positions resampled by `grid` against the closed forms (borders clamp to the first / last cell centre)"""
    xs, ys = 8 * torch.arange(88) + 3.5, 8 * torch.arange(32) + 3.5
    ramp = torch.stack([xs.view(1, 88).expand(32, 88), ys.view(32, 1).expand(32, 88)])[None]
    got = F.grid_sample(ramp, grid, mode="bilinear", padding_mode="border", align_corners=align_corners)[0]
    xb, yb = torch.arange(88) * 703 / 87, torch.arange(32) * 255 / 31
    want = torch.stack([(0.44 / 0.48 * (xb + 32)).view(1, 88).expand(32, 88), (0.44 / 0.48 * (yb + 176) - 140).view(32, 1).expand(32, 88)])
    want = torch.stack([want[0].clamp(3.5, 8 * 87 + 3.5), want[1].clamp(3.5, 8 * 31 + 3.5)])
    return float((got - want).abs().max())


# 1 contract ------------------------------------------------------------------------------------------------------------------------------------------------------
m = make(pretrained=False).eval()
img = torch.randint(0, 256, (2, 3, 256, 704), dtype=torch.uint8)
with torch.no_grad():
    out = m(img)
    out_f = m(img.float())
check(not problems(out, 2), f"shapes of the six outputs at 256 x 704 (H16 {tuple(out['prob16'].shape[-2:])}, H8 {tuple(out['prob8'].shape[-2:])}), softmax, normalised")
check(all(torch.equal(out[k], out_f[k]) for k in out), "a uint8 image and the same values as float give the same outputs")
seen = []
hook = m.backbone.register_forward_pre_hook(lambda mod, inp: seen.append(inp[0]))
with torch.no_grad():
    m(torch.tensor([255, 0, 0], dtype=torch.uint8).view(1, 3, 1, 1).expand(1, 3, 256, 704).contiguous())
hook.remove()
check(torch.allclose(seen[0][0, :, 0, 0], torch.tensor([(255 - 123.675) / 58.395, -116.28 / 57.12, -103.53 / 57.375]), atol=1e-5),
      "the backbone gets (RGB - mean) / std with the DualBEV constants: a pure red pixel -> (2.249, -2.036, -1.804)")
try:
    m(img[..., :128, :352])
    check(False, "a window other than 256 x 704 is refused")
except ValueError:
    check(True, "a window other than 256 x 704 is refused")
mutants = {"H16 at stride 32": lambda o: {**o, "prob16": F.avg_pool2d(o["prob16"], 2)}, "H8 at stride 16": lambda o: {**o, "prob8": o["prob8"][:, :, ::2, ::2]},
           "ctx16 at stride 8": lambda o: {**o, "ctx16": F.interpolate(o["ctx16"], scale_factor=2)}, "80 depth bins": lambda o: {**o, "prob8": o["prob8"][:, :80]},
           "unnormalised probabilities": lambda o: {**o, "prob16": 0.5 * o["prob16"]}}
for name, mut in mutants.items():
    check(bool(problems(mut(out), 2)), f"the contract check catches the mutant '{name}'")

# 2 weights, resnet18 ---------------------------------------------------------------------------------------------------------------------------------------------
if os.path.exists(S.WEIGHTS["efficientvit_b1"]):
    from safetensors.torch import load_file
    t0 = time.time()
    real = make()
    ref = load_file(S.WEIGHTS["efficientvit_b1"])
    own = {re.sub(r"^stages_(\d)", r"stages.\1", k.replace("stem_in_conv", "stem.in_conv").replace("stem_res0", "stem.res0")): v for k, v in real.backbone.state_dict().items()}   # timm's feature net flattens the first names
    check(len(own) == 261 and all(k in ref and torch.equal(v, ref[k]) for k, v in own.items()), f"EfficientViT-B1 weights loaded from the local file (all {len(own)} backbone tensors equal the file's, {time.time() - t0:.1f} s)")
    check(4.6e6 < sum(p.numel() for p in real.backbone.parameters()) < 4.7e6, "backbone has 4.6 M parameters (no classifier)")
    del real
else:
    print("  SKIP EfficientViT-B1 weights: file missing")
r18 = make("resnet18", pretrained=os.path.exists(S.WEIGHTS["resnet18"])).eval()
with torch.no_grad():
    check(not problems(r18(img[:1]), 1), "resnet18: the same output shapes (pretrained weights " + ("loaded strictly)" if os.path.exists(S.WEIGHTS["resnet18"]) else "missing: random)"))
del r18

# 3 the H8 geometry -----------------------------------------------------------------------------------------------------------------------------------------------
grid, xy = S.h8_warp_grid()
check(ramp_error(grid) < 1e-3, f"the warp grid puts a ramp where the closed forms say (max error {ramp_error(grid):.1e} px)")
check(abs(float(xy[0, 0, 0]) - 32 * 11 / 12) < 1e-3 and abs(float(xy[0, 0, 1]) - (176 * 11 / 12 - 140)) < 1e-3 and abs(float(xy[-1, -1, 1]) - (431 * 11 / 12 - 140)) < 1e-3,
      "corner cells: (29.33, 21.33) at the top left, row 255.08 at the bottom (BEVFusion's window is a part of the JPEG's)")


class RampHead(nn.Module):
    """ignores its input: native pixel x / y of every stride-8 cell centre in the first two channels"""

    def forward(self, x):
        B, _, h, w = x.shape
        o = torch.zeros(B, 198, h, w)
        o[:, 0] = (8 * torch.arange(w) + 3.5).view(1, 1, w)
        o[:, 1] = (8 * torch.arange(h) + 3.5).view(1, h, 1)
        return o


zero_feats = [torch.zeros(1, c, 256 // s, 704 // s) for c, s in ((32, 4), (64, 8), (128, 16), (256, 32))]
ramp_model = make(pretrained=False).eval()
ramp_model.head8 = RampHead()
with torch.no_grad():
    got = torch.stack([ramp_model.decode(zero_feats)["logit8"][0, 0], ramp_model.decode(zero_feats)["logit8"][0, 1]])
    ramp_model.grid8 = None
    native = ramp_model.decode(zero_feats)["logit8"][0, :2]
want = torch.stack([0.44 / 0.48 * (torch.arange(88) * 703 / 87 + 32).view(1, 88).expand(32, 88), (0.44 / 0.48 * (torch.arange(32) * 255 / 31 + 176) - 140).view(32, 1).expand(32, 88)])
want = torch.stack([want[0].clamp(3.5, 8 * 87 + 3.5), want[1].clamp(3.5, 8 * 31 + 3.5)])
check(float((got - want).abs().max()) < 1e-3, "through the model's decode: H8 carries the position of the image point BEVFusion's cell looks at")
check(float((native - want).abs().max()) > 20, "negative control: warp_h8 off leaves H8 in the JPEG's window (cells off by more than 20 px)")
identity = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 88), torch.linspace(-1, 1, 32), indexing="xy"), -1)[None]
for name, g, ac in (("no warp (identity grid)", identity, True), ("align_corners=False", grid, False), ("x and y swapped", grid.flip(-1), True)):
    check(ramp_error(g, ac) > 1e-3, f"the geometry check catches the mutant '{name}' ({ramp_error(g, ac):.2f} px)")

# 4 training ------------------------------------------------------------------------------------------------------------------------------------------------------
x1 = img[:1]


def all_terms(o):
    return sum(o[k].pow(2).mean() for k in ("logit16", "ctx16", "logit8", "ctx8"))


mf = make(pretrained=False)
mf.freeze_backbone(True)
mf.train()
bn = next(mod for mod in mf.backbone.modules() if isinstance(mod, nn.BatchNorm2d))
before = bn.running_mean.clone()
all_terms(mf(x1)).backward()
back = [p for p in mf.backbone.parameters()]
rest = [(n, p) for n, p in mf.named_parameters() if not n.startswith("backbone.")]
check(all(p.grad is None for p in back) and not any(p.requires_grad for p in back), "frozen: no backbone parameter gets a gradient")
dead = [n for n, p in rest if p.grad is None or float(p.grad.abs().sum()) == 0.0]
check(len(rest) == 20 and not dead, f"frozen: the gradient reaches every FPN / head parameter ({len(rest)} tensors, dead: {dead})")
check(not mf.backbone.training and torch.equal(bn.running_mean, before), "frozen: the backbone's BatchNorm stays in eval mode and keeps its statistics under .train()")
mu = make(pretrained=False)
mu.train()
bn_u = next(mod for mod in mu.backbone.modules() if isinstance(mod, nn.BatchNorm2d))
before_u = bn_u.running_mean.clone()
all_terms(mu(x1)).backward()
dead = [n for n, p in mu.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
check(not dead and not torch.equal(bn_u.running_mean, before_u), f"unfrozen: the gradient reaches every parameter ({sum(1 for _ in mu.parameters())} tensors, dead: {dead[:3]}), BatchNorm statistics move")

mf.zero_grad(set_to_none=True)
teacher = copy.deepcopy(mf).eval()
torch.manual_seed(1)
with torch.no_grad():
    for n, p in teacher.named_parameters():
        if not n.startswith("backbone."):
            p.add_(0.5 * p.std() * torch.randn_like(p))
    feats = mf.eval().encode(x1)
    to = teacher.decode(feats)
    tgt = {"p16": to["prob16"], "c16": to["ctx16"], "p8": to["prob8"], "c8": to["ctx8"]}
mf.train()
opt = torch.optim.Adam([p for p in mf.parameters() if p.requires_grad], lr=3e-3)
losses = []
for _ in range(30):
    opt.zero_grad()
    loss = D.distill_terms(mf.decode([f.detach() for f in feats]), tgt)["total"]
    loss.backward()
    opt.step()
    losses.append(float(loss))
check(losses[-1] < 0.5 * losses[0], f"a perturbed teacher is fitted by the real loss: {losses[0]:.3f} -> {losses[-1]:.3f} in 30 steps")

groups = param_groups(mu, nn.Module(), 1e-3, 1e-2, mu.lr_groups(1e-3, 0.1))
bb_ids = {id(p) for p in mu.backbone.parameters()}
bb_lr = {g["base_lr"] for g in groups if "base_lr" in g and all(id(p) in bb_ids for p in g["params"])}
check(bb_lr == {1e-4} and sum(len(g["params"]) for g in groups) == sum(1 for _ in mu.parameters()) and not any(id(p) in bb_ids for g in groups if "base_lr" not in g for p in g["params"]),
      "lr_groups: the backbone's parameters in groups of 0.1 x lr, FPN and heads at lr, no parameter lost")
check(mf.lr_groups(1e-3) == {}, "lr_groups of a frozen backbone is empty (trainkit refuses a prefix without a trainable parameter)")

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "student.pt")
    torch.save(mu.state_dict(), path)
    twin = make(seed=5, pretrained=False, warp_h8=False)
    twin.load_state_dict(torch.load(path, weights_only=True))
    twin.grid8 = mu.grid8
    mu.eval(), twin.eval()
    with torch.no_grad():
        a, b = mu(x1), twin(x1)
    check(all(torch.equal(a[k], b[k]) for k in a), "state_dict round trip: identical outputs")
    check(not any(k in mu.state_dict() for k in ("mean", "std", "grid8")), "the fixed buffers are not in the checkpoint (a warp on / off checkpoint loads either way)")

# 5 loss, bands, gate ---------------------------------------------------------------------------------------------------------------------------------------------
B, L = 2, 118
zero = {k: torch.zeros(B, L, h, w) for k, (h, w) in (("logit16", (16, 44)), ("logit8", (32, 88)))}
onehot = {"p16": torch.zeros(B, L, 16, 44), "p8": torch.zeros(B, L, 32, 88)}
for p in onehot.values():
    p[:, 0] = 1.0
ones = {"c16": torch.ones(B, 80, 16, 44), "c8": torch.ones(B, 80, 32, 88)}
same = {"logit16": torch.randn(B, L, 16, 44), "logit8": torch.randn(B, L, 32, 88), "ctx16": torch.randn(B, 80, 16, 44), "ctx8": torch.randn(B, 80, 32, 88)}
t_same = D.distill_terms(same, {"p16": same["logit16"].softmax(1), "p8": same["logit8"].softmax(1), "c16": same["ctx16"], "c8": same["ctx8"]})
check(max(abs(float(v)) for v in t_same.values()) < 1e-5, "student = teacher: every term is 0")
hand = D.distill_terms({**zero, "ctx16": -ones["c16"], "ctx8": -ones["c8"]}, {**onehot, **ones})
check(all(abs(float(hand[k + s]) - v) < 1e-5 for s in ("16", "8") for k, v in (("kl", np.log(118)), ("cosl", 2.0), ("mse", 4.0))) and abs(float(hand["total"]) - 2 * (np.log(118) + 6)) < 1e-4,
      f"one-hot teacher against a uniform student: KL = log 118, opposite context: 1 - cos = 2, mse = 4 (total {float(hand['total']):.4f})")
half = D.distill_terms({**zero, "ctx16": ones["c16"], "ctx8": ones["c8"]}, {"p16": 0.5 * onehot["p16"], "p8": 0.5 * onehot["p8"], **ones})
check(abs(float(half["kl16"]) - np.log(118)) < 1e-5, "an unnormalised teacher (float16 storage) is renormalised before the KL")
rows16, rows8 = D.band_rows(D.head_rows(m)["16"]), D.band_rows(D.head_rows(m)["8"])
check({k: v.tolist() for k, v in rows16.items()} == {"near": list(range(8, 16)), "mid": [7], "far": [6]}, "H16 rows by flat-ground range: far {6} (30.7 m), mid {7} (19.3 m), near 8..15 (< 15 m)")
check({k: v.tolist() for k, v in rows8.items()} == {"near": list(range(15, 32)), "mid": [11, 12, 13, 14], "far": [10]},
      "H8 rows (BEVFusion frustum rows seen in the JPEG's window, 7.54 px per row from row 21.3): far {10}, mid 11..14, near 15..31; rows 0..9 are above the horizon or beyond 60 m")
t16, t8, c16, c8 = torch.randn(B, L, 16, 44), torch.randn(B, L, 32, 88), torch.randn(B, 80, 16, 44), torch.randn(B, 80, 32, 88)
bad16, bad8 = c16.clone(), c8.clone()
bad16[:, :, 6] += 1                                                            # a context error of 1 in H16 row 6 (far band) and H8 row 10 (far band)
bad8[:, :, 10] += 1


class Fixed(nn.Module):
    grid8 = torch.ones(1)                                                      # a warped H8: its rows are BEVFusion's frustum rows

    def forward(self, img):
        return {"logit16": t16, "logit8": t8, "ctx16": bad16, "ctx8": bad8}


val = D.validate(Fixed(), [{"img": torch.zeros(B, 3, 1, 1), "p16": t16.softmax(1), "p8": t8.softmax(1), "c16": c16, "c8": c8}], "cpu")
want = {"mse16.far": 1.0, "mse16.mid": 0.0, "mse16.near": 0.0, "mse16": 1 / 16, "mse8.far": 1.0, "mse8.mid": 0.0, "mse8.near": 0.0, "mse8": 1 / 32, "kl16": 0.0, "kl8.far": 0.0}
check(val["n"] == B and all(abs(val[k] - v) < 1e-5 for k, v in want.items()), "validate: an error in one far-band row of each head shows in that head's far band and its overall mean, nowhere else")
p16u, p8u = t16.softmax(1), t8.softmax(1)
batch = {"img": torch.zeros(B, 3, 1, 1), "p16": p16u, "p8": p8u, "c16": c16, "c8": c8}
uniform = {"p16": torch.full((118, 16, 44), 1 / 118), "c16": torch.zeros(80, 16, 44), "p8": torch.full((118, 32, 88), 1 / 118), "c8": torch.zeros(80, 32, 88)}


class Teacher(nn.Module):
    grid8 = torch.ones(1)

    def __init__(self, l16, l8, k16, k8):
        super().__init__()
        self.o = {"logit16": l16, "logit8": l8, "ctx16": k16, "ctx8": k8}

    def forward(self, img):
        return self.o


perfect, _ = D.score(Teacher(t16, t8, c16, c8), [batch], "cpu", uniform)
kl_uniform = {k: float((p * (p.log() + np.log(118))).sum(1).mean()) for k, p in (("16", p16u), ("8", p8u))}
check(all(abs(perfect["null.kl" + k] - kl_uniform[k]) < 1e-4 and abs(perfect["kl" + k]) < 1e-6 and abs(perfect["skill" + k] - 1) < 1e-6 and perfect["null.cos" + k] == 0.0 for k in ("16", "8"))
      and all(f"{k}{n}.{b}" in perfect for k in ("null.kl", "skill") for n in ("16", "8") for b in ("near", "mid", "far")),
      f"null baseline scored on the same frames and bands: a uniform null has KL {kl_uniform['16']:.3f} / {kl_uniform['8']:.3f} (by hand), cosine 0, the exact student skill 1")
as_null, _ = D.score(Teacher(torch.zeros(B, 118, 16, 44), torch.zeros(B, 118, 32, 88), torch.zeros(B, 80, 16, 44), torch.zeros(B, 80, 32, 88)), [batch], "cpu", uniform)
check(all(abs(as_null["skill" + k]) < 1e-6 and abs(as_null["skill" + k + ".far"]) < 1e-6 for k in ("16", "8")), "a student that equals the null predictor has skill 0 (overall and in a band)")
s16 = t16.clone()
s16[:, :, 3] = 0                                                               # H16 row 3: a uniform student; the KL of that row is the teacher's KL against uniform
_, curves = D.score(Teacher(s16, t8, c16, c8), [batch], "cpu")
cell = (p16u * (p16u.log() + np.log(118))).sum(1)                              # (B, 16, 44)
by_row, by_col = curves["by_row"]["kl16"], curves["by_col"]["kl16"]
check(len(by_row) == 16 and len(by_col) == 44 and len(curves["by_row"]["kl8"]) == 32 and len(curves["by_col"]["kl8"]) == 88 and abs(by_row[3] - float(cell[:, 3].mean())) < 1e-4
      and max(abs(x) for i, x in enumerate(by_row) if i != 3) < 1e-5 and max(abs(by_col[j] - float(cell[:, 3, j].mean()) / 16) for j in range(44)) < 1e-4,
      "KL per output row (mean over columns) and per output column (mean over rows) against a planted error in row 3: only that row, spread over the columns")


class FakeStore:
    """a feature store of three frames: probability scale (1 = normalised) and context value per token; the probability sits in bin 0"""
    spec = {"t0": (1.0, 1.0), "t1": (0.5, 3.0), "t2": (1.0, 100.0)}               # t1 is not normalised; t2 is outside the first two frames

    def __init__(self, hw):
        self.hw = hw

    def has(self, token):
        return token in self.spec

    def get(self, token):
        scale, ctx = self.spec[token]
        p = torch.zeros(118, *self.hw, dtype=torch.float16)
        p[0] = scale
        return p, torch.full((80, *self.hw), ctx, dtype=torch.float16)


nb = D.null_baseline(D.KeyFrames(["t0", "t1", "t2"], "unused", FakeStore((16, 44)), FakeStore((32, 88)), images=False), 2)
check(torch.allclose(nb["c16"], torch.full((80, 16, 44), 2.0)) and abs(float(nb["p16"][0, 0, 0]) - 1 / (1 + 117 * D.NULL_FLOOR)) < 1e-6 and abs(float(nb["p16"][5, 3, 3]) - D.NULL_FLOOR / (1 + 117 * D.NULL_FLOOR)) < 1e-9
      and torch.allclose(nb["p16"].sum(0), torch.ones(16, 44)), "null baseline: the mean over the FIRST n frames (the third is left out), each frame's probabilities renormalised first, a floor per bin, sums to 1")


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


clk = Clock()
tp, rates = D.Throughput(4, window=3, skip=2, clock=clk), []
for dur, wait in ((10, 0), (1, 0.5), (1, 0.5), (1, 0.5), (2, 1.0), (2, 1.0)):
    tp.begin_step(len(rates) + 1)
    with tp.data_wait():
        clk.t += wait
    clk.t += dur - wait
    rates.append(tp.end_step())
check(abs(rates[3]["samples_per_s"] - 4.0) < 1e-9 and abs(rates[5]["samples_per_s"] - 2.4) < 1e-9 and abs(rates[5]["data_wait_s"] - 2.5 / 3) < 1e-9 and abs(tp.mean_rate() - 16 / 6) < 1e-9,
      "samples/s on a fake clock: batch x steps in the window / their time (4.0, 2.4), the data wait per step, the mean rate after the 2 start-up steps (4 steps of 6 s in total: 2.667)")
check(D.check_resources("cuda", 2000, 1000) and D.check_resources("cuda", 8000, 7000) and D.check_resources("cuda", 8000, None) and not D.check_resources("cuda", 8000, 2000)
      and not D.check_resources("cpu", 8000, None) and not D.check_resources("cuda", None, 2000), "resource gate: RAM < 3 GB, GPU > 6 GB in use or unreadable refuse; cpu and an unreadable RAM do not")

# 6 data and CLI on a synthetic tree ------------------------------------------------------------------------------------------------------------------------------
del m, mf, mu, teacher, twin, ramp_model, opt, feats, tgt, out, out_f, a, b, to, loss                                  # the CLI runs below need the memory (the test stays under 2 GB)
gc.collect()
with tempfile.TemporaryDirectory() as tmp:
    rng = np.random.default_rng(0)
    root16, root8, root_img, root_splits = (os.path.join(tmp, d) for d in ("f16", "f8", "img", "splits"))
    os.makedirs(root16), os.makedirs(root8), os.makedirs(root_splits)
    scenes = {"scene-a": ["aa000001", "aa000002", "ab000003", "ab000004", "ab000005"], "scene-b": ["bb000001", "bb000002", "bb000003"], "scene-c": ["cc000001"]}
    keys = {"scene-a": ["aa000001", "aa000002", "ab000003", "ab000004"], "scene-b": ["bb000001", "bb000002"], "scene-c": []}      # the others are sweep frames: no JPEG
    for scene, toks in scenes.items():
        for root, (h, w) in ((root16, (16, 44)), (root8, (32, 88))):
            arr = np.concatenate([np.exp(rng.normal(size=(len(toks), 118, h, w))), rng.normal(size=(len(toks), 80, h, w))], 1).astype(np.float32)
            arr[:, :118] /= arr[:, :118].sum(1, keepdims=True)
            write_shard(root, scene, arr.astype(np.float16), toks)
        for t in keys[scene]:
            os.makedirs(os.path.join(root_img, t[:2]), exist_ok=True)
            Image.fromarray(np.tile(np.linspace(0, 255, 704).astype(np.uint8)[None, :, None], (256, 1, 3))).save(os.path.join(root_img, t[:2], t + ".jpg"), quality=95)
    with open(os.path.join(root_splits, "syn.json"), "w") as f:
        json.dump({"about": "test", "train": ["scene-a"], "val": ["scene-b"]}, f)
    s16, s8 = D.FeatureStore(root16), D.FeatureStore(root8)
    tr, va = D.split_tokens(s16, ["scene-a"], root_img), D.split_tokens(s16, ["scene-b"], root_img)
    check(tr == sorted(keys["scene-a"]) and va == keys["scene-b"] and D.split_tokens(s16, ["scene-c"], root_img) == [], "key frames = shard tokens of the scene that have a JPEG (sweep frames dropped)")
    try:
        D.split_tokens(s16, ["scene-z"], root_img)
        check(False, "a scene without a shard is an error")
    except ValueError:
        check(True, "a scene without a shard is an error")
    item = D.KeyFrames(tr, root_img, s16, s8)[1]
    (p16, c16), (p8, c8) = s16.get(tr[1]), s8.get(tr[1])
    check(item["img"].shape == (3, 256, 704) and item["img"].dtype == torch.uint8 and item["p16"].shape == (118, 16, 44) and item["c8"].shape == (80, 32, 88) and item["p8"].dtype == torch.float16
          and torch.equal(item["p16"], p16) and torch.equal(item["c16"], c16) and torch.equal(item["p8"], p8) and torch.equal(item["c8"], c8) and not torch.equal(p8, s8.get(tr[2])[0]),
          "one item: uint8 JPEG and the float16 teacher outputs of the same token, from the DualBEV store (H16) and the BEVFusion store (H8)")
    check("img" not in D.KeyFrames(tr, root_img, s16, s8, images=False)[1], "images=False reads no JPEG (the null baseline)")
    cfg3 = D.DistillConfig(name="x", split="syn", batch=3, workers=0, img_root=root_img, feats16=root16, feats8=root8, splits_dir=root_splits)
    sets3, train_l, val_l = D.build_data(cfg3)
    check(len(train_l) == 1 and len(val_l) == 1 and len(next(iter(val_l))["p16"]) == 2 and isinstance(train_l.sampler, torch.utils.data.RandomSampler)
          and isinstance(val_l.sampler, torch.utils.data.SequentialSampler), "loaders: 4 training frames at batch 3 drop the incomplete batch and shuffle, the 2 validation frames stay (one batch of 2) in order")
    try:
        D.build_data(D.DistillConfig(name="x", split="syn", batch=5, workers=0, img_root=root_img, feats16=root16, feats8=root8, splits_dir=root_splits))
        check(False, "a batch larger than the training set is refused")
    except SystemExit:
        check(True, "a batch larger than the training set is refused")
    s8_short = D.FeatureStore(root8)                                           # a token only in the DualBEV cache
    del s8_short.index["aa000001"]
    try:
        D.KeyFrames(tr, root_img, s16, s8_short)
        check(False, "a token missing from the BEVFusion cache is an error")
    except ValueError:
        check(True, "a token missing from the BEVFusion cache is an error")

    base = ["--name", "smoke", "--split", "syn", "--freeze-backbone", "--max-steps", "4", "--log-every", "2", "--batch", "2", "--workers", "0", "--device", "cpu", "--force", "--out-root", os.path.join(tmp, "runs"),
            "--img-root", root_img, "--feats16", root16, "--feats8", root8, "--splits-dir", root_splits]
    calls = []

    def stop_at_2(out_dir):
        calls.append(1)
        return "test stop" if len(calls) == 3 else None

    run_dir = os.path.join(tmp, "runs", "smoke")
    try:
        D.main(base, stop_check=stop_at_2)
        code = None
    except SystemExit as e:
        code = e.code
    check(code == 75 and os.path.exists(os.path.join(run_dir, "STOPPED")) and os.path.exists(os.path.join(run_dir, "ckpt_00000002.pt")) and not os.path.exists(os.path.join(run_dir, "metrics.json")),
          "a stop request at step 2: checkpoint, STOPPED, exit 75, no metrics")
    check(D.main(base) == 2, "the run folder has a checkpoint and --resume is missing: exit 2")
    seen_w, real_score = [], D.score

    def spy_score(model, loader, device, null=None):
        seen_w.append(model.head16.out.weight.detach().clone())              # the weights this validation runs with
        return real_score(model, loader, device, null)
    D.score = spy_score
    code = D.main(base + ["--resume"])
    D.score = real_score
    check(code == 0, "--resume goes on to the end")
    ck = torch.load(os.path.join(run_dir, "ckpt_00000004.pt"), map_location="cpu", weights_only=False)
    met = json.load(open(os.path.join(run_dir, "metrics.json")))
    check(met["weights"] == "ema" and len(seen_w) == 1 and torch.equal(seen_w[0], ck["ema"]["head16.out.weight"]) and not torch.equal(seen_w[0], ck["state"]["model.head16.out.weight"]),
          "metrics.json says weights 'ema' and the final validation really ran with the checkpoint's EMA tensors, not the live ones (which differ)")
    rows = [json.loads(line) for line in open(os.path.join(run_dir, "log.jsonl"))]
    train_rows = [r for r in rows if "total" in r]
    check(met["resumed"] and met["steps"] == 4 and met["val"]["n"] == 2 and all(f"{n}{k}.{b}" in met["val"] for n in ("kl", "cos", "mse") for k in ("16", "8") for b in ("near", "mid", "far"))
          and np.isfinite(met["val"]["total"]) and met["freeze_backbone"] and met["trained_params_m"] < 1.0, f"metrics.json: resumed, 4 steps, val on 2 frames, every head / band ({met['trained_params_m']:.2f} M trained)")
    check(met["null_frames"] == 4 and all(f"{n}.{k}{h}{b}" in met["val"] for n in ("null",) for k in ("kl", "cos", "mse") for h in ("16", "8") for b in ("", ".near", ".far")) and "skill16.mid" in met["val"]
          and "skill8" in met["val"] and [len(met["val_by_row"]["kl" + h]) for h in ("16", "8")] == [16, 32] and [len(met["val_by_col"]["kl" + h]) for h in ("16", "8")] == [44, 88],
          "metrics.json: the null baseline (4 training frames, all there are) as null.*, the skill per head and band, the KL per output row and column")
    check(train_rows and train_rows[-1]["step"] == 4 and all(k in train_rows[-1] for k in ("kl16", "kl8", "cosl16", "cosl8", "mse16", "mse8", "total", "samples_per_s", "data_wait_s", "lr")),
          "log.jsonl: a row with every loss term, samples/s and the data wait; the steps continue after the resume")
    check(os.path.exists(os.path.join(run_dir, "STOPPED.old")) and not os.path.exists(os.path.join(run_dir, "STOPPED")), "the stop file of the earlier stop was moved aside at the restart")
    all_w = ["--name", "smoke_all", "--split", "syn", "--max-steps", "2", "--log-every", "1", "--backbone-mult", "0.1", "--batch", "1"] + base[base.index("--workers"):]      # stage A2: nothing frozen
    check(D.main(all_w) == 0, "all weights trainable: a short run goes through")
    rows_bb = [json.loads(line) for line in open(os.path.join(tmp, "runs", "smoke_all", "log.jsonl")) if '"lr.backbone"' in line]
    cfg2 = json.load(open(os.path.join(tmp, "runs", "smoke_all", "config.json")))
    check(bool(rows_bb) and abs(rows_bb[-1]["lr.backbone"] / rows_bb[-1]["lr"] - 0.1) < 1e-9 and cfg2["lr_groups"] == {"backbone": {"lr": 1e-4}} and not cfg2["freeze_backbone"],
          "all weights: the backbone's learning rate is 0.1 x the heads' in the log and the config")
    for tag, every in (("ee_end", "4"), ("ee_mid", "2")):                       # --eval-every: equal to the steps = the validation at the end only; 2 of 4 = steps 2 and 4
        D.main(["--name", tag, "--eval-every", every] + base[base.index("--split"):])
        val_steps = [json.loads(line)["step"] for line in open(os.path.join(tmp, "runs", tag, "log.jsonl")) if '"val"' in line]
        check(val_steps == ([] if every == "4" else [2, 4]), f"--eval-every {every} with 4 steps validates at steps {val_steps}")
    ev_args = [x for x in base if x not in ("--freeze-backbone",)] + ["--eval-only", "--threads", "1"]
    before = open(os.path.join(run_dir, "metrics.json"), "rb").read()
    check(D.main(ev_args) == 0 and torch.get_num_threads() == 1, "--eval-only goes through with the threads it was given (--threads 1)")
    torch.set_num_threads(2)
    ev = json.load(open(os.path.join(run_dir, "eval_metrics.json")))
    check(open(os.path.join(run_dir, "metrics.json"), "rb").read() == before and ev["checkpoint"] == "ckpt_00000004.pt" and ev["step"] == 4 and ev["weights"] == "ema", "--eval-only: metrics.json untouched, the newest checkpoint")
    same = [k for k in met["val"] if k != "n"]
    check(all(abs(ev["val"][k] - met["val"][k]) < 1e-5 or (np.isnan(ev["val"][k]) and np.isnan(met["val"][k])) for k in same) and ev["val"]["n"] == 2 and np.allclose(ev["val_by_row"]["kl16"] + ev["val_by_row"]["kl8"], met["val_by_row"]["kl16"] + met["val_by_row"]["kl8"], atol=1e-4)
          and np.allclose(ev["val_by_col"]["kl16"] + ev["val_by_col"]["kl8"], met["val_by_col"]["kl16"] + met["val_by_col"]["kl8"], atol=1e-4), f"--eval-only reproduces the run's own validation (EMA weights, same frames, same null) in all {len(same)} numbers and both curves")
    check(D.main([x if x != "smoke" else "nothing" for x in ev_args]) == 2 and not os.path.exists(os.path.join(tmp, "runs", "nothing", "eval_metrics.json")), "--eval-only on a run without a checkpoint: exit 2")
    real_av, torch.cuda.is_available = torch.cuda.is_available, (lambda: False)
    code = D.main([x if x != "cpu" else "cuda" for x in base] + ["--name", "nocuda"])
    torch.cuda.is_available = real_av
    check(code == 3 and not os.path.exists(os.path.join(tmp, "runs", "nocuda")), "cuda asked for and not there: exit 3, no silent CPU run (--force does not override it)")
    D.mem_available_mb, real_mem = (lambda: 100), D.mem_available_mb
    code = D.main([a for a in base if a != "--force"] + ["--name", "gate"])
    D.mem_available_mb = real_mem
    check(code == 3 and not os.path.exists(os.path.join(tmp, "runs", "gate")), "low free RAM without --force: exit 3 before anything is written")

print("FAILED: " + "; ".join(fails) if fails else "student encoder + distillation tests: all checks passed")
sys.exit(1 if fails else 0)
