"""Three changes that cut host (CPU) work of a training step without changing a number; no training run, CPU, a few seconds, toy data. Each is compared with the code it replaced,
which is kept in this file as the reference:

  1. object_head.build_targets computes the Gaussian peak only inside its |dx|, |dy| <= r window (it is 0 outside): the targets are bit-identical to the whole-grid version,
     also for boxes at the edge of the grid, boxes outside it, very large and very small radii and the max_obj cap
  2. losses.object_head_terms reads the weight map to the host ONCE per call (it read one element per box: one GPU synchronisation per box, about 270 per step): the
     keep / drop split, the targets and every loss term are identical to the per-box version, and the weight map is read by one slice and no element read
  3. train.EMA.update averages with two multi-tensor calls instead of two kernels per state tensor: the same shadow weights (on the CPU bit for bit; on the GPU the last bit of
     a fused multiply-add may differ, the average is not on a gradient path), the frozen keys stay as they are, the integer buffers are left alone
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_host_overhead.py
"""
import math
import random
import sys

import torch
from torch import nn

sys.path.insert(0, "training")
import losses as L  # noqa: E402
import object_head as oh  # noqa: E402
import train as T  # noqa: E402
from lift_pd import Grid  # noqa: E402

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


# ---------------------------------------------------------------- 1 build_targets
def build_targets_whole_grid(boxes_batch, grid, groups=oh.GROUPS, max_obj=64, min_radius=2):
    """the version before the window (the whole grid per object); the reference"""
    B, G, X, Y = len(boxes_batch), len(groups), grid.nx, grid.ny
    gidx = {name: i for i, (name, _) in enumerate(groups)}
    t = {"hm": torch.zeros(B, G, X, Y), "ind": torch.zeros(B, max_obj, dtype=torch.long),
         "mask": torch.zeros(B, max_obj), "group": torch.zeros(B, max_obj, dtype=torch.long),
         "off": torch.zeros(B, max_obj, 2), "size": torch.zeros(B, max_obj, 2), "rot": torch.zeros(B, max_obj, 2),
         "rad": torch.zeros(B, max_obj, 1), "vel": torch.zeros(B, max_obj, 2), "is_box": torch.zeros(B, max_obj),
         "dir": torch.zeros(B, max_obj, 1)}
    gx = torch.arange(X, dtype=torch.float32).view(X, 1)
    gy = torch.arange(Y, dtype=torch.float32).view(1, Y)
    for b, boxes in enumerate(boxes_batch):
        n = 0
        for bx in boxes:
            fx = (bx["cx"] - grid.x_min) / grid.res - 0.5
            fy = (bx["cy"] - grid.y_min) / grid.res - 0.5
            ix, iy = int(math.floor(fx + 0.5)), int(math.floor(fy + 0.5))
            if not (0 <= ix < X and 0 <= iy < Y) or n >= max_obj:
                continue
            g = gidx[bx["group"]]
            kind = groups[g][1]
            if kind == "box":
                r = oh.gaussian_radius(bx["length"] / grid.res, bx["width"] / grid.res)
            else:
                d = 2 * bx["radius"] / grid.res
                r = oh.gaussian_radius(d, d)
            r = max(min_radius, int(r))
            sigma = (2 * r + 1) / 6
            peak = torch.exp(-((gx - ix) ** 2 + (gy - iy) ** 2) / (2 * sigma * sigma))
            peak = peak * (((gx - ix).abs() <= r) & ((gy - iy).abs() <= r))
            t["hm"][b, g] = torch.maximum(t["hm"][b, g], peak)
            t["ind"][b, n], t["mask"][b, n], t["group"][b, n] = ix * Y + iy, 1.0, g
            t["off"][b, n] = torch.tensor([fx - ix, fy - iy])
            t["vel"][b, n] = torch.tensor([bx["vx"], bx["vy"]])
            if kind == "box":
                t["size"][b, n] = torch.tensor([math.log(bx["length"]), math.log(bx["width"])])
                t["rot"][b, n] = torch.tensor([math.sin(bx["yaw"]), math.cos(bx["yaw"])])
                t["dir"][b, n] = oh.direction_class(bx["yaw"])
                t["is_box"][b, n] = 1.0
            else:
                t["rad"][b, n] = math.log(bx["radius"])
            n += 1
    return t


grid = Grid()
rng = random.Random(1)


def random_box(near_edge=False):
    grp, kind = rng.choice(L.NUSC_GROUPS)
    cx = rng.choice([grid.x_min - 0.2, grid.x_min + 0.1, grid.x_max - 0.1, grid.x_max + 0.3]) if near_edge else rng.uniform(grid.x_min - 3, grid.x_max + 3)
    cy = rng.choice([grid.y_min + 0.1, grid.y_max - 0.1, grid.y_max + 0.2]) if near_edge else rng.uniform(grid.y_min - 3, grid.y_max + 3)
    bx = {"group": grp, "cx": cx, "cy": cy, "vx": rng.uniform(-9, 9), "vy": rng.uniform(-9, 9)}
    if kind == "box":
        bx.update(length=rng.choice([0.4, 4.5, 12.0, 60.0]) * rng.uniform(0.9, 1.1), width=rng.choice([0.3, 1.9, 3.0, 40.0]) * rng.uniform(0.9, 1.1), yaw=rng.uniform(-3.1, 3.1))
    else:
        bx["radius"] = rng.choice([0.05, 0.4, 0.9, 25.0])
    return bx


same = True
n_boxes = n_edge = 0
for trial in range(60):
    batch = [[random_box(near_edge=rng.random() < 0.3) for _ in range(rng.choice([0, 1, 6, 30]))] for _ in range(rng.choice([1, 3]))]
    cap = rng.choice([4, 128])
    ref = build_targets_whole_grid(batch, grid, L.NUSC_GROUPS, cap)
    new = oh.build_targets(batch, grid, L.NUSC_GROUPS, cap)
    same &= all(torch.equal(ref[k], new[k]) for k in ref) and set(ref) == set(new)
    n_boxes += sum(len(b) for b in batch)
check(same, f"build_targets: window == whole grid, bit for bit, on {n_boxes} random boxes (edges, outside the grid, radius 0.05 m ... 60 m, max_obj cap)")
one = oh.build_targets([[{"group": "vehicle", "cx": 20.0, "cy": 0.0, "vx": 0.0, "vy": 0.0, "length": 4.5, "width": 1.9, "yaw": 0.3}]], grid, L.NUSC_GROUPS, 8)
check(float(one["hm"].max()) == 1.0 and float(one["hm"][0, 0, int(20 / grid.res), int(26 / grid.res)]) > 0.99 and int((one["hm"] > 0).sum()) > 4, "a single box: peak 1 at its centre cell, a blob around it")


# ---------------------------------------------------------------- 2 object_head_terms: one read, the same result
def object_head_terms_per_box(out, boxes_batch, w_obj, grid, groups=L.NUSC_GROUPS, w_vel=0.2, max_obj=128, w_mov=1.0):
    """the version before: float(w_obj[b, 0, ix, iy]) per box; the reference"""
    X, Y = w_obj.shape[-2:]
    keep, drop = [], []
    for b, boxes in enumerate(boxes_batch):
        k, d = [], []
        for bx in boxes:
            hb = L.head_box(bx, groups)
            if hb is None:
                continue
            ix = int(math.floor((hb["cx"] - grid.x_min) / grid.res))
            iy = int(math.floor((hb["cy"] - grid.y_min) / grid.res))
            if not (0 <= ix < X and 0 <= iy < Y):
                continue
            (k if bx.get("observed", True) and float(w_obj[b, 0, ix, iy]) > 0 else d).append(hb)
        keep.append(k)
        drop.append(d)
    dev = out["hm"].device
    t = oh.build_targets(keep, grid, groups, max_obj)
    ign = (w_obj <= 0).expand_as(out["hm"])
    if any(drop):
        ign = ign | ((oh.build_targets(drop, grid, groups, max_obj)["hm"].to(dev) > 0) & (t["hm"].to(dev) == 0))
    hm = torch.where(ign, torch.full_like(out["hm"], -1e4), out["hm"])
    return oh.head_loss(dict(out, hm=hm), t, w_vel, w_mov=w_mov), t


def gt_box(group, cx, cy, observed=True):
    kind = dict(L.NUSC_GROUPS)[group]
    size = (1.9, 4.5, 1.6) if kind == "box" else (0.7, 0.7, 1.7)
    return {"group": group, "center": (cx, cy, 0.0), "size": size, "yaw": 0.4, "vel": (1.0, -2.0), "observed": observed}


X, Y = grid.nx, grid.ny
B = 3
g = torch.Generator().manual_seed(0)
out = {"hm": torch.randn(B, 3, X, Y, generator=g, requires_grad=True), "off": torch.randn(B, 2, X, Y, generator=g), "size": torch.randn(B, 2, X, Y, generator=g),
       "rot": torch.randn(B, 2, X, Y, generator=g), "rad": torch.randn(B, 1, X, Y, generator=g), "vel": torch.randn(B, 2, X, Y, generator=g)}
w_obj = torch.ones(B, 1, X, Y)
w_obj[:, :, :, :30] = 0.0                                                    # a band without weight: the boxes in it are dropped (ignored), not negatives
w_obj[1, 0, 40:60, 50:70] = 0.0                                              # a patch of sample 1 only, another of sample 2 only: a box read from the wrong sample is wrong
w_obj[2, 0, 10:25, 80:100] = 0.0
boxes = [[gt_box("vehicle", rng.uniform(0, 59), rng.uniform(-25, 25), observed=rng.random() > 0.2) for _ in range(12)]
         + [gt_box("pedestrian", rng.uniform(0, 59), rng.uniform(-25, 25)) for _ in range(5)] + [{"group": "static", "center": (5, 5, 0), "size": (1, 1, 1), "yaw": 0.0, "vel": (0, 0)}]
         for _ in range(B)]
cell = lambda ix, iy: (grid.x_min + (ix + 0.5) * grid.res, grid.y_min + (iy + 0.5) * grid.res)       # noqa: E731 - the centre of a cell
boxes[1] += [gt_box("vehicle", *cell(45 + i, 55 + i)) for i in range(5)]                     # inside the zero patch of sample 1 (weight 1 in samples 0 and 2): dropped there
boxes[2] += [gt_box("vehicle", *cell(12 + 2 * i, 85 + 2 * i)) for i in range(5)]             # inside the zero patch of sample 2


class Spy(torch.Tensor):
    """counts the calls on the weight map: everything that reads it to the host is a __getitem__, .cpu() / .item() / float()"""
    calls = []

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        Spy.calls.append(getattr(func, "__name__", str(func)))
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **(kwargs or {}))


ref_lo, ref_t = object_head_terms_per_box(out, boxes, w_obj, grid)
Spy.calls.clear()
new_lo, new_t = L.object_head_terms(out, boxes, w_obj.as_subclass(Spy), grid)
reads = [c for c in Spy.calls if c in ("__getitem__", "cpu", "item", "__float__", "tolist", "numpy", "__bool__")]
check(all(torch.equal(ref_t[k], new_t[k]) for k in ref_t), "object_head_terms: the same targets as the per-box version (kept and dropped boxes, an unobserved box, a group the head does not have)")
check(set(ref_lo) == set(new_lo) and all(torch.equal(ref_lo[k], new_lo[k]) for k in ref_lo), f"object_head_terms: every loss term identical ({', '.join(sorted(ref_lo))})")
check(int((w_obj[0, 0, :, :30] == 0).sum()) > 0 and sum(1 for b in boxes for bx in b if bx["center"][1] < grid.y_min + 30 * grid.res) > 0, "the toy data does hit the drop branch (boxes on a zero-weight band)")
check(len(reads) == 1 and reads == ["__getitem__"], f"the weight map is read by ONE slice and no element read (reads seen: {reads})")


# ---------------------------------------------------------------- 3 EMA
def ema_update_loop(ema, model, step):
    """the version before: two kernels per state tensor; the reference"""
    d = min(ema.decay, (1 + step) / (10 + step))
    for k, v in ema._tensors(model).items():
        if k not in ema.skip:
            ema.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)


torch.manual_seed(0)
net = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(), nn.Conv2d(8, 4, 3), nn.BatchNorm2d(4))
net.register_buffer("b16", torch.randn(5).bfloat16())                       # state that is not float32: a multi-tensor call would round the decay 0.99 to bf16 (0.98828125)
net.register_buffer("h16", torch.randn(5).half())
net.register_buffer("d64", torch.randn(5).double())
skip = {"3.weight", "3.bias"}                                               # a frozen layer: its average stays the value itself
a, b = T.EMA(net, 0.99, skip), T.EMA(net, 0.99, skip)
opt = torch.optim.SGD(net.parameters(), lr=0.05)
before = {k: v.clone() for k, v in a.shadow.items()}
for step in range(12):
    net.train()
    net(torch.randn(4, 3, 12, 12)).square().mean().backward()
    opt.step()
    opt.zero_grad()
    with torch.no_grad():
        net.b16.add_(torch.randn(5).bfloat16())
        net.h16.mul_(1.01)
        net.d64.add_(torch.randn(5).double())
    a.update(net, step)
    ema_update_loop(b, net, step)
check(all(torch.equal(a.shadow[k], b.shadow[k]) for k in a.shadow), "EMA.update: the shadow weights equal the per-tensor loop after 12 steps, bit for bit (CPU), batch-norm statistics, bfloat16, float16 and float64 state included")
nothing = T.EMA(net, 0.9, set(a.shadow))
snap = {k: v.clone() for k, v in nothing.shadow.items()}
nothing.update(net, 0)
check(all(torch.equal(nothing.shadow[k], snap[k]) for k in snap), "EMA.update: every key frozen (an empty list) does nothing and does not raise")
check(all(torch.equal(a.shadow[k], before[k]) for k in skip) and not torch.equal(a.shadow["0.weight"], before["0.weight"]), "EMA.update: the frozen keys keep their value, the others move")
check("1.num_batches_tracked" not in a.shadow, "EMA.update: integer buffers are not averaged")
calls = []
orig_mul, orig_add = torch._foreach_mul_, torch._foreach_add_
torch._foreach_mul_ = lambda *x, **k: (calls.append("mul"), orig_mul(*x, **k))[1]
torch._foreach_add_ = lambda *x, **k: (calls.append("add"), orig_add(*x, **k))[1]
try:
    a.update(net, 12)
finally:
    torch._foreach_mul_, torch._foreach_add_ = orig_mul, orig_add
check(calls == ["mul", "add"], f"EMA.update: two multi-tensor calls per update ({calls})")

print("HOST OVERHEAD GATE: " + ("PASS" if not fails else "FAIL (" + str(len(fails)) + ")"))
sys.exit(1 if fails else 0)
