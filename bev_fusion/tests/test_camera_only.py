"""Gate of the camera-only decoder (training/camera_decoder.py): semantic map + all objects from one camera alone.

Synthetic world (tests/synthetic_world.py): per sample a random scene at the output time and the same scene 0.5 s
earlier (Scene.step(-0.5)); both are rendered by the ray-cast front camera (128 x 352, f = 280, 1.5 m high) with exact
depth and class per pixel. Depth target per feature cell (stride 8): the histogram of all its pixel depths over the
118 bins (tests/test_oracle_lift.depth_hist, the oracle gate's fix); cells that are all sky are ignored.
GT at the output time (synthetic_world.gt_bev): lane lines, vehicles, pedestrians as coverage; boxes / circles of ALL
objects with velocity. Losses and metrics only on the ground the camera sees (cell centres at z = 0 that project into
the image); objects: those with at least one pixel in the current image ('seen'), plus a recall over all objects whose
centre lies in the camera's ground field of view.
Metrics on held-out scenes: semantic IoU at 0.5 per class and range band; objects AP at 0.5 / 1 / 2 m and the match
errors (tests/test_object_head.evaluate); depth: mean absolute error of the expected depth on non-sky cells.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_camera_only.py [--quick]
"""
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "training")
sys.path.insert(0, "tests")
from camera_decoder import DEPTH_CENTRES, SEM_CLASSES, Z_LEVELS, CameraOnlyDecoder  # noqa: E402

SEM_THR = (0.05, 0.5, 0.25)             # coverage threshold per class of SEM_CLASSES (a 0.6 m pedestrian often
                                        # covers four 0.5 m cells by about 0.36 each: 0.5 left some without a cell)
from lift_pd import Grid  # noqa: E402
from object_head import GROUPS, build_targets, decode, head_loss  # noqa: E402
from synthetic_world import camera, front_camera, gt_bev, random_scene  # noqa: E402
from test_object_head import evaluate, show  # noqa: E402
from test_oracle_lift import depth_hist  # noqa: E402

G = Grid(z_levels=Z_LEVELS)
DEV = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 3e9 else "cpu"
QUICK = "--quick" in sys.argv
K64, T64, HW = front_camera(h=128, w=352, f=280.0)
K, T = K64.float(), T64.float()
AGE, STEPS_PER = 0.5, 6
xs, ys = G.xy_centres()
GX, GY = torch.meshgrid(xs, ys, indexing="ij")


def ground_fov():
    p = torch.stack([GX, GY, torch.zeros_like(GX), torch.ones_like(GX)], -1).double() @ T64.T
    z = p[..., 2]
    u = K64[0, 0] * p[..., 0] / z.clamp(min=1e-6) + K64[0, 2]
    v = K64[1, 1] * p[..., 1] / z.clamp(min=1e-6) + K64[1, 2]
    return (z > 0.5) & (u >= 0) & (u <= HW[1] - 1) & (v >= 0) & (v <= HW[0] - 1)


MASK = ground_fov()


def boxes_of(scene, cls_map):
    out = []
    for k, b in enumerate(scene.boxes):
        d = {"group": b.cls, "cx": b.cx, "cy": b.cy, "vx": b.vx, "vy": b.vy, "seen": bool((cls_map == 3 + k).any())}
        if b.cls == "pedestrian":
            d["radius"] = max(b.length, b.width) / 2
        else:
            d.update(length=b.length, width=b.width, yaw=b.heading)
        out.append(d)
    return out


def sample(rng):
    scene = random_scene(rng)
    imgs, dts = [], []
    cls_now = None
    for age in (0.0, AGE):
        sc = scene.step(-age) if age > 0 else scene
        img, depth, cls = camera(sc, K64, T64, HW)
        imgs.append((img.clamp(0, 1) * 255).to(torch.uint8))
        dts.append(depth_hist(depth, 8).half())
        if age == 0:
            cls_now = cls
    g = gt_bev(scene, G)
    sem = torch.stack([g["lanes"], g["vehicles"], g["pedestrians"]])
    # per-class threshold of the coverage: lane lines are 0.15 m wide in 0.5 m cells (coverage <= 0.3), so a lane
    # cell is any covered cell; vehicles / pedestrians from half coverage (a 0.5 threshold for all made the lane
    # target empty: lane IoU 0 by construction in the first quick run, 2026-10-26)
    thr = torch.tensor(SEM_THR).view(-1, 1, 1)
    return torch.stack(imgs), torch.stack(dts), (sem > thr).float(), boxes_of(scene, cls_now), (scene.v, scene.omega)


def steps_for(v, w):
    return [[], [(AGE / STEPS_PER, v, w)] * STEPS_PER]


def run_model(model, items):
    """one rig, but the ego motion differs per sample: run per sample (the window's steps are per sample)."""
    outs = []
    for img, _, _, _, (v, w) in items:
        x = (img.float() / 255.0)[None].to(DEV)
        outs.append(model(x, K.to(DEV), T.to(DEV), None, steps_for(v, w)))
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def losses(model, items):
    out = run_model(model, items)
    m = MASK.to(DEV).float()
    sem_t = torch.stack([s for _, _, s, _, _ in items]).to(DEV)
    sem = (F.binary_cross_entropy_with_logits(out["sem"], sem_t, pos_weight=torch.tensor(3.0, device=DEV), reduction="none") * m).sum() / (m.sum() * len(items) * len(SEM_CLASSES))
    dt = torch.stack([d for _, d, _, _, _ in items]).to(DEV).float()                    # (B, F, D, h, w)
    valid = (dt.sum(2) > 0.5).float()
    logp = out["depth_logits"].log_softmax(2)
    depth = -(dt * logp).sum(2).mul(valid).sum() / valid.sum().clamp(min=1)
    obj = head_loss({k: out[k] for k in ("hm", "off", "size", "rot", "rad", "vel")},
                    build_targets([[b for b in bx if b["seen"]] for _, _, _, bx, _ in items], G))
    return sem, depth, obj["total"]


def train(model, data, iters, bs, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=iters, pct_start=0.1)
    model.train()
    t0 = time.time()
    for it in range(iters):
        s, d, o = losses(model, [data[random.randrange(len(data))] for _ in range(bs)])
        loss = s + d + o
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()
        if it % max(1, iters // 5) == 0 or it == iters - 1:
            print(f"  iter {it:5d} sem {s.item():.3f} depth {d.item():.3f} objects {o.item():.3f} ({time.time() - t0:.0f} s)", flush=True)


def in_fov(b):
    ix, iy = int((b["cx"] - G.x_min) / G.res), int((b["cy"] - G.y_min) / G.res)
    return 0 <= ix < G.nx and 0 <= iy < G.ny and bool(MASK[ix, iy])


@torch.no_grad()
def run_eval(model, data, title):
    model.eval()
    inter = torch.zeros(len(SEM_CLASSES), 3)
    union = torch.zeros(len(SEM_CLASSES), 3)
    bands = ((0, 20), (20, 40), (40, 60))
    dets, dep_err, dep_n = [], 0.0, 0
    for i in range(0, len(data), 4):
        items = data[i:i + 4]
        out = run_model(model, items)
        pred = (torch.sigmoid(out["sem"]) > 0.5).float().cpu()
        tgt = torch.stack([s for _, _, s, _, _ in items])
        for bi, (lo, hi) in enumerate(bands):
            mb = (MASK & (GX >= lo) & (GX < hi)).float()
            inter[:, bi] += (pred * tgt * mb).sum((0, 2, 3))
            union[:, bi] += (((pred + tgt) > 0).float() * mb).sum((0, 2, 3))
        dt = torch.stack([d[0] for _, d, _, _, _ in items]).float()
        p = out["depth_logits"][:, 0].softmax(1).cpu()
        valid = dt.sum(1) > 0.5
        e = ((p * DEPTH_CENTRES.view(1, -1, 1, 1)).sum(1) - (dt * DEPTH_CENTRES.view(1, -1, 1, 1)).sum(1)).abs()
        dep_err += e[valid].sum().item()
        dep_n += int(valid.sum())
        dets.extend(decode({k: out[k].cpu() for k in ("hm", "off", "size", "rot", "rad", "vel")}, G, threshold=0.2))
    print(title)
    for c, name in enumerate(SEM_CLASSES):
        tot = inter[c].sum() / union[c].sum().clamp(min=1)
        print(f"  {name:12s} IoU {tot:.3f}; per range band " +
              ", ".join(f"{lo}-{hi} m {inter[c, b] / max(1.0, union[c, b].item()):.3f}" for b, (lo, hi) in enumerate(bands)))
    print(f"  depth: mean absolute error of the expected depth {dep_err / max(1, dep_n):.2f} m on non-sky cells")
    seen_gt = [[b for b in bx if b["seen"] and in_fov(b)] for _, _, _, bx, _ in data]
    show(evaluate([[d for d in ds if in_fov(d)] for ds in dets], seen_gt), "  objects seen by the camera")


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    print(f"device {DEV}")
    t0 = time.time()
    n_train, n_test = (150, 16) if QUICK else (1000, 48)
    cache = f"temp/checks/camera_only_data_{n_train}_{n_test}_thr{'_'.join(map(str, SEM_THR))}.pt"   # scratch, not in git
    if os.path.exists(cache):
        train_set, test_set = torch.load(cache)
    else:
        rng = random.Random(11)
        train_set = [sample(rng) for _ in range(n_train)]
        rng = random.Random(5150)
        test_set = [sample(rng) for _ in range(n_test)]
        torch.save((train_set, test_set), cache)
    print(f"data: {n_train} training / {n_test} held-out scenes, {time.time() - t0:.0f} s")
    model = CameraOnlyDecoder(grid=G).to(DEV)
    print("(1) overfit one scene")
    train(model, train_set[:1], 150 if QUICK else 400, 1)
    run_eval(model, train_set[:1], "    that scene")
    model = CameraOnlyDecoder(grid=G).to(DEV)
    print(f"(2) train on {n_train} scenes")
    train(model, train_set, 300 if QUICK else 3000, 4)
    run_eval(model, test_set, "    held-out scenes")
    run_eval(model, train_set[:n_test], "    training scenes (same count)")
