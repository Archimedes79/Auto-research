"""Gate of S5b (plan 8.3): object boxes as a function of time, on the synthetic world.

Three ways to get the boxes at t + dt (dt in -1, -0.5, +0.5, +1 s; GT from scene.step(dt): exact ego motion and
constant object velocity, boxes in the ego frame of t + dt):
  oracle   the GT boxes at t moved with the splat's position function and their true velocity (tests the maths)
  A        S5a detections at t (ObjectHead, trained at dt = 0) moved with the position function and their PREDICTED
           velocity (propagate_boxes)
  ego      the same detections moved by the ego motion only (velocity 0): the reference without object motion
  B        the head as a function of time (TimeObjectHead, FiLM on dt), trained with dt in {-1, -0.5, 0, 0.5, 1} against
           the GT boxes at t + dt, predicting them directly
Both networks see the same input at t (vehicle / pedestrian coverage, lane lines, all objects 0.5 s earlier), use the
same trunk and the same budget. Measured on held-out scenes per dt: AP per class at 1 m centre distance, centre error
and BEV IoU of the matches (tests/test_object_head.py evaluate()).
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python -u tests/test_box_time.py
"""
import math
import random
import sys
import time

import torch

sys.path.insert(0, "training")
sys.path.insert(0, "tests")
from bev_trunk import BEVTrunk  # noqa: E402
from object_head import ObjectHead, TimeObjectHead, build_targets, decode, head_loss, propagate_boxes  # noqa: E402
from synthetic_world import random_scene  # noqa: E402
from test_object_head import DEV, G, evaluate, scene_boxes, scene_inputs  # noqa: E402

DTS = (-1.0, -0.5, 0.5, 1.0)
TRAIN_DTS = (-1.0, -0.5, 0.0, 0.5, 1.0)
N_TRAIN, N_TEST, ITERS, BS = 300, 30, 900, 8


def make(time_head):
    trunk = BEVTrunk(cin=4, width=0.5)
    c = trunk.img_bev_encoder_neck.out_channels
    return trunk.to(DEV), (TimeObjectHead(c) if time_head else ObjectHead(c)).to(DEV)


def train(trunk, head, data, time_head):
    params = list(trunk.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=ITERS, pct_start=0.1)
    t0 = time.time()
    for it in range(ITERS):
        batch = [data[random.randrange(len(data))] for _ in range(BS)]
        x = torch.stack([b["x"] for b in batch]).to(DEV)
        if time_head:
            dts = [random.choice(TRAIN_DTS) for _ in batch]
            out = head(trunk.features(x), torch.tensor(dts))
            tgt = build_targets([b["boxes"][dt] for b, dt in zip(batch, dts)], G)
        else:
            out = head(trunk.features(x))
            tgt = build_targets([b["boxes"][0.0] for b in batch], G)
        loss = head_loss(out, tgt)
        opt.zero_grad()
        loss["total"].backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        sched.step()
        if it % (ITERS // 4) == 0 or it == ITERS - 1:
            print(f"  iter {it:4d} loss {loss['total'].item():.3f} {time.time() - t0:.0f} s", flush=True)


@torch.no_grad()
def detect(trunk, head, data, dt=None):
    trunk.eval()
    head.eval()
    res = []
    for i in range(0, len(data), 8):
        x = torch.stack([d["x"] for d in data[i:i + 8]]).to(DEV)
        f = trunk.features(x)
        out = head(f, torch.full((x.shape[0],), dt)) if dt is not None else head(f)
        res.extend(decode({k: v.cpu() for k, v in out.items()}, G))
    return res


def line(name, r):
    ap = [r.get(f"AP {c} @1.0", float("nan")) for c in ("car", "truck", "pedestrian")]
    mc = r.get("match car", {})
    return (f"{name:9s} | AP@1 car {ap[0]:.2f} truck {ap[1]:.2f} ped {ap[2]:.2f} | car centre "
            f"{mc.get('centre', float('nan')):.2f} m, IoU {mc.get('IoU', float('nan')):.2f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    print(f"device {DEV}", flush=True)
    t0 = time.time()

    def sample(scene):
        x, b0 = scene_inputs(scene, ss=2)
        boxes = {0.0: b0}
        for dt in TRAIN_DTS:
            if dt != 0.0:
                boxes[dt] = scene_boxes(scene.step(dt))
        return {"x": x, "boxes": boxes, "steps": {dt: [(dt, scene.v, scene.omega)] for dt in TRAIN_DTS}}
    import os
    cache = f"temp/cache/box_time_{N_TRAIN}_{N_TEST}.pt"          # rendered scenes (not in git), reused by reruns
    if os.path.exists(cache):
        train_set, test_set = torch.load(cache, weights_only=False)
        print(f"loaded {N_TRAIN} + {N_TEST} rendered scenes from {cache}", flush=True)
    else:
        rng = random.Random(2)
        train_set = [sample(random_scene(rng)) for _ in range(N_TRAIN)]
        rng = random.Random(999)
        test_set = [sample(random_scene(rng)) for _ in range(N_TEST)]
        os.makedirs("temp/cache", exist_ok=True)
        torch.save((train_set, test_set), cache)
        print(f"rendered {N_TRAIN} + {N_TEST} scenes in {time.time() - t0:.0f} s, cached in {cache}", flush=True)

    print("oracle: GT boxes at t moved with the position function and their true velocity", flush=True)
    for dt in DTS:
        moved = [propagate_boxes(d["boxes"][0.0], d["steps"][dt]) for d in test_set]
        errs = [math.hypot(m["cx"] - g["cx"], m["cy"] - g["cy"]) for ms, d in zip(moved, test_set)
                for m, g in zip(ms, d["boxes"][dt])]
        yerr = [abs(math.remainder(m["yaw"] - g["yaw"], 2 * math.pi)) for ms, d in zip(moved, test_set)
                for m, g in zip(ms, d["boxes"][dt]) if "yaw" in g]
        print(f"  dt {dt:+.1f} s: centre error max {max(errs):.4f} m, yaw error max {math.degrees(max(yerr)):.4f} deg", flush=True)

    print("A: S5a head (dt = 0) + propagation", flush=True)
    trunk, head = make(False)
    train(trunk, head, train_set, False)
    det0 = detect(trunk, head, test_set)
    print(line("dt +0.0", evaluate(det0, [d["boxes"][0.0] for d in test_set])), flush=True)
    resA, resE = {}, {}
    for dt in DTS:
        gts = [d["boxes"][dt] for d in test_set]
        prop = [propagate_boxes(ds, d["steps"][dt]) for ds, d in zip(det0, test_set)]
        ego = [propagate_boxes([dict(b, vx=0.0, vy=0.0) for b in ds], d["steps"][dt]) for ds, d in zip(det0, test_set)]
        resA[dt], resE[dt] = evaluate(prop, gts), evaluate(ego, gts)

    print("B: head as a function of time (FiLM on dt)", flush=True)
    trunk, head = make(True)
    train(trunk, head, train_set, True)
    print(line("dt +0.0", evaluate(detect(trunk, head, test_set, 0.0), [d["boxes"][0.0] for d in test_set])), flush=True)
    resB = {dt: evaluate(detect(trunk, head, test_set, dt), [d["boxes"][dt] for d in test_set]) for dt in DTS}

    print("\nheld-out scenes, boxes at t + dt")
    for dt in DTS:
        print(f"dt {dt:+.1f} s")
        print("  " + line("A prop", resA[dt]))
        print("  " + line("ego only", resE[dt]))
        print("  " + line("B time", resB[dt]))
