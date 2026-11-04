"""scratch (CPU speed analysis, 2026-11-03): the t20 chain (configs/t20_base.json) on a synthetic cached batch with synthetic GT, one clip keyframe at a time.
Run from the bev_fusion folder with the gcar python. Everything is CPU, 2 threads."""
import dataclasses
import json
import math
import os
import random
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "training"), os.path.join(ROOT, "tests"), os.path.join(ROOT, "tests", "support"), os.path.join(ROOT, ".."),
                os.path.join(ROOT, "..", "autoresearch")]
os.chdir(ROOT)
torch.set_num_threads(2)

import synthetic_world as sw  # noqa: E402
import train as T  # noqa: E402
from losses import synthetic_gt  # noqa: E402
from test_slots import batch_for  # noqa: E402


def t20_config(**over):
    d = json.load(open(os.path.join(ROOT, "training", "configs", "t20_base.json")))
    d = {k: v for k, v in d.items() if not k.startswith("_") and k != "split"}
    d["model"] = dict(d["model"], pretrained="", trunk_pretrained="")
    d.update(over)
    names = {f.name for f in dataclasses.fields(T.TrainConfig)}
    return T.TrainConfig(**{k: v for k, v in d.items() if k in names}), sorted(set(d) - names)


def scene_with(n_boxes, seed=0):
    rnd = random.Random(seed)
    classes = ["car", "car", "car", "truck", "pedestrian", "car", "pedestrian"]
    boxes = []
    for i in range(n_boxes):
        c = classes[i % len(classes)]
        L, W, H = {"car": (4.5, 1.8, 1.5), "truck": (8.0, 2.5, 3.0), "pedestrian": (0.6, 0.6, 1.8)}[c]
        moving = rnd.random() < 0.4
        boxes.append(sw.Box(rnd.uniform(6, 55), rnd.uniform(-20, 20), L, W, H, rnd.uniform(-3, 3),
                            rnd.uniform(-8, 8) if moving else 0.0, rnd.uniform(-1, 1) if moving else 0.0, c, rnd.uniform(-5, 15)))
    return sw.Scene(boxes, lanes=[-5.25, -1.75, 1.75, 5.25], v=8.0, omega=0.05)


def with_gt(batch, grid, n_boxes, B):
    items = []
    for b in range(B):
        scene = scene_with(n_boxes, seed=b)
        items.append({"prev": synthetic_gt(scene.step(-0.5), grid), "cur": synthetic_gt(scene, grid), "next": synthetic_gt(scene.step(0.5), grid)})
    return dict(batch, gt=items)


def make(B=1, n_boxes=15, grid_kwargs=None, **over):
    cfg, unknown = t20_config(**over)
    if grid_kwargs:
        cfg = dataclasses.replace(cfg, model=dict(cfg.model, grid_kwargs=grid_kwargs))
    model, loss, cached = T.build(cfg)
    if grid_kwargs:
        loss = type(loss)(loss.cfg, grid=model.grid)
    assert cached
    model.train()
    torch.manual_seed(0)
    with torch.no_grad():
        for q in model.parameters():
            if q.requires_grad and q.abs().max() == 0:
                q.normal_(0, 1e-2)
    batch = batch_for(model.cfg, B=B)
    batch = with_gt(batch, model.grid, n_boxes, B)
    batch["lane_img"] = torch.randint(0, 255, (B, 3, *batch["hw"]), generator=torch.Generator().manual_seed(1), dtype=torch.uint8)
    batch["gap_steps"] = [{"prev": [(-1 / 12, 8.0, 0.05)] * 6, "next": [(1 / 12, 8.0, 0.05)] * 6} for _ in range(B)]
    return cfg, model, loss, batch, unknown
