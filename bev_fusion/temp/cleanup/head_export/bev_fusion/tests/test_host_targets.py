"""The host side of the loss, bit for bit (D43 follow-up, 2026-11-03): `object_head.build_targets` (numpy fields, one torch.exp per frame) and
`losses._stack_many` (every GT layer of all samples moved with ONE copy) against the code they replaced.

    python tests/test_host_targets.py [--frames 36] [--gt-cache DIR]

1. build_targets == the old function (a textual copy below, the oracle) for all 11 tensors with torch.equal and the same dtype: random boxes of every
   group, boxes outside the grid and on its edge, the max_obj cap, empty frames, and the boxes of real cached GT frames (skipped without data/cache/gt)
2. _stack_many == _stack per key (equal values, dtype, shape), and build_layer_targets through the new path gives exactly the tensors of the old path
3. the timings of the host work (informational; the saving of the stack is the number of host-to-device copies: 1 instead of one per layer per sample)
CPU, seconds, no model, no training step."""
import argparse
import glob
import math
import os
import statistics
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.join(ROOT, "tests", "support"))
sys.path.insert(0, os.path.dirname(ROOT))
import losses as L                                              # noqa: E402
import object_head as O                                         # noqa: E402
from lift_pd import Grid                                        # noqa: E402
from object_head import GROUPS, direction_class, gaussian_radius   # noqa: E402,F401


# ---- the oracle: build_targets as it was before 2026-11-03 (torch.tensor per field per box), copied unchanged
def build_targets_old(boxes_batch, grid, groups=GROUPS, max_obj=64, min_radius=2):
    """boxes_batch: list (B) of lists of box dicts -> dict of target tensors (hm, ind, mask, group, off, size, rot,
    rad, vel, is_box)."""
    B, G, X, Y = len(boxes_batch), len(groups), grid.nx, grid.ny
    gidx = {name: i for i, (name, _) in enumerate(groups)}
    t = {"hm": torch.zeros(B, G, X, Y), "ind": torch.zeros(B, max_obj, dtype=torch.long),
         "mask": torch.zeros(B, max_obj), "group": torch.zeros(B, max_obj, dtype=torch.long),
         "off": torch.zeros(B, max_obj, 2), "size": torch.zeros(B, max_obj, 2), "rot": torch.zeros(B, max_obj, 2),
         "rad": torch.zeros(B, max_obj, 1), "vel": torch.zeros(B, max_obj, 2), "is_box": torch.zeros(B, max_obj),
         "dir": torch.zeros(B, max_obj, 1)}
    if any("vel_known" in bx for boxes in boxes_batch for bx in boxes):    # LossConfig.hygiene_2026_11_02: 1 where the velocity of the object is known (a box dict may carry `loss_w`, its loss weight, as well)
        t["known"] = torch.ones(B, max_obj)
    for b, boxes in enumerate(boxes_batch):
        n = 0
        for bx in boxes:
            fx = (bx["cx"] - grid.x_min) / grid.res - 0.5                 # continuous cell index of the centre
            fy = (bx["cy"] - grid.y_min) / grid.res - 0.5
            ix, iy = int(math.floor(fx + 0.5)), int(math.floor(fy + 0.5))
            if not (0 <= ix < X and 0 <= iy < Y) or n >= max_obj:
                continue
            g = gidx[bx["group"]]
            kind = groups[g][1]
            if kind == "box":
                r = gaussian_radius(bx["length"] / grid.res, bx["width"] / grid.res)
            else:
                d = 2 * bx["radius"] / grid.res
                r = gaussian_radius(d, d)
            r = max(min_radius, int(r))
            sigma = (2 * r + 1) / 6
            x0, x1, y0, y1 = max(0, ix - r), min(X, ix + r + 1), max(0, iy - r), min(Y, iy + r + 1)     # the peak is 0 outside |dx|, |dy| <= r: only that window is computed (same values, 2x less CPU)
            gx = torch.arange(x0, x1, dtype=torch.float32).view(-1, 1)
            gy = torch.arange(y0, y1, dtype=torch.float32).view(1, -1)
            peak = torch.exp(-((gx - ix) ** 2 + (gy - iy) ** 2) / (2 * sigma * sigma))
            w = t["hm"][b, g, x0:x1, y0:y1]
            w.copy_(torch.maximum(w, peak))
            t["ind"][b, n], t["mask"][b, n], t["group"][b, n] = ix * Y + iy, bx.get("loss_w", 1.0), g
            if "known" in t:
                t["known"][b, n] = 1.0 if bx.get("vel_known", True) else 0.0
            t["off"][b, n] = torch.tensor([fx - ix, fy - iy])
            t["vel"][b, n] = torch.tensor([bx["vx"], bx["vy"]])
            if kind == "box":
                t["size"][b, n] = torch.tensor([math.log(bx["length"]), math.log(bx["width"])])
                t["rot"][b, n] = torch.tensor([math.sin(bx["yaw"]), math.cos(bx["yaw"])])
                t["dir"][b, n] = direction_class(bx["yaw"])
                t["is_box"][b, n] = 1.0
            else:
                t["rad"][b, n] = math.log(bx["radius"])
            n += 1
    return t


def med(fn, n=7):
    fn()
    ts = []
    for _ in range(n):
        a = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - a) * 1000)
    return statistics.median(ts)


def same(a, b, what):
    assert a.keys() == b.keys(), (what, sorted(a), sorted(b))
    for k in a:
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and torch.equal(a[k], b[k]), (what, k)


def random_frames(rng, grid, groups, n_frames=4, n_boxes=40):
    names = [g for g, _ in groups]
    kinds = dict(groups)
    frames = []
    for _ in range(n_frames):
        boxes = []
        for _ in range(n_boxes):
            g = names[rng.integers(len(names))]
            bx = {"group": g, "cx": float(rng.uniform(grid.x_min - 3, grid.x_max + 3)), "cy": float(rng.uniform(grid.y_min - 3, grid.y_max + 3)),
                  "vx": float(rng.normal() * 5), "vy": float(rng.normal() * 5)}
            if kinds[g] == "box":
                bx.update(length=float(rng.uniform(0.4, 12)), width=float(rng.uniform(0.4, 3.5)), yaw=float(rng.uniform(-math.pi, math.pi)))
            else:
                bx["radius"] = float(rng.uniform(0.2, 1.2))
            boxes.append(bx)
        frames.append(boxes)
    frames.append([])                                          # an empty frame
    frames.append([dict(frames[0][0], cx=grid.x_min - 50.0)])  # only a box outside the grid
    return frames


def raises(fn, *a):
    try:
        fn(*a)
    except Exception as e:                                     # noqa: BLE001 - the type is the result
        return type(e).__name__
    return None


def edge_cases(grid):
    """the inputs the random frames do not reach: per-box loss_w / vel_known (the hygiene flag), a key in one frame of the batch only, unknown groups
    outside the grid, boxes in one cell, numpy scalars as values, and the inputs that used to raise (the new code must still raise, the same type)"""
    rng = np.random.default_rng(1)
    G = L.NUSC_GROUPS
    base = random_frames(rng, grid, G, n_frames=3, n_boxes=30)
    mixed = [[dict(bx, **({"loss_w": float(rng.uniform(0.1, 1))} if rng.random() < 0.5 else {}),
                   **({"vel_known": bool(rng.random() < 0.5)} if rng.random() < 0.5 else {})) for bx in fr] for fr in base]
    same(build_targets_old(mixed, grid, G, 64), O.build_targets(mixed, grid, G, 64), "loss_w / vel_known mixed in the frames")
    one_frame = [base[0], [dict(bx, vel_known=False, loss_w=0.5) for bx in base[1]], base[2]]
    same(build_targets_old(one_frame, grid, G, 64), O.build_targets(one_frame, grid, G, 64), "a key in one frame of the batch only")
    same(build_targets_old(base, grid, G, 64), O.build_targets(base, grid, G, 64), "no loss_w / vel_known at all")
    assert "known" not in O.build_targets(base, grid, G, 64) and "known" in O.build_targets(one_frame, grid, G, 64)
    alien = [[{"group": "alien", "cx": grid.x_min - 40.0, "cy": 0.0, "vx": 0.0, "vy": 0.0, "radius": 0.5}] + base[0][:5]]
    same(build_targets_old(alien, grid, G, 64), O.build_targets(alien, grid, G, 64), "an unknown group outside the grid is skipped")
    twin = [[dict(base[0][0], cx=10.02, cy=1.01), dict(base[0][0], cx=10.02, cy=1.01, length=2.0, vx=3.0)]]
    same(build_targets_old(twin, grid, G, 64), O.build_targets(twin, grid, G, 64), "two boxes in one cell")
    f32 = [[{k: (np.float32(v) if isinstance(v, float) else v) for k, v in bx.items()} for bx in fr] for fr in base]
    same(build_targets_old(f32, grid, G, 64), O.build_targets(f32, grid, G, 64), "numpy float32 values")
    for name, bad in (("length 0", dict(length=0.0)), ("width -1", dict(width=-1.0)), ("NaN centre", dict(cx=float("nan")))):
        bx = dict(base[0][0], cx=10.0, cy=1.0, group="vehicle", length=4.0, width=2.0, yaw=0.1)
        bx.update(bad)
        fr = [[bx]]
        e_old, e_new = raises(build_targets_old, fr, grid, G, 64), raises(O.build_targets, fr, grid, G, 64)
        assert e_old is not None and e_new == e_old, (name, e_old, e_new)
    fr = [[{"group": "pedestrian", "cx": 10.0, "cy": 1.0, "vx": 0.0, "vy": 0.0, "radius": -0.2}]]
    assert raises(build_targets_old, fr, grid, G, 64) is not None and raises(O.build_targets, fr, grid, G, 64) == raises(build_targets_old, fr, grid, G, 64)
    print("build_targets == old also for loss_w / vel_known (mixed, one frame only, none), unknown groups outside the grid, two boxes in one cell, "
          "numpy float32 values; length 0, width < 0, radius < 0 and a NaN centre still raise the same error")


def test_build_targets(n_real, cache):
    grid = Grid()
    rng = np.random.default_rng(0)
    for groups in (L.NUSC_GROUPS, GROUPS):
        fr = random_frames(rng, grid, groups)
        for cap in (64, 7):                                    # the second one cuts the lists
            same(build_targets_old(fr, grid, groups, cap), O.build_targets(fr, grid, groups, cap), f"random {groups[0][0]} cap {cap}")
    # boxes exactly on the edge cells
    edge = [[{"group": "vehicle", "cx": grid.x_min + 0.01, "cy": grid.y_min + 0.01, "vx": 1.0, "vy": 2.0, "length": 4.0, "width": 2.0, "yaw": 0.3},
             {"group": "pedestrian", "cx": grid.x_max - 0.01, "cy": grid.y_max - 0.01, "vx": 0.0, "vy": 0.0, "radius": 0.4}]]
    same(build_targets_old(edge, grid, L.NUSC_GROUPS), O.build_targets(edge, grid, L.NUSC_GROUPS), "edge cells")
    edge_cases(grid)
    files = sorted(glob.glob(os.path.join(cache, "*_0.5_4_*.npz")))[::97][:n_real]
    if not files:
        print("build_targets == old, bit for bit: random frames and edge cells (no cached GT here: the real-frame part is skipped)")
        return None
    gts = [np.load(f, allow_pickle=True)["gt"].item()["cur"] for f in files]
    fr = [[b for b in (L.head_box(x) for x in g["meta"]["boxes"]) if b] for g in gts]
    n = sum(len(f) for f in fr)
    same(build_targets_old(fr, grid, L.NUSC_GROUPS, 128), O.build_targets(fr, grid, L.NUSC_GROUPS, 128), "real frames")
    t_old = med(lambda: build_targets_old(fr, grid, L.NUSC_GROUPS, 128))
    t_new = med(lambda: O.build_targets(fr, grid, L.NUSC_GROUPS, 128))
    print(f"build_targets == old, bit for bit: random (2 group sets, cap 64 and 7, empty and outside frames), edge cells, {n} boxes of {len(fr)} real frames;"
          f" {t_old:.1f} -> {t_new:.1f} ms for the real frames")
    return gts


def test_stack(gts):
    cfg = L.LossConfig(w_obj_vel=1.0)
    if gts is None:
        import synthetic_world as sw
        scene = L._demo_scene(sw)
        gts = [L.synthetic_gt(scene, Grid()) for _ in range(3)]
    keys = ["drivable_area", "lane_divider_dist", "vehicle", "velocity", "camera", "radar", "occluded", "undefined", "instance"]
    new = L._stack_many(gts, keys, "cpu")
    for k in keys:
        old = L._stack(gts, k, "cpu")
        assert new[k].dtype == old.dtype and new[k].shape == old.shape and torch.equal(new[k], old), k
    # the whole build_layer_targets through the new path against the same function with the old per-layer path
    real = L._stack_many
    old_path = lambda g, ks, device: {k: L._stack(g, k, device) for k in dict.fromkeys(ks)}      # noqa: E731
    T_new = L.build_layer_targets(gts, cfg)
    try:
        L._stack_many = old_path
        T_old = L.build_layer_targets(gts, cfg)
        t_old = med(lambda: L.build_layer_targets(gts, cfg))
    finally:
        L._stack_many = real
    assert T_new.keys() == T_old.keys()
    for k in T_new:
        if k == "boxes":
            assert T_new[k] == T_old[k]
        else:
            assert T_new[k].dtype == T_old[k].dtype and torch.equal(T_new[k], T_old[k]), k
    t_new = med(lambda: L.build_layer_targets(gts, cfg))
    print(f"_stack_many == _stack for {len(keys)} keys; build_layer_targets identical in all {len(T_new)} entries ({len(gts)} samples); CPU {t_old:.1f} -> {t_new:.1f} ms"
          f" (on the GPU the saving is the copies: the layers move with 1 copy instead of about {len(gts) * 19}; the per-sample interpolated flag and w_reg stay, {len(gts) + 2} small ones)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--gt-cache", default=os.path.join(ROOT, "data", "cache", "gt"), help="folder of the cached GT (an empty one tests the synthetic fallback)")
    a = ap.parse_args()
    torch.set_num_threads(2)
    g = test_build_targets(a.frames, a.gt_cache)
    test_stack(g)
    print("host_targets passed")
