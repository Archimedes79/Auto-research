"""throw-away, CPU only: cost of object_head.build_targets per training step (~272 objects: 9 keyframe calls of 3 samples x 10 boxes) and the same with a window
around each centre instead of the whole grid; the two must give bit-identical targets"""
import math, os, random, sys, time
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion/temp/speed_base/bev_fusion/training")
import torch
import object_head as oh
from lift_pd import Grid
from losses import NUSC_GROUPS

grid = Grid()
print("grid", grid.nx, "x", grid.ny, "threads", torch.get_num_threads())


def build_targets_win(boxes_batch, grid, groups=oh.GROUPS, max_obj=64, min_radius=2):
    B, G, X, Y = len(boxes_batch), len(groups), grid.nx, grid.ny
    gidx = {name: i for i, (name, _) in enumerate(groups)}
    t = {"hm": torch.zeros(B, G, X, Y), "ind": torch.zeros(B, max_obj, dtype=torch.long),
         "mask": torch.zeros(B, max_obj), "group": torch.zeros(B, max_obj, dtype=torch.long),
         "off": torch.zeros(B, max_obj, 2), "size": torch.zeros(B, max_obj, 2), "rot": torch.zeros(B, max_obj, 2),
         "rad": torch.zeros(B, max_obj, 1), "vel": torch.zeros(B, max_obj, 2), "is_box": torch.zeros(B, max_obj),
         "dir": torch.zeros(B, max_obj, 1)}
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
            x0, x1, y0, y1 = max(0, ix - r), min(X, ix + r + 1), max(0, iy - r), min(Y, iy + r + 1)     # the peak is 0 outside |dx|, |dy| <= r
            gx = torch.arange(x0, x1, dtype=torch.float32).view(-1, 1)
            gy = torch.arange(y0, y1, dtype=torch.float32).view(1, -1)
            peak = torch.exp(-((gx - ix) ** 2 + (gy - iy) ** 2) / (2 * sigma * sigma))
            w = t["hm"][b, g, x0:x1, y0:y1]
            w.copy_(torch.maximum(w, peak))
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


rng = random.Random(0)


def make_boxes(n):
    out = []
    for _ in range(n):
        grp = rng.choice(NUSC_GROUPS)
        bx = {"group": grp[0], "cx": rng.uniform(-1, 61), "cy": rng.uniform(-27, 27), "vx": rng.uniform(-8, 8), "vy": rng.uniform(-8, 8)}
        if grp[1] == "box":
            bx.update(length=rng.uniform(1.5, 12), width=rng.uniform(0.6, 3), yaw=rng.uniform(-3.1, 3.1))
        else:
            bx["radius"] = rng.uniform(0.3, 0.9)
        out.append(bx)
    return out


calls = [[make_boxes(10) for _ in range(3)] for _ in range(9)]            # 9 keyframes x 3 samples x 10 boxes = 270 objects
for fn_name, fn in (("whole grid (HEAD)", oh.build_targets), ("window", build_targets_win)):
    fn(calls[0], grid, NUSC_GROUPS, 128)
    t0 = time.perf_counter()
    for _ in range(5):
        res = [fn(c, grid, NUSC_GROUPS, 128) for c in calls]
    dt = (time.perf_counter() - t0) / 5
    print(f"{fn_name:18} {dt * 1000:7.1f} ms per step (270 objects)")
ref = [oh.build_targets(c, grid, NUSC_GROUPS, 128) for c in calls]
new = [build_targets_win(c, grid, NUSC_GROUPS, 128) for c in calls]
same = all(torch.equal(a[k], b[k]) for a, b in zip(ref, new) for k in a)
print("bit-identical targets on", sum(len(b) for c in calls for b in c), "boxes:", same)
