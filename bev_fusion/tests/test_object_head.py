"""Gate of the object head, stage 1 (plan.md 8.3, S5a): centre points -> oriented boxes (cars, trucks) and circles
(pedestrians) on the synthetic world, (1) overfit one scene, (2) train on random scenes, measure on held-out scenes.

Input of the test network (what a BEV trunk would see from the sensors, here rendered from the synthetic GT):
vehicle coverage at t, pedestrian coverage at t, lane lines, and all objects 0.5 s earlier (in the current frame, as
an ego-aligned past would be) - so the velocity is inferable, the car / truck class from the size, and the heading of
a moving object from its motion (a static box's heading is ambiguous by 180 deg: yaw is reported modulo pi and modulo
2 pi). Network: training/bev_trunk.py BEVTrunk(width 0.5).features + training/object_head.py ObjectHead.
Metrics on held-out scenes: AP per class at centre distances 0.5 / 1 / 2 m (all-point AP, simplified), and for the
matches at 1 m: centre error, BEV IoU (shapely polygons; circles as polygons), size error, yaw error, radius error,
velocity error.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_object_head.py [--quick]
"""
import math
import random
import sys
import time

import torch
from shapely.geometry import Point, Polygon

sys.path.insert(0, "training")
sys.path.insert(0, "tests")
from bev_trunk import BEVTrunk  # noqa: E402
from lift_pd import Grid  # noqa: E402
from object_head import GROUPS, ObjectHead, build_targets, decode, head_loss  # noqa: E402
from synthetic_world import gt_bev, random_scene  # noqa: E402

G = Grid()
DEV = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 3e9 else "cpu"
QUICK = "--quick" in sys.argv


def scene_inputs(scene):
    """(4, X, Y) input maps and the GT box list of a scene."""
    now = gt_bev(scene, G)
    past = scene.__class__(boxes=[b.__class__(**{**b.__dict__, "cx": b.cx - 0.5 * b.vx, "cy": b.cy - 0.5 * b.vy})
                                  for b in scene.boxes], lanes=scene.lanes, lane_pose=scene.lane_pose)
    pg = gt_bev(past, G)
    x = torch.stack([now["vehicles"], now["pedestrians"], now["lanes"], torch.maximum(pg["vehicles"], pg["pedestrians"])])
    boxes = []
    for b in scene.boxes:
        d = {"group": b.cls, "cx": b.cx, "cy": b.cy, "vx": b.vx, "vy": b.vy}
        if b.cls == "pedestrian":
            d["radius"] = max(b.length, b.width) / 2
        else:
            d.update(length=b.length, width=b.width, yaw=b.heading)
        boxes.append(d)
    return x, boxes


def shape(b):
    if "radius" in b:
        return Point(b["cx"], b["cy"]).buffer(b["radius"], 32)
    c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
    hl, hw = b["length"] / 2, b["width"] / 2
    return Polygon([(b["cx"] + c * a - s * w, b["cy"] + s * a + c * w) for a, w in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))])


def evaluate(dets, gts):
    """dets, gts: lists (scenes) of box lists -> metrics dict."""
    res = {}
    for name, kind in GROUPS:
        for thr in (0.5, 1.0, 2.0):
            cand = [(d["score"], s, d) for s, ds in enumerate(dets) for d in ds if d["group"] == name]
            cand.sort(key=lambda c: -c[0])
            n_gt = sum(1 for gs in gts for g in gs if g["group"] == name)
            used, tp, fp, prec, rec = set(), 0, 0, [], []
            matches = []
            for sc, s, d in cand:
                best, bd = None, thr
                for j, g in enumerate(gts[s]):
                    if g["group"] != name or (s, j) in used:
                        continue
                    dist = math.hypot(d["cx"] - g["cx"], d["cy"] - g["cy"])
                    if dist <= bd:
                        best, bd = j, dist
                if best is None:
                    fp += 1
                else:
                    used.add((s, best))
                    tp += 1
                    matches.append((d, gts[s][best], bd))
                prec.append(tp / (tp + fp))
                rec.append(tp / max(1, n_gt))
            ap, pmax = 0.0, 0.0
            for i in range(len(prec) - 1, -1, -1):                    # all-point interpolated AP
                pmax = max(pmax, prec[i])
                prec[i] = pmax
            last_r = 0.0
            for p, r in zip(prec, rec):
                ap += p * (r - last_r)
                last_r = r
            res[f"AP {name} @{thr}"] = ap if n_gt else float("nan")
            if thr == 1.0 and matches:
                ce = [m[2] for m in matches]
                iou = [shape(d).intersection(shape(g)).area / shape(d).union(shape(g)).area for d, g, _ in matches]
                ve = [math.hypot(d["vx"] - g["vx"], d["vy"] - g["vy"]) for d, g, _ in matches]
                row = {"n": len(matches), "centre": sum(ce) / len(ce), "IoU": sum(iou) / len(iou), "vel": sum(ve) / len(ve)}
                if kind == "box":
                    row["size"] = sum(abs(d["length"] - g["length"]) + abs(d["width"] - g["width"]) for d, g, _ in matches) / (2 * len(matches))
                    row["yaw mod pi"] = sum(abs(math.degrees(math.remainder(d["yaw"] - g["yaw"], math.pi))) for d, g, _ in matches) / len(matches)
                    row["yaw mod 2pi"] = sum(abs(math.degrees(math.remainder(d["yaw"] - g["yaw"], 2 * math.pi))) for d, g, _ in matches) / len(matches)
                else:
                    row["radius"] = sum(abs(d["radius"] - g["radius"]) for d, g, _ in matches) / len(matches)
                res[f"match {name}"] = row
    return res


def make_model():
    trunk = BEVTrunk(cin=4, width=0.5)
    head = ObjectHead(cin=trunk.img_bev_encoder_neck.out_channels)
    return trunk.to(DEV), head.to(DEV)


def train(trunk, head, data, iters, bs, lr=2e-3):
    opt = torch.optim.AdamW(list(trunk.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=iters, pct_start=0.1)
    trunk.train()
    head.train()
    t0 = time.time()
    for it in range(iters):
        batch = [data[random.randrange(len(data))] for _ in range(bs)]
        x = torch.stack([b[0] for b in batch]).to(DEV)
        t = build_targets([b[1] for b in batch], G)
        loss = head_loss(head(trunk.features(x)), t)
        opt.zero_grad()
        loss["total"].backward()
        torch.nn.utils.clip_grad_norm_(list(trunk.parameters()) + list(head.parameters()), 10.0)
        opt.step()
        sched.step()
        if it % max(1, iters // 5) == 0 or it == iters - 1:
            print(f"  iter {it:5d} loss {loss['total'].item():.3f} (hm {loss['hm'].item():.3f}) {time.time() - t0:.0f} s", flush=True)


@torch.no_grad()
def predict(trunk, head, data):
    trunk.eval()
    head.eval()
    dets = []
    for i in range(0, len(data), 8):
        x = torch.stack([d[0] for d in data[i:i + 8]]).to(DEV)
        out = {k: v.cpu() for k, v in head(trunk.features(x)).items()}
        dets.extend(decode(out, G, threshold=0.2))
    return dets


def show(res, title):
    print(title)
    for k, v in res.items():
        if k.startswith("AP"):
            print(f"  {k:24s} {v:.3f}")
    for k, v in res.items():
        if k.startswith("match"):
            print(f"  {k:18s} " + ", ".join(f"{a} {b:.3f}" if isinstance(b, float) else f"{a} {b}" for a, b in v.items()))


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    print(f"device {DEV}")
    rng = random.Random(1)
    one = [scene_inputs(random_scene(rng))]
    trunk, head = make_model()
    print("(1) overfit one scene")
    train(trunk, head, one, 150 if QUICK else 400, 1)
    r1 = evaluate(predict(trunk, head, one), [one[0][1]])
    show(r1, "    result on that scene")
    n_train, n_test = (200, 20) if QUICK else (1500, 50)
    t0 = time.time()
    rng = random.Random(2)
    train_set = [scene_inputs(random_scene(rng)) for _ in range(n_train)]
    rng = random.Random(999)
    test_set = [scene_inputs(random_scene(rng)) for _ in range(n_test)]
    print(f"(2) {n_train} training scenes, {n_test} held-out scenes (rendered in {time.time() - t0:.0f} s)")
    trunk, head = make_model()
    train(trunk, head, train_set, 400 if QUICK else 4000, 8)
    show(evaluate(predict(trunk, head, test_set), [d[1] for d in test_set]), "    held-out scenes")
    show(evaluate(predict(trunk, head, train_set[:n_test]), [d[1] for d in train_set[:n_test]]), "    training scenes (same count)")
