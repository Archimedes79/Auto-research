"""Is the box head optimal on the simplest question? And if not, why? (Mirko, 2026-10-26)

The simplest question: boxes of a synthetic scene are rendered into a pixel map (vehicle / pedestrian coverage, lane lines, all
objects 0.5 s earlier, as studies/early/object_head_synth.py), a trunk + the object head must give the boxes back. With a perfect input the
answer is (nearly) determined, so the trained head should approach it. This script separates the possible reasons for a gap:
  A  the INFORMATION CEILING: an analytic decoder (no learning) reads the same input: connected components of the coverage maps,
     centroid, area and second moments -> box (centre, length / width, yaw modulo pi, circle radius); velocity from the past map with
     the true association (so only the measurement precision is tested). Its metrics are what any decoder can get from this input.
  B  what an ERROR COSTS in IoU (analytic): centre shift, yaw error, velocity error x dt for a car and a pedestrian circle.
  C  LEARNING: the same network trained with more steps and more scenes: does it converge to the ceiling (then the gap is
     optimisation / data) or stay away from it (then capacity, loss or decoding)?
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python studies/early/box_optimality.py [--configs 200:400,800:2000] [--quick]
"""
import math
import random
import sys
import time

import numpy as np
import torch
from scipy import ndimage
from shapely import affinity
from shapely.geometry import Point, box as sbox

sys.path.insert(0, "training")
sys.path.insert(0, "tests/support")
import object_head_synth as T  # noqa: E402
from synthetic_world import random_scene  # noqa: E402

G, RES = T.G, T.G.res
QUICK = "--quick" in sys.argv
CFG = "200:400,200:2000,800:2000,800:6000"
for i, a in enumerate(sys.argv):
    if a == "--configs":
        CFG = sys.argv[i + 1]
ORACLE_ONLY = "--oracle-only" in sys.argv
CONFIGS = [] if ORACLE_ONLY else [(60, 40)] if QUICK else [tuple(int(v) for v in c.split(":")) for c in CFG.split(",")]
XS = G.x_min + (np.arange(G.nx) + 0.5) * RES
YS = G.y_min + (np.arange(G.ny) + 0.5) * RES
TRUCK_LENGTH = 6.2        # synthetic classes: car 4.5 x 1.8, truck 8.0 x 2.5, pedestrian 0.6 x 0.6


# ------------------------------------------------------------------------------------------------ A: analytic decoder
def components(cov, thr=0.02):
    lab, n = ndimage.label(cov > thr, structure=np.ones((3, 3)))
    return lab, n


def moments(cov, lab, k):
    ix, iy = np.nonzero(lab == k)
    w = cov[ix, iy]
    px, py = XS[ix], YS[iy]
    s = w.sum()
    mx, my = (w * px).sum() / s, (w * py).sum() / s
    c = np.array([[(w * (px - mx) ** 2).sum(), (w * (px - mx) * (py - my)).sum()],
                  [(w * (px - mx) * (py - my)).sum(), (w * (py - my) ** 2).sum()]]) / s
    ev, evec = np.linalg.eigh(c)                                # ascending
    return mx, my, s * RES * RES, ev, evec


def oracle_decode(x, gt_boxes):
    """x (4, X, Y) input maps -> list of box dicts (score 1). Velocity: centroid now minus centroid of the past map in a window
    around the true past position (true association), divided by 0.5 s."""
    veh, ped, past = x[0].numpy(), x[1].numpy(), x[3].numpy()
    dets = []
    for cov, kind in ((veh, "veh"), (ped, "ped")):
        lab, n = components(cov)
        for k in range(1, n + 1):
            mx, my, area, ev, evec = moments(cov, lab, k)
            # the cell-averaged coverage is the object convolved with the cell: variances add, so subtract res^2 / 12 per axis
            L = math.sqrt(max(12 * ev[1] - RES * RES, 1e-6))
            W = math.sqrt(max(12 * ev[0] - RES * RES, 1e-6))
            yaw = math.atan2(evec[1, 1], evec[0, 1])
            d = {"cx": mx, "cy": my, "score": 1.0, "vx": 0.0, "vy": 0.0}
            if kind == "ped":
                d.update(group="pedestrian", radius=L / 2)
            else:
                d.update(group="truck" if L > TRUCK_LENGTH else "car", length=L, width=W, yaw=yaw)
            dets.append(d)
    for d in dets:                                              # velocity with the true association (nearest GT box of the same kind)
        near = min((g for g in gt_boxes if (g["group"] == "pedestrian") == (d["group"] == "pedestrian")),
                   key=lambda g: math.hypot(g["cx"] - d["cx"], g["cy"] - d["cy"]), default=None)
        if near is None:
            continue
        px, py = near["cx"] - 0.5 * near["vx"], near["cy"] - 0.5 * near["vy"]
        win = (np.abs(XS[:, None] - px) <= 2.0) & (np.abs(YS[None, :] - py) <= 2.0)
        w = past * win
        if w.sum() > 1e-6:
            qx, qy = (w * XS[:, None]).sum() / w.sum(), (w * YS[None, :]).sum() / w.sum()
            d["vx"], d["vy"] = (d["cx"] - qx) / 0.5, (d["cy"] - qy) / 0.5
    return dets


# ------------------------------------------------------------------------------------------------ B: what an error costs
def iou_tables():
    car = sbox(-2.25, -0.9, 2.25, 0.9)
    ped = Point(0, 0).buffer(0.3, 32)
    print("B  IoU of a perfect box shifted / rotated by an error (analytic, shapely)")
    print(f"   {'error':22s}{'car 4.5 x 1.8':>16s}{'pedestrian r 0.3':>20s}")
    for d in (0.02, 0.05, 0.10, 0.15, 0.25, 0.50):
        row = []
        for shp in (car, ped):
            s = affinity.translate(shp, 0.0, d)                # sideways shift (the narrow side of the car)
            row.append(s.intersection(shp).area / s.union(shp).area)
        print(f"   {'centre ' + format(d, '.2f') + ' m (lateral)':22s}{row[0]:16.3f}{row[1]:20.3f}")
    for a in (2, 5, 10):
        r = affinity.rotate(car, a, origin="center")
        print(f"   {'yaw ' + str(a) + ' deg':22s}{r.intersection(car).area / r.union(car).area:16.3f}{'-':>20s}")
    print("   forecast: a velocity error sigma_v moves a box by sigma_v x dt (the centre rows above): at dt = 0.5 s")
    for sv in (0.2, 0.5, 1.0, 2.0):
        s = affinity.translate(car, 0.0, sv * 0.5)
        print(f"   {'sigma_v ' + format(sv, '.1f') + ' m/s':22s}{s.intersection(car).area / s.union(car).area:16.3f}   (car, shift {sv * 0.5:.2f} m)")


# ------------------------------------------------------------------------------------------------ table
def row_of(res):
    c, t, p = res.get("match car", {}), res.get("match truck", {}), res.get("match pedestrian", {})
    f = lambda d, k: d.get(k, float("nan"))
    return {"car AP@1": res["AP car @1.0"], "car IoU": f(c, "IoU"), "car centre": f(c, "centre"), "car yaw/pi": f(c, "yaw mod pi"),
            "car size": f(c, "size"), "car vel": f(c, "vel"), "truck AP@1": res["AP truck @1.0"], "ped AP@1": res["AP pedestrian @1.0"],
            "ped IoU": f(p, "IoU"), "ped centre": f(p, "centre"), "ped radius": f(p, "radius")}


def print_table(rows):
    keys = list(next(iter(rows.values())).keys())
    print(f"{'':30s}" + "".join(f"{k:>12s}" for k in keys))
    for name, r in rows.items():
        print(f"{name:30s}" + "".join(f"{r[k]:12.3f}" for k in keys), flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    print(f"device {T.DEV}; configs (training scenes : iterations) {CONFIGS}")
    iou_tables()
    n_test = 20 if QUICK else 50
    rng = random.Random(999)
    test = [T.scene_inputs(random_scene(rng)) for _ in range(n_test)]
    gts = [d[1] for d in test]
    rows = {}
    ora = [oracle_decode(d[0], d[1]) for d in test]
    rows["A analytic ceiling"] = row_of(T.evaluate(ora, gts))
    print(f"\nA  analytic decoder from the same input ({n_test} held-out scenes): the information ceiling")
    print_table(rows)
    print("\nC  learned: the same network trained with more steps and scenes (held-out scenes; the last column set = 20 training scenes)")
    rng = random.Random(2)
    pool = [T.scene_inputs(random_scene(rng)) for _ in range(max([c[0] for c in CONFIGS] or [0]))]
    for n_train, iters in CONFIGS:
        data = pool[:n_train]
        torch.manual_seed(0)
        random.seed(0)
        trunk, head = T.make_model()
        t0 = time.time()
        T.train(trunk, head, data, iters, 8)
        name = f"C {n_train} scenes, {iters} it"
        rows[name] = row_of(T.evaluate(T.predict(trunk, head, test), gts))
        rows[name + " (train)"] = row_of(T.evaluate(T.predict(trunk, head, data[:20]), [d[1] for d in data[:20]]))
        print(f"   {name}: {time.time() - t0:.0f} s", flush=True)
    print("\nsummary (car / truck / pedestrian: AP at 1 m, IoU, centre error m, yaw error deg mod pi, size error m, velocity error m/s)")
    print_table(rows)
