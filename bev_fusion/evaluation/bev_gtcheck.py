"""GT against GT: does the evaluation do what it says? Real cached ground truth, no model, no GPU.

    python evaluation/bev_gtcheck.py [--n 24] [--sig 15efbde2d1] [--seed 0]        table: the GT used as the prediction, and degraded copies of it
    python evaluation/bev_gtcheck.py --versions 15efbde2d1 962e788566 [--n 200]     two stored versions of the SAME frames compared layer by layer
    python evaluation/bev_gtcheck.py --list                                          the stored GT signatures and how many frames each holds
    python evaluation/bev_gtcheck.py --selftest                                      the table with its expected answers asserted

The prediction variants run through the REAL metrics of the suite (BevSegMetric, BevDetMetric, BevCurveMetric; masks, dt keys, range bins as in
`bev_suite.py eval`), each frame as cur / prev / next = dt 0 / -0.5 / +0.5:
    oracle      the GT itself (maps, distance fields, boxes with score 1, velocity): every number must be at its ceiling (IoU 1, AP 1, AUC 1, R2 1, EPE 0)
    shift 1/3   the maps rolled by k cells (0.5 m) along x, the boxes moved by k x 0.5 m: tolerance 0 falls, tolerance >= k cells recovers
    noise       Gaussian noise on the probabilities: the ranking numbers (AUC, AP) fall smoothly
    dropout     half of the positive cells and half of the boxes removed
    shuffled    the GT of ANOTHER frame: low everywhere (the metrics see through a plausible but wrong answer)
    null        all probabilities 0, no boxes, velocity 0: IoU 0, AUC 0.5, AP = prevalence
What it does NOT test: the model, the GT generator's correctness against the world (that is autogt's own selftest and the visual checks), the
training path. A failing line here means the evaluation, the masks or the GT cache is wrong, not the network."""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")]
SIDE_DT = {"cur": 0.0, "prev": -0.5, "next": 0.5}
LINES = ("lane_divider", "road_divider")


def cache_dir():
    return os.path.join(ROOT, "data", "cache", "gt")


def signatures():
    out = {}
    for f in glob.glob(os.path.join(cache_dir(), "*.npz")):
        out[f.rsplit("_", 1)[-1][:-4]] = out.get(f.rsplit("_", 1)[-1][:-4], 0) + 1
    return out


def load(path):
    return np.load(path, allow_pickle=True)["gt"].item()


def complete(g):
    return all(isinstance(g.get(k), dict) and g[k] for k in SIDE_DT)


def pick(n, sig, seed):
    files = sorted(glob.glob(os.path.join(cache_dir(), f"*_{sig}.npz")))
    rng = np.random.RandomState(seed)
    rng.shuffle(files)
    out = []
    for f in files:
        g = load(f)
        if complete(g):
            out.append((os.path.basename(f), {SIDE_DT[k]: g[k] for k in SIDE_DT}))
        if len(out) == n:
            break
    return out


def f32(a):
    return np.asarray(a, np.float32)


def oracle_output(g, spec, groups):
    from evaluation.bev_views import gt_boxes
    prob = {name: f32(g[key]) for name, kind, key in spec if kind in ("area", "line", "object") and key in g}
    dist = {ln: f32(g[ln + "_dist"]) for ln in LINES if ln + "_dist" in g}
    return {"prob": prob, "dist": dist, "mass": None, "dets": [dict(b, score=1.0) for b in gt_boxes(g, groups)], "detr": None,
            "vhat": f32(g["velocity"])}


def shifted(o, k):
    return {"prob": {n: np.roll(a, k, axis=0) for n, a in o["prob"].items()}, "dist": {n: np.roll(a, k, axis=0) for n, a in o["dist"].items()},
            "mass": None, "dets": [dict(d, cx=d["cx"] + 0.5 * k) for d in o["dets"]], "detr": None, "vhat": np.roll(o["vhat"], k, axis=1)}


def noisy(o, sigma, rng):
    return dict(o, prob={n: np.clip(a + rng.normal(0, sigma, a.shape), 0, 1).astype(np.float32) for n, a in o["prob"].items()})


def dropped(o, rng):
    return dict(o, prob={n: np.where((a > 0.5) & (rng.rand(*a.shape) < 0.5), 0.0, a).astype(np.float32) for n, a in o["prob"].items()},
                dets=[d for d in o["dets"] if rng.rand() >= 0.5])


def null_output(o):
    return {"prob": {n: np.zeros_like(a) for n, a in o["prob"].items()}, "dist": {n: np.full_like(a, 2.0) for n, a in o["dist"].items()},
            "mass": None, "dets": [], "detr": None, "vhat": np.zeros_like(o["vhat"])}


def variants(samples, spec, groups, seed=0):
    """name -> [ {dt: output} per sample ]"""
    rng = np.random.RandomState(seed)
    orc = [{dt: oracle_output(g, spec, groups) for dt, g in s.items()} for _, s in samples]
    v = {"oracle": orc}
    for k in (1, 3):
        v[f"shift {k}"] = [{dt: shifted(o, k) for dt, o in s.items()} for s in orc]
    v["noise 0.2"] = [{dt: noisy(o, 0.2, rng) for dt, o in s.items()} for s in orc]
    v["noise 0.5"] = [{dt: noisy(o, 0.5, rng) for dt, o in s.items()} for s in orc]
    v["dropout 50 %"] = [{dt: dropped(o, rng) for dt, o in s.items()} for s in orc]
    v["shuffled"] = [orc[(i + 1) % len(orc)] for i in range(len(orc))]
    v["null"] = [{dt: null_output(o) for dt, o in s.items()} for s in orc]
    return v


def run(samples, outputs, task):
    from evalsuite import Prediction, Sample
    from evalsuite.contracts import FrameRef
    metrics = [m for m in task.metrics() if m.name != "loss"]
    counts = {m.name: [] for m in metrics}
    for i, ((name, s), out) in enumerate(zip(samples, outputs)):
        sample = Sample(ref=FrameRef("gt", i, name), inputs=None, reference=s)
        pred = Prediction(outputs={"dt": out})
        for m in metrics:
            counts[m.name].append(m.frame(sample, pred))
    res = {}
    for m in metrics:
        for k, v in m.reduce(counts[m.name]).items():
            res[f"{m.name}/{k}"] = v
    return res


SHOW = [("drivable IoU@0", "seg/fov/dt+0.00/drivable_area/all/all/iou@0"), ("lane div IoU@0", "seg/fov/dt+0.00/lane_divider/all/all/iou@0"),
        ("lane div IoU@0.5", "seg/fov/dt+0.00/lane_divider/all/all/iou@0.5"), ("lane div IoU@1", "seg/fov/dt+0.00/lane_divider/all/all/iou@1"),
        ("vehicle IoU@0", "seg/obj/dt+0.00/vehicle/all/all/iou@0"), ("vehicle AP@2", "det/dt+0.00/vehicle/all/ap@2"),
        ("drivable AP", "curve/dt+0.00/drivable_area/ap"), ("drivable AUC", "curve/dt+0.00/drivable_area/roc_auc"),
        ("lane div AP", "curve/dt+0.00/lane_divider/ap"), ("lane div AUC", "curve/dt+0.00/lane_divider/roc_auc"),
        ("lane div prevalence", "curve/dt+0.00/lane_divider/prevalence"), ("vel EPE moving", "curve/dt+0.00/velocity/moving/epe"),
        ("vel R2 moving", "curve/dt+0.00/velocity/moving/r2"), ("lane div IoU@0 dt+0.5", "seg/fov/dt+0.50/lane_divider/all/all/iou@0")]


def build_task():
    from evaluation.bev_task import BEVTask
    from lift_pd import Grid
    from losses import NUSC_GROUPS, layer_spec
    task = BEVTask(Grid(), layer_spec(False, True), [g for g, _ in NUSC_GROUPS])
    task.curves = True
    return task


def table(n, sig, seed):
    task = build_task()
    samples = pick(n, sig, seed)
    assert len(samples) >= min(n, 4), f"only {len(samples)} complete cached frames with signature {sig} in {cache_dir()}"
    rows = {name: run(samples, outs, task) for name, outs in variants(samples, task.spec, task.groups, seed).items()}
    return samples, rows


def print_table(rows):
    names = list(rows)
    print(f"{'':24s}" + "".join(f"{n:>14s}" for n in names))
    for label, key in SHOW:
        print(f"{label:24s}" + "".join(f"{rows[n].get(key, float('nan')):14.4f}" for n in names))


def print_alignment(iou, dev):
    print("GT frames against their poses (static layers of cur warped into the ego frame of prev / next, IoU of the binary maps where both exist):")
    for layer in STATIC:
        print(f"  {layer:14s}" + "".join(f"  dt{dt:+.1f}: unwarped {iou[(layer, dt, 'unwarped')]:.3f} -> warped {iou[(layer, dt, 'warped')]:.3f}" for dt in (-0.5, 0.5)))
    print(f"  ego motion of the recorded poses against the one integrated from meta['steps'] (x, y, yaw): xy mean {dev['xy_mean']:.3f} m, max {dev['xy_max']:.3f} m, yaw max {dev['yaw_max']:.5f} rad")


def selftest(n=24):
    samples, rows = table(n, "15efbde2d1", 0)
    g = lambda v, k: rows[v][k]  # noqa: E731
    L0, L1 = "seg/fov/dt+0.00/lane_divider/all/all/iou@0", "seg/fov/dt+0.00/lane_divider/all/all/iou@1"
    assert g("oracle", L0) > 0.999 and g("oracle", "seg/fov/dt+0.00/drivable_area/all/all/iou@0") > 0.999, rows["oracle"]
    assert g("oracle", "det/dt+0.00/vehicle/all/ap@2") > 0.999 and g("oracle", "curve/dt+0.00/lane_divider/ap") > 0.999
    assert g("oracle", "curve/dt+0.00/lane_divider/roc_auc") > 0.999 and g("oracle", "curve/dt+0.00/velocity/moving/epe") < 1e-3
    assert g("oracle", "curve/dt+0.00/velocity/moving/r2") > 0.999
    assert g("oracle", L0) > g("shift 1", L0) > g("shift 3", L0) and g("shift 1", L1) > 0.99 > g("shift 3", L0)       # tolerance recovers a 1-cell shift
    assert g("oracle", "curve/dt+0.00/lane_divider/roc_auc") > g("noise 0.2", "curve/dt+0.00/lane_divider/roc_auc") > g("noise 0.5", "curve/dt+0.00/lane_divider/roc_auc") > 0.5
    assert g("shuffled", L0) < 0.5 * g("oracle", L0)
    assert g("null", L0) == 0.0 and abs(g("null", "curve/dt+0.00/lane_divider/roc_auc") - 0.5) < 1e-9
    assert abs(g("null", "curve/dt+0.00/lane_divider/ap") - g("null", "curve/dt+0.00/lane_divider/prevalence")) < 1e-9
    assert g("null", "det/dt+0.00/vehicle/all/ap@2") == 0.0
    assert g("dropout 50 %", "det/dt+0.00/vehicle/all/ap@2") < g("oracle", "det/dt+0.00/vehicle/all/ap@2")
    iou, dev = alignment(samples)
    for layer, floor in (("drivable_area", 0.95), ("walkway", 0.9), ("road_divider", 0.5)):
        for dt in (-0.5, 0.5):
            assert iou[(layer, dt, "warped")] > max(floor, iou[(layer, dt, "unwarped")] + 0.1), (layer, dt, iou)          # thin lines: 0.61 - 0.66 measured, floor 0.5
    assert dev["xy_max"] < 0.15 and dev["yaw_max"] < 0.01, f"the ego motion of the poses and of meta['steps'] differ: {dev}"       # measured: 0.07 m, 9e-5 rad
    print(f"bev_gtcheck selftest ok ({len(samples)} real frames, 3 dt each)")
    print_table(rows)
    print_alignment(iou, dev)


RES, X0, Y0 = 0.5, 0.0, -26.0                              # the front grid: axis 0 = x forward from 0, axis 1 = y (left positive) from -26 m
STATIC = ("drivable_area", "walkway", "road_divider")


def to_frame(g_from, g_to):
    """2D rigid transform (R, t): a point in the ego frame of g_from -> in the ego frame of g_to, from the recorded ego poses T_e2g"""
    T = np.linalg.inv(np.asarray(g_to["meta"]["T_e2g"], np.float64)) @ np.asarray(g_from["meta"]["T_e2g"], np.float64)
    return T[:2, :2], T[:2, 3]


def warp(g_from, g_to, key):
    """the map `key` of g_from seen from the ego frame of g_to (nearest cell) and the mask of the cells that have a source"""
    R, t = to_frame(g_from, g_to)
    h, w = np.asarray(g_from[key]).shape
    xs, ys = X0 + (np.arange(h) + 0.5) * RES, Y0 + (np.arange(w) + 0.5) * RES
    P = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)           # cells of g_to, metres in its frame
    src = (P - t) @ R                                                             # = R^T (p - t): the same points in the frame of g_from
    i, j = np.floor((src[:, 0] - X0) / RES).astype(int), np.floor((src[:, 1] - Y0) / RES).astype(int)
    ok = (i >= 0) & (i < h) & (j >= 0) & (j < w)
    out = np.zeros(h * w, np.float32)
    out[ok] = f32(g_from[key])[i[ok], j[ok]]
    return out.reshape(h, w), ok.reshape(h, w)


def rel_pose(Ta, Tb):
    """the pose of ego b in the ego frame of a (x, y, yaw) from the 4x4 poses ego -> global"""
    T = np.linalg.inv(np.asarray(Ta, np.float64)) @ np.asarray(Tb, np.float64)
    return float(T[0, 3]), float(T[1, 3]), float(math.atan2(T[1, 0], T[0, 0]))


def steps_displacement(steps):
    """(x, y, yaw) of the ego after the CTRV sub-steps (dt, speed, yaw rate) of meta['steps']"""
    x = y = th = 0.0
    for dt, v, w in steps:
        x += v * np.cos(th) * dt
        y += v * np.sin(th) * dt
        th += w * dt
    return x, y, th


def alignment(samples):
    """Are the GT frames consistent with their poses? The static layers of `cur` warped into the ego frame of `prev` / `next` with the recorded
    poses must agree with that frame's own GT where both are defined; unwarped they must not (the ego moves 4 - 11 m). Also the displacement of the
    poses against the one integrated from meta['steps'] (what the model's time mover uses)."""
    acc, dev = {}, []
    for _, s in samples:
        for dt in (-0.5, 0.5):
            for layer in STATIC:
                w, ok = warp(s[0.0], s[dt], layer)
                tgt = f32(s[dt][layer]) > 0.5
                for tag, src in (("warped", w > 0.5), ("unwarped", f32(s[0.0][layer]) > 0.5)):
                    m = ok if tag == "warped" else np.ones_like(ok)
                    inter, union = float((src & tgt & m).sum()), float(((src | tgt) & m).sum())
                    r = acc.setdefault((layer, dt, tag), [0.0, 0.0])
                    r[0] += inter
                    r[1] += union
            side = s[dt]["meta"]
            if side.get("steps"):                                       # steps: the ego motion from cur to the neighbour, backwards in time for prev (what the time mover is given)
                px, py, pyaw = rel_pose(s[0.0]["meta"]["T_e2g"], s[dt]["meta"]["T_e2g"])
                sx, sy, syaw = steps_displacement(side["steps"])
                dev.append((float(np.hypot(px - sx, py - sy)), abs(math.remainder(pyaw - syaw, 2 * math.pi))))
    mean_xy = float(np.mean([d[0] for d in dev])) if dev else float("nan")
    return {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in acc.items()}, {"xy_mean": mean_xy, "xy_max": max((d[0] for d in dev), default=float("nan")),
                                                                                   "yaw_max": max((d[1] for d in dev), default=float("nan"))}


def compare_versions(a, b, n, seed):
    """the same frames in two stored GT versions: per layer the share of cells that differ and the IoU of the binarised maps; masks and boxes"""
    fa = {os.path.basename(f).rsplit("_", 1)[0]: f for f in glob.glob(os.path.join(cache_dir(), f"*_{a}.npz"))}
    fb = {os.path.basename(f).rsplit("_", 1)[0]: f for f in glob.glob(os.path.join(cache_dir(), f"*_{b}.npz"))}
    common = sorted(set(fa) & set(fb))
    rng = np.random.RandomState(seed)
    rng.shuffle(common)
    common = common[:n]
    print(f"{a}: {len(fa)} frames, {b}: {len(fb)} frames, in both: {len(set(fa) & set(fb))}, compared: {len(common)}")
    acc = {}
    for t in common:
        ga, gb = load(fa[t]).get("cur"), load(fb[t]).get("cur")
        if not ga or not gb:
            continue
        for k in sorted(set(ga) & set(gb)):
            xa, xb = ga[k], gb[k]
            if not hasattr(xa, "shape") or xa.shape != getattr(xb, "shape", None) or xa.dtype == object:
                continue
            xa, xb = f32(xa), f32(xb)
            ba, bb = xa > 0.5, xb > 0.5
            r = acc.setdefault(k, [0, 0.0, 0.0, 0.0, 0.0])
            r[0] += 1
            r[1] += float(np.mean(np.abs(xa - xb) > 1e-3))
            r[2] += float((ba & bb).sum())
            r[3] += float((ba | bb).sum())
            r[4] += float(np.abs(xa - xb).max())
        na, nb = len(ga["meta"]["boxes"]), len(gb["meta"]["boxes"])
        r = acc.setdefault("boxes (count)", [0, 0.0, 0.0, 0.0, 0.0])
        r[0] += 1
        r[1] += float(na != nb)
    print(f"{'layer':22s}{'frames':>8s}{'cells differ':>14s}{'IoU(bin)':>10s}{'max |d|':>10s}")
    for k, r in acc.items():
        if k == "boxes (count)":
            print(f"{k:22s}{r[0]:8d}{r[1] / r[0]:14.3f}  (share of frames with another number of boxes)")
        else:
            print(f"{k:22s}{r[0]:8d}{r[1] / r[0]:14.4f}{(r[2] / r[3]) if r[3] else float('nan'):10.4f}{r[4]:10.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--sig", default="15efbde2d1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--versions", nargs=2, metavar=("SIG_A", "SIG_B"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.list:
        for s, c in sorted(signatures().items(), key=lambda x: -x[1]):
            print(f"{s}  {c} frames")
    elif a.selftest:
        selftest(a.n)
    elif a.versions:
        compare_versions(a.versions[0], a.versions[1], a.n, a.seed)
    else:
        samples, rows = table(a.n, a.sig, a.seed)
        print(f"{len(samples)} real GT frames (cur / prev / next), signature {a.sig}")
        print_table(rows)
        print_alignment(*alignment(samples))


if __name__ == "__main__":
    main()
