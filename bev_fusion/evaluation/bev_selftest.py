"""Known-answer tests of the BEV task of the evaluation suite (evaluation/bev_task.py, bev_views.py, bev_suite.py, bev_legacy.py).

    python evaluation/bev_selftest.py             the tests on a synthetic BEV world (no checkpoint, about 1 minute)
    python evaluation/bev_selftest.py --golden    ONLY: the numbers of the suite on 24 real val frames against the numbers of the former
                                                  evaluate_run.py on the same frames (CPU, 3 minutes; needs runs/t1_r4_soa)
    python evaluation/bev_selftest.py --real      ONLY: the adapter on a real run: contract checker, every view, a quantitative run, outliers
(--golden and --real are separate processes on purpose: one run's pinned code snapshot per process)

The synthetic world is a curved road with lane / road dividers, vehicles and a pedestrian (GT layers and boxes like autogt's), and a
"network" that predicts the GT with noise; chosen frames are corrupted on purpose, so the outlier search has a known answer.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import traceback
import zlib

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, ".."))                          # the repository root: `import autogt` (the suite lives there since 2026-11-01)
sys.path.insert(0, os.path.join(ROOT, "..", "autoresearch"))
from evaluation import bev_adapter as A  # noqa: E402

import torch  # noqa: E402

from evalsuite import EvalSuite, FrameRef, Prediction, Sample, Source, Subject  # noqa: E402
from evalsuite.render import figure  # noqa: E402
from evaluation import bev_legacy, bev_suite  # noqa: E402
from evaluation.bev_task import BEVTask, BevDetMetric, BevSegMetric, MASKS, add_sample  # noqa: E402
from evaluation.bev_views import gt_boxes  # noqa: E402
from evaluation.metrics import DetMeter, SegMeter, dt_key, flatten  # noqa: E402
from evaluation.metrics import match as metrics_match  # noqa: E402
from evalsuite.comparison import boxes as BX  # noqa: E402

AREAS = ("drivable_area", "ped_crossing", "walkway", "stop_line", "carpark_area")
LINES = ("lane_divider", "road_divider")
OBJECTS = ("vehicle", "pedestrian", "two_wheeler")
GRID, SPEC, DTS = None, None, (-0.5, 0.0, 0.5)          # set by init_synthetic()
CHECKS = []
PLANTED = {("bev-02", 3), ("bev-04", 5)}


def init_synthetic():
    """the grid and the head layout of the working tree's chain. NOT done for --golden / --real: one run's pinned code snapshot per
    process, and importing the working tree's modules first would shadow it."""
    global GRID, SPEC
    A.setup_paths(None)
    from lift_pd import Grid
    from losses import layer_spec
    GRID, SPEC = Grid(), layer_spec(False, True)


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


# ------------------------------------------------------------------------------------------------ the synthetic world
def _centres():
    xs = GRID.x_min + (np.arange(GRID.nx) + 0.5) * GRID.res
    ys = GRID.y_min + (np.arange(GRID.ny) + 0.5) * GRID.res
    return np.meshgrid(xs, ys, indexing="ij")


def synth_gt(seed: int, dt: float = 0.0) -> dict:
    """GT layers + masks + meta of one frame (the keys of autogt's keyframe dict that the BEV task reads)"""
    rng = np.random.default_rng(seed)
    X, Y = _centres()
    ph = rng.uniform(0, 6)
    yc = 2.0 * np.sin(X / 12.0 + ph)
    g = {"drivable_area": (np.abs(Y - yc) < 5.0).astype(np.float32),
         "walkway": ((np.abs(Y - yc) >= 5.0) & (np.abs(Y - yc) < 8.0)).astype(np.float32),
         "ped_crossing": ((X > 20) & (X < 23) & (np.abs(Y - yc) < 5)).astype(np.float32),
         "stop_line": np.zeros_like(X, np.float32), "carpark_area": np.zeros_like(X, np.float32)}
    d_lane, d_road = np.abs(Y - yc), np.minimum(np.abs(Y - yc - 5.0), np.abs(Y - yc + 5.0))
    g["lane_divider_dist"], g["road_divider_dist"] = np.minimum(d_lane, 2.0).astype(np.float32), np.minimum(d_road, 2.0).astype(np.float32)
    g["lane_divider"], g["road_divider"] = (d_lane < GRID.res / 2).astype(np.float32), (d_road < GRID.res / 2).astype(np.float32)
    g["vehicle"] = np.zeros_like(X, np.float32)
    g["pedestrian"], g["two_wheeler"] = np.zeros_like(X, np.float32), np.zeros_like(X, np.float32)
    inst, boxes, vel = -np.ones(X.shape, np.int32), [], np.zeros((2,) + X.shape, np.float32)
    for k in range(3):
        cx = 12 + 14 * k + rng.uniform(-2, 2)
        cy = float(2.0 * np.sin(cx / 12.0 + ph) + rng.uniform(-1.5, 1.5)) + (0.0 if k % 2 == 0 else 0.0)
        yaw, speed = 0.0, float(rng.uniform(0, 8))
        c, s = math.cos(yaw), math.sin(yaw)
        inside = (np.abs(c * (X - cx) + s * (Y - cy)) <= 2.25) & (np.abs(-s * (X - cx) + c * (Y - cy)) <= 0.9)
        g["vehicle"][inside] = 1.0
        inst[inside] = len(boxes)
        vel[0][inside] = speed
        boxes.append({"group": "vehicle", "center": [cx, cy, 0.8], "size": [1.8, 4.5, 1.6], "yaw": yaw, "vel": [speed, 0.0],
                      "observed": True, "stationary": speed < 0.5})
    px, py = 8.0, float(2.0 * np.sin(8.0 / 12.0 + ph) + 3.0)
    ped = (np.hypot(X - px, Y - py) <= 0.5)
    g["pedestrian"][ped] = 1.0
    boxes.append({"group": "pedestrian", "center": [px, py, 0.9], "size": [0.8, 0.8, 1.7], "yaw": 0.0, "vel": [0.0, 1.0],
                  "observed": True, "stationary": False})
    ang = np.abs(np.arctan2(Y, X))
    cam = (ang <= math.atan(800.0 / 1266.4) + 0.0).astype(np.float32)
    g.update({"instance": inst, "velocity": vel, "camera": cam, "occluded": np.zeros_like(cam), "undefined": np.zeros_like(cam),
              "ignore": np.zeros_like(cam), "radar": np.ones_like(cam)})
    xs = np.linspace(0, 55, 40)
    g["meta"] = {"boxes": boxes, "K": np.array([[1266.4, 0, 800], [0, 1266.4, 450], [0, 0, 1.0]]),
                 "T_c2e": np.array([[0, 0, 1, 1.5], [-1, 0, 0, 0], [0, -1, 0, 1.5], [0, 0, 0, 1.0]]),
                 "vectors": {"lane_divider": [np.c_[xs, 2.0 * np.sin(xs / 12.0 + ph)]]}}
    return g


class SynthSource(Source):
    def __init__(self, n_scenes=6, n_frames=8):
        self.n_scenes, self.n_frames = n_scenes, n_frames

    def scenes(self):
        return [f"bev-{i:02d}" for i in range(self.n_scenes)]

    def frames(self, scene):
        i = int(scene.split("-")[1])
        return [FrameRef(scene, t, f"k{i}x{t}") for t in range(self.n_frames)]

    def load(self, ref):
        i, t = int(ref.scene.split("-")[1]), ref.index
        rng = np.random.default_rng(100 * i + t)
        n = 60
        batch = {"prob": torch.softmax(torch.randn(1, 1, 118, 16, 44), 2),
                 "radar": {"x": torch.tensor(rng.uniform(0, 60, n), dtype=torch.float32), "y": torch.tensor(rng.uniform(-25, 25, n), dtype=torch.float32),
                           "vr": torch.tensor(rng.normal(0, 5, n), dtype=torch.float32), "rcs": torch.tensor(rng.normal(0, 5, n), dtype=torch.float32)}}
        return Sample(ref, batch, {dt: synth_gt(1000 * i + 10 * t, dt) for dt in DTS}, {}, {"planted": (ref.scene, t) in PLANTED})

    def describe(self):
        return {"class": "SynthSource", "scenes": self.n_scenes, "frames": self.n_frames}


class SynthSubject(Subject):
    """predicts the GT with noise (a carried bias grows along a scene: the state matters); frames in PLANTED are badly wrong"""
    warmup = 1

    def __init__(self, noise=0.15, shift=0, name="synth"):
        self.noise, self.shift, self.name = noise, shift, name
        self.t = 0

    def reset(self):
        self.t = 0

    def describe(self):
        return {"class": "SynthSubject", "name": self.name, "noise": self.noise, "shift": self.shift}

    def predict(self, sample, taps=()):
        rng = np.random.default_rng(zlib.crc32(f"{sample.ref.label}|{self.name}".encode()))      # stable across processes
        bad = sample.meta["planted"]
        sh = 7 if bad else self.shift
        dts = {}
        for dt, g in sample.reference.items():
            prob, dist = {}, {}
            for name, kind, key in SPEC:
                if kind in ("area", "line", "object"):
                    a = np.roll(np.asarray(g[key], np.float32), sh, axis=1)
                    prob[name] = np.clip(a + rng.normal(0, self.noise, a.shape), 0, 1).astype(np.float32)
                elif kind == "dist":
                    dist[name[:-5]] = np.clip(np.roll(np.asarray(g[key], np.float32), sh, axis=1) + rng.normal(0, 0.1, g[key].shape), 0, 2).astype(np.float32)
            dets = []
            if not bad:
                for b in gt_boxes(g, ["vehicle", "pedestrian", "two_wheeler"]):
                    d = dict(b, cx=b["cx"] + rng.normal(0, 0.4), cy=b["cy"] + rng.normal(0, 0.4), score=float(rng.uniform(0.6, 0.99)))
                    dets.append(d)
            dets += [{"group": "vehicle", "cx": float(rng.uniform(5, 50)), "cy": float(rng.uniform(-10, 10)), "length": 4.5, "width": 1.8,
                      "yaw": 0.0, "vx": 0.0, "vy": 0.0, "score": float(rng.uniform(0.05, 0.3))} for _ in range(4 if not bad else 12)]
            mass = np.ones(g["camera"].shape, np.float32) if dt == 0 else (np.arange(120)[:, None] < 110).astype(np.float32) * np.ones((1, 104), np.float32)
            dts[dt] = {"prob": prob, "dist": dist, "mass": mass, "dets": dets, "detr": [dict(d, score=d["score"] * 0.9) for d in dets],
                       "vhat": (np.asarray(g["velocity"], np.float32) * (0.0 if bad else 0.9))}
        self.t += 1
        return Prediction(outputs={"dt": dts}, info={"loss": {"total": 5.0 + (20.0 if bad else 0.0), "group.areas": 1.0}})


def make_suite(out, **kw):
    state = kw.pop("state", "carry")
    task = BEVTask(GRID, SPEC, ["vehicle", "two_wheeler", "pedestrian"], refine=True)
    return EvalSuite(SynthSource(), SynthSubject(**kw), task, out=out, state=state)


def frames_pred(n_frames=20):
    src, sub = SynthSource(), SynthSubject()
    out = []
    for sc in src.scenes()[:5]:
        sub.reset()
        for ref in src.frames(sc)[: n_frames // 5]:
            s = src.load(ref)
            out.append((s, sub.predict(s)))
    return out


def same(a, b, tol=1e-12):
    return (a != a and b != b) or abs(a - b) <= tol * max(1.0, abs(a))


# ------------------------------------------------------------------------------------------------ tests
@check("BevSegMetric: reduce(all frames) = the SegMeters fed directly (every key); reduce(keys=) and per-scene sums; score = the frame's own IoU")
def _():
    m = BevSegMetric(GRID, SPEC)
    fp = frames_pred(20)
    counts = [m.frame(s, p) for s, p in fp]
    meters = {k: SegMeter(GRID) for k in MASKS}
    for s, p in fp:
        for dt, o in p.outputs["dt"].items():
            add_sample(meters, o["prob"], o["dist"], s.reference[dt], SPEC, dt, o["mass"] if dt else None)
    want = {}
    for mname, mt in meters.items():
        want.update({f"{mname}/" + k[len("seg/"):]: v for k, v in flatten({"seg": mt.result()}).items()})
    got = m.reduce(counts)
    assert set(got) == set(want) and all(same(got[k], want[k]) for k in want), [k for k in want if not same(got.get(k, -1), want[k])][:3]
    head = list(m.headline)
    part = m.reduce(counts, keys=head)
    assert list(part) == head and all(same(part[k], got[k]) for k in head)
    # one scene = the direct meters over its frames
    sc0 = [(s, p) for s, p in fp if s.ref.scene == fp[0][0].ref.scene]
    d0 = SegMeter(GRID)
    for s, p in sc0:
        for dt, o in p.outputs["dt"].items():
            if dt == 0.0:
                add_sample({"fov": d0, "seen": SegMeter(GRID), "out": SegMeter(GRID), "obj": SegMeter(GRID)}, o["prob"], o["dist"], s.reference[dt], SPEC, dt, None)
    r0 = m.reduce([c for (s, _), c in zip(fp, counts) if s.ref.scene == fp[0][0].ref.scene], keys=head)
    assert same(r0[head[0]], d0.result()[0.0]["drivable_area"]["all"]["all"]["iou@0"])
    # the frame's own score equals the direct meter on that single frame
    s, p = fp[3]
    one = {k: SegMeter(GRID) for k in MASKS}
    for dt, o in p.outputs["dt"].items():
        add_sample(one, o["prob"], o["dist"], s.reference[dt], SPEC, dt, o["mass"] if dt else None)
    assert same(m.score(counts[3])["drivable_iou"], one["fov"].result()[0.0]["drivable_area"]["all"]["all"]["iou@0"])
    assert m.support(counts[3])["drivable_iou"] == one["fov"].result()[0.0]["drivable_area"]["all"]["all"]["n_gt"]
    assert m.reduce([]) is not None


@check("BevDetMetric: reduce(all frames) = DetMeter fed directly (every key, plain and refined boxes); selective keys; headline AP")
def _():
    fp = frames_pred(20)
    groups = ["vehicle", "two_wheeler", "pedestrian"]
    for src_name, name in (("dets", "det"), ("detr", "detr")):
        m = BevDetMetric(groups, name, src_name)
        counts = [m.frame(s, p) for s, p in fp]
        dm = DetMeter(classes=groups)
        for s, p in fp:
            for dt, o in p.outputs["dt"].items():
                dm.add(o[src_name], gt_boxes(s.reference[dt], groups), dt=dt)
        want = {k[len("det/"):]: v for k, v in flatten({"det": dm.result()}).items()}
        got = m.reduce(counts)
        vel = {k: v for k, v in got.items() if k.rsplit("/", 1)[-1] in BX.velocity_numbers(np.zeros(9))}     # the moving / static split: its own check below
        got = {k: v for k, v in got.items() if k not in vel}
        assert set(got) == set(want) and all(same(got[k], want[k]) for k in want), [k for k in want if not same(got.get(k, -1), want[k])][:3]
        assert len(vel) == len(DTS) * len(groups) * len(BX.velocity_numbers(np.zeros(9))) and all(k.split("/")[2] == "all" for k in vel)
        part = m.reduce(counts, keys=list(m.headline))
        assert all(same(part[k], want[k]) for k in m.headline if k in want) and all(same(part[k], vel[k]) for k in m.headline if k in vel)
        assert set(m.headline) <= set(want) | set(vel)
        sc = m.score(counts[0])
        assert 0 <= sc["recall2"] <= 1 and sc["fp_hi"] >= 0 and set(m.support(counts[0])) == set(sc)


@check("BevDetMetric velocity split: moving / static / zero predictor / predicted-standing against a plain-loop count on a planted world; the two groups pool to the class's ave; scenes add up; a run without the counts gives nan")
def _():
    groups = ["vehicle", "two_wheeler", "pedestrian"]
    src, sub = SynthSource(), SynthSubject()
    fp = []
    for sc in src.scenes()[:5]:
        sub.reset()
        for ref in src.frames(sc)[:4]:
            s = src.load(ref)
            for g in s.reference.values():                                    # per frame: vehicle 0 stands (0.2 m/s), vehicle 1 moves slowly (0.8 m/s)
                veh = [b for b in g["meta"]["boxes"] if b["group"] == "vehicle"]
                veh[0]["vel"], veh[1]["vel"] = [0.2, 0.0], [0.8, 0.0]
            p = sub.predict(s)
            for o in p.outputs["dt"].values():                                # the network says half of every velocity, and a little sideways
                o["dets"] = [dict(d, vx=d["vx"] * 0.5, vy=d["vy"] * 0.5 + 0.1) for d in o["dets"]]
            fp.append((s, p))
    m = BevDetMetric(groups, "det", "dets")
    counts = [m.frame(s, p) for s, p in fp]
    meter = DetMeter(classes=groups)
    got = m.reduce(counts)
    for dt in DTS:                                                            # a plain-loop count of the pairs, per class
        dk = dt_key(dt)
        for cls in groups:
            frames = [([d for d in meter.keep(p.outputs["dt"][dt]["dets"]) if d["group"] == cls],
                       [b for b in meter.keep(gt_boxes(s.reference[dt], groups)) if b["group"] == cls]) for s, p in fp]
            rows, _ = metrics_match(frames, 2.0)
            acc = {"m": [], "s": []}
            for _score, ok, d, gt in rows:
                if ok:
                    speed = math.hypot(gt["vx"], gt["vy"])
                    acc["m" if speed > 0.5 else "s"].append((math.hypot(d["vx"] - gt["vx"], d["vy"] - gt["vy"]), speed, math.hypot(d["vx"], d["vy"])))
            for tag, name in (("m", "moving"), ("s", "static")):
                a = acc[tag]
                n = len(a)
                assert got[f"{dk}/{cls}/all/n_tp_{name}"] == n
                for key, want in ((f"ave_{name}", sum(x[0] for x in a) / n if n else float("nan")), (f"ave0_{name}", sum(x[1] for x in a) / n if n else float("nan"))):
                    assert same(got[f"{dk}/{cls}/all/{key}"], want, 1e-9), (dk, cls, key)
            slow = sum(1 for x in acc["m"] if x[2] < 0.5)
            assert same(got[f"{dk}/{cls}/all/ave_slow_moving"], slow / len(acc["m"]) if acc["m"] else float("nan"), 1e-9)
            if cls == "vehicle":
                assert len(acc["m"]) > 10 and len(acc["s"]) >= 10 and slow >= 5, (len(acc["m"]), len(acc["s"]), slow)    # the planted world has all three
                assert same(got[f"{dk}/vehicle/all/ave_skill_moving"], 1 - sum(x[0] for x in acc["m"]) / sum(x[1] for x in acc["m"]), 1e-9)
                n_all = len(acc["m"]) + len(acc["s"])                           # the groups pooled = the class's ave (the same pairs: matches at 2 m among all boxes)
                pooled = (got[f"{dk}/vehicle/all/ave_moving"] * len(acc["m"]) + got[f"{dk}/vehicle/all/ave_static"] * len(acc["s"])) / n_all
                assert same(pooled, got[f"{dk}/vehicle/all/ave"], 1e-9), (pooled, got[f"{dk}/vehicle/all/ave"])
    k = f"{dt_key(0.0)}/vehicle/all/n_tp_moving"                              # scenes add up (the counts are additive per frame)
    per_scene = [m.reduce([c for c, (s, _) in zip(counts, fp) if s.ref.scene == sc], keys=[k])[k] for sc in src.scenes()[:5]]
    assert sum(per_scene) == got[k]
    sel = m.reduce(counts, keys=[f"{dt_key(0.0)}/vehicle/all/ave_skill_moving", f"{dt_key(0.0)}/vehicle/all/ap@2"])      # selective keys include the split
    assert same(sel[f"{dt_key(0.0)}/vehicle/all/ave_skill_moving"], got[f"{dt_key(0.0)}/vehicle/all/ave_skill_moving"]) and m.direction("dt+0.00/vehicle/all/ave_moving") == "low" \
        and m.direction("dt+0.00/vehicle/all/ave_skill_moving") == "high" and m.direction("dt+0.00/vehicle/all/ap@2") == "high"
    old = [{kk: v for kk, v in c.items() if not kk.endswith("|vel")} for c in counts]                  # a stored run from before the split
    pre = m.reduce(old)                                                      # no counts: the keys are left out of the full table, nan when asked for
    assert f"{dt_key(0.0)}/vehicle/all/ave_skill_moving" not in pre and same(pre[f"{dt_key(0.0)}/vehicle/all/ap@2"], got[f"{dt_key(0.0)}/vehicle/all/ap@2"])
    assert math.isnan(m.reduce(old, keys=[f"{dt_key(0.0)}/vehicle/all/ave_skill_moving"])[f"{dt_key(0.0)}/vehicle/all/ave_skill_moving"])
    assert f"{dt_key(0.0)}/vehicle/all/ave_moving" not in m.reduce(old[:10] + counts[10:]), "frames with and without the counts must not be mixed"


@check("to_legacy: the table of the suite becomes the flat names of the former metrics.json; headline_table prints")
def _():
    suite = make_suite(tempfile.mkdtemp())
    st = suite.quant.run(scenes=["bev-00"], name="a", progress=0)
    tab = suite.quant.table(st)
    leg = bev_suite.to_legacy(tab)
    assert "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0" in leg and "seg_out/seg/dt+0.50/walkway/known/0-20/iou@1" in leg
    assert "det/dt+0.00/vehicle/all/ap@2" in leg and "detr/dt+0.00/mean/all/mAP" in leg and "val_loss/total" in leg
    assert not any(k.startswith(("seg.", "loss.")) for k in leg)
    assert "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0" in bev_suite.headline_table(leg)
    assert bev_suite.tag_of(True, True, "zero_radar") == "_warm_train_ctl_zero_radar" and bev_suite.tag_of(False, False, "") == ""
    # the card's "better" map (read by autoresearch/tools/decide.py): every number but the counts, each by its metric's own direction
    bt = bev_suite.better_of(tab, suite.task.metrics())
    assert set(bt) <= set(leg) and all(v in ("lower", "higher") for v in bt.values())
    assert set(leg) - set(bt) == {k for k in leg if k.rsplit("/", 1)[-1] == "n" or k.rsplit("/", 1)[-1].startswith("n_")}, "only counts undeclared"
    assert bt["seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0"] == "higher" and bt["det/dt+0.00/vehicle/all/ap@2"] == "higher"
    assert bt["det/dt+0.00/vehicle/all/ate"] == bt["detr/dt+0.00/vehicle/all/ave_moving"] == bt["val_loss/total"] == "lower"
    assert bt["seg_fov/seg/dt+0.00/lane_divider:zc/all/all/field_mae"] == "lower" and bt["det/dt+0.00/vehicle/all/ave_skill_moving"] == "higher"
    assert bev_suite.better_of(tab, suite.task.metrics(), "prior")["prior_fov/seg/dt+0.00/drivable_area/all/all/iou@0"] == "higher"
    assert bev_suite.ms_better({"ms/median_ms": 3.0, "ms/idle": True, "info/step": 5}) == {"ms/median_ms": "lower"}
    assert bt["val_loss/group.areas"] == "lower", "every loss term is lower-is-better, not only the total"
    # the fallback of autoresearch for cards without the map (project.json lower_is_better, matched by the key's last part) reads every
    # declared number the same way
    sys.path.insert(0, os.path.join(bev_suite.ROOT, "..", "autoresearch", "tools"))
    import decide as DC
    import card as CARD
    import experiment as EXP
    assert not [k for k in bt if (bt[k] == "lower") != DC.named_lower(k, bev_suite.ROOT)], "project.json lower_is_better disagrees with the evaluator"
    # the prior-map baseline adds its map numbers only (its det / loss numbers would carry the model's names and overwrite them)
    pflat, pbt = bev_suite.prior_numbers(tab, suite.task.metrics())
    assert pflat and all(k.startswith("prior_") for k in [*pflat, *pbt]) and set(pbt) <= set(pflat)
    # the contract card: valid, finite numbers only, read by decide.py with its directions
    run = tempfile.mkdtemp()
    json.dump({"seed": 3}, open(os.path.join(run, "config.json"), "w"))
    flat = dict(leg, **{"info/step": 7, "ms/idle": True, "ms/median_ms": 4.0, "det/dt+0.00/vehicle/all/nanx": float("nan")})
    card = bev_suite.contract_card(run, "val", "_warm", flat, dict(bt, **{"ms/median_ms": "lower", "det/dt+0.00/vehicle/all/nanx": "lower"}),
                                   24, "sig1")
    assert CARD.validate(card) == [], CARD.validate(card)
    assert card["commit"] == "+?" and card["seed"] == 3 and card["name"] == os.path.basename(run) + "_warm" and card["protocol"] == {"gt_signature": "sig1"}
    assert "info/step" not in card["metrics"] and "ms/idle" not in card["metrics"] and "det/dt+0.00/vehicle/all/nanx" not in card["better"]
    path = os.path.join(run, "card_warm.json")
    json.dump(card, open(path, "w"))
    m, full, _ = EXP.card_ref(path, run)
    assert full is not None and DC.card_lower("det/dt+0.00/vehicle/all/ate", [(m, full, "a")]) is True
    assert DC.card_lower("seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0", [(m, full, "a")]) is False


@check("provenance: the card carries the reproducibility block (GT, evaluator hash, frame hash, model, seed); a collector that cannot run leaves a marked gap, not a lost evaluation")
def _():
    import hashlib
    from types import SimpleNamespace
    sys.path.insert(0, os.path.join(ROOT, "..", "autoresearch", "tools"))
    import card as CARD
    run = tempfile.mkdtemp()
    json.dump({"seed": 3}, open(os.path.join(run, "config.json"), "w"))
    json.dump({"gt_signature": "0.5_4_15efbde2d1"}, open(os.path.join(run, "gt.json"), "w"))
    with open(os.path.join(run, "ckpt_00000007.pt"), "wb") as f:
        f.write(b"checkpoint bytes")
    keys = ["scene-0001/1", "scene-0001/0"]
    store = SimpleNamespace(frames=[{"key": k} for k in keys], scene_names=lambda: ["scene-0001"])
    a, e = SimpleNamespace(run=run, split="val", control="", ckpt_dir=None), SimpleNamespace(n=2, warm=False, prior=0)
    subj = {"checkpoint": "ckpt_00000007.pt", "step": 7}
    pv = bev_suite.provenance_block(a, e, store, subj, "0.5_4_15efbde2d1", "", "reset")
    assert "error" not in pv, pv
    assert pv["subject"]["run"] == os.path.basename(run) and pv["subject"]["step"] == 7 and pv["subject"]["ckpt"] == "ckpt_00000007.pt"
    assert pv["subject"]["ckpt_hash"] not in (None, "unknown") and pv["subject"]["config_hash"] not in (None, "unknown"), pv["subject"]
    assert pv["subject"]["train_gt"]["signature"] == "15efbde2d1", pv["subject"]
    assert pv["data"]["n"] == 2 and pv["data"]["split"] == "val"
    use = pv["data"]["use"]                                                    # which data was used, readable without code (2026-11-01)
    assert use["scene_names"] == ["scene-0001"] and use["scenes_evaluated"] == 1 and use["frames_evaluated"] == 2 and "nuScenes official split 'val'" in use["split_rule"], use
    assert pv["data"]["frame_hash"] == hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()[:12], "the frame hash is over the sorted keys"
    assert pv["gt"]["signature"] == "15efbde2d1" and pv["gt"]["key"] == "0.5_4_15efbde2d1"
    assert pv["evaluator"]["paths"] == ["bev_fusion/evaluation", "autoresearch/evalsuite"] and pv["evaluator"]["code_hash"] not in (None, "unknown"), pv["evaluator"]
    assert pv["evaluator"]["settings"]["n"] == 2 and pv["evaluator"]["settings"]["weights"] == "EMA" and pv["seed"] == 3
    name = "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0"
    card = bev_suite.contract_card(run, "val", "", {name: 0.5}, {name: "higher"}, 2, "0.5_4_15efbde2d1")
    card["provenance"] = pv
    assert CARD.validate(card) == [], CARD.validate(card)                      # the card with the block stays valid
    moved = tempfile.mkdtemp()                                                 # a checkpoint that was moved is found through --ckpt-dir
    os.replace(os.path.join(run, "ckpt_00000007.pt"), os.path.join(moved, "ckpt_00000007.pt"))
    a.ckpt_dir = moved
    assert bev_suite.provenance_block(a, e, store, subj, "0.5_4_15efbde2d1", "", "reset")["subject"]["ckpt_hash"] == pv["subject"]["ckpt_hash"]
    keep, bev_suite.PROVENANCE_DIR = bev_suite.PROVENANCE_DIR, os.path.join(run, "nowhere")
    try:                                                                       # no collector: a marked gap, the evaluation goes on
        gap = bev_suite.provenance_block(a, e, store, subj, "0.5_4_15efbde2d1", "", "reset")
    finally:
        bev_suite.PROVENANCE_DIR = keep
    assert gap.get("error") and gap["schema"] == 1, gap
    # the START of the evaluation: hashes taken before the first frame; a change while it runs is marked, not hidden; the card stays serialisable
    import contextlib
    import io
    before = list(sys.path)
    suite0 = SimpleNamespace(subject=SimpleNamespace(ckpt=os.path.join(moved, "ckpt_00000007.pt")))
    start = bev_suite.provenance_start(suite0)
    assert sys.path == before, "loading the collector must not leave its folder first on sys.path"
    assert start["tree"] and start["ckpt"] and "error" not in start, start
    ok = bev_suite.provenance_block(a, e, store, subj, "0.5_4_15efbde2d1", "", "reset", start=start)
    assert "changed_during_run" not in ok["evaluator"] and "changed_during_run" not in ok["subject"], ok
    json.dumps(dict(card, provenance=ok))
    assert ok["evaluator"]["settings"]["store"] == {"created": None, "resumed": None, "commit": None} and "device" in ok["evaluator"]["settings"]
    edited = bev_suite.provenance_block(a, e, store, subj, "0.5_4_15efbde2d1", "", "reset", start=dict(start, tree="000000000000"))
    assert edited["evaluator"]["dirty"] == "unknown" and edited["evaluator"]["changed_during_run"], edited["evaluator"]
    assert edited["evaluator"]["code_hash_at_start"] == "000000000000", edited["evaluator"]
    with open(os.path.join(moved, "ckpt_00000007.pt"), "ab") as f:                     # the checkpoint is rewritten while the evaluation runs
        f.write(b"more")
    swapped = bev_suite.provenance_block(a, e, store, subj, "0.5_4_15efbde2d1", "", "reset", start=start)
    assert swapped["subject"]["ckpt_hash"] == "unknown" and swapped["subject"]["changed_during_run"], swapped["subject"]
    os.remove(os.path.join(moved, "ckpt_00000007.pt"))                                  # ... or pruned: the hash of the start stands
    gone = bev_suite.provenance_block(a, e, store, subj, "0.5_4_15efbde2d1", "", "reset", start=start)
    assert gone["subject"]["ckpt_hash"] == start["ckpt"] and gone["subject"]["gone_at_end"], gone["subject"]
    buf = io.StringIO()                                                                 # a collector that cannot be loaded is said loudly: at the start and at the end
    bev_suite.PROVENANCE_DIR = os.path.join(run, "nowhere")
    try:
        with contextlib.redirect_stdout(buf):
            bad0 = bev_suite.provenance_start(suite0)
            bad1 = bev_suite.provenance_block(a, e, store, subj, "g", "", "reset", start=bad0)
    finally:
        bev_suite.PROVENANCE_DIR = keep
    assert buf.getvalue().count("WARNING: provenance") == 2 and bad0.get("error") and bad1.get("error"), buf.getvalue()


@check("views: overview / outputs / dashboard / stages (with taps) render pages with the error colours; camera panel carries the projected GT")
def _():
    suite = make_suite(tempfile.mkdtemp(), shift=1)                 # a 1-cell shift: false positives and misses on every boundary
    src, sub = suite.source, suite.subject
    s = src.load(src.frames("bev-01")[2])
    p = sub.predict(s)
    p.taps = {"lift": np.random.default_rng(0).random((8 * 4, 120, 104)).astype(np.float32), "cam_temporal": np.random.default_rng(1).random((6, 120, 104)),
              "radar_raster": np.random.default_rng(2).random((3, 120, 104)), "trunk": np.random.default_rng(3).random((8, 120, 104)),
              "camera_visibility": (np.arange(120)[:, None] < 100) * np.ones((1, 104))}
    s.display["camera"] = np.full((900, 1600, 3), 90, np.uint8)
    s.display["camera_model"] = np.full((256, 704, 3), 90, np.uint8)
    task = suite.task
    task.z_levels = tuple(range(8))
    ctx = suite.qual.context(s, p)
    for v in ("overview", "outputs", "dashboard", "stages"):
        page = suite.qual.page(s, p, v, ctx)
        img = figure.render_page(page)
        assert page.panels and img.min() != img.max(), v
    ov = task.views()["overview"].render(s, p, ctx)
    assert ov.panels[0].kind == "image" and len(ov.panels[0].shapes) > 10, "GT boxes and map lines are projected into the camera image"
    err = ov.panels[3].data                                          # the drivable-area error map: FP red, FN blue exist for a noisy prediction
    assert (np.abs(err - np.array(figure.FP_COLOR)).max(-1) < 1e-6).any() and (np.abs(err - np.array(figure.FN_COLOR)).max(-1) < 1e-6).any()
    assert "drivable" in ov.panels[3].title and "missed" in ov.panels[3].title


@check("BEVTask.checks: a clean frame has no failure, the planted corrupted frame is flagged")
def _():
    suite = make_suite(tempfile.mkdtemp())
    src, sub = suite.source, suite.subject
    good = src.load(src.frames("bev-01")[1])
    flags_good = suite.task.checks(good, sub.predict(good))
    bad = src.load(src.frames("bev-02")[3])
    flags_bad = suite.task.checks(bad, sub.predict(bad))
    assert not any(f.status == "fail" for f in flags_good)
    assert len(flags_bad) > len(flags_good) and any("drivable" in f.name or "boxes" in f.name for f in flags_bad), [f.name for f in flags_bad]


@check("the whole suite on the synthetic BEV world: contract checker, run, outliers find the planted frames, report, compare, legacy export")
def _():
    with tempfile.TemporaryDirectory() as d:
        suite = make_suite(d)
        res = suite.check()
        assert all(r["ok"] for r in res), [r for r in res if not r["ok"]]
        det_rec, detr_rec = {}, {}

        def keep(sample, pred):
            for dt, o in pred.outputs["dt"].items():
                gb = gt_boxes(sample.reference[dt], suite.task.groups)
                det_rec.setdefault(sample.ref.scene, {}).setdefault(dt, []).append((list(o["dets"]), gb))
                detr_rec.setdefault(sample.ref.scene, {}).setdefault(dt, []).append((list(o["detr"]), gb))
        st = suite.quant.run(name="a", progress=0, on_frame=keep)
        assert len(st.frames) == 48 and st.manifest["finished"]
        osr = suite.quant.outliers(st, k=4)
        planted = {f"{s}/{i:03d}" for s, i in PLANTED}
        iou = [r["label"] for r in osr["frames"] if r["metric"] == "seg.drivable_iou"]
        assert planted <= set(iou[:4]), iou
        assert [r["scene"] for r in osr["scenes"] if r["metric"] == bev_task_head()][:2] and \
            {"bev-02", "bev-04"} <= {r["scene"] for r in osr["scenes"] if r["metric"] == bev_task_head()}
        rep = suite.quant.report(st, render=2)
        assert os.path.exists(rep)
        tr = suite.quant.trace(st, "bev-02/003", ("overview",))
        assert abs(tr.scores["seg.drivable_iou"] - st.frame("bev-02/003")["scores"]["seg.drivable_iou"]) < 1e-12
        # compare: a worse network is found (store-based)
        make_suite(d, noise=0.45, shift=2, name="worse").quant.run(name="b", progress=0)
        cmp = suite.quant.compare("a", "b", boot=200)["metrics"]
        assert cmp[bev_task_head()]["diff"] < -0.02 and cmp[bev_task_head()]["hi"] < 0
        # the legacy export: scene_records.pkl written from the store; its per-scene headline counts equal the store's sums
        os.makedirs(os.path.join(d, "runs", "a", "eval"), exist_ok=True)
        m0 = suite.task.metrics()[0]
        bev_legacy.write_scene_records(os.path.join(d, "runs", "a", "eval", "scene_records.pkl"), m0, st, det_rec, detr_rec)
        import pickle
        rec = pickle.load(open(os.path.join(d, "runs", "a", "eval", "scene_records.pkl"), "rb"))
        assert set(rec["seg"]) == set(suite.source.scenes()) and set(rec["det"]) == set(rec["seg"])
        for layer in bev_legacy.HEADLINE_SEG:
            assert layer[0] in rec["seg"]["bev-01"], layer
        sc = bev_legacy.scene_counts(rec, "drivable_area")["bev-01"]
        assert sc.shape == (2, 4) and sc[0, 0] > 0
        # AP of the whole run: the suite's rows-based AP = the former global greedy matching (legacy det_ap) on the same boxes
        frames_by_scene = {s: r[0.0] for s, r in rec["det"].items()}
        legacy_ap = bev_legacy.det_ap(frames_by_scene, list(frames_by_scene), "vehicle", 2.0)
        assert same(suite.quant.table(st)["det.dt+0.00/vehicle/all/ap@2"], legacy_ap)
        # and the legacy paired comparison of two stored runs agrees with the suite's point estimates
        os.makedirs(os.path.join(d, "runs", "b", "eval"), exist_ok=True)
        stb = RunStoreOpen(os.path.join(d, "quant", "b"))
        det_b = {}
        sub_b, src_b = SynthSubject(noise=0.45, shift=2, name="worse"), SynthSource()
        for sc_ in src_b.scenes():
            sub_b.reset()
            for ref in src_b.frames(sc_):
                s_ = src_b.load(ref)
                p_ = sub_b.predict(s_)
                for dt, o in p_.outputs["dt"].items():
                    det_b.setdefault(sc_, {}).setdefault(dt, []).append((list(o["dets"]), gt_boxes(s_.reference[dt], suite.task.groups)))
        bev_legacy.write_scene_records(os.path.join(d, "runs", "b", "eval", "scene_records.pkl"), m0, stb, det_b)
        old = os.getcwd()
        os.chdir(d)
        try:
            leg, _ = bev_legacy.compare("runs/a", "runs/b", n_boot=50)
        finally:
            os.chdir(old)
        assert same(leg["drivable_area iou@0"][3], cmp[bev_task_head()]["a"], 1e-9) and same(leg["drivable_area iou@0"][4], cmp[bev_task_head()]["b"], 1e-9)


def bev_task_head():
    return "seg.fov/dt+0.00/drivable_area/all/all/iou@0"


def RunStoreOpen(path):
    from evalsuite.store import RunStore
    return RunStore.open(path)


@check("bev_probe: the chain's names resolve lazily; helpers agree with the adapter's; gt_boxes takes one argument as before")
def _():
    from evaluation import bev_probe as P
    assert P.ROOT == A.ROOT and P.layer_index(SPEC)["vehicle"][1] == "object"
    for name in ("forward_loss", "make_datasets", "make_loader", "val_subset", "TrainConfig", "NUSC_GROUPS", "decode", "collate_cached"):
        assert getattr(P, name) is not None, name
    g = synth_gt(1)
    assert len(P.gt_boxes(g)) == 4 and P.masks(g)["fov"].any() and set(P.masks(g)) == {"fov", "seen", "obj_fov", "out"}
    try:
        P.nonexistent
        raise AssertionError("must raise")
    except AttributeError:
        pass
    import torch                                     # the camera controls take EVERY camera input away (t10 prob2 / ctx2, t7 lane_img)
    keys = ("prob", "ctx", "prob2", "ctx2", "img", "lane_img")
    batch = {"tokens": ["a", "b"], **{k: torch.arange(2.0).view(2, 1) + 1 for k in keys}}
    P.perturb(batch, "zero_cam")
    assert all(float(batch[k].abs().sum()) == 0 for k in keys), {k: batch[k] for k in keys}
    batch = {"tokens": ["a", "b"], **{k: torch.arange(2.0).view(2, 1) for k in keys}}
    P.perturb(batch, "shuffle_cam")
    assert all(batch[k].view(-1).tolist() == [1.0, 0.0] for k in keys), {k: batch[k] for k in keys}
    # the same on one dataset item (the path of `bev_suite.py eval --control`): every frame key and the lane image
    def item(v):
        return {"frames": [{k: torch.full((2,), float(v)) for k in A.CAMERA_FRAME_KEYS} | {"token": "t"} for _ in range(3)],
                "lane_img": torch.full((3, 2), v, dtype=torch.uint8)}
    it, other = item(5), item(9)
    A.perturb_camera(it, other, "zero_cam")
    assert all(float(f[k].abs().sum()) == 0 for f in it["frames"] for k in A.CAMERA_FRAME_KEYS) and int(it["lane_img"].sum()) == 0
    it = item(5)
    A.perturb_camera(it, other, "shuffle_cam")
    assert all(float(f[k][0]) == 9 for f in it["frames"] for k in A.CAMERA_FRAME_KEYS) and int(it["lane_img"][0, 0]) == 9
    try:
        A.perturb_camera({"frames": [{"prob3": torch.ones(1)}]}, {"frames": [{"prob3": torch.ones(1)}]}, "zero_cam")
        raise AssertionError("an unknown camera key must be refused")
    except ValueError:
        pass


@check("BEVSubject.ops: the GMACs of the cached chain, counted by torch, on a one-convolution model with a known answer; the note names what is not counted")
def _():
    import torch
    from types import SimpleNamespace
    Sub, _prior = A._make_subject_classes()

    class OneConv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

        def forward(self, batch, dts=()):
            return [self.conv(batch) for _ in dts]               # one pass per prediction time, like the chain
    sub = object.__new__(Sub)                                    # no checkpoint: only what ops() reads
    sub.device, sub.model = "cpu", OneConv().eval()
    seen = []
    sub.ns = SimpleNamespace(T=SimpleNamespace(to_device=lambda x, device: seen.append(device) or x))
    r = sub.ops(SimpleNamespace(inputs=torch.zeros(1, 3, 8, 8)))
    assert abs(r["gmacs"] - 3 * 4 * 3 * 3 * 3 * 8 * 8 / 1e9) < 1e-12 and r["method"] == "measured", r   # 6912 MACs of the 3 -> 4 channel 3x3 convolution on 8 x 8, once per dt
    assert seen == ["cpu"], "the sample goes to the subject's device first"
    assert "pre-trained camera heads" in r["note"] and "dts -0.5 / 0 / +0.5" in r["note"], r["note"]


@check("eval: a stored run of another or an unrecorded GT is refused (a resume would label old counts with the new GT); "
       "--force-resume only for an unrecorded one; a legacy pickle that fits the store is kept (review 2026-10-31 pass 2)")
def _():
    import tempfile
    from evaluation import bev_suite as S
    with tempfile.TemporaryDirectory() as d:
        assert S.gt_conflict(os.path.join(d, "none"), "0.5_4_new", False) == (None, "0.5_4_new"), "no stored run: no conflict"
        for name, src in (("old", {"gt_signature": "0.5_4_old"}), ("unrec", {}), ("same", {"gt_signature": "0.5_4_new"})):
            os.makedirs(os.path.join(d, name))
            with open(os.path.join(d, name, "manifest.json"), "w") as f:
                json.dump({"source": src}, f)
        assert S.gt_conflict(os.path.join(d, "same"), "0.5_4_new", False) == (None, "0.5_4_new")
        msg, _ = S.gt_conflict(os.path.join(d, "old"), "0.5_4_new", True)
        assert msg and "0.5_4_old" in msg, "another recorded GT is refused even with --force-resume"
        msg, _ = S.gt_conflict(os.path.join(d, "unrec"), "0.5_4_new", False)
        assert msg and "--force-resume" in msg
        assert S.gt_conflict(os.path.join(d, "unrec"), "0.5_4_new", True) == (None, "mixed:unrecorded+0.5_4_new")
        import pickle
        p = os.path.join(d, "rec.pkl")
        with open(p, "wb") as f:
            pickle.dump({"meta": {"frames": ["a", "b"], "gt_signature": "0.5_4_new"}}, f)
        assert S.pickle_meta(p) == {"frames": ["a", "b"], "gt_signature": "0.5_4_new"} and S.pickle_meta(os.path.join(d, "x")) is None


def refused(fn) -> str:
    """the message of the SystemExit that fn must raise ('' if it returned)"""
    try:
        fn()
    except SystemExit as e:
        return str(e)
    return ""


@check("compare: refused on different GT signatures (evaluation / training) or different frames of legacy records, warned when not "
       "recorded, --force; the legacy record's name carries n (review 2026-10-31 F3, F4)")
def _():
    import contextlib
    import io
    # the GT rule alone
    same_sig = {"evaluation GT": ("0.5_4_aaa", "0.5_4_aaa"), "training GT": ("0.5_4_aaa", "0.5_4_aaa")}
    assert bev_suite.gt_check("a", "b", same_sig) == []
    w = bev_suite.gt_check("a", "b", {"evaluation GT": ("0.5_4_aaa", None), "training GT": (None, None)})
    assert len(w) == 2 and "not recorded for b" in w[0] and "a and b" in w[1], w
    assert "different GT" in refused(lambda: bev_suite.gt_check("a", "b", {"evaluation GT": ("0.5_4_aaa", "0.5_4_bbb")}))
    assert "different GT" in refused(lambda: bev_suite.gt_check("a", "b", {"training GT": ("0.5_4_aaa", "0.5_4_bbb")}))
    assert "--force" in bev_suite.gt_check("a", "b", {"evaluation GT": ("0.5_4_aaa", "0.5_4_bbb")}, force=True)[0]
    # legacy records through the command line
    with tempfile.TemporaryDirectory() as d:
        suite = make_suite(d)
        det_rec = {}

        def keep(sample, pred):
            for dt, o in pred.outputs["dt"].items():
                det_rec.setdefault(sample.ref.scene, {}).setdefault(dt, []).append((list(o["dets"]), gt_boxes(sample.reference[dt], suite.task.groups)))
        st = suite.quant.run(name="x", progress=0, on_frame=keep)
        m0, keys = suite.task.metrics()[0], [fr["key"] for fr in st.frames]
        assert len(keys) == 48

        def run(name, frames=None, sig="0.5_4_aaa", n=48, train_sig=None, old=False, metrics=None):
            r = os.path.join(d, name)
            os.makedirs(os.path.join(r, "eval"), exist_ok=True)
            pkl = "scene_records.pkl" if old else f"scene_records_n{n}.pkl"
            bev_legacy.write_scene_records(os.path.join(r, "eval", pkl), m0, st, det_rec,
                                           meta=None if old else {"frames": frames if frames is not None else keys, "gt_signature": sig})
            if metrics is not None:
                json.dump(metrics, open(os.path.join(r, "eval", "metrics.json"), "w"))
            if train_sig:
                json.dump({"gt_signature": train_sig}, open(os.path.join(r, "gt.json"), "w"))
            return r

        def cmp(*args):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = bev_suite.cmd_compare(list(args) + ["--boot", "20"])
            return rc, out.getvalue()
        a, b = run("a", train_sig="0.5_4_aaa"), run("b", train_sig="0.5_4_aaa")
        rc, out = cmp(a, b, "--n", "48")
        assert rc == 0 and "b-a +0.0000" in out and "WARNING" not in out, out            # the same records: zero difference, no warning
        assert "no evaluation" in refused(lambda: cmp(a, b, "--n", "24"))              # another n: another file name, never the newest eval
        c = run("c", frames=keys[:-1] + ["another-frame"])
        assert "different frame sets (47 common" in refused(lambda: cmp(a, c, "--n", "48"))
        e = run("e", sig="0.5_4_bbb", train_sig="0.5_4_aaa")
        assert "evaluation GT 0.5_4_aaa" in refused(lambda: cmp(a, e, "--n", "48"))
        rc, out = cmp(a, e, "--n", "48", "--force")
        assert rc == 0 and "WARNING (--force)" in out
        f_ = run("f", train_sig="0.5_4_ccc")
        assert "training GT 0.5_4_aaa" in refused(lambda: cmp(a, f_, "--n", "48"))
        rc, out = cmp(a, run("g"), "--n", "48")                                         # training GT not recorded on one side: a warning
        assert rc == 0 and "WARNING: training GT signature not recorded for" in out, out
        # the F4 case: a suite-written record whose store names its frames against a pre-suite record (frames unknown) -> refused
        h = run("h", old=True, metrics={"suite_run": os.path.relpath(st.path, os.path.join(d, "h")), "metrics": {"info/n_samples": 48}})
        p_ = run("p", old=True, metrics={"metrics": {"info/n_samples": 48}})
        assert "are not recorded" in refused(lambda: cmp(h, p_, "--n", "48"))
        assert "evaluated before the suite" in refused(lambda: cmp(a, p_, "--n", "48"))     # a new record against a pre-suite one
        q = run("q", old=True, metrics={"metrics": {"info/n_samples": 1000}})
        assert "frame counts 48 / 1000" in refused(lambda: cmp(p_, q))                 # two pre-suite records: equal counts needed
        rc, out = cmp(p_, run("r", old=True, metrics={"metrics": {"info/n_samples": 48}}), "--n", "48")
        assert rc == 0 and "same count 48 assumed" in out, out


# ------------------------------------------------------------------------------------------------ optional: real data
def golden():
    """the suite's numbers on 24 real val frames = the numbers of the former evaluate_run.py on the same frames (CPU, deterministic)"""
    fixture = os.path.join(ROOT, "evaluation", "golden", "t1_r4_soa_n24_old.json")
    run, scratch = "runs/t1_r4_soa", "runs/_golden_t1_r4_soa"
    assert os.path.exists(os.path.join(ROOT, run, "config.json")), f"{run} is needed for the golden test"
    os.chdir(ROOT)
    os.makedirs(scratch, exist_ok=True)
    cfg = json.load(open(os.path.join(run, "config.json")))
    frozen = json.load(open("data/scenes_parts1_6.json"))
    cfg["val_scenes"], cfg["train_scenes"] = frozen["val"], frozen["train"]
    json.dump(cfg, open(os.path.join(scratch, "config.json"), "w"), indent=1)
    if not os.path.exists(os.path.join(scratch, "code")):
        import shutil
        shutil.copytree(os.path.join(run, "code"), os.path.join(scratch, "code"))
    ckpt_dir = run if any(f.startswith("ckpt_") for f in os.listdir(run)) else os.path.join("data", "checkpoints", os.path.basename(run))     # data/checkpoints: link to D: (data/NuScenes/tools/link_storage.sh)
    old_store = os.path.join(scratch, "eval", "suite")
    if os.path.isdir(old_store):
        import shutil
        shutil.rmtree(old_store)
    rc = bev_suite.main([scratch, "--ckpt-dir", ckpt_dir, "--device", "cpu", "--workers", "0", "eval", "--n", "24"])
    assert rc == 0
    prov = json.load(open(os.path.join(scratch, "eval", "card.json"))).get("provenance") or {}      # the wiring of cmd_eval (a card written by hand would not show it)
    assert "error" not in prov and prov["data"]["n"] == 24 and prov["subject"]["ckpt_hash"] not in (None, "unknown"), prov
    assert prov["evaluator"]["code_hash"] and not prov["evaluator"].get("changed_during_run"), prov["evaluator"]
    new = json.load(open(os.path.join(scratch, "eval", "metrics.json")))["metrics"]
    old = json.load(open(fixture))["metrics"]
    missing = [k for k in old if k not in new]
    bad = [(k, old[k], new[k]) for k in old if k in new and isinstance(old[k], (int, float)) and not same(old[k], new[k], 1e-9)]
    assert not missing and not bad, (len(missing), bad[:5])
    print(f"      golden: {len(old)} numbers of the former evaluate_run.py reproduced (relative difference <= 1e-9), {len(new) - len(old)} more keys")


def real():
    """the adapter on a real run: contract checker, every view of a frame, a quantitative run with the state carried, outliers, report"""
    os.chdir(ROOT)
    from evaluation.bev_adapter import build
    with tempfile.TemporaryDirectory() as d:
        src, subj, task = build("runs/t1_r4_soa", device="cuda", workers=0)
        suite = EvalSuite(src, subj, task, out=d)
        res = suite.check()
        assert all(r["ok"] for r in res), [r for r in res if not r["ok"]]
        q = suite.qual.frame(src.frames(src.scenes()[0])[10].label, ("overview", "outputs", "stages", "dashboard"), history="none")
        assert all(os.path.exists(p) for p in q.images.values()) and len(q.images) == 4
        st = suite.quant.run(n_scenes=2, max_frames=6, name="r", progress=0)
        assert len(st.frames) == 12 and suite.quant.outliers(st, k=2)["frames"]
        assert os.path.exists(suite.quant.report(st, render=1))
        v = suite.qual.videos(1, max_frames=4)[0]
        assert v["video"]["frames"] == 4
        suite.close()


def main() -> int:
    ap_golden, ap_real = "--golden" in sys.argv, "--real" in sys.argv
    if ap_golden or ap_real:                                                 # real data: each in a process of its own
        todo = ([("golden: the numbers of the former evaluate_run.py on 24 real frames", golden)] if ap_golden else []) + \
            ([("real run: adapter contract, views, quantitative run, outliers, report, video", real)] if ap_real else [])
    else:
        init_synthetic()
        todo = list(CHECKS)
    t0, bad = time.time(), 0
    for name, fn in todo:
        t = time.time()
        try:
            fn()
            print(f"ok    {name}  ({time.time() - t:.1f} s)")
        except Exception:                                                  # noqa: BLE001
            bad += 1
            print(f"FAIL  {name}\n{traceback.format_exc()}")
    print(f"bev selftest: {len(todo) - bad}/{len(todo)} checks passed in {time.time() - t0:.0f} s")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
