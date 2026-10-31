"""The BEV Task of the evaluation suite (evalsuite): what the outputs of the chain mean.

    BevSegMetric   maps: tolerant IoU per layer / range bin / dt / subset and line zero-crossing F1 (metrics.SegMeter)
    BevDetMetric   boxes: nuScenes-style AP, TP errors per class / bin / dt (metrics.DetMeter rows; 'det' = plain decode,
                   'detr' = boxes refined by S5c)
    BevLossMetric  the training loss of the frame (means over frames; per-frame totals rank frames)
    BEVTask        the three plus the views of bev_views.py and the plausibility checks of a frame

Every Metric stores ADDITIVE per-frame counts: for the maps the SegMeter counts of the three masks (fov = camera cells, seen =
camera or radar cells, obj = camera cells without ignored boxes) packed into arrays; for the boxes the true-positive flags of
every detection and the GT counts (DetMeter.frame_rows). Any set of frames (all, a scene, all but one scene, a bootstrap
resample) reduces to the same numbers the direct meters give. Flat number names: seg: `<mask>/dt+0.00/<layer>/<subset>/<bin>/<metric>`,
det: `dt+0.00/<class>/<bin>/<metric>`; with the metric name in front they are the keys of `suite.quant.table`.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from evalsuite import Flag, Metric, Prediction, Sample, Task, View
from evalsuite.qualitative import DebugView
from evaluation.metrics import DetMeter, SegMeter, bin_label, dt_key, in_front
from evaluation.bev_views import (DashboardView, OutputsView, OverviewView, StagesView, gt_boxes, layer_index, masks)

MASKS = ("fov", "seen", "out", "obj")


# ------------------------------------------------------------------------------------------------ maps
def add_sample(meters, prob, dist, g, spec, dt, mass) -> None:
    """all SegMeter updates of one sample at one dt; meters: {"fov": SegMeter, "seen": SegMeter, "obj": SegMeter}"""
    k = layer_index(spec)
    m = masks(g)
    areas = {n: prob[n] for n, (_, kind, _) in k.items() if kind in ("area", "line")}       # line = the 1-cell mask head
    objs = {n: prob[n] for n, (_, kind, _) in k.items() if kind == "object"}
    gt_a = {n: np.asarray(g[k[n][2]], np.float32) for n in areas}
    gt_o = {n: np.asarray(g[k[n][2]], np.float32) for n in objs}
    gt_d = {n: np.asarray(g[n + "_dist"], np.float32) for n in dist}
    for mname in ("fov", "seen", "out"):                    # out = the whole grid outside the current view (where a memory must help)
        meters[mname].add(areas, gt_a, mask=m[mname], mass=mass, dt=dt)
        meters[mname].add_lines(dist, gt_d, mask=m[mname], mass=mass, dt=dt)
    meters["obj"].add(objs, gt_o, mask=m["obj_fov"], mass=mass, dt=dt)


def _ikey(mname, dt, layer, subset, b) -> str:
    return f"{mname}|{float(dt):+.2f}|{layer}|{subset}|{b}"


def _flat_to_ikey(flat: str) -> str:
    mname, dtk, layer, subset, b, _ = flat.split("/")
    return f"{mname}|{dtk[2:]}|{layer}|{subset}|{b}"


class BevSegMetric(Metric):
    name = "seg"
    HEAD = {"drivable_iou": "fov/dt+0.00/drivable_area/all/all/iou@0", "walkway_iou": "fov/dt+0.00/walkway/all/all/iou@0",
            "lane_divider_iou": "fov/dt+0.00/lane_divider/all/all/iou@0.5", "lane_divider_f1": "fov/dt+0.00/lane_divider:zc/all/all/f1@0.5",
            "vehicle_iou": "obj/dt+0.00/vehicle/all/all/iou@0"}
    headline = tuple(HEAD.values())
    better = {}                                                            # all numbers: higher is better

    def __init__(self, grid, spec):
        self.grid, self.spec = grid, spec

    def frame(self, sample: Sample, pred: Prediction):
        meters = {m: SegMeter(self.grid) for m in MASKS}
        for dt, o in pred.outputs["dt"].items():
            mass = o["mass"] if (o["mass"] is not None and abs(dt) > 1e-6) else None
            add_sample(meters, o["prob"], o["dist"], sample.reference[dt], self.spec, dt, mass)
        keys, cnt, n_gt, err, n_err = [], [], [], [], []
        for mname, m in meters.items():
            for (dt, name, sname, b), acc in m.c.items():
                keys.append(_ikey(mname, dt, name, sname, b))
                cnt.append(acc["cnt"])
                n_gt.append(acc["n_gt"])
                err.append(acc["err"])
                n_err.append(acc["n_err"])
        return {"keys": np.array(keys), "cnt": np.asarray(cnt, np.float32), "n_gt": np.asarray(n_gt, np.int32),
                "err": np.asarray(err, np.float64), "n_err": np.asarray(n_err, np.int32)}

    def _accumulate(self, counts, want: Optional[set]):
        groups: Dict[bytes, dict] = {}
        for c in counts:
            g = groups.setdefault(c["keys"].tobytes(), {"keys": c["keys"], "cnt": 0.0, "n_gt": 0, "err": 0.0, "n_err": 0})
            g["cnt"] = g["cnt"] + c["cnt"].astype(np.float64)
            g["n_gt"] = g["n_gt"] + c["n_gt"].astype(np.int64)
            g["err"] = g["err"] + c["err"]
            g["n_err"] = g["n_err"] + c["n_err"].astype(np.int64)
        acc: Dict[str, list] = {}
        for g in groups.values():
            for i, key in enumerate(g["keys"]):
                key = str(key)
                if want is not None and key not in want:
                    continue
                a = acc.get(key)
                if a is None:
                    acc[key] = [g["cnt"][i].copy(), int(g["n_gt"][i]), float(g["err"][i]), int(g["n_err"][i])]
                else:
                    a[0] += g["cnt"][i]
                    a[1] += int(g["n_gt"][i])
                    a[2] += float(g["err"][i])
                    a[3] += int(g["n_err"][i])
        return acc

    def reduce(self, counts, keys: Optional[Sequence[str]] = None):
        want = None if keys is None else {_flat_to_ikey(k) for k in keys}
        acc = self._accumulate(counts, want)
        meters = {m: SegMeter(self.grid) for m in MASKS}
        for key, (cnt, n_gt, err, n_err) in acc.items():
            mname, dt, layer, subset, b = key.split("|")
            meters[mname].c[(round(float(dt), 3), layer, subset, b)] = {"cnt": cnt, "n_gt": n_gt, "err": err, "n_err": n_err}
        out: Dict[str, float] = {}
        for mname, m in meters.items():
            for dt, layers in m.result().items():
                for layer, subsets in layers.items():
                    for sname, bins in subsets.items():
                        for b, row in bins.items():
                            for metric, v in row.items():
                                out[f"{mname}/{dt_key(dt)}/{layer}/{sname}/{b}/{metric}"] = float(v)
        if keys is not None:
            return {k: out.get(k, float("nan")) for k in keys}
        return out

    def score(self, counts):
        v = self.reduce([counts], keys=list(self.HEAD.values()))
        return {n: v[k] for n, k in self.HEAD.items()}

    def support(self, counts):
        keys = {n: k.rsplit("/", 1)[0] for n, k in self.HEAD.items()}                     # the row of the number: its n_gt
        want = {_flat_to_ikey(k + "/n_gt") for k in keys.values()}
        acc = self._accumulate([counts], want)
        return {n: float(acc[_flat_to_ikey(k + "/n_gt")][1]) if _flat_to_ikey(k + "/n_gt") in acc else 0.0 for n, k in keys.items()}

    def direction(self, key):
        return "high"


# ------------------------------------------------------------------------------------------------ boxes
class BevDetMetric(Metric):
    HEADLINE = ("vehicle/all/ap@1", "vehicle/all/ap@2", "pedestrian/all/ap@1", "pedestrian/all/ap@2", "vehicle/all/ave_skill_moving")
    better = {"ate": "low", "fp_hi": "low"}
    LOW = ("ate", "ase", "aoe", "ave", "mATE", "mASE", "mAOE", "mAVE", "ave_moving", "ave_static", "ave0_moving", "ave0_static",
           "ave_slow_moving")                                      # the last part of a number's name: errors (and the share of moving boxes predicted to stand) are better low

    def __init__(self, groups: Sequence[str], name: str = "det", source: str = "dets"):
        self.name, self.source, self.groups = name, source, list(groups)
        self.meter = DetMeter(classes=self.groups)
        self.headline = tuple(f"dt+0.00/{k}" for k in self.HEADLINE) + ("dt+0.00/mean/all/mAP",)
        self.col_tp2 = 3 + list(self.meter.thr).index(2.0)
        self.col_ate = 3 + 2 * len(self.meter.thr)

    def frame(self, sample: Sample, pred: Prediction):
        out = {}
        for dt, o in pred.outputs["dt"].items():
            dets = o.get(self.source)
            if dets is None:
                continue
            gts = self.meter.keep(gt_boxes(sample.reference[dt], self.groups))
            fr = self.meter.frame_rows(self.meter.keep(dets), gts, self.groups)
            out[f"{dt_key(dt)}|rows"] = fr["rows"]                                     # float64: ties and errors stay exact
            out[f"{dt_key(dt)}|n_gt"] = fr["n_gt"].astype(np.int32)
            out[f"{dt_key(dt)}|vel"] = self.meter.frame_velocity(self.meter.keep(dets), gts, self.groups)     # moving / static velocity counts
        return out

    def reduce(self, counts, keys: Optional[Sequence[str]] = None):
        dts = sorted({k.split("|")[0] for c in counts for k in c})
        only = None
        if keys is not None:
            if any("/mean/" in k for k in keys):
                only = None                                                                  # the mean needs every class
            else:
                only = {tuple(k.split("/")[1:3]) for k in keys}
                dts = sorted({k.split("/")[0] for k in keys})
        out: Dict[str, float] = {}
        for dk in dts:
            frames = [{"rows": c[f"{dk}|rows"].astype(np.float64), "n_gt": c[f"{dk}|n_gt"]} for c in counts if f"{dk}|rows" in c]
            res = self.meter.reduce_rows(frames, self.groups, only=only)
            for cls, bins in res.items():
                for b, row in bins.items():
                    for metric, v in row.items():
                        out[f"{dk}/{cls}/{b}/{metric}"] = float(v)
            vel = [c[f"{dk}|vel"] for c in counts if f"{dk}|vel" in c]
            if vel and len(vel) == len(frames):                                              # (a stored run from before the split has none: nan)
                for cls, row in self.meter.reduce_velocity(vel, self.groups).items():
                    for metric, v in row.items():
                        out[f"{dk}/{cls}/all/{metric}"] = float(v)                           # scope all boxes of the class; no range bins
        if keys is not None:
            return {k: out.get(k, float("nan")) for k in keys}
        return out

    def direction(self, key):
        return "low" if key.rsplit("/", 1)[-1] in self.LOW else self.better.get(key, "high")

    def _frame_vals(self, c):
        rows = c.get("dt+0.00|rows")
        n_gt = c.get("dt+0.00|n_gt")
        if rows is None:
            return None
        tp = rows[:, self.col_tp2] > 0
        ate = rows[tp, self.col_ate]
        ate = ate[ate == ate]
        return {"recall2": float(tp.sum() / n_gt[:, 0].sum()) if n_gt[:, 0].sum() else float("nan"),
                "fp_hi": float(((rows[:, 2] >= 0.3) & ~tp).sum()), "ate": float(ate.mean()) if len(ate) else float("nan")}, \
               {"recall2": float(n_gt[:, 0].sum()), "fp_hi": 1.0, "ate": float(tp.sum())}

    def score(self, counts):
        v = self._frame_vals(counts)
        return v[0] if v else {}

    def support(self, counts):
        v = self._frame_vals(counts)
        return v[1] if v else {}


# ------------------------------------------------------------------------------------------------ loss
class BevLossMetric(Metric):
    name = "loss"
    headline = ("mean/total",)
    better = {"mean/total": "low", "total": "low"}

    def frame(self, sample: Sample, pred: Prediction):
        d = pred.info.get("loss") or {}
        names = sorted(d)
        return {"names": np.array(names), "v": np.asarray([d[n] for n in names], np.float64)}

    def reduce(self, counts, keys=None):
        acc: Dict[str, List[float]] = {}
        for c in counts:
            for n, v in zip(c["names"], c["v"]):
                acc.setdefault(str(n), []).append(float(v))
        out = {f"mean/{n}": float(np.mean(v)) for n, v in acc.items()}
        return {k: out.get(k, float("nan")) for k in keys} if keys is not None else out

    def score(self, counts):
        names = [str(n) for n in counts["names"]]
        return {"total": float(counts["v"][names.index("total")])} if "total" in names else {}


# ------------------------------------------------------------------------------------------------ checks and Task
GUIDE = """Panels of the overview: camera with the ground truth drawn in (3D boxes: vehicle red, pedestrian magenta, two-wheeler orange, grey = no lidar/radar point;
lines: lane dividers yellow, road dividers orange, centre-lines cyan); bird's-eye maps with x forward and +y on the left; error maps: grey = right,
RED = false, BLUE = missed, cream = not evaluated (outside the camera's view). Stages (debug view): 1 input, 2-3 camera head depth and uncertainty,
4 radar points, 5-6 lift, 7-8 camera BEV before / after temporal fusion, 9 what the past supplies, 10-11 radar, 12 trunk. Red flags: black input,
flat depth, energy outside the camera wedge, energy that does not follow the GT boxes, features without raster dots, one single colour in the trunk,
large red or blue regions, boxes in empty space, objects that do not move at dt +-0.5 s."""


def _gt_in_front(g, groups):
    return [b for b in gt_boxes(g, groups) if in_front(b)]


class BEVTask(Task):
    name = "bev"
    default_video_view = "dashboard"

    def __init__(self, grid, spec, groups: Sequence[str], depth_centres=None, z_levels: Sequence[float] = (), refine: bool = False):
        self.grid, self.spec, self.groups = grid, spec, list(groups)
        self.depth_centres, self.z_levels, self.refine = depth_centres, tuple(z_levels), refine

    def metrics(self) -> List[Metric]:
        ms: List[Metric] = [BevSegMetric(self.grid, self.spec), BevDetMetric(self.groups, "det", "dets")]
        if self.refine:
            ms.append(BevDetMetric(self.groups, "detr", "detr"))
        ms.append(BevLossMetric())
        return ms

    def views(self) -> Dict[str, View]:
        stages = StagesView(self)
        return {"overview": OverviewView(self), "outputs": OutputsView(self), "stages": stages, "debug": stages,
                "dashboard": DashboardView(self), "taps": DebugView()}

    def guide(self) -> str:
        return GUIDE

    def checks(self, sample: Sample, pred: Prediction) -> List[Flag]:
        """plausibility of the OUTPUTS of one frame (heuristic thresholds: they flag, a human judges)"""
        out: List[Flag] = []
        g0 = sample.reference[0.0]
        batch = sample.inputs
        o0 = pred.outputs["dt"][0.0]
        cam = np.asarray(g0["camera"], np.float32) > 0
        if "prob" in batch and self.depth_centres is not None:
            p = batch["prob"][0, 0].float().numpy()
            med = float(np.median((p * self.depth_centres()[:, None, None]).sum(0)))
            if not 4 < med < 40:
                out.append(Flag("depth head: median expected depth", "warn", med, "a road scene: pixel groups mostly 5-40 m away"))
        n_rad = int(len(batch["radar"]["x"]))
        if n_rad <= 20:
            out.append(Flag("radar detections", "warn", n_rad, "forward radars: tens to hundreds per 0.5 s"))
        gd = float((np.asarray(g0["drivable_area"], np.float32) > 0.5)[cam].mean()) if cam.any() else float("nan")
        pd = float((o0["prob"]["drivable_area"] > 0.5)[cam].mean()) if cam.any() else float("nan")
        if cam.any() and abs(pd - gd) >= 0.15:
            out.append(Flag("drivable area share of camera cells", "warn", f"{pd:.2f} vs GT {gd:.2f}", "a gross offset means a systematic bias"))
        P, G = o0["prob"]["drivable_area"] > 0.5, np.asarray(g0["drivable_area"], np.float32) > 0.5
        iou = float((P & G & cam).sum() / max((P | G)[cam].sum(), 1))
        if iou <= 0.5:
            out.append(Flag("drivable area IoU on camera cells", "warn", iou, "this frame only"))
        boxes = _gt_in_front(g0, self.groups)
        left = list(boxes)
        dets = [d for d in o0["dets"] if d["score"] >= 0.3 and in_front(d)]
        hit = 0
        for d in sorted(dets, key=lambda d: -d["score"]):
            j = min(range(len(left)), key=lambda q: math.hypot(left[q]["cx"] - d["cx"], left[q]["cy"] - d["cy"]), default=None)
            if j is not None and math.hypot(left[j]["cx"] - d["cx"], left[j]["cy"] - d["cy"]) < 2.0:
                left.pop(j)
                hit += 1
        if not (hit >= 0.5 * len(boxes) and len(dets) <= 2 * max(hit, 1) + 2):
            out.append(Flag("boxes in view: found / detected / GT", "warn", f"{hit} / {len(dets)} / {len(boxes)}",
                            "far and small objects are in the GT: recall is not expected to be 100 %"))
        for dt in (0.5, -0.5):
            od = pred.outputs["dt"].get(dt)
            if od is not None and od["mass"] is not None:
                frac = float((od["mass"] >= 0.5).mean())
                if not 0.3 < frac < 0.999:
                    out.append(Flag(f"splat: known cells at dt {dt:+.1f} s", "warn", f"{frac:.1%}",
                                    "cells with no source: the ego drives into new ground, a few percent must be unknown"))
        if o0.get("vhat") is not None and "velocity" in g0:
            v, gv = o0["vhat"], np.asarray(g0["velocity"], np.float32)
            sp, gsp = np.hypot(v[0], v[1]), np.hypot(gv[0], gv[1])
            mov = (np.asarray(g0["vehicle"], np.float32) > 0.5) & (gsp > 1.0)
            if mov.any():
                r = float(sp[mov].mean() / max(gsp[mov].mean(), 1e-6))
                if r <= 0.5:
                    out.append(Flag("velocity head: predicted / GT speed on moving vehicle cells", "fail" if r <= 0.2 else "warn",
                                    f"{sp[mov].mean():.2f} / {gsp[mov].mean():.2f} m/s", "the head should reproduce the speed of moving vehicles"))
        return out
