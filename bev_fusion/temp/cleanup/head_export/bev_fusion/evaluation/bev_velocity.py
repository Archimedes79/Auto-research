"""Velocity of the dense field and of the box centres, split the way the velocity cards need it (t12, t15): `bev_suite.py eval --velocity`.

    cells     the field `vhat` (the time mover's / motion block's velocity, dt 0) against the GT velocity layer on the object cells of the camera view
              (`masks()['obj_fov']` x any object layer > 0.5 x known velocity: `flow_weight` > 0 where the GT has it; autogt writes 0 m/s with
              flow_weight 0 for a box of unknown velocity, which would count as static), with the counts of bev_curves.velocity_sums: sets
              static (GT speed <= MOVING_SPEED, the evaluation's 0.5 m/s), and per class <cls> / <cls>_moving (the class layer > 0.5). The sets obj / moving
              themselves are `curve` numbers (`--curves`) and are not repeated here
    centres   at the GT box centres (observed boxes of known velocity (meta `vel_known`) inside the front window, centre cell in the grid and in `obj_fov`; the cell rule of object_head.targets:
              floor((c - min) / res)), the GT box velocity against three predictions read at that cell: `field` (vhat), `objhead` (the object head's dense
              `vel` map as it is, before any decode gate: the adapter's `objvel`), `zero` (always 0 m/s, the null); rows all / moving (GT speed > 0.5)
    numbers   n, epe, rmse, r2, bias (bev_curves.regression_numbers); keys `dt+0.00/cells/<set>/<number>` and `dt+0.00/centre/<source>/<all|moving>/<number>`
              `centre/objhead/all` is not comparable across models with and without the velocity hurdle (`object_moving`): a hurdle head's raw `vel` is trained
              on moving centres only (decode zeroes it where `mov` says static), so its static centres carry untrained values; compare `moving`

The guard "vehicle IoU at dt +-0.5 s" of the velocity cards is the standard `seg` number (`seg_obj/seg/dt+0.50/vehicle/all/all/iou@0`), not repeated here.
Counts are additive per frame (paired bootstrap of `compare --velocity`). Response rows in evaluation/bev_response.py, mutants in evaluation/bev_mutants.py.
    python evaluation/bev_velocity.py --selftest"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Sequence

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")]
from evalsuite import Metric, Prediction, Sample  # noqa: E402
from evaluation import bev_curves as C  # noqa: E402
from evaluation.bev_views import masks  # noqa: E402
from evaluation.metrics import dt_key, gt_boxes_from_meta, in_front  # noqa: E402

SOURCES = ("field", "objhead", "zero")
ROWS = ("all", "moving")


def object_cells(g, groups) -> np.ndarray:
    layers = [np.asarray(g[c], np.float32) > 0.5 for c in groups if c in g]
    if not layers:
        raise KeyError(f"none of the object layers {list(groups)} is in the GT")
    cells = masks(g)["obj_fov"] & np.any(layers, axis=0)
    return cells & (np.asarray(g["flow_weight"], np.float32) > 0) if "flow_weight" in g else cells


def cell_sets(g, groups) -> Dict[str, np.ndarray]:
    """the cell sets whose velocity_sums are kept: static and per class (row 0 = the set, row 1 = its moving part)"""
    cells = object_cells(g, groups)
    gv = np.asarray(g["velocity"], np.float64)
    out = {"static": cells & ~(np.hypot(gv[0], gv[1]) > C.MOVING_SPEED)}
    out.update({c: cells & (np.asarray(g[c], np.float32) > 0.5) for c in groups if c in g})
    return out


def scored_boxes(g, groups):
    """the GT boxes whose centre velocity is scored: observed, of the groups, inside the front window, velocity known (meta `vel_known`, True if absent)"""
    raw = [b for b in g["meta"]["boxes"] if b.get("observed", True)]
    return [b for b, rb in zip(gt_boxes_from_meta(raw), raw) if b["group"] in groups and in_front(b) and rb.get("vel_known", True)]


def centre_index(boxes, grid, valid):
    """(ix, iy, kept boxes): the centre cell of each box (object_head.targets' rule), boxes outside the grid or outside `valid` dropped"""
    ix, iy, kept = [], [], []
    for b in boxes:
        i, j = int(np.floor((b["cx"] - grid.x_min) / grid.res)), int(np.floor((b["cy"] - grid.y_min) / grid.res))
        if 0 <= i < valid.shape[0] and 0 <= j < valid.shape[1] and valid[i, j]:
            ix.append(i), iy.append(j), kept.append(b)
    return np.array(ix, int), np.array(iy, int), kept


class BevVelocityMetric(Metric):
    """see the module text; a Metric of the evalsuite (frame / reduce), counts additive per frame"""
    name = "vel"
    headline = ("dt+0.00/centre/field/moving/epe", "dt+0.00/centre/objhead/moving/epe", "dt+0.00/cells/static/epe", "dt+0.00/cells/vehicle/epe")
    LOW = ("epe", "rmse", "bias")

    def __init__(self, grid, groups: Sequence[str]):
        self.grid, self.groups = grid, list(groups)

    def direction(self, key):
        last = key.rsplit("/", 1)[-1]
        return "low" if last in self.LOW else "none" if last == "n" else "high"

    def frame(self, sample: Sample, pred: Prediction):
        out = {}
        for dt, o in pred.outputs["dt"].items():
            g = sample.reference.get(dt)
            if g is None or "velocity" not in g or (o.get("vhat") is None and o.get("objvel") is None):
                continue
            dk = dt_key(dt)
            if o.get("vhat") is not None:
                for name, sel in cell_sets(g, self.groups).items():
                    out[f"{dk}|cells|{name}"] = C.velocity_sums(o["vhat"], g["velocity"], sel)
            ix, iy, kept = centre_index(scored_boxes(g, self.groups), self.grid, masks(g)["obj_fov"])
            if not kept:
                continue
            gc = np.array([[b["vx"] for b in kept], [b["vy"] for b in kept]], np.float64)
            for src, field in (("field", o.get("vhat")), ("objhead", o.get("objvel")), ("zero", 0)):
                if field is None:
                    continue
                pc = np.zeros_like(gc) if src == "zero" else np.asarray(field, np.float64)[:, ix, iy]
                out[f"{dk}|centre|{src}"] = C.velocity_sums(pc, gc, np.ones(len(kept), bool))
        return out

    def reduce(self, counts, keys: Optional[Sequence[str]] = None):
        sums: Dict[str, np.ndarray] = {}
        for c in counts:
            for k, a in c.items():
                sums[k] = sums[k] + a if k in sums else np.array(a, np.float64)
        out: Dict[str, float] = {}
        for k, a in sums.items():
            dk, kind, name = k.split("|")
            for r, row in enumerate(ROWS):
                key = (f"{dk}/cells/{name if r == 0 else name + '_moving'}" if kind == "cells" else f"{dk}/centre/{name}/{row}")
                if kind == "cells" and name == "static" and r == 1:
                    continue                                             # static cells have no moving part
                for mk, v in C.regression_numbers(a[r]).items():
                    out[f"{key}/{mk}"] = float(v)
        if keys is not None:
            return {k: out.get(k, float("nan")) for k in keys}
        return out


def selftest():
    from types import SimpleNamespace
    from evalsuite.contracts import FrameRef
    grid = SimpleNamespace(x_min=0.0, y_min=-2.0, res=0.5)
    X, Y = 8, 8
    z = np.zeros((X, Y), np.float32)
    cam, veh = np.ones((X, Y), np.float32), z.copy()
    veh[2:4, 2:4] = 1                                                    # a moving vehicle (4 cells) at x 1.0 - 2.0, y -1.0 - 0.0
    gv = np.zeros((2, X, Y), np.float32)
    gv[0, 2:4, 2:4] = 3.0                                                # 3 m/s along x
    veh[6, 6] = 1                                                        # a parked one (1 cell)
    veh[1, 7] = 1                                                        # a car of unknown velocity (GT 0 m/s, flow_weight 0): left out
    fw = np.ones((X, Y), np.float32)
    fw[1, 7] = 0
    meta = {"boxes": [{"group": "vehicle", "center": [1.6, -0.6, 0], "size": [1, 1, 1], "yaw": 0.0, "vel": [3.0, 0.0], "observed": True},
                      {"group": "vehicle", "center": [3.2, 1.2, 0], "size": [0.5, 0.5, 1], "yaw": 0.0, "vel": [0.0, 0.0], "observed": True},
                      {"group": "vehicle", "center": [9.0, 0.0, 0], "size": [1, 1, 1], "yaw": 0.0, "vel": [5.0, 0.0], "observed": True},     # outside the grid
                      {"group": "vehicle", "center": [0.6, 1.6, 0], "size": [0.5, 0.5, 1], "yaw": 0.0, "vel": [0.0, 0.0], "observed": True, "vel_known": False}]}
    g = {"camera": cam, "undefined": z, "vehicle": veh, "velocity": gv, "flow_weight": fw, "meta": meta}
    m = BevVelocityMetric(grid, ["vehicle"])
    v = gv + np.array([0.3, 0.4], np.float32)[:, None, None]             # field off by 0.5 m/s everywhere
    objvel = np.zeros_like(gv)
    objvel[0, 3, 2] = 2.0                                                # the object head: 2 m/s at the moving box's centre cell (3, 2)
    c = m.frame(Sample(ref=FrameRef("gt", 0, "s"), inputs=None, reference={0.0: g}), Prediction(outputs={"dt": {0.0: {"vhat": v, "objvel": objvel}}}))
    r = m.reduce([c, c])
    assert r["dt+0.00/cells/vehicle/n"] == 10 and abs(r["dt+0.00/cells/vehicle/epe"] - 0.5) < 1e-6, r                        # 5 cells x 2 frames (float32 field)
    assert r["dt+0.00/cells/vehicle_moving/n"] == 8 and r["dt+0.00/cells/static/n"] == 2, r
    assert "dt+0.00/cells/static_moving/n" not in r and "dt+0.00/cells/obj/epe" not in r, "static has no moving part; obj / moving belong to `curve`"
    assert r["dt+0.00/centre/field/all/n"] == 4 and r["dt+0.00/centre/field/moving/n"] == 2, r                                 # the box outside the grid dropped
    assert abs(r["dt+0.00/centre/field/moving/epe"] - 0.5) < 1e-6 and abs(r["dt+0.00/centre/objhead/moving/epe"] - 1.0) < 1e-9, r
    assert abs(r["dt+0.00/centre/zero/moving/epe"] - 3.0) < 1e-9 and abs(r["dt+0.00/centre/zero/all/epe"] - 1.5) < 1e-9, r
    ix, iy, kept = centre_index([{"cx": 1.6, "cy": -0.6}, {"cx": 0.24, "cy": -2.0}], grid, cam > 0)
    assert list(ix) == [3, 0] and list(iy) == [2, 0], (ix, iy)                                                                 # floor((c - min) / res)
    c2 = m.frame(Sample(ref=FrameRef("gt", 0, "s"), inputs=None, reference={0.0: g}), Prediction(outputs={"dt": {0.0: {"vhat": None, "objvel": None}}}))
    assert c2 == {}, "no velocity output, no counts"
    assert m.direction("dt+0.00/cells/static/epe") == "low" and m.direction("dt+0.00/centre/zero/all/r2") == "high" and m.direction("x/n") == "none"
    print("bev_velocity selftest ok: cell sets (static, per class, moving part), centre cells and drop rule, unknown velocity left out, field / object head / zero at the centres, directions")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
