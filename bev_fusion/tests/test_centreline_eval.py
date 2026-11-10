"""Gate of the 1 m centre-line head in the evaluation (D48, plan 2026-11-03 section 2): the evaluation scores what `p["sem_1m"]` says. CPU, seconds.

  UNITS (no data)   `adapter.predictions_1m` on a hand-made output: sigmoid of the presence channel, each 1 m cell copied to its 2 x 2 cells, the direction pair
                    from the channel NAMES of `losses.layer_spec_1m`; `task_spec`; `SegMeter.add_directions` on a hand example with a known mean (wrap-around, a reversed line
                    is pi, a zero vector is pi, cells outside the GT / the mask not counted, range bins); the table and the card direction; the two debug panels (arrows
                    at exact plot coordinates; none for a model without the head; the cell count is the metric's); the real tiny model's `sem_1m` through the adapter
  HAND FRAME        four GT centre cells through the REAL `_BEVSubject.predict` (a fake forward) and BevSegMetric: the angle on every meter (fov / seen / out), every mass subset
                    (all / known / unknown) and dt, each with a known answer, the presence not needed
  REAL GT          cached frames (data/cache/gt, 3 dt each, no model): a synthetic head output made by the training code's own pooling (`build_layer_targets`) from the GT is
                    the ORACLE: IoU@1 = 1, IoU@0 equals an independent numpy count (the pooled mask is a superset: |G| / |P|), the angle error equals an independent numpy
                    pooling; a constant forward direction scores the mean |angle of the GT|; GT of another frame scores low; null scores 0. And OLD RUNS: with no 1 m head the
                    SegMeter counts of `add_sample` and its `result()` equal those of frozen copies of the code before the change, on oracle / noisy / shuffled 0.5 m outputs, and
                    the real predict of a model without `spec_1m` gives the old output keys
  MUTANTS           planted one at a time into the adapter / metric; each must make a check above fail (wrong presence channel, no upsample, a shifted upsample, wrong GT layer,
                    no sigmoid, cos / sin swapped, a flipped direction, an angle without the wrap, an axis angle that cannot tell a reversed line, a missing layer, the direction
                    on the fov meter only / unmasked on seen and out / without the splat mass, a changed result(), predict dropping the layer or the direction)
Run (from the bev_fusion folder): wsl ~/miniconda3/envs/gcar/bin/python tests/test_centreline_eval.py   (PYTHONPATH = the repository root for GtGenerator)
"""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests"), os.path.join(ROOT, "tests", "support"), os.path.join(ROOT, "training"), os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")]
from evaluation import bev_adapter as A, bev_gtcheck as GC, bev_suite, bev_views as V, metrics as M  # noqa: E402
import evaluation.bev_task as BT  # noqa: E402
from evaluation.bev_task import BEVTask, BevSegMetric, add_sample  # noqa: E402
from evalsuite import Prediction, Sample  # noqa: E402
from evalsuite.contracts import FrameRef  # noqa: E402
import layers as LY  # noqa: E402
import losses as L  # noqa: E402
from lift_pd import Grid  # noqa: E402

torch.set_num_threads(2)
GRID = Grid()
SPEC, SPEC_1M = L.layer_spec(False, True), L.layer_spec_1m(True)
GROUPS = [g for g, _ in L.NUSC_GROUPS]
PRE = "dt+0.00/centreline/all/all/"                  # a key of BevSegMetric.reduce: <mask>/<dt>/<layer>/<subset>/<range bin>/<metric>
EPS = 1e-9


class Fails(list):
    """the checks of one run: `ck(cond, msg)` prints (when loud) and records the failures"""

    def __init__(self, loud):
        super().__init__()
        self.loud = loud

    def __call__(self, cond, msg):
        if self.loud:
            print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            self.append(msg)


# ------------------------------------------------------------------------------------------------ units
def p_hand():
    """a 1 m head output of 4 x 3 blocks (grid 8 x 6 cells) with distinct values, channels as losses.layer_spec_1m: presence, distance, cos, sin"""
    sem_1m = torch.zeros(1, 4, 4, 3)
    sem_1m[0, 0, 1, 2] = 0.3                                          # presence logit of block (1, 2): sigmoid 0.5744
    sem_1m[0, 0, 3, 0] = -2.0
    sem_1m[0, 1] = 7.0                                                # the distance channel (not a presence layer)
    sem_1m[0, 2, 1, 2], sem_1m[0, 3, 1, 2] = 0.6, -0.8                # direction of block (1, 2)
    return {"sem": torch.zeros(1, len(SPEC), 8, 6), "sem_1m": sem_1m}


def unit_checks(ck):
    prob, direction = A.predictions_1m(p_hand(), 0, SPEC_1M)
    pr, dr = prob["centreline"], direction["centreline"]
    sig = lambda v: 1.0 / (1.0 + math.exp(-v))  # noqa: E731
    ck(list(prob) == ["centreline"] and pr.shape == (8, 6) and dr.shape == (2, 8, 6), f"predictions_1m: one presence layer 'centreline' (8, 6) and its direction (2, 8, 6) on the 0.5 m grid ({pr.shape}, {dr.shape})")
    ck(np.allclose(pr[2:4, 4:6], sig(0.3), atol=1e-6) and np.allclose(pr[6:8, 0:2], sig(-2.0), atol=1e-6) and abs(pr[0, 0] - 0.5) < 1e-6,
       "presence = sigmoid of the presence channel, block (i, j) -> cells 2i .. 2i + 1, 2j .. 2j + 1 (nearest, the alignment of the max-pool)")
    ck(abs(pr[1, 4] - 0.5) < 1e-6 and abs(pr[2, 3] - 0.5) < 1e-6 and abs(pr[4, 4] - 0.5) < 1e-6, "... and not one cell further")
    ck(np.allclose(dr[0, 2:4, 4:6], 0.6) and np.allclose(dr[1, 2:4, 4:6], -0.8) and dr[:, 0, 0].tolist() == [0.0, 0.0],
       "direction = the (cos, sin) channels in that order, raw (not normalised), copied the same way")
    model = SimpleNamespace(spec=SPEC, spec_1m=SPEC_1M)
    ck(A.task_spec(model)[:len(SPEC)] == SPEC and A.task_spec(model)[len(SPEC):] == [("centreline", "line", "centreline")],
       "task_spec: the model's channels + the presence layer scored as a line layer against the GT mask `centreline` (not `centreline@1m`)")
    ck(A.task_spec(SimpleNamespace(spec=SPEC)) is SPEC and A.task_spec(SimpleNamespace(spec=SPEC, spec_1m=[])) is SPEC, "task_spec: a model without the head keeps its spec (the same list)")

    # the angle: a hand example. GT cells (x cell, y cell, GT direction, predicted vector) in the three range bins; x = (ix + 0.5) * 0.5 m
    X, Y = 120, 104
    cells = [((10, 10), 0.0, (2.0, 0.0)),                             # 0 deg vs 0 deg (an unnormalised vector): 0                      range 0-20
             ((20, 10), 90.0, (1.0, 0.0)),                            # 90 vs 0: 90                                                      range 0-20
             ((50, 10), 180.0, (1.0, 0.0)),                           # a reversed line: 180                                             range 20-40
             ((60, 10), 10.0, (math.cos(math.radians(-10)), math.sin(math.radians(-10)))),   # 10 vs -10: 20, not 340                       range 20-40
             ((90, 10), 0.0, (0.0, 0.0))]                             # no direction: 180                                                range 40-60
    gt_mask, gd, pd = np.zeros((X, Y), np.float32), np.zeros((2, X, Y), np.float32), np.zeros((2, X, Y), np.float32)
    for (i, j), a, v in cells:
        gt_mask[i, j], gd[:, i, j], pd[:, i, j] = 1.0, (math.cos(math.radians(a)), math.sin(math.radians(a))), v
    pd[:, 30, 30] = (0.0, 1.0)                                        # a wrong direction on a cell outside the GT layer: not counted
    gt_mask[40, 40], gd[:, 40, 40], pd[:, 40, 40] = 1.0, (1.0, 0.0), (-1.0, 0.0)
    valid = np.ones((X, Y), bool)
    valid[40, 40] = False                                             # a GT cell outside the evaluated mask: not counted
    m = M.SegMeter(GRID)
    m.add_directions({"centreline": pd}, {"centreline": gd}, {"centreline": gt_mask}, mask=valid)
    r = m.result()[0.0]["centreline:dir"]["all"]
    want = {"all": (0 + 90 + 180 + 20 + 180) / 5, "0-20": 45.0, "20-40": 100.0, "40-60": 180.0}
    got = {b: math.degrees(r[b]["angle_mae"]) for b in want}
    ck(all(abs(got[b] - want[b]) < 1e-4 for b in want) and r["all"]["n_gt"] == 5 and set(r["all"]) == {"angle_mae", "n_gt"},
       f"angle_mae by hand (0, 90, 180, 20, 180 deg -> {want['all']:.1f} deg; ranges 45 / 100 / 180): {got['all']:.4f} deg, {got['0-20']:.2f} / {got['20-40']:.2f} / {got['40-60']:.2f}; one number and the GT count per row")
    mt = BevSegMetric(GRID, SPEC)
    ck(mt.direction("fov/dt+0.00/centreline:dir/all/all/angle_mae") == "low" and mt.direction("fov/dt+0.00/centreline/all/all/iou@1") == "high"
       and mt.direction("fov/dt+0.00/lane_divider:zc/all/all/field_mae") == "low", "the card direction: angle_mae lower is better, IoU higher (field_mae as before)")
    flat = {"seg_fov/seg/dt+0.00/centreline/all/all/iou@1": 0.5, "seg_fov/seg/dt+0.00/centreline:dir/all/all/angle_mae": 0.1, "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0": 0.8}
    old = {k: v for k, v in flat.items() if "centreline" not in k}
    ck("centreline/all/all/iou@1" in bev_suite.headline_table(flat) and "centreline:dir/all/all/angle_mae" in bev_suite.headline_table(flat) and "centreline" not in bev_suite.headline_table(old),
       "bev_suite.headline_table lists the centre-line rows of a run that has them, and an old run's table gains none")

    # the debug panels: arrows at exact plot coordinates (horizontal = y, vertical = x); none for a model without the head
    g1 = {"centreline": np.zeros((X, Y), np.float32), "centreline_dir": np.zeros((2, X, Y), np.float32)}
    g1["centreline"][40, 30], g1["centreline_dir"][:, 40, 30] = 1.0, (0.0, 1.0)                    # one GT cell, heading +y (left)
    o1 = {"prob": {"centreline": np.ones((X, Y), np.float32)}, "dir": {"centreline": np.zeros((2, X, Y), np.float32)}}
    o1["dir"]["centreline"][:, 40, 30] = (3.0, 0.0)                                              # predicted heading +x (forward), not a unit vector
    g1["centreline"][50, 40], g1["centreline_dir"][:, 50, 40] = 1.0, (0.3, 0.0)                  # a GT cell without a direction (|dir| <= 0.5): the metric skips it, so does the panel
    panels = V.centreline_panels(GRID, g1, o1, np.ones((X, Y), bool))
    arrows = [s for s in panels[1].shapes if s.kind == "arrow"] if len(panels) == 2 else []
    x0, y0 = GRID.x_min + 40.5 * GRID.res, GRID.y_min + 30.5 * GRID.res
    ck(len(arrows) == 2 and np.allclose(arrows[0].xy, (y0, x0, y0 + 2.0, x0)) and np.allclose(arrows[1].xy, (y0, x0, y0, x0 + 2.0)) and arrows[0].color != arrows[1].color,
       f"debug panels: a GT arrow +y and a predicted arrow +x at the cell, 2 m long, plot coordinates (y, x, y', x') ({[tuple(np.round(s.xy, 2)) for s in arrows]})")
    ck(len(panels) == 2 and panels[0].data.shape == (X, Y, 3) and panels[1].data.shape == (X, Y, 3) and "90.0 deg, 1 cells" in panels[1].title and "missed 0" in panels[0].title,
       f"... a presence panel (the error map) and a direction panel whose title says the mean error ({panels[1].title if panels else ''})")
    ck(V.centreline_panels(GRID, g1, {"prob": o1["prob"]}, np.ones((X, Y), bool)) == [] and V.centreline_panels(GRID, {}, o1, np.ones((X, Y), bool)) == [],
       "no panel for a model without the head (no `dir`) or a GT without the layer")
    outputs_of_checks(ck)


def outputs_of_checks(ck):
    """`bev_adapter.outputs_of`, the one place that builds a Subject's outputs (BEVSubject.predict and bev_free.FreeSubject call it): keys, the optional parts, the 1 m head"""
    ns = SimpleNamespace(decode=lambda obj, grid, groups, k, threshold: [["det"]], NUSC_GROUPS=L.NUSC_GROUPS)
    model = SimpleNamespace(grid=GRID, spec=SPEC, spec_1m=None, boxes=lambda res, dt, k, threshold, rescore: [["refined"]])
    bare = {"sem": torch.zeros(1, len(SPEC), 8, 6), "obj": {"hm": torch.zeros(1, 1, 8, 6)}}
    keys = {"prob", "dist", "mass", "dets", "detr", "vhat", "objvel"}
    o = A.outputs_of(model, {"out": {0.0: bare}}, ns, False)[0.0]
    ck(set(o) == keys and o["dets"] == ["det"] and o["detr"] is None and o["mass"] is None and o["vhat"] is None and o["objvel"] is None,
       f"outputs_of: a bare result gives the keys {sorted(keys)}, no refined boxes, no mass / velocity ({sorted(o)})")
    ck(A.outputs_of(model, {"out": {0.0: bare}}, ns, True)[0.0]["detr"] == ["refined"], "outputs_of: refine gives the refined boxes")
    full = {**bare, "mass": torch.ones(1, 1, 8, 6), "v_hat": torch.zeros(1, 2, 8, 6), "obj": {"hm": torch.zeros(1, 1, 8, 6), "vel": torch.zeros(1, 2, 8, 6)}}
    o = A.outputs_of(model, {"out": {0.0: full, 0.5: full}}, ns, False)
    ck(sorted(o) == [0.0, 0.5] and o[0.0]["mass"].shape == (8, 6) and o[0.0]["vhat"].shape == (2, 8, 6) and o[0.0]["objvel"].shape == (2, 8, 6) and "dir" not in o[0.0],
       "outputs_of: every dt, the mass and the velocity maps when the result has them, no direction without the 1 m head")
    model.spec_1m = SPEC_1M
    o = A.outputs_of(model, {"out": {0.0: {**full, "sem_1m": p_hand()["sem_1m"]}}}, ns, False)[0.0]
    ck("centreline" in o["prob"] and o["dir"]["centreline"].shape == (2, 8, 6), "outputs_of: a model with the 1 m head adds its layer to `prob` and the direction under `dir`")


def meter_checks(ck):
    """the direction on each meter / mask / mass subset with a known answer, through the real predict and BevSegMetric (frame -> reduce -> SegMeter.result). Four GT centre cells
    (GT direction 0 deg), the predicted direction 0 / 90 / 180 / 45 deg (errors 0 / 90 / 180 / 45): A, B in the camera view, C outside it but seen, D outside it and undefined;
    at dt +0.5 s the splat mass says B and C are unknown, A and D known. The presence is nowhere predicted: the direction is scored on the GT cells anyway"""
    X, Y = 120, 104
    g = {key: np.zeros((X, Y), np.float32) for _, _, key in SPEC}
    g.update(camera=np.zeros((X, Y), np.float32), undefined=np.zeros((X, Y), np.float32), centreline=np.zeros((X, Y), np.float32), centreline_dir=np.zeros((2, X, Y), np.float32))
    cells = {"A": (10, 10), "B": (10, 20), "C": (10, 30), "D": (10, 40)}
    g["camera"][cells["A"]] = g["camera"][cells["B"]] = 1.0
    g["undefined"][cells["D"]] = 1.0
    sem_1m, mass = torch.zeros(1, 4, X // 2, Y // 2), torch.full((1, 1, X, Y), 0.9)
    sem_1m[0, 0] = -20.0
    for name, deg in (("A", 0.0), ("B", 90.0), ("C", 180.0), ("D", 45.0)):
        i, j = cells[name]
        g["centreline"][i, j], g["centreline_dir"][:, i, j] = 1.0, (1.0, 0.0)
        sem_1m[0, 2:4, i // 2, j // 2] = torch.tensor([2 * math.cos(math.radians(deg)), 2 * math.sin(math.radians(deg))])        # not a unit vector
    mass[0, 0, cells["B"][0], cells["B"][1]] = mass[0, 0, cells["C"][0], cells["C"][1]] = 0.1
    p = {"sem": torch.zeros(1, len(SPEC), X, Y), "sem_1m": sem_1m, "mass": mass}
    metric = BevSegMetric(GRID, A.task_spec(SimpleNamespace(spec=SPEC, spec_1m=SPEC_1M)))
    r = metric.reduce([metric.frame(Sample(ref=FrameRef("gt", 0, "x"), inputs=None, reference={0.0: g, 0.5: g}), Prediction(outputs={"dt": predict_outputs({0.0: p, 0.5: p})}))])
    want = {("fov", "all"): (45.0, 2), ("fov", "known"): (0.0, 1), ("fov", "unknown"): (90.0, 1),
            ("seen", "all"): (90.0, 3), ("seen", "known"): (0.0, 1), ("seen", "unknown"): (135.0, 2),
            ("out", "all"): (112.5, 2), ("out", "known"): (45.0, 1), ("out", "unknown"): (180.0, 1)}
    got = {k: (math.degrees(r.get(f"{k[0]}/dt+0.50/centreline:dir/{k[1]}/all/angle_mae", float("nan"))), r.get(f"{k[0]}/dt+0.50/centreline:dir/{k[1]}/all/n_gt")) for k in want}
    ck(all(abs(got[k][0] - want[k][0]) < 1e-4 and got[k][1] == want[k][1] for k in want),
       "direction on every meter and subset (dt +0.5 s; fov = camera cells, seen = defined cells, out = outside the camera; all / known / unknown by the splat mass), a hand frame: "
       + "  ".join(f"{k[0]}/{k[1]} {got[k][0]:.1f} (n {got[k][1]})" for k in want))
    ck(all(abs(math.degrees(r[f"{m}/dt+0.00/centreline:dir/all/all/angle_mae"]) - want[m, "all"][0]) < 1e-4 for m in ("fov", "seen", "out")) and "fov/dt+0.00/centreline:dir/known/all/angle_mae" not in r,
       "the same at dt 0 without a mass map: the 'all' rows only")
    ck(r["fov/dt+0.50/centreline/all/all/iou@1"] == 0.0 and r["fov/dt+0.50/centreline/all/all/n_gt"] == 2,
       "the presence is scored on the same frame (nothing predicted: IoU 0 on the two GT cells in view)")


def tiny_check(ck):
    """the real tiny model's sem_1m (the 32 x 32 grid of test_slots) through the adapter"""
    from feature_cache import CachedBEVFusionModel
    from model import ModelConfig
    from test_slots import BASE, batch_for
    tiny = CachedBEVFusionModel(ModelConfig(**dict(BASE, centreline_1m=True))).eval()
    with torch.no_grad():
        p = tiny(batch_for(tiny.cfg), dts=(0.0,))["out"][0.0]
    pm, dm = A.predictions_1m(p, 1, tiny.spec_1m)
    ck(pm["centreline"].shape == (tiny.grid.nx, tiny.grid.ny) and dm["centreline"].shape == (2, tiny.grid.nx, tiny.grid.ny) and np.isfinite(pm["centreline"]).all()
       and np.array_equal(pm["centreline"][::2, ::2], torch.sigmoid(p["sem_1m"][1, 0]).numpy()) and np.array_equal(dm["centreline"][1, 1::2, 1::2], p["sem_1m"][1, 3].numpy()),
       f"the model's own sem_1m (batch item 1) through predictions_1m: {pm['centreline'].shape}, = sigmoid / copy of its channels")
    ck(A.task_spec(tiny)[-1] == ("centreline", "line", "centreline") and A.task_spec(tiny)[:len(tiny.spec)] == tiny.spec, "task_spec of the real model")


# ------------------------------------------------------------------------------------------------ real GT
def head_p(g, forward=False, null=False):
    """the model output of a frame whose 1 m head is the ORACLE: logits +-20 of the pooled GT presence, the pooled distance, the pooled (cos, sin) - the targets of the training
    code (`build_layer_targets` / `layers.pool_polyline`), channels in the order of `losses.layer_spec_1m`. forward: the direction (1, 0) everywhere; null: nothing present"""
    T = L.build_layer_targets([g], L.LossConfig(centreline_1m=True))
    ch = []
    for name, kind, key in SPEC_1M:
        t = T[key]
        if kind in ("line", "centreline"):
            ch.append(torch.full_like(t, -20.0) if null else 20.0 * (2 * t - 1))
        elif kind == "dist":
            ch.append(t)
        else:
            c = t[:, 0:1] if name.endswith("_cos") else t[:, 1:2]
            ch.append(torch.full_like(c, 1.0 if name.endswith("_cos") else 0.0) if forward else c)
    return {"sem": torch.zeros(1, len(SPEC), g["centreline"].shape[0], g["centreline"].shape[1]), "sem_1m": torch.cat(ch, 1)}


SUBJECT = []


def predict_outputs(ps, head=True):
    """the REAL `_BEVSubject.predict` around a fake forward: {dt: p} (the model output of each target time, read from the batch) -> {dt: output} as the Task receives it. head:
    the fake model has `spec_1m` (a model without the head has no such attribute, as the snapshots of old runs)"""
    if not SUBJECT:
        SUBJECT.append(A._make_subject_classes()[0])
    forward = lambda model, loss, batch, frac, device, amp, cfg, train_terms=False, **kw: ({"total": torch.tensor(1.0)}, {"out": batch["_p"]}, None)  # noqa: E731
    s = object.__new__(SUBJECT[0])
    s.ns = SimpleNamespace(T=SimpleNamespace(forward_loss=forward), clip_step=None, detach_tree=None, decode=lambda *a, **k: [[]], NUSC_GROUPS=L.NUSC_GROUPS)
    s.model = SimpleNamespace(grid=GRID, spec=SPEC, **({"spec_1m": SPEC_1M} if head else {}))
    s.loss, s.cfg, s.device, s.state, s.prev, s.refine = None, SimpleNamespace(amp=False), "cpu", None, None, False
    s.rec = SimpleNamespace(arm=lambda names: None, read=lambda i: {})
    return s.predict(Sample(ref=FrameRef("gt", 0, "x"), inputs={"_p": {dt: dict(p, obj={}) for dt, p in ps.items()}}, reference={})).outputs["dt"]


def numbers(samples, make):
    """BevSegMetric over the frames: make(i, dt, g) -> the model output p of that frame (a list of GT frames {dt: g}); -> the reduced numbers"""
    metric = BevSegMetric(GRID, A.task_spec(SimpleNamespace(spec=SPEC, spec_1m=SPEC_1M)))
    counts = []
    for i, (name, s) in enumerate(samples):
        sample = Sample(ref=FrameRef("gt", i, name), inputs=None, reference=s)
        counts.append(metric.frame(sample, Prediction(outputs={"dt": predict_outputs({dt: make(i, dt, g) for dt, g in s.items()})})))
    return metric.reduce(counts)


def pooled(a, how):
    """independent numpy pooling by 2 x 2 blocks: max / sum of (X, Y) -> repeated back to (X, Y)"""
    X, Y = a.shape
    b = a.reshape(X // 2, 2, Y // 2, 2)
    return np.repeat(np.repeat(getattr(b, how)((1, 3)), 2, 0), 2, 1)


def reference(samples):
    """the numbers the oracle must give at dt 0 on the evaluated (camera) cells, counted straight from the GT: IoU@0 of the pooled-and-repeated mask (a superset of the GT mask:
    |G| / |P|), the mean angle between each GT cell's direction and the 1 m block's mask-weighted mean direction, and of the constant direction (1, 0)"""
    n_g = n_p = n = 0
    err = err_fwd = 0.0
    for _, s in samples:
        g = s[0.0]
        G, cam = np.asarray(g["centreline"], np.float64) > 0.5, np.asarray(g["camera"], np.float32) > 0
        d = np.asarray(g["centreline_dir"], np.float64)
        has = G & (np.hypot(d[0], d[1]) > 0.5) & cam
        P = pooled(G.astype(np.float64), "max") > 0.5
        n_g, n_p = n_g + (G & cam).sum(), n_p + (P & cam).sum()
        w = G.astype(np.float64)
        wsum = pooled(w, "sum")
        u = np.stack([pooled(w * d[k], "sum") for k in (0, 1)]) / np.maximum(wsum, 1e-9)           # the mean direction of the block's line cells, repeated to its cells
        ang = np.where(np.hypot(u[0], u[1]) > 1e-6, np.abs(np.arctan2(d[0] * u[1] - d[1] * u[0], d[0] * u[0] + d[1] * u[1])), np.pi)
        err, n = err + ang[has].sum(), n + has.sum()
        err_fwd += np.abs(np.arctan2(d[1], d[0]))[has].sum()
    return {"iou0": n_g / n_p, "angle": err / n, "angle_forward": err_fwd / n, "n_gt": n_g, "n_dir": n}


def real_checks(ck, samples):
    ref = reference(samples)
    oracle = numbers(samples, lambda i, dt, g: head_p(g))
    ck(oracle["fov/" + PRE + "iou@1"] == 1.0 and oracle["fov/" + PRE + "n_gt"] == ref["n_gt"],
       f"ORACLE ({len(samples)} real frames, {int(ref['n_gt'])} GT centre cells in view): IoU@1 = {oracle['fov/' + PRE + 'iou@1']:.4f} (the pooled mask is within 1 m of every GT cell and the other way round)")
    ck(abs(oracle["fov/" + PRE + "iou@0"] - ref["iou0"]) < EPS and 0.3 < ref["iou0"] < 0.8 and oracle["fov/" + PRE + "iou@0"] < oracle["fov/" + PRE + "iou@0.5"] < 1.0,
       f"ORACLE IoU@0 = {oracle['fov/' + PRE + 'iou@0']:.4f} = the independent count |G| / |pooled| {ref['iou0']:.4f} (any shift of the 2 x 2 copy moves it), IoU@0.5 {oracle['fov/' + PRE + 'iou@0.5']:.3f}")
    a = oracle["fov/dt+0.00/centreline:dir/all/all/angle_mae"]
    ck(abs(a - ref["angle"]) < 1e-5 and a < 0.1 and oracle["fov/dt+0.00/centreline:dir/all/all/n_gt"] == ref["n_dir"],
       f"ORACLE angle_mae = {a:.5f} rad = the independent numpy pooling {ref['angle']:.5f} (a few degrees: a 1 m block averages its cells' headings), on {int(ref['n_dir'])} cells")
    def reversed_head(g):
        p = head_p(g)
        return dict(p, sem_1m=p["sem_1m"] * torch.tensor([1.0, 1.0, -1.0, -1.0]).view(1, 4, 1, 1))
    rev = numbers(samples, lambda i, dt, g: reversed_head(g))
    ck(abs(rev["fov/dt+0.00/centreline:dir/all/all/angle_mae"] - math.pi) < 0.2 and rev["fov/" + PRE + "iou@1"] == 1.0,
       f"all directions reversed: angle_mae {rev['fov/dt+0.00/centreline:dir/all/all/angle_mae']:.3f} rad ~ pi (a reversed line is not equal to its axis), the presence is untouched")
    fwd = numbers(samples, lambda i, dt, g: head_p(g, forward=True))
    ck(abs(fwd["fov/dt+0.00/centreline:dir/all/all/angle_mae"] - ref["angle_forward"]) < 1e-5 and ref["angle_forward"] > 2 * ref["angle"],
       f"constant direction (1, 0): angle_mae {fwd['fov/dt+0.00/centreline:dir/all/all/angle_mae']:.4f} rad = mean |angle of the GT| {ref['angle_forward']:.4f} (the baseline to beat; the oracle's is {ref['angle']:.4f})")
    for mask in ("seen", "out"):
        k = f"{mask}/{PRE}iou@1"
        ck(k in oracle and (oracle[k] == 1.0 or oracle[k] != oracle[k]), f"the other cell sets are scored too ({mask}: {oracle.get(k)})")
    sh = numbers(samples, lambda i, dt, g: head_p(samples[(i + 1) % len(samples)][1][dt]))
    ck(sh["fov/" + PRE + "iou@1"] < 0.5 * oracle["fov/" + PRE + "iou@1"],
       f"SHUFFLED (the GT of another frame): IoU@1 {sh['fov/' + PRE + 'iou@1']:.3f} < half the oracle's")
    nl = numbers(samples, lambda i, dt, g: head_p(g, null=True))
    ck(nl["fov/" + PRE + "iou@1"] == 0.0 and nl["fov/" + PRE + "iou@0"] == 0.0 and nl["fov/" + PRE + "n_gt"] == ref["n_gt"], "NULL (nothing present): IoU 0 at every tolerance on the same GT cells")
    ck(all(oracle[f"fov/dt{dt}/centreline/all/all/iou@1"] == 1.0 for dt in ("+0.50", "-0.50")), "the same at dt +-0.5 s (the GT of that frame)")


def view_checks(ck, samples):
    """every view and every metric family renders / runs for a model WITH the head (the Task's spec carries the extra row, the views look the GT up by it); the debug view
    (stages) has the two panels more than without the head, and the other views keep their panel count"""
    from evalsuite import ViewContext
    task = BEVTask(GRID, A.task_spec(SimpleNamespace(spec=SPEC, spec_1m=SPEC_1M)), GROUPS)
    task.curves = task.velocity = True
    _, s = samples[0]
    rad = {k: torch.tensor(v, dtype=torch.float32) for k, v in (("x", [5.0, 12.0]), ("y", [0.0, 3.0]), ("vr", [0.0, 5.0]))}
    sample = Sample(ref=FrameRef("gt", 0, "x"), inputs={"radar": rad}, reference=s, display={"camera": None})
    orc = GC.variants(samples[:1], SPEC, GROUPS)["oracle"][0]                                  # the 0.5 m maps, boxes and velocity of the oracle
    heads = predict_outputs({dt: head_p(s[dt]) for dt in orc})
    with_head = {dt: dict(o, prob={**o["prob"], "centreline": heads[dt]["prob"]["centreline"]}, dir=heads[dt]["dir"]) for dt, o in orc.items()}
    for m in task.metrics():
        m.reduce([m.frame(sample, Prediction(outputs={"dt": with_head}))])
    n = {}
    for tag, outs in (("head", with_head), ("old", orc)):
        for vn, view in task.views().items():
            if vn != "taps":
                n[tag, vn] = len(view.render(sample, Prediction(outputs={"dt": outs}), ViewContext()).panels)
    ck(all(n["head", v] == n["old", v] + (2 if v in ("stages", "debug") else 0) for v in ("overview", "outputs", "stages", "debug", "dashboard")),
       f"every view and metric runs for a model with the head; the debug view gains two panels ({n['old', 'stages']} -> {n['head', 'stages']}), the other views none")


def legacy_add_sample(meters, prob, dist, g, spec, dt, mass):
    """add_sample as it was before the centre-line head (a frozen copy)"""
    from evaluation.bev_views import layer_index, masks
    k = layer_index(spec)
    m = masks(g)
    areas = {n: prob[n] for n, (_, kind, _) in k.items() if kind in ("area", "line")}
    objs = {n: prob[n] for n, (_, kind, _) in k.items() if kind == "object"}
    gt_a = {n: np.asarray(g[k[n][2]], np.float32) for n in areas}
    gt_o = {n: np.asarray(g[k[n][2]], np.float32) for n in objs}
    gt_d = {n: np.asarray(g[n + "_dist"], np.float32) for n in dist}
    for mname in ("fov", "seen", "out"):
        meters[mname].add(areas, gt_a, mask=m[mname], mass=mass, dt=dt)
        meters[mname].add_lines(dist, gt_d, mask=m[mname], mass=mass, dt=dt)
    meters["obj"].add(objs, gt_o, mask=m["obj_fov"], mass=mass, dt=dt)


def legacy_result(m):
    """SegMeter.result as it was before the direction layers (a frozen copy)"""
    res = {}
    for (dt, name, sname, b), acc in sorted(m.c.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        row = {}
        for i, t in enumerate(m.tol):
            tp_p, fp, tp_g, fn = acc["cnt"][i]
            tp = (tp_p + tp_g) / 2
            if name.endswith(":zc"):
                p = tp_p / (tp_p + fp) if tp_p + fp else float("nan")
                r = tp_g / (tp_g + fn) if tp_g + fn else float("nan")
                row["f1@%g" % t] = 2 * p * r / (p + r) if p + r > 0 else (0.0 if p == p and r == r else float("nan"))
                row["p@%g" % t], row["r@%g" % t] = p, r
            else:
                row["iou@%g" % t] = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
        if name.endswith(":zc"):
            row["field_mae"] = acc["err"] / acc["n_err"] if acc["n_err"] else float("nan")
        row["n_gt"] = acc["n_gt"]
        res.setdefault(dt, {}).setdefault(name, {}).setdefault(sname, {})[b] = row
    return res


def same_numbers(a, b):
    """nested dicts of numbers equal, nan = nan"""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_numbers(a[k], b[k]) for k in a)
    return a == b or (a != a and b != b)


def old_run_checks(ck, samples):
    """a run WITHOUT the 1 m head: the counts of every layer / mask / dt / subset / range bin equal those of the code before the change, bit for bit, on real GT with the
    ceiling, a noisy, a half-dropped and a wrong (shuffled) 0.5 m output; the keys are the old ones (no centreline key, no `dir`)"""
    task_spec = A.task_spec(SimpleNamespace(spec=SPEC))
    ck(task_spec is SPEC, "no 1 m head: the Task's spec is the model's spec itself")
    variants = GC.variants(samples, SPEC, GROUPS, seed=1)
    same, same_res, n_keys = True, True, 0
    for name in ("oracle", "noise 0.2", "dropout 50 %", "shuffled"):
        new, old = {m: M.SegMeter(GRID) for m in ("fov", "seen", "out", "obj")}, {m: M.SegMeter(GRID) for m in ("fov", "seen", "out", "obj")}
        for (_, s), outs in zip(samples, variants[name]):
            for dt, o in outs.items():
                mass = np.tile(np.where(np.arange(104) < 52, 0.2, 0.9).astype(np.float32), (120, 1)) if dt else None      # half known, half unknown cells at dt != 0
                add_sample(new, o["prob"], o["dist"], s[dt], task_spec, dt, mass)
                legacy_add_sample(old, o["prob"], o["dist"], s[dt], task_spec, dt, mass)
        for mname in new:
            same &= new[mname].c.keys() == old[mname].c.keys() and all(np.array_equal(new[mname].c[k]["cnt"], old[mname].c[k]["cnt"]) and new[mname].c[k]["n_gt"] == old[mname].c[k]["n_gt"]
                                                                       and new[mname].c[k]["err"] == old[mname].c[k]["err"] and new[mname].c[k]["n_err"] == old[mname].c[k]["n_err"] for k in old[mname].c)
            n_keys += len(old[mname].c)
            same &= not any("centreline" in k[1] for k in new[mname].c)
            same_res &= same_numbers(new[mname].result(), legacy_result(old[mname]))
    ck(same and n_keys > 100, f"NO 1 m HEAD: the SegMeter counts equal those of the code before the change (4 outputs, {n_keys} count rows, equal bit for bit), and no key mentions the centre-line")
    ck(same_res, "NO 1 m HEAD: SegMeter.result() of the counts equals a frozen copy of the old result() (every IoU / F1 / precision / recall / field_mae / n_gt)")
    s0 = samples[0][1][0.0]
    out = predict_outputs({0.0: {"sem": torch.zeros(1, len(SPEC), *s0["centreline"].shape)}}, head=False)[0.0]
    ck(set(out) == {"prob", "dist", "mass", "dets", "detr", "vhat", "objvel"} and set(out["prob"]) == {n for n, k, _ in SPEC if k in ("area", "line", "object")},
       "NO 1 m HEAD: the real predict gives the old output keys (no `dir`) and the old prob layers for a model without `spec_1m` (as the snapshot of an old run)")


def run_all(ck, samples, tiny=False):
    unit_checks(ck)
    meter_checks(ck)
    if tiny:
        tiny_check(ck)
    if samples:
        real_checks(ck, samples)
        old_run_checks(ck, samples)
        view_checks(ck, samples)


# ------------------------------------------------------------------------------------------------ mutants
def mutants():
    """(label, patch): each patch is a deliberate bug; it returns the function that undoes it"""
    def swap(obj, name, fn):
        real = getattr(obj, name)
        setattr(obj, name, fn(real))
        return lambda: setattr(obj, name, real)
    def with_p(f):
        return lambda real: (lambda p, b, spec: f(real, p, b, spec))

    def no_sigmoid(real):
        def f(p, b, spec):
            keep, torch.sigmoid = torch.sigmoid, (lambda x: x)
            try:
                return real(p, b, spec)
            finally:
                torch.sigmoid = keep
        return f
    roll = lambda t, k: torch.roll(t, k, 1)  # noqa: E731

    def direction_bug(only=("fov", "seen", "out"), unmasked=(), no_mass=False):
        """bev_task.add_sample with the direction scored wrongly: only on the meters `only`, without the mask on `unmasked`, without the splat mass"""
        def f(real):
            def add(meters, prob, dist, g, spec, dt, mass, direction=None):
                real(meters, prob, dist, g, spec, dt, mass, None)
                if direction:
                    m, gt_dir = BT.masks(g), {n: np.asarray(g[n + "_dir"], np.float32) for n in direction}
                    for mname in only:
                        meters[mname].add_directions(direction, gt_dir, {n: np.asarray(g[n], np.float32) for n in direction}, mask=None if mname in unmasked else m[mname],
                                                     mass=None if no_mass else mass, dt=dt)
            return add
        return lambda: swap(BT, "add_sample", f)

    def drop_field_mae(real):
        def f(self):
            res = real(self)
            for layers in res.values():
                for subsets in layers.values():
                    for bins in subsets.values():
                        for row in bins.values():
                            row.pop("field_mae", None)
            return res
        return f
    return [
        ("direction scored on the fov meter only (not seen / out)", direction_bug(only=("fov",))),
        ("direction of the seen / out meters without their cell mask", direction_bug(unmasked=("seen", "out"))),
        ("direction without the splat mass (no known / unknown subsets)", direction_bug(no_mass=True)),
        ("SegMeter.result loses the line field error", lambda: swap(M.SegMeter, "result", drop_field_mae)),
        ("the presence layer is not put into prob (predict)", lambda: swap(A, "predictions_1m", with_p(lambda real, p, b, s: (lambda r: ({}, r[1]))(real(p, b, s))))),
        ("the direction is not handed to the Task (predict)", lambda: swap(A, "predictions_1m", with_p(lambda real, p, b, s: (lambda r: (r[0], None))(real(p, b, s))))),
        ("presence read from the next channel (off by one)", lambda: swap(A, "predictions_1m", with_p(lambda real, p, b, s: real(dict(p, sem_1m=roll(p["sem_1m"], 1)), b, s)))),
        ("no upsample (the 1 m arrays are scored as they are)", lambda: swap(A, "predictions_1m", with_p(lambda real, p, b, s: real(dict(p, sem=p["sem"][..., ::2, ::2]), b, s)))),
        ("upsample shifted by one 0.5 m cell", lambda: swap(A, "predictions_1m", with_p(lambda real, p, b, s: (lambda r: ({k: np.roll(v, 1, 0) for k, v in r[0].items()}, r[1]))(real(p, b, s))))),
        ("direction not upsampled", lambda: swap(A, "predictions_1m", with_p(lambda real, p, b, s: (lambda r: (r[0], {k: v[:, ::2, ::2] for k, v in r[1].items()}))(real(p, b, s))))),
        ("scored against the wrong GT layer (lane_divider)", lambda: swap(A, "task_spec", lambda real: (lambda m: [r if r[0] != "centreline" else ("centreline", "line", "lane_divider") for r in real(m)]))),
        ("the layer is not scored (task_spec adds nothing)", lambda: swap(A, "task_spec", lambda real: (lambda m: m.spec))),
        ("an old run's Task gains the centre-line layer", lambda: swap(A, "task_spec", lambda real: (lambda m: list(m.spec) + [("centreline", "line", "centreline")]))),
        ("presence without the sigmoid (the logit itself)", lambda: swap(A, "predictions_1m", no_sigmoid)),
        ("cos and sin swapped", lambda: swap(A, "predictions_1m", with_p(lambda real, p, b, s: real(dict(p, sem_1m=p["sem_1m"][:, [0, 1, 3, 2]]), b, s)))),
        ("the direction sign flipped", lambda: swap(A, "predictions_1m", with_p(lambda real, p, b, s: (lambda r: (r[0], {k: -v for k, v in r[1].items()}))(real(p, b, s))))),
        ("angle without the wrap-around", lambda: swap(M, "angle_error", lambda real: (lambda p, g: np.abs(np.arctan2(np.asarray(p)[1], np.asarray(p)[0]) - np.arctan2(np.asarray(g)[1], np.asarray(g)[0]))))),
        ("axis angle (a reversed line scores 0)", lambda: swap(M, "angle_error", lambda real: (lambda p, g: np.minimum(real(p, g), np.pi - real(p, g))))),
        ("a zero prediction vector scores 0", lambda: swap(M, "angle_error", lambda real: (lambda p, g: np.where(np.hypot(*np.asarray(p, np.float64)) > 1e-6, real(p, g), 0.0)))),
    ]


NEEDS_REAL_GT = {"SegMeter.result loses the line field error"}      # killed only by the real-GT check of the line field error (the tiny toy GT has no field error to lose)


def main():
    sig = "15efbde2d1"
    samples = GC.pick(8, sig, 0) if os.path.isdir(GC.cache_dir()) and sig in GC.signatures() else []
    if not samples:
        print(f"  skip the real-GT checks and the mutants of them: no cached GT of signature {sig} in {GC.cache_dir()}")
    ck = Fails(loud=True)
    run_all(ck, samples, tiny=True)
    survived = []
    for label, patch in mutants():
        if not samples and label in NEEDS_REAL_GT:                  # the message above promises it: without cached GT these mutants cannot be killed, so they are not run
            print(f"  skipped  mutant: {label}  (needs the cached GT)")
            continue
        undo = patch()
        quiet = Fails(loud=False)
        try:
            run_all(quiet, samples)
            killed = bool(quiet)
        except Exception as e:                                         # noqa: BLE001  a shape error is a kill too
            killed = True
            quiet.append(f"{type(e).__name__}: {str(e)[:60]}")
        finally:
            undo()
        print(f"  {'killed  ' if killed else 'SURVIVED'} mutant: {label}" + (f"  [{quiet[0][:70]}]" if killed else ""))
        if not killed:
            survived.append(label)
    bad = list(ck) + [f"mutant survived: {s}" for s in survived]
    print("test_centreline_eval " + ("FAILED: " + "; ".join(bad) if bad else "passed"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
