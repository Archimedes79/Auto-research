"""Known-answer tests of the lane Task of the evalsuite (`../lane_task/`: geometry, raster, measures, causes, task, views, the synthetic
world `synth.py`) and of the core probing on it. Moved out of `autoresearch/evalsuite/selftest_lanes.py` on 2026-10-31 (D24); the evalsuite
selftest no longer runs lane checks.   tools/env/_py14.sh test_lane_task.py [name filter ...]
Every number is compared with a hand calculation or an independent brute-force computation, and the
planted causes of the synthetic world must be found where they were planted."""
from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from typing import Callable, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))   # evaluation/: the lane_task package
from lane_task import Grid, LaneGraph, LaneTask, rasterize, resample, simplify, skeleton_graph, thin  # noqa: E402  (puts the evalsuite on the path)
from lane_task import causes as C  # noqa: E402
from lane_task import measures as M  # noqa: E402
from lane_task import synth as LS  # noqa: E402
from lane_task import task as T  # noqa: E402
from lane_task.raster import edt  # noqa: E402

from evalsuite import EvalSuite  # noqa: E402
from evalsuite.render import report  # noqa: E402

CHECKS: List[Tuple[str, Callable]] = []


def check(name: str):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def line(u0, v0, u1, v1, n=2):
    return np.c_[np.linspace(u0, u1, n), np.linspace(v0, v1, n)]


def brute(pred: LaneGraph, gt: LaneGraph, ds, taus, clip, ft, ff):
    """geometry counts by an independent brute-force computation (scipy cdist, python loops over lanes)"""
    from scipy.spatial.distance import cdist
    P, G = pred.dense(ds), gt.dense(ds)
    D = cdist(P.xy, G.xy) if len(P) and len(G) else np.zeros((len(P), len(G)))
    dpg = D.min(1) if len(G) else np.full(len(P), np.inf)
    dgp = D.min(0) if len(P) else np.full(len(G), np.inf)
    out = {"p_ok": [(dpg <= t).sum() for t in taus], "g_ok": [(dgp <= t).sum() for t in taus],
           "cham": [np.minimum(dpg, clip).sum(), np.minimum(dgp, clip).sum()]}
    found = 0
    lines = np.unique(G.line)
    for ln in lines:
        found += int((dgp[G.line == ln] <= ft).mean() >= ff)
    out["lanes"] = [found, len(lines)]
    return out


@check("lanes.geometry: resample spacing, simplify keeps the corner, LaneGraph.of (dict / list), point graph merges a junction and links edges")
def t_geometry():
    p = resample(line(0, 0, 0, 10), 0.5)
    assert len(p) == 21 and np.allclose(np.diff(p[:, 1]), 0.5) and np.allclose(p[:, 0], 0)
    L = np.r_[line(0, 0, 0, 10, 21)[:-1], line(0, 10, 10, 10, 21)]
    k = simplify(L, 0.05)
    assert list(k) == [0, 20, len(L) - 1], k                                  # start, the corner, the end
    assert len(simplify(line(0, 0, 3, 3, 30), 0.01)) == 2                      # a straight line collapses
    g1 = LaneGraph.of({"pts": [line(0, 0, 0, 10), line(0, 10, 5, 20), line(0, 10, -5, 20)], "edges": [(0, 1), (0, 2)]})
    g2 = LaneGraph.of([line(0, 0, 0, 10)])
    assert len(g1) == 3 and g1.edges == [(0, 1), (0, 2)] and g1.directed and len(g2) == 1 and not g2.edges
    assert abs(g1.length() - (10 + 2 * np.hypot(5, 10))) < 1e-9
    xy, ed = g1.point_graph(1.0)
    deg = np.bincount(np.unique(np.vstack([ed, ed[:, ::-1]]), axis=0)[:, 0], minlength=len(xy))
    j = xy[deg >= 3]
    assert len(j) == 1 and np.allclose(j[0], [0, 10]), j                         # one junction, at the fork
    assert g1.vertices() == 6 and LaneGraph([line(0, 0, 0, 10, 50)], nodes=[[0, 1, 2]]).vertices() == 3   # DP vertices / given ids
    assert abs(g1.turning_deg() - 0.0) < 40 and LaneGraph([np.c_[np.cos(np.linspace(0, np.pi / 2, 50)), np.sin(np.linspace(0, np.pi / 2, 50))] * 20]).turning_deg() > 80


@check("lanes.raster: grid cells, rasterize, distance map in metres, thinning keeps one connected cell-wide line, a Y is traced back into three chains")
def t_raster():
    g = Grid(-10.0, 0.0, 0.5, 40, 80)
    assert g.shape == (80, 40) and g.extent == (-10.0, 10.0, 0.0, 40.0)
    r, c, ok = g.cell([[0.1, 0.1], [-10.0, 0.0], [10.0, 5.0], [0.0, 41.0]])
    assert list(ok) == [True, True, False, False] and (r[0], c[0]) == (0, 20)
    m = rasterize(line(0, 2, 0, 38), g)
    assert m.sum() == 73 and (m.sum(1)[4:77] == 1).all() and m[:4].sum() == 0 and m[77:].sum() == 0 and m[:, 20].all() == False and m[4:77, 20].all()
    d = edt(m, g)
    assert d[40, 20] == 0 and abs(d[40, 24] - 2.0) < 1e-6 and abs(d[40, 20 + 3] - 1.5) < 1e-6
    bar = np.zeros((40, 60), bool)
    bar[15:26, 5:55] = True                                                      # a thick bar (11 cells)
    sk = thin(bar)
    rows = np.nonzero(sk)[0]
    assert sk.sum() >= 30 and sk.sum(0).max() <= 2 and 18 <= rows.min() and rows.max() <= 22                  # one line along the middle of the bar
    from scipy.ndimage import label
    assert label(sk, structure=np.ones((3, 3)))[1] == 1
    Y = [line(0, 2, 0, 15, 27), line(0, 15, 6, 38, 47), line(0, 15, -6, 38, 47)]
    gy = skeleton_graph(rasterize(LaneGraph(Y), g, 1.2), g, 2.0, 0.3)
    assert len(gy) == 3, len(gy)
    true_len = sum(float(np.linalg.norm(np.diff(y, axis=0), axis=1).sum()) for y in Y)
    assert abs(gy.length() - true_len) < 0.12 * true_len, (gy.length(), true_len)
    f = M.geometry_numbers(M.geometry_counts(gy, LaneGraph(Y), 0.5, (0.5, 1.0), 5.0, 0.8), (0.5, 1.0), 5.0)
    assert f["F1@1.0"] > 0.95 and f["chamfer"] < 0.4, f                          # picture -> graph -> geometry round trip
    jn = M._junctions(*gy.point_graph(1.0))
    assert len(jn) >= 1 and np.hypot(*(jn[np.argmin(np.abs(jn[:, 1] - 15))] - [0, 15])) < 4.0      # the junction is where the fork is (the arms are
    #                                                                                               merged by the 1.2 m line width for the first few metres)


@check("lanes.measures (sparse): counts equal an independent brute force on random graphs; shift / half length / duplicate / reversal by hand")
def t_sparse():
    rng = np.random.default_rng(3)
    taus = (0.5, 1.0, 1.5)
    for _ in range(6):
        gt = LaneGraph([np.cumsum(rng.normal(0, 1, (rng.integers(5, 12), 2)), axis=0) * 2 for _ in range(rng.integers(1, 5))])
        pr = LaneGraph([p + rng.normal(0, 0.6, p.shape) for p in gt.lines[:-1]] + [np.cumsum(rng.normal(0, 1, (6, 2)), axis=0) * 2])
        c = M.geometry_counts(pr, gt, 0.5, taus, 5.0, 0.8)
        b = brute(pr, gt, 0.5, taus, 5.0, 1.0, 0.8)
        assert np.allclose(c["p_ok"], b["p_ok"]) and np.allclose(c["g_ok"], b["g_ok"]) and np.allclose(c["cham"], b["cham"]) and list(c["lanes"]) == b["lanes"]
    gt = LaneGraph([line(0, 0, 0, 40)])
    n = M.geometry_numbers(M.geometry_counts(LaneGraph([line(0.7, 0, 0.7, 40)]), gt, 0.5, taus, 5.0, 0.8), taus, 5.0)
    assert n["F1@0.5"] == 0 and n["F1@1.0"] == 1 and abs(n["chamfer"] - 0.7) < 1e-9 and n["lanes_found"] == 1     # 0.7 m aside
    n = M.geometry_numbers(M.geometry_counts(LaneGraph([line(0, 0, 0, 20)]), gt, 0.5, taus, 5.0, 0.8), taus, 5.0)
    assert n["P@1.0"] == 1 and abs(n["R@1.0"] - 43 / 81) < 1e-9 and n["lanes_found"] == 0 and abs(n["len_ratio"] - 0.5) < 1e-9     # half the lane
    n = M.geometry_numbers(M.geometry_counts(LaneGraph([line(0, 0, 0, 40), line(0.6, 0, 0.6, 40)]), gt, 0.5, taus, 5.0, 0.8), taus, 5.0)
    assert n["P@1.0"] == 1 and n["R@1.0"] == 1 and abs(n["dup_frac"] - 0.5) < 1e-9 and abs(n["F1_1to1"] - 2 * 0.5 / 1.5) < 1e-9    # a duplicate: 1:1 F1 drops
    n1 = M.geometry_numbers(M.geometry_counts(gt, gt, 0.5, taus, 5.0, 0.8), taus, 5.0)
    n2 = M.geometry_numbers(M.geometry_counts(LaneGraph([line(0, 40, 0, 0)]), gt, 0.5, taus, 5.0, 0.8), taus, 5.0)
    assert n1["dir_acc"] == 1 and n2["dir_acc"] == 0 and n2["F1@0.5"] == 1                    # reversed: geometry perfect, direction wrong
    e = M.geometry_numbers(M.geometry_counts(LaneGraph([]), gt, 0.5, taus, 5.0, 0.8), taus, 5.0)
    assert e["P@1.0"] == 1 and e["R@1.0"] == 0 and e["F1@1.0"] == 0 and e["chamfer"] == 5.0 and e["lanes_found"] == 0   # nothing predicted
    z = M.geometry_numbers(M.geometry_counts(LaneGraph([]), LaneGraph([]), 0.5, taus, 5.0, 0.8), taus, 5.0)
    assert z["F1@1.0"] == 1 and z["chamfer"] == 0                                              # nothing to find, nothing predicted
    a, b2 = M.geometry_counts(gt, gt, 0.5, taus, 5.0, 0.8), M.geometry_counts(LaneGraph([line(0.7, 0, 0.7, 40)]), gt, 0.5, taus, 5.0, 0.8)
    s = T._sum([a, b2])
    assert np.allclose(s["p_ok"], a["p_ok"] + b2["p_ok"]) and M.geometry_numbers(s, taus, 5.0)["F1@0.5"] == 0.5     # dataset number = sum of counts


@check("lanes.measures (topology): TOPO marbles and junctions on a fork: identical graph = 1, missed branch lowers recall, reversed graph ~ 0")
def t_topo():
    gt = LaneGraph([line(0, 0, 0, 20), line(0, 20, 0, 40), line(0, 20, 6, 40)], [(0, 1), (0, 2)])
    n = M.topo_numbers(M.topo_counts(gt, gt))
    assert n["topo_F1"] == 1 and n["junction_F1"] == 1, n
    no_branch = LaneGraph([line(0, 0, 0, 20), line(0, 20, 0, 40)], [(0, 1)])
    n = M.topo_numbers(M.topo_counts(no_branch, gt))
    assert n["topo_R"] < 0.95 and n["junction_R"] == 0 and n["junction_P"] == 1, n
    cut = LaneGraph([line(0, 0, 0, 19), line(0, 21, 0, 40), line(0, 21, 6, 40)])               # the fork is not linked to its lane
    n_cut = M.topo_numbers(M.topo_counts(cut, gt))
    assert n_cut["topo_R"] < 1.0 and n_cut["topo_F1"] < 1.0, n_cut
    rev = LaneGraph([p[::-1] for p in (line(0, 20, 0, 40), line(0, 0, 0, 20), line(6, 40, 0, 20))], [(0, 1)])
    r = M.topo_numbers(M.topo_counts(LaneGraph([line(0, 40, 0, 0)]), LaneGraph([line(0, 0, 0, 40)])))
    assert r["topo_F1"] < 0.3, r                                                                # directed: the reachable sets are opposite
    und = LaneGraph([line(0, 40, 0, 0)], directed=False)
    assert M.topo_numbers(M.topo_counts(und, LaneGraph([line(0, 0, 0, 40)], directed=False)))["topo_F1"] > 0.95


@check("lanes.measures (image): strict vs tolerant overlap, shift by 4 cells, soft Dice, tolerant AP = 1 for a clean picture and lower with false alarms")
def t_image():
    g = Grid(-10.0, 0.0, 0.5, 40, 80)
    gt = rasterize(line(0, 2, 0, 38), g)
    kw = dict(grid=g, thr=0.5, taus=(0.5, 1.0, 1.5), sweep=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    num = lambda p: M.image_numbers(M.image_counts(p, gt, **kw), kw["taus"], kw["sweep"])
    perfect = num(gt.astype(float))
    assert perfect["iou"] == perfect["dice"] == perfect["F1_rel@1.0"] == perfect["soft_dice"] == perfect["AP_rel"] == 1.0 and perfect["quality_rel@1.0"] == 1.0
    from scipy.ndimage import binary_dilation
    fat = binary_dilation(gt, np.ones((3, 3))).astype(float)
    n = num(fat)
    assert abs(n["iou"] - 73 / 225) < 1e-9 and n["F1_rel@0.5"] > 0.98 and n["R_rel@0.5"] == 1.0 and n["F1_rel@1.0"] == 1.0 and n["F1_skel@1.0"] > 0.95, n   # 3 cells wide, right place
    n = num(np.roll(gt, 4, axis=1).astype(float))                                           # 2 m aside
    assert n["iou"] == 0 and n["F1_rel@1.0"] == 0 and n["F1_rel@1.5"] == 0 and n["F1_rel@0.5"] == 0
    n = num(np.roll(gt, 2, axis=1).astype(float))                                           # 1 m aside: tolerant F1 sees it at 1.0 only from tau = 1.0
    assert n["F1_rel@0.5"] == 0 and n["F1_rel@1.0"] == 1.0
    rng = np.random.default_rng(0)
    noisy = np.clip(gt * 1.0 + rng.uniform(0, 0.2, gt.shape), 0, 1)
    n = num(noisy)
    assert n["AP_rel"] == 1.0 and n["best_F1_rel"] == 1.0 and n["best_thr"] >= 0.2 and n["P_rel@1.0"] == 1.0        # the threshold-free number ignores the weak background
    false_alarm = np.clip(gt * 1.0 + (rng.uniform(0, 1, gt.shape) > 0.97) * 0.95, 0, 1)
    assert num(false_alarm)["AP_rel"] < 1.0 and num(false_alarm)["P_rel@1.0"] < 1.0
    c1, c2 = M.image_counts(gt, gt, **kw), M.image_counts(fat, gt, **kw)
    s = T._sum([c1, c2])
    assert np.allclose(s["px"], c1["px"] + c2["px"])
    e = M.image_numbers(M.image_counts(np.zeros(gt.shape), gt, **kw), kw["taus"], kw["sweep"])
    assert e["F1_rel@1.0"] == 0 and e["AP_rel"] == 0 and e["px_P"] == 1.0                   # nothing predicted


@check("lanes.causes: the error budget adds up to the brute-force missed / false length; planted cases land in the planted cause (unit cases)")
def t_causes_unit():
    gt = LaneGraph([line(0, 0, 0, 60)])
    eb = lambda pr, **kw: C.error_budget(pr, gt, 0.5, 1.0, 3.0, 35.0, (-25, 25, 0, 60), **kw)
    gap = eb(LaneGraph([line(0, 0, 0, 20), line(0, 30, 0, 60)]))
    assert abs(gap["fn.gap"] - 7.5) < 1e-9 and gap["fn.truncated"] == 0 and gap["fn.offset"] == 0 and gap["fp_total"] == 0, gap   # 10 m hole minus 1 m of tolerance at each end
    short = eb(LaneGraph([line(0, 0, 0, 30)]))
    assert short["fn.truncated"] > 25 and short["fn.gap"] == 0, short
    gt2 = LaneGraph([line(0, 0, 0, 60), line(8, 40, 8, 60)])
    miss_far = C.error_budget(LaneGraph([line(0, 0, 0, 60)]), gt2, 0.5, 1.0, 3.0, 35.0, (-25, 25, 0, 60))
    assert abs(miss_far["fn.missed_far"] - 20) < 1.1, miss_far                                  # a whole lane beyond far_m
    near_lane = C.error_budget(LaneGraph([line(0, 0, 0, 60)]), LaneGraph([line(0, 0, 0, 60), line(8, 5, 8, 25)]), 0.5, 1.0, 3.0, 35.0, (-25, 25, 0, 60))
    assert abs(near_lane["fn.missed_near"] - 20) < 1.1, near_lane
    one = eb(LaneGraph([line(1.6, 0, 1.6, 60)]))
    assert one["fn.shift"] > 55 and one["fp.shift"] > 55 and abs(one["shift_u"] + 1.5) <= 0.26, one     # one lane 1.6 m aside: a displacement; shift_u = the correction that is applied (-1.5)
    rng = np.random.default_rng(1)
    off = eb(LaneGraph([line(0, 0, 0, 60, 60) + rng.normal(0, 0.8, (60, 2))]))
    assert off["fp.offset"] > 5 and off["fp_total"] == off["fp.offset"] and off["fn.shift"] == 0 and off["fp.shift"] == 0, off     # scatter around the lane: localisation
    sh =C.error_budget(LaneGraph([line(1.6, 0, 1.6, 60), line(5.1, 0, 5.1, 60)]), LaneGraph([line(0, 0, 0, 60), line(3.5, 0, 3.5, 60)]), 0.5, 1.0, 3.0, 35.0, (-25, 25, 0, 60))
    assert sh["fn.shift"] > 100 and sh["fp.shift"] > 100 and abs(sh["shift_u"] + 1.5) <= 0.26, sh   # two lanes both displaced by 1.6: a shift of the whole prediction
    dup = eb(LaneGraph([line(0, 0, 0, 60), line(1.4, 0, 1.4, 60)]))
    assert dup["fp.duplicate"] > 55 and dup["fn_total"] == 0 and dup["fp.shift"] == 0, dup     # a copy beside a correct lane
    ghost = eb(LaneGraph([line(0, 0, 0, 60), line(-9, 0, -9, 60)]))
    assert ghost["fp.hallucinated"] > 55 and ghost["fn_total"] == 0, ghost
    over = C.error_budget(LaneGraph([line(0, 0, 0, 60)]), LaneGraph([line(0, 0, 0, 40)]), 0.5, 1.0, 3.0, 35.0, (-25, 25, 0, 60))
    assert over["fp.overshoot"] > 18 and over["fn_total"] == 0, over
    cross = C.error_budget(LaneGraph([line(0, 0, 0, 60)]), LaneGraph([line(0, 0, 0, 60), line(-20, 30, 20, 30, 41)]), 0.5, 1.0, 3.0, 35.0, (-25, 25, 0, 60))
    assert cross["fn.offset"] == 0 and cross["fn.missed_near"] + cross["fn.missed_far"] > 36, cross     # a lane that only CROSSES the missing one is not a displaced prediction
    spur = eb(LaneGraph([line(0, 0, 0, 60), line(-9, 10, -9, 14)]))
    assert 3.5 <= spur["fp.spur"] <= 4.6 and spur["fp.hallucinated"] == 0, spur
    g = Grid(-25.0, -5.0, 0.5, 100, 130)
    u, _ = g.centres()
    unobs = C.error_budget(LaneGraph([]), LaneGraph([line(0, 0, 0, 60), line(22, 0, 22, 60)]), 0.5, 1.0, 3.0, 35.0, (-25, 25, 0, 60), np.abs(u) <= 18, g)
    assert abs(unobs["fn.unobservable"] - 60) < 1.1 and unobs["fn.missed_near"] + unobs["fn.missed_far"] > 55, unobs      # outside the view: not the model's fault
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(5)
    for _ in range(5):
        a = LaneGraph([p + rng.normal(0, 0.7, p.shape) for p in (line(0, 0, 3, 60, 30), line(4, 0, 6, 50, 25))] + [line(-8, 10, -8, 30, 10)])
        b = C.error_budget(a, LaneGraph([line(0, 0, 3, 60, 30), line(4, 0, 6, 55, 25), line(12, 20, 12, 40, 12)]), 0.5, 1.0, 3.0, 35.0, (-25, 25, 0, 60))
        dg_, dp_ = LaneGraph([line(0, 0, 3, 60, 30), line(4, 0, 6, 55, 25), line(12, 20, 12, 40, 12)]).dense(0.5, (-25, 25, 0, 60)), a.dense(0.5, (-25, 25, 0, 60))
        D = cdist(dg_.xy, dp_.xy)
        assert abs(sum(b[k] for k in C.FN) - (D.min(1) > 1.0).sum() * 0.5) < 1e-9 and abs(sum(b[k] for k in C.FP) - (D.min(0) > 1.0).sum() * 0.5) < 1e-9   # adds up exactly


def _suite(tmp, emit="both", n_frames=6, **kw):
    src, subj, task = LS.build(emit=emit, n_frames=n_frames, **kw)
    return EvalSuite(src, subj, task, out=tmp, state="carry")


@check("lane task: the picture-only head is scored as a picture and as the graph read from it; the graph-only head is drawn for the picture level; contract checker passes")
def t_task_levels():
    with tempfile.TemporaryDirectory() as d:
        for emit in ("both", "lines", "prob"):
            s = _suite(d, emit, 3)
            res = s.check()
            assert all(r["ok"] for r in res), [r for r in res if not r["ok"]]
        src, subj, task = LS.build(emit="prob", n_frames=3)
        smp = src.load(src.frames("road-00")[1])
        pr = subj.predict(smp)
        dd = T.lane_data(task.cfg, smp, pr)
        assert dd.pred_from == "prob" and len(dd.pred) >= 3 and dd.pred.directed is False                  # read from the picture, undirected
        src2, subj2, task2 = LS.build(emit="lines", n_frames=3)
        pr2 = subj2.predict(smp)
        d2 = T.lane_data(task2.cfg, smp, pr2)
        assert d2.pred_from == "lanes" and d2.prob.max() == 1.0 and d2.prob.shape == task2.cfg.grid.shape
        g_a = task.metrics()[1].reduce([task.metrics()[1].frame(smp, pr)])
        g_b = task2.metrics()[1].reduce([task2.metrics()[1].frame(smp, pr2)])
        assert g_a["F1@1.0"] > 0.9 and g_b["F1@1.0"] > 0.95                                              # clean scene: both paths agree on the good result
        fl = {f.name: f.status for f in LaneTask(task.cfg).checks(src.load(src.frames("road-08")[0]), LS.LaneSynthSubject().predict(src.load(src.frames("road-08")[0])))}
        assert fl["prediction not empty"] == "warn" and fl["ground truth has lanes"] == "ok"             # the empty scene is flagged


@check("lane suite: quant run = direct computation, planted causes are found in their scenes, budget / situations / sample / scan / CLI")
def t_lane_suite():
    with tempfile.TemporaryDirectory() as d:
        s = _suite(d, "both", 6)
        st = s.quant.run(name="r")
        assert len(st.frames) == 66 and len(st.scene_names()) == 11
        tab = s.quant.table(st)
        # the suite's dataset number = the number computed directly from the counts of the same frames, without the suite
        src, subj, task = LS.build(n_frames=6)
        ms = {m.name: m for m in task.metrics()}
        cs = []
        for sc in src.scenes():
            for ref in src.frames(sc):
                smp = src.load(ref)
                cs.append(ms["geo"].frame(smp, subj.predict(smp)))
        assert abs(tab["geo.F1@1.0"] - ms["geo"].reduce(cs)["F1@1.0"]) < 1e-12
        # the planted cause of every scene
        want = {"shift": ("fn.shift", "fp.shift"), "far_miss": ("fn.missed_far",), "gap": ("fn.gap",), "dup": ("fp.duplicate",), "ghost": ("fp.hallucinated",),
                "short": ("fn.truncated",), "jitter": ("fn.offset", "fp.offset"), "empty": ("fn.missed_near", "fn.missed_far"), "gt_ends": ("fp.overshoot",)}
        rows = s.probe.frames(st)
        for sc, (mode, _) in zip(src.scenes(), LS.SCENES):
            tot = {}
            for r in rows:
                if r["scene"] == sc:
                    for k, v in r["causes"].items():
                        tot[k] = tot.get(k, 0.0) + v
            if mode == "clean":
                assert sum(tot.values()) < 10, tot
            elif mode != "mixed":
                top = sorted(tot, key=tot.get, reverse=True)[:2]
                assert any(t in want[mode] for t in top), (mode, sorted(tot.items(), key=lambda kv: -kv[1])[:4])
                assert sum(tot.get(k, 0) for k in want[mode]) >= 0.5 * sum(tot.values()), (mode, tot)
        # budget adds up; situations name the planted scenes
        bud = s.probe.budget(st)
        assert abs(sum(b["amount"] for b in bud) - sum(r["error"] for r in rows)) < 1e-6 and abs(sum(b["share"] for b in bud) - 1) < 1e-9
        assert all(b["what"] and b["fix"] and b["target"] for b in bud if b["amount"] > 0)
        assert abs(sum(s.probe.targets(bud).values()) - 1) < 1e-9
        sits = {x["name"]: x for x in s.probe.situations(st)}
        gap_scenes = {l.rsplit("/", 1)[0] for l in sits["dominated by fn.gap"]["labels"]}
        assert "road-03" in gap_scenes and sits["dominated by fn.gap"]["n"] >= 6
        empty = set(sits["pred_empty"]["labels"])
        assert {f.label for f in src.frames("road-08")} <= empty and {l.rsplit("/", 1)[0] for l in empty} <= {"road-08", "road-10"} and sits["pred_empty"]["excess"] > 0   # the empty scene and some frames of the mixed one
        assert abs(sits["is_fork"]["share"] - 24 / 66) < 1e-9                                   # forks: roads 0, 2, 7, 10 (6 frames each)
        # sampling: video first, then frame; reproducible; no repeats
        a = s.probe.draw(["A/0"] + [f"B/{i}" for i in range(9)], 1, 0)
        assert a == s.probe.draw(["A/0"] + [f"B/{i}" for i in range(9)], 1, 0)
        hit = sum(s.probe.draw(["A/0"] + [f"B/{i}" for i in range(9)], 1, k)[0].startswith("A") for k in range(1000))
        assert 440 < hit < 560, hit                                                           # each VIDEO equally likely: 1 of 10 frames is still 50 %
        dr = s.probe.draw(sits["dominated by fn.gap"]["labels"], 8, 1, {l: l.rsplit("/", 1)[0] for l in sits["dominated by fn.gap"]["labels"]})
        assert len(set(dr)) == len(dr) == 8
        smp = s.probe.sample(st, "dominated by fn.gap", 2, seed=3)
        assert len(smp) == 2 and all(os.path.exists(x["images"]["overview"]) for x in smp) and all(x["top_cause"] == "fn.gap" for x in smp)
        res = s.probe.scan(st, k=2, top=3, seed=1)
        assert os.path.exists(res["index"]) and all(os.path.exists(os.path.join(os.path.dirname(res["index"]), l)) for l in report.links_in(res["index"]))
        assert os.path.exists(os.path.join(res["dir"], "budget.csv")) and len(res["situations"]) == 3
        try:
            s.probe.sample(st, "no such situation")
            raise AssertionError("an unknown situation must raise")
        except KeyError:
            pass
        # the same through the command line
        from evalsuite.cli import main
        out = os.path.join(d, "cli")
        base = ["--adapter", "lane_task.synth:build", "--arg", "n_frames=4", "--out", out]
        assert main(base + ["quant", "run", "--name", "c"]) == 0
        assert main(base + ["probe", "budget", "c"]) == 0 and main(base + ["probe", "situations", "c"]) == 0
        assert main(base + ["probe", "sample", "c", "dominated by fn.gap", "-n", "1"]) == 0
        assert main(base + ["probe", "scan", "c", "-k", "1", "--top", "2", "--no-render"]) == 0


@check("lane views: the overview has the six panels; blue = ground truth without prediction, red = prediction without ground truth at the planted places")
def t_lane_views():
    from evalsuite.contracts import ViewContext
    from evalsuite.render.figure import FN_COLOR, FP_COLOR, render_page
    src, subj, task = LS.build(n_frames=3)
    v = task.views()["overview"]
    sm = src.load(src.frames("road-06")[0])                                                    # short: lanes end at 30 m
    pg = v.render(sm, subj.predict(sm), ViewContext(scores={"geo.F1@1.0": 0.5}))
    assert [p.title.split(" (")[0] for p in pg.panels[:3]] == ["ground truth", "prediction", "errors on the lines"] and len(pg.panels) == 6, [p.title for p in pg.panels]
    shapes = pg.panels[2].shapes
    blue = [s for s in shapes if tuple(s.color) == tuple(FN_COLOR)]
    assert blue and all(np.asarray(s.xy)[:, 1].min() >= 29.0 for s in blue) and not [s for s in shapes if tuple(s.color) == tuple(FP_COLOR)]   # only the part beyond 30 m is missed
    sm = src.load(src.frames("road-04")[0])                                                    # dup: a false copy
    pg = v.render(sm, subj.predict(sm), ViewContext())
    assert [s for s in pg.panels[2].shapes if tuple(s.color) == tuple(FP_COLOR)] and not [s for s in pg.panels[2].shapes if tuple(s.color) == tuple(FN_COLOR)]
    img = render_page(pg)
    assert img.ndim == 3 and img.shape[0] > 300 and img.shape[1] > 600


@check("lane probe of a constant prediction (S1): a Subject that hands out ONE LaneGraph (or one Prediction) for every frame gets each frame's own tags and error budget")
def t_constant_prediction():
    from evalsuite.contracts import Prediction, Subject

    class Const(Subject):                                    # a null baseline / a prior: the same object for every frame
        def __init__(self, g, same_prediction=False):
            self.g, self.p = g, (Prediction({"lanes": g}) if same_prediction else None)

        def predict(self, sample, taps=()):
            return self.p if self.p is not None else Prediction({"lanes": self.g})

    src, _, task = LS.build(n_frames=3)
    frames = [src.load(r) for sc in src.scenes()[:4] for r in src.frames(sc)]
    for g in (LaneGraph([]), LaneGraph([line(0, 0, 0, 50, 26)])):
        for same in (False, True):
            subj = Const(g, same)
            for smp in frames:
                own = task.probe(smp, Prediction({"lanes": LaneGraph([p.copy() for p in g.lines])}))   # a fresh object: nothing cached
                got = task.probe(smp, subj.predict(smp))
                assert got == own, (smp.ref.label, same, {k: (got[k], own[k]) for k in own if got.get(k) != own[k]})
                d = T.lane_data(task.cfg, smp, subj.predict(smp))
                assert d.gt is not None and abs(d.gt.length() - LaneGraph.of(smp.reference["lanes"]).length()) < 1e-9   # the frame's own ground truth
    with tempfile.TemporaryDirectory() as d:                 # end to end: the stored probe tags of a run of the empty baseline
        s = EvalSuite(src, Const(LaneGraph([])), task, out=d)
        st = s.quant.run(scenes=src.scenes()[:4], name="null", progress=0)
        gl = {fr["label"]: fr["info"]["probe"]["tag.gt_len"] for fr in st.frames}
        want = {smp.ref.label: round(LaneGraph.of(smp.reference["lanes"]).length(), 4) for smp in frames}
        assert gl == want and len(set(gl.values())) > 1, (gl, want)


def main(only=()) -> int:
    checks = [c for c in CHECKS if not only or any(o in c[0] for o in only)]
    t0, bad = time.time(), 0
    for name, fn in checks:
        t = time.time()
        try:
            fn()
            print(f"ok    {name}  ({time.time() - t:.1f} s)")
        except Exception:                                                  # noqa: BLE001
            bad += 1
            print(f"FAIL  {name}\n{traceback.format_exc()}")
    print(f"lane task tests: {len(checks) - bad}/{len(checks)} checks passed in {time.time() - t0:.0f} s")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
