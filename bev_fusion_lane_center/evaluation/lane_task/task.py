"""The lane-centre-line Task: metrics at two levels (image, then sparse), plausibility checks, descriptors of the situation and the
error budget for the probing. One class for every network that outputs lane centre lines, as a picture or as a graph.

What an adapter hands over (plain dicts, see `lanes/__init__.py`):
  Sample.reference  {"lanes": LaneGraph | dict | list of polylines (ground truth),   and / or  "mask": bool image of the lines,
                     "valid": bool image of the evaluated cells (optional),  "visible": bool image of what the input can observe (optional)}
  Prediction.outputs {"prob": float image (probability / heat map),  and / or  "lanes": LaneGraph | dict | list of polylines}
A dense head gives "prob" only: the sparse metrics then read the graph from the thinned picture (`skeleton_graph`), so the same output
is scored first as an image and then sparsely. A graph head gives "lanes" only: the image metrics then draw it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from evalsuite.contracts import Flag, Metric, Prediction, Sample, Task
from . import causes as C
from . import measures as M
from .geometry import LaneGraph
from .raster import Grid, rasterize, skeleton_graph


@dataclass
class LaneConfig:
    grid: Grid
    taus: Tuple[float, ...] = (0.5, 1.0, 1.5)       # tolerances in metres (the middle one is the main one)
    ds: float = 0.5                                 # spacing of the dense points of a graph
    box: Optional[Tuple[float, float, float, float]] = None   # sparse region (u0, u1, v0, v1); default = the grid
    thr: float = 0.5                                # threshold of the probability image
    sweep: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)   # ascending
    clip: float = 5.0                               # chamfer distances are clipped at this many metres
    found_frac: float = 0.8                         # a ground-truth lane is found when this share of it is within tau
    skeleton_min_len: float = 2.0
    skeleton_tol: float = 0.3
    render_width: float = 0.0                       # width of a predicted graph drawn as an image (0 = one cell)
    topo: Dict[str, float] = None                   # TOPO settings: ds, seed_step, L, snap, r, j_tol
    near_factor: float = 3.0                        # causes: 'displaced' = within near_factor * tau
    far_m: float = 40.0                             # causes / tags: range from the origin beyond which a lane is 'far'
    origin: Tuple[float, float] = (0.0, 0.0)        # the ego position in the frame of the coordinates

    def __post_init__(self):
        if self.topo is None:
            self.topo = dict(ds=1.0, seed_step=5.0, L=30.0, snap=2.0, r=2.0, j_tol=3.0)
        if self.box is None:
            self.box = self.grid.extent

    @property
    def tau(self) -> float:
        return self.taus[min(1, len(self.taus) - 1)]

    @property
    def tk(self) -> str:
        return M.tkey(self.tau)


@dataclass
class LaneData:
    gt: LaneGraph
    pred: LaneGraph
    mask: np.ndarray          # bool, ground-truth lines as an image
    prob: np.ndarray          # float, prediction as an image
    valid: Optional[np.ndarray]
    visible: Optional[np.ndarray]
    pred_from: str            # 'lanes' | 'prob' (read from the picture)
    gt_from: str
    probe: Optional[Dict[str, float]] = None   # lane_probe() of this frame, computed once


def lane_data(cfg: LaneConfig, sample: Sample, pred: Prediction) -> LaneData:
    """the graph and the image form of ground truth and prediction of one frame, computed once and cached on the Prediction together
    with the frame it belongs to (a Subject may hand out one constant Prediction or LaneGraph for every frame: a null baseline, a prior)"""
    cache = pred.info.setdefault("_lane", {})
    hit = cache.get(id(cfg))
    if hit is not None and hit[0] == sample.ref:
        return hit[1]
    ref, out = sample.reference or {}, pred.outputs or {}
    # ground truth
    if ref.get("lanes") is not None:
        gt, gf = LaneGraph.of(ref["lanes"]), "lanes"
    elif ref.get("mask") is not None:
        gt, gf = skeleton_graph(ref["mask"], cfg.grid, cfg.skeleton_min_len, cfg.skeleton_tol), "mask"
    else:
        raise ValueError("Sample.reference needs 'lanes' or 'mask'")
    mask = np.asarray(ref["mask"], bool) if ref.get("mask") is not None else rasterize(gt, cfg.grid)
    # prediction
    if out.get("prob") is not None:
        prob = np.asarray(out["prob"], np.float32)
    elif out.get("lanes") is not None:
        prob = rasterize(out["lanes"], cfg.grid, cfg.render_width).astype(np.float32)
    else:
        raise ValueError("Prediction.outputs needs 'prob' or 'lanes'")
    if out.get("lanes") is not None:
        pg, pf = LaneGraph.of(out["lanes"]), "lanes"
    else:
        pg, pf = skeleton_graph(prob >= cfg.thr, cfg.grid, cfg.skeleton_min_len, cfg.skeleton_tol), "prob"
    d = LaneData(gt, pg, mask, prob, None if ref.get("valid") is None else np.asarray(ref["valid"], bool),
                 None if ref.get("visible") is None else np.asarray(ref["visible"], bool), pf, gf)
    cache[id(cfg)] = (sample.ref, d)
    return d


def _sum(counts: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for c in counts:
        for k, v in c.items():
            out[k] = out[k] + v if k in out else np.array(v, np.float64)
    return out


def _reduce(fn, counts, *args) -> Dict[str, float]:
    """numbers of the summed counts of any set of frames (an empty set has no numbers)"""
    c = _sum(counts)
    return {k: float(v) for k, v in fn(c, *args).items()} if c else {}


class LaneImageMetric(Metric):
    """the picture of the prediction against the ground-truth lines: strict pixel overlap, tolerant (relaxed) precision / recall /
    F1 and quality, the same on the thinned prediction, soft Dice, threshold-free tolerant average precision"""
    name = "img"

    def __init__(self, cfg: LaneConfig):
        self.cfg = cfg
        self.headline = (f"F1_rel@{cfg.tk}", "AP_rel", f"F1_skel@{cfg.tk}", "dice")
        self.better = {}

    def frame(self, sample, pred):
        c, d = self.cfg, lane_data(self.cfg, sample, pred)
        return M.image_counts(d.prob, d.mask, c.grid, c.thr, c.taus, c.sweep, d.valid)

    def reduce(self, counts, keys=None):
        return _reduce(M.image_numbers, counts, self.cfg.taus, self.cfg.sweep)

    def score(self, counts):
        r = M.image_numbers(counts, self.cfg.taus, self.cfg.sweep)
        return {k: r[k] for k in (f"F1_rel@{self.cfg.tk}", "dice", f"P_rel@{self.cfg.tk}", f"R_rel@{self.cfg.tk}")}

    def support(self, counts):
        n = float(counts["px"][1])
        return {k: n for k in (f"F1_rel@{self.cfg.tk}", "dice", f"P_rel@{self.cfg.tk}", f"R_rel@{self.cfg.tk}")}


class LaneSparseMetric(Metric):
    """polylines against polylines: P / R / F1 at tau, chamfer (accuracy, completeness), lanes found, one-to-one F1 and duplicate share,
    direction accuracy, vertices per 100 m, length ratio"""
    name = "geo"

    def __init__(self, cfg: LaneConfig):
        self.cfg = cfg
        self.headline = (f"F1@{cfg.tk}", "chamfer", "lanes_found", "vert100")
        self.better = {"chamfer": "low", "chamfer_acc": "low", "chamfer_comp": "low", "dup_frac": "low", "vert100": "low"}

    def frame(self, sample, pred):
        c, d = self.cfg, lane_data(self.cfg, sample, pred)
        return M.geometry_counts(d.pred, d.gt, c.ds, c.taus, c.clip, c.found_frac, c.box)

    def reduce(self, counts, keys=None):
        return _reduce(M.geometry_numbers, counts, self.cfg.taus, self.cfg.clip)

    def score(self, counts):
        r = M.geometry_numbers(counts, self.cfg.taus, self.cfg.clip)
        return {k: r[k] for k in (f"F1@{self.cfg.tk}", "chamfer", "lanes_found", f"P@{self.cfg.tk}", f"R@{self.cfg.tk}")}

    def support(self, counts):
        n = float(counts["n"][1])
        return {f"F1@{self.cfg.tk}": n, "chamfer": n, "lanes_found": float(counts["lanes"][1]), f"P@{self.cfg.tk}": float(counts["n"][0]),
                f"R@{self.cfg.tk}": n}


class LaneTopoMetric(Metric):
    """connectivity: TOPO marbles (what can be reached along the graph) and junctions (forks / merges) matched one-to-one"""
    name = "topo"

    def __init__(self, cfg: LaneConfig):
        self.cfg = cfg
        self.headline = ("topo_F1", "junction_F1")
        self.better = {}

    def frame(self, sample, pred):
        c, d = self.cfg, lane_data(self.cfg, sample, pred)
        return M.topo_counts(d.pred, d.gt, box=c.box, **c.topo)

    def reduce(self, counts, keys=None):
        return _reduce(M.topo_numbers, counts)

    def score(self, counts):
        r = M.topo_numbers(counts)
        return {"topo_F1": r["topo_F1"], "topo_R": r["topo_R"], "topo_P": r["topo_P"]}

    def support(self, counts):
        n = float(counts["marb"][0])
        return {"topo_F1": n, "topo_R": n, "topo_P": float(counts["marb"][2])}


def lane_probe(cfg: LaneConfig, d: LaneData) -> Dict[str, float]:
    """descriptors ('tag.*') and the error budget ('cause.*', metres of lane) of one frame, cached on the frame's LaneData (never on the
    LaneGraph: a constant prediction object would carry the first frame's tags into every other frame)"""
    if d.probe is not None:
        return d.probe
    gt, pr = d.gt, d.pred
    glen, plen = gt.length(), pr.length()
    dg = gt.dense(cfg.ds, cfg.box)
    far = float((np.hypot(dg.xy[:, 0] - cfg.origin[0], dg.xy[:, 1] - cfg.origin[1]) >= cfg.far_m).mean()) if len(dg) else 0.0
    gx, ge = gt.point_graph(1.0)
    eb = C.error_budget(pr, gt, cfg.ds, cfg.tau, cfg.near_factor, cfg.far_m, cfg.box, d.visible, cfg.grid, origin=cfg.origin)
    out = {"tag.gt_lanes": float(len(gt)), "tag.gt_len": glen, "tag.is_fork": float(len(M._junctions(gx, ge)) > 0),
           "tag.curviness": gt.turning_deg() / max(glen / 100.0, 1e-9) if glen > 0 else 0.0, "tag.far_share": far,
           "tag.pred_empty": float(plen == 0.0), "tag.gt_empty": float(glen == 0.0), "tag.pred_pieces": float(len(pr)),
           "tag.len_ratio": plen / glen if glen > 0 else 0.0, "tag.vert100": 100.0 * pr.vertices() / plen if plen > 0 else 0.0,
           "tag.shift": float(np.hypot(eb["shift_u"], eb["shift_v"]))}
    out.update({f"cause.{k}": eb[k] for k in C.CAUSES})
    d.probe = out
    return out


class LaneTask(Task):
    """levels: any of 'img', 'geo', 'topo' (default all); `extra_tags(sample) -> {name: float}` adds descriptors of the situation
    that only the adapter knows (night, rain, scene type ...)"""
    name = "lanes"

    def __init__(self, cfg: LaneConfig, levels: Sequence[str] = ("img", "geo", "topo"), extra_tags: Optional[Callable[[Sample], Dict[str, float]]] = None,
                 view_factory: Optional[Callable[[LaneConfig], Any]] = None, extra_metrics: Sequence[Metric] = ()):
        self.cfg, self.levels, self.extra_tags, self._extra_metrics = cfg, tuple(levels), extra_tags, list(extra_metrics)
        self.default_video_view = "overview"
        self._view_factory = view_factory

    def metrics(self) -> List[Metric]:
        make = {"img": LaneImageMetric, "geo": LaneSparseMetric, "topo": LaneTopoMetric}
        return [make[k](self.cfg) for k in self.levels] + self._extra_metrics

    def views(self):
        from .views import LaneOverviewView
        return {"overview": (self._view_factory or LaneOverviewView)(self.cfg)}

    def checks(self, sample: Sample, pred: Prediction) -> List[Flag]:
        d = lane_data(self.cfg, sample, pred)
        fl = []
        glen, plen = d.gt.length(), d.pred.length()
        fl.append(Flag("ground truth has lanes", "ok" if glen > 0 else "warn", glen, "no ground-truth lane in this frame: only false positives can occur"))
        fl.append(Flag("prediction not empty", "ok" if plen > 0 or glen == 0 else "warn", plen, "nothing predicted although there is ground truth"))
        ratio = plen / glen if glen > 0 else 0.0
        fl.append(Flag("prediction length plausible", "ok" if glen == 0 or 0.3 <= ratio <= 3.0 else "warn", round(ratio, 2),
                       "predicted length differs from the ground truth by more than a factor of 3: fragmentation, duplicates or a missing head"))
        p = d.prob
        fl.append(Flag("probabilities finite", "ok" if np.isfinite(p).all() else "fail", None, "NaN / inf in the lane image"))
        if self.cfg.thr is not None and p.size and 0.0 <= float(np.nanmin(p)) and float(np.nanmax(p)) <= 1.0 + 1e-6:
            fl.append(Flag("probabilities not saturated", "ok" if 0.0 < float((p > 0.5).mean()) < 0.5 else "warn", round(float((p > 0.5).mean()), 3),
                           "more than half of the picture is 'lane' or nothing at all"))
        return fl

    def probe(self, sample: Sample, pred: Prediction) -> Dict[str, float]:
        d = lane_data(self.cfg, sample, pred)
        out = dict(lane_probe(self.cfg, d))
        if self.extra_tags is not None:
            out.update({f"tag.{k}": float(v) for k, v in self.extra_tags(sample).items()})
        return out

    def causes(self) -> Dict[str, Tuple[str, str, str]]:
        """cause -> (what it is, what to improve, where to work: model / code / evaluation)"""
        return {k: (what, fix, C.target_of(fix)) for k, (what, fix) in C.CAUSES.items()}

    def guide(self) -> str:
        c = self.cfg
        return (f"Lane centre lines. Grey = right, RED = false (prediction without ground truth within {c.tau:g} m), BLUE = missed (ground truth without "
                f"prediction within {c.tau:g} m). Panels: ground truth, prediction (dots = vertices), the same errors on the lines, the errors on "
                f"the image, the error budget (which cause the missing / false metres are assigned to). Metrics: img = picture level (tolerant "
                f"F1, AP), geo = polylines (F1 at {', '.join(M.tkey(t) for t in c.taus)} m, chamfer, lanes found, vertices per 100 m), "
                f"topo = connectivity (TOPO, junctions).")
