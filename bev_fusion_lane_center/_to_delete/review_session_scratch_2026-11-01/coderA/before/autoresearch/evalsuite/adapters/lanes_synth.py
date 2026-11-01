"""Synthetic lane world: the test bed of the lane suite and the porting template for a lane network.

World: scenes `road-00 ...` of 10 frames; a plan view (u right, v forward, ego at the origin), a road with 2-4 directed lanes, a
curvature, a slow lateral drift, in some scenes a fork that starts beyond 35 m. The ground truth is the set of lane polylines (graph with
successor edges) and their image. The "network" is the ground truth corrupted in a way that depends on the scene, so every cause of error
has a scene where it is planted, and the probing must find it there (selftest):

  clean     small noise (0.08 m)                          -> almost no error
  shift     the whole prediction displaced by (1.4, 0.6) m -> fn.shift, fp.shift
  far_miss  the fork beyond 35 m is not found              -> fn.missed_far
  gap       an 8 m hole in every lane                      -> fn.gap
  dup       a parallel copy 1.4 m aside of every lane      -> fp.duplicate
  ghost     an invented lane 6 m left of the road          -> fp.hallucinated
  short     lanes end at 30 m                              -> fn.truncated
  jitter    every vertex displaced by 0.7 m                -> fn.offset, fp.offset
  empty     nothing predicted                              -> fn.missed_near / fn.missed_far
  gt_ends   ground truth ends at 40 m (unmapped), the prediction continues -> fp.overshoot
  mixed     every frame one of the above, drawn at random  -> a bit of everything, planted bad frames

The Subject can emit the lane graph, the probability picture (a dense head) or both (`emit`), so the image level, the sparse level and
the path 'picture -> thinned graph' are all tested on the same world.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..contracts import FrameRef, Prediction, Sample, Source, Subject, TapSpec
from ..lanes import Grid, LaneConfig, LaneGraph, LaneTask, rasterize

GRID = Grid(u0=-25.0, v0=-5.0, res=0.5, nu=100, nv=130)
BOX = (-25.0, 25.0, 0.0, 60.0)
N_FRAMES = 10
SCENES: List[Tuple[str, bool]] = [("clean", True), ("shift", False), ("far_miss", True), ("gap", False), ("dup", False), ("ghost", False),
                                  ("short", False), ("jitter", True), ("empty", False), ("gt_ends", False), ("mixed", True)]
MIXED = ("clean", "shift", "gap", "dup", "ghost", "short", "jitter", "empty")
FORK_V = 38.0


def config() -> LaneConfig:
    return LaneConfig(GRID, taus=(0.5, 1.0, 1.5), ds=0.5, box=BOX, far_m=35.0, clip=5.0)


def make_road(idx: int, t: int, fork: bool, end_v: float = 60.0) -> LaneGraph:
    """ground-truth lanes of road `idx` at frame `t`: directed polylines (v increasing), a fork as three pieces with successor edges"""
    n = 2 + idx % 3
    c = (0.0, 1 / 300.0, -1 / 300.0)[idx % 3]
    off = 0.15 * t

    def u(k, v):
        return (k - (n - 1) / 2.0) * 3.5 + off + 0.5 * c * v ** 2
    vs = np.arange(0.0, end_v + 1e-9, 2.0)
    lines = [np.c_[u(k, vs), vs] for k in range(n - 1 if fork else n)]
    edges: List[Tuple[int, int]] = []
    if fork:
        k = n - 1
        va, vb = vs[vs <= FORK_V], vs[vs >= FORK_V]
        a = np.c_[u(k, va), va]
        b = np.c_[u(k, vb), vb]
        f = np.c_[u(k, vb) + 0.12 * (vb - FORK_V) + 0.0, vb]
        f[:, 0] += 3.5 * np.clip((vb - FORK_V) / 10.0, 0, 1)             # the fork leaves the lane
        ia = len(lines)
        lines += [a, b, f]
        edges += [(ia, ia + 1), (ia, ia + 2)]
    return LaneGraph(lines, edges, True)


class LaneSynthSource(Source):
    def __init__(self, n_frames: int = N_FRAMES, seed: int = 0):
        self.n_frames, self.seed = n_frames, seed

    def scenes(self) -> List[str]:
        return [f"road-{i:02d}" for i in range(len(SCENES))]

    def frames(self, scene: str) -> List[FrameRef]:
        i = int(scene.split("-")[1])
        return [FrameRef(scene, t, key=f"lane{i:02d}x{t:03d}") for t in range(self.n_frames)]

    def describe(self) -> Dict[str, Any]:
        return {"class": "LaneSynthSource", "scenes": len(SCENES), "frames": self.n_frames, "modes": [m for m, _ in SCENES]}

    def load(self, ref: FrameRef) -> Sample:
        i, t = int(ref.scene.split("-")[1]), ref.index
        mode, fork = SCENES[i]
        gt = make_road(i, t, fork, end_v=40.0 if mode == "gt_ends" else 60.0)
        u, v = GRID.centres()
        visible = np.abs(u) <= 18.0                                          # the input only sees +-18 m sideways (diagnostic mask)
        return Sample(ref, inputs={"road": i, "t": t, "mode": mode, "fork": fork, "seed": self.seed},
                      reference={"lanes": gt, "mask": rasterize(gt, GRID), "visible": visible}, display={}, meta={"mode": mode})


def corrupt(gt: LaneGraph, mode: str, rng: np.random.Generator, road: int, t: int, fork: bool, full: Optional[LaneGraph] = None) -> LaneGraph:
    """the planted network error"""
    lines = [p.copy() for p in gt.lines]
    edges = list(gt.edges)
    if mode == "mixed":
        mode = MIXED[int(rng.integers(len(MIXED)))]
    if mode == "gt_ends":                                                    # the network knows the whole road
        full = make_road(road, t, fork, end_v=60.0)
        lines, edges = [p.copy() for p in full.lines], list(full.edges)
        mode = "clean"
    if mode == "clean":
        lines = [p + rng.normal(0, 0.08, p.shape) for p in lines]
    elif mode == "shift":
        lines = [p + np.array([1.4, 0.6]) + rng.normal(0, 0.05, p.shape) for p in lines]
    elif mode == "far_miss":
        keep = list(range(len(lines) - 1)) if fork else list(range(len(lines)))        # the fork piece is the last line
        remap = {old: new for new, old in enumerate(keep)}
        lines = [lines[i] + rng.normal(0, 0.08, lines[i].shape) for i in keep]
        edges = [(remap[a], remap[b]) for a, b in edges if a in remap and b in remap]
    elif mode == "gap":
        out, remap = [], {}
        for i, p in enumerate(lines):
            a, b = p[p[:, 1] < 22.0], p[p[:, 1] > 30.0]
            for q in (a, b):
                if len(q) >= 2:
                    out.append(q + rng.normal(0, 0.08, q.shape))
        lines, edges = out, []
    elif mode == "dup":
        lines = [p + rng.normal(0, 0.08, p.shape) for p in lines] + [p + np.array([1.4, 0.0]) + rng.normal(0, 0.08, p.shape) for p in lines]
        edges = edges
    elif mode == "ghost":
        lines = [p + rng.normal(0, 0.08, p.shape) for p in lines]
        ref = gt.lines[0]                                                    # the leftmost lane
        lines.append(np.c_[ref[:, 0] - 6.0, ref[:, 1]])
    elif mode == "short":
        lines = [p[p[:, 1] <= 30.0] + rng.normal(0, 0.08, (int((p[:, 1] <= 30.0).sum()), 2)) for p in lines]
        lines = [p for p in lines if len(p) >= 2]
        edges = []
    elif mode == "jitter":
        lines = [p + rng.normal(0, 0.7, p.shape) for p in lines]
    elif mode == "empty":
        lines, edges = [], []
    else:
        raise ValueError(mode)
    return LaneGraph(lines, edges, True)


class LaneSynthSubject(Subject):
    """emit: 'both' (graph and picture) | 'lines' (graph head) | 'prob' (dense head: only the picture)"""
    warmup = 0

    def __init__(self, emit: str = "both", blur: float = 1.1):
        self.emit, self.blur = emit, blur

    def reset(self) -> None:
        pass

    def taps(self) -> List[TapSpec]:
        return []

    def describe(self) -> Dict[str, Any]:
        return {"name": f"lanes_synth_{self.emit}", "class": "LaneSynthSubject", "emit": self.emit}

    def predict(self, sample: Sample, taps=()) -> Prediction:
        from scipy.ndimage import gaussian_filter
        inp = sample.inputs
        rng = np.random.default_rng(10007 * (inp["seed"] + 1) + 131 * inp["road"] + inp["t"])
        g = corrupt(sample.reference["lanes"], inp["mode"], rng, inp["road"], inp["t"], inp["fork"])
        out: Dict[str, Any] = {}
        if self.emit in ("both", "lines"):
            out["lanes"] = g
        if self.emit in ("both", "prob"):
            img = rasterize(g, GRID).astype(np.float32)
            img = gaussian_filter(img, self.blur)
            img = img / max(float(img.max()), 1e-6) * 0.95 if img.max() > 0 else img
            out["prob"] = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1).astype(np.float32)
        return Prediction(out)


def build(emit: str = "both", n_frames: int = N_FRAMES, seed: int = 0, levels: str = "img,geo,topo", **_):
    cfg = config()
    return LaneSynthSource(int(n_frames), int(seed)), LaneSynthSubject(emit), LaneTask(cfg, tuple(levels.split(",")))
