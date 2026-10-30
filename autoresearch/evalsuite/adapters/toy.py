"""The toy adapter: synthetic videos, a numpy "network" with a memory, two metrics, an overview view. It is the test bed of the
suite (selftest.py checks every function against known answers on it) and the TEMPLATE for porting the suite to a real network:
copy this file, replace the three classes' bodies, keep the shapes of what they return.

World: scenes toy-00 ... with `n_frames` frames of a 48 x 48 plane; two rectangles move with constant velocity; the ground
truth is their mask and boxes. The input is the mask plus noise. PLANTED frames are corrupted on purpose (a noise burst, a hole),
so an outlier search must find exactly them (and, with the memory carried, the frame after a burst suffers too).

Subject ("network"): 3x3 box blur -> memory (running average with the previous frame, carried along the scene) -> logits; taps
'blur', 'memory', 'logits', 'features' (3 channels). `bias` shifts the decision threshold (a worse network for compare tests).

Metrics: 'seg' (additive TP / FP / FN counts for the near and far half of the plane -> iou, iou_near, iou_far, precision, recall),
'objs' (ragged: centre error of every GT box to the nearest predicted blob + blob counts -> center_err, count_abs_err).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..contracts import (Flag, FrameRef, Metric, Page, Panel, Prediction, Sample, Shape, Source, Subject, TapSpec, Task, View,
                         ViewContext)
from ..render.figure import ERROR_LEGEND, TP_COLOR, error_counts, error_rgb

SIZE = 48
PLANTED = {("toy-03", 5): "noise burst", ("toy-03", 6): "noise burst", ("toy-07", 2): "hole"}


# ------------------------------------------------------------------------------------------------ Source
class ToySource(Source):
    def __init__(self, n_scenes: int = 8, n_frames: int = 12, seed: int = 0, planted: Optional[Dict] = None):
        self.n_scenes, self.n_frames, self.seed = n_scenes, n_frames, seed
        self.planted = PLANTED if planted is None else planted

    def scenes(self) -> List[str]:
        return [f"toy-{i:02d}" for i in range(self.n_scenes)]

    def frames(self, scene: str) -> List[FrameRef]:
        i = int(scene.split("-")[1])
        return [FrameRef(scene, t, key=f"tok{i:02d}x{t:03d}") for t in range(self.n_frames)]

    def describe(self) -> Dict[str, Any]:
        return {"class": "ToySource", "scenes": self.n_scenes, "frames": self.n_frames, "seed": self.seed}

    def boxes(self, i: int, t: int) -> List[Tuple[float, float, float, float]]:
        """(cx, cy, w, h) of the two rectangles of scene i at frame t"""
        rng = np.random.default_rng(1000 * self.seed + i)
        out = []
        for _ in range(2):
            cx0, cy0 = rng.uniform(10, SIZE - 10, 2)
            vx, vy = rng.uniform(-0.8, 0.8, 2)
            w, h = rng.uniform(8, 13, 2)
            out.append((float(np.clip(cx0 + vx * t, 7, SIZE - 7)), float(np.clip(cy0 + vy * t, 7, SIZE - 7)), float(w), float(h)))
        return out

    def load(self, ref: FrameRef) -> Sample:
        i, t = int(ref.scene.split("-")[1]), ref.index
        boxes = self.boxes(i, t)
        yy, xx = np.mgrid[0:SIZE, 0:SIZE] + 0.5
        mask = np.zeros((SIZE, SIZE), bool)
        for cx, cy, w, h in boxes:
            mask |= (np.abs(xx - cx) <= w / 2) & (np.abs(yy - cy) <= h / 2)
        rng = np.random.default_rng(7919 * self.seed + 131 * i + t)
        img = mask.astype(np.float32) + rng.normal(0, 0.25, mask.shape).astype(np.float32)
        what = self.planted.get((ref.scene, t))
        if what == "noise burst":
            img = mask.astype(np.float32) + rng.normal(0, 2.0, mask.shape).astype(np.float32)
        elif what == "hole":
            img[:, SIZE // 3: 2 * SIZE // 3] = 0.0
        return Sample(ref=ref, inputs=img, reference={"mask": mask, "boxes": boxes}, display={"input": img},
                      meta={"planted": what or ""})


# ------------------------------------------------------------------------------------------------ Subject
def _blur(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    return sum(p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)) / 9.0


class ToySubject(Subject):
    warmup = 2

    def __init__(self, bias: float = 0.0, memory: float = 0.5, name: str = "toy"):
        self.bias, self.memory, self.name = bias, memory, name
        self.mem: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.mem = None

    def taps(self) -> List[TapSpec]:
        return [TapSpec("blur", "map", "front end", "the input, smoothed: rectangles with soft edges", "noise still visible"),
                TapSpec("memory", "map", "memory", "blur averaged with the previous frame", "all zero: the state never fills"),
                TapSpec("logits", "map", "head", "positive inside the rectangles", "positive everywhere"),
                TapSpec("features", "feature_map", "head", "3 channels: blur, memory, input", "")]

    def describe(self) -> Dict[str, Any]:
        return {"class": "ToySubject", "name": self.name, "bias": self.bias, "memory": self.memory}

    def predict(self, sample: Sample, taps: Sequence[str] = ()) -> Prediction:
        x = np.asarray(sample.inputs, np.float32)
        b = _blur(x)
        self.mem = b if self.mem is None else (self.memory * self.mem + (1 - self.memory) * b)
        logits = (self.mem - 0.5 - self.bias) * 4.0
        allt = {"blur": b, "memory": self.mem.copy(), "logits": logits,
                "features": np.stack([b, self.mem, x]).astype(np.float32)}
        return Prediction(outputs={"mask": logits > 0, "logits": logits}, taps={k: v for k, v in allt.items() if k in set(taps)})


# ------------------------------------------------------------------------------------------------ Metrics
class SegMetric(Metric):
    """counts (2, 4): rows = near half / far half of the plane (x < 24, x >= 24), columns = TP, FP, FN, TN"""
    name = "seg"
    headline = ("iou", "iou_near", "iou_far")
    better = {"iou": "high", "iou_near": "high", "iou_far": "high"}

    def frame(self, sample: Sample, pred: Prediction):
        p, g = pred.outputs["mask"], sample.reference["mask"]
        c = np.zeros((2, 4))
        for r, sl in enumerate((slice(0, SIZE // 2), slice(SIZE // 2, SIZE))):
            a, b = p[:, sl], g[:, sl]
            c[r] = [(a & b).sum(), (a & ~b).sum(), (~a & b).sum(), (~a & ~b).sum()]
        return {"cnt": c}

    @staticmethod
    def _iou(c):
        d = c[0] + c[1] + c[2]
        return float(c[0] / d) if d else float("nan")

    def reduce(self, counts, keys=None):
        c = sum((x["cnt"] for x in counts), np.zeros((2, 4)))
        t = c.sum(0)
        return {"iou": self._iou(t), "iou_near": self._iou(c[0]), "iou_far": self._iou(c[1]),
                "precision": float(t[0] / (t[0] + t[1])) if t[0] + t[1] else float("nan"),
                "recall": float(t[0] / (t[0] + t[2])) if t[0] + t[2] else float("nan")}

    def score(self, counts):
        return {"iou": self._iou(counts["cnt"].sum(0))}

    def support(self, counts):
        t = counts["cnt"].sum(0)
        return {"iou": float(t[0] + t[2])}                    # ground-truth cells


class ObjMetric(Metric):
    """ragged counts: 'err' (n_gt,) centre error of every GT box to the nearest predicted blob (capped at 24), 'n' (2,) blobs, boxes"""
    name = "objs"
    headline = ("center_err",)
    better = {"center_err": "low", "count_abs_err": "low"}

    def frame(self, sample: Sample, pred: Prediction):
        from scipy import ndimage
        lab, n = ndimage.label(pred.outputs["mask"])
        cents = [np.array(ndimage.center_of_mass(pred.outputs["mask"], lab, k)) for k in range(1, n + 1)
                 if (lab == k).sum() >= 6]
        gts = sample.reference["boxes"]
        err = []
        for cx, cy, _, _ in gts:                                      # array axes: rows = y, columns = x
            d = [np.hypot(c[1] - cx, c[0] - cy) for c in cents]
            err.append(min(min(d), 24.0) if d else 24.0)
        return {"err": np.asarray(err, float), "n": np.array([len(cents), len(gts)], float)}

    def reduce(self, counts, keys=None):
        err = np.concatenate([c["err"] for c in counts]) if counts else np.zeros(0)
        n = np.stack([c["n"] for c in counts]) if counts else np.zeros((0, 2))
        return {"center_err": float(err.mean()) if len(err) else float("nan"),
                "count_abs_err": float(np.abs(n[:, 0] - n[:, 1]).mean()) if len(n) else float("nan")}

    def score(self, counts):
        return {"center_err": float(counts["err"].mean()) if len(counts["err"]) else float("nan")}

    def support(self, counts):
        return {"center_err": float(len(counts["err"]))}


# ------------------------------------------------------------------------------------------------ Views and Task
def _box_shape(b, color="k", ls="--"):
    cx, cy, w, h = b
    return Shape("polygon", [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)],
                 color=color, ls=ls, lw=1.0)


class OverviewView(View):
    """input | ground truth | prediction (GT boxes dashed) | errors highlighted (FP red, FN blue)"""
    name = "overview"
    description = "ground truth next to the prediction, errors highlighted"

    def render(self, sample: Sample, pred: Prediction, ctx: ViewContext) -> Page:
        g, p = sample.reference["mask"], pred.outputs["mask"]
        tp, fp, fn = error_counts(p, g)
        ext = (0, SIZE, 0, SIZE)
        boxes = [_box_shape(b) for b in sample.reference["boxes"]]
        gray = lambda m: np.where(m[..., None], np.array([0.25, 0.25, 0.25]), np.array([1.0, 1.0, 1.0]))      # noqa: E731  (red / blue are for errors only)
        note = "PLANTED: " + sample.meta["planted"] if sample.meta.get("planted") else ""
        return Page(
            title=f"toy overview  {note}",
            panels=[Panel("raster", "input (mask + noise)", sample.get("input"), extent=ext, cmap="gray", vmin=-1, vmax=2),
                    Panel("rgb", "ground truth", gray(g), extent=ext, shapes=boxes),
                    Panel("rgb", "prediction (GT boxes dashed)", gray(p), extent=ext, shapes=boxes),
                    Panel("rgb", f"errors: TP {tp}  FP {fp}  FN {fn}", error_rgb(p, g), extent=ext)],
            ncols=4, legend=ERROR_LEGEND, panel_size=(3.6, 3.6))


class ToyTask(Task):
    name = "toy"

    def metrics(self) -> List[Metric]:
        return [SegMetric(), ObjMetric()]

    def views(self) -> Dict[str, View]:
        return {"overview": OverviewView()}

    def checks(self, sample: Sample, pred: Prediction) -> List[Flag]:
        out = []
        if sample.reference["mask"].any() and not pred.outputs["mask"].any():
            out.append(Flag("empty prediction", "warn", 0, "the ground truth has objects, the prediction none"))
        if not np.isfinite(sample.inputs).all():
            out.append(Flag("input finite", "fail", None, "NaN in the input"))
        return out

    def guide(self) -> str:
        return "toy world: dark = rectangles; in the error panel red = false positive, blue = false negative, grey = true positive."


def build(n_scenes: int = 8, n_frames: int = 12, bias: float = 0.0, memory: float = 0.5, seed: int = 0, name: str = "toy"):
    """the entry point of an adapter: (Source, Subject, Task)"""
    return ToySource(n_scenes, n_frames, seed), ToySubject(bias, memory, name), ToyTask()
