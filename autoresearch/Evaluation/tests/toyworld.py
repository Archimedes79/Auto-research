"""The world of the engine checks: the discs of the minimal adapter with three frames corrupted on purpose (an outlier search must find exactly these), a network with a memory, one
number where higher is better and one where lower is."""
from __future__ import annotations

import numpy as np

from .. import FrameRef, Metric, Page, Panel, Prediction, Sample, Subject, TapSpec, Task, View
from ..comparison.maps import confusion, scores
from ..minimal import Discs

PLANTED = (("toy-3", 5), ("toy-3", 6), ("toy-1", 2))


class World(Discs):
    """4 scenes of 8 frames; in the planted frames a noise burst drowns the disc"""
    def __init__(self, seed: int = 0):
        super().__init__()
        self.seed = seed

    def scenes(self):
        return [f"toy-{i}" for i in range(4)]

    def frames(self, scene):
        return [FrameRef(scene, t, key=f"{scene}:{t}") for t in range(8)]

    def load(self, ref):
        s = int(ref.scene[-1])
        yy, xx = np.mgrid[:32, :32]
        truth = (xx - (6 + 2 * ref.index + s + self.seed)) ** 2 + (yy - 16) ** 2 < 36
        spread = 3.0 if (ref.scene, ref.index) in PLANTED else 0.2
        noise = np.random.default_rng(1000 * self.seed + 100 * s + ref.index).normal(0, spread, truth.shape)
        return Sample(ref, inputs=(truth + noise).astype(np.float32), reference=truth)

    def describe(self):
        return {"dataset": "toy", "seed": self.seed}


def blur(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    return sum(p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)) / 9.0


class Smooth(Subject):
    """3 x 3 blur -> a memory (the average with the previous frame, carried along a scene) -> threshold at 0.5 + bias"""
    warmup = 2

    def __init__(self, bias: float = 0.0):
        self.bias, self.mem = bias, None

    def reset(self):
        self.mem = None

    def taps(self):
        return [TapSpec("blur", "map"), TapSpec("memory", "map")]

    def describe(self):
        return {"class": "Smooth", "bias": self.bias}

    def predict(self, sample, taps=()):
        b = blur(sample.inputs)
        self.mem = b if self.mem is None else 0.5 * self.mem + 0.5 * b
        tapped = {"blur": b, "memory": self.mem.copy()}
        return Prediction(outputs=self.mem > 0.5 + self.bias, taps={k: v for k, v in tapped.items() if k in taps})


class Seg(Metric):
    name, headline = "seg", ("iou",)

    def frame(self, sample, pred):
        return {"c": confusion(pred.outputs, sample.reference)}

    def reduce(self, counts, keys=None):
        return scores(sum((c["c"] for c in counts), np.zeros(4, dtype=np.int64)))

    def support(self, counts):
        return {"iou": float(counts["c"][0] + counts["c"][2])}                 # ground-truth cells


class Wrong(Metric):
    """wrong cells per frame, lower is better; the counts are ragged (one entry per wrong row of the picture)"""
    name, headline, better = "wrong", ("cells",), {"cells": "low"}

    def frame(self, sample, pred):
        per_row = (pred.outputs != sample.reference).sum(1)
        return {"rows": per_row[per_row > 0]}

    def reduce(self, counts, keys=None):
        return {"cells": float(np.mean([c["rows"].sum() for c in counts])) if counts else float("nan")}


class Picture(View):
    def render(self, sample, pred, ctx):
        return Page("toy", [Panel("image", "ground truth", sample.reference), Panel("image", "prediction", pred.outputs)], ncols=2)


class ToyTask(Task):
    name = "toy"

    def metrics(self):
        return [Seg(), Wrong()]

    def views(self):
        return {"overview": Picture()}
