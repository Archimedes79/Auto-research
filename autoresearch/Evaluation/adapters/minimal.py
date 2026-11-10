"""The smallest working adapter: copy this file and replace the bodies. Everything the suite can do beyond this is in EXTENSIONS.md.

World: 3 scenes ("videos") of 6 frames; a frame is a 32 x 32 picture with one moving disc plus noise; the "network" thresholds it.
    python -m Evaluation --adapter Evaluation.adapters.minimal:build --out ../eval_out qual random 3
    python -m Evaluation --adapter Evaluation.adapters.minimal:build --out ../eval_out quant run --name a
A copy anywhere works too (it imports Evaluation, not its neighbours): --adapter path/to/my_adapter.py:build
"""
from __future__ import annotations

import numpy as np

from Evaluation import FrameRef, Metric, Page, Panel, Prediction, Sample, Source, Subject, Task, View
from Evaluation.comparison.maps import confusion, scores
from Evaluation.render.figure import ERROR_LEGEND, error_rgb


class MinSource(Source):                    # 1. the data: scenes ("videos"), their frames in time order, how to load one frame
    def scenes(self):
        return ["demo-0", "demo-1", "demo-2"]

    def frames(self, scene):
        return [FrameRef(scene, t) for t in range(6)]

    def load(self, ref):
        s = int(ref.scene[-1])
        yy, xx = np.mgrid[:32, :32]
        truth = (xx - (6 + 3 * ref.index + 2 * s)) ** 2 + (yy - 16) ** 2 < 25
        noise = np.random.default_rng(100 * s + ref.index).normal(0, 0.25, truth.shape)
        return Sample(ref, inputs=truth + noise, reference=truth)        # inputs -> the network, reference -> metrics and pictures


class MinSubject(Subject):                  # 2. the network: one frame in, one prediction out (reset() / warmup are for memory)
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def predict(self, sample, taps=()):
        return Prediction(outputs=sample.inputs > self.threshold)

    def describe(self):                     # optional: what the run record says about the network (a resume with other values is refused)
        return {"class": "MinSubject", "threshold": self.threshold}


class IoU(Metric):                          # 3. what to measure: ADDITIVE counts of one frame, and how counts of any frames become numbers
    name, headline = "iou", ("iou",)        # headline = the numbers that tables, outlier search and comparison use

    def frame(self, sample, pred):
        return {"c": confusion(pred.outputs, sample.reference)}               # [tp, fp, fn, tn]; the comparison itself is Evaluation/comparison/

    def reduce(self, counts, keys=None):
        return scores(sum((c["c"] for c in counts), np.zeros(4, dtype=np.int64)))        # iou, precision, recall, f1


class Overview(View):                       # 4. how to draw one frame: a Page of Panels (pure data, no plotting code)
    def render(self, sample, pred, ctx):
        g, p = sample.reference, pred.outputs
        return Page("minimal demo", [Panel("image", "ground truth", g), Panel("image", "prediction", p),
                                     Panel("image", "errors", error_rgb(p, g))], ncols=3, legend=ERROR_LEGEND)


class MinTask(Task):
    def metrics(self):
        return [IoU()]

    def views(self):
        return {"overview": Overview()}     # 'overview' is required


def build(threshold=0.5):                   # the entry point: (Source, Subject, Task); --arg threshold=0.3 reaches it
    return MinSource(), MinSubject(threshold), MinTask()
