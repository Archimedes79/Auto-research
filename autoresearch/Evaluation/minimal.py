"""The smallest project adapter: copy it to <problem>/evaluation/adapter.py and replace the bodies (README.md runs it, EXTENSIONS.md says what more an adapter can give).

World: 3 scenes ("videos") of 6 frames; a frame is a 32 x 32 picture with a moving disc plus noise. A network is a run folder; its config.json {"threshold": 0.5} is all it has learned."""
from __future__ import annotations

import json
import os

import numpy as np

from Evaluation import FrameRef, Metric, Page, Panel, Prediction, Sample, Source, Subject, Task, View
from Evaluation.comparison.maps import confusion, scores
from Evaluation.defaults import file_hash
from Evaluation.render.figure import ERROR_LEGEND, error_rgb

METRICS = ("iou",)                          # the metrics of an evaluation that names none


class Discs(Source):                        # 1. the data: scenes, their frames in time order, how to load one
    def __init__(self, split="val"):
        self.split = split

    def scenes(self):
        return ["demo-0", "demo-1", "demo-2"]

    def frames(self, scene):
        return [FrameRef(scene, t) for t in range(6)]

    def load(self, ref):
        s = int(ref.scene[-1])
        yy, xx = np.mgrid[:32, :32]
        truth = (xx - (6 + 3 * ref.index + 2 * s)) ** 2 + (yy - 16) ** 2 < 25
        noise = np.random.default_rng(100 * s + ref.index).normal(0, 0.25, truth.shape)
        return Sample(ref, inputs=truth + noise, reference=truth)          # inputs -> the network, reference -> metrics and pictures

    def describe(self):                     # required: which ground truth this is, and a hash of the code that shapes the numbers
        return {"dataset": "discs", "split": self.split, "gt_signature": "discs-1", "evaluator_hash": file_hash(os.path.abspath(__file__))[:12]}


class Threshold(Subject):                   # 2. the network: one frame in, one prediction out (a memory: reset() starts a scene)
    def __init__(self, threshold):
        self.threshold = threshold

    def predict(self, sample, taps=()):
        return Prediction(outputs=sample.inputs > self.threshold)


class IoU(Metric):                          # 3. a number: ADDITIVE counts of one frame; reduce() turns the counts of any frames into numbers
    name, headline = "iou", ("iou",)        # headline: what comparisons and the outlier search use

    def frame(self, sample, pred):
        return {"c": confusion(pred.outputs, sample.reference)}            # [tp, fp, fn, tn]

    def reduce(self, counts, keys=None):
        return scores(sum((c["c"] for c in counts), np.zeros(4, dtype=np.int64)))          # iou, precision, recall, f1


class Overview(View):                       # 4. a picture: a page of panels (data, no plotting code)
    def render(self, sample, pred, ctx):
        truth, found = sample.reference, pred.outputs
        return Page("disc", [Panel("image", "ground truth", truth), Panel("image", "prediction", found), Panel("image", "errors", error_rgb(found, truth))], legend=ERROR_LEGEND)


class DiscTask(Task):
    def metrics(self):
        return [IoU()]

    def views(self):
        return {"overview": Overview()}


# 5. the adapter: these four functions (folders, names and identities have defaults: defaults.py)
def sets():
    return {"val": 3}                       # set name -> number of scenes


def source(net, of, workers=0, control=""):
    return Discs(of)


def subject(net, src, device="cpu"):
    with open(os.path.join(net, "config.json"), encoding="utf-8") as f:
        return Threshold(json.load(f)["threshold"])


def task(net, metrics=METRICS):
    return DiscTask()


def checkpoint(net):                        # the file that holds the weights (default: the last ckpt_*.pt of the run folder)
    return os.path.join(net, "config.json")
