"""The adapter of the toy problem for autoresearch/Evaluation: the sets, the three boxes Source / Subject / Task, the view, and the timing.
Everything else (names, folders, checkpoints, identities, the Runner) is the core's default (Evaluation/defaults.py)."""
from __future__ import annotations

import hashlib
import json
import os
import time as clock

import numpy as np
import torch

import Evaluation.comparison.maps as maps
from Evaluation import FrameRef, Metric, Page, Panel, Prediction, Sample, Source, Subject, Task, View
from Evaluation.contracts import TapSpec
from Evaluation.defaults import Defaults
from Evaluation.render.figure import ERROR_LEGEND, error_rgb
from Evaluation.torch_taps import count_ops
from gt import generate as GT
from net.model.net import DiscNet
from net.prep import shards as S

ROOT = S.ROOT
METRICS = ("seg",)
EVALUATOR_FILES = (os.path.abspath(__file__), os.path.abspath(S.__file__), os.path.abspath(maps.__file__))   # every file that shapes a number


def evaluator_now():
    return hashlib.sha256(b"".join(open(p, "rb").read() for p in EVALUATOR_FILES)).hexdigest()[:12]


def _config(net):
    with open(os.path.join(net, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def sets():
    count = {}
    for s in S.manifest()["shards"]:
        count[s["split"]] = count.get(s["split"], 0) + s["scenes"]
    return count


# ------------------------------------------------------------------------------------------------ the three boxes
class ToySource(Source):
    def __init__(self, split, root, supersample):
        self.split, self.signature = split, GT.signature(supersample)
        self.data = S.load_split(split, root, self.signature)
        self.rows = {(str(s), int(i)): k for k, (s, i) in enumerate(zip(self.data["scene"], self.data["index"]))}
        self.names = sorted({s for s, _ in self.rows})

    def scenes(self):
        return list(self.names)

    def frames(self, scene):
        return [FrameRef(scene, i, key=f"{scene}/{i}") for i in sorted(i for s, i in self.rows if s == scene)]

    def load(self, ref):
        k = self.rows[(ref.scene, ref.index)]
        return Sample(ref, inputs=self.data["frames"][k], reference=self.data["masks"][k])

    def describe(self):
        return {"class": "ToySource", "dataset": "toy-disc", "data_hash": S.manifest_hash(), "split": self.split,
                "set_hash": hashlib.sha256(json.dumps(self.names).encode()).hexdigest()[:12], "scenes": len(self.names), "frames": len(self.rows),
                "gt_signature": self.signature, "gt_version": self.signature, "evaluator_hash": evaluator_now()}


class ToySubject(Subject):
    def __init__(self, net, device="cpu"):
        cfg = _config(net)
        torch.set_num_threads(cfg.get("threads", 2))
        ck = torch.load(Defaults(ROOT).checkpoint(net), map_location="cpu", weights_only=False)
        self.model = DiscNet(cfg["width"])
        state = {k[len("model."):]: v for k, v in ck["state"].items() if k.startswith("model.")}
        state.update(ck["ema"])                                        # the averaged weights are what is evaluated
        self.model.load_state_dict(state)
        self.model.eval()
        self.device, self.width = "cpu", cfg["width"]
        self._trunk = None
        self.model.mid.register_forward_hook(lambda mod, inp, out: setattr(self, "_trunk", out))

    def predict(self, sample, taps=()):
        x = torch.from_numpy(np.array(sample.inputs, np.float32))[None, None]
        t0 = clock.perf_counter()
        with torch.no_grad():
            logits = self.model(x)                                     # the model's own forward: an edit of it is what is evaluated
        ms = (clock.perf_counter() - t0) * 1000.0
        tapped = {"trunk": torch.relu(self._trunk)[0].numpy()} if "trunk" in taps else {}
        return Prediction(outputs=(logits[0, 0] > 0).numpy(), taps=tapped, info={"predict_ms": ms, "device": self.device})

    def taps(self):
        return [TapSpec("trunk", kind="feature_map", stage="trunk", note="the features before the last convolution")]

    def parameters(self):
        return int(sum(p.numel() for p in self.model.parameters()))

    def ops(self, sample):
        x = torch.from_numpy(np.array(sample.inputs, np.float32))[None, None]
        return count_ops(lambda: self.model(x))

    def describe(self):
        return {"class": "ToySubject", "width": self.width, "parameters": self.parameters()}


class SegIoU(Metric):
    name, headline, better = "seg", ("iou", "precision", "recall"), {"iou": "high", "precision": "high", "recall": "high"}

    def frame(self, sample, pred):
        return {"c": maps.confusion(pred.outputs, sample.reference)}                  # additive: [tp, fp, fn, tn]

    def reduce(self, counts, keys=None):
        return maps.scores(sum((c["c"] for c in counts), np.zeros(4, dtype=np.int64)))

    def support(self, counts):
        return {"iou": float(counts["c"][0] + counts["c"][2])}                        # the number of ground-truth pixels


class Overview(View):
    def render(self, sample, pred, ctx):
        truth, found = sample.reference, pred.outputs
        panels = [Panel("image", "input", sample.inputs, cmap="gray"), Panel("image", "ground truth", truth, cmap="gray"),
                  Panel("image", "prediction", found, cmap="gray"), Panel("image", "errors", error_rgb(found, truth))]
        return Page(f"toy {sample.ref.label}", panels, ncols=4, legend=ERROR_LEGEND, panel_size=(2.6, 2.6))


class ToyTask(Task):
    name = "disc"

    def metrics(self):
        return [SegIoU()]

    def views(self):
        return {"overview": Overview()}

    def guide(self):
        return "input, ground truth, prediction, errors (the colours are named in the legend)"


def source(net, of, workers=0, control=""):
    if control:
        raise ValueError("the toy has no control inputs")
    cfg = _config(net)
    return ToySource(of, cfg["data_root"], cfg.get("gt_supersample", GT.SUPERSAMPLE))


def subject(net, src, device="cpu"):
    return ToySubject(net, device)


def task(net, metrics=METRICS):
    unknown = set(metrics) - set(METRICS)
    if unknown:
        raise ValueError(f"unknown metrics {sorted(unknown)}: the toy has {list(METRICS)}")
    return ToyTask()


def time(net, n=200):
    """milliseconds per frame at batch 1 (the forward pass and the threshold, no loading), operations and parameters; the machine's load is part of
    the result: a number from a busy machine says little"""
    cfg = _config(net)
    src, sub = ToySource("val", cfg["data_root"], cfg.get("gt_supersample", GT.SUPERSAMPLE)), ToySubject(net)
    sample = src.load(src.frames(src.scenes()[0])[0])
    bench, ops = sub.benchmark(sample, n=n, warmup=20), sub.ops(sample)
    load = os.getloadavg()[0]
    return {**bench, "parameters": sub.parameters(), "gmacs": ops["gmacs"], "ops_note": ops["note"], "threads": torch.get_num_threads(), "device": "cpu",
            "load_1min": round(load, 2), "quiet": load < 1.0, "protocol": "batch 1, one val frame, predict() = forward pass and threshold"}
