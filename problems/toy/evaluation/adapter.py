"""The adapter of the toy problem for autoresearch/Evaluation: what a network and its data are (the functions the class Evaluation asks for), the three boxes
Source / Subject / Task, the metric card of an evaluation and the timing.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import time as clock

import numpy as np
import torch

import Evaluation.comparison.maps as maps
from Evaluation import FrameRef, Metric, Page, Panel, Prediction, Sample, Source, Subject, Task, View
from Evaluation.contracts import TapSpec
from Evaluation.render.figure import ERROR_LEGEND, error_rgb
from Evaluation.torch_taps import count_ops
from gt import generate as GT
from net.model.net import DiscNet
from net.prep import shards as S

ROOT = S.ROOT
EVALUATOR_FILES = (os.path.abspath(__file__), os.path.abspath(S.__file__), os.path.abspath(maps.__file__))   # every file that shapes a number


def _json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _short(data):
    return hashlib.sha256(data).hexdigest()[:12]


def evaluator_hash():
    return _short(b"".join(open(p, "rb").read() for p in EVALUATOR_FILES))


evaluator_now = evaluator_hash


# ------------------------------------------------------------------------------------------------ networks
def network(name):
    return os.path.join(ROOT, "runs", name)


def resolve(text):
    """a run id (E1.c_s0), a run folder name (E1_c_s0) or a path -> the run folder"""
    for p in (text, os.path.join(ROOT, text), network(os.path.basename(str(text).rstrip("/")).replace(".", "_"))):
        if os.path.isfile(os.path.join(p, "config.json")):
            return os.path.abspath(p)
    raise OSError(f"no run folder with a config.json for {text!r}")


def name(net):
    return os.path.basename(net)


def out(net):
    return os.path.join(net, "eval")


def config(net):
    return _json(os.path.join(net, "config.json"))


def checkpoint(net):
    found = sorted(glob.glob(os.path.join(net, "ckpt_*.pt")))
    if not found:
        raise OSError(f"{net}: no checkpoint")
    return found[-1]


def identity(net):
    with open(os.path.join(net, "config.json"), "rb") as f:
        cfg = _short(f.read())
    return {"config": cfg, "checkpoint": os.path.basename(checkpoint(net)), "checkpoint_hash": S.file_hash(checkpoint(net))[:12]}


def sets():
    count = {}
    for s in S.manifest()["shards"]:
        count[s["split"]] = count.get(s["split"], 0) + s["scenes"]
    return count


# ------------------------------------------------------------------------------------------------ the three boxes
class ToySource(Source):
    def __init__(self, split, root, supersample):
        self.split, self.signature = split, GT.signature(supersample)
        data = S.load_split(split, root, self.signature)
        self.data = data
        self.rows = {(str(s), int(i)): k for k, (s, i) in enumerate(zip(data["scene"], data["index"]))}
        self.names = sorted({s for s, _ in self.rows})

    def scenes(self):
        return list(self.names)

    def frames(self, scene):
        return [FrameRef(scene, i, key=f"{scene}/{i}") for i in sorted(i for s, i in self.rows if s == scene)]

    def load(self, ref):
        k = self.rows[(ref.scene, ref.index)]
        return Sample(ref, inputs=self.data["frames"][k], reference=self.data["masks"][k])

    def describe(self):
        return {"class": "ToySource", "dataset": "toy-disc", "data_hash": S.manifest_hash(), "split": self.split, "set_hash": _short(json.dumps(self.names).encode()),
                "scenes": len(self.names), "frames": len(self.rows), "gt_signature": self.signature, "gt_version": self.signature, "evaluator_hash": evaluator_hash()}


class ToySubject(Subject):
    def __init__(self, net, device="cpu"):
        cfg = config(net)
        torch.set_num_threads(cfg.get("threads", 2))
        ck = torch.load(checkpoint(net), map_location="cpu", weights_only=False)
        self.model = DiscNet(cfg["width"])
        state = {k[len("model."):]: v for k, v in ck["state"].items() if k.startswith("model.")}
        state.update(ck["ema"])                                        # the averaged weights are what is evaluated
        self.model.load_state_dict(state)
        self.model.eval()
        self.device, self.width = "cpu", cfg["width"]

    def predict(self, sample, taps=()):
        x = torch.from_numpy(np.array(sample.inputs, np.float32))[None, None]
        t0 = clock.perf_counter()
        with torch.no_grad():
            features = self.model.features(x)
            logits = self.model.head(features)
        ms = (clock.perf_counter() - t0) * 1000.0
        tapped = {"trunk": features[0].numpy()} if "trunk" in taps else {}
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
    name, headline, better = "seg", ("iou",), {"iou": "high"}

    def frame(self, sample, pred):
        return {"c": maps.confusion(pred.outputs, sample.reference)}                  # additive: [tp, fp, fn, tn]

    def reduce(self, counts, keys=None):
        return maps.scores(sum((c["c"] for c in counts), np.zeros(4, dtype=np.int64)))

    def support(self, counts):
        return {"iou": float(counts["c"][0] + counts["c"][2])}                        # the number of truth pixels


class Overview(View):
    def render(self, sample, pred, ctx):
        truth, found = sample.reference, pred.outputs
        panels = [Panel("image", "input", sample.inputs, cmap="gray"), Panel("image", "truth", truth, cmap="gray"), Panel("image", "prediction", found, cmap="gray"),
                  Panel("image", "errors", error_rgb(found, truth))]
        return Page(f"toy {sample.ref.label}", panels, ncols=4, legend=ERROR_LEGEND, panel_size=(2.6, 2.6))


class ToyTask(Task):
    name = "disc"

    def metrics(self):
        return [SegIoU()]

    def views(self):
        return {"overview": Overview()}

    def guide(self):
        return "input, truth, prediction, errors (the colours are named in the legend)"


def source(net, of, workers=0, control=""):
    if control:
        raise ValueError("the toy has no control inputs")
    cfg = config(net)
    return ToySource(of, cfg.get("data_root", ""), cfg.get("gt_supersample", GT.SUPERSAMPLE))


def subject(net, src, device="cpu"):
    return ToySubject(net, device)


def task(net, metrics=("seg",)):
    unknown = set(metrics) - {"seg"}
    if unknown:
        raise ValueError(f"unknown metrics {sorted(unknown)}: the toy has 'seg' (pass --metrics seg)")
    return ToyTask()


# ------------------------------------------------------------------------------------------------ card and cost
def _dirty(commit):
    """the evaluator files that differ from `commit` (a list; [] = the numbers were made by that commit's code), 'unknown' without a commit. Files outside this
    repository (a copy of the problem somewhere else) are left out"""
    if not commit:
        return "unknown"
    git = lambda *a: subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True)          # noqa: E731
    top = git("rev-parse", "--show-toplevel").stdout.strip()
    changed = []
    for path in EVALUATOR_FILES:
        rel = os.path.relpath(path, top).replace(os.sep, "/") if top else ".."
        if rel.startswith(".."):
            continue
        now, then = git("hash-object", path).stdout.strip(), git("rev-parse", "--verify", "--quiet", f"{commit}:{rel}").stdout.strip()
        if now != then:
            changed.append(rel)
    return changed


def card(info, store, quant, rs):
    """the metric card of one evaluation (autoresearch/schemas/card.schema.json): what decide.py reads"""
    net = info["network"]
    table, cfg, commit = quant.table(store), config(net), os.environ.get("AR_COMMIT", "")
    numbers = {k: float(table[f"seg.{k}"]) for k in ("iou", "precision", "recall", "f1")}
    numbers = {k: v for k, v in numbers.items() if np.isfinite(v)}                 # a ratio that is not defined (nothing predicted) is left out: a card holds numbers only
    frame_hash = _short(json.dumps(sorted(fr["label"] for fr in store.frames)).encode())
    return {"schema": 1, "name": name(net), "split": str(info["frames"]), "frames": info["n_frames"], "seed": cfg["seed"], "metrics": numbers,
            "better": {k: "higher" for k in numbers}, "commit": commit or "unknown (not started by run.py)", "date": clock.strftime("%Y-%m-%d %H:%M"),
            "info": {"eval_id": info["id"], "results": info["results"], "n_scenes": info["n_scenes"], "gt": info["gt"], "width": cfg["width"], "checkpoint": os.path.basename(checkpoint(net))},
            "provenance": {"schema": 1, "gt": {"version": info["gt_signature"], "signature": info["gt_signature"]},
                           "data": {"manifest_hash": S.manifest_hash(), "frame_hash": frame_hash, "split": str(info["frames"]), "n": info["n_frames"]},
                           "evaluator": {"code_hash": info.get("evaluator"), "commit": commit or "unknown", "dirty": _dirty(commit)},
                           "subject": {"run": name(net), "checkpoint_hash": identity(net)["checkpoint_hash"]}}}


def time(net, n=200):
    """milliseconds per frame at batch 1 (the forward pass and the threshold, no loading), operations and parameters; the box it was measured on is part of the result"""
    src, net_ = ToySource("val", config(net).get("data_root", ""), config(net).get("gt_supersample", GT.SUPERSAMPLE)), ToySubject(net)
    sample = src.load(src.frames(src.scenes()[0])[0])
    bench, ops = net_.benchmark(sample, n=n, warmup=20), net_.ops(sample)
    load = os.getloadavg()[0]
    return {**bench, "parameters": net_.parameters(), "gmacs": ops["gmacs"], "ops_note": ops["note"], "threads": torch.get_num_threads(), "device": "cpu",
            "load_1min": round(load, 2), "quiet": load < 1.0, "protocol": "batch 1, one val frame, predict() = forward pass and threshold; valid only on a quiet machine"}
