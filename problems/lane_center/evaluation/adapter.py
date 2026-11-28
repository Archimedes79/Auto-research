"""The adapter of the lane problem for autoresearch/Evaluation (the evaluation step of a run, the anchor test, the pictures of a report).

The network itself ran in the step `predict` (evaluation/lane_predict.py: the live dump of B' and the token head, at the batch sizes of its GPU
parity). The Subject here only reads that dump and decodes it at the run's decode (config "decode": the link thresholds, the keep threshold,
the none gate; jtoken_eval.decode_variant + frame_prediction): what the system delivers. The metric is the project's own scorer
(qgraph_eval.score_frames, lean, inside the 2 m lattice): its record of a frame is kept whole as the frame's counts, and the numbers of any set
of frames are qgraph_eval.graph_numbers of their records - the same numbers as jtoken_eval's cards (sums of integer counts; the float length
sums of vert100 only to the last bits). A scene is the scene number of poses_val.npz; VAL = 873 frames in 22 scenes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("evaluation", "training", "net/prep", "net/model", "net/loss", "training/data", "training/data/prep", "training/train", "../.."):
    p_ = os.path.normpath(os.path.join(ROOT, sub_))
    if p_ not in sys.path:
        sys.path.insert(0, p_)
import bflc_paths as BP  # noqa: E402
from Evaluation import FrameRef, Metric, Page, Panel, Prediction, Sample, Shape, Source, Subject, Task, View  # noqa: E402
from Evaluation.defaults import file_hash  # noqa: E402

METRICS = ("graph",)
HEADLINE = ("topo_F1", "junction_F1", "F1@1.0", "lanes_found", "dup_frac", "vert100_dp", "Q", "topo_P", "topo_R")
LOWER = ("dup_frac", "vert100_dp")
LATTICE = 2.0
_FRAMES = {}


def _qe():
    import qgraph_eval as QE
    return QE


def frames_of(split):
    """the scorer's frames of a split (qgraph_eval.Frame: k, row, token, scene, gt) in the order of data/splits.json"""
    if split not in _FRAMES:
        QE = _qe()
        rows = np.asarray(json.load(open(os.path.join(ROOT, "data", "splits.json")))[split], int)
        tokens = json.load(open(os.path.join(BP.B123, "feats", "val_tokens.json")))
        scene = np.load(os.path.join(BP.B123, "poses_val.npz"))["scene"]
        gt = pickle.load(open(os.path.join(BP.B123, "graph_gt_val.pkl"), "rb"))
        _FRAMES[split] = [QE.Frame(k, int(r), tokens[r], int(scene[r]), gt[r]) for k, r in enumerate(rows)]
    return _FRAMES[split]


def sets():
    return {s: len({f.scene for f in frames_of(s)}) for s in ("val", "test")}


def checkpoint(net):
    path = os.path.join(net, "model.pt")
    if not os.path.isfile(path):
        raise OSError(f"{net}: no model.pt")
    return path


# ------------------------------------------------------------------------------------------------ the three boxes
class LaneSource(Source):
    def __init__(self, split):
        self.split = split
        self.fr = frames_of(split)
        self.by_scene = {}
        for f in self.fr:
            self.by_scene.setdefault(str(f.scene), []).append(f)
        self.refs = {}

    def scenes(self):
        return sorted(self.by_scene, key=int)

    def frames(self, scene):
        out = []
        for i, f in enumerate(self.by_scene[scene]):
            ref = FrameRef(scene, i, key=f.token)
            self.refs[(scene, i)] = f
            out.append(ref)
        return out

    def load(self, ref):
        if (ref.scene, ref.index) not in self.refs:
            self.frames(ref.scene)
        f = self.refs[(ref.scene, ref.index)]
        return Sample(ref, inputs=f.token, reference=f)

    def describe(self):
        gt = os.path.join(BP.B123, "graph_gt_val.pkl")
        return {"class": "LaneSource", "split": self.split, "frames": len(self.fr), "scenes": len(self.by_scene), "gt_signature": file_hash(gt)[:12],
                "gt_version": "graph_gt_val.pkl", "data_hash": file_hash(os.path.join(ROOT, "data", "splits.json"))[:12], "evaluator_hash": evaluator_now()}


class LaneSubject(Subject):
    """the delivered graph of every frame, from the run's dump at the run's decode"""

    def __init__(self, net, split):
        import jtoken_eval as JE
        cfg = json.load(open(os.path.join(net, "config.json"), encoding="utf-8"))
        a = argparse.Namespace(run=os.path.basename(os.path.normpath(net)), seed=cfg["seed"], split=split, live=True, every=1)
        r = JE.read_dump(a, os.path.dirname(os.path.normpath(net)), 0)
        JE.check_record(r["fr"], r["valid"])
        d = cfg["decode"]
        self.decode, self.preds = d, {}
        for t, f, v in zip(r["toks"], r["fr"], r["valid"]):
            dec = JE.decode_variant(f, v, d["link_thr"], d["keep_thr"], d["none_gate"], d["link_thr_long"])
            self.preds[f["token"]] = JE.frame_prediction(t, dec, "rec", d["keep_thr"])
        self.meta = {k: r["meta"].get(k) for k in ("bs_b", "bs_head", "device", "env", "tokens_exact")}

    def predict(self, sample, taps=()):
        return Prediction(outputs=self.preds[sample.inputs], info={"device": "dump"})

    def taps(self):
        return []

    def describe(self):
        return {"class": "LaneSubject", "decode": self.decode, "dump": self.meta}


class LaneGraph(Metric):
    name, headline = "graph", HEADLINE
    better = {k: ("low" if k in LOWER else "high") for k in HEADLINE}

    def frame(self, sample, pred):
        import jtok as JT
        QE = _qe()
        with JT.lattice(LATTICE):
            _, r = QE.score_frames([pred.outputs], [sample.reference], records=True, lean=True)
        return {"rec": np.frombuffer(pickle.dumps(r["graph"][0], protocol=4), np.uint8)}

    def reduce(self, counts, keys=None):
        QE = _qe()
        recs = [pickle.loads(np.asarray(c["rec"], np.uint8).tobytes()) for c in counts]
        if not recs:
            return {k: float("nan") for k in HEADLINE}
        out = {k: float(v) for k, v in QE.graph_numbers(recs).items() if isinstance(v, (int, float, np.floating, np.integer))}
        return {k: out[k] for k in (keys or out) if k in out}

    def support(self, counts):
        return {"topo_F1": 1.0}


class Overview(View):
    """the delivered graph (red) over the ground truth graph (green), lane frame (x right, y forward), the 100 m square"""

    def render(self, sample, pred, ctx):
        QE = _qe()
        graph, _ = QE.readout(pred.outputs, 0.5, 0.5)
        raw, gt = QE.to_raw(graph), sample.reference.gt
        shapes = [Shape("line", np.asarray(p, float)[:, :2], color="tab:green", lw=2.0) for p in gt["pts"] if len(p) > 1]
        shapes += [Shape("line", np.asarray(p, float)[:, :2], color="tab:red", lw=1.2) for p in raw["pts"] if len(p) > 1]
        blank = np.ones((2, 2, 3))
        title = " ".join(f"{k} {v:.3f}" for k, v in ctx.scores.items() if k.split(".")[-1] in ("topo_F1", "F1@1.0"))
        return Page(f"lane {sample.ref.label}  {title}", [Panel("rgb", "ground truth (green), delivered graph (red)", blank, extent=(-50, 50, -50, 50),
                                                                shapes=shapes)], ncols=1, panel_size=(6.0, 6.0))


class LaneTask(Task):
    name = "lane_center"

    def metrics(self):
        return [LaneGraph()]

    def views(self):
        return {"overview": Overview()}

    def guide(self):
        return "the delivered lane-centre graph (red) over the ground-truth graph (green), 100 m around the car"


def source(net, of, workers=0, control=""):
    if control:
        raise ValueError("the lane problem has no control inputs")
    return LaneSource(of.split(".")[-1])


def subject(net, src, device="cpu"):
    return LaneSubject(net, src.split)


def task(net, metrics=METRICS):
    unknown = set(metrics) - set(METRICS)
    if unknown:
        raise ValueError(f"unknown metrics {sorted(unknown)}: the lane problem has {list(METRICS)}")
    return LaneTask()


EVALUATOR_FILES = [os.path.abspath(__file__)] + [os.path.join(ROOT, "evaluation", f) for f in ("qgraph_eval.py", "jtoken_eval.py")] + \
    [os.path.join(ROOT, "evaluation", "sparse_lanes", f) for f in ("bootstrap.py", "graph_eval.py", "sparse_metrics.py", "topo_metrics.py")] + \
    [os.path.join(ROOT, "evaluation", "lane_task", "measures.py")]


def evaluator_now():
    return hashlib.sha256(b"".join(open(p, "rb").read() for p in EVALUATOR_FILES)).hexdigest()[:12]
