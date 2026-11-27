"""The BEV system as a project of the evaluation class (`autoresearch/Evaluation/api.py`, named in `evaluation/evaluation.json`): what a network, a set and a frame are here.

    AR_PROJECT=../problems/bev_fusion RESEARCH_RUNS=<runs root>/bev_fusion  python -m Evaluation Evaluation.run t21_ctl_s0 small.val --gt-key v13
                                                    (from autoresearch/; the whole interface: `python -m Evaluation methods`)

It adds no logic of its own: the Source and the Subject are `bev_adapter.build`, the sets are `bev_sets.py` (the training's `splits/`), the Task is `_task_only` below (the metric
definitions without a model; maps / boxes are always on, `curves`, `fusion`, `velocity` switch their families on, `board` is the score of the problem). A network is a run folder
(its config.json and checkpoints; by path or by name in the run root, `Evaluation/defaults.py`); everything of an evaluation goes under its `eval/` (`eval/results/<id>` the
predictions, `eval/suite/quant/<id>` the scored evaluation).

The memory is always carried (D70: whole scenes in time order from the network's own start, the frames asked for decide what is kept and scored), the predictions are stored, so a
new GT is scored without the network, and the network is run by the Runner (D72, `Evaluation/netrunner.py`), which reads no ground truth: a prediction carries no validation loss,
so `loss.*` / `val_loss/*` are not among the numbers of an Evaluation.
Limits: the identity of a network reads its last checkpoint, so predictions of a run whose checkpoints were archived or slimmed are found under the identity of the file that was there, and a
changed file is another network; a network is addressed by the name of its run folder (`runs/x` and `archive/x` would share the id `x@...`).
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
import sys
from types import SimpleNamespace
from typing import Any, Dict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_CODE = {"path": None}                           # the code snapshot of the first network built in this process: its modules stay imported, no other snapshot can run here


def _abs(net: str) -> str:
    return os.path.abspath(net)


_PREDICTION_FILES = (os.path.join(ROOT, "evaluation", "bev_adapter.py"), os.path.join(ROOT, "evaluation", "bev_free.py"),
                     os.path.join(ROOT, "..", "..", "autoresearch", "Evaluation", "netrunner.py"), os.path.join(ROOT, "..", "..", "autoresearch", "Evaluation", "torch_taps.py"))       # the code that makes a prediction: the output
# decoding of both paths, the frame loop (memory carry / reset) and the taps; part of the identity of the predictions, so an edit of any of them predicts again


def _claim_code(net: str) -> str:
    """the code of this run (its snapshot, else the live training/ and net/) is imported into this process once and for all: any later network with other code is refused, whatever it is asked for
    (a task, a timing, a probe, a source). -> the code folder"""
    from Evaluation.api import EvaluationError
    code = os.path.abspath(os.path.join(_abs(net), "code")) if os.path.isdir(os.path.join(_abs(net), "code")) else os.path.join(ROOT, "training")
    if _CODE["path"] not in (None, code):
        raise EvaluationError(f"one network per process: this process imported the code of {_CODE['path']}, {net} has {code}; run it in another process (compare the evaluations by their ids)")
    _CODE["path"] = code
    return code


def _A():
    from evaluation import bev_adapter
    return bev_adapter


def sets() -> Dict[str, Any]:
    """val / train (the run's own lists: no count here) and every `<split>.<part>` of splits/ with its number of scenes"""
    from evaluation import bev_sets
    return {"val": None, "train": None, **bev_sets.counts()}


def _file_hash(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _tree_hash(*folders: str) -> str:
    """the hash of the python files of the folders (name and content, line endings ignored): the code a run without a snapshot is evaluated with"""
    h = hashlib.sha1()
    for p in (p for folder in folders for p in sorted(glob.glob(os.path.join(folder, "*.py")))):
        with open(p, "rb") as f:
            h.update(os.path.basename(p).encode() + hashlib.sha1(f.read().replace(b"\r\n", b"\n")).digest())
    return h.hexdigest()[:12]


def identity(net: str) -> Dict[str, Any]:
    """what the predictions of this network depend on, read without loading it: the last checkpoint (name and the hash of the whole file), the config, the code (the snapshot's record, or a
    hash of the live code for a run that has none: it is imported then), and Subject.predict / the output decoding in bev_adapter.py and bev_free.py (the two paths give the same predictions)"""
    net = _abs(net)
    ckpts = sorted(glob.glob(os.path.join(net, "ckpt_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint (ckpt_*.pt) in {net}")
    last = ckpts[-1]
    try:
        with open(os.path.join(net, "code", "SNAPSHOT.txt"), encoding="utf-8") as f:
            code = f.read().strip()
    except OSError:
        code = "live code " + _tree_hash(*[os.path.join(ROOT, d) for d in ("training", "net", "net/prep", "net/model", "net/loss")])
    adapter = hashlib.sha1("".join(_file_hash(p) for p in _PREDICTION_FILES).encode()).hexdigest()[:12]
    return {"run": os.path.basename(os.path.normpath(net)), "checkpoint": os.path.basename(last), "sha1": _file_hash(last), "config": _file_hash(os.path.join(net, "config.json")), "code": code, "adapter": adapter}


def source(net: str, of: str, workers: int = 2, control: str = ""):
    """the Source of the set `of` for this run (val / train: its own lists, `<split>.<part>`: a file of splits/); the Subject is built with it (on the CPU) and handed out by `subject`"""
    from evaluation import bev_sets
    bev_sets.parse(of)                                                  # TEST and malformed names are refused before anything is built
    _claim_code(net)
    src, subj, _ = _A().build(net, device="cpu", split=of, workers=workers, control=control)
    src._bev_subject = subj
    return src


def subject(net: str, src, device: str):
    """the Subject `source` built with it, on `device` (a model built on the CPU is moved; cuda asked for and not there is an error, never a quiet CPU run)"""
    subj = getattr(src, "_bev_subject", None)
    if subj is None:
        raise RuntimeError("the Subject of this source was handed out or released already")
    if str(device).startswith("cuda") and not str(subj.device).startswith("cuda"):
        import torch
        from Evaluation.api import EvaluationError
        if not torch.cuda.is_available():
            raise EvaluationError(f"device {device} was asked for and CUDA is not available (pass device=cpu to run on the CPU)")
        subj.model.to(device)
        subj.loss.to(device)
        subj.device = device
    src._bev_subject = None
    return subj


def release(src) -> None:
    """drop the Subject a source holds (a check or a rescoring never runs it)"""
    src._bev_subject = None
    import gc
    gc.collect()                                                        # a built network and its nuScenes tables are several GB: give them back before the next build


def free_source(net: str, of: str, workers: int = 2, control: str = ""):
    """the Source the Runner reads: the same set, frames and dataset keys as `source`, without the ground truth (bev_free.FreeSource); the Subject travels with it"""
    from evaluation.bev_free import free_source as make
    built = source(net, of, workers, control)
    free = make(built)
    built._bev_subject = None                                           # the free Source holds it now
    return free


def free_subject(net: str, src, device: str):
    """the network without the validation loss (bev_free.FreeSubject over the BEVSubject `subject` hands out)"""
    from evaluation.bev_free import FreeSubject
    return FreeSubject(subject(net, src, device))


METRICS = ("maps", "boxes")


def _task_only(run: str, curves: bool = False, fusion: bool = True, velocity: bool = False):
    """the Task of a run without loading its checkpoint (the metric definitions only)"""
    import json
    from evaluation import bev_adapter as B
    from evaluation.bev_task import BEVTask
    import netpath
    run = os.path.abspath(run)
    os.chdir(netpath.SITE)
    B.setup_paths(run)
    ns = B._chain()
    with open(os.path.join(run, "config.json")) as fh:
        cfg = ns.T.TrainConfig(**{k: v for k, v in json.load(fh).items() if not k.startswith("_")})
    model, _, _ = ns.T.build(cfg)
    refine = bool(cfg.model.get("refine")) and hasattr(model, "boxes")
    t = BEVTask(model.grid, B.task_spec(model), [g for g, _ in ns.NUSC_GROUPS], refine=refine)
    t.curves, t.fusion, t.velocity, t.junction = curves, fusion, velocity, None
    return t


def task(net: str, metrics=METRICS):
    """the metric definitions of this run without loading it; `metrics` = maps, boxes, fusion (always on since D31; the name is still accepted), curves, velocity
    (and board, the score of the problem: always there, made of the maps and boxes counts)"""
    unknown = sorted(set(metrics) - {"maps", "boxes", "curves", "fusion", "velocity", "board"})
    if unknown:
        raise ValueError(f"unknown metrics {unknown}: maps, boxes, curves, fusion, velocity, board")
    _claim_code(net)
    t = _task_only(net, "curves" in metrics, True, "velocity" in metrics)       # the fusion-area scoring is always on (D31: the network's field of view is the headline)
    metrics_of = t.metrics
    def metrics_now():
        from evaluation.bev_task import Board7Metric
        ms = [m for m in metrics_of() if m.name != "loss"]                # the Runner reads no ground truth, so a prediction carries no validation loss (D72)
        return ms + [Board7Metric(next(m for m in ms if m.name == "seg"), next(m for m in ms if m.name == "det"))]
    t.metrics = metrics_now
    return t


def reference(src, ref):
    """the ground truth of a frame without its inputs: what a stored prediction is scored against (the dataset's own GT cache, then the training's per-side targets)"""
    gts = src.ns.sample_targets([src.ds._gt(ref.key)])
    return {dt: gts[dt][0] for dt in gts}


def _step(checkpoint: str):
    m = re.search(r"(\d+)", checkpoint or "")
    return int(m.group(1)) if m else None


def time(net: str, n: int = 30) -> Dict[str, Any]:
    from Evaluation.api import EvaluationError
    from evaluation import bev_timing
    _claim_code(net)
    try:
        return bev_timing.run_timing(_abs(net), n=n)
    except SystemExit as ex:                                           # the idle-GPU guard refuses with a message
        raise EvaluationError(str(ex)) from None


def probe(net: str, frames: int = 3, device: str = "cpu") -> Dict[str, Any]:
    from evaluation import bev_modules
    _claim_code(net)
    return bev_modules.probe(_abs(net), frames=frames, device=device)


def evaluator_now():
    """the evaluator hash of the code as it is now (what a stored evaluation was scored with is in its record): `show_numbers` warns when they differ"""
    from evaluation import bev_adapter
    return bev_adapter.evaluator_hash()


COMPARE_IGNORE = {"source": ("feature_cache", "control", "evaluator_hash")}   # the arms may differ in what the model is fed; the evaluator hash only guards a resume
MEMORY_TAP = "memory_conf"                                                   # the tap whose mean is the fill of the camera memory: the D70 health check (empty at the first frame of a scene, filling after)
