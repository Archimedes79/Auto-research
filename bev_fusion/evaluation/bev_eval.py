"""The BEV system as a project of the evaluation class (`autoresearch/evalsuite/api.py`, named in `evaluation/evalsuite.json`): what a network, a set and a frame are here.

    AR_PROJECT=../bev_fusion  python -m evalsuite Evaluation.run runs/t23_C_s1 small.val --gt-key v8      (from autoresearch/; the whole interface: `python -m evalsuite methods`)

It adds no logic of its own: the Source and the Subject are `bev_adapter.build` (so the objects, and with them the numbers, are those of `bev_suite.py eval`), the sets are `bev_sets.py`
(the training's `splits/`), the Task is `bev_suite._task_only` (the metric definitions without a model; the metric flags: maps / boxes are always on, `curves`, `fusion`, `velocity`
switch the families of `--curves --fusion --velocity`), the card is `bev_suite.contract_card`. A network is a run folder, `runs/<name>`; everything of an evaluation goes under its
`eval/` (`eval/results/<id>` the predictions, `eval/suite/quant/<id>` the scored evaluation).

What differs from `bev_suite.py eval`: the memory is always carried (D70: whole scenes in time order from the network's own start, the frames asked for decide what is kept and scored; the
old `eval` without `--warm` starts every frame cold), the predictions are stored, so a new GT is scored without the network, and the network is run by the Runner (D72), which reads no ground
truth: a prediction carries no validation loss, so `loss.*` / `val_loss/*` are not among the numbers of an Evaluation (`bev_suite.py eval` keeps them).
Limits: the identity of a network reads its last checkpoint, so predictions of a run whose checkpoints were archived or slimmed (keep3) are found under the identity of the file that was there, and a
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
from evaluation import bev_suite  # noqa: E402

_CODE = {"path": None}                           # the code snapshot of the first network built in this process: its modules stay imported, no other snapshot can run here


def _abs(net: str) -> str:
    return net if os.path.isabs(net) else os.path.join(ROOT, net)


def _claim_code(net: str) -> str:
    """the code of this run (its snapshot, else the live training/) is imported into this process once and for all: any later network with other code is refused, whatever it is asked for
    (a task, a timing, a probe, a source). -> the code folder"""
    from evalsuite.api import EvaluationError
    code = os.path.abspath(os.path.join(_abs(net), "code")) if os.path.isdir(os.path.join(_abs(net), "code")) else os.path.join(ROOT, "training")
    if _CODE["path"] not in (None, code):
        raise EvaluationError(f"one network per process: this process imported the code of {_CODE['path']}, {net} has {code}; run it in another process (compare the evaluations by their ids)")
    _CODE["path"] = code
    return code


def resolve(text: str) -> str:
    """a run name (`t23_C_s1`), a path relative to bev_fusion (`runs/t23_C_s1`) or an absolute one -> the run folder, as a path relative to bev_fusion when it lies inside it"""
    tried = [os.path.abspath(text if os.path.isabs(text) else os.path.join(ROOT, text)), os.path.abspath(os.path.join(ROOT, "runs", text))]
    for p in tried:
        if os.path.isfile(os.path.join(p, "config.json")):
            return os.path.relpath(p, ROOT) if p.startswith(ROOT + os.sep) else p
    raise FileNotFoundError(f"{text}: no run folder with a config.json (looked at {', '.join(tried)})")


def _A():
    os.chdir(ROOT)
    from evaluation import bev_adapter
    return bev_adapter


def name(network: str) -> str:
    return os.path.basename(os.path.normpath(network))


def network(nm: str) -> str:
    return os.path.join("runs", nm)


def out(net: str) -> str:
    return os.path.join(_abs(net), "eval")


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


def _tree_hash(folder: str) -> str:
    """the hash of the python files of a folder (name and content, line endings ignored): the code a run without a snapshot is evaluated with"""
    h = hashlib.sha1()
    for p in sorted(glob.glob(os.path.join(folder, "*.py"))):
        with open(p, "rb") as f:
            h.update(os.path.basename(p).encode() + hashlib.sha1(f.read().replace(b"\r\n", b"\n")).digest())
    return h.hexdigest()[:12]


def identity(net: str) -> Dict[str, Any]:
    """what the predictions of this network depend on, read without loading it: the last checkpoint (name and the hash of the whole file), the config, the code (the snapshot's record, or a
    hash of the live training/ code for a run that has none: it is imported then), and Subject.predict / the output decoding in bev_adapter.py and bev_free.py (the two paths give the same predictions)"""
    net = _abs(net)
    ckpts = sorted(glob.glob(os.path.join(net, "ckpt_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint (ckpt_*.pt) in {net}")
    last = ckpts[-1]
    try:
        with open(os.path.join(net, "code", "SNAPSHOT.txt"), encoding="utf-8") as f:
            code = f.read().strip()
    except OSError:
        code = "live training/ " + _tree_hash(os.path.join(ROOT, "training"))
    adapter = hashlib.sha1("".join(_file_hash(os.path.join(ROOT, "evaluation", f)) for f in ("bev_adapter.py", "bev_free.py")).encode()).hexdigest()[:12]
    return {"run": name(net), "checkpoint": os.path.basename(last), "sha1": _file_hash(last), "config": _file_hash(os.path.join(net, "config.json")), "code": code, "adapter": adapter}


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
        from evalsuite.api import EvaluationError
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


def task(net: str, metrics=("maps", "boxes")):
    """the metric definitions of this run without loading it; `metrics` = maps, boxes (always), curves, fusion, velocity"""
    from evaluation import bev_suite
    unknown = sorted(set(metrics) - {"maps", "boxes", "curves", "fusion", "velocity"})
    if unknown:
        raise ValueError(f"unknown metrics {unknown}: maps, boxes, curves, fusion, velocity")
    _claim_code(net)
    t = bev_suite._task_only(net, "curves" in metrics, "fusion" in metrics, "velocity" in metrics)
    metrics_of = t.metrics
    t.metrics = lambda: [m for m in metrics_of() if m.name != "loss"]       # the Runner reads no ground truth, so a prediction carries no validation loss (D72); `bev_suite.py eval` keeps it
    return t


def reference(src, ref):
    """the ground truth of a frame without its inputs: what a stored prediction is scored against (the dataset's own GT cache, then the training's per-side targets)"""
    gts = src.ns.sample_targets([src.ds._gt(ref.key)])
    return {dt: gts[dt][0] for dt in gts}


def _step(checkpoint: str):
    m = re.search(r"(\d+)", checkpoint or "")
    return int(m.group(1)) if m else None


def card(info: Dict[str, Any], store, quant, rs):
    """the metric card of an evaluation in the contract of autoresearch/tools/decide.py (what `bev_suite.py eval` writes as eval/card.json): the numbers with their directions,
    protocol = GT, memory carried, and a provenance block; `subset` names the frames when they are not the whole set"""
    from evaluation import bev_adapter
    from evaluation.bev_suite import better_of, contract_card, provenance_block, to_legacy
    from evalsuite.api import EvaluationError, Frames
    if info.get("evaluator") and info["evaluator"] != bev_adapter.evaluator_hash():
        raise EvaluationError(f"{info['id']} was scored with other evaluator code than the one now ({info['evaluator']} / {bev_adapter.evaluator_hash()}): score the predictions again "
                              f"(`score {info['results']}`, no network is run), then make the card of the new evaluation")
    tab = quant.table(store)
    flat, better = to_legacy(tab), better_of(tab, quant.s.task.metrics())
    ident = rs.manifest["identity"]["network"] if rs else {}
    run = _abs(info["network"])                                                # the run folder as a path that works from any working directory
    flat["info/step"], flat["info/n_samples"], flat["info/n_scenes"] = _step(ident.get("checkpoint")), len(store.frames), len(store.scene_names())
    fr = Frames.parse(info["frames"])
    devices = sorted({fr_["info"]["device"] for fr_ in store.frames if fr_["info"].get("device")})
    card_ = contract_card(run, fr.of, "", flat, better, len(store.frames), info.get("gt_signature"), info.get("state", "carry"), info.get("control", ""))
    if fr.kind != "all":
        card_["subset"] = str(fr)
    a = SimpleNamespace(run=run, ckpt_dir=None, device=",".join(devices) or "unknown", control=info.get("control", ""), split=fr.of)
    e = SimpleNamespace(n=len(store.frames), warm=True, prior=0)
    subj = {"checkpoint": ident.get("checkpoint"), "step": _step(ident.get("checkpoint")), "device": a.device, "code": ident.get("code")}
    card_["provenance"] = provenance_block(a, e, store, subj, info.get("gt_signature"), "", info.get("state", "carry"), start={}, avail=store.scene_names())
    card_["provenance"]["launcher"] = "evalsuite Evaluation.run: a direct call, no run.py (no ledger row, no GPU slot): a diagnostic unless it was started through run.py"
    return card_


def time(net: str, n: int = 30) -> Dict[str, Any]:
    from evalsuite.api import EvaluationError
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


def evaluator_extra() -> str:
    """the code that shapes the numbers outside the source's evaluator hash (bev_adapter.NUMBER_PATHS): this adapter (the reference, the metrics without the loss), the label-free path and
    bev_views.py / bev_annotation_pts.py (the masks that are every map's denominator, the boxes and the layer index: imported by the metrics but not in NUMBER_PATHS, a frozen file);
    it goes into the id of an evaluation, so an edit of any of them scores the stored predictions again instead of serving old numbers"""
    files = ("bev_eval.py", "bev_free.py", "bev_views.py", "bev_annotation_pts.py")
    return hashlib.sha1("".join(_file_hash(os.path.join(ROOT, "evaluation", f)) for f in files).encode()).hexdigest()[:12]


def evaluator_now():
    """the evaluator hash of the code as it is now (what a stored evaluation was scored with is in its record): `show_numbers` warns when they differ"""
    from evaluation import bev_adapter
    return bev_adapter.evaluator_hash()


COMPARE_IGNORE = bev_suite.COMPARE_IGNORE                                    # the arms may differ in what the model is fed; the evaluator hash only guards a resume
MEMORY_TAP = "memory_conf"                                                   # the tap whose mean is the fill of the camera memory: the D70 health check (empty at the first frame of a scene, filling after)
