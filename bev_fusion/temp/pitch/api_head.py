"""The interface of the evaluation: one class, `Evaluation(project)`, plain data in and out, every method also a command (`python -m evalsuite Evaluation.run ...`, apicli.py; the pattern of
INTERFACE.md at the root of autoresearch/).

    ev = Evaluation()                                              # the project is AR_PROJECT (default); its adapter says what a network and its data are (below)
    r = ev.run("runs/t23_C_s1", "large.val", gt_key="v8")         # predict (the network runs once on what is missing), then score: a Result
    r.quant();  r.by("scene");  r.worst("seg", n=5);  r.cost();  r.report();  r.card()
    r.compare(control)                                             # paired bootstrap, r - control;   ev.qual_frame(r["id"], "scene-0061/12")
    ev.check("runs/t23_C_s1", "large.val")                         # the dry run: what `run` would do, with its problems; starts nothing

`run` is `predict` + `score`. Only `predict` (hence `run`) and `time`, `probe`, `qual_frame` execute the network; everything else reads what was stored. D70: a network with a memory always
runs warm: whole scenes in time order from the network's own start; the frames you ask for decide which results are kept and scored, a frame already stored is not predicted again (a subset
of an earlier selection costs no network). What was predicted is a ResultSet (results.py), one per identity of the question "this network on this data"; scoring it again with another
ground truth or metric needs the Source (for the ground truth) but not the network. Ids: `<network>@r<hash>` (predictions), `<network>@e<hash>` (a scored evaluation of chosen frames, a ground
truth and metrics); the same question gives the same id. `run` here is not `autoresearch/tools/run.py`: it books no ledger row and takes no GPU slot; a number that feeds a verdict is made
with `run.py ... -- python -m evalsuite Evaluation.run ...`, this direct call is a diagnostic.

Frames: a text or a Frames object. `val`, `small.val` = every frame of a set; `spread:1000:large.val` = 1000 frames evenly spaced over all its frames; `random:3:7:val` = 3 random scenes (seed
7); `scenes:scene-0061,scene-0103:val` = those scenes; `frames:scene-0061/12,scene-0003/5:val` = those frames (their scenes run whole). The set is last and defaults to `val`.

The adapter (a module of the project, named in `<project>/evaluation/evalsuite.json` as {"adapter": "evaluation.bev_eval"}, or passed to Evaluation as a module or its name): functions
    name(network) -> str, out(network) -> folder, network(name) -> path, sets() -> {set: scenes}, identity(network) -> dict   (cheap: no model is loaded)
    source(network, of, workers, control) -> Source,  subject(network, source, device) -> Subject,  task(network, metrics) -> Task
  optional: resolve(text) -> network path (a run name, a relative path), release(source), reference(source, ref) -> the ground truth of a frame without loading the inputs,
  inputs(source) -> what the source was fed, COMPARE_IGNORE, card(...), time(network, n), probe(network, frames, device).
Numpy and the standard library only here."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .contracts import FrameRef, Source
from .quantitative import Quantitative
from .results import Locked, ResultSet, result_id, writer_lock
from .runner import Runner
from .sampling import Sampler

RESULT_DROP = ("gt_signature", "gt_version", "evaluator_hash", "set_hash", "scenes", "frames", "split")      # what a source says about the GT and the set: not what the network was fed


class EvaluationError(Exception):
    """the message says what to fix"""


