"""The interface of the evaluation: one class, `Evaluation(project)`, plain data in and out, every method also a command (`python -m evalsuite Evaluation.run ...`, apicli.py; the pattern of
INTERFACE.md at the root of autoresearch/).

    ev = Evaluation()                                              # the project is AR_PROJECT (default); its adapter says what a network and its data are (below)
    r = ev.run("runs/t23_C_s1", "large.val", gt_key="v8")         # predict (the network runs once on what is missing), then score: a Result
    r.quant();  r.by("scene");  r.worst("seg", n=5);  r.cost();  r.report();  r.card()
    ev.compare(a_id, b_id)                                         # paired bootstrap, b - a (a = the control);   ev.qual_frame(r["id"], "scene-0061/12")
    ev.check("runs/t23_C_s1", "large.val")                         # the dry run: what `run` would do, with its problems; the data and the checkpoint are opened, no frame runs

One network per process: a project's network code (its modules) is imported once per process, so a second network of another code snapshot cannot be run in the same one (the adapter may
refuse it). Run each network in its own process and compare the two evaluations by their ids, which only reads what was stored.

`run` is `predict` + `score`. Only `predict` (hence `run`) and `time`, `probe`, `qual_frame` execute the network; everything else reads what was stored. D70: a network with a memory always
runs warm: whole scenes in time order from the network's own start; the frames you ask for decide which results are kept and scored, a frame already stored is not predicted again (a subset
of an earlier selection costs no network). What was predicted is a ResultSet (results.py), one per identity of the question "this network on this data"; scoring it again with another
ground truth or metric needs the Source (for the ground truth) but not the network. Ids: `<network>@r<hash>` (predictions), `<network>@e<hash>` (a scored evaluation of chosen frames, a ground
truth and metrics); the same question gives the same id. `run` here is not `autoresearch/tools/run.py`: it books no ledger row and takes no GPU slot; a number that feeds a verdict is made
with `run.py ... -- python -m evalsuite Evaluation.run ...`, this direct call is a diagnostic.

Frames: a text or a Frames object. `val`, `small.val` = every frame of a set; `spread:1000:large.val` = 1000 frames evenly spaced over all its frames (the scenes they lie in run up to
the last of them); `random:3[:SEED]:val` = 3 random scenes (seed 0 when none is named); `scenes:scene-0061,scene-0103:val` = those scenes; `frames:scene-0061/12,scene-0003/5:val` = those frames
(their scenes run whole up to them). The set is last and defaults to `val`; it must be one the project defines (`sets`).

The adapter (a module of the project, named in `<project>/evaluation/evalsuite.json` as {"adapter": "evaluation.bev_eval"}, or passed to Evaluation as a module or its name): functions
    name(network) -> str, out(network) -> folder, network(name) -> path, sets() -> {set: scenes}, identity(network) -> dict   (cheap: no model is loaded)
    source(network, of, workers, control) -> Source,  subject(network, source, device) -> Subject,  task(network, metrics) -> Task
  optional: resolve(text) -> network path (a run name, a relative path), release(source), reference(source, ref) -> the ground truth of a frame without loading the inputs,
  inputs(source) -> what the source was fed, COMPARE_IGNORE, card(...), time(network, n), probe(network, frames, device), MEMORY_TAP = the name of a tap whose mean is the fill of the
  network's memory (the health check: below).
Intermediate results: `predict(..., taps=("fused", ...))` stores the named taps of the Subject (its `taps()`) with the predictions; a frame stored without a tap is predicted again when the
tap is asked for. The health of the memory (D70) is checked on every scene the network runs, from the MEMORY_TAP means: empty at the first frame, growing over the frames; a scene that
fails is a warning in the result, never an error.
Numpy and the standard library only here."""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .contracts import FrameRef, Prediction, Source
from .quantitative import Quantitative
from .results import Locked, ResultSet, digest, result_id, writer_lock
from .runner import Runner
from .sampling import Sampler

CARRY = "carry"                                                               # the memory is always carried along a scene (D70)
MEMORY_START_MAX, MEMORY_GROWTH_MIN, MEMORY_MIN_FRAMES = 0.05, 0.02, 4        # the health of a carried memory (D70): empty at the first frame of a scene, filling over its frames
RESULT_DROP = ("gt_signature", "gt_version", "evaluator_hash", "set_hash", "scenes", "frames", "split")      # what a source says about the GT and the set: not what the network was fed


class EvaluationError(Exception):
    """the message says what to fix"""


# ------------------------------------------------------------------------------------------------ frames
class Frames:
    """which frames of which set: a value; `resolve(source)` -> (the scenes to run, the frames to keep and score)"""
    KINDS = {"all": 0, "spread": 1, "random": 2, "scenes": 1, "frames": 1}

    def __init__(self, kind: str = "all", args: Sequence[Any] = (), of: str = "val"):
        if kind not in self.KINDS or len(args) != self.KINDS[kind]:
            raise EvaluationError(f"frames {kind!r} takes {self.KINDS.get(kind)} argument(s), got {len(args)}")
        self.kind, self.args, self.of = kind, tuple(args), of

    @classmethod
    def all(cls, of: str = "val") -> "Frames":
        return cls("all", (), of)

    @classmethod
    def spread(cls, n: int, of: str = "val") -> "Frames":
        return cls("spread", (int(n),), of)

    @classmethod
    def random(cls, n: int, seed: int, of: str = "val") -> "Frames":
        return cls("random", (int(n), int(seed)), of)

    @classmethod
    def scenes(cls, names: Sequence[str], of: str = "val") -> "Frames":
        return cls("scenes", (tuple(names),), of)

    @classmethod
    def pick(cls, ids: Sequence[str], of: str = "val") -> "Frames":
        return cls("frames", (tuple(ids),), of)

    @classmethod
    def parse(cls, text: "str | Frames") -> "Frames":
        if isinstance(text, Frames):
            return text
        parts = str(text).split(":")
        kind = parts[0]
        if kind not in cls.KINDS or kind == "all":
            return cls.all(str(text))                                           # a bare set name
        usage = {"spread": "spread:N[:SET]", "random": "random:N[:SEED][:SET]", "scenes": "scenes:NAME,NAME[:SET]", "frames": "frames:SCENE/INDEX,SCENE/INDEX[:SET]"}[kind]
        rest = parts[1:]
        if not rest or len(rest) > 3:
            raise EvaluationError(f"frames {text!r}: {usage}")
        try:
            if kind == "random":
                n, seed = int(rest[0]), 0
                if len(rest) > 1 and rest[1].lstrip("-").isdigit():              # random:3:7:val, random:3:7, random:3:val
                    seed, rest = int(rest[1]), rest[:1] + rest[2:]
                args, of = (n, seed), (rest[1] if len(rest) > 1 else "val")
            elif kind == "spread":
                args, of = (int(rest[0]),), (rest[1] if len(rest) > 1 else "val")
            else:
                args, of = (tuple(x for x in rest[0].split(",") if x),), (rest[1] if len(rest) > 1 else "val")
        except ValueError:
            raise EvaluationError(f"frames {text!r}: a number is expected: {usage}") from None
        if len(rest) > 2:
            raise EvaluationError(f"frames {text!r}: {usage}")
        return cls(kind, args, of)

    def __str__(self) -> str:
        if self.kind == "all":
            return self.of
        a = [",".join(x) if isinstance(x, tuple) else str(x) for x in self.args]
        return ":".join([self.kind] + a + [self.of])

    __repr__ = __str__

    def __eq__(self, other) -> bool:
        return isinstance(other, Frames) and str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))

    def resolve(self, source: Source) -> Tuple[List[str], List[FrameRef]]:
        """the scenes to run (whole, in time order) and the frames to keep; an unknown scene or frame is an error that lists what exists"""
        have = list(source.scenes())
        if self.kind == "all":
            keep = [r for s in have for r in source.frames(s)]
        elif self.kind == "spread":
            keep = Sampler(source, 0).spread(self.args[0])
        elif self.kind == "random":
            keep = [r for s in Sampler(source, self.args[1]).scenes(self.args[0]) for r in source.frames(s)]
        elif self.kind == "scenes":
            bad = [s for s in self.args[0] if s not in have]
            if bad or not self.args[0]:
                raise EvaluationError(f"scenes {bad or 'none named'}: not in {self.of} ({len(have)} scenes, e.g. {', '.join(have[:3])})")
            keep = [r for s in self.args[0] for r in source.frames(s)]
        else:
            keep = []
            for text in self.args[0]:
                parsed = FrameRef.parse(text)
                if parsed is None or parsed[0] not in have or parsed[1] >= len(source.frames(parsed[0])):
                    raise EvaluationError(f"frame {text!r}: not a frame of {self.of} (scene/index, e.g. {have[0]}/0)")
                keep.append(source.frames(parsed[0])[parsed[1]])
        if not keep:
            raise EvaluationError(f"frames {self}: no frame selected")
        keep = sorted(set(keep))
        return sorted({r.scene for r in keep}), keep


# ------------------------------------------------------------------------------------------------ the adapter
ADAPTER_FUNCTIONS = ("name", "out", "network", "sets", "source", "subject", "task", "identity")


def load_adapter(spec: Any, project: Optional[str]):
    """the adapter module: given (a module or object with the functions), named in `spec` (a module path), or named in <project>/evaluation/evalsuite.json"""
    project = os.path.abspath(project or os.environ.get("AR_PROJECT") or ".")
    if spec is not None and not isinstance(spec, str):
        mod = spec
    else:
        if spec is None:
            cfg = os.path.join(project, "evaluation", "evalsuite.json")
            try:
                with open(cfg, encoding="utf-8") as f:
                    spec = json.load(f)["adapter"]
            except (OSError, ValueError, KeyError):
                raise EvaluationError(f"no adapter: pass one, or write {cfg} as {{\"adapter\": \"module.path\"}}") from None
        if project not in sys.path:
            sys.path.insert(0, project)
        try:
            mod = importlib.import_module(spec)
        except ImportError as e:
            raise EvaluationError(f"adapter {spec!r} cannot be imported: {e}") from None
    missing = [f for f in ADAPTER_FUNCTIONS if not hasattr(mod, f)]
    if missing:
        raise EvaluationError(f"the adapter lacks {', '.join(missing)}")
    return mod


# ------------------------------------------------------------------------------------------------ the class
class Evaluation:
    def __init__(self, project: Optional[str] = None, adapter: Any = None):
        self.ad = load_adapter(adapter, project)
        self._tasks: Dict[Tuple[str, Tuple[str, ...]], Any] = {}
        self._networks: Dict[str, str] = {}                               # the networks this object has seen: name -> path (an id carries only the name)

    # ---------------------------------------------------------------------------------------- what exists, and the dry run
    def sets(self) -> Dict[str, Any]:
        """the sets of scenes the project defines: name -> number of scenes (`val` and `train` are the network's own lists: no number)"""
        return dict(self.ad.sets())

    def list(self, network: str) -> List[Dict[str, Any]]:
        """the stored predictions and evaluations of a network, with their size"""
        net = self._net(network)
        out, name, rows = self.ad.out(net), self.ad.name(net), []
        base = os.path.join(out, "results")
        for d in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            try:
                rs = ResultSet.open(os.path.join(base, d))
            except (OSError, ValueError):
                continue
            rows.append({"id": f"{name}@{d}", "kind": "predictions", "frames": rs.n_frames(), "megabytes": rs.size_mb(), "created": rs.manifest.get("created")})
        base = os.path.join(out, "suite", "quant")
        for d in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            try:
                with open(os.path.join(base, d, "manifest.json"), encoding="utf-8") as f:
                    ev = json.load(f).get("evaluation")
            except (OSError, ValueError):
                continue
            if ev:
                rows.append({"id": f"{name}@{d}", "kind": "evaluation", "frames": ev["n_frames"], "set": ev["frames"], "gt": ev["gt"], "metrics": ev["metrics"]})
        return rows

    def check(self, network: str, frames: str = "val", metrics: Sequence[str] = ("maps", "boxes"), workers: int = 0) -> Dict[str, Any]:
        """the dry run: what `run` would do with these frames and what is wrong (`ok`, `problems`). No frame runs; the data and the checkpoint are opened (a BEV network is built on the CPU)"""
        problems: List[str] = []
        res: Dict[str, Any] = {"network": network, "frames": str(frames), "ok": False, "problems": problems}
        try:
            net, fr = self._net(network), Frames.parse(frames)
            res["network"] = net
            ident = self._net_identity(net)
            self._task(net, tuple(metrics))
            source = self._source(net, fr.of, workers, "")
        except (EvaluationError, ValueError, KeyError, OSError) as e:
            problems.append(str(e))
            return res
        try:
            desc = source.describe()
            scenes, keep = fr.resolve(source)
            rid = result_id(self._identity(ident, source))
            rs = self._stored(net, rid)
            have = sum(1 for r in keep if rs and rs.has(r))
            want = {r.label for r in keep}
            to_run = 0                                                           # the network runs each scene up to its last kept frame, the frames before it included (the memory)
            for sc in scenes:
                kept = [r for r in source.frames(sc) if r.label in want]
                to_run += (kept[-1].index + 1) if any(not (rs and rs.has(r)) for r in kept) else 0
            res.update({"scenes_run": len(scenes), "frames_kept": len(keep), "frames_stored": have, "frames_to_run": to_run, "results": f"{self.ad.name(net)}@{rid}", "out": self.ad.out(net),
                        "gt": desc.get("gt_version") or desc.get("gt_signature"), "state": "carry (whole scenes in time order, D70)",
                        "network_runs": have < len(keep), "id": f"{self.ad.name(net)}@{self._eval_id(rid, keep, desc, metrics)}"})
            if not desc.get("gt_signature"):
                problems.append("the data has no ground-truth signature: the numbers could not be told from another GT's")
            self._probe_reference(source, keep, problems)
        except (EvaluationError, ValueError, KeyError, OSError) as e:
            problems.append(str(e))
        finally:
            self._release(source)
        res["ok"] = not problems
        return res

    # ---------------------------------------------------------------------------------------- running (the only code that executes the network)
    def predict(self, network: str, frames: str = "val", taps: Sequence[str] = (), device: str = "cuda", workers: int = 2, control: str = "", progress: int = 50) -> Dict[str, Any]:
        """run the network on the frames that are not stored yet (whole scenes in time order, warm: D70) and store what it predicts, with the intermediate results `taps` when asked;
        -> the id of the predictions, what was done and the warnings"""
        net, fr = self._net(network), Frames.parse(frames)
        ident = self._net_identity(net)
        source = self._source(net, fr.of, workers, control)
        try:
            desc = source.describe()
            self._need_gt(desc, None)
            scenes, keep = fr.resolve(source)
            return self._predict(net, source, desc, ident, scenes, keep, tuple(taps), device, progress)[1]
        finally:
            self._release(source)

    def score(self, results: str, frames: str = "val", gt_key: Optional[str] = None, metrics: Sequence[str] = ("maps", "boxes"), workers: int = 0, progress: int = 50) -> "Result":
        """score stored predictions (an id `<network>@r<hash>`) on the frames against the ground truth; the network is not run; frames that were not predicted are an error"""
        name, _, rid = str(results).partition("@")
        net, fr = self._net(self.ad.network(name) if name not in self._networks else self._networks[name]), Frames.parse(frames)
        rs = self._stored(net, rid)
        if rs is None:
            raise EvaluationError(f"no predictions {results}: nothing at {os.path.join(self.ad.out(net), 'results', rid)}")
        source = self._source(net, fr.of, workers, rs.manifest["identity"]["inputs"].get("control") or "")
        try:
            desc = source.describe()
            scenes, keep = fr.resolve(source)
            self._need_gt(desc, gt_key)
            missing = sum(not rs.has(r) for r in keep)
            if missing:
                raise EvaluationError(f"{missing} of the {len(keep)} frames are not in {results}: `predict` them first, or use `run`")
            return self._score(net, source, desc, rs, keep, fr, metrics, 0.0, 0, True, progress)
        finally:
            self._release(source)

    def run(self, network: str, frames: str = "val", gt_key: Optional[str] = None, metrics: Sequence[str] = ("maps", "boxes"), taps: Sequence[str] = (), device: str = "cuda",
            workers: int = 2, control: str = "", progress: int = 50) -> "Result":
        """predict (what is not stored yet), then score against the ground truth; gt_key = the GT version the numbers must be on (refused when the data has another);
        metrics = what the project offers (BEV: maps, boxes, curves, fusion, velocity). -> a Result: a dict with the id, whose methods read the evaluation"""
        net, fr, metrics = self._net(network), Frames.parse(frames), tuple(metrics)
        ident = self._net_identity(net)
        self._task(net, metrics)                                          # a typo in the metrics fails here, not after the network ran
        source = self._source(net, fr.of, workers, control)
        try:
            desc = source.describe()
            self._need_gt(desc, gt_key)
            scenes, keep = fr.resolve(source)
            problems: List[str] = []
            self._probe_reference(source, keep, problems)                 # the ground truth of the first kept frame is readable: before the network runs, not after
            if problems:
                raise EvaluationError(problems[0])
            rs, pinfo = self._predict(net, source, desc, ident, scenes, keep, tuple(taps), device, progress)
            return self._score(net, source, desc, rs, keep, fr, metrics, pinfo["seconds_network"], pinfo["frames_predicted"], pinfo["frames_predicted"] == 0, progress)
        finally:
            self._release(source)

    # ------------------------------------------------------------ the two halves
    def _predict(self, net, source, desc, ident, scenes, keep, taps, device, progress):
        out = self.ad.out(net)
        identity = self._identity(ident, source)
        rid = result_id(identity)
        rs = ResultSet.create(os.path.join(out, "results", rid), identity)
        todo = [r for r in keep if not rs.has(r, taps)]
        t0 = time.time()
        if todo:
            try:
                with writer_lock(rs.path):
                    rs = ResultSet.open(rs.path, "a")                          # another writer may have stored frames while we waited for the lock
                    todo = [r for r in keep if not rs.has(r, taps)]
                    if todo:
                        self._run_network(net, source, scenes, keep, rs, ident, desc.get("gt_signature"), taps, device, progress)
            except Locked as e:
                raise EvaluationError(str(e)) from None
            if self._net_identity(net) != ident:
                raise EvaluationError(f"{net} changed while the network ran (a new checkpoint?): the predictions belong to the old one; run again")
        return rs, {"results": f"{self.ad.name(net)}@{rid}", "frames_kept": len(keep), "frames_predicted": len(todo), "seconds_network": round(time.time() - t0, 1),
                    "predictions_reused": not todo, "warnings": self._warnings(rs, {r.scene for r in keep})}

    def _run_network(self, net, source, scenes, keep, rs, ident, gt, taps, device, progress):
        subject = self.ad.subject(net, source, device)
        if self._net_identity(net) != ident:                                   # a training may have written a new checkpoint while the network was loaded: refuse before a frame is stored
            subject.close()
            raise EvaluationError(f"{net} changed while it was loaded (a new checkpoint?): nothing was stored; run again")
        declared = {t.name for t in subject.taps()}
        bad = [t for t in taps if t not in declared]
        if bad:
            subject.close()
            raise EvaluationError(f"unknown taps {bad}: the network offers {sorted(declared)}")
        mem = getattr(self.ad, "MEMORY_TAP", None)
        mem = mem if mem in declared else None                                 # the health check needs the project's memory tap
        names = list(taps) + ([mem] if mem and mem not in taps else [])
        runner = Runner(source, subject, CARRY)
        want, t0, n, ms = {r.label for r in keep}, time.time(), 0, []
        total = sum(1 for r in keep if not rs.has(r, taps))
        try:
            for sc in scenes:
                kept = [r for r in source.frames(sc) if r.label in want]
                if all(rs.has(r, taps) for r in kept):
                    continue
                refs = source.frames(sc)[: kept[-1].index + 1]                 # nothing after the last frame asked for is needed; the frames before it always run (the memory)
                fill = []
                for sample, pred in runner.iter_scene(sc, refs, taps=names):
                    if mem and mem in pred.taps:
                        fill.append(float(np.mean(pred.taps[mem])))
                    if sample.ref.label in want and not rs.has(sample.ref, taps):
                        info = {**pred.info, "gt_at_predict": gt, "device": str(getattr(subject, "device", device)), **({"memory_fill": fill[-1]} if fill else {})}
                        rs.add(sample.ref, Prediction(pred.outputs, {k: v for k, v in pred.taps.items() if k in taps}, info))
                        n += 1
                        ms.append(float(pred.info.get("predict_ms", 0.0)))
                        if progress and n % progress == 0:
                            print(f"[predict {rs.id}] {n}/{total} frames, {(time.time() - t0) / n:.2f} s per frame", file=sys.stderr, flush=True)
                self._memory_health(rs, sc, fill)
        finally:
            rs.close(wall_s=time.time() - t0, cost={"median_ms": sorted(ms)[len(ms) // 2], "n": len(ms)} if ms else None)
            subject.close()

    @staticmethod
    def _memory_health(rs: ResultSet, scene: str, fill: List[float]) -> None:
        """D70: a carried memory is empty at the first frame of a scene and fills over its frames; a scene where it is not is recorded in the result set (a warning to every evaluation of it)"""
        rs.manifest.setdefault("health", {}).pop(scene, None)
        if not fill:
            return
        if fill[0] > MEMORY_START_MAX:
            rs.manifest["health"][scene] = f"{scene}: the memory is not empty at the first frame ({fill[0]:.2f} > {MEMORY_START_MAX}): not initialised"
        elif len(fill) >= MEMORY_MIN_FRAMES and max(fill) - fill[0] < MEMORY_GROWTH_MIN:
            rs.manifest["health"][scene] = f"{scene}: the memory does not fill over {len(fill)} frames ({fill[0]:.2f} -> {fill[-1]:.2f}): not used"

    @staticmethod
    def _warnings(rs: Optional[ResultSet], scenes, store=None, scored_gt: Optional[str] = None) -> List[str]:
        """what the reader of a number should know: from the frames the evaluation scored (the ground truth and the device they were predicted under) and from the memory's health"""
        out = []
        if store is not None:
            gts = sorted({str(fr["info"]["gt_at_predict"]) for fr in store.frames if fr["info"].get("gt_at_predict")})
            if gts and gts != [str(scored_gt)]:
                out.append(f"frames were predicted while the data had ground truth {gts}, the numbers are scored against {scored_gt}: what a prediction computes itself "
                           "(the val loss) belongs to the earlier ground truth; maps, boxes and the rest are scored against the present one")
            devs = sorted({fr["info"]["device"] for fr in store.frames if fr["info"].get("device")})
            if len(devs) > 1:
                out.append(f"the frames were predicted on different devices {devs}: their float results differ slightly")
        if rs is not None:
            out += [msg for sc, msg in rs.manifest.get("health", {}).items() if sc in scenes]
        return out

    def _score(self, net, source, desc, rs, keep, fr, metrics, seconds_network, predicted, reused, progress):
        out, name, metrics = self.ad.out(net), self.ad.name(net), tuple(metrics)
        eid = self._eval_id(rs.id, keep, desc, metrics)
        task = self._task(net, metrics)
        ref_of = (lambda ref: self.ad.reference(source, ref)) if hasattr(self.ad, "reference") else None
        quant = Quantitative(SimpleNamespace(out=os.path.join(out, "suite"), source=source, subject=None, task=task, state=CARRY))
        t1 = time.time()
        try:
            with writer_lock(os.path.join(out, "suite", "quant", eid)):
                store = quant.run(frames=keep, state=CARRY, name=eid, results=rs, reference=ref_of, progress=progress)
        except Locked as e:
            raise EvaluationError(str(e)) from None
        info = store.manifest.get("evaluation") or {"id": f"{name}@{eid}", "results": f"{name}@{rs.id}", "network": net, "frames": str(fr), "gt": desc.get("gt_version") or desc.get("gt_signature"),
                                                    "gt_signature": desc.get("gt_signature"), "metrics": list(metrics), "n_frames": len(store.frames), "n_scenes": len(store.scene_names()),
                                                    "control": rs.manifest["identity"]["inputs"].get("control") or "", "state": CARRY, "evaluator": desc.get("evaluator_hash")}
        if "evaluation" not in store.manifest:
            store.manifest["evaluation"] = info                               # written once: a later call of the same question does not overwrite its record
            store._write_manifest()
        return Result(self, {**info, "warnings": self._warnings(rs, {r.scene for r in keep}, store, info["gt_signature"]), "predictions_reused": reused, "frames_predicted": predicted, "seconds_network": seconds_network,
                             "seconds_score": round(time.time() - t1, 1)})

    # ------------------------------------------------------------ small helpers
    def _net(self, network: str) -> str:
        try:
            net = self.ad.resolve(network) if hasattr(self.ad, "resolve") else network
        except OSError as e:
            raise EvaluationError(str(e)) from None
        self._networks[self.ad.name(net)] = net
        return net

    def _net_identity(self, net: str) -> Dict[str, Any]:
        try:
            return self.ad.identity(net)
        except OSError as e:
            raise EvaluationError(f"{net}: {e}") from None

    def _identity(self, net_identity: Dict[str, Any], source: Source) -> Dict[str, Any]:
        inputs = self.ad.inputs(source) if hasattr(self.ad, "inputs") else {k: v for k, v in source.describe().items() if k not in RESULT_DROP}
        return {"network": net_identity, "inputs": inputs, "state": CARRY}

    @staticmethod
    def _eval_id(rid: str, keep: Sequence[FrameRef], desc: Dict[str, Any], metrics: Sequence[str]) -> str:
        return "e" + digest([rid, digest([[r.label, r.key] for r in keep], 40), desc.get("gt_signature"), sorted(metrics), desc.get("evaluator_hash")])

    @staticmethod
    def _need_gt(desc: Dict[str, Any], gt_key: Optional[str]) -> None:
        if not desc.get("gt_signature"):
            raise EvaluationError("the data has no ground-truth signature: numbers could not be told from those of another GT; fix the data adapter")
        if gt_key and gt_key not in (desc.get("gt_version"), desc.get("gt_signature")):
            raise EvaluationError(f"the data has ground truth {desc.get('gt_version') or desc.get('gt_signature')} ({desc.get('gt_signature')}), not {gt_key}")

    def _source(self, net: str, of: str, workers: int, control: str) -> Source:
        """the Source of a set; a set the project does not define is refused before anything is opened, and whatever the adapter raises (SystemExit included) is an EvaluationError"""
        known = self.ad.sets()
        if of not in known:
            raise EvaluationError(f"set {of!r}: not one of {sorted(known)}" + (" (the TEST split is not offered)" if "test" in of else ""))
        try:
            return self.ad.source(net, of, workers=workers, control=control)
        except SystemExit as e:
            raise EvaluationError(str(e.code) if e.code is not None else "the data adapter stopped") from None
        except (ValueError, KeyError, OSError) as e:
            raise EvaluationError(str(e.args[0] if isinstance(e, KeyError) and e.args else e)) from None

    def _probe_reference(self, source: Source, keep: Sequence[FrameRef], problems: List[str]) -> None:
        try:
            (self.ad.reference(source, keep[0]) if hasattr(self.ad, "reference") else source.load(keep[0]).reference)
        except Exception as e:                                                # noqa: BLE001  whatever the data adapter raises: its message is the problem
            problems.append(f"the ground truth of {keep[0].label} cannot be read: {type(e).__name__}: {e}")

    def _release(self, source: Source) -> None:
        if hasattr(self.ad, "release"):
            self.ad.release(source)

    def _stored(self, net: str, rid: str) -> Optional[ResultSet]:
        path = os.path.join(self.ad.out(net), "results", rid)
        return ResultSet.open(path) if os.path.exists(os.path.join(path, "manifest.json")) else None

    def _task(self, net: str, metrics: Sequence[str]):
        key = (net, tuple(metrics))
        if key not in self._tasks:
            try:
                self._tasks[key] = self.ad.task(net, tuple(metrics))
            except ValueError as e:
                raise EvaluationError(str(e)) from None
        return self._tasks[key]

    # ---------------------------------------------------------------------------------------- reading what was stored
    def _open(self, eval_id: str):
        """(info, store, quantitative) of a stored evaluation"""
        name, _, eid = str(eval_id).partition("@")
        if not eid.startswith("e"):
            raise EvaluationError(f"{eval_id!r} is not an evaluation id (<network>@e<hash>; `list NETWORK` shows them)")
        net = self._networks.get(name) or self._net(self.ad.network(name))
        path = os.path.join(self.ad.out(net), "suite", "quant", eid)
        from .store import RunStore
        try:
            store = RunStore.open(path)
        except OSError:
            raise EvaluationError(f"no evaluation {eval_id}: nothing at {path}") from None
        info = store.manifest.get("evaluation")
        if not info:
            raise EvaluationError(f"{eval_id} is a store without an evaluation record")
        task = self._task(info["network"], tuple(info["metrics"]))
        return info, store, Quantitative(SimpleNamespace(out=os.path.join(self.ad.out(info["network"]), "suite"), source=None, subject=None, task=task, state=CARRY))

    def _predictions(self, info: Dict[str, Any]) -> Optional[ResultSet]:
        return self._stored(info["network"], info["results"].partition("@")[2])

    def tap(self, results: str, frame: str, name: str = "") -> Dict[str, Any]:
        """the intermediate results stored with a frame's predictions: without `name` which taps there are, with it the path of its .npy file, its shape and dtype"""
        rname, _, rid = str(results).partition("@")
        net = self._networks.get(rname) or self._net(self.ad.network(rname))
        rs = self._stored(net, rid)
        parsed = FrameRef.parse(frame)
        if rs is None or parsed is None:
            raise EvaluationError(f"no predictions {results}" if rs is None else f"{frame!r} is not scene/index")
        ref = FrameRef(parsed[0], parsed[1])
        if not rs.has(ref):
            raise EvaluationError(f"{frame} is not in {results}")
        if not name:
            return {"results": results, "frame": ref.label, "taps": rs.tap_names(ref)}
        if name not in rs.tap_names(ref):
            raise EvaluationError(f"{frame} of {results} has no tap {name!r}; it has {rs.tap_names(ref)} (ask for it with `predict --taps {name}`)")
        arr = np.load(rs._tap_file(ref, name), mmap_mode="r")
        return {"results": results, "frame": ref.label, "tap": name, "path": rs._tap_file(ref, name), "shape": list(arr.shape), "dtype": str(arr.dtype)}

    def quant(self, id: str, match: Optional[str] = None) -> Dict[str, Any]:
        """the dataset-level numbers of an evaluation ('metric.key' -> value, only those containing `match`) and which way is better ('high' / 'low')"""
        info, store, q = self._open(id)
        tab = q.table(store)
        better = {f"{m.name}.{k}": m.direction(k) for m in q._metrics().values() for k in m.headline}
        keep = (lambda k: match in k) if match else (lambda k: True)
        return {"id": id, "n_frames": info["n_frames"], "gt": info["gt"], "numbers": {k: v for k, v in tab.items() if keep(k)}, "better": {k: v for k, v in better.items() if keep(k)},
                "warnings": self._warnings(self._predictions(info), {fr["scene"] for fr in store.frames}, store, info["gt_signature"])}

    def by(self, id: str, axis: str = "scene") -> Dict[str, Any]:
        """the headline numbers per scene (axis 'scene') or the scores of every frame (axis 'frame')"""
        _, store, q = self._open(id)
        if axis == "scene":
            return q.per_scene(store)
        if axis == "frame":
            return {fr["label"]: fr["scores"] for fr in store.frames}
        raise EvaluationError(f"axis {axis!r}: scene or frame")

    def worst(self, id: str, match: Optional[str] = None, n: int = 10) -> Dict[str, List[dict]]:
        """where the numbers come from: the scenes whose removal improves a headline number most and the frames furthest out in the bad direction (rows that contain `match`); a
        stored evaluation has no plausibility flags (they need the inputs)"""
        _, store, q = self._open(id)
        res = q.outliers(store, k=n)
        if match:
            res = {k: [row for row in v if match in json.dumps(row, default=str)] for k, v in res.items()}
        return res

    def compare(self, a: str, b: str, boot: int = 1000, seed: int = 0) -> Dict[str, Any]:
        """paired scene bootstrap of the headline numbers, b - a (a = the control, b = the variant); refused when the two do not answer the same question (other GT, other frames)"""
        _, sa, qa = self._open(a)
        _, sb, _ = self._open(b)
        try:
            return qa.compare(sa, sb, boot, seed, ignore=getattr(self.ad, "COMPARE_IGNORE", None))
        except ValueError as e:
            raise EvaluationError(str(e)) from None

    def cost(self, id: str) -> Dict[str, Any]:
        """ms per frame of the network as it ran (the median over the frames of the last session that ran frames, on that day's machine)"""
        info, store, q = self._open(id)
        rs = self._predictions(info)
        return {"id": id, "predict": rs.manifest.get("cost") if rs else None, "devices": sorted({fr["info"]["device"] for fr in store.frames if fr["info"].get("device")})}

    def report(self, id: str) -> str:
        """the html page of an evaluation (numbers, scenes by drag, outliers); returns its path"""
        info, store, q = self._open(id)
        return q.report(store, render=0)

    def card(self, id: str) -> Dict[str, Any]:
        """the metric card in the project's contract (what its decision tool reads); refused when the checkpoint changed since the predictions"""
        if not hasattr(self.ad, "card"):
            raise EvaluationError("this project's adapter defines no card")
        info, store, q = self._open(id)
        rs = self._predictions(info)
        self._same_network(info, rs)
        return self.ad.card(info, store, q, rs)

    def _same_network(self, info: Dict[str, Any], rs: Optional[ResultSet]) -> None:
        if rs is not None and self._net_identity(info["network"]) != rs.manifest["identity"]["network"]:
            raise EvaluationError(f"{info['network']} is not the network of these predictions any more (another checkpoint, config or code): "
                                  "what it would draw or certify is not what was evaluated")

    # ---------------------------------------------------------------------------------------- about the network itself
    def qual_frame(self, id: str, frame: str, view: str = "overview", device: str = "cpu", workers: int = 0) -> Dict[str, Any]:
        """draw one frame of an evaluation: the network runs on it again, on `device` and with the history of its scene (its scores can differ slightly from the stored ones when the
        evaluation ran on another device); returns the picture paths and the frame's scores"""
        info, store, q = self._open(id)
        self._same_network(info, self._predictions(info))
        from .suite import EvalSuite
        fr = Frames.parse(info["frames"])
        source = self._source(info["network"], fr.of, workers, info.get("control", ""))
        suite = EvalSuite(source, self.ad.subject(info["network"], source, device), self._task(info["network"], tuple(info["metrics"])),
                          out=os.path.join(self.ad.out(info["network"]), "suite"), state=CARRY)
        try:
            res = suite.qual.frame(frame, views=(view,), history="full")
        finally:
            suite.close()
            self._release(source)
        return {"frame": res.ref, "images": res.images, "scores": res.scores}

    def time(self, network: str, n: int = 30) -> Dict[str, Any]:
        """milliseconds per module and per frame at batch 1 (idle GPU only: the adapter refuses a busy one)"""
        if not hasattr(self.ad, "time"):
            raise EvaluationError("this project's adapter defines no timing")
        return self.ad.time(self._net(network), n)

    def probe(self, network: str, frames: int = 3, device: str = "cpu") -> Dict[str, Any]:
        """the module probe: statistics and dead channels of every module, the response when the camera or the radar is removed (frames = how many frames of the first scene)"""
        if not hasattr(self.ad, "probe"):
            raise EvaluationError("this project's adapter defines no probe")
        return self.ad.probe(self._net(network), frames, device)

    def selftest(self) -> Dict[str, Any]:
        """the checks of this interface on a toy project (a few seconds; no data, no network)"""
        from .selftest_api import run_all
        return run_all()


Evaluation.METHODS = tuple(n for n, f in vars(Evaluation).items() if callable(f) and not n.startswith("_"))      # the methods are the commands: no second list


class Result(dict):
    """what `run` returns: a plain dict (the id, the frames, the GT, what was done, warnings) whose methods read the evaluation of this id; pickled or copied it is the plain dict"""

    def __init__(self, ev: Evaluation, info: Dict[str, Any]):
        super().__init__(info)
        self._ev = ev

    def __reduce__(self):
        return (dict, (dict(self),))

    def quant(self, match: Optional[str] = None):
        return self._ev.quant(self["id"], match)

    def by(self, axis: str = "scene"):
        return self._ev.by(self["id"], axis)

    def worst(self, match: Optional[str] = None, n: int = 10):
        return self._ev.worst(self["id"], match, n)

    def compare(self, a: "Result", boot: int = 1000, seed: int = 0):
        """this evaluation (b) against `a` (the control): b - a"""
        return self._ev.compare(a["id"], self["id"], boot, seed)

    def cost(self):
        return self._ev.cost(self["id"])

    def report(self):
        return self._ev.report(self["id"])

    def card(self):
        return self._ev.card(self["id"])
