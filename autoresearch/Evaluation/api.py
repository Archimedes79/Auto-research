"""The interface of the evaluation: one class, `Evaluation(project)`, plain data in and out, every method also a command (`python -m Evaluation <method>`, cli.py).

    ev = Evaluation()                                             # the project is AR_PROJECT (default); its adapter says what a network and its data are (EXTENSIONS.md)
    r = ev.run("t23_C_s1", "large.val", gt_key="v8")              # predict (the network runs once on what is missing), then score: a dict with the id and what was done
    ev.show_numbers(r["id"]);  ev.split_numbers(r["id"], "scene");  ev.find_worst(r["id"], "seg", n=5);  ev.get_cost(r["id"]);  ev.report(r["id"])
    ev.compare(a_id, b_id)                                        # paired bootstrap, b - a (a = the control);   ev.draw_frame(r["id"], "scene-0061/12")
    ev.check("t23_C_s1", "large.val")                             # the dry run: what `run` would do, with its problems; the data and the checkpoint are opened, no frame runs

One path: the Runner (netrunner.py) is the only code that runs a network (D72: it reads no ground truth) and stores what the network predicts (a ResultSet, results.py), one per identity of
"this network on this data"; `score` counts stored predictions against the ground truth into a RunStore (store.py: additive counts per frame); every other method reads what was stored.
Scoring again with another ground truth or metric needs the Source but not the network. D70: a network with a memory always runs warm: whole scenes in time order from the network's own
start; the frames you ask for decide which results are kept and scored, a frame already stored is not predicted again.
Ids: `<network>@r<hash>` (predictions), `<network>@e<hash>` (a scored evaluation of chosen frames, a ground truth and metrics); the same question gives the same id. One network per
process (a project's code modules are imported once, the adapter may refuse a second code snapshot): run each network in its own process and compare the evaluations by their ids.
`run` here is not `research start`: it writes no record and takes no GPU slot; a number that feeds a verdict comes from the evaluation step of a registered run (research/evaluate.py).

Frames: a text (`Frames.parse`). `val`, `small.val` = every frame of a set; `spread:1000:large.val` = 1000 frames evenly spaced over all its frames (the scenes they lie in run up to
the last of them); `random:3[:SEED]:val` = 3 random scenes (seed 0 when none is named); `scenes:scene-0061,scene-0103:val` = those scenes; `frames:scene-0061/12,scene-0003/5:val` = those frames
(their scenes run whole up to them). The set is last and defaults to `val`; it must be one the project defines (`sets`).

The adapter is a module of the project, named in `<project>/evaluation/evaluation.json` as {"adapter": "evaluation.bev_eval"} or passed to Evaluation (a module or its name); EXTENSIONS.md
lists its functions. A source's `describe()` must give `gt_signature` and `evaluator_hash` (the hash of the files that shape the numbers); without either the Evaluation refuses (a hash
that cannot be computed is not "unchanged"). Numpy and the standard library only here."""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import defaults, qualitative
from .contracts import FrameRef, Metric, Source
from .results import Locked, ResultSet, digest, writer_lock
from .sampling import Sampler
from .store import RunStore
from .suite import EvalSuite

CARRY = "carry"                                                               # the memory is always carried along a scene (D70)
MEMORY_START_MAX, MEMORY_GROWTH_MIN, MEMORY_MIN_FRAMES = 0.05, 0.02, 4        # the health of a carried memory (D70): empty at the first frame of a scene, filling over its frames
MIN_SCENES_FOR_INTERVAL = 30                                                  # the percentile scene bootstrap is anti-conservative below this (the review chat simulated it: 12.1 % false at 10 scenes, 7.4 % at 30)
NO_EVALUATOR_HASH = ("the data adapter gives no evaluator hash (the files that shape the numbers could not be hashed: one is missing or renamed): an old evaluation would be reused across edits of the metric "
                     "code; fix the adapter")
RESULT_DROP = ("class", "gt_signature", "gt_version", "evaluator_hash", "set_hash", "scenes", "frames", "split")      # what a source says about the GT and the set: not what the network was fed


class EvaluationError(Exception):
    """the message says what to fix"""


# ------------------------------------------------------------------------------------------------ frames
class Frames:
    """which frames of which set, written as the text of the module docstring; `resolve(source)` -> (the scenes to run, the frames to keep and score)"""
    KINDS = {"all": 0, "spread": 1, "random": 2, "scenes": 1, "frames": 1}

    def __init__(self, kind: str = "all", args: Sequence[Any] = (), of: str = "val"):
        if kind not in self.KINDS or len(args) != self.KINDS[kind]:
            raise EvaluationError(f"frames {kind!r} takes {self.KINDS.get(kind)} argument(s), got {len(args)}")
        self.kind, self.args, self.of = kind, tuple(args), of

    @classmethod
    def parse(cls, text: "str | Frames") -> "Frames":
        """a Frames from its text (see the module text); a Frames is returned as it is"""
        if isinstance(text, Frames):
            return text
        parts = str(text).split(":")
        kind = parts[0]
        if kind not in cls.KINDS or kind == "all":
            return cls("all", (), str(text))                                    # a bare set name
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
def load_adapter(spec: Any, project: Optional[str]):
    """the adapter module: given (a module or object with the functions), named in `spec` (a module path), or named in <project>/evaluation/evaluation.json. A "runner" key there and
    an adapter function `inputs` are refused: the Runner and the identity of predictions are the package's own, and a project that counts on its own must hear it"""
    project = os.path.abspath(project or os.environ.get("AR_PROJECT") or ".")
    cfg = os.path.join(project, "evaluation", "evaluation.json")
    try:
        with open(cfg, encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError):
        config = {}
    config = config if isinstance(config, dict) else {}
    if "runner" in config:
        raise EvaluationError(f"{cfg}: the key \"runner\" is not supported, every project runs on netrunner.Runner: remove it")
    if spec is not None and not isinstance(spec, str):
        mod = spec
    else:
        if spec is None:
            spec = config.get("adapter")
            if not spec:
                raise EvaluationError(f"no adapter: pass one, or write {cfg} as {{\"adapter\": \"module.path\"}}")
        if project not in sys.path:
            sys.path.insert(0, project)
        try:
            mod = importlib.import_module(spec)
        except ImportError as e:
            raise EvaluationError(f"adapter {spec!r} cannot be imported: {e}") from None
    missing = [f for f in defaults.REQUIRED if not hasattr(mod, f)]
    if missing:
        raise EvaluationError(f"the adapter lacks {', '.join(missing)}")
    if hasattr(mod, "inputs"):
        raise EvaluationError("the adapter defines `inputs`, which is not supported: what the network is fed is what the Source's describe() says; remove it")
    if hasattr(mod, "free_source") != hasattr(mod, "free_subject"):
        raise EvaluationError("the adapter defines only one of free_source / free_subject: they go together (a Source without ground truth and the Subject that fits it)")
    return mod if isinstance(mod, defaults.Adapter) else defaults.Adapter(mod, project)      # name, out, network, ... where the adapter has none


def names(x: "str | Sequence[str]") -> Tuple[str, ...]:
    """a list of names from a name or a list (a bare string is one name, not its letters)"""
    return (x,) if isinstance(x, str) else tuple(x)


# ------------------------------------------------------------------------------------------------ the class
class Helpers:
    """what the Runner and the Evaluation both need: the adapter, the names of networks, the identity of predictions, the stored sets. Not commands."""

    def __init__(self, project: Optional[str] = None, adapter: Any = None):
        self.ad = load_adapter(adapter, project)
        self._networks: Dict[str, str] = {}                   # the networks this object has seen: name -> path (an id carries only the name)

    def resolve(self, network: str) -> str:
        try:
            net = self.ad.resolve(network) if hasattr(self.ad, "resolve") else network
        except OSError as e:
            raise EvaluationError(str(e)) from None
        self._networks[self.ad.name(net)] = net
        return net

    def net_identity(self, net: str) -> Dict[str, Any]:
        try:
            return self.ad.identity(net)
        except OSError as e:
            raise EvaluationError(f"{net}: {e}") from None

    def identity(self, net_identity: Dict[str, Any], source: Source) -> Dict[str, Any]:
        """what the predictions depend on: the network, what it was fed (what a source says about the ground truth and the set is not part of it) and the memory mode"""
        return {"network": net_identity, "inputs": {k: v for k, v in source.describe().items() if k not in RESULT_DROP}, "state": CARRY}

    def stored(self, net: str, rid: str) -> Optional[ResultSet]:
        path = os.path.join(self.ad.out(net), "results", rid)
        return ResultSet.open(path) if os.path.exists(os.path.join(path, "manifest.json")) else None

    @staticmethod
    def health(rs: Optional[ResultSet], scenes) -> List[str]:
        """the warnings recorded for these scenes (the memory's health)"""
        return [msg for sc, msg in rs.manifest.get("health", {}).items() if sc in scenes] if rs is not None else []

    def _release(self, source: Source) -> None:
        if hasattr(self.ad, "release"):
            self.ad.release(source)

    def _source(self, net: str, of: str, workers: int, control: str, labelled: bool) -> Source:
        """the Source of a set: with its ground truth (the adapter's source) for the Evaluation, without (free_source, else source) for the Runner. A set the project does not define is
        refused before anything is opened, and whatever the adapter raises (SystemExit included) is an EvaluationError"""
        held = defaults.holdout_sets(getattr(self.ad, "project", ""))
        if (of in held or of.split(".")[-1] in held) and not defaults.holdout_open():
            raise EvaluationError(f"set {of!r} is a holdout split of this problem: it opens only in a run of `research test` (a row of the TEST log under a decided decision)")
        known = self.ad.sets()
        if of not in known:
            raise EvaluationError(f"set {of!r}: not one of {sorted(known)}" + (" (the TEST split is not offered)" if "test" in of else ""))
        try:
            make = self.ad.source if labelled else getattr(self.ad, "free_source", self.ad.source)
            return make(net, of, workers=workers, control=control)
        except SystemExit as e:
            raise EvaluationError(str(e.code) if e.code is not None else "the data adapter stopped") from None
        except (ValueError, KeyError, OSError) as e:
            raise EvaluationError(str(e.args[0] if isinstance(e, KeyError) and e.args else e)) from None


def _python_only(method):
    """a public method that is no command: it returns objects, not plain data"""
    method.command = False
    return method


class Evaluation(Helpers):
    def __init__(self, project: Optional[str] = None, adapter: Any = None):
        super().__init__(project, adapter)
        from .netrunner import Runner                         # (netrunner imports this module)
        self.runner = Runner(project, self.ad)
        self._tasks: Dict[Tuple[str, Tuple[str, ...]], Any] = {}

    def _metric_names(self, metrics: Optional[Sequence[str]]) -> Tuple[str, ...]:
        """the metrics asked for; none named = the adapter's METRICS"""
        if metrics is None:
            if not getattr(self.ad, "METRICS", None):
                raise EvaluationError("name the metrics (the adapter defines no METRICS, the list used when none are named)")
            return names(self.ad.METRICS)
        return names(metrics)

    # ---------------------------------------------------------------------------------------- what exists, and the dry run
    def list_sets(self) -> Dict[str, Any]:
        """the sets of scenes the project defines: name -> number of scenes (`val` and `train` are the network's own lists: no number)"""
        return dict(self.ad.sets())

    def list(self, network: str) -> List[Dict[str, Any]]:
        """the stored predictions and evaluations of a network, with their size"""
        net = self.resolve(network)
        rows, name, base = list(self.runner.list_results(net)), self.ad.name(net), os.path.join(self.ad.out(net), "suite", "quant")
        for d in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            try:
                with open(os.path.join(base, d, "manifest.json"), encoding="utf-8") as f:
                    ev = json.load(f).get("evaluation")
            except (OSError, ValueError):
                continue
            if ev:
                rows.append({"id": f"{name}@{d}", "kind": "evaluation", "frames": ev["n_frames"], "set": ev["frames"], "gt": ev["gt"], "metrics": ev["metrics"]})
        return rows

    def check(self, network: str, frames: str = "val", metrics: Optional[Sequence[str]] = None, taps: Sequence[str] = (), workers: int = 0, control: str = "") -> Dict[str, Any]:
        """the dry run: what `run` would do with these frames (the Runner's plan) and what is wrong with the ground truth and the metrics (`ok`, `problems`). No frame runs; the data
        and the checkpoint are opened (a BEV network is built on the CPU)"""
        metrics = self._metric_names(metrics)
        res = dict(self.runner.plan(network, str(frames), names(taps), workers, control))
        problems: List[str] = res["problems"]
        if "results" not in res:                                              # the Runner could not even plan: its problems are the answer
            return res
        try:
            net, fr = self.resolve(network), Frames.parse(frames)
            self._task(net, metrics)
            source = self._source(net, fr.of, workers, control, True)
        except (EvaluationError, ValueError, KeyError, OSError) as e:
            problems.append(str(e))
            res["ok"] = False
            return res
        try:
            self._release(source)                                             # only the ground truth is needed from here: the network a source may hold is given back
            desc = source.describe()
            _, keep = fr.resolve(source)
            res["gt"] = desc.get("gt_version") or desc.get("gt_signature")
            res["id"] = f"{self.ad.name(net)}@{self._eval_id(res['results'].partition('@')[2], keep, desc, metrics)}"
            if not desc.get("gt_signature"):
                problems.append("the data has no ground-truth signature: the numbers could not be told from another GT's")
            if not desc.get("evaluator_hash"):
                problems.append(NO_EVALUATOR_HASH)
            self._probe_reference(source, keep, problems)
        except (EvaluationError, ValueError, KeyError, OSError) as e:
            problems.append(str(e))
        finally:
            self._release(source)
        res["ok"] = not problems
        return res

    # ---------------------------------------------------------------------------------------- scoring (the network is run by the Runner)
    def score(self, results: str, frames: str = "val", gt_key: Optional[str] = None, metrics: Optional[Sequence[str]] = None, workers: int = 0, progress: int = 50) -> Dict[str, Any]:
        """score stored predictions (an id `<network>@r<hash>`) on the frames against the ground truth; the network is not run; frames that were not predicted are an error"""
        return self._score_stored(results, Frames.parse(frames), gt_key, self._metric_names(metrics), workers, progress, 0.0, 0, True)

    def run(self, network: str, frames: str = "val", gt_key: Optional[str] = None, metrics: Optional[Sequence[str]] = None, taps: Sequence[str] = (), device: str = "cuda",
            workers: int = 2, control: str = "", progress: int = 50) -> Dict[str, Any]:
        """predict (what is not stored yet, through the Runner), then score against the ground truth; gt_key = the GT version the numbers must be on (refused when the data has another);
        metrics = what the project offers (BEV: maps, boxes, curves, fusion, velocity). -> a dict: the id, what was done, the warnings"""
        net, fr, metrics = self.resolve(network), Frames.parse(frames), self._metric_names(metrics)
        self._task(net, metrics)                                          # a typo in the metrics fails here, not after the network ran
        pinfo = self.runner.predict(net, str(fr), names(taps), device, workers, control, progress)       # the network runs and is released; the labelled data is opened after it
        try:
            return self._score_stored(pinfo["results"], fr, gt_key, metrics, workers, progress, pinfo["seconds_network"], pinfo["frames_predicted"], pinfo["frames_predicted"] == 0)
        except EvaluationError as e:
            raise EvaluationError(f"{e} (the predictions are stored as {pinfo['results']}: `score` them when this is fixed)") from None

    def _score_stored(self, results, fr, gt_key, metrics, workers, progress, seconds_network, predicted, reused):
        name, _, rid = str(results).partition("@")
        net = self.resolve(self._networks.get(name) or self.ad.network(name))
        rs = self.stored(net, rid)
        if rs is None:
            raise EvaluationError(f"no predictions {results}: nothing at {os.path.join(self.ad.out(net), 'results', rid)}")
        source = self._source(net, fr.of, workers, rs.manifest["identity"]["inputs"].get("control") or "", True)
        try:
            self._release(source)                                             # scoring needs the ground truth, not the network a source may hold: give it back now
            desc = source.describe()
            _, keep = fr.resolve(source)
            self._need_gt(desc, gt_key)
            missing = [r.label for r in keep if not rs.has(r)]
            if missing:
                raise EvaluationError(f"{len(missing)} of the {len(keep)} frames (e.g. {missing[0]}) are not in {results}: predict them first (`run`, or the Runner's `predict`)")
            return self._score(net, source, desc, rs, keep, fr, metrics, seconds_network, predicted, reused, progress)
        finally:
            self._release(source)

    def _score(self, net, source, desc, rs, keep, fr, metrics, seconds_network, predicted, reused, progress):
        out, name, metrics = self.ad.out(net), self.ad.name(net), tuple(metrics)
        eid = self._eval_id(rs.id, keep, desc, metrics)
        task = self._task(net, metrics)
        ref_of = (lambda ref: self.ad.reference(source, ref)) if hasattr(self.ad, "reference") else None
        quant = EvalSuite(source, None, task, out=os.path.join(out, "suite"), state=CARRY).quant
        t1 = time.time()
        try:
            with writer_lock(os.path.join(out, "suite", "quant", eid)):
                store = quant.run(frames=keep, state=CARRY, name=eid, results=rs, reference=ref_of, progress=progress)
                info = store.manifest.get("evaluation") or {"id": f"{name}@{eid}", "results": f"{name}@{rs.id}", "network": net, "frames": str(fr), "gt": desc.get("gt_version") or desc.get("gt_signature"),
                                                            "gt_signature": desc.get("gt_signature"), "metrics": list(metrics), "n_frames": len(store.frames), "n_scenes": len(store.scene_names()),
                                                            "control": rs.manifest["identity"]["inputs"].get("control") or "", "state": CARRY, "evaluator": desc.get("evaluator_hash")}
                if "evaluation" not in store.manifest:
                    store.manifest["evaluation"] = info                       # written once, under the lock: a later call of the same question does not overwrite its record
                    store._write_manifest()
        except Locked as e:
            raise EvaluationError(str(e)) from None
        except ValueError as e:                                               # the store of this id holds another question (the same frames asked through another set, ...)
            raise EvaluationError(str(e)) from None
        return ({**info, "warnings": self._warnings(rs, {r.scene for r in keep}, store), "predictions_reused": reused, "frames_predicted": predicted,
                             "seconds_network": seconds_network, "seconds_score": round(time.time() - t1, 1)})

    # ------------------------------------------------------------ small helpers
    @staticmethod
    def _warnings(rs: Optional[ResultSet], scenes, store=None) -> List[str]:
        """what the reader of a number should know: the devices the frames were predicted on (from the frames the evaluation scored) and the memory's health"""
        out = []
        if store is not None:
            devs = sorted({fr["info"]["device"] for fr in store.frames if fr["info"].get("device")})
            if len(devs) > 1:
                out.append(f"the frames were predicted on different devices {devs}: their float results differ slightly")
        return out + Helpers.health(rs, scenes)

    def _eval_id(self, rid: str, keep: Sequence[FrameRef], desc: Dict[str, Any], metrics: Sequence[str]) -> str:
        """what the numbers depend on: the predictions, the frames (with their dataset keys), the set they were asked through, the ground truth, the metrics, the evaluator code (the source's
        `evaluator_hash`: it must cover every file that shapes a number)"""
        return "e" + digest([rid, digest([[r.label, r.key] for r in keep], 40), desc.get("split"), desc.get("set_hash"), desc.get("gt_signature"), sorted(metrics), desc.get("evaluator_hash")])

    @staticmethod
    def _need_gt(desc: Dict[str, Any], gt_key: Optional[str]) -> None:
        if not desc.get("gt_signature"):
            raise EvaluationError("the data has no ground-truth signature: numbers could not be told from those of another GT; fix the data adapter")
        if not desc.get("evaluator_hash"):
            raise EvaluationError(NO_EVALUATOR_HASH)
        if gt_key and gt_key not in (desc.get("gt_version"), desc.get("gt_signature")):
            raise EvaluationError(f"the data has ground truth {desc.get('gt_version') or desc.get('gt_signature')} ({desc.get('gt_signature')}), not {gt_key}")

    def _probe_reference(self, source: Source, keep: Sequence[FrameRef], problems: List[str]) -> None:
        try:
            (self.ad.reference(source, keep[0]) if hasattr(self.ad, "reference") else source.load(keep[0]).reference)
        except Exception as e:                                                # noqa: BLE001  whatever the data adapter raises: its message is the problem
            problems.append(f"the ground truth of {keep[0].label} cannot be read: {type(e).__name__}: {e}")

    def _task(self, net: str, metrics: Sequence[str]):
        key = (net, tuple(metrics))
        if key not in self._tasks:
            try:
                self._tasks[key] = self.ad.task(net, tuple(metrics))
            except ValueError as e:
                raise EvaluationError(str(e)) from None
        return self._tasks[key]

    # ---------------------------------------------------------------------------------------- reading what was stored
    def _open_store(self, eval_id: str) -> Tuple[Dict[str, Any], RunStore]:
        """(info, store) of a stored evaluation: the counts and the record of what was asked, no metric definitions (no Task is built)"""
        name, _, eid = str(eval_id).partition("@")
        if not eid.startswith("e"):
            raise EvaluationError(f"{eval_id!r} is not an evaluation id (<network>@e<hash>; `list NETWORK` shows them)")
        net = self._networks.get(name) or self.resolve(self.ad.network(name))
        path = os.path.join(self.ad.out(net), "suite", "quant", eid)
        try:
            store = RunStore.open(path)
        except OSError:
            raise EvaluationError(f"no evaluation {eval_id}: nothing at {path}") from None
        info = store.manifest.get("evaluation")
        if not info:
            raise EvaluationError(f"{eval_id} is a store without an evaluation record")
        return info, store

    def _open(self, eval_id: str):
        """(info, store, quantitative) of a stored evaluation"""
        info, store = self._open_store(eval_id)
        task = self._task(info["network"], tuple(info["metrics"]))
        return info, store, EvalSuite(None, None, task, out=os.path.join(self.ad.out(info["network"]), "suite"), state=CARRY).quant

    def _predictions(self, info: Dict[str, Any]) -> Optional[ResultSet]:
        return self.stored(info["network"], info["results"].partition("@")[2])

    @_python_only
    def stored_counts(self, id: str) -> RunStore:
        """the additive counts of an evaluation (`RunStore.by_scene(metric)`), for code that computes its own statistics (the judge); no Task is built"""
        return self._open_store(id)[1]

    @_python_only
    def metrics_of(self, id: str) -> Dict[str, Metric]:
        """the metric definitions of an evaluation: name -> Metric (`reduce` turns counts into numbers, `direction` says which way is better)"""
        return self._open(id)[2]._metrics()

    def show_numbers(self, id: str, match: Optional[str] = None) -> Dict[str, Any]:
        """the dataset-level numbers of an evaluation ('metric.key' -> value, only those containing `match`) and which way is better ('high' / 'low')"""
        info, store, q = self._open(id)
        tab = q.table(store)
        better = {f"{m.name}.{k}": m.direction(k) for m in q._metrics().values() for k in m.headline}
        keep = (lambda k: match in k) if match else (lambda k: True)
        warn = self._warnings(self._predictions(info), {fr["scene"] for fr in store.frames}, store)
        now = self.ad.evaluator_now() if hasattr(self.ad, "evaluator_now") else info.get("evaluator")
        if info.get("evaluator") and not now:
            warn.append("the evaluator code cannot be hashed now (a file that shapes the numbers is missing or renamed): whether it changed since this evaluation cannot be told")
        elif now and info.get("evaluator") and now != info["evaluator"]:
            warn.append(f"scored with evaluator code {info['evaluator']}, the code is now {now}: the counts are the old ones but they are reduced with today's metrics; score again (`run`) for a clean number")
        return {"id": id, "n_frames": info["n_frames"], "gt": info["gt"], "numbers": {k: v for k, v in tab.items() if keep(k)}, "better": {k: v for k, v in better.items() if keep(k)},
                "warnings": warn}

    def split_numbers(self, id: str, axis: str = "scene") -> Dict[str, Any]:
        """the headline numbers per scene (axis 'scene') or the scores of every frame (axis 'frame')"""
        _, store, q = self._open(id)
        if axis == "scene":
            return q.per_scene(store)
        if axis == "frame":
            return {fr["label"]: fr["scores"] for fr in store.frames}
        raise EvaluationError(f"axis {axis!r}: scene or frame")

    def find_worst(self, id: str, match: Optional[str] = None, n: int = 10) -> Dict[str, List[dict]]:
        """where the numbers come from: the scenes whose removal improves a headline number most and the frames furthest out in the bad direction (rows that contain `match`); a
        stored evaluation has no plausibility flags (they need the inputs)"""
        _, store, q = self._open(id)
        res = q.outliers(store, k=n)
        if match:
            res = {k: [row for row in v if match in json.dumps(row, default=str)] for k, v in res.items()}
        return res

    def compare(self, a: str, b: str, boot: int = 1000, seed: int = 0) -> Dict[str, Any]:
        """paired scene bootstrap of the headline numbers, b - a (a = the control, b = the variant); refused when the two do not answer the same question (other source, task or GT, or a frame
        that is another frame in the two); `warnings` say what the interval needs a reader to know (frames in only one of the two, fewer than 30 scenes, other evaluator code)"""
        ia, sa, qa = self._open(a)
        ib, sb = self._open_store(b)                                          # only the control's metric definitions are used; the variant may come from other code
        try:
            res = qa.compare(sa, sb, boot, seed, ignore=getattr(self.ad, "COMPARE_IGNORE", None))
        except ValueError as e:
            raise EvaluationError(str(e)) from None
        both = {fr["label"] for fr in sa.frames} & {fr["label"] for fr in sb.frames}
        n_scenes = len({fr["scene"] for fr in sa.frames if fr["label"] in both})
        warn = []
        if ia.get("evaluator") != ib.get("evaluator"):
            warn.append(f"the two were scored with different evaluator code ({ia.get('evaluator')} / {ib.get('evaluator')}): the difference may come from the evaluator")
        if res.get("only_a") or res.get("only_b"):
            warn.append(f"the two share {res['n_common_frames']} frames only ({res.get('only_a', 0)} only in the control, {res.get('only_b', 0)} only in the variant): the interval is on the shared frames")
        if n_scenes < MIN_SCENES_FOR_INTERVAL:
            warn.append(f"only {n_scenes} scenes: the scene bootstrap is too narrow below about {MIN_SCENES_FOR_INTERVAL} scenes (simulated: 12 % false 'excludes 0' with 10 scenes, 5 % nominal)")
        return {**res, "n_scenes": n_scenes, "warnings": warn}

    def get_cost(self, id: str) -> Dict[str, Any]:
        """ms per frame of the network as it ran (the median over the frames of the last session that ran frames, on that day's machine)"""
        info, store, _ = self._open(id)
        rs = self._predictions(info)
        return {"id": id, "predict": rs.manifest.get("cost") if rs else None, "devices": sorted({fr["info"]["device"] for fr in store.frames if fr["info"].get("device")})}

    def report(self, id: str) -> str:
        """the html page of an evaluation (numbers, scenes by drag, outliers); returns its path"""
        _, store, q = self._open(id)
        return q.report(store)

    # ---------------------------------------------------------------------------------------- a picture
    def draw_frame(self, id: str, frame: str, view: str = "overview", out: str = "", workers: int = 0) -> Dict[str, Any]:
        """draw one frame (scene/index or the dataset's key) of an evaluation from its stored prediction, with the ground truth the data gives now, headed by the frame's label and scores;
        no network runs. `view`: one of the Task's, or `debug`; a view that looks inside the network gets the taps stored with the frame (`run --taps`) and what the subject says they are
        (it is built for that, not run). Refused when the predictions are gone or the ground truth is not the one that was scored.
        -> {frame, image (the png: `out`, else below the network's eval folder), scores}"""
        out = os.path.abspath(out) if out else ""                       # before the project's code runs: an adapter may change the working folder
        info = self._open_store(id)[0]
        rs = self._predictions(info)
        if rs is None:
            raise EvaluationError(f"the predictions {info['results']} of {id} are gone: score it again (`run`)")
        net = info["network"]
        task = self._task(net, tuple(info["metrics"]))
        views = qualitative.views_of(task)
        if view not in views:
            raise EvaluationError(f"no view {view!r}: {sorted(views)}")
        source = self._source(net, Frames.parse(info["frames"]).of, workers, info.get("control", ""), True)
        try:
            inside = getattr(views[view], "needs_taps", False)
            specs, described = (), {}
            if inside:                                                        # the kind, plane, note and alive check of every tap are the subject's word
                subject = self.ad.subject(net, source, "cpu")
                try:
                    specs, described = subject.taps(), subject.describe()
                finally:
                    subject.close()
                subject = None
            self._release(source)                                             # from here a picture needs the data, not the network
            now, then = source.describe().get("gt_signature"), info.get("gt_signature")     # info["gt"] is the version's name when it has one
            if now and then and str(now) != str(then):
                raise EvaluationError(f"{id} was evaluated on ground truth {info.get('gt')} ({then}), the source gives {now} now: the picture would not show what was scored")
            try:
                ref = source.find(frame)
            except KeyError as e:
                raise EvaluationError(str(e.args[0])) from None
            if not rs.has(ref):
                raise EvaluationError(f"{ref.label} is not among the predictions {info['results']}")
            taps = [s.name for s in specs if rs.has(ref, (s.name,))]
            if inside and not taps:
                raise EvaluationError(f"the view {view!r} shows intermediate results and none are stored for {ref.label}: predict them (`run ... --taps NAMES`; "
                                      f"the network offers {[s.name for s in specs]})")
            path = out or os.path.abspath(os.path.join(self.ad.out(net), "suite", "pictures", id.partition("@")[2], f"{ref.slug}_{view}.png"))
            scores = qualitative.draw(task, view, source.load(ref), rs.load(ref, taps), path, specs, described)
            return {"frame": ref.label, "image": path, "scores": scores}
        finally:
            self._release(source)

    # ---------------------------------------------------------------------------------------- what executes the network belongs to the Runner: passed through
    def tap(self, results: str, frame: str, name: str = "") -> Dict[str, Any]:
        """the intermediate results stored with a frame's predictions: without `name` which taps there are, with it the path of its .npy file, its shape and dtype (the Runner's)"""
        return self.runner.tap(results, frame, name)

    def time(self, network: str, n: int = 30) -> Dict[str, Any]:
        """milliseconds per module and per frame at batch 1 (idle GPU only: the adapter refuses a busy one; the Runner's)"""
        return self.runner.time(network, n)

    def probe(self, network: str, frames: int = 3, device: str = "cpu") -> Dict[str, Any]:
        """the module probe: statistics and dead channels of every module, the response when the camera or the radar is removed (frames = how many frames of the first scene; the Runner's)"""
        return self.runner.probe(network, frames, device)


Evaluation.METHODS = tuple(n for n, f in vars(Evaluation).items() if callable(f) and not n.startswith("_") and getattr(f, "command", True))      # the methods are the commands: no second list
