"""What the Runner (netrunner.py) and the Evaluation (api.py) share of a project: the error class, the frames text, the project's adapter, and `Helpers` (the names of networks, the
identity of predictions, the stored sets, the Source of a set behind the holdout gate). Numpy and the standard library only."""
from __future__ import annotations

import importlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import defaults
from .contracts import FrameRef, Source
from .results import ResultSet
from .sampling import Sampler

CARRY = "carry"                                                               # the memory is always carried along a scene (D70)
RESULT_DROP = ("class", "gt_signature", "gt_version", "evaluator_hash", "set_hash", "scenes", "frames", "split")      # what a source says about the GT and the set: not what the network was fed


class EvaluationError(Exception):
    """the message says what to fix"""


# ------------------------------------------------------------------------------------------------ frames
class Frames:
    """which frames of which set, as a text: `val`, `small.val` = every frame of a set; `spread:1000:large.val` = 1000 frames evenly spaced over all its frames (the scenes they lie in run
    up to the last of them); `random:3[:SEED]:val` = 3 random scenes (seed 0 when none is named); `scenes:scene-0061,scene-0103:val` = those scenes;
    `frames:scene-0061/12,scene-0003/5:val` = those frames (their scenes run whole up to them). The set is last and defaults to `val`; it must be one the project defines (`sets`).
    `resolve(source)` -> (the scenes to run, the frames to keep and score)"""
    KINDS = {"all": 0, "spread": 1, "random": 2, "scenes": 1, "frames": 1}

    def __init__(self, kind: str = "all", args: Sequence[Any] = (), of: str = "val"):
        if kind not in self.KINDS or len(args) != self.KINDS[kind]:
            raise EvaluationError(f"frames {kind!r} takes {self.KINDS.get(kind)} argument(s), got {len(args)}")
        self.kind, self.args, self.of = kind, tuple(args), of

    @classmethod
    def parse(cls, text: "str | Frames") -> "Frames":
        """a Frames from its text; a Frames is returned as it is"""
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
    """the adapter module: given (a module or object with the functions), named in `spec` (a module path), or named in <project>/evaluation/evaluation.json. That file may be missing; one
    that is there and is not a json object is an error that names the file and what is wrong with it. A "runner" key there and an adapter function `inputs` are refused: the Runner and
    the identity of predictions are the package's own, and a project that counts on its own must hear it"""
    project = os.path.abspath(project or os.environ.get("AR_PROJECT") or ".")
    cfg = os.path.join(project, "evaluation", "evaluation.json")
    try:
        with open(cfg, encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {}
    except (OSError, ValueError) as e:
        raise EvaluationError(f"{cfg} cannot be read: {type(e).__name__}: {e}") from None
    if not isinstance(config, dict):
        raise EvaluationError(f"{cfg} must hold a json object such as {{\"adapter\": \"module.path\"}}, it holds a {type(config).__name__}")
    if "runner" in config:
        raise EvaluationError(f"{cfg}: the key \"runner\" is not supported, every project runs on netrunner.Runner: remove it")
    if spec is not None and not isinstance(spec, str):
        mod = spec
    else:
        if spec is None:
            spec = config.get("adapter")
            if not spec:
                raise EvaluationError(f"no adapter: pass one, or write {cfg} as {{\"adapter\": \"module.path\"}}")
        packaged = spec.split(".")[0] == os.path.basename(project)             # the project is a package named like its folder: its root is the folder above
        root = os.path.dirname(project) if packaged else project
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            mod = importlib.import_module(spec)
        except ImportError as e:
            raise EvaluationError(f"adapter {spec!r} cannot be imported: {e}") from None
        file = os.path.realpath(getattr(mod, "__file__", None) or os.devnull)
        own = spec.split(".")[0] == __package__.split(".")[0]                 # an adapter this package ships (minimal, its tests) belongs to no project
        if not own and not file.startswith(os.path.realpath(project) + os.sep):   # found elsewhere on the module path: another tree's code under this project's hashes
            raise EvaluationError(f"adapter {spec!r} was imported from {file}, outside the project {project}: the module path names another tree first, or the "
                                  "project is a copy whose adapter still carries the other folder's package name")
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


# ------------------------------------------------------------------------------------------------ what both classes need
class Helpers:
    """the adapter, the names of networks, the identity of predictions, the stored sets. Not commands."""

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
        """the Source of a set: with its ground truth (the adapter's source) for the Evaluation, without (free_source, else source) for the Runner. A holdout split is refused unless
        the holdout gate is open (defaults.holdout_open), a set the project does not define is refused, both before anything is opened; whatever the adapter raises (SystemExit
        included) is an EvaluationError"""
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
