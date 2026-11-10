"""runner - a trained BEV network run on frames, as ONE class. The command line above it is `python -m runner` (__main__.py): every method of `Runner` is a command.

    r = Runner()                                                     # the project: this one (bev_fusion); its evaluation adapter (evaluation/evalsuite.json) says what a network and its data are
    r.plan("runs/t23_C_s1", "small.val")                             # the dry run: scenes, frames, what is stored, what would run, the problems
    r.predict("runs/t23_C_s1", "small.val", taps=["fused"])          # the network on the frames that are not stored yet, no ground truth; -> {results: "<net>@r<hash>", ...}
    r.list_results("runs/t23_C_s1");  r.tap(id, "scene-0061/12", "fused");  r.time("runs/t23_C_s1");  r.probe("runs/t23_C_s1")

No queue, no launcher, no ledger row: a call runs in this process (a run that feeds a verdict is started through autoresearch/tools/run.py with this command as its command). A network with a
memory always runs warm (D70): whole scenes in time order from the scene's own start, the frames asked for only decide which results are kept. The run reads no ground truth and computes no
validation loss (D72): the adapter's `free_source` / `free_subject` (evaluation/bev_free.py) give a Source whose frames have no reference and a Subject that only predicts. What is stored is
the ResultSet of autoresearch/evalsuite/results.py, the same whoever asks (the identity rules of evalsuite/api.py); `Evaluation.score` scores it later when a ground truth exists.
The health of a carried memory (the adapter's MEMORY_TAP, the mean of that tap per frame): empty at the first frame of a scene, filling over its frames; a scene where it is not is a warning in
the result, never an error. `time` and `probe` are here because they execute the network. The pieces (Frames, ResultSet, the frame iterator, the Source / Subject contracts, the constants of
the memory check) are evalsuite's and imported here; evalsuite reaches this class only by the name in evaluation/evalsuite.json (key "runner")."""
from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
AUTORESEARCH = os.path.abspath(os.path.join(ROOT, "..", "autoresearch"))
for _p in (ROOT, AUTORESEARCH):                                     # the helpers of the evaluation and the suites (the code snapshot of a run keeps the first place)
    if _p not in sys.path:
        sys.path.append(_p)

from evalsuite.api import CARRY, MEMORY_GROWTH_MIN, MEMORY_MIN_FRAMES, MEMORY_START_MAX, EvaluationError, Frames, Helpers, names  # noqa: E402
from evalsuite.contracts import FrameRef, Prediction  # noqa: E402
from evalsuite.results import Locked, ResultSet, result_id, writer_lock  # noqa: E402
from evalsuite.runner import Runner as FrameRunner  # noqa: E402     # the frame iterator of the engine (a Subject over a Source, scene by scene)

RunnerError = EvaluationError                                       # the data, the checkpoint, the memory: the errors are the evaluation's, one class for both


class Runner(Helpers):
    METHODS = ("predict", "plan", "list_results", "tap", "time", "probe")         # every one is a command; resolve and stored (from Helpers) are helpers of the Evaluation

    def __init__(self, project: Optional[str] = None, adapter: Any = None):
        super().__init__(project or ROOT, adapter)

    # ---------------------------------------------------------------------------------------- run
    def predict(self, network: str, frames: str = "val", taps: Sequence[str] = (), device: str = "cuda", workers: int = 2, control: str = "", progress: int = 50) -> Dict[str, Any]:
        """run the network on the frames that are not stored yet (whole scenes in time order, warm: D70) and store what it predicts, with the intermediate results `taps` when asked; no ground truth is read. -> the id of the predictions, what was done and the warnings of the memory's health"""
        net, fr, taps = self.resolve(network), Frames.parse(frames), names(taps)
        ident = self.net_identity(net)
        source = self._source(net, fr.of, workers, control, False)
        try:
            scenes, keep = fr.resolve(source)
            identity = self.identity(ident, source)
            rid = result_id(identity)
            path = os.path.join(self.ad.out(net), "results", rid)
            t0, todo, memory_checked = time.time(), self._todo(self.stored(net, rid), keep, taps), None
            if todo:
                try:
                    with writer_lock(path):                                      # one writer per set; the set itself is made at the first frame, so a refused run leaves none
                        have = self.stored(net, rid)                             # another writer may have stored frames before we got the lock
                        todo = self._todo(have, keep, taps)
                        if todo:
                            memory_checked = self._run_network(net, source, scenes, keep, path, identity, have, ident, taps, device, progress)
                except Locked as e:
                    raise EvaluationError(str(e)) from None
                if self.net_identity(net) != ident:
                    raise EvaluationError(f"{net} changed while the network ran (a new checkpoint?): the predictions {self.ad.name(net)}@{rid} belong to the old one; run again")
            return {"results": f"{self.ad.name(net)}@{rid}", "frames_kept": len(keep), "frames_predicted": len(todo), "seconds_network": round(time.time() - t0, 1),
                    "predictions_reused": not todo, "warnings": self.health(self.stored(net, rid), {r.scene for r in keep}), "memory_checked": memory_checked, "network": net}
        finally:
            self._release(source)

    def plan(self, network: str, frames: str = "val", taps: Sequence[str] = (), workers: int = 0, control: str = "") -> Dict[str, Any]:
        """the dry run of `predict`: what it would do with these frames (scenes, what is stored, what would run) and what is wrong; no frame runs, the data and the checkpoint are opened. -> {ok, problems, results, frames_to_run, ...}"""
        problems: List[str] = []
        taps = names(taps)
        res: Dict[str, Any] = {"network": network, "frames": str(frames), "ok": False, "problems": problems}
        try:
            net, fr = self.resolve(network), Frames.parse(frames)
            res["network"] = net
            ident = self.net_identity(net)
            source = self._source(net, fr.of, workers, control, False)
        except (EvaluationError, ValueError, KeyError, OSError) as e:
            problems.append(str(e))
            return res
        try:
            scenes, keep = fr.resolve(source)
            rid = result_id(self.identity(ident, source))
            rs = self.stored(net, rid)
            have = sum(1 for r in keep if rs and rs.has(r, taps))
            want = {r.label for r in keep}
            to_run = 0                                                           # the network runs each scene up to its last kept frame, the frames before it included (the memory)
            for sc in scenes:
                kept = [r for r in source.frames(sc) if r.label in want]
                to_run += (kept[-1].index + 1) if any(not (rs and rs.has(r, taps)) for r in kept) else 0
            res.update({"scenes_run": len(scenes), "frames_kept": len(keep), "frames_stored": have, "frames_to_run": to_run, "results": f"{self.ad.name(net)}@{rid}",
                        "out": self.ad.out(net), "state": "carry (whole scenes in time order, D70)", "network_runs": have < len(keep)})
            subject = self._subject(net, source, "cpu")                          # what `predict` refuses before its first frame: a network too old for this run, taps it does not declare
            try:
                self._declared(subject, taps)
            finally:
                subject.close()
        except (EvaluationError, ValueError, KeyError, OSError) as e:
            problems.append(str(e))
        finally:
            self._release(source)
        res["ok"] = not problems
        return res

    # ---------------------------------------------------------------------------------------- what is stored
    def list_results(self, network: str) -> List[Dict[str, Any]]:
        """the stored predictions of a network with their size. -> a list of {id, kind, frames, megabytes, created}"""
        net = self.resolve(network)
        base, name, rows = os.path.join(self.ad.out(net), "results"), self.ad.name(net), []
        for d in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            try:
                rs = ResultSet.open(os.path.join(base, d))
            except (OSError, ValueError):
                continue
            rows.append({"id": f"{name}@{d}", "kind": "predictions", "frames": rs.n_frames(), "megabytes": rs.size_mb(), "created": rs.manifest.get("created")})
        return rows

    def tap(self, results: str, frame: str, name: str = "") -> Dict[str, Any]:
        """the intermediate results stored with a frame's predictions: without `name` which taps there are, with it the path of its .npy file, its shape and dtype."""
        rname, _, rid = str(results).partition("@")
        net = self._networks.get(rname) or self.resolve(self.ad.network(rname))
        rs = self.stored(net, rid)
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
        arr = np.load(rs.tap_path(ref, name), mmap_mode="r")
        return {"results": results, "frame": ref.label, "tap": name, "path": rs.tap_path(ref, name), "shape": list(arr.shape), "dtype": str(arr.dtype)}

    # ---------------------------------------------------------------------------------------- about the network itself
    def time(self, network: str, n: int = 30) -> Dict[str, Any]:
        """milliseconds per module and per frame of a network at batch 1 (idle GPU only: the project's adapter refuses a busy one)."""
        if not hasattr(self.ad, "time"):
            raise EvaluationError("this project's adapter defines no timing")
        return self.ad.time(self.resolve(network), n)

    def probe(self, network: str, frames: int = 3, device: str = "cpu") -> Dict[str, Any]:
        """the module probe: statistics and dead channels of every module and the response when the camera or the radar is removed (frames = how many frames of the first scene)."""
        if not hasattr(self.ad, "probe"):
            raise EvaluationError("this project's adapter defines no probe")
        return self.ad.probe(self.resolve(network), frames, device)

    # ---------------------------------------------------------------------------------------- the work behind predict (the only code that executes the network)
    @staticmethod
    def _todo(have: Optional[ResultSet], keep: Sequence[FrameRef], taps: Sequence[str]) -> List[FrameRef]:
        """the kept frames that are not stored (with these taps) yet"""
        return [r for r in keep if not (have and have.has(r, taps))]

    def _subject(self, net: str, source, device: str):
        """the network without the validation loss, built for this source (the adapter's free_subject, else its subject)"""
        return getattr(self.ad, "free_subject", self.ad.subject)(net, source, device)

    @staticmethod
    def _declared(subject, taps: Sequence[str]) -> set:
        """the taps the network declares; asking for one it does not declare is an error"""
        declared = {t.name for t in subject.taps()}
        bad = [t for t in taps if t not in declared]
        if bad:
            raise EvaluationError(f"unknown taps {bad}: the network offers {sorted(declared)}")
        return declared

    def _run_network(self, net, source, scenes, keep, path, identity, have, ident, taps, device, progress):
        """run the scenes with frames to store; -> whether the memory could be checked (None when it does not apply: the adapter names no MEMORY_TAP or the network does not declare it).
        The set is made at the first frame to store: a refused run (a network too old, an unknown tap, a changed checkpoint) leaves none"""
        want, t0, n, ms, rs, produced, checked = {r.label for r in keep}, time.time(), 0, [], None, False, None
        subject = self._subject(net, source, device)
        try:
            if self.net_identity(net) != ident:                                # a training may have written a new checkpoint while the network was loaded: refuse before a frame is stored
                raise EvaluationError(f"{net} changed while it was loaded (a new checkpoint?): nothing was stored; run again")
            declared = self._declared(subject, taps)
            mem = getattr(self.ad, "MEMORY_TAP", None)
            mem = mem if mem in declared else None                             # the health check needs the project's memory tap
            tapped = list(taps) + ([mem] if mem and mem not in taps else [])
            checked = False if mem else None
            frame_runner = FrameRunner(source, subject, CARRY)
            total = len(self._todo(have, keep, taps))
            for sc in scenes:
                kept = [r for r in source.frames(sc) if r.label in want]
                if not self._todo(have, kept, taps):
                    continue
                refs = source.frames(sc)[: kept[-1].index + 1]                 # nothing after the last frame asked for is needed; the frames before it always run (the memory)
                fill = []
                for sample, pred in frame_runner.iter_scene(sc, refs, taps=tapped):
                    if not produced:                                           # a tap that is declared (every tap of the BEV network is optional) but not produced would be dropped silently
                        absent = [t for t in taps if t not in pred.taps]
                        if absent:
                            raise EvaluationError(f"the network does not produce the taps {absent} (the module they read is not in this network): nothing was stored")
                        produced = True
                    if mem and mem in pred.taps:
                        fill.append(float(np.mean(pred.taps[mem])))
                    if sample.ref.label in want and not (have and have.has(sample.ref, taps)):
                        rs = rs or ResultSet.create(path, identity)
                        info = {**pred.info, "device": str(getattr(subject, "device", device)), **({"memory_fill": fill[-1]} if fill else {})}
                        rs.add(sample.ref, Prediction(pred.outputs, {k: v for k, v in pred.taps.items() if k in taps}, info))
                        n += 1
                        ms.append(float(pred.info.get("predict_ms", 0.0)))
                        if progress and n % progress == 0:
                            print(f"[predict {rs.id}] {n}/{total} frames, {(time.time() - t0) / n:.2f} s per frame", file=sys.stderr, flush=True)
                if rs is not None:
                    checked = bool(fill) or checked
                    self._memory_health(rs, sc, fill)
                    rs.save()                                                  # a killed writer keeps what it recorded so far (frames are atomic, the manifest per scene)
            return checked
        finally:
            if rs is not None:
                rs.close(wall_s=time.time() - t0, cost={"median_ms": sorted(ms)[len(ms) // 2], "n": len(ms)} if ms else None)
            subject.close()

    @staticmethod
    def _memory_health(rs: ResultSet, scene: str, fill: List[float]) -> None:
        """D70: a carried memory is empty at the first frame of a scene and fills over its frames; a scene where it is not is recorded in the result set (a warning to every evaluation of it).
        The scene always runs from its start, so the first frame is always judged; whether the memory fills is judged only by a run long enough to tell (a shorter run, for a look at one tap,
        leaves what a longer run recorded)"""
        health = rs.manifest.setdefault("health", {})
        if not fill:
            return
        if not all(math.isfinite(f) for f in fill):
            health[scene] = f"{scene}: the memory fill is not a finite number: the memory is broken"
        elif fill[0] > MEMORY_START_MAX:
            health[scene] = f"{scene}: the memory is not empty at the first frame ({fill[0]:.2f} > {MEMORY_START_MAX}): not initialised"
        elif len(fill) >= MEMORY_MIN_FRAMES:
            if max(fill) - fill[0] < MEMORY_GROWTH_MIN:
                health[scene] = f"{scene}: the memory does not fill over {len(fill)} frames ({fill[0]:.2f} -> {fill[-1]:.2f}): not used"
            else:
                health.pop(scene, None)
