"""Runner: the class that executes a network and stores what it predicts (D70, D72). The Evaluation calls it; it also works alone, to try a network on frames that have no labels.

    rn = Runner()                                                  # the project is AR_PROJECT (default), its adapter says what a network and its data are (api.py)
    rn.check("runs/t23_C_s1", "large.val")                         # the dry run: scenes, frames, what is stored, what would run, the problems
    rn.run("runs/t23_C_s1", "large.val", taps=("fused",))          # predict what is not stored yet, store it (ResultSet, results.py): the id `<network>@r<hash>`
    rn.tap(id, "scene-0061/12", "fused");  rn.list("runs/t23_C_s1")

It reads no ground truth and computes no validation loss: the adapter's `free_source` / `free_subject` (when it has them; else its `source` / `subject`) give a Source whose frames have
no reference and a Subject that only predicts. A network with a memory always runs warm, whole scenes in time order from its own start (D70); the frames asked for decide which results are kept.
What is stored is the same ResultSet whoever asks (same identity rules as api.py): `Evaluation.score` scores it later when a ground truth exists. Also here, because they execute the
network: `time` (ms per module and frame, idle GPU) and `probe` (the module probe). `run` is not `autoresearch/tools/run.py`: no ledger row, no GPU slot.
The health of a carried memory (the adapter's MEMORY_TAP, the mean of that tap per frame): empty at the first frame of a scene, filling over its frames; a scene that fails is a warning
in the result of `run`, never an error. Not the engine's `runner.Runner` (that one only iterates frames through a Subject)."""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Sequence

import numpy as np

from .api import CARRY, MEMORY_GROWTH_MIN, MEMORY_MIN_FRAMES, MEMORY_START_MAX, EvaluationError, Frames, Helpers, names
from .contracts import FrameRef, Prediction
from .results import Locked, ResultSet, result_id, writer_lock
from .runner import Runner as FrameRunner                      # the engine's frame iterator


class Runner(Helpers):
    # ---------------------------------------------------------------------------------------- the dry run and what is stored
    def check(self, network: str, frames: str = "val", taps: Sequence[str] = (), workers: int = 0, control: str = "") -> Dict[str, Any]:
        """the dry run: what `run` would do with these frames and what is wrong (`ok`, `problems`). No frame runs; the data and the checkpoint are opened (a BEV network is built on the CPU)"""
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
        except (EvaluationError, ValueError, KeyError, OSError) as e:
            problems.append(str(e))
        finally:
            self._release(source)
        res["ok"] = not problems
        return res

    def list(self, network: str) -> List[Dict[str, Any]]:
        """the stored predictions of a network, with their size"""
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
        """the intermediate results stored with a frame's predictions: without `name` which taps there are, with it the path of its .npy file, its shape and dtype"""
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
            raise EvaluationError(f"{frame} of {results} has no tap {name!r}; it has {rs.tap_names(ref)} (ask for it with `run --taps {name}`)")
        path = rs.tap_path(ref, name)
        arr = np.load(path, mmap_mode="r")
        return {"results": results, "frame": ref.label, "tap": name, "path": path, "shape": list(arr.shape), "dtype": str(arr.dtype)}

    # ---------------------------------------------------------------------------------------- running (the only code that executes the network)
    def run(self, network: str, frames: str = "val", taps: Sequence[str] = (), device: str = "cuda", workers: int = 2, control: str = "", progress: int = 50) -> Dict[str, Any]:
        """run the network on the frames that are not stored yet (whole scenes in time order, warm: D70) and store what it predicts, with the intermediate results `taps` when asked;
        no ground truth is read. -> the id of the predictions, what was done and the warnings of the memory's health"""
        net, fr, taps = self.resolve(network), Frames.parse(frames), names(taps)
        ident = self.net_identity(net)
        source = self._source(net, fr.of, workers, control, False)
        try:
            scenes, keep = fr.resolve(source)
            identity = self.identity(ident, source)
            rid = result_id(identity)
            rs = ResultSet.create(os.path.join(self.ad.out(net), "results", rid), identity)
            todo = [r for r in keep if not rs.has(r, taps)]
            t0 = time.time()
            if todo:
                try:
                    with writer_lock(rs.path):
                        rs = ResultSet.open(rs.path, "a")                          # another writer may have stored frames while we waited for the lock
                        todo = [r for r in keep if not rs.has(r, taps)]
                        if todo:
                            self._run_network(net, source, scenes, keep, rs, ident, taps, device, progress)
                except Locked as e:
                    raise EvaluationError(str(e)) from None
                if self.net_identity(net) != ident:
                    raise EvaluationError(f"{net} changed while the network ran (a new checkpoint?): the predictions belong to the old one; run again")
            return {"results": f"{self.ad.name(net)}@{rid}", "frames_kept": len(keep), "frames_predicted": len(todo), "seconds_network": round(time.time() - t0, 1),
                    "predictions_reused": not todo, "warnings": self.health(rs, {r.scene for r in keep}), "network": net}
        finally:
            self._release(source)

    def _run_network(self, net, source, scenes, keep, rs, ident, taps, device, progress):
        want, t0, n, ms = {r.label for r in keep}, time.time(), 0, []
        subject = getattr(self.ad, "free_subject", self.ad.subject)(net, source, device)
        try:
            if self.net_identity(net) != ident:                                # a training may have written a new checkpoint while the network was loaded: refuse before a frame is stored
                raise EvaluationError(f"{net} changed while it was loaded (a new checkpoint?): nothing was stored; run again")
            declared = {t.name for t in subject.taps()}
            bad = [t for t in taps if t not in declared]
            if bad:
                raise EvaluationError(f"unknown taps {bad}: the network offers {sorted(declared)}")
            mem = getattr(self.ad, "MEMORY_TAP", None)
            mem = mem if mem in declared else None                             # the health check needs the project's memory tap
            names = list(taps) + ([mem] if mem and mem not in taps else [])
            runner = FrameRunner(source, subject, CARRY)
            total = sum(1 for r in keep if not rs.has(r, taps))
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
                        info = {**pred.info, "device": str(getattr(subject, "device", device)), **({"memory_fill": fill[-1]} if fill else {})}
                        rs.add(sample.ref, Prediction(pred.outputs, {k: v for k, v in pred.taps.items() if k in taps}, info))
                        n += 1
                        ms.append(float(pred.info.get("predict_ms", 0.0)))
                        if progress and n % progress == 0:
                            print(f"[run {rs.id}] {n}/{total} frames, {(time.time() - t0) / n:.2f} s per frame", file=sys.stderr, flush=True)
                self._memory_health(rs, sc, fill)
                rs.save()                                                      # a killed writer keeps what it recorded so far (frames are atomic, the manifest per scene)
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

    # ---------------------------------------------------------------------------------------- about the network itself
    def time(self, network: str, n: int = 30) -> Dict[str, Any]:
        """milliseconds per module and per frame at batch 1 (idle GPU only: the adapter refuses a busy one)"""
        if not hasattr(self.ad, "time"):
            raise EvaluationError("this project's adapter defines no timing")
        return self.ad.time(self.resolve(network), n)

    def probe(self, network: str, frames: int = 3, device: str = "cpu") -> Dict[str, Any]:
        """the module probe: statistics and dead channels of every module, the response when the camera or the radar is removed (frames = how many frames of the first scene)"""
        if not hasattr(self.ad, "probe"):
            raise EvaluationError("this project's adapter defines no probe")
        return self.ad.probe(self.resolve(network), frames, device)


Runner.METHODS = tuple(n for n, f in vars(Runner).items() if callable(f) and not n.startswith("_"))      # the methods are the commands: no second list
