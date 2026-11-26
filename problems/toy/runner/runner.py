"""The Runner of the toy problem: a trained network on frames, the predictions stored as a ResultSet (the contract of autoresearch/Evaluation: every project
names its Runner in evaluation/evaluation.json). It reads checkpoints and frames and computes no number about quality.

A network with a memory runs whole scenes in time order (CARRY); the toy has none, but the order and the stored identity are the same as for any problem.
"""
from __future__ import annotations

import os
import time

from Evaluation.api import CARRY, EvaluationError, Frames, Helpers, names
from Evaluation.contracts import FrameRef
from Evaluation.results import Locked, ResultSet, result_id, writer_lock
from Evaluation.runner import Runner as FrameRunner

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


class Runner(Helpers):
    METHODS = ("predict", "plan", "list_results", "tap", "time", "probe")

    def __init__(self, project=None, adapter=None):
        super().__init__(project or ROOT, adapter)

    def _question(self, network, frames, workers, control):
        net, fr = self.resolve(network), Frames.parse(frames)
        ident = self.net_identity(net)
        return net, fr, ident, self._source(net, fr.of, workers, control, False)

    def predict(self, network, frames="val", taps=(), device="cpu", workers=0, control="", progress=0):
        """run the network on the frames that are not stored yet and store what it predicts; -> the id of the predictions and what was done"""
        net, fr, ident, source = self._question(network, frames, workers, control)
        taps = names(taps)
        try:
            scenes, keep = fr.resolve(source)
            identity = self.identity(ident, source)
            rid = result_id(identity)
            path = os.path.join(self.ad.out(net), "results", rid)
            have = self.stored(net, rid)
            todo = {r for r in keep if not (have and have.has(r, taps))}
            started = time.time()
            if todo:
                try:
                    with writer_lock(path):
                        self._run(net, source, scenes, todo, path, identity, ident, taps, device, started)
                except Locked as e:
                    raise EvaluationError(str(e)) from None
            return {"results": f"{self.ad.name(net)}@{rid}", "frames_kept": len(keep), "frames_predicted": len(todo), "seconds_network": round(time.time() - started, 2),
                    "predictions_reused": not todo, "warnings": [], "network": net}
        finally:
            self._release(source)

    def _run(self, net, source, scenes, todo, path, identity, ident, taps, device, started):
        subject, rs, ms = self.ad.subject(net, source, device), None, []
        try:
            if self.net_identity(net) != ident:
                raise EvaluationError(f"{net} changed while it was loaded (a new checkpoint?): nothing was stored; run again")
            declared = {t.name for t in subject.taps()}
            if set(taps) - declared:
                raise EvaluationError(f"unknown taps {sorted(set(taps) - declared)}: the network offers {sorted(declared)}")
            walk = FrameRunner(source, subject, CARRY)
            for scene in scenes:
                if not any(r.scene == scene for r in todo):
                    continue
                for sample, pred in walk.iter_scene(scene, source.frames(scene), taps=taps):
                    if sample.ref in todo:
                        rs = rs or ResultSet.create(path, identity)
                        rs.add(sample.ref, pred)
                        ms.append(float(pred.info.get("predict_ms", 0.0)))
                if rs is not None:
                    rs.save()
        finally:
            if rs is not None:
                rs.close(wall_s=time.time() - started, cost={"median_ms": sorted(ms)[len(ms) // 2], "n": len(ms)} if ms else None)
            subject.close()

    def plan(self, network, frames="val", taps=(), workers=0, control=""):
        """the dry run of `predict`: nothing runs"""
        res = {"network": network, "frames": str(frames), "ok": False, "problems": []}
        try:
            net, fr, ident, source = self._question(network, frames, workers, control)
        except (EvaluationError, ValueError, KeyError, OSError) as e:
            res["problems"].append(str(e))
            return res
        try:
            scenes, keep = fr.resolve(source)
            rid = result_id(self.identity(ident, source))
            rs = self.stored(net, rid)
            have = sum(1 for r in keep if rs and rs.has(r, names(taps)))
            res.update({"ok": True, "network": net, "results": f"{self.ad.name(net)}@{rid}", "scenes_run": len(scenes), "frames_kept": len(keep), "frames_stored": have,
                        "frames_to_run": len(keep) - have, "out": self.ad.out(net)})
        finally:
            self._release(source)
        return res

    def list_results(self, network):
        net = self.resolve(network)
        base, name, rows = os.path.join(self.ad.out(net), "results"), self.ad.name(net), []
        for d in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            try:
                rs = ResultSet.open(os.path.join(base, d))
            except (OSError, ValueError):
                continue
            rows.append({"id": f"{name}@{d}", "kind": "predictions", "frames": rs.n_frames(), "megabytes": rs.size_mb(), "created": rs.manifest.get("created")})
        return rows

    def tap(self, results, frame, name=""):
        rname, _, rid = str(results).partition("@")
        net = self._networks.get(rname) or self.resolve(self.ad.network(rname))
        rs, parsed = self.stored(net, rid), FrameRef.parse(frame)
        if rs is None or parsed is None or not rs.has(FrameRef(*parsed)):
            raise EvaluationError(f"no stored frame {frame!r} in {results}")
        ref = FrameRef(*parsed)
        if not name:
            return {"results": results, "frame": ref.label, "taps": rs.tap_names(ref)}
        if name not in rs.tap_names(ref):
            raise EvaluationError(f"{frame} has no tap {name!r} (predict --taps {name})")
        return {"results": results, "frame": ref.label, "tap": name, "path": rs.tap_path(ref, name)}

    def time(self, network, n=200):
        return self.ad.time(self.resolve(network), n)

    def probe(self, network, frames=3, device="cpu"):
        raise EvaluationError("the toy defines no probe")
