"""Layer 2 of runsuite, the class interface: `Runner`, the trained network as an object. Layer 1 is the command line (cli.py: every method below is a command), layer 3 the functionality
(engine.py: the work). This file holds no work of its own: a method here is the documented signature of a command, and its body is one call of the engine.

    r = Runner("../bev_fusion")                                      # the project (default: AR_PROJECT); its evaluation adapter says what a network and its data are
    r.plan("runs/t23_C_s1", "small.val")                             # the dry run: what is stored, what would run, the problems
    r.predict("runs/t23_C_s1", "small.val", taps=["fused"])          # the network on the frames, no ground truth, stored; -> {results: "<net>@r<hash>", ...}
    r.list_results("runs/t23_C_s1");  r.tap(id, "scene-0061/12", "fused");  r.time("runs/t23_C_s1");  r.probe("runs/t23_C_s1")

No queue and no launcher: a call runs in this process. The same call as a job of the project (ledger row, GPU slot, the queue of the trainings) is `RunJobs` (jobs.py), whose mode "predict"
starts `python -m runsuite Runner.predict`."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import suitecli
from evalsuite.api import EvaluationError as RunnerError          # the data, the checkpoint, the memory: the errors of the engine are the evaluation's, one class for both suites

from .engine import Engine


class Runner:
    METHODS = ("predict", "plan", "list_results", "tap", "time", "probe")         # every one is a command; resolve and stored are helpers of the Evaluation

    def __init__(self, project: Optional[str] = None, adapter: Any = None):
        self.project, self._adapter, self._engine = project, adapter, None

    @classmethod
    def methods(cls) -> List[Dict[str, str]]:
        """the methods that are commands: [{name, arguments, summary}]"""
        return suitecli.method_list(cls)

    @property
    def engine(self) -> Engine:
        """layer 3, made when first needed: it opens the project's evaluation adapter"""
        if self._engine is None:
            self._engine = Engine(self.project or None, self._adapter)
        return self._engine

    def predict(self, network: str, frames: str = "val", taps: Sequence[str] = (), device: str = "cuda", workers: int = 2, control: str = "", progress: int = 50) -> Dict[str, Any]:
        """run the network on the frames that are not stored yet (whole scenes in time order, warm: D70) and store what it predicts, with the intermediate results `taps` when asked; no ground truth is read. -> the id of the predictions, what was done and the warnings of the memory's health"""
        return self.engine.predict(network, frames, taps, device, workers, control, progress)

    def plan(self, network: str, frames: str = "val", taps: Sequence[str] = (), workers: int = 0, control: str = "") -> Dict[str, Any]:
        """the dry run of `predict`: what it would do with these frames (scenes, what is stored, what would run) and what is wrong; no frame runs, the data and the checkpoint are opened. -> {ok, problems, results, frames_to_run, ...}"""
        return self.engine.plan(network, frames, taps, workers, control)

    def list_results(self, network: str) -> List[Dict[str, Any]]:
        """the stored predictions of a network with their size. -> a list of {id, frames, megabytes, created}"""
        return self.engine.list_results(network)

    def tap(self, results: str, frame: str, name: str = "") -> Dict[str, Any]:
        """the intermediate results stored with a frame's predictions: without `name` which taps there are, with it the path of its .npy file, its shape and dtype."""
        return self.engine.tap(results, frame, name)

    def time(self, network: str, n: int = 30) -> Dict[str, Any]:
        """milliseconds per module and per frame of a network at batch 1 (idle GPU only: the project's adapter refuses a busy one)."""
        return self.engine.time(network, n)

    def probe(self, network: str, frames: int = 3, device: str = "cpu") -> Dict[str, Any]:
        """the module probe: statistics and dead channels of every module and the response when the camera or the radar is removed (frames = how many frames of the first scene)."""
        return self.engine.probe(network, frames, device)

    # ---------------------------------------------------------------------------------------- helpers of the Evaluation (not commands)
    def resolve(self, text: str) -> str:
        """a network named by run name or path -> its path"""
        return self.engine.resolve(text)

    def stored(self, net: str, rid: str) -> Any:
        """the stored predictions `rid` of a network, or None"""
        return self.engine.stored(net, rid)
