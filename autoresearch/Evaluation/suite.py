"""EvalSuite: the counting engine on a Source, a Subject (or None) and a Task. The Evaluation (api.py) builds one for each call; it is used directly to run a network that has no project
adapter (the tests do).

    suite = EvalSuite(source, subject, task, out="eval_out")
    store = suite.quant.run(name="a")          # every frame through the Subject, or results=<ResultSet>: its stored predictions -> counts per frame (store.py)
    suite.quant.table(store);  suite.quant.per_scene(store);  suite.quant.outliers(store);  suite.quant.compare("a", "b");  suite.quant.report(store)
`state`: how the Subject's memory is treated: 'carry' (frame to frame along a scene, as deployed), 'reset' or 'warmup' (runner.py)."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from .contracts import Source, Subject, Task
from .quantitative import Quantitative


class Counting(Quantitative):
    """the engine of quantitative.py (a file of the evaluator hash) with what this package offers: a row of the outlier search names the command that draws its frame, the report has no
    pictures, and its two methods whose parts are gone refuse"""

    def trace(self, *args, **kwargs):
        raise NotImplementedError("quant.trace is not offered: draw a stored frame with `Evaluation.draw_frame(id, frame, view)`")

    def pareto(self, *args, **kwargs):
        raise NotImplementedError("quant.pareto is not offered: compare two evaluations with `compare` and read the ms per frame with `get_cost`")

    def outliers(self, run, *args, **kwargs) -> Dict[str, List[dict]]:
        res, store = super().outliers(run, *args, **kwargs), self.open(run)
        eval_id = store.manifest.get("evaluation", {}).get("id", "ID")
        first: Dict[str, str] = {}                                             # a scene row is looked at through its first scored frame
        for frame in store.frames:
            first.setdefault(frame["scene"], frame["label"])
        for row in (row for rows in res.values() for row in rows):
            row["view"] = f"draw-frame {eval_id} {row.get('label') or first[row['scene']]}"
        return res

    def report(self, run, k: int = 8) -> str:
        return super().report(run, render=0, k=k)


class EvalSuite:
    def __init__(self, source: Source, subject: Optional[Subject], task: Task, out: str = "eval_out", state: str = "carry"):
        if state not in ("carry", "reset", "warmup"):
            raise ValueError("state must be 'carry', 'reset' or 'warmup'")
        self.source, self.subject, self.task = source, subject, task
        self.out, self.state = os.path.abspath(out), state
        os.makedirs(self.out, exist_ok=True)
        self.quant = Counting(self)

    def close(self) -> None:
        if self.subject is not None:
            self.subject.close()
