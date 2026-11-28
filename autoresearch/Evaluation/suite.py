"""EvalSuite: the counting engine on a Source, a Subject (or None) and a Task. The Evaluation (api.py) builds one for each call; it is used directly to run a network that has no project
adapter (the tests do).

    suite = EvalSuite(source, subject, task, out="eval_out")
    store = suite.quant.run(name="a")          # every frame through the Subject, or results=<ResultSet>: its stored predictions -> counts per frame (store.py)
    suite.quant.table(store);  suite.quant.per_scene(store);  suite.quant.outliers(store);  suite.quant.compare("a", "b");  suite.quant.report(store)
`state`: how the Subject's memory is treated: 'carry' (frame to frame along a scene, as deployed), 'reset' or 'warmup' (runner.py)."""
from __future__ import annotations

import os
from typing import Optional

from .contracts import Source, Subject, Task
from .quantitative import Quantitative


class EvalSuite:
    def __init__(self, source: Source, subject: Optional[Subject], task: Task, out: str = "eval_out", state: str = "carry"):
        if state not in ("carry", "reset", "warmup"):
            raise ValueError("state must be 'carry', 'reset' or 'warmup'")
        self.source, self.subject, self.task = source, subject, task
        self.out, self.state = os.path.abspath(out), state
        os.makedirs(self.out, exist_ok=True)
        self.quant = Quantitative(self)

    def close(self) -> None:
        if self.subject is not None:
            self.subject.close()
