"""EvalSuite: the one object the rest of the toolchain uses.

    suite = EvalSuite(source, subject, task, out="eval_out")
    suite.qual.frame("scene-0061/12", views=("overview", "debug"))      # one concrete frame
    suite.qual.random(6, seed=1)                                        # first a random video, then a random frame, 6 times
    suite.qual.videos(2)                                                # 2 random sequences as videos
    run = suite.quant.run(n_scenes=20)                                  # numbers over many videos, stored per frame
    suite.quant.table(run); suite.quant.outliers(run); suite.quant.report(run)
    suite.check()                                                       # does the adapter keep the contracts?
`state` is the suite-wide default of how the Subject's memory is treated: 'carry' (frame to frame along a scene, as deployed) or
'reset' (every frame independent).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from .contracts import Source, Subject, Task


class EvalSuite:
    def __init__(self, source: Source, subject: Subject, task: Task, out: str = "eval_out", state: str = "carry"):
        if state not in ("carry", "reset", "warmup"):
            raise ValueError("state must be 'carry', 'reset' or 'warmup'")
        self.source, self.subject, self.task = source, subject, task
        self.out, self.state = os.path.abspath(out), state
        os.makedirs(self.out, exist_ok=True)
        from .qualitative import Qualitative
        from .quantitative import Quantitative
        self.qual = Qualitative(self)
        self.quant = Quantitative(self)

    def check(self) -> List[Dict[str, Any]]:
        """run the contract checker of the adapter (see check.py); a list of {name, ok, note}"""
        from .check import check_adapter
        return check_adapter(self)

    def close(self) -> None:
        self.subject.close()
