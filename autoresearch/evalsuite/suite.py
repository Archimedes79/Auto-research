"""EvalSuite: the one object the rest of the toolchain uses.

    suite = EvalSuite(source, subject, task, out="eval_out")
    suite.qual.frame("scene-0061/12", views=("overview", "debug"))      # one concrete frame
    suite.qual.random(6, seed=1)                                        # first a random video, then a random frame, 6 times
    suite.qual.videos(2)                                                # 2 random sequences as videos
    run = suite.quant.run(n_scenes=20)                                  # numbers over many videos, stored per frame
    suite.quant.table(run); suite.quant.outliers(run); suite.quant.report(run)
    suite.probe.scan(run)                                               # where do the errors come from: budget, situations, random examples
    suite.check()                                                       # does the adapter keep the contracts?
`state` is the suite-wide default of how the Subject's memory is treated: 'carry' (frame to frame along a scene, as deployed) or
'reset' (every frame independent).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .contracts import Source, Subject, Task


class EvalSuite:
    """the engine's workbench: a Source, a Subject (None when nothing is run: scoring stored predictions, reading counts) and a Task, with the qualitative and the quantitative part on them.
    The class interface (api.py, `Evaluation`) builds one per call and is what a person uses; `EvalSuite` itself is the older entry point of bev_suite.py, the lane project and the selftests."""
    def __init__(self, source: Source, subject: Optional[Subject], task: Task, out: str = "eval_out", state: str = "carry"):
        if state not in ("carry", "reset", "warmup"):
            raise ValueError("state must be 'carry', 'reset' or 'warmup'")
        self.source, self.subject, self.task = source, subject, task
        self.out, self.state = os.path.abspath(out), state
        os.makedirs(self.out, exist_ok=True)
        from .qualitative import Qualitative
        from .quantitative import Quantitative
        self.qual = Qualitative(self)
        self.quant = Quantitative(self)
        self._probe = None

    @property
    def probe(self):
        """the probing (see probing.py): loaded when it is first used, needs a Task that implements probe()"""
        if self._probe is None:
            from .probing import Prober
            self._probe = Prober(self)
        return self._probe

    def check(self) -> List[Dict[str, Any]]:
        """run the contract checker of the adapter (see check.py); a list of {name, ok, note}"""
        from .check import check_adapter
        return check_adapter(self)

    def close(self) -> None:
        if self.subject is not None:
            self.subject.close()
