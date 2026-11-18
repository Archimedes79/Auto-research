"""E374 (S3) inference cost with the aging wait (scratch, 2026-11-18): runs `jmap_eval.py time <args>` with `jtiming.gpu_state` waiting 2.5 s before its SECOND call (the after-sample), the wait `jtiming.py`'s own protocol has and `jmap_eval.py time` lacks (nvidia-smi averages over its window, so an at-once sample reads this process's own kernels: 65 - 70 % and a false NOT QUIET).
usage: python time_s3_aged.py --run e374_m_s0   (the arguments after are those of `jmap_eval.py time`)"""
import os
import runpy
import sys
import time

R = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training/data", "training/grid", "training/train"):
    sys.path.insert(0, os.path.join(R, sub))
import jtiming as JTG  # noqa: E402

_orig, _n = JTG.gpu_state, [0]


def _aged():
    _n[0] += 1
    if _n[0] >= 2:
        time.sleep(2.5)
    return _orig()


JTG.gpu_state = _aged
sys.argv = ["jmap_eval.py", "time"] + sys.argv[1:]
runpy.run_path(os.path.join(R, "evaluation", "jmap_eval.py"), run_name="__main__")
