"""scratch: evaluation/bev_timing.py with ONE change of the idle guard, in this process only (the frozen file is not touched):
on this machine nvidia-smi now lists the timing process ITSELF as a compute app ('<pid>, [N/A]') once its model is loaded, so the guard's second check
('the GPU got busy while the model was loading') always refuses. Here the process's own pid is dropped from `other_processes`; every other check stays
(utilisation < 10 %, no other compute app, no trainer / evaluation process in ps). Reported as an open finding for the owner of bev_timing.py.
    python temp/t16/timing_patched.py RUN [--n 30]     (from the bev_fusion folder)"""
import os
import sys

sys.path.insert(0, os.getcwd())
from evaluation import bev_timing as BT  # noqa: E402

_orig = BT.gpu_state


def gpu_state(samples=3, gap=0.5):
    st = _orig(samples, gap)
    own = str(os.getpid())
    st["other_processes"] = [p for p in st["other_processes"] if p.split(",")[0].strip() != own]
    return st


BT.gpu_state = gpu_state
sys.exit(BT.main(sys.argv[1:]))
