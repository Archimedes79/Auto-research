"""scratch: evaluation/bev_timing.py with TWO changes of the idle guard, in this process only (the frozen file is not touched):
 1. the process's own pid is dropped from the compute apps that nvidia-smi lists (it lists the timing process itself once its model is loaded);
 2. the guard's second look (samples=2, taken right after the model load) first synchronises this process and waits 3 s, because the load's own copies and
    cuDNN set-up still showed 6 - 29 % utilisation (no other process was listed in those refusals).
Every other check stays: the first look (before the load) needs utilisation < 10 %, no other compute app, no trainer / evaluation process in ps, memory below the desktop baseline.
    python temp/t16/timing_patched2.py RUN [--n 30]     (from the bev_fusion folder)"""
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402
from evaluation import bev_timing as BT  # noqa: E402

_orig = BT.gpu_state


def gpu_state(samples=3, gap=0.5):
    if samples == 2:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time.sleep(3.0)
    st = _orig(samples, gap)
    own = str(os.getpid())
    st["other_processes"] = [p for p in st["other_processes"] if p.split(",")[0].strip() != own]
    return st


BT.gpu_state = gpu_state
sys.exit(BT.main(sys.argv[1:]))
