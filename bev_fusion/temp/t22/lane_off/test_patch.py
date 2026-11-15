"""scratch: load-only check of suite_lane_off.install() on the CPU (2 threads, no data): the run loads, the lane branch is there before and gone after, and the parameter count drops by exactly
the lane branch's. Run from bev_fusion/ (after make_runs.py):  OMP_NUM_THREADS=2 python temp/t22/lane_off/test_patch.py"""
import os
import sys

import torch

torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(__file__))
import suite_lane_off  # noqa: E402

sys.path.insert(0, suite_lane_off.ROOT)
from evaluation import bev_adapter  # noqa: E402

bev_adapter.setup_paths("runs/diag_t20_L4on")           # no code/ folder in the diagnostic run: the live training/ (as the suite does it)
plain_load = bev_adapter.load_run                       # the original, before the patch
ba = suite_lane_off.install()
cfg, model, loss, step, path = plain_load("runs/diag_t20_L4on", "cpu", "runs/t20_base_s0")
print("twin (original loader): lane present:", model.lane is not None, "step", step)
assert model.lane is not None
n_all = sum(p.numel() for p in model.parameters())
n_lane = sum(p.numel() for p in model.lane.parameters())
print("parameters:", n_all, "of which lane:", n_lane)
cfg2, model2, loss2, step2, path2 = ba.load_run("runs/diag_t20_L4off", "cpu", "runs/t20_base_s0")
assert model2.lane is None
n2 = sum(p.numel() for p in model2.parameters())
print("after the drop:", n2, "= all minus lane:", n2 == n_all - n_lane)
assert n2 == n_all - n_lane and step2 == step
print("OK")
