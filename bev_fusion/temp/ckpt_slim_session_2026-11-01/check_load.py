"""the evaluation's own load path for one run: the run's pinned code snapshot (runs/<name>/code) imports train.py, bev_adapter.load_run
loads the last checkpoint and applies its EMA. Prints a checksum of the weights.   usage (from bev_fusion/): check_load.py RUN [CKPT_DIR]"""
import os
import sys

import torch

torch.set_num_threads(2)
run = sys.argv[1]
ckpt_dir = sys.argv[2] if len(sys.argv) > 2 else None
sys.path.insert(0, os.path.abspath("evaluation"))
import bev_adapter as A  # noqa: E402

A.setup_paths(run)
ns = A._chain()
cfg, model, loss, step, ckpt = A.load_run(run, "cpu", ckpt_dir, ns)
total = sum(float(p.double().sum()) for p in model.parameters())
print(f"LOADED {os.path.basename(run)} step {step} {os.path.basename(ckpt)} snapshot {os.path.basename(os.path.dirname(os.path.abspath(sys.modules[ns.T.__name__].__file__)))} weights-sum {total:.6f}")
