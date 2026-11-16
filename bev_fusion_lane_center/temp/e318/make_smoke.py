"""E318 scratch: a CPU-built smoke checkpoint of the unc model (E307 maps of seed 0 + an untrained JEnds(unc=True, scales=True)) as runs/e318_smoke_cpu_s0/model.pt, to test `jends_eval.py dump` with the logsig key.
usage: wsl bash tools/env/_py14.sh e318/make_smoke.py    (CPU only; the run folder is scratch, a smoke run like e312_smoke_cpu_s0)"""
import os
import sys

import torch

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import jends_train as JT  # noqa: E402

torch.manual_seed(0)
ck = torch.load(os.path.join(ROOT, "runs", "e307_n_s0", "model.pt"), map_location="cpu")
m = JT.JEndModel(ck["args"], order=True, evidence=True, scales=True, e2e=True, unc=True)
m.jmodel.load_state_dict(ck["state"])
a = dict(name="e318_smoke_cpu_s0", seed=0, maps=os.path.join(ROOT, "runs", "e307_n_s0", "model.pt"), no_order=False, no_evidence=False, cross_w=1.0, scales=True, driven_w=0.0, capacity=False, ntok=False,
         geom_w=0.5, unc=True, e2e=True, maps_w=1.0, raster="legacy", steps=0, lr=1e-3, bs=8, workers=0, smoke=True, maps_args=ck["args"])
d = os.path.join(ROOT, "runs", "e318_smoke_cpu_s0")
os.makedirs(d, exist_ok=True)
torch.save(dict(state=m.state_dict(), args=a), os.path.join(d, "model.pt"))
print("wrote", d)
