"""Scratch: a student checkpoint written by the distilling trainer loads in the pair-stage model (JEndModel, jends_train) with maps_args from the checkpoint. CPU."""
import os
import sys

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/train", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import torch  # noqa: E402
import jends_train as JT  # noqa: E402

ck = torch.load(os.path.join(ROOT, "runs/smoke_e339_s0/model.pt"), map_location="cpu", weights_only=False)
print({k: ck["args"][k] for k in ("d", "ctx", "sigma", "teacher", "distill_w")})
m = JT.JEndModel(ck["args"], order=True, evidence=True, scales=True, e2e=True)
r = m.jmodel.load_state_dict(ck["state"], strict=True)
print("JModel strict load:", r, "| head params", sum(p.numel() for p in m.jmodel.jm.parameters()) / 1e6, "M")
