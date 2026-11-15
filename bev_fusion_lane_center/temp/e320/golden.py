"""E320 scratch: the golden numbers of the E312 pair head (runs/e312_p_s0, the `je.*` weights) on a seeded synthetic input, from the ORIGINAL module (the committed 7386e0e1): pinned in the self-test of jends.py as the flag-off identity check that survives in the repo"""
import os
import sys

import torch

torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("training", "training/grid", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
from importlib.machinery import SourceFileLoader  # noqa: E402
import importlib.util  # noqa: E402

ld = SourceFileLoader("jends_o", os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "orig", "jends.pysrc"))
J = importlib.util.module_from_spec(importlib.util.spec_from_loader("jends_o", ld))
ld.exec_module(J)
ck = torch.load(os.path.join(ROOT, "runs", "e312_p_s0", "model.pt"), map_location="cpu")
st = {k[3:]: v for k, v in ck["state"].items() if k.startswith("je.")}
m = J.JEnds(d=64, order=True, evidence=True, scales=True)
m.load_state_dict(st)
m.eval()
g = torch.Generator().manual_seed(7)
B, d = 2, 64
jout = dict(j1=torch.randn(B, 8, 100, 100, generator=g) * 0.5 - 2.5, j05=torch.randn(B, 2, 200, 200, generator=g) - 2.0, f1=torch.randn(B, d, 100, 100, generator=g))
P = dict(centre1=torch.randn(B, 12, 100, 100, generator=g))
with torch.no_grad():
    out = m(jout, P, gt=None)
print("GOLDEN", dict(logit=round(float(out["logit"].sum()), 4), kp_pos=round(float(out["kp_pos"].sum()), 4), scale=round(float(out["scale"].sum()), 4), kp_p=round(float(out["kp_p"].sum()), 4), allowed=int(out["allowed"].sum()), valid=int(out["valid"].sum())))
