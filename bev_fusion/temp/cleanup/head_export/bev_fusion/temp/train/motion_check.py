"""One-off: does the motion (velocity) head of a run train? Norms and changes of its weights across the run's checkpoints."""
import glob
import sys

import torch

run = sys.argv[1]
cks = sorted(glob.glob(f"{run}/ckpt_*.pt"))
sds = [torch.load(c, map_location="cpu", weights_only=False) for c in cks]
keys = [k for k in sds[0]["state"] if "mover.motion" in k]
print("checkpoints", [c.split("/")[-1] for c in cks])
for k in keys:
    a = [s["state"][k].float() for s in sds]
    e = [s["ema"].get(k[6:], torch.zeros(1)).float() for s in sds]
    print(f"{k[6:]:45s} shape {tuple(a[0].shape)} norm {[round(float(x.norm()), 5) for x in a]} "
          f"change {[round(float((a[i + 1] - a[i]).norm()), 6) for i in range(len(a) - 1)]} ema norm {[round(float(x.norm()), 5) for x in e]}")
