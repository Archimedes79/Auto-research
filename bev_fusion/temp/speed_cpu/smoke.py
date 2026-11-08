import resource
import sys
import time

import common as C
import torch

B = int(sys.argv[1]) if len(sys.argv) > 1 else 1
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 15
t0 = time.time()
cfg, model, loss, batch, unknown = C.make(B=B, n_boxes=nb)
print("unknown keys:", unknown, "params", sum(p.numel() for p in model.parameters()) / 1e6, "M", "grid", model.grid.nx, model.grid.ny, "build %.1fs" % (time.time() - t0), flush=True)
print("rss MB after build", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, flush=True)
t0 = time.time()
out, res, _ = C.T.forward_loss(model, loss, batch, 0.5, "cpu", "off", cfg)
print("forward+loss %.1fs total %.3f" % (time.time() - t0, float(out["total"])), "rss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, flush=True)
t0 = time.time()
out["total"].backward()
print("backward %.1fs" % (time.time() - t0), "rss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, flush=True)
