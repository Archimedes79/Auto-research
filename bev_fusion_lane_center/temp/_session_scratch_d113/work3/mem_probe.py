import os, sys, resource, time
import torch
R = os.environ["BFLC_ROOT"]
for sub in ("training/grid", "training/data"):
    sys.path.insert(0, os.path.join(R, sub))
import jmap as JM
torch.manual_seed(0)
blk = JM._trained(0)
B, K, M = 8, 512, 512
x, pos, dirv, valid = JM._tokens(B, K, 1, 0)
maps = JM._maps(B, M, 2, 0)
r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
t = time.time()
x = x.requires_grad_()
out = blk(x, pos, dirv, valid, maps)
out.square().mean().backward()
print("fp32 B8 K512 M512 forward+backward: %.1f s, peak RSS %.0f MB (before %.0f MB), parameters %d" % (time.time() - t, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, r0, sum(p.numel() for p in blk.parameters())))
