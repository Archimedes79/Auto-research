import os, sys, resource, time
import torch
R = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/data/prep", "training/train"):
    sys.path.insert(0, os.path.join(R, sub))
import jmap as JM
import jtoken as JK
arm = sys.argv[1]
B, K, M = 8, 512, 704
torch.manual_seed(0)
torch.set_num_threads(4)
b = JK.synthetic_batch(B, K, seed=1)
head = JK.JTokenHead(map_block=JM.MapBlock()) if arm == "m" else JK.JTokenHead()
if arm == "m":
    JM_ = head.map_block
    JM._trained(0)
    for p in (head.map_block.o.weight, head.map_block.o.bias):
        torch.nn.init.normal_(p, std=0.3)
maps = JM._maps(B, M, 2, 0) if arm == "m" else None
head.train()
r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
t = time.time()
out = head(b, maps=maps)
loss, _ = JK.jtoken_loss(out, b)
loss.backward()
r1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print("arm %s B8 K512 M704 fp32 forward+loss+backward: %.1f s, peak RSS before %.0f MB, after %.0f MB (+%.0f MB), loss %.3f" % (arm, time.time() - t, r0, r1, r1 - r0, float(loss)))
if arm == "m":
    # block only: the bias path tensors
    x = torch.randn(B, K, 64)
    blk = JM._trained(1)
    pos, dirv = b["pos"].float(), b["dirv"].float()
    valid = b["valid"].bool()
    r2 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    xx = x.requires_grad_()
    blk(xx, pos, dirv, valid, maps).square().mean().backward()
    print("block alone: peak RSS %.0f MB (before %.0f)" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, r2))
