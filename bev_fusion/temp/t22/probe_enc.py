import os, sys, time, resource
sys.path.insert(0, "training")
import torch
torch.set_num_threads(2)
from student_encoder import StudentEncoder, h8_warp_grid
g, xy = h8_warp_grid()
print("grid", g.shape, "range x", float(g[...,0].min()), float(g[...,0].max()), "y", float(g[...,1].min()), float(g[...,1].max()))
print("xy corners", xy[0,0].tolist(), xy[0,-1].tolist(), xy[-1,0].tolist(), xy[-1,-1].tolist())
m = StudentEncoder()
print("params M", sum(p.numel() for p in m.parameters())/1e6, "train", sum(p.numel() for p in m.parameters() if p.requires_grad)/1e6)
x = torch.randint(0, 256, (1, 3, 256, 704), dtype=torch.uint8)
for flag in (True, False):
    m.freeze_backbone(flag); m.train()
    t = time.time()
    for _ in range(3):
        out = m(x)
        loss = sum(out[k].pow(2).mean() for k in ("logit16","ctx16","logit8","ctx8"))
        m.zero_grad(); loss.backward()
    print("frozen" if flag else "unfrozen", "fwd+bwd s", (time.time()-t)/3, "maxrss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
print({k: tuple(v.shape) for k, v in out.items()})
