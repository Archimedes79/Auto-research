import os, sys, time
ROOT = os.environ["BFLC_ROOT"]
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.join(ROOT, "training/data"))
import lane_live_tok as LL
torch.set_num_threads(2)
def tf32(t):
    i = t.contiguous().view(torch.int32)
    return ((i + 0x1000) & ~0x1FFF).view(torch.float32)
def emulate(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            def fwd(x, m=m):
                return F.conv2d(tf32(x.float()), tf32(m.weight), m.bias, m.stride, m.padding, m.dilation, m.groups)
            m.forward = fwd
n, B = 16, 4
ds = LL.LiveData("val", 0)
res = {}
for emu in (False, True):
    src = LL.LiveSource(os.path.join(ROOT, "runs/e292_a_s0"), 512, False).eval()
    if emu:
        emulate(src.model)
    outs = []
    for i0 in range(0, n, B):
        b = LL.collate([ds[i] for i in range(i0, i0 + B)])
        with torch.no_grad():
            got = src(b["x"], b["table"], b["token"], b["row"])["batch"]
        for j in range(B):
            live = {k: got[k][j].numpy() for k in LL.GRAPH_KEYS + ("valid", "ytype", "y", "ylong", "yscale")}
            live.update(gt_out=int(got["gt_out"][j]), cell=got["tokens_np"][j]["cell"], label={k: v[j].numpy() for k, v in got["label"].items()})
            outs.append(live)
    res[emu] = outs
cmp = [LL.compare_frame(a, w) for a, w in zip(res[True], res[False])]
print("TF32-emulated CPU vs plain float32 CPU: same cell list on", sum(c["same_order"] for c in cmp), "of", len(cmp), "| max diff", {k: float("%.1e" % max(c["floats"][k] for c in cmp)) for k in cmp[0]["floats"]}, "| entries", {k: sum(c["ints"][k] for c in cmp) for k in cmp[0]["ints"]})
