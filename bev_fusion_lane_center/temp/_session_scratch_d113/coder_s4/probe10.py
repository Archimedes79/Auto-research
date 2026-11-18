import os, sys, time
ROOT = os.environ["BFLC_ROOT"]
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.join(ROOT, "training/data"))
import lane_live_tok as LL
import lane_ptok_data as LP
torch.set_num_threads(2)
def tf32(t):                                   # round to the 10-bit mantissa of TF32 (round half away), accumulation stays float32
    i = t.contiguous().view(torch.int32)
    return ((i + 0x1000) & ~0x1FFF).view(torch.float32)
def emulate(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            def fwd(x, m=m):
                return F.conv2d(tf32(x.float()), tf32(m.weight), m.bias, m.stride, m.padding, m.dilation, m.groups)
            m.forward = fwd
n, B = 16, 4
ds, pt = LL.LiveData("val", 0), LP.PTokData("val", 0, "e292_a")
for emu in (False, True):
    src = LL.LiveSource(os.path.join(ROOT, "runs/e292_a_s0"), 512, False).eval()
    if emu:
        emulate(src.model)
    cmp = []
    for i0 in range(0, n, B):
        idx = list(range(i0, i0 + B))
        b = LL.collate([ds[i] for i in idx])
        with torch.no_grad():
            got = src(b["x"], b["table"], b["token"], b["row"])["batch"]
        for j, i in enumerate(idx):
            live = {k: got[k][j].numpy() for k in LL.GRAPH_KEYS + ("valid", "ytype", "y", "ylong", "yscale")}
            live.update(gt_out=int(got["gt_out"][j]), cell=got["tokens_np"][j]["cell"], label={k: v[j].numpy() for k, v in got["label"].items()})
            o = pt[i]
            cmp.append(LL.compare_frame(live, dict({k: o[k] for k in live if k not in ("gt_out", "cell", "label")}, gt_out=o["gt_out"], cell=o["tokens_np"]["cell"], label=o["label"])))
    print("TF32 emulated convs" if emu else "plain float32 CPU", ": frames with the same cell list", sum(c["same_order"] for c in cmp), "of", len(cmp),
          "| max diff", {k: float("%.1e" % max(c["floats"][k] for c in cmp)) for k in cmp[0]["floats"]}, "| differing entries", {k: sum(c["ints"][k] for c in cmp) for k in cmp[0]["ints"]})
