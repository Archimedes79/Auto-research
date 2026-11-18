import os, sys, time
ROOT = os.environ["BFLC_ROOT"]
for s in ("training", "training/grid", "training/data", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, s))
import torch
torch.set_num_threads(2)
import qgraph as QG
import lane_live_tok as LLT
import jtoken as JK
torch.manual_seed(0)
m = QG.QGraph(width=16, soft_nms=["mnms", 1.75, 0.1])
with torch.no_grad():
    m.nodes.head1.bias[0] = 3.0
print("nodes params", sum(p.numel() for p in m.nodes.parameters()), "all", sum(p.numel() for p in m.parameters()))
src = LLT.LiveSource(m, K=64)
x = torch.randn(2, 39, 200, 200).half()
t = time.time()
no, tk = src.tokens(x)
print("tokens", time.time() - t)
head = JK.JTokenHead(d=32, heads=4, layers=2, ffn=64, live_inputs=True)
t = time.time()
out = head(LLT.head_inputs(tk))
tabs = [LLT.toy_table([[-9.0, 5.0]], [], [])] * 2
b = src.labelled(tk, tabs, ["a", "b"], [0, 1], gate=False)
l, info = JK.jtoken_loss(out, b)
print("head+labels", time.time() - t, float(l), info)
t = time.time()
tg = {k: torch.from_numpy(__import__("numpy").stack([LLT.node_targets(tt)[k] for tt in tabs])) for k in LLT.NODE_KEYS}
lb, ib = LLT.b_loss(no, tg)
(l + lb).backward()
print("backward", time.time() - t, float(lb))
print(sorted(no))
