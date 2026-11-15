import os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "training", "grid"))
import jstate as JS
torch.set_num_threads(2)
n, d = 64, 64
L = JS.StateAttn(d=d)
macs = [0]
def hk(m, i, o):
    if isinstance(m, torch.nn.Linear): macs[0] += i[0].numel() // m.in_features * m.in_features * m.out_features
for m in L.modules(): m.register_forward_hook(hk)
t = torch.randn(1, n, d); pos = torch.randn(1, n, 2) * 20; D = torch.nn.functional.normalize(torch.randn(1, n, 2), dim=-1)
K = n // 2
st = dict(pos=pos, dir=D, p=torch.rand(1, n), valid=torch.ones(1, n, dtype=torch.bool), logit=torch.randn(1, K, K), allowed=torch.ones(1, K, K, dtype=torch.bool), has=torch.ones(1, dtype=torch.bool))
L(t, pos, D, st)
lin = macs[0]
att = n * n * L.dh + n * n * 2 + n * n * JS.NF + n * n * 2   # scores q.k, geometry, w@sf, offset
sf = 2 * K * K * 8  # row summaries (softmax weights @ pos, dir)
print("params", sum(p.numel() for p in L.parameters()), "Linear MACs", lin, "attention + aggregation MACs", att, "row summary", sf, "total ~", lin + att + sf)
with torch.no_grad():
    for _ in range(5): L(t, pos, D, st)
    t0 = time.time()
    for _ in range(100): L(t, pos, D, st)
print("CPU ms per call (batch 1)", (time.time() - t0) * 10)
