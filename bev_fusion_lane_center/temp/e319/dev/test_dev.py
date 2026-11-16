"""E319 dev tests (scratch): the modified copy `jmaps_dev` against the original `jmaps_orig` (git HEAD). CPU only.
usage: OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= wsl bash tools/env/_py14.sh e319/dev/test_dev.py"""
import hashlib
import time

import torch
import torch.nn.functional as F

import jmaps_dev as D
import jmaps_orig as O

torch.set_num_threads(2)


def mk_P(seed=1, B=2):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    return dict(feat2=r(B, 144, 50, 50), feat1=r(B, 48, 100, 100), feat05=r(B, 32, 200, 200), centre1=r(B, 12, 100, 100))


def mk_batch(B=2):
    b = dict(jz=torch.zeros(B, 2, 100, 100), jheat=torch.zeros(B, 2, 100, 100), jdir=torch.zeros(B, 4, 100, 100), jpaint=torch.zeros(B, 2, 200, 200))
    b["jz"][:, :, 40:60, 40:60] = 1
    b["jheat"][:, 0, 30, 50] = 1
    b["jdir"][:, 0, 30, 50], b["jdir"][:, 1, 30, 50] = 0.0, 1.0
    return b


def build(mod, ctx, seed=0):
    torch.manual_seed(seed)
    return mod.JMaps(ctx=ctx)


P, batch = mk_P(), mk_batch()

# T1: the flag off is bit-identical to the original: weights, outputs, loss, gradients
for ctx in ("none", "dil", "attn"):
    a, b = build(O, ctx), build(D, ctx)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa), ctx
    oa, ob = a(P), b(P)
    assert all(torch.equal(oa[k], ob[k]) for k in oa), ctx
    la, _ = O.jmaps_loss(oa, batch)
    lb, _ = D.jmaps_loss(ob, batch)
    assert torch.equal(la, lb), ctx
    la.backward()
    lb.backward()
    assert all(torch.equal(p.grad, q.grad) for p, q in zip(a.parameters(), b.parameters())), ctx
print("T1 flag off == original (weights, outputs, loss, gradients): ok")

# T2: flag on at initialisation == flag off (same seed): the block starts as the identity
a, b = build(D, "dil"), build(D, "dil+dirx")
sa, sb = a.state_dict(), b.state_dict()
assert all(torch.equal(sa[k], sb[k]) for k in sa) and set(sb) - set(sa) and all(k.startswith("dirx.") for k in set(sb) - set(sa))
oa, ob = a(P), b(P)
assert all(torch.equal(oa[k], ob[k]) for k in oa)
print("T2 flag on at init == flag off (bit-identical outputs; the other", len(sa), "tensors identical; new tensors", len(set(sb) - set(sa)), "): ok")

# T3: gradients: step 0 reaches the last layers only (zero start), after two steps every tensor of the block
m = build(D, "dil+dirx")
loss, _ = D.jmaps_loss(m(P), batch)
loss.backward()
live0 = [n for n, p in m.named_parameters() if n.startswith("dirx.") and p.grad is not None and float(p.grad.abs().sum()) > 0]
print("T3 step 0 live dirx tensors:", live0)
assert live0 and all(n.endswith("3.weight") or n.endswith("3.bias") for n in live0)
for _ in range(2):
    with torch.no_grad():
        for p in m.parameters():
            if p.grad is not None:
                p.sub_(0.05 * p.grad)
    m.zero_grad()
    loss, _ = D.jmaps_loss(m(P), batch)
    loss.backward()
dead = [n for n, p in m.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
assert not dead, dead
print("T3 after 2 steps no dead tensor in the whole head: ok")

# T4: the axis convention of shift_along: rows = x
H = 100
f = torch.arange(H).float().view(1, 1, H, 1).expand(1, 1, H, H).contiguous()                  # f = row
t = torch.zeros(1, 2, H, H)
t[:, 0] = 1.0                                                                                  # heading along +x = along the rows
o = D.shift_along(f, t, 2.0)
assert torch.allclose(o[0, 0, :90, 5], f[0, 0, 2:92, 5], atol=1e-5), o[0, 0, :5, 5]
o = D.shift_along(f, t, -3.0)
assert torch.allclose(o[0, 0, 5:90, 5], f[0, 0, 2:87, 5], atol=1e-5)
t2 = torch.zeros(1, 2, H, H)
t2[:, 1] = 1.0                                                                                 # heading along +y = along the columns: a function of the row does not change
assert torch.allclose(D.shift_along(f, t2, 4.0)[0, 0, :, 5:90], f[0, 0, :, 5:90], atol=1e-5)
fl = (2.0 * torch.arange(H).float().view(1, 1, H, 1) + 3.0 * torch.arange(H).float().view(1, 1, 1, H)).expand(1, 1, H, H).contiguous()   # linear in (row, col): bilinear sampling is exact
c, s_ = 0.6, 0.8
t3 = torch.zeros(1, 2, H, H)
t3[:, 0], t3[:, 1] = c, s_
o = D.shift_along(fl, t3, 5.0)
assert torch.allclose(o[0, 0, 10:80, 10:80], fl[0, 0, 10:80, 10:80] + 5.0 * (2 * c + 3 * s_), atol=1e-4)
print("T4 shift_along axes (rows = x), sign (negative = behind) and exact bilinear sampling on a linear field: ok")

# T5: behind == ahead of the reversed heading; T6: differentiable in the heading
fr = torch.randn(1, 3, H, H)
tr = F.normalize(torch.randn(1, 2, H, H), dim=1)
assert torch.equal(D.shift_along(fr, tr, -4.0), D.shift_along(fr, -tr, 4.0))
tg = tr.clone().requires_grad_(True)
D.shift_along(fr, tg, 4.0).square().sum().backward()
assert float(tg.grad.abs().sum()) > 0
print("T5 / T6 mirror symmetry and gradient into the heading: ok")

# T7: a JModel round trip: the state of a flag-off model loads into the flag-on model (only the dirx tensors are new), and the flag-on state loads strictly
Jo, Jd = D.JModel(ctx="dil"), D.JModel(ctx="dil+dirx")
rep = Jd.load_state_dict(Jo.state_dict(), strict=False)
assert not rep.unexpected_keys and rep.missing_keys and all(k.startswith("jm.dirx.") for k in rep.missing_keys), rep
Jd2 = D.JModel(ctx="dil+dirx")
Jd2.load_state_dict(Jd.state_dict())
print("T7 JModel round trip: ok; new tensors", len(rep.missing_keys))

# T8: size and cost of the block (batch 1 at 100 x 100, CPU)
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

P1 = mk_P(B=1)
for ctx in ("dil", "dil+dirx"):
    m = build(D, ctx).eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(P1)
    npar = sum(p.numel() for p in m.parameters())
    ts = []
    with torch.no_grad():
        m(P1)
        for _ in range(10):
            t0 = time.perf_counter()
            m(P1)
            ts.append(time.perf_counter() - t0)
    print(f"T8 JMaps {ctx:9s} params {npar / 1e3:8.1f} k   {fc.get_total_flops() / 2e9:6.3f} GMAC (conv / matmul)   CPU 2 threads, batch 1: median {sorted(ts)[len(ts) // 2] * 1e3:7.1f} ms")
dx = D.DirX().eval()
f1 = torch.randn(1, 64, 100, 100)
with torch.no_grad(), FlopCounterMode(display=False) as fc:
    dx(f1, P1)
ts = []
with torch.no_grad():
    for _ in range(10):
        t0 = time.perf_counter()
        dx(f1, P1)
        ts.append(time.perf_counter() - t0)
print(f"T8 DirX alone: params {sum(p.numel() for p in dx.parameters())}, {fc.get_total_flops() / 2e9:.3f} GMAC (conv only; grid_sample not counted: 2 rounds x 6 shifts x 64 ch x 1e4 cells x 4 taps = {2 * 6 * 64 * 1e4 * 4 / 1e9:.3f} G bilinear taps), CPU median {sorted(ts)[len(ts) // 2] * 1e3:.1f} ms")

# golden numbers of the ORIGINAL for the permanent test (seed 0 weights, seeded input)
for ctx in ("none", "dil", "attn"):
    o = build(O, ctx)(P)
    npar = sum(p.numel() for p in build(O, ctx).parameters())
    print("GOLDEN", ctx, npar, "j1 %.6f j05 %.6f f1 %.6f" % (float(o["j1"].double().sum()), float(o["j05"].double().sum()), float(o["f1"].double().sum())),
          hashlib.md5(b"".join(k.encode() for k in build(O, ctx).state_dict())).hexdigest()[:8])
