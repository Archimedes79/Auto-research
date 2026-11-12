"""E319 tests (CPU, torch): the heading-guided exchange `jmaps.DirX` (ctx "dil+dirx", `jmaps_train.py --dirx`).
 - the flag off is the E307 head: weights and outputs reproduce golden numbers taken from the code before E319 (seeded weights, seeded input), the parameter counts are unchanged;
 - the flag on starts as the flag off (bit-identical outputs for the same seed: the block's last layers are zero, its tensors are created last) and a model state without the block loads into the model with it (only `dirx` tensors are new);
 - the gradient: the zero start opens in two steps (step 0 reaches the last layers, after two steps no tensor of the head is dead), the sampling is differentiable in the heading;
 - the geometry of `shift_along`: rows = x, negative step = behind, exact bilinear sampling of a linear field, mirror symmetry;
 - the cost of the block (parameters, MACs) is printed.
Run: python test_jmaps_dirx.py   (`_py14.sh grid/test_jmaps_dirx.py`)"""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jmaps as J  # noqa: E402

torch.set_num_threads(2)
GOLDEN = {"none": (148554, -243740.090987, -640955.480696, 514074.034008), "dil": (444490, -235164.857782, -637990.757052, 514848.862545), "attn": (517962, -235640.350036, -637654.999036, 519382.281528)}   # params, sum j1, sum j05, sum f1 of the code before E319


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


def build(ctx, seed=0):
    torch.manual_seed(seed)
    return J.JMaps(ctx=ctx)


def main():
    P, batch = mk_P(), mk_batch()
    # the flag off reproduces the head of E286 / E307
    for ctx, (npar, s1, s05, sf) in GOLDEN.items():
        m = build(ctx)
        assert sum(p.numel() for p in m.parameters()) == npar and m.dirx is None, ctx
        o = m(P)
        for name, ref in (("j1", s1), ("j05", s05), ("f1", sf)):
            got = float(o[name].double().sum())
            assert abs(got - ref) <= 1e-4 * abs(ref), (ctx, name, got, ref)
    print("flag off: parameter counts and outputs reproduce the head before E319: ok")
    # the flag on starts as the flag off
    a, b = build("dil"), build("dil+dirx")
    sa, sb = a.state_dict(), b.state_dict()
    new = set(sb) - set(sa)
    assert new and all(k.startswith("dirx.") for k in new) and all(torch.equal(sa[k], sb[k]) for k in sa)
    oa, ob = a(P), b(P)
    assert all(torch.equal(oa[k], ob[k]) for k in oa)
    Jo, Jd = J.JModel(ctx="dil"), J.JModel(ctx="dil+dirx")
    rep = Jd.load_state_dict(Jo.state_dict(), strict=False)
    assert not rep.unexpected_keys and rep.missing_keys and all(k.startswith("jm.dirx.") for k in rep.missing_keys), rep
    Jd.load_state_dict(Jd.state_dict())
    print("flag on at the start == flag off (bit-identical), %d new tensors, a state without the block loads: ok" % len(new))
    # gradients
    m = build("dil+dirx")
    loss, _ = J.jmaps_loss(m(P), batch)
    loss.backward()
    live0 = [n for n, p in m.named_parameters() if n.startswith("dirx.") and p.grad is not None and float(p.grad.abs().sum()) > 0]
    assert live0 and all(n.endswith("3.weight") or n.endswith("3.bias") for n in live0), live0
    for _ in range(2):
        with torch.no_grad():
            for p in m.parameters():
                if p.grad is not None:
                    p.sub_(0.05 * p.grad)
        m.zero_grad()
        loss, _ = J.jmaps_loss(m(P), batch)
        loss.backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    assert not dead, dead
    print("gradient: step 0 reaches the zero-initialised last layers, after two steps no tensor of the head is dead: ok")
    # geometry of shift_along
    H = 100
    f = torch.arange(H).float().view(1, 1, H, 1).expand(1, 1, H, H).contiguous()
    t = torch.zeros(1, 2, H, H)
    t[:, 0] = 1.0
    assert torch.allclose(J.shift_along(f, t, 2.0)[0, 0, :90, 5], f[0, 0, 2:92, 5], atol=1e-5)
    assert torch.allclose(J.shift_along(f, t, -3.0)[0, 0, 5:90, 5], f[0, 0, 2:87, 5], atol=1e-5)
    t2 = torch.zeros(1, 2, H, H)
    t2[:, 1] = 1.0
    assert torch.allclose(J.shift_along(f, t2, 4.0)[0, 0, :, 5:90], f[0, 0, :, 5:90], atol=1e-5)
    fl = (2.0 * torch.arange(H).float().view(1, 1, H, 1) + 3.0 * torch.arange(H).float().view(1, 1, 1, H)).expand(1, 1, H, H).contiguous()
    t3 = torch.zeros(1, 2, H, H)
    t3[:, 0], t3[:, 1] = 0.6, 0.8
    assert torch.allclose(J.shift_along(fl, t3, 5.0)[0, 0, 10:80, 10:80], fl[0, 0, 10:80, 10:80] + 5.0 * (2 * 0.6 + 3 * 0.8), atol=1e-4)
    fr = torch.randn(1, 3, H, H)
    tr = F.normalize(torch.randn(1, 2, H, H), dim=1)
    assert torch.equal(J.shift_along(fr, tr, -4.0), J.shift_along(fr, -tr, 4.0))
    tg = tr.clone().requires_grad_(True)
    J.shift_along(fr, tg, 4.0).square().sum().backward()
    assert float(tg.grad.abs().sum()) > 0
    print("shift_along: rows = x, negative step = behind, exact on a linear field, mirror symmetry, gradient into the heading: ok")
    # the cost
    from torch.utils.flop_counter import FlopCounterMode
    P1 = mk_P(B=1)
    dx = J.DirX().eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        dx(torch.randn(1, 64, 100, 100), P1)
    print("DirX: %d parameters, %.3f GMAC in the 1 x 1 convolutions (batch 1, 100 x 100) next to %.3f GMAC of the head without it" % (sum(p.numel() for p in dx.parameters()), fc.get_total_flops() / 2e9, _gmac(build("dil"), P1)))
    print("test_jmaps_dirx ok")


def _gmac(m, P):
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m.eval()(P)
    return fc.get_total_flops() / 2e9


if __name__ == "__main__":
    main()
