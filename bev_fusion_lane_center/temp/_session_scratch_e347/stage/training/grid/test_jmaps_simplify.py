"""E332 tests (CPU, torch): the junction maps head without its auxiliary maps (`jmaps.JMaps` ctx "dil+nopaint" / "dil+nopaint+nozone", `jmaps_train.py --no-paint --no-zone`; D84: simplify).
 - the flags off are the E307 head: the golden numbers of the code before E319 (`test_jmaps_dirx.GOLDEN`) still hold;
 - `nopaint` is the flag-off head minus the 0.5 m branch: the state of a flag-off head loads into it with only `m05` / `head05` unexpected, and the 1 m outputs and features are then bit-identical; the parameter count drops by exactly those two blocks;
 - the loss: without `j05` the paint term is gone (and only it), with `skip = ("zone",)` the zone / core term is gone (and only it);
 - the gradient: every tensor of a reduced head gets a gradient, the zone / core output channels get none under `nozone`, the heat and heading channels do;
 - the pair head runs on the reduced head with `evidence=False` and the pair loss reaches the maps head (the E305 `--e2e` path); with the evidence on and the default `paint_ev=True` it fails loudly; `paint_ev=False` (E341, set by `JEndModel` from a `nopaint` ctx) reads the four evidence inputs that need no paint map and the pair loss reaches the zone / core channels through them (it reads the zone, core and paint maps);
 - the cost (parameters, MACs, batch 1) of the head variants is printed: what each removal saves.
Run: python test_jmaps_simplify.py   (`_py14.sh grid/test_jmaps_simplify.py`)"""
import os
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jmaps as J  # noqa: E402
import test_jmaps_dirx as TD  # noqa: E402

torch.set_num_threads(2)


def gmac(m, P):
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m.eval()(P)
    return fc.get_total_flops() / 2e9


def main():
    P, batch = TD.mk_P(), TD.mk_batch()
    # 1. the flags off: the golden numbers of the head before E319
    for ctx, (npar, s1, s05, sf) in TD.GOLDEN.items():
        m = TD.build(ctx)
        assert sum(p.numel() for p in m.parameters()) == npar and m.paint and m.zone, ctx
        o = m(P)
        assert "skip" not in o
        for name, ref in (("j1", s1), ("j05", s05), ("f1", sf)):
            assert abs(float(o[name].double().sum()) - ref) <= 1e-4 * abs(ref), (ctx, name)
    print("flags off: the head of E307 (golden numbers): ok")
    # 2. nopaint = the head minus the 0.5 m branch
    full = TD.build("dil")
    npn = TD.build("dil+nopaint")
    n_branch = sum(p.numel() for p in full.m05.parameters()) + sum(p.numel() for p in full.head05.parameters())
    assert sum(p.numel() for p in full.parameters()) - sum(p.numel() for p in npn.parameters()) == n_branch and not hasattr(npn, "m05") and not hasattr(npn, "head05")
    rep = npn.load_state_dict(full.state_dict(), strict=False)
    assert not rep.missing_keys and rep.unexpected_keys and all(k.startswith(("m05.", "head05.")) for k in rep.unexpected_keys), rep
    of, on = full(P), npn(P)
    assert "j05" not in on and torch.equal(of["j1"], on["j1"]) and torch.equal(of["f1"], on["f1"]) and "skip" not in on
    print("nopaint: %d parameters less (m05 + head05), the 1 m outputs and features are bit-identical to the full head with the same weights: ok" % n_branch)
    # 3. the loss terms
    lf, inf = J.jmaps_loss(of, batch)
    lnp, inp = J.jmaps_loss(on, batch)
    assert abs(float(lf) - inf["paint"] - float(lnp)) < 1e-5 and inp["paint"] == 0.0 and abs(inp["zone"] - inf["zone"]) < 1e-6 and abs(inp["heat"] - inf["heat"]) < 1e-6, (inf, inp)
    onz = dict(on, skip=("zone",))
    lnz, inz = J.jmaps_loss(onz, batch)
    assert abs(float(lnp) - inf_zone(inp) - float(lnz)) < 1e-5 and inz["zone"] == 0.0 and abs(inz["heat"] - inp["heat"]) < 1e-6 and abs(inz["dir"] - inp["dir"]) < 1e-6
    print("loss: nopaint drops exactly the paint term, skip=('zone',) exactly the zone / core term: ok")
    # 4. gradients
    for ctx in ("dil+nopaint", "dil+nopaint+nozone"):
        m = TD.build(ctx)
        o = m(P)
        assert ("skip" in o) == ("nozone" in ctx)
        loss, _ = J.jmaps_loss(o, batch)
        loss.backward()
        dead = [n for n, p in m.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
        assert not dead, (ctx, dead)
        gz = float(m.head1.weight.grad[:2].abs().sum()) + float(m.head1.bias.grad[:2].abs().sum())
        gh = float(m.head1.weight.grad[2:].abs().sum())
        assert gh > 0 and ((gz == 0.0) == ("nozone" in ctx)), (ctx, gz, gh)
    print("gradient: no dead tensor in the reduced heads, the zone / core channels get no gradient under nozone, heat and heading do: ok")
    # 5. the pair head on a reduced head (evidence off), the E305 e2e path
    import jends as JE
    m = TD.build("dil+nopaint+nozone")
    o = m(P)
    je = JE.JEnds(d=64, order=True, evidence=False, f1=64)
    out = je(o, P)
    (out["logit"].sum() + out["kp_pos"].sum()).backward()
    assert float(m.head1.weight.grad[2:].abs().sum()) > 0 and float(sum(p.grad.abs().sum() for p in m.blocks.parameters() if p.grad is not None)) > 0
    try:
        JE.JEnds(d=64, order=True, evidence=True, f1=64)(m(P), P)
        raise AssertionError("evidence on a head without the paint maps must fail")
    except KeyError:
        pass
    # E341: the evidence of four inputs (no paint maxima) on a head without the 0.5 m branch: the pair loss reaches the zone / core channels through it, and JEndModel builds that head from the ctx of the maps
    m = TD.build("dil+nopaint")
    out = JE.JEnds(d=64, order=True, evidence=True, f1=64, paint_ev=False)(m(P), P)
    (out["logit"].sum() + out["kp_pos"].sum()).backward()
    assert float(m.head1.weight.grad[:2].abs().sum()) > 0, "the pair loss must reach the zone / core channels through the four-input evidence"
    import jends_train as JT
    for ctx_, nin_ in (("dil", 152), ("dil+nopaint", 150)):
        assert JT.JEndModel(dict(d=64, ctx=ctx_), order=True, evidence=True).je.pair[0].in_features == nin_, ctx_
    try:
        JT.JEndModel(dict(d=64, ctx="dil+nopaint+nozone"), order=True, evidence=True)
        refused = False
    except AssertionError:
        refused = True
    assert refused, "evidence on a nozone head must be refused"
    J2 = J.JModel(ctx="dil+nopaint+nozone")
    assert "j05" not in J2.jm(P) and sum(p.numel() for p in J2.jm.parameters()) < sum(p.numel() for p in J.JModel(ctx="dil").jm.parameters())
    print("pair head (evidence off) on the reduced head: the pair loss reaches the maps head; JEnds(paint_ev=True) on it fails loudly, JEnds(paint_ev=False) / JEndModel on a nopaint ctx read the four-input evidence (150 inputs), nozone with evidence is refused: ok")
    # 6. the cost of the variants
    P1 = TD.mk_P(B=1)
    base = gmac(TD.build("dil"), P1)
    print("cost of the maps head, batch 1 (GMACs / parameters in M): full dil %.3f / %.3f" % (base, 0.444490))
    for name, ctx, d in (("nopaint", "dil+nopaint", 64), ("nopaint + nozone", "dil+nopaint+nozone", 64), ("width 48", "dil", 48), ("width 32", "dil", 32), ("width 32 + nopaint", "dil+nopaint", 32)):
        torch.manual_seed(0)
        m = J.JMaps(c2=144, c1=48, c05=32, d=d, ctx=ctx)
        g = gmac(m, P1)
        print("  %-20s %.3f GMACs (%+.0f %%), %.3f M parameters" % (name, g, 100 * (g / base - 1), sum(p.numel() for p in m.parameters()) / 1e6))
    g_np = gmac(TD.build("dil+nopaint"), P1)
    assert base - g_np > 1.2, (base, g_np)                                                       # the 0.5 m branch is about 1.48 GMACs of the 3.26
    # 7. E347: the head with its output on the 2 m grid (ctx '+r2', `jmaps_train.py --no-paint --r2`): no 1 m block, no 0.5 m branch, 12 channels on 50 x 50 cells
    r2 = TD.build("dil+nopaint+r2")
    o2 = r2(P)
    assert not any(hasattr(r2, n) for n in ("m1", "head1", "m05", "head05")) and r2.dirx is None and r2.r2 and not r2.onehot, "the 2 m head has no 1 m or 0.5 m branch"
    assert o2["j1"].shape == (2, 12, 50, 50) and o2["f1"].shape == (2, 64, 50, 50) and "j05" not in o2 and o2["r2"] is True and "onehot" not in o2 and "skip" not in o2, {k: getattr(v, "shape", v) for k, v in o2.items()}
    n_out = sum(p.numel() for p in npn.m1.parameters()) + sum(p.numel() for p in npn.head1.parameters())
    assert sum(p.numel() for p in r2.parameters()) == sum(p.numel() for p in npn.parameters()) - n_out + 64 * J.NCH2 + J.NCH2, "the 2 m head = the no-paint head minus (m1 + head1) plus its output convolution"
    assert r2.head2.weight.shape == (12, 64, 1, 1) and torch.equal(r2.head2.bias, torch.tensor([-3.0, -3.0, -4.0, -4.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])), "the 8 channels start as the 1 m head, the 4 offset channels at 0"
    assert all(torch.equal(npn.state_dict()[k], v) for k, v in r2.state_dict().items() if k.startswith(("p2.", "blocks."))), "the context blocks are the same tensors (same seed) as in the no-paint head"
    with torch.no_grad():
        c2 = r2.p2(P["feat2"])
        for blk in r2.blocks:
            c2 = blk(c2)
        assert torch.equal(o2["f1"], c2) and torch.equal(o2["j1"], r2.head2(c2)), "f1 = the 2 m features of the context blocks, j1 = the output convolution on them"
    for bad in ("dil+r2", "dil+nopaint+dirx+r2", "dil+nopaint+onehot", "dil+onehot"):
        try:
            J.JMaps(ctx=bad)
            raise RuntimeError(f"{bad} must be refused")
        except AssertionError as ex:
            assert "needs" in str(ex) or "ctx" in str(ex), ex
    o3 = TD.build("dil+nopaint+r2+onehot")(P)
    o4 = TD.build("dil+nopaint+nozone+r2")(P)
    assert o3["onehot"] is True and o3["r2"] is True and o4["skip"] == ("zone",) and o4["j1"].shape == (2, 12, 50, 50)
    r2.train()
    loss, _ = J.jmaps_loss(r2(P), dict(jz=torch.zeros(2, 2, 50, 50), jheat=torch.zeros(2, 2, 50, 50), jdir=torch.zeros(2, 4, 50, 50), jpaint=torch.zeros(2, 2, 200, 200), jend=torch.ones(2, 2, 50, 50), joff=torch.ones(2, 4, 50, 50), joffm=torch.ones(2, 2, 50, 50)))
    loss.backward()
    dead = [n for n, p in r2.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    assert not dead, dead                                                                         # every tensor of the 2 m head gets a gradient, the offset channels through the offset term
    g_r2 = gmac(TD.build("dil+nopaint+r2"), P1)
    assert 0.6 < g_r2 < 0.95 and g_np - g_r2 > 0.9, (g_r2, g_np)                                  # the context blocks at 50 x 50 only: about 0.76 of the 1.78 GMACs of the no-paint head
    print("2 m head (E347): no m1 / m05, 12 channels on 50 x 50, the no-paint head minus (m1 + head1) plus head2, the rules of the ctx flags, no dead tensor; cost %.3f GMACs (the no-paint head %.3f, the full head %.3f)" % (g_r2, g_np, base))
    print("test_jmaps_simplify ok")


def inf_zone(info):
    return info["zone"]


if __name__ == "__main__":
    main()
