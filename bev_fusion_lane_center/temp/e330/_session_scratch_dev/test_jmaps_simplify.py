"""E332 tests (CPU, torch): the junction maps head without its auxiliary maps (`jmaps.JMaps` ctx "dil+nopaint" / "dil+nopaint+nozone", `jmaps_train.py --no-paint --no-zone`; D84: simplify).
 - the flags off are the E307 head: the golden numbers of the code before E319 (`test_jmaps_dirx.GOLDEN`) still hold;
 - `nopaint` is the flag-off head minus the 0.5 m branch: the state of a flag-off head loads into it with only `m05` / `head05` unexpected, and the 1 m outputs and features are then bit-identical; the parameter count drops by exactly those two blocks;
 - the loss: without `j05` the paint term is gone (and only it), with `skip = ("zone",)` the zone / core term is gone (and only it);
 - the gradient: every tensor of a reduced head gets a gradient, the zone / core output channels get none under `nozone`, the heat and heading channels do;
 - the pair head runs on the reduced head with `evidence=False` and the pair loss reaches the maps head (the E305 `--e2e` path); with the evidence on it fails loudly (it reads the zone, core and paint maps);
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
    J2 = J.JModel(ctx="dil+nopaint+nozone")
    assert "j05" not in J2.jm(P) and sum(p.numel() for p in J2.jm.parameters()) < sum(p.numel() for p in J.JModel(ctx="dil").jm.parameters())
    print("pair head (evidence off) on the reduced head: the pair loss reaches the maps head; evidence on fails loudly: ok")
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
    print("test_jmaps_simplify ok")


def inf_zone(info):
    return info["zone"]


if __name__ == "__main__":
    main()
