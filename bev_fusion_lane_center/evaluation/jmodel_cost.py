"""Parameters and MACs per block of the junction system (E330, D84: simplify): the frozen pyramid, the junction maps head (context blocks, 1 m and 0.5 m output branches, the exchange block), the pair head (keypoint token, set blocks, pair MLP, scale head), on CPU, batch 1, torch FlopCounterMode.

    wsl bash tools/env/_py14.sh jmodel_cost.py [--run e312_p_s0] [--base e273_p_s0] [--out file.json]      # a checkpoint of a pair run (its maps args and head flags) + the parameter table of B from its checkpoint
    wsl bash tools/env/_py14.sh jmodel_cost.py --ctx dil --d 32 [--evidence 0] [--scales 1]                # a configuration with fresh weights (MACs do not depend on the weights)
    wsl bash tools/env/_py14.sh jmodel_cost.py selftest

MACs = FLOPs / 2 of the multiply-add ops the counter knows (convolution, matmul, linear, attention); element-wise ops, norms, the bilinear sampling (`grid_sample`) and the top-K keypoint choice are not counted: they are small next to the convolutions but not zero.
The ms per frame of the same blocks come from `temp/jr/timing_junction.py` on an idle GPU (batch 1, bf16); B's own stage times are in `runs/e273/<run>/timing.json`.
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))


def npar(m):
    return sum(p.numel() for p in m.parameters())


def cost(model, x=None):
    """-> dict block name -> dict(params_m, macs_g): the blocks of a `jends_train.JEndModel` at batch 1 (the whole forward, then every named child by its own counted FLOPs)"""
    if x is None:
        x = torch.randn(1, 39, 200, 200)
    model.eval()
    with torch.no_grad(), FlopCounterMode(mods=model, depth=4, display=False) as fc:
        model(x, gt=None)
    counts = fc.get_flop_counts()                                                       # {module path: {op: flops}}
    out = {}

    def macs(path):
        return sum(counts.get(path, {}).values()) / 2e9

    out["total"] = dict(params_m=npar(model) / 1e6, macs_g=macs("Global"))
    jm = model.jmodel.jm
    parts = {"pyramid (B's, frozen)": ("jmodel.pyr", model.jmodel.pyr), "junction maps head": ("jmodel.jm", jm), "pair head": ("je", model.je)}
    name = type(model).__name__
    for k, (path, mod) in parts.items():
        out[k] = dict(params_m=npar(mod) / 1e6, macs_g=macs(f"{name}.{path}"))
    # inside the maps head: context (p2 + blocks + attention), the 1 m branch (m1 + head1), the 0.5 m branch (m05 + head05, the paint maps), the exchange block
    sub = {"maps: context (p2 + dilated blocks)": [("p2", jm.p2), ("blocks", jm.blocks)], "maps: 1 m branch (m1 + head1)": [("m1", jm.m1), ("head1", jm.head1)],
           "maps: 0.5 m paint branch (m05 + head05)": [("m05", getattr(jm, "m05", None)), ("head05", getattr(jm, "head05", None))], "maps: heading exchange (E319)": [("dirx", getattr(jm, "dirx", None))]}
    for k, items in sub.items():
        pm, mm = 0.0, 0.0
        for n_, mod in items:
            if mod is None:
                continue
            pm += npar(mod) / 1e6
            mm += macs(f"{name}.jmodel.jm.{n_}")
            for i in range(len(mod) if isinstance(mod, torch.nn.ModuleList) else 0):                    # a ModuleList is not called itself: its blocks are
                pass
        out[k] = dict(params_m=pm, macs_g=mm)
    if getattr(jm, "attn", None) is not None:
        out["maps: attention (attn)"] = dict(params_m=npar(jm.attn) / 1e6, macs_g=macs(f"{name}.jmodel.jm.attn"))
    # blocks is a ModuleList: sum the counted blocks.N
    mb = sum(macs(f"{name}.jmodel.jm.blocks.{i}") for i in range(len(jm.blocks)))
    out["maps: context (p2 + dilated blocks)"]["macs_g"] += mb
    je = model.je
    pairs = {"pair: keypoint token + 2 set blocks": [je.kp, je.blocks], "pair: pair MLP": [je.pair], "pair: handle-scale head": [getattr(je, "scale", None)], "pair: position scale (E318)": [getattr(je, "logsig", None)]}
    for k, mods in pairs.items():
        pm = sum(npar(m) / 1e6 for m in mods if m is not None)
        mm = 0.0
        for m in mods:
            if m is None:
                continue
            nm = next((n for n, c in je.named_children() if c is m), None)
            if nm is not None:
                mm += macs(f"{name}.je.{nm}")
                if isinstance(m, torch.nn.ModuleList):
                    mm += sum(macs(f"{name}.je.{nm}.{i}") for i in range(len(m)))
        out[k] = dict(params_m=pm, macs_g=mm)
    return out


def build(a):
    import jends_train as JT
    if a.run:
        ck = torch.load(os.path.join(ROOT, "runs", a.run, "model.pt"), map_location="cpu")
        ar = ck["args"]
        return JT.JEndModel(ar["maps_args"], order=not ar.get("no_order", False), evidence=not ar.get("no_evidence", False), scales=bool(ar.get("scales", False)), capacity=bool(ar.get("capacity", False)),
                            ntok=bool(ar.get("ntok", False)), unc=bool(ar.get("unc", False)))
    return JT.JEndModel(dict(d=a.d, ctx=a.ctx), order=True, evidence=bool(a.evidence), scales=bool(a.scales))


def b_params(run):
    """parameters of B's blocks from its checkpoint (the prefixes of the state dict: pyr, nodes, links, menc, asg ...)"""
    p = os.path.join(ROOT, "runs", run, "model.pt")
    if not os.path.exists(p):
        return None
    sd = torch.load(p, map_location="cpu")["state"]
    res = {}
    for k, v in sd.items():
        res[k.split(".")[0]] = res.get(k.split(".")[0], 0) + v.numel()
    return {k: v / 1e6 for k, v in res.items()}


def table(c):
    w = max(len(k) for k in c)
    return "\n".join(f"{k:{w}s}  {v['params_m']:7.3f} M  {v['macs_g']:7.3f} GMACs" for k, v in c.items())


def selftest():
    import jends_train as JT
    m = JT.JEndModel(dict(d=64, ctx="dil"), order=True, evidence=True, scales=True)
    c = cost(m)
    tot = c["total"]["macs_g"]
    parts = c["pyramid (B's, frozen)"]["macs_g"] + c["junction maps head"]["macs_g"] + c["pair head"]["macs_g"]
    assert abs(tot - parts) / tot < 0.02, (tot, parts)                                                   # the three blocks make the whole forward (up to ops outside every block)
    assert c["maps: 0.5 m paint branch (m05 + head05)"]["macs_g"] > 0.5 * c["junction maps head"]["macs_g"] * 0.3, c
    small = JT.JEndModel(dict(d=32, ctx="dil"), order=True, evidence=True, scales=True)
    assert cost(small)["junction maps head"]["macs_g"] < c["junction maps head"]["macs_g"], "a narrower head must cost less"
    print("jmodel_cost selftest ok:", f"total {tot:.2f} GMACs, maps head {c['junction maps head']['macs_g']:.2f}, paint branch {c['maps: 0.5 m paint branch (m05 + head05)']['macs_g']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="cost", choices=("cost", "selftest"))
    ap.add_argument("--run", default=None)
    ap.add_argument("--base", default="e273_p_s0")
    ap.add_argument("--ctx", default="dil")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--evidence", type=int, default=1)
    ap.add_argument("--scales", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(2)
    if a.cmd == "selftest":
        selftest()
        sys.exit(0)
    res = cost(build(a))
    print(table(res))
    bp = b_params(a.base)
    if bp:
        print("B's parameters by block (", a.base, "):", {k: round(v, 3) for k, v in bp.items()}, "M")
    if a.out:
        json.dump(dict(blocks=res, b_params_m=bp), open(a.out, "w"), indent=1)
