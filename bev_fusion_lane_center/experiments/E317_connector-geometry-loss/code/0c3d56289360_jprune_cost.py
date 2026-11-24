"""Parameters and MACs of the junction system (frozen pyramid, junction maps head, pair head of a `jends_train.JEndModel`) after a SET of removals (E345, D99 step 0): the cost side of `jprune_probe.py`.

    from jprune_cost import Cost, Edit
    c = Cost(model)                                  # the layer table of the model (positions measured by one forward at batch 1 on the CPU) + the tensor wiring
    c.saved([Edit("chan", tap="jm.blocks.1.f.2", idx=(0, 5)), Edit("skip", block="jm.blocks.2")])      # -> dict(macs, params, macs_pyr, macs_maps, macs_pair, macs_mha)
    wsl bash tools/env/_py14.sh jprune_cost.py selftest

An `Edit` is one removal: `chan` (channels `idx` of the tensor `tap` replaced by constants), `cols` (input columns of the pair head's two MLPs), `skip` (a residual block of the maps head or a `SetBlock` taken out: output = input), `head` (one attention head of a `SetBlock`).
Tensor names are module paths relative to the model with `jmodel.` dropped (`pyr.e1`, `jm.blocks.0.f.2`, `je.pair.1`); the block output of a `head.block` is the Sequential's path, its hidden tensor `<path>.2`.
Cost of a set (savings multiply, they do not add): every conv / linear counts positions x kernel x KEPT input channels x KEPT output channels, a channel is kept when no edit of the set drops it; a skipped block counts nothing (also no biases, norms);
GroupNorm / LayerNorm / bias parameters follow the kept channels. MACs follow `jmodel_cost.py` (torch FlopCounterMode): convolutions and linears; `nn.MultiheadAttention` is not counted in eval mode (its native fast path is invisible to the counter),
so `macs` leaves it out and `macs_mha` carries it (projections + the two attention matmuls), `mha=True` adds it. Parts outside the table (`jm.dirx`, `jm.attn`, `je.logsig`, `je.attach`, `nodes`) are never edited and not counted (`uncovered`).
"""
import dataclasses

import torch
import torch.nn as nn


@dataclasses.dataclass(frozen=True)
class Edit:
    kind: str                                          # "chan" | "cols" | "skip" | "head"
    tap: str = ""                                      # chan / cols: the tensor
    idx: tuple = ()                                    # chan / cols: its channels (columns)
    block: str = ""                                    # skip / head: path of the block (`jm.blocks.2`, `je.blocks.0`)
    head: int = -1                                     # head: which attention head
    val: object = dataclasses.field(default=None, compare=False, hash=False, repr=False)   # the replacement (chan / cols: a tensor of len(idx); head: the head output mean); not part of the cost


PREFIX = (("jmodel.pyr.", "pyr."), ("jmodel.jm.", "jm."), ("je.", "je."))
UNCOVERED = ("jmodel.jm.dirx", "jmodel.jm.attn", "jmodel.jm.pos", "je.logsig", "je.attach")


def short(path):
    """module path of the JEndModel -> tensor / unit name (`jmodel.` dropped)"""
    return path.replace("jmodel.", "", 1) if path.startswith("jmodel.") else path


def sub(model, name):
    """the module of a short name (inverse of `short`)"""
    return model.get_submodule(name if name == "je" or name.startswith("je.") else "jmodel." + name)


def covered(path):
    return path.startswith(("jmodel.pyr.", "jmodel.jm.", "je.")) and not path.startswith(UNCOVERED)


def uncovered(model):
    """parameters outside the cost table: (name, count) of the parts the table does not know (never edited)"""
    out = []
    for n, m in model.named_children():
        if n == "nodes":
            out.append((n, sum(p.numel() for p in m.parameters())))
    for pre, mod in (("jmodel.jm.", model.jmodel.jm), ("je.", model.je)):
        for n, m in mod.named_children():
            if (pre + n).startswith(UNCOVERED):
                out.append((pre + n, sum(p.numel() for p in m.parameters())))
    return out


def wiring(model):
    """layer path -> (input tensors, output tensor, alive-block or None) of every conv / linear of the covered parts; the concatenations of the pyramid and the maps head are the lists of inputs"""
    pyr, jm, je = model.jmodel.pyr, model.jmodel.jm, model.je
    W = {}

    def blk(p, ins):                                                         # `head.block`: conv GN ReLU conv GN ReLU
        W[f"{p}.0"] = (ins, f"{p}.2", None)
        W[f"{p}.3"] = ([f"{p}.2"], p, None)

    blk("pyr.stem", ["x"])
    blk("pyr.e1", ["pyr.stem"])
    blk("pyr.e2", ["pyr.e1"])
    blk("pyr.e3", ["pyr.e2"])
    blk("pyr.d2", ["pyr.e3", "pyr.e2"])                                      # up(e3) and e2
    blk("pyr.d1", ["pyr.d2", "pyr.e1"])
    blk("pyr.d0", ["pyr.d1", "pyr.stem"])
    W["pyr.centre1"] = (["pyr.d0"], "pyr.centre1", None)
    W["pyr.road2"] = (["pyr.d1"], "pyr.road2", None)
    W["pyr.s05.0"] = (["x"], "pyr.s05", None)
    blk("pyr.m05", ["pyr.d0", "pyr.s05"])
    W["pyr.lines05"] = (["pyr.m05"], "pyr.lines05", None)
    W["jm.p2.0"] = (["pyr.d1", "pyr.d0"], "jm.p2", None)                     # feat2 = [d1, avg_pool(d0)]
    prev = "jm.p2"
    for i in range(len(jm.blocks)):
        p = f"jm.blocks.{i}"
        W[f"{p}.f.0"] = ([prev], f"{p}.f.2", p)
        W[f"{p}.f.3"] = ([f"{p}.f.2"], p, p)                                 # its output channels are the stream channels of the block output
        prev = p
    blk("jm.m1", ["pyr.d0", prev])                                           # [feat1, up(stream)]
    W["jm.head1"] = (["jm.m1"], "jm.head1", None)
    if getattr(jm, "paint", True):
        blk("jm.m05", ["pyr.m05", "jm.m1"])                                  # [feat05, up(x1)]
        W["jm.head05"] = (["jm.m05"], "jm.head05", None)
    W["je.kp"] = (["je.kp.in"], "je.tok", None)
    for i in range(len(je.blocks)):
        p = f"je.blocks.{i}"
        W[f"{p}.ffn.0"] = (["je.tok"], f"{p}.ffn.1", p)
        W[f"{p}.ffn.2"] = ([f"{p}.ffn.1"], "je.tok", p)
    W["je.pair.0"] = (["je.fcat"], "je.pair.1", None)
    W["je.pair.2"] = (["je.pair.1"], "je.pair.3", None)
    W["je.pair.4"] = (["je.pair.3"], "je.pair.4", None)
    if hasattr(je, "scale"):
        W["je.scale.0"] = (["je.fcat"], "je.scale.1", None)
        W["je.scale.2"] = (["je.scale.1"], "je.scale.2", None)
    return W


def _next_is_gn(model, n):
    parent, _, i = n.rpartition(".")
    seq = sub(model, parent) if i.isdigit() else None
    return isinstance(seq, nn.Sequential) and int(i) + 1 < len(seq) and isinstance(seq[int(i) + 1], nn.GroupNorm)


class Cost:
    def __init__(self, model, x=None):
        model = model.eval()
        dev = next(model.parameters()).device
        W = wiring(model)
        mods = {short(n): m for n, m in model.named_modules() if covered(n) and ".att." not in n and isinstance(m, (nn.Conv2d, nn.Linear))}      # an attention module is counted as a whole (`atts`)
        miss = sorted(set(mods) - set(W))
        assert not miss, f"layers without wiring (the table is incomplete): {miss}"
        pos = {}
        hooks = [m.register_forward_hook(lambda _m, _a, o, n=n: pos.__setitem__(n, int((o.shape[2:] if isinstance(_m, nn.Conv2d) else o.shape[:-1]).numel()))) for n, m in mods.items()]
        with torch.no_grad():
            model(torch.randn(1, model.jmodel.pyr.c_in, 200, 200, device=dev) if x is None else x, gt=None)
        for h in hooks:
            h.remove()
        self.layers, self.C = [], {}
        for n, (ins, out, blk) in W.items():
            if n in mods:
                m = mods[n]
                conv = isinstance(m, nn.Conv2d)
                cin, cout = (m.in_channels, m.out_channels) if conv else (m.in_features, m.out_features)
                self.layers.append(dict(name=n, k2=m.kernel_size[0] * m.kernel_size[1] if conv else 1, pos=pos[n], cin=cin, cout=cout, bias=m.bias is not None, gn=_next_is_gn(model, n), ins=ins, out=out, block=blk, group=n.split(".")[0]))
                self.C[out] = cout
        for L in self.layers:                                                # a tensor that no layer writes (the input, the token input, the pair features) is as wide as its readers say
            for t in L["ins"]:
                if t not in self.C and len(L["ins"]) == 1:
                    self.C[t] = L["cin"]
        for L in self.layers:
            assert sum(self.C[t] for t in L["ins"]) == L["cin"], (L["name"], L["ins"], [self.C[t] for t in L["ins"]], L["cin"])
        self.atts = []
        for i, b in enumerate(model.je.blocks):
            att = b.att
            self.atts.append(dict(block=f"je.blocks.{i}", d=att.embed_dim, H=att.num_heads, hd=att.embed_dim // att.num_heads, n=pos[f"je.blocks.{i}.ffn.0"]))
        self.ln = {a["block"]: 4 * a["d"] for a in self.atts}                # n1, n2: weight + bias of d channels each
        self.all = self.totals()

    def totals(self, edits=()):
        """-> dict(macs, macs_mha, params, macs_pyr, macs_maps, macs_pair) of the model with the edits (macs: conv + linear, `macs_mha`: the attention modules, not in `macs`)"""
        dropped, skip, heads = {}, set(), {}
        for e in edits:
            if e.kind in ("chan", "cols"):
                assert e.tap in self.C and all(0 <= c < self.C[e.tap] for c in e.idx), (e.tap, e.idx)
                dropped.setdefault(e.tap, set()).update(e.idx)
            elif e.kind == "skip":
                skip.add(e.block)
            elif e.kind == "head":
                heads.setdefault(e.block, set()).add(e.head)
        kept = {t: c - len(dropped.get(t, ())) for t, c in self.C.items()}
        out = dict(macs=0.0, macs_mha=0.0, params=0, macs_pyr=0.0, macs_maps=0.0, macs_pair=0.0)
        for L in self.layers:
            if L["block"] in skip:
                continue
            cin, cout = sum(kept[t] for t in L["ins"]), kept[L["out"]]
            m = L["pos"] * L["k2"] * cin * cout
            out["macs"] += m
            out["macs_" + {"pyr": "pyr", "jm": "maps", "je": "pair"}[L["group"]]] += m
            out["params"] += L["k2"] * cin * cout + (cout if L["bias"] else 0) + (2 * cout if L["gn"] else 0)
        for A in self.atts:
            if A["block"] in skip:
                continue
            hk = A["H"] - len(heads.get(A["block"], ()))
            d, hd, n = A["d"], A["hd"], A["n"]
            out["params"] += 3 * hd * hk * d + 3 * hd * hk + d * hd * hk + d + self.ln[A["block"]]
            out["macs_mha"] += n * d * 3 * hd * hk + 2 * hk * n * n * hd + n * hd * hk * d
        return out

    def saved(self, edits, mha=False):
        """what the set of edits saves: dict(macs, params, macs_pyr, macs_maps, macs_pair, macs_mha) (intact minus edited); `mha=True` adds the attention modules to `macs`"""
        t = self.totals(edits)
        s = {k: self.all[k] - t[k] for k in self.all}
        if mha:
            s["macs"] += s["macs_mha"]
        return s


def selftest():
    import jends_train as JT
    from torch.utils.flop_counter import FlopCounterMode
    torch.manual_seed(0)
    m = JT.JEndModel(dict(d=64, ctx="dil"), order=True, evidence=True, scales=True).eval()
    c = Cost(m)
    x = torch.randn(1, 39, 200, 200)
    with torch.no_grad(), FlopCounterMode(mods=m, depth=4, display=False) as fc:
        m(x, gt=None)
    counts = fc.get_flop_counts()
    ref = {k: sum(counts.get(f"JEndModel.{p}", {}).values()) / 2 for k, p in (("macs_pyr", "jmodel.pyr"), ("macs_maps", "jmodel.jm"), ("macs_pair", "je"))}
    for k, v in ref.items():
        assert abs(c.all[k] - v) <= 1e-9 * v, (k, c.all[k], v)                                  # the layer table makes the counter's MACs, part by part
    npar = lambda mod: sum(p.numel() for p in mod.parameters())  # noqa: E731
    assert c.all["params"] == npar(m) - sum(n for _, n in uncovered(m)), (c.all["params"], npar(m))     # and the parameters of the whole model (nothing is left out)
    assert c.saved([])["macs"] == 0 and c.saved([])["params"] == 0
    s = c.saved([Edit("chan", tap="jm.blocks.0.f.2", idx=tuple(range(16)))])                    # 16 of the 64 hidden channels of one block: conv 1 loses 16 outputs, conv 2 16 inputs, 50 x 50 positions
    assert s["macs"] == 2 * 2500 * 9 * 64 * 16 and s["params"] == 2 * 9 * 64 * 16 + 2 * 16, s
    s = c.saved([Edit("skip", block="jm.blocks.1")])
    assert s["macs"] == 2 * 2500 * 9 * 64 * 64 and s["params"] == 2 * 9 * 64 * 64 + 2 * 2 * 64, s
    a, b = Edit("chan", tap="jm.blocks.0.f.2", idx=(0, 1)), Edit("skip", block="jm.blocks.0")
    assert c.saved([a, b])["macs"] == c.saved([b])["macs"], "a skipped block counts its hidden width once"
    print(f"jprune_cost selftest ok: {c.all['macs'] / 1e9:.3f} GMACs (pyramid {c.all['macs_pyr'] / 1e9:.3f}, maps {c.all['macs_maps'] / 1e9:.3f}, pair {c.all['macs_pair'] / 1e9:.4f}), {c.all['params'] / 1e6:.3f} M parameters")


if __name__ == "__main__":
    import os
    import sys
    ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for sub_ in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
        sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
    torch.set_num_threads(2)
    if sys.argv[1:] == ["selftest"]:
        selftest()
    else:
        print(__doc__)
        sys.exit(2)
