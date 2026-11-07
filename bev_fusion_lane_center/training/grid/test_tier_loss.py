"""E276 tests (CPU, torch): the tier / visibility switches of the node and link losses of qgraph. The defaults must be the E273 loss (the new maps in the batch change nothing); each switch must
move the gradient in the intended way. Run: python test_tier_loss.py   (`_py14.sh test_tier_loss.py`)"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qgraph as QG  # noqa: E402

B = 1
CELLS = {"t1": (10, 10), "t2": (10, 12), "t3": (10, 14), "inv": (10, 16)}              # four GT nodes: tier 1, 2, 3, and a tier-1 node no camera sees
NEG = {"z1": (20, 10), "z3": (20, 14)}                                                 # two cells without a node, the first near the own lane (zone 1), the second in zone 3


def make(with_maps=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = 50
    no = {"e": torch.zeros(B, n, n), "flags": torch.zeros(B, 8, n, n), "off": torch.zeros(B, 2, n, n), "dir_raw": torch.randn(B, 2, n, n, generator=g) + 2.0,
          "s": torch.zeros(B, n, n), "uout": torch.zeros(B, n, n), "uin": torch.zeros(B, n, n)}
    for v in no.values():
        v.requires_grad_(True)
    b = {"exists": torch.zeros(B, n, n), "flags": torch.zeros(B, 8, n, n), "offset": torch.zeros(B, 2, n, n), "dirv": torch.zeros(B, 2, n, n), "sstart": torch.zeros(B, n, n),
         "uout": torch.zeros(B, n, n), "uin": torch.zeros(B, n, n), "road2": torch.zeros(B, 2, n, n)}
    b["dirv"][:, 1] = 1.0
    tier, vis, z = torch.zeros(B, n, n), torch.ones(B, n, n), torch.zeros(B, n, n)
    for k, (i, j) in CELLS.items():
        b["exists"][0, i, j] = 1.0
        tier[0, i, j] = {"t1": 1, "t2": 2, "t3": 3, "inv": 1}[k]
    vis[0, CELLS["inv"][0], CELLS["inv"][1]] = 0.0
    for k, (i, j) in NEG.items():
        z[0, i, j] = 1.0 if k == "z1" else 3.0
    if with_maps:
        b.update(tier=tier, vis=vis, ztier=z)
    return no, b


def grads(no, b):
    loss, _ = QG.node_loss(no, b)
    return loss, {k: torch.autograd.grad(loss, v, retain_graph=True, allow_unused=True)[0] for k, v in no.items()}


def reset():
    QG.CFG.update(tier_w=None, vis_w=1.0, neg_w=None, link_tier_w=None, node_w_evt=0.0, node_w_inter=0.0)


def main():
    reset()
    # 1. the defaults are the E273 loss: the new maps in the batch change nothing, bit for bit
    no, b = make(True)
    no2, b2 = make(False)
    l1, g1 = grads(no, b)
    l2, g2 = grads(no2, b2)
    assert float(l1) == float(l2) and all(torch.equal(g1[k], g2[k]) for k in g1), "defaults must reproduce the E273 loss"
    # layer 1 of the card: a FROZEN value, computed with the committed E273 qgraph.py (HEAD of 2026-11-07, equal bit for bit on this batch): a change of the default loss fails here
    assert abs(float(l1) - 60.220611572265625) < 1e-6, f"the default node loss changed: {float(l1)!r}"
    ge = lambda g, key: {k: float(g[key][0, i, j]) for k, (i, j) in {**CELLS, **NEG}.items()}   # noqa: E731
    base = ge(g1, "e")
    # 2. tier_w: the gradient of the existence loss at a GT node scales with the tier weight (same logit, same focal factor)
    QG.CFG.update(tier_w=(3.0, 1.5, 1.0))
    _, g = grads(*make(True))
    w = ge(g, "e")
    assert abs(w["t1"] / base["t1"] - 3.0) < 1e-5 and abs(w["t2"] / base["t2"] - 1.5) < 1e-5 and abs(w["t3"] / base["t3"] - 1.0) < 1e-5, (w, base)
    assert w["z1"] == base["z1"] and w["z3"] == base["z3"], "negatives are untouched by tier_w"
    i, j = CELLS["t1"]
    for key in ("flags", "dir_raw"):                                                                # the other positive terms follow the same factor
        a, c = float(g[key][0, :, i, j].abs().sum()), float(g1[key][0, :, i, j].abs().sum())
        assert c > 0 and abs(a / c - 3.0) < 1e-4, (key, a, c)
    # 3. vis_w = 0: an invisible GT node is not demanded (no gradient in any positive term), the visible ones keep theirs, negatives are untouched
    reset()
    QG.CFG.update(vis_w=0.0)
    _, g = grads(*make(True))
    i, j = CELLS["inv"]
    assert float(g["e"][0, i, j]) == 0.0 and float(g["flags"][0, :, i, j].abs().sum()) == 0.0 and float(g["dir_raw"][0, :, i, j].abs().sum()) == 0.0, "invisible node: no gradient"
    assert ge(g, "e")["t1"] == base["t1"] and ge(g, "e")["z1"] == base["z1"], "visible node and negatives unchanged"
    # 4. neg_w: a false node costs more near the own lane (zone 1) than far from it (zone 3)
    reset()
    QG.CFG.update(neg_w=(2.0, 1.5, 1.0, 1.0))
    _, g = grads(*make(True))
    w = ge(g, "e")
    assert abs(w["z1"] / base["z1"] - 2.0) < 1e-5 and abs(w["z3"] / base["z3"] - 1.0) < 1e-5 and w["t1"] == base["t1"], (w, base)
    # 5. the links: the weight of a candidate pair follows the tier of its source node
    reset()

    def links():
        act = torch.tensor([[10 * 50 + 10, 10 * 50 + 12, 10 * 50 + 14]])
        L = dict(logit=torch.zeros(3, requires_grad=True), label=torch.ones(3), is_long=torch.zeros(3, dtype=torch.bool), valid=torch.ones(1, 3, dtype=torch.bool), pb=torch.zeros(3, dtype=torch.long),
                 pa=torch.arange(3), pc=torch.tensor([1, 2, 0]), act=act, sa=torch.zeros(3), sb=torch.zeros(3))
        bb = {"lk": -torch.ones(1, 1, 4, dtype=torch.long)}
        bb2 = dict(bb)
        t, v = torch.zeros(1, 50, 50), torch.ones(1, 50, 50)
        t[0, 10, 10], t[0, 10, 12], t[0, 10, 14] = 1, 2, 3
        v[0, 10, 14] = 0.0
        bb.update(tier=t, vis=v)
        return L, bb, bb2
    L, bb, bb0 = links()
    l0, _ = QG.link_loss(L, bb0)
    g0 = torch.autograd.grad(l0, L["logit"])[0]
    L, bb, _ = links()
    lbase, _ = QG.link_loss(L, bb)
    assert float(lbase) == float(l0), "defaults: the tier maps change nothing in the link loss"
    QG.CFG.update(link_tier_w=(3.0, 1.5, 1.0))
    L, bb, _ = links()
    lw, _ = QG.link_loss(L, bb)
    gw = torch.autograd.grad(lw, L["logit"])[0]
    assert torch.allclose(gw / g0, torch.tensor([3.0, 1.5, 1.0]), atol=1e-5), (gw, g0)
    QG.CFG.update(vis_w=0.0)
    L, bb, _ = links()
    lv, _ = QG.link_loss(L, bb)
    gv = torch.autograd.grad(lv, L["logit"])[0]
    assert float(gv[2]) == 0.0 and float(gv[0]) != 0.0, "a link from an invisible node is not demanded"
    reset()
    print("test_tier_loss (qgraph): defaults = E273 bit for bit; tier_w x3 / x1.5 / x1 on the positives; vis_w = 0 silences an invisible node; neg_w x2 near the own lane; link weights by the source tier: ok")


if __name__ == "__main__":
    main()
