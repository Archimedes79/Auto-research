"""E339 second critic pass: the normaliser = the number of peak cells (the same mask as the heading), a selftest that drives the exit side, pins the 0.3 boundary and the normaliser by a hand-computed value."""


def rd(p):
    return open(p, encoding="utf-8").read()


def sub(s, a, b):
    assert s.count(a) == 1, a[:80]
    return s.replace(a, b)


p = "training/grid/jmaps.py"
s = rd(p)
s = sub(s, """summed over the cells and divided by the teacher's probability mass (the number of positives, as CenterNet / FCOS normalise: a mean over the 10 000 cells would
    dilute the gradient of the ends to nothing; a softmax over the cells, CWD, would give a weak end next to a strong one a fraction of a percent of the mass), + the cosine""",
        """summed over the cells and divided by the number of peak cells of the teacher (probability > peak, the mask of the heading term: the number of positives, as CenterNet / FCOS
    normalise; a mean over the 10 000 cells would dilute the gradient of the ends to nothing, the probability mass would make the scale depend on the teacher's background calibration; a softmax over the cells, CWD, would give a weak end next to a strong
    one a fraction of a percent of the mass), + the cosine""")
s = sub(s, """    l_h = F.binary_cross_entropy_with_logits(s[:, 2:4], pt, reduction="sum") / pt.sum().clamp_min(1.0)
    l_d = 0.0
    for k in range(2):
        ps, th = s[:, 4 + 2 * k: 6 + 2 * k], t[:, 4 + 2 * k: 6 + 2 * k]
        ps = ps / ps.norm(dim=1, keepdim=True).clamp_min(1e-4)
        th = th / th.norm(dim=1, keepdim=True).clamp_min(1e-4)
        m = (pt[:, k] > peak).float()
""", """    pk = (pt > peak).float()
    l_h = F.binary_cross_entropy_with_logits(s[:, 2:4], pt, reduction="sum") / pk.sum().clamp_min(1.0)
    l_d = 0.0
    for k in range(2):
        ps, th = s[:, 4 + 2 * k: 6 + 2 * k], t[:, 4 + 2 * k: 6 + 2 * k]
        ps = ps / ps.norm(dim=1, keepdim=True).clamp_min(1e-4)
        th = th / th.norm(dim=1, keepdim=True).clamp_min(1e-4)
        m = pk[:, k]
""")
i0 = s.index("    for hx, hy, want in ((0.0, 1.0, 0.0)")
i1 = s.index('    print("jmaps selftest ok; params"')
s = s[:i0] + """    # one fixed pair, both kinds, pins the exit side, the 0.3 boundary and the normaliser: the teacher has an entry peak (p 0.98) at (30, 50), an entry cell at p 0.4 at (60, 20), an entry cell at p 0.2 at (70, 70) (below the mask) and an exit peak at (40, 40);
    # teacher headings: entry (0, 1), exit (1, 0). Student = teacher except: heat +0.5 at the entry peak, +1.0 at the exit peak; entry heading (0, -1) (opposite) everywhere but (0, 1) (the same) at the 0.4 cell; exit heading (1, 0) (the same)
    # -> heading loss = (2 + 0) / 2 for the entry kind + 0 for the exit kind = 1; heat loss = the BCE of every cell (written out with logsigmoid) / 3 peak cells
    t2 = torch.full((1, 8, 100, 100), -7.0)
    t2[0, 2, 30, 50], t2[0, 2, 60, 20], t2[0, 2, 70, 70], t2[0, 3, 40, 40] = 4.0, -0.405465, -1.386294, 4.0
    t2[0, 4], t2[0, 5], t2[0, 6], t2[0, 7] = 0.0, 1.0, 1.0, 0.0
    s2 = t2.clone()
    s2[0, 2, 30, 50] += 0.5
    s2[0, 3, 40, 40] += 1.0
    s2[0, 4], s2[0, 5], s2[0, 6], s2[0, 7] = 0.0, -1.0, 1.0, 0.0
    s2[0, 4, 60, 20], s2[0, 5, 60, 20] = 0.0, 1.0
    o2 = distill_loss(dict(j1=s2), dict(j1=t2))[1]
    p_, z_ = torch.sigmoid(t2[:, 2:4]).double(), s2[:, 2:4].double()
    want_bce = float((-(p_ * F.logsigmoid(z_)) - (1 - p_) * F.logsigmoid(-z_)).sum()) / 3.0
    assert abs(o2["bce"] - want_bce) < 1e-3 * want_bce and abs(o2["dir"] - 1.0) < 1e-4, (o2, want_bce)
""" + s[i1:]
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
