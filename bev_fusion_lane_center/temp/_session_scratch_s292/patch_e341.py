"""E341: the pair head's evidence along the curve without the 0.5 m paint features (4 of the 6 inputs: lane heat of the pyramid, zone and core of the 1 m branch), for a maps head without the 0.5 m branch."""


def rd(p):
    return open(p, encoding="utf-8").read()


def sub(s, a, b):
    assert s.count(a) == 1, a[:90]
    return s.replace(a, b)


p = "training/grid/jends.py"
s = rd(p)
s = sub(s, """    def __init__(self, d=64, hid=64, k=K, order=True, evidence=True, f1=64, scales=False, ntok=0, unc=False, attach=False):
        super().__init__()
        self.k, self.order, self.evidence, self.scales, self.ntok, self.unc = k, order, evidence, scales, ntok, unc
""", """    def __init__(self, d=64, hid=64, k=K, order=True, evidence=True, f1=64, scales=False, ntok=0, unc=False, attach=False, paint_ev=True):
        super().__init__()
        self.k, self.order, self.evidence, self.scales, self.ntok, self.unc, self.paint_ev = k, order, evidence, scales, ntok, unc, paint_ev   # E341: paint_ev = False: no 0.5 m paint maps (stop line, crossing) among the evidence, 4 of the 6 inputs
""")
s = sub(s, "nin = 2 * d + NG + (NEV if evidence else 0) +", "nin = 2 * d + NG + ((NEV if paint_ev else NEV - 2) if evidence else 0) +")
s = sub(s, """            pnt = _sample(jout["j05"].float().sigmoid(), pts)
            ev = torch.stack([lane.mean(-1), lane.min(-1)[0], zc[..., 0].mean(-1), zc[..., 1].mean(-1), pnt[..., 0].max(-1)[0], pnt[..., 1].max(-1)[0]], -1)
            feats.append(ev)
""", """            ev = [lane.mean(-1), lane.min(-1)[0], zc[..., 0].mean(-1), zc[..., 1].mean(-1)]
            if self.paint_ev:                                                                                # E341: the two paint maxima need the 0.5 m branch
                pnt = _sample(jout["j05"].float().sigmoid(), pts)
                ev += [pnt[..., 0].max(-1)[0], pnt[..., 1].max(-1)[0]]
            feats.append(torch.stack(ev, -1))
""")
# selftest: the evidence without the paint maps
s = sub(s, """    for order, evidence in ((False, False), (True, False), (True, True)):
        m = JEnds(d=d, order=order, evidence=evidence)
""", """    seen = {}
    for order, evidence, paint_ev in ((False, False, True), (True, False, True), (True, True, True), (True, True, False)):                                    # E341: the last case has no 0.5 m branch
        m = JEnds(d=d, order=order, evidence=evidence, paint_ev=paint_ev)
        jo = jout if paint_ev else {k_: v_ for k_, v_ in jout.items() if k_ != "j05"}
        if evidence:
            torch.manual_seed(0)
            m.pair.register_forward_pre_hook(lambda _m, a_, key=paint_ev: seen.__setitem__(key, a_[0].detach().clone()))
""")
s = sub(s, """        out = m(jout, P, gt=gt)
        assert out["logit"].shape == (B, K, K) and out["kp_pos"].shape == (B, 2 * K, 2), out["logit"].shape
        loss, info = jends_loss(out, batch)
        loss.backward()
        dead = [n for n, p in m.named_parameters() if p.grad is None]
        assert not dead, (order, evidence, dead)
""", """        out = m(jo, P, gt=gt)
        assert out["logit"].shape == (B, K, K) and out["kp_pos"].shape == (B, 2 * K, 2), out["logit"].shape
        loss, info = jends_loss(out, batch)
        loss.backward()
        dead = [n for n, p in m.named_parameters() if p.grad is None]
        assert not dead, (order, evidence, paint_ev, dead)
""")
s = sub(s, "        assert info[\"pos_pairs\"] >= 2 * B, info\n        if order and evidence:", "        assert info[\"pos_pairs\"] >= 2 * B, info\n        if order and evidence and paint_ev:")
open(p, "w", encoding="utf-8", newline="\n").write(s)

p = "training/train/jends_train.py"
s = rd(p)
s = sub(s, "unc=unc, attach=attach)\n        self.nodes = QN.NodeHead", "unc=unc, attach=attach, paint_ev=\"nopaint\" not in maps_args.get(\"ctx\", \"dil\"))   # E341: a maps head without the 0.5 m branch gives the evidence without the two paint maxima\n        self.nodes = QN.NodeHead")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
