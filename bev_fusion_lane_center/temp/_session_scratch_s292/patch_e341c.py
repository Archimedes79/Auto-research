"""E341 critic answers: a JEndModel test on a nopaint ctx, a nozone assert, a zone != core column test, stale texts, the README row, junction_view without j05."""


def rd(p):
    return open(p, encoding="utf-8").read()


def sub(s, a, b):
    assert s.count(a) == 1, a[:90]
    return s.replace(a, b)


def wr(p, s):
    open(p, "w", encoding="utf-8", newline="\n").write(s)


# 1. jends.py selftest: the zone and the core column must not be interchangeable
p = "training/grid/jends.py"
s = rd(p)
s = sub(s, """    # E308: with node tokens the pair logits depend on the tokens (and the kp layer takes them)
""", """    jz = dict(jout)                                                                                    # E341: zone about sigmoid(-1) = 0.27, core about sigmoid(-2) = 0.12 (the two evidence columns 2 and 3 must read their own channel, not each other's)
    jz["j1"] = jout["j1"].clone()
    jz["j1"][:, 0] += 1.0
    torch.manual_seed(0)
    mz = JEnds(d=d, order=True, evidence=True, paint_ev=False)
    mz.pair.register_forward_pre_hook(lambda _m, a_: seen.__setitem__("z", a_[0].detach().clone()))
    mz({k_: v_ for k_, v_ in jz.items() if k_ != "j05"}, P, gt=gt)
    zc_ = seen["z"][..., c0 + 2: c0 + 4]
    assert abs(float(zc_[..., 0].mean()) - 0.269) < 0.03 and abs(float(zc_[..., 1].mean()) - 0.119) < 0.03, (float(zc_[..., 0].mean()), float(zc_[..., 1].mean()))
    # E308: with node tokens the pair logits depend on the tokens (and the kp layer takes them)
""")
wr(p, s)

# 2. jends_train.py: nozone with evidence is refused
p = "training/train/jends_train.py"
s = rd(p)
s = sub(s, """        super().__init__()
        self.e2e = e2e
        self.jmodel = JM.JModel(""", """        super().__init__()
        self.e2e = e2e
        assert not (evidence and "nozone" in maps_args.get("ctx", "dil")), "evidence on a maps head without the zone / core supervision (E332 arm b) would read unsupervised channels: train the pair head with --no-evidence"
        self.jmodel = JM.JModel(""")
wr(p, s)

# 3. test_jmaps_simplify.py: the JEndModel on a nopaint ctx, the gradient into the zone / core channels through the four-input evidence
p = "training/grid/test_jmaps_simplify.py"
s = rd(p)
s = sub(s, """    J2 = J.JModel(ctx="dil+nopaint+nozone")""", """    # E341: the evidence of four inputs (no paint maxima) on a head without the 0.5 m branch: the pair loss reaches the zone / core channels through it, and JEndModel builds that head from the ctx of the maps
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
    J2 = J.JModel(ctx="dil+nopaint+nozone")""")
s = sub(s, """    print("pair head (evidence off) on the reduced head: the pair loss reaches the maps head; evidence on fails loudly: ok")""", """    print("pair head (evidence off) on the reduced head: the pair loss reaches the maps head; JEnds(paint_ev=True) on it fails loudly, JEnds(paint_ev=False) / JEndModel on a nopaint ctx read the four-input evidence (150 inputs), nozone with evidence is refused: ok")""")
s = sub(s, """with the evidence on it fails loudly (it reads the zone, core and paint""", """with the evidence on and the default `paint_ev=True` it fails loudly; `paint_ev=False` (E341, set by `JEndModel` from a `nopaint` ctx) reads the four evidence inputs that need no paint map and the pair loss reaches the zone / core channels through them (it reads the zone, core and paint""")
wr(p, s)

# 4. stale texts
p = "training/grid/jmaps.py"
s = rd(p)
s = sub(s, """The pair head must then run `--no-evidence` (the evidence along the
curve reads the zone, core and paint maps).""", """The pair head then reads the evidence along the curve without its two paint inputs (E341: `JEnds(paint_ev=False)`, set by
`JEndModel` from the ctx) or runs `--no-evidence` (E332); `nozone` needs `--no-evidence` (the evidence reads the zone and core maps).""")
wr(p, s)
p = "training/train/jmaps_train.py"
s = rd(p)
s = sub(s, "The pair head must then run --no-evidence\"", "The pair head then runs with the four-input evidence (E341) or --no-evidence (E332)\"")
wr(p, s)

# 5. README row
p = "training/grid/README.md"
s = rd(p)
s = sub(s, "E318 `unc`: a position scale per keypoint", "E341 `paint_ev=False` (set by `JEndModel` from a `nopaint` maps ctx): the evidence along the curve without the two 0.5 m paint maxima (4 inputs, 150 inputs of the pair MLP); E318 `unc`: a position scale per keypoint")
wr(p, s)

# 6. junction_view: a nopaint run has no j05
p = "tools/viz/junction_view.py"
s = rd(p)
s = sub(s, """    j1, j05 = out["j1"][b].float().sigmoid().numpy(), out["j05"][b].float().sigmoid().numpy()""", """    j1 = out["j1"][b].float().sigmoid().numpy()
    j05 = out["j05"][b].float().sigmoid().numpy() if "j05" in out else None                                  # E341 / E332: a head without the 0.5 m branch has no paint maps""")
s = sub(s, """    axs[1].imshow(np.ma.masked_less(j05[0], 0.5).T, origin="lower", extent=ext, cmap="autumn", vmin=0, vmax=1)""", """    if j05 is not None:
        axs[1].imshow(np.ma.masked_less(j05[0], 0.5).T, origin="lower", extent=ext, cmap="autumn", vmin=0, vmax=1)""")
wr(p, s)
print("ok")
