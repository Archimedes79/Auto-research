def rd(p):
    return open(p, encoding="utf-8").read()


def sub(s, a, b):
    assert s.count(a) == 1, a[:90]
    return s.replace(a, b)


p = "training/grid/jends.py"
s = rd(p)
s = sub(s, """        m = JEnds(d=d, order=order, evidence=evidence, paint_ev=paint_ev)
        jo = jout if paint_ev else {k_: v_ for k_, v_ in jout.items() if k_ != "j05"}
        if evidence:
            torch.manual_seed(0)
            m.pair.register_forward_pre_hook(""", """        torch.manual_seed(0)                                                                           # the token layers are built before the pair MLP: the same seed gives the same tokens, so the two evidence cases can be compared column by column
        m = JEnds(d=d, order=order, evidence=evidence, paint_ev=paint_ev)
        jo = jout if paint_ev else {k_: v_ for k_, v_ in jout.items() if k_ != "j05"}
        if evidence:
            m.pair.register_forward_pre_hook(""")
# after the loop: find the end of the loop body by the line that follows it
marker = """            _, iz = jends_loss(out, batch, driven_w=2.0)
            assert abs(iz["bce"] - i0["bce"]) < 1e-6, (iz, i0)
"""
assert s.count(marker) == 1
s = s.replace(marker, marker + """    c0 = 2 * d + NG                                                                                    # E341: the input of the pair MLP without the paint maps is the one with them minus the two paint columns (evidence columns 4 and 5), nothing else moves
    assert seen[True].shape[-1] == seen[False].shape[-1] + 2 and torch.equal(torch.cat([seen[True][..., :c0 + 4], seen[True][..., c0 + 6:]], -1), seen[False]), "the evidence without the paint maps must be the same features minus the two paint columns"
    assert not torch.equal(seen[True][..., c0 + 4: c0 + 6], torch.zeros_like(seen[True][..., c0 + 4: c0 + 6])), "the paint columns of the full evidence must carry the 0.5 m maps"
""")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
