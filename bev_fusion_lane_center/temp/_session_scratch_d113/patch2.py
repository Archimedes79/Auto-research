import io
p = "training/data/lane_tokcache.py"
s = io.open(p, encoding="utf-8").read()
a = '''assert torch.equal(t["flags_p"][b, k], no["flags"][b, :, i, j].sigmoid()) and torch.equal(t["s_p"][b, k], no["s"][b, i, j].sigmoid()) and torch.equal(t["e_p"][b, k], e[b, i, j].sigmoid())'''
b = '''assert torch.equal(t["flags_p"][b, k], fp[b, :, i, j]) and torch.equal(t["s_p"][b, k], sp[b, i, j]) and torch.equal(t["e_p"][b, k], ep_[b, i, j])             # the whole-map sigmoid, as predict makes it (a scalar sigmoid can differ by an ulp)'''
assert s.count(a) == 1
s = s.replace(a, b)
a2 = '''    for b in range(B):                                                                                       # against a plain loop'''
b2 = '''    fp, sp, ep_ = no["flags"].sigmoid(), no["s"].sigmoid(), e.sigmoid()
    for b in range(B):                                                                                       # against a plain loop'''
assert s.count(a2) == 1
s = s.replace(a2, b2)
io.open(p, "w", encoding="utf-8").write(s)
