def rd(p):
    return open(p, encoding="utf-8").read()


def sub(s, a, b):
    assert s.count(a) == 1, a[:80]
    return s.replace(a, b)


p = "training/grid/jmaps.py"
s = rd(p)
s = sub(s, '''    assert float(g[30, 50]) > 0 and float(g[10, 10]) == 0.0 and float(g[30, 51]) == 0.0, "the heading term must act at the teacher's peak only"
''', '''    assert float(g[30, 50]) > 0 and float(g[10, 10]) == 0.0 and float(g[30, 51]) == 0.0, "the heading term must act at the teacher's peak only"
    for hx, hy, want in ((0.0, 1.0, 0.0), (0.0, -1.0, 2.0), (1.0, 0.0, 1.0)):                               # the cosine loss at the peak: the same heading 0, the opposite one 2, a right angle 1
        sj = torch.full((1, 8, 100, 100), -7.0)
        sj[0, 2, 30, 50], sj[0, 4], sj[0, 5] = 4.0, hx, hy
        assert abs(distill_loss(dict(j1=sj), dict(j1=tj))[1]["dir"] - want) < 1e-4, (hx, hy, want)
''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
