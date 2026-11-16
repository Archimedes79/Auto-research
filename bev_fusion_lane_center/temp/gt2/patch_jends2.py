p = "training/grid/jends.py"
s = open(p, encoding="utf-8").read()
old = "    P = dict(centre1=torch.randn(B, 12, 100, 100))\n"
new = "    jout[\"j1\"][:, 4], jout[\"j1\"][:, 6] = 0.0, 0.0                              # heading (0, 1) = north for entries and exits\n    jout[\"j1\"][:, 5], jout[\"j1\"][:, 7] = 5.0, 5.0\n    P = dict(centre1=torch.randn(B, 12, 100, 100))\n"
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
