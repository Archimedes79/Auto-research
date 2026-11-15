p = "training/grid/jends.py"
s = open(p, encoding="utf-8").read()
a = s.index("@torch.no_grad()\ndef rank_features")
b = s.index("class JEnds")
s = s[:a] + s[b:]
open(p, "w", encoding="utf-8").write(s)
m = "training/grid/jmaps.py"
u = open(m, encoding="utf-8").read()
assert "return dict(j1=j1, j05=j05)" in u
u = u.replace("return dict(j1=j1, j05=j05)", "return dict(j1=j1, j05=j05, f1=x1)", 1)
u = u.replace('        out["j05"]  (B, 2, 200, 200)   0.5 m logits: stop lines, pedestrian crossings', '        out["j05"]  (B, 2, 200, 200)   0.5 m logits: stop lines, pedestrian crossings\n        out["f1"]   (B, d, 100, 100)   the 1 m features of the head (read by `jends.JEnds`)', 1)
open(m, "w", encoding="utf-8").write(u)
print("ok")
