p = "evaluation/jwire_verdict.py"
s = open(p, encoding="utf-8").read()
old = '''for sp in seeds(dict(good, wa=agree(360))) + seeds(dict(good, wa=agree(float("nan")))) + seeds(dict(good, d=dict(topo_F1=float("nan"))))::'''
old = '''    for sp in seeds(dict(good, wa=agree(360))) + seeds(dict(good, wa=agree(float("nan")))) + seeds(dict(good, d=dict(topo_F1=float("nan")))):'''
new = '''    for sp in seeds(dict(good, wa=agree(360))) + seeds(dict(good, wa=agree(float("nan")))) + seeds(dict(good, d=dict(topo_F1=float("nan")), iv=None)):    # (a card with NaN has no usable bootstrap log: INVALID comes first)'''
assert s.count(old) == 1
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
