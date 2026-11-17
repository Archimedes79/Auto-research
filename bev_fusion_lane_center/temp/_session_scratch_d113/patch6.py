import io
p = "training/data/lane_tokcache.py"
s = io.open(p, encoding="utf-8").read()
a = '''        t["sstart"][0] = True
        for l, g in enumerate(longs):'''
b = '''        t["sstart"][:1] = True
        for l, g in enumerate(longs):'''
assert s.count(a) == 1
io.open(p, "w", encoding="utf-8").write(s.replace(a, b))
