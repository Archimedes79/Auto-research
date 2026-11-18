import io
p = "training/data/lane_tokcache.py"
s = io.open(p, encoding="utf-8").read()
a = '''        if n < cache.K and nc == n and (f["e_p"][i, n:] >= e_min).any():'''
b = '''        if n < cache.K and nc == n and n < b_cap and (f["e_p"][i, n:] >= e_min).any():                            # (a full active set is cut by B itself: tokens at or above e_min behind it are expected)'''
assert s.count(a) == 1
io.open(p, "w", encoding="utf-8").write(s.replace(a, b))
