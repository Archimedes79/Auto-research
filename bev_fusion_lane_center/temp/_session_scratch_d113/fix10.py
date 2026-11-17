import io
p = "training/data/lane_tokcache.py"
s = io.open(p, encoding="utf-8").read()
a = "    assert not compare_dump(cap, dd4, b_cap=3)[0] or True\n"
b = "    assert not compare_dump(cap, dd4, b_cap=3)[0]\n"
assert s.count(a) == 1
io.open(p, "w", encoding="utf-8").write(s.replace(a, b))
p = "training/grid/test_lane_tokcache.py"
s = io.open(p, encoding="utf-8").read()
a = '''    (TC, "the fixture does not look behind the stored nodes", "if n < cache.K and nc == n and (f[\\"e_p\\"][i, n:] >= e_min).any():", "if False:"),
'''
b = '''    (TC, "the fixture does not look behind the stored nodes", "if n < cache.K and nc == n and n < b_cap and (f[\\"e_p\\"][i, n:] >= e_min).any():", "if False:"),
'''
assert s.count(a) == 1, s.count(a)
s = s.replace(a, b)
a2 = '''    (TC, "the look-behind check ignores the cap of the active set", "and n < b_cap and (f[\\"e_p\\"][i, n:] >= e_min).any():", "and (f[\\"e_p\\"][i, n:] >= e_min).any():"),
'''
assert s.count(a2) == 1
b2 = '''    (TC, "the look-behind check ignores the cap of the active set", "if n < cache.K and nc == n and n < b_cap and", "if n < cache.K and nc == n and"),
'''
s = s.replace(a2, b2)
io.open(p, "w", encoding="utf-8").write(s)
