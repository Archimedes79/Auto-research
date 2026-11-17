import io
p = "training/grid/test_lane_tokcache.py"
s = io.open(p, encoding="utf-8").read()
a = '''    (TC, "the gate is read from the wrong pair",'''
b = '''    (TC, "a boundary tie at any existence", "all(ed[c_] == last for c_ in only_d) and all(ec[c_] == last for c_ in only_c)", "True"),
    (TC, "a boundary tie without a full active set", "if n >= b_cap and nc == n and all(", "if all("),
    (TC, "the look-behind check ignores the cap of the active set", "and n < b_cap and (f[\\"e_p\\"][i, n:] >= e_min).any():", "and (f[\\"e_p\\"][i, n:] >= e_min).any():"),
    (TC, "the gate is read from the wrong pair",'''
assert s.count(a) == 1
io.open(p, "w", encoding="utf-8").write(s.replace(a, b))
