p = "tests/test_stream_temporal.py"
s = open(p, encoding="utf-8").read()
a = '''        s[3] = run(r, mk([3]), s[2]["state"], [(torch.full((B,), 1 / 6), VG, WG)])'''
b = '''        dt_o = torch.full((B,), 0.5) - sum((x[0] for x in SEGS[3] + SEGS[4]), torch.zeros(B))        # the step to frame 3: the gap less its own chain
        s[3] = run(r, mk([3]), s[2]["state"], [(dt_o, VG, WG)])'''
assert s.count(a) == 1
s = s.replace(a, b)
open(p, "w", encoding="utf-8", newline="").write(s)
