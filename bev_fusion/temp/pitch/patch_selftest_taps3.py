p = "evalsuite/selftest_api.py"
s = open(p, encoding="utf-8").read()
old = '''and back.info == {"predict_ms": 2.5} and np.array_equal(back.taps["t"], np.ones(2)), "keys and types survive"'''
new = '''and back.info == {"predict_ms": 2.5} and back.taps == {} and np.array_equal(rs.load(refs[0], ("t",)).taps["t"], np.ones(2)), "keys and types survive; taps come on request"'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("ok")
