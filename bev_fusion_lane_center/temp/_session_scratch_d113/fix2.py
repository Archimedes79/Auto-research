import io
p = "temp/_session_scratch_d113/work/lane_tokcache.py"
s = io.open(p, encoding="utf-8").read()
a = '''        assert not ok and rep["worst"][key] > 0, (key, rep)'''
b = '''        assert not ok and rep["mismatches"] > 0 and (key == "e" or rep["worst"][key] > 0), (key, rep)             # a changed existence is caught by the order check before the value check'''
assert s.count(a) == 1
io.open(p, "w", encoding="utf-8").write(s.replace(a, b))
