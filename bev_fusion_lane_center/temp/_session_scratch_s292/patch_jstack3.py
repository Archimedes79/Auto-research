import io
p = "evaluation/jstack.py"
s = io.open(p, encoding="utf-8").read()
a = "    assert not node_diff(dict(good, pos=t(float(\"nan\"))), good, False)[1]"
assert s.count(a) == 1
s = s.replace(a, "    assert not node_diff(dict(good, e=t(1.0005)), good, False)[1] and node_diff(dict(good, e=t(1.5)), good, False)[0] == dict(e=0.5, pos=0.0), \"a difference of 5e-4 fails, the largest difference is returned\"\n" + a)
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
