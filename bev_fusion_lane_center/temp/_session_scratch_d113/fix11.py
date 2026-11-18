import io
p = "training/data/lane_tokcache.py"
s = io.open(p, encoding="utf-8").read()
a = "    assert not compare_dump(extra, dump_of(extra))[0]\n"
b = "    assert not compare_dump(extra, dump_of(extra))[0]\n    assert compare_dump(extra, dump_of(extra), b_cap=3)[0], \"a full active set is cut by B itself: tokens behind it are expected\"\n"
assert s.count(a) == 1
io.open(p, "w", encoding="utf-8").write(s.replace(a, b))
