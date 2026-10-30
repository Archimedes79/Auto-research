p = "tests/test_flow_explains.py"
s = open(p, encoding="utf-8").read()
def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)
rep("        return torch.load(path)\n", "        return torch.load(path, weights_only=False)        # our own scratch cache\n", 2)
rep('''    s01 = [(-dt, v, w) for dt, v, w in reversed(g["prev"]["meta"]["steps"])]
    s12 = [tuple(s) for s in g["next"]["meta"]["steps"]]''', '''    s01 = [(-float(dt), float(v), float(w)) for dt, v, w in reversed(g["prev"]["meta"]["steps"])]
    s12 = [(float(dt), float(v), float(w)) for dt, v, w in g["next"]["meta"]["steps"]]''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
