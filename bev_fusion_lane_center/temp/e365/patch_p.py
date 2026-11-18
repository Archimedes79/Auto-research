import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
a = '''    d = g("d")
    return JK.JTokenHead('''
assert s.count(a) == 1
s = s.replace(a, '''    d = g("d")
    assert d % g("heads") == 0, f"--d {d} is not a multiple of --heads {g('heads')}"
    return JK.JTokenHead(''')
open(p, "w", encoding="utf-8").write(s)
