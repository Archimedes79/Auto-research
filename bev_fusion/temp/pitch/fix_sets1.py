p = "evaluation/bev_sets.py"
s = open(p, encoding="utf-8").read()
old = '''    for bad in ("test", "small.test", "large.test", "", "small", "small.", ".val", "Small.val", "small.VAL", "small.val.x", "a/b.val", "..val", "small val", None, 3):
        try:'''
new = '''    bads = ("test", "small.test", "large.test", "", "small", "small.", ".val", "Small.val", "small.VAL", "small.val.x", "a/b.val", "..val", "small val", None, 3)
    for bad in bads:
        try:'''
assert s.count(old) == 1
s = s.replace(old, new)
old2 = "the grammar (TEST and 14 malformed names refused)"
assert s.count(old2) == 1
s = s.replace(old2, "the grammar (TEST and {len(bads) - 1} malformed names refused)")
open(p, "w", encoding="utf-8").write(s)
print("ok")
