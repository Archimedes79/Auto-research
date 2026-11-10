p = "evaluation/evaluate.py"
s = open(p, encoding="utf-8").read()
old1 = '''    try:
        bev_sets.parse(a.split)
    except ValueError as ex:
        return str(ex)
    if not os.path.isfile(os.path.join(a.run, "config.json")):'''
new1 = '''    if not os.path.isfile(os.path.join(a.run, "config.json")):'''
old2 = '''    if a.all and a.split != "val":'''
new2 = '''    try:
        bev_sets.parse(a.split)                                                     # a set that does not exist, or TEST: refused before any step
    except ValueError as ex:
        raise SystemExit(str(ex))
    if a.all and a.split != "val":'''
for o, n in ((old1, new1), (old2, new2)):
    assert s.count(o) == 1, o[:60]
    s = s.replace(o, n)
open(p, "w", encoding="utf-8").write(s)
print("ok")
