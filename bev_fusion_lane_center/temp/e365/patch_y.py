p = "evaluation/jtoken_eval.py"
s = open(p, encoding="utf-8").read()
a = '''pick = lambda rec: {k: [v[int(i)] for i in r["idx"]] for k, v in rec.items()}  # noqa: E731   # the records of plain B / B' on the frames of this read-out (the cache order = the VAL order)'''
assert s.count(a) == 1
s = s.replace(a, '''pick = lambda rec: dict(graph=[rec["graph"][int(i)] for i in r["idx"]])  # noqa: E731   # the graph records of plain B / B' on the frames of this read-out (the cache order = the VAL order); the paired interval reads these only''')
open(p, "w", encoding="utf-8").write(s)
