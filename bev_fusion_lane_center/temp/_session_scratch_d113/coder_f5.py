p = "evaluation/jpast_eval.py"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)
rep('''    sibling(runs, a.run, a.mode)
    meta = dict(run=name,''', '''    meta = dict(run=name,''')
rep('''    assert not os.path.exists(path), f"a stored dump is never overwritten: {path}"
    donor = states''', '''    assert not os.path.exists(path), f"a stored dump is never overwritten: {path}"
    sibling(runs, a.run, a.mode)
    donor = states''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
