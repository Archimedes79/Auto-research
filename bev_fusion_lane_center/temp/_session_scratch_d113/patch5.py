import io
p = "training/data/lane_tokcache.py"
s = io.open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)


rep('''def _path(d, name):''', '''def _abs(p):
    """a path relative to the project root (the wrappers run a script from its own folder)"""
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def _path(d, name):''')
rep('''        print(json.dumps(report(a.cache), indent=1))''', '''        print(json.dumps(report(_abs(a.cache)), indent=1))''')
rep('''        c = TokCache(a.cache)
        ok, rep = compare_dump(c, pickle.load(open(a.dump if os.path.isabs(a.dump) else os.path.join(ROOT, a.dump), "rb")), prefix=a.prefix)''', '''        c = TokCache(_abs(a.cache))
        ok, rep = compare_dump(c, pickle.load(open(_abs(a.dump), "rb")), prefix=a.prefix)''')
rep('''    run = a.run if os.path.isabs(a.run) else os.path.join(ROOT, a.run)''', '''    run = _abs(a.run)''')
io.open(p, "w", encoding="utf-8").write(s)
