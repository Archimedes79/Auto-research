import io
p = "training/data/lane_tokcache.py"
s = io.open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)


rep('''def compare_dump(cache, dump, e_min=0.2):''', '''def compare_dump(cache, dump, e_min=0.2, prefix=False):''')
rep('''    assert len(dump) == len(cache.rows), f"{len(dump)} dump frames, {len(cache.rows)} cache frames"''', '''    if prefix:                                                                                                  # a limited (smoke) cache: its frames are the first ones of the dump
        dump = dump[:len(cache.rows)]
    assert len(dump) == len(cache.rows), f"{len(dump)} dump frames, {len(cache.rows)} cache frames"''')
rep('''    s.add_argument("--dump", required=True)
    s = sub.add_parser("report")''', '''    s.add_argument("--dump", required=True)
    s.add_argument("--prefix", action="store_true", help="the cache is a limited build: compare the first frames of the dump only")
    s = sub.add_parser("report")''')
rep('''compare_dump(c, pickle.load(open(a.dump if os.path.isabs(a.dump) else os.path.join(ROOT, a.dump), "rb")))''', '''compare_dump(c, pickle.load(open(a.dump if os.path.isabs(a.dump) else os.path.join(ROOT, a.dump), "rb")), prefix=a.prefix)''')
io.open(p, "w", encoding="utf-8").write(s)
