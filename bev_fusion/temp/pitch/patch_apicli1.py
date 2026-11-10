def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/apicli.py", [
    ('''LEGACY = ("list", "quant", "qual", "selftest", "selftest-adapter")''',
     '''LEGACY = ("list", "quant", "qual", "selftest", "selftest-adapter")
POSITIONAL = ("network", "id", "a", "b", "frame", "frames")                      # the things a command is about: positional even when they have a default (`run NETWORK SET`)'''),
    ('''        if p.default is inspect.Parameter.empty:
            ap.add_argument(k)
        elif isinstance(p.default, bool):''', '''        if p.default is inspect.Parameter.empty:
            ap.add_argument(k)
        elif k in POSITIONAL:
            ap.add_argument(k, nargs="?", default=p.default, type=type(p.default) if isinstance(p.default, int) else str)
        elif isinstance(p.default, bool):'''),
    ('''A parameter without a default is positional, every other one an option''', '''A parameter without a default (and `network`, `id`, `frame`, `frames`) is positional, every other one an option'''),
])
print("ok")
