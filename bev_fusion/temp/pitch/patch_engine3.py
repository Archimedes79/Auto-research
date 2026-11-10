def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/quantitative.py", [
    ('''            if ms and len(store.frames) > n_before:           # every stored frame, also those of earlier sessions of a resumed run''',
     '''            if ms and len(store.frames) > n_before and results is None:       # every stored frame, also those of earlier sessions of a resumed run (a stored result set has the load of its own day)'''),
])
print("ok")
