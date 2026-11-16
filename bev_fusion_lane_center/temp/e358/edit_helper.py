import sys
def apply(path, pairs):
    s = open(path, encoding="utf-8", newline="").read()
    for old, new in pairs:
        assert s.count(old) == 1, (s.count(old), old[:120])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8", newline="").write(s)
