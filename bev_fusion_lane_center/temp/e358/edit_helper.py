import os
import sys


def apply(path, pairs):
    """replace each `old` (exactly once) by `new` in the file; the new content is written to a temp file next to it and moved over it (atomic: a process that starts reading sees the old or the new file)"""
    s = open(path, encoding="utf-8", newline="").read()
    for old, new in pairs:
        assert s.count(old) == 1, (s.count(old), old[:140])
        s = s.replace(old, new)
    tmp = f"{path}.{os.getpid()}.edit"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(s)
    os.replace(tmp, path)
    print("edited", os.path.basename(path), file=sys.stderr)
