import os
for name, old in (("land_patch.py", 'with open(os.path.join(root, "temp", "landing", "changed.txt"), "w", encoding="utf-8") as fh:'), ("perf_patch.py", 'with open(os.path.join(root, "temp", "landing", "perf_changed.txt"), "w", encoding="utf-8") as fh:')):
    p = os.path.join(os.environ["BFLC_ROOT"], "temp", "landing", name)
    s = open(p, encoding="utf-8").read()
    assert s.count(old) == 1, name
    s = s.replace(old, 'os.makedirs(os.path.join(root, "temp", "landing"), exist_ok=True)\n    ' + old)
    open(p, "w", encoding="utf-8").write(s)
print("ok")
