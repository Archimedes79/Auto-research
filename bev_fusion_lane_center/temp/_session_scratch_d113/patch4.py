import io
p = "training/data/lane_tokcache.py"
s = io.open(p, encoding="utf-8").read()
a = '''    out = os.path.join(CACHE, f"{name}_K{a.K}", split)'''
b = '''    out = os.path.join(CACHE, f"{name}_K{a.K}" + (f"_first{a.limit}" if a.limit else ""), split)                  # a limited build never shares a directory with the full one'''
assert s.count(a) == 1
io.open(p, "w", encoding="utf-8").write(s.replace(a, b))
