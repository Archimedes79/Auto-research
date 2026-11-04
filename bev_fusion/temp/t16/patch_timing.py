p = "bev_fusion/temp/t16/time_modules.py"
s = open(p, encoding="utf-8").read()
a = """    dts = (-0.5, 0.0, 0.5)
    with torch.no_grad():
        for _ in range(a.warmup):
"""
b = """    dts = (-0.5, 0.0, 0.5)
    orig_features = model.trunk.features                                       # the trunk is called through .features(), not forward: wrap it

    def timed_features(*args, **kw):
        sync(dev)
        t0 = time.perf_counter()
        out = orig_features(*args, **kw)
        sync(dev)
        acc["trunk"].append((time.perf_counter() - t0) * 1e3)
        return out
    model.trunk.features = timed_features
    with torch.no_grad():
        for _ in range(a.warmup):
"""
assert s.count(a) == 1
s = s.replace(a, b)
a = """    for h in hooks:
        h.remove()
    per = """
b = """    for h in hooks:
        h.remove()
    model.trunk.features = orig_features
    per = """
assert s.count(a) == 1
s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("patched")
