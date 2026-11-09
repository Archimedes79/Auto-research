import sys, importlib.util
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion/training")
import torch
import losses as NEW
def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
OLD = load("losses_head", "/mnt/c/Auto-research/bev_fusion/temp/critic_line_bce/losses_head.py")
torch.manual_seed(1)
B, X, Y = 2, 16, 18
def mk(spec, level):
    T = {}
    for n, k, key in spec:
        if k in ("area", "line", "object", "centreline"):
            T[key] = (torch.rand(B, 1, X, Y) > 0.9).float()
        elif k == "dist":
            T[key] = torch.rand(B, 1, X, Y) * 2.5
        elif k == "dir":
            T[key] = torch.randn(B, 2, X, Y)
    return T
ok = True
for cl, c1 in ((False, False), (True, False), (False, True)):
    for tag, spec in (("0.5", NEW.layer_spec(cl, True)), ("1m", NEW.layer_spec_1m(c1))):
        if not spec: continue
        T = mk(spec, tag); lg = torch.randn(B, len(spec), X, Y)
        w = {"area": torch.rand(B, 1, X, Y), "line": torch.rand(B, 1, X, Y), "object": torch.rand(B, 1, X, Y), "stationary": torch.rand(B, 1, X, Y)}
        for kw in ({}, {"line_boundary": True}):
            for sf in (0.0, 0.7):
                a = NEW.map_terms(lg, T, spec, w, NEW.LossConfig(**kw), sf)
                b = OLD.map_terms(lg, T, spec, w, OLD.LossConfig(**kw), sf)
                same = a[0].keys() == b[0].keys() and a[1].keys() == b[1].keys() and all(torch.equal(a[0][k], b[0][k]) for k in a[0]) and all(torch.equal(torch.as_tensor(a[1][k]), torch.as_tensor(b[1][k])) for k in a[1])
                ok &= same
                print(tag, cl, c1, kw, sf, "bit-identical:", same)
print("ALL", ok)
print("new LossConfig fields vs head:", sorted(set(NEW.LossConfig.__dataclass_fields__) - set(OLD.LossConfig.__dataclass_fields__)))
