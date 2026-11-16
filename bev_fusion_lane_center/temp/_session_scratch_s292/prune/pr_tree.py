import os, sys, torch
ROOT = os.environ.get("BFLC_ROOT") or os.getcwd()
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import jends_train as JT
run = sys.argv[1]
ck = torch.load(os.path.join(ROOT, "runs", run, "model.pt"), map_location="cpu")
ar = ck["args"]
m = JT.JEndModel(ar["maps_args"], order=not ar.get("no_order", False), evidence=not ar.get("no_evidence", False), scales=bool(ar.get("scales", False)), ntok=bool(ar.get("ntok", False)), unc=bool(ar.get("unc", False)))
print("maps_args", ar["maps_args"])
for n, mod in m.named_modules():
    if len(list(mod.children())) == 0 and sum(p.numel() for p in mod.parameters()) > 0:
        shp = tuple(next(mod.parameters()).shape)
        print(f"{n:60s} {type(mod).__name__:14s} {shp} {sum(p.numel() for p in mod.parameters())}")
