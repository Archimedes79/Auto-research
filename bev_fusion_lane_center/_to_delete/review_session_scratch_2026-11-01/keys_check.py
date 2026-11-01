"""print the field set, shapes and dtypes of frame 0 of the member caches of the union entries (one file in memory at a time)"""
import gc
import pickle
import evaluate as EV

reg = EV.registry()
names = sorted({m for e in reg.values() if e["kind"] == "union" for m in e["members"]})
for n in names:
    p = next((p for p in EV.cache_paths(reg[n], "val") if __import__("os").path.exists(p)), None)
    raw, gts = pickle.load(open(p, "rb"))
    r = raw[0]
    desc = {k: (getattr(v, "shape", len(v)), str(getattr(v, "dtype", type(v).__name__))) for k, v in sorted(r.items())}
    print(n, len(raw), p.split("/")[-1], desc, "lens_ok", all(len(x["pi"]) == len(x["parent"]) == len(x["pos"]) for x in raw))
    del raw, gts, r
    gc.collect()
