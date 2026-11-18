import os, sys, time, json
import numpy as np
torch_threads = 2
import torch
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for s in ("training/data", "training/grid", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, s))
import lane_ptok_data as LP, lane_tokcache as LC, lane_past_data as LPD
d = LP.cache_dir("val", 0, "e292_a")
print("dir", d)
t0 = time.perf_counter(); ds = LP.PTokData("val", 0, "e292_a"); print("open", time.perf_counter() - t0)
# warm
for i in range(5): ds[i]
t0 = time.perf_counter()
n = 60
for i in range(n): o = ds[i * 10]
dt = (time.perf_counter() - t0) / n
print("PTokData item ms", dt * 1000)
sz = sum(v.nbytes for v in o.values() if isinstance(v, np.ndarray))
print("item bytes", sz, {k: (v.dtype, v.shape) for k, v in o.items() if isinstance(v, np.ndarray)})
# collate bs 8
items = [ds[i] for i in range(8)]
t0 = time.perf_counter(); b = LP.collate(items); print("collate bs8 ms", (time.perf_counter() - t0) * 1000, "yscale MB", b["yscale"].numel() * 4 / 1e6)
# pairs: duplicates?
dup = 0; maxp = 0
for i in range(len(ds)):
    yp = np.array(ds.c.f["ypair"][i]); ok = yp[:, 0] >= 0
    k = yp[ok]; maxp = max(maxp, len(k))
    dup += len(k) - len({(int(a), int(b_)) for a, b_ in k})
print("dup pairs", dup, "max pairs", maxp)
# prev idx sanity
ext = LPD.load_ext(d)
prev = ext["prev"]; has = prev >= 0
print("prev == i-1 where has:", bool((prev[has] == (np.arange(len(prev)) - 1)[has]).all()), "starts", int((~has).sum()))
# frames/rows equality across the caches of the same split
import glob
for p in sorted(glob.glob(os.path.join(LC.CACHE, "b*_v1", "val", "frames.json"))):
    fr = json.load(open(p)); print(os.path.relpath(p, LC.CACHE), len(fr["rows"]), hash(tuple(fr["rows"])) , hash(tuple(fr["tokens"])))
