import os, sys, pickle, time
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
t0 = time.time()
d = pickle.load(open(os.path.join(ROOT, "temp/e276/predfull_e273_p_s0.pkl"), "rb"))
print(len(d), "frames", time.time() - t0)
x = d[0]
print({k: (type(v).__name__, getattr(v, "shape", None), getattr(v, "dtype", None)) for k, v in x.items()})
print([ (len(f["e"]), len(f["link_p"])) for f in d[:6]])
