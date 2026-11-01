import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "evaluation"))
import evaluate as EV
reg = EV.registry()
raw, gts, p = EV.builder_raw(reg["e24_j1rc_s0"], "val", require_cache=True)
r = raw[0]
for k, v in r.items():
    if isinstance(v, np.ndarray):
        print(k, v.shape, v.dtype, float(np.nanmin(v)) if v.size else "", float(np.nanmax(v)) if v.size else "")
    else:
        print(k, type(v).__name__, len(v) if hasattr(v, "__len__") else v, [ (type(x).__name__, getattr(x, "shape", None)) for x in (v[:3] if isinstance(v, list) else [])])
print(p)
