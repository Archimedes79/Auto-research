import numpy as np, sys
g = np.load(sys.argv[1], allow_pickle=True)["gt"].item()
for k, v in g.items():
    if isinstance(v, np.ndarray):
        print(k, v.shape, v.dtype, round(float(v.mean()), 3) if v.dtype.kind in "fiub" else "")
    elif isinstance(v, dict):
        print(k, "dict", list(v.keys())[:14])
    else:
        print(k, type(v).__name__)
