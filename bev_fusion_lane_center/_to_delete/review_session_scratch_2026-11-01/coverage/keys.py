import numpy as np, sys
d = np.load(sys.argv[1], allow_pickle=True)
for k in d.files:
    v = d[k]
    print(k, v.shape, v.dtype, (float(v.mean()) if v.dtype != object and v.size > 1 and v.dtype.kind in "fiub" else ""))
