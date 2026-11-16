import pickle, sys
import numpy as np
c = pickle.load(open(sys.argv[1], "rb"))
f = c["head"][100]
for k, v in f.items():
    if isinstance(v, dict):
        print(k, {kk: (np.asarray(vv).tolist() if np.asarray(vv).size < 12 else np.asarray(vv).shape) for kk, vv in v.items()})
    else:
        print(k, np.asarray(v).tolist())
print("scenes", sorted(set(c["scenes"])), len(set(c["scenes"])))
b = c["base"][100]
print("base sec", np.asarray(b["sec"]).tolist())
