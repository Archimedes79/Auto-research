import pickle, numpy as np
d = pickle.load(open(__import__("os").environ["BFLC_ROOT"] + "/temp/e276/predfull_e273_p_s0.pkl", "rb"))
print(len(d), type(d[0]), sorted(d[0].keys()))
for k in ("pos", "e", "flags", "section"):
    v = np.asarray(d[0][k]); print(k, v.shape, v.dtype, float(v.min()), float(v.max()))
