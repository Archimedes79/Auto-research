import os, pickle, numpy as np
d = pickle.load(open(os.environ["BFLC_ROOT"] + "/temp/e276/predfull_e273_p_s0.pkl", "rb"))
fr = d[0]
idx = np.asarray(fr["idx"]); pos = np.asarray(fr["pos"])
print("idx", idx.dtype, idx.shape, idx.min(), idx.max(), "distinct", len(set(idx.tolist())))
c = -50 + 2 * (np.arange(50) + 0.5)
for name, (iy, ix) in {"ij": (idx // 50, idx % 50), "ji": (idx % 50, idx // 50)}.items():
    for sw in (False, True):
        for fy in (1, -1):
            for fx in (1, -1):
                cy, cx = fy * c[iy], fx * c[ix]
                cc = np.stack([cx, cy], 1) if sw else np.stack([cy, cx], 1)
                print(name, "swap" if sw else "keep", fy, fx, float(np.abs(pos - cc).max()))
print(pos[:4], idx[:4])
