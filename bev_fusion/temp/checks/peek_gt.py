import glob, numpy as np, json
f = sorted(glob.glob("data/cache/gt/*_0.5_4_*.npz"))[0]
z = np.load(f, allow_pickle=True)
print(f)
for k in z.files:
    v = z[k]
    print(f"  {k:28s} {str(v.dtype):10s} {v.shape}" if v.dtype != object else f"  {k:28s} object {v.shape}")
