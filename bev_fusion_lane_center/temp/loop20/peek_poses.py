import os, numpy as np, json
H = os.path.expanduser("~")
z = np.load(f"{H}/exp_gcar_b123/poses_val.npz", allow_pickle=True)
print({k: (z[k].shape, z[k].dtype) for k in z.files})
for k in z.files:
    if z[k].dtype.kind in "US" or z[k].dtype == object:
        print(k, np.unique(z[k])[:12])
print("scene unique", np.unique(z["scene"])[:30], len(np.unique(z["scene"])))
