import os

import numpy as np

for part in ("train", "val"):
    p = os.path.expanduser(f"~/exp_gcar_b123/poses_{part}.npz")
    print(part, os.path.exists(p))
    if os.path.exists(p):
        z = np.load(p)
        print({k: (z[k].shape, z[k].dtype) for k in z.files})
        print("t[:2]", z["t"][:2], "scene[:5]", z["scene"][:5] if "scene" in z.files else None)
