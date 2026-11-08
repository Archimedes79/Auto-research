import glob, os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sig in ("15efbde2d1", "6a7aa7864a", "7fe1149566", "962e788566"):
    files = sorted(glob.glob(os.path.join(ROOT, "data", "cache", "gt", f"*_{sig}.npz")))
    print(sig, len(files))
    for f in files[:1]:
        g = np.load(f, allow_pickle=True)["gt"].item()
        print(sorted(g.keys()))
        c = g["cur"]
        for k in ("centreline", "centreline_dir", "lane_divider", "lane_divider_dist"):
            if k in c:
                a = c[k]
                print(" ", k, a.shape, a.dtype, float(np.asarray(a, np.float32).sum()))
            else:
                print(" ", k, "MISSING")
