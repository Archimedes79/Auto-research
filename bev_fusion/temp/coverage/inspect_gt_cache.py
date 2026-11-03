"""Box fields of one GT cache file and the split file (for the footprint-share study, D38 b). Run from bev_fusion (WSL, gcar python, CPU)."""
import glob
import json

import numpy as np

files = sorted(glob.glob("data/cache/gt/*_0.5_4_15efbde2d1.npz"))
z = np.load(files[100], allow_pickle=True)
meta = z["gt"].item()["cur"]["meta"]
print({k: (str(v)[:80] if not isinstance(v, (list, dict)) else type(v).__name__ + " len " + str(len(v))) for k, v in meta.items()})
for b in meta["boxes"][:3]:
    print({k: (round(v, 3) if isinstance(v, float) else (str(v)[:40])) for k, v in b.items()})
sp = json.load(open("data/splits.json"))
print({k: (len(v) if hasattr(v, "__len__") else v) for k, v in sp.items()})
print("val scenes sample:", list(sp.get("val", []))[:4])
