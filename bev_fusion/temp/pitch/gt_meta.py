import sys
sys.path.insert(0, ".")
import numpy as np
import evaluation.bev_gtcheck as G
for name, s in G.pick(3, "15efbde2d1", 1):
    for dt, g in s.items():
        m = g["meta"]
        print(name[:8], dt, {k: (np.round(v, 3).tolist() if hasattr(v, "__len__") and not isinstance(v, (str, list, dict)) else (v if isinstance(v, (int, float, str)) else type(v).__name__)) for k, v in m.items() if k != "boxes"})
        break
    print("   steps next:", s[0.5]["meta"].get("steps"), "dt", s[0.5]["meta"].get("dt"))
