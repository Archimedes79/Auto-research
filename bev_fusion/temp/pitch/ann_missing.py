import glob, os, sys
sys.path.insert(0, ".")
import evaluation.bev_gtcheck as G
from evaluation.bev_annotation_pts import load
pts = load()
n = 0
for f in sorted(glob.glob(os.path.join(G.cache_dir(), "*_15efbde2d1.npz")))[:200]:
    g = G.load(f).get("cur")
    for b in g["meta"]["boxes"]:
        if pts(b["token"]) is None:
            n += 1
            if n <= 6:
                print(os.path.basename(f)[:12], {k: (v if not hasattr(v, "__len__") or isinstance(v, str) else list(v)[:3]) for k, v in b.items()})
print(n)
