import sys, glob, os, json
sys.path.insert(0, ".")
import numpy as np
import evaluation.bev_gtcheck as G
sig = sys.argv[1]
files = sorted(glob.glob(os.path.join(G.cache_dir(), f"*_{sig}.npz")))
rows = []
for i, f in enumerate(files):
    d = G.load(f)
    for side in ("cur", "prev", "next"):
        g = d.get(side)
        if not g:
            continue
        for k in ("velocity", "flow"):
            v = np.asarray(g[k], np.float32); sp = np.hypot(v[0], v[1]); m = float(sp.max())
            if m > 45:
                rows.append((os.path.basename(f)[:12], side, k, round(m, 1), int((sp > 45).sum())))
print(sig, len(files), "files;", len(rows), "(file, side, layer) entries with a cell > 45 m/s")
for r in sorted(rows, key=lambda r: -r[3])[:15]:
    print(r)
