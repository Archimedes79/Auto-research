import sys, glob, os
sys.path.insert(0, ".")
import numpy as np
import evaluation.bev_gtcheck as G
for sig in ("15efbde2d1", "7fe1149566", "02a1217742"):
    files = sorted(glob.glob(os.path.join(G.cache_dir(), f"*_{sig}.npz")))
    rng = np.random.RandomState(0); rng.shuffle(files)
    bad, mx, n, nf = [], 0.0, 0, 0
    for f in files[:600]:
        g = G.load(f).get("cur")
        if not g:
            continue
        n += 1
        for k in ("velocity", "flow"):
            v = np.asarray(g[k], np.float32); sp = np.hypot(v[0], v[1])
            m = float(sp.max()); mx = max(mx, m)
            if m > 60:
                bad.append((os.path.basename(f)[:10], k, round(m, 1), int((sp > 60).sum())))
        bx = [np.hypot(*b["vel"][:2]) for b in g["meta"]["boxes"] if b.get("vel") is not None]
        if bx and max(bx) > 60:
            nf += 1
    print(sig, "frames", n, "max speed", round(mx, 1), "| frames with a cell > 60 m/s:", len(bad), "| frames with a box > 60 m/s:", nf, bad[:3])
