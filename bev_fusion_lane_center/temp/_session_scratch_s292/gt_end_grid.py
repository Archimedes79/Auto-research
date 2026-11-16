"""scratch: do the GT entry / exit ends of a frame fit one per cell of a 2 m (or 1 m) grid? nearest-neighbour distances and cell collisions over the 873 VAL frames."""
import os, sys, numpy as np
ROOT = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import lane_gt_data as LD
LD.QData.DEFAULT_JUNCTION = True
ds = LD.QData("val_clean", map_mode="error", pixels=False)
print("frames", len(ds))
res = {k: dict(n=0, nn=[], coll2=0, coll1=0, frames=0) for k in ("ent", "ext")}
nfr_j = 0
for i in range(len(ds)):
    it = ds[i]
    if "ent" not in it:
        continue
    for k in ("ent", "ext"):
        E = np.asarray(it[k], float).reshape(-1, 4)[:, :2]
        if len(E) < 1:
            continue
        r = res[k]; r["n"] += len(E); r["frames"] += 1
        for g, key in ((2.0, "coll2"), (1.0, "coll1")):
            c = np.floor((E + 50.0) / g).astype(int)
            _, cnt = np.unique(c[:, 0] * 1000 + c[:, 1], return_counts=True)
            r[key] += int((cnt[cnt > 1] - 1).sum())
        if len(E) > 1:
            D = np.linalg.norm(E[:, None] - E[None], axis=-1) + np.eye(len(E)) * 1e9
            r["nn"].extend(D.min(1).tolist())
for k, r in res.items():
    nn = np.array(r["nn"])
    print(f"{k}: {r['n']} ends in {r['frames']} frames ({r['n'] / max(r['frames'], 1):.1f} per frame); ends that SHARE a cell with another end: 2 m grid {r['coll2']} ({100 * r['coll2'] / r['n']:.1f} %), 1 m grid {r['coll1']} ({100 * r['coll1'] / r['n']:.1f} %); nearest other end of the same kind: median {np.median(nn):.2f} m, share < 1 m {100 * (nn < 1).mean():.1f} %, < 2 m {100 * (nn < 2).mean():.1f} %, < 3.5 m {100 * (nn < 3.5).mean():.1f} %")
