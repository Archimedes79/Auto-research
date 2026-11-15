import os, pickle, sys
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
PF = pickle.load(open(os.path.join(ROOT, "temp", "e309", "perframe_s0.pkl"), "rb"))
keys = ("B", "e284", "e287p2")
def arr(k, f): return np.array([r[k]["num"][f] if f in r[k]["num"] else np.nan for r in PF])
def lar(k, f): return np.array([r[k]["len"][f] for r in PF])
print("frames", len(PF))
for k in keys:
    cov, prec = lar(k, "cov"), lar(k, "prec")
    print(f"{k:8s} per-frame mean: topo_F1 {np.nanmean(arr(k,'topo_F1')):.3f} topoP {np.nanmean(arr(k,'topo_P')):.3f} topoR {np.nanmean(arr(k,'topo_R')):.3f} | lenient2m cov {np.nanmean(cov):.3f} prec {np.nanmean(prec):.3f} | F1@1 {np.nanmean(arr(k,'F1@1.0')):.3f} lanes_found {np.nanmean(arr(k,'lanes_found')):.3f}")
cov, prec, t = lar("B", "cov"), lar("B", "prec"), arr("B", "topo_F1")
ok = ~np.isnan(cov) & ~np.isnan(prec) & ~np.isnan(t)
print("corr(topo_F1, cov)", np.corrcoef(t[ok], cov[ok])[0,1], " corr(topo_F1, prec)", np.corrcoef(t[ok], prec[ok])[0,1])
good = ok & (cov >= 0.85) & (prec >= 0.85)
print(f"frames that LOOK good by the lenient 2 m measure (cov >= .85 and prec >= .85): {good.sum()} of {ok.sum()}; their topo F1: mean {t[good].mean():.3f}, median {np.median(t[good]):.3f}, share below 0.6: {(t[good] < 0.6).mean():.2f}")
for lo, hi in ((0, .3), (.3, .5), (.5, .7), (.7, .85), (.85, 1.01)):
    m = ok & (t >= lo) & (t < hi)
    print(f"topo F1 in [{lo},{hi}): {m.sum():4d} frames, lenient cov {cov[m].mean():.2f} prec {prec[m].mean():.2f}, GT length {np.mean([PF[i]['B']['len']['n_gt'] for i in np.where(m)[0]]):.0f} m, GT pieces {np.mean([PF[i]['n_gt_lines'] for i in np.where(m)[0]]):.1f}")
# junction frames: many GT pieces
npc = np.array([r["n_gt_lines"] for r in PF])
for lo, hi in ((0, 6), (6, 12), (12, 24), (24, 999)):
    m = ok & (npc >= lo) & (npc < hi)
    print(f"GT pieces in [{lo},{hi}): {m.sum():4d} frames, topo F1 {t[m].mean():.3f} cov {cov[m].mean():.3f} prec {prec[m].mean():.3f}")
