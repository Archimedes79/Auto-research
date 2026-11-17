"""scratch: how regular are the GT ends of one arm? along-heading spread (stop line alignment) and lateral gaps between neighbouring ends of a group."""
import os, sys, numpy as np
ROOT = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import lane_gt_data as LD
import jgt as JG
LD.QData.DEFAULT_JUNCTION = True
ds = LD.QData("val_clean", map_mode="error", pixels=False)
along, gaps, gsz = [], [], []
for i in range(len(ds)):
    it = ds[i]
    for k in ("ent", "ext"):
        E = np.asarray(it[k], float).reshape(-1, 4)
        if len(E) < 2:
            continue
        g = JG._groups(E)
        for gi in set(g.tolist()):
            idx = np.where(g == gi)[0]
            if len(idx) < 2:
                continue
            t = E[idx, 2:].mean(0); t /= np.linalg.norm(t) + 1e-9
            n = np.array([-t[1], t[0]])
            c = E[idx, :2] - E[idx, :2].mean(0)
            along.append(float(np.std(c @ t))); gsz.append(len(idx))
            lat = np.sort(c @ n)
            gaps.extend(np.diff(lat).tolist())
along, gaps = np.array(along), np.array(gaps)
print(f"{len(along)} groups of >= 2 ends (mean size {np.mean(gsz):.1f}); along-heading spread (std of the position along the arm's heading): median {np.median(along):.2f} m, 90th percentile {np.percentile(along, 90):.2f} m")
print(f"lateral gap between neighbouring ends of a group: median {np.median(gaps):.2f} m, 10th - 90th percentile {np.percentile(gaps, 10):.2f} - {np.percentile(gaps, 90):.2f} m")
