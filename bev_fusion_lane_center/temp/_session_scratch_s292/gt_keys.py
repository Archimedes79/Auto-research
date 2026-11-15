import os, sys, numpy as np
ROOT = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import lane_gt_data as LD
LD.QData.DEFAULT_JUNCTION = True
ds = LD.QData("val_clean", map_mode="error", pixels=False)
for i in (0, 100, 400):
    it = ds[i]
    print(i, {k: (np.asarray(v).shape if hasattr(v, "__len__") else v) for k, v in it.items() if k in ("ent", "ext", "ent_n", "n_ent", "pairs", "jg") or "ent" in k or "ext" in k})
