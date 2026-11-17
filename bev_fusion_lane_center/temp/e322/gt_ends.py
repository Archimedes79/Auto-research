"""E322 scratch: the GT ends of the 873 VAL frames (entry / exit position + heading, mask, true pairs) from the loader of the dump -> temp/e322/gt_ends_val.pkl {row: dict}. CPU."""
import os, pickle, sys
import numpy as np
ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import lane_gt_data as LD
LD.QData.DEFAULT_JUNCTION = True
ds, dl = LD.loader("val_clean", bs=8, workers=0, shuffle=False, map_mode="error", pixels=False)
assert ds.missing == 0
out = {}
for b in dl:
    for i in range(len(b["token"])):
        out[int(b["row"][i])] = dict(ent=b["ent"][i].numpy(), ent_m=b["ent_m"][i].numpy(), ext=b["ext"][i].numpy(), ext_m=b["ext_m"][i].numpy(), pairs=b["pairs"][i].numpy())
pickle.dump(out, open(os.path.join(ROOT, "temp", "e322", "gt_ends_val.pkl"), "wb"))
print(len(out), "frames; ent per frame mean", np.mean([o["ent_m"].sum() for o in out.values()]), "pairs per frame", np.mean([(o["pairs"][:, 0] >= 0).sum() for o in out.values()]))
