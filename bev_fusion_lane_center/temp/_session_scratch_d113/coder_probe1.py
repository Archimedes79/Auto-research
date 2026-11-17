import os, sys, json
import numpy as np
R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import p16_data as PD
import lane_tokcache as LC
import lane_seq_data as LS
print("D", PD.D)
for part in ("train", "val"):
    z = dict(np.load(f"{PD.D}/poses_{part}.npz"))
    print(part, {k: (v.shape, v.dtype) for k, v in z.items()})
for split in ("val", "train"):
    d = os.path.join(R, f"runs/e365_cache/be292_a_0_K512_9ad1d4ce30_v1/{split}")
    c = LC.TokCache(d)
    print(split, len(c), c.rows[:5], c.tokens[:2], c.done, c.K)
    pose = dict(np.load(f"{PD.D}/poses_{split}.npz"))
    prev = pose["prev"][np.array(c.rows)]
    print(" prev[:6]", prev[:6], "rows[:6]", c.rows[:6], "sorted rows?", all(np.diff(c.rows) > 0))
    row2i = {int(r): i for i, r in enumerate(c.rows)}
    n_in = sum(int(p) in row2i for p in prev)
    n_neg = int((prev < 0).sum())
    print(" prev in cache", n_in, "prev -1", n_neg, "not in cache and >=0", len(prev) - n_in - n_neg)
    # is prev == row of previous cache index?
    ok = sum(1 for i, p in enumerate(prev) if int(p) >= 0 and int(p) in row2i and row2i[int(p)] == i - 1)
    print(" prev is i-1:", ok)
sp = json.load(open(os.path.join(R, "data", "splits.json")))
print({k: len(v) for k, v in sp.items()})
lane_gt, store = None, None
import lane_gt_data as LD
lg, store = LD.lane_store()
print(store)
g = lg.load(c.tokens[0]) if False else None
c = LC.TokCache(os.path.join(R, "runs/e365_cache/be292_a_0_K512_9ad1d4ce30_v1/val"))
g = lg.load(c.tokens[10])
print(type(g), list(g.keys()) if hasattr(g, "keys") else None)
print(len(g["pieces"]), g["pieces"][0].keys(), np.asarray(g["pieces"][0]["pts"]).shape)
