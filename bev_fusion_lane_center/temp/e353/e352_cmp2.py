import os, sys, pickle
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
ref = pickle.load(open(os.path.join(ROOT, "temp/e276/predfull_e273_p_s0.pkl"), "rb"))[:1]
new = pickle.load(open(os.path.join(ROOT, "temp/e352/check_s0/predfull_null_s0.pkl"), "rb"))[:1]
a, b = ref[0], new[0]
d = np.abs(a["pos"] - b["pos"]).max(1)
bad = np.flatnonzero(d > 0.01)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
for j in bad[:14]:
    print(j, "ref", a["pos"][j], "new", b["pos"][j], "e", round(a["e"][j], 3), round(b["e"][j], 3), "dirv", a["dirv"][j], b["dirv"][j], "flags0", round(a["flags"][j][0], 2), round(b["flags"][j][0], 2))
print("all bad idx", a["idx"][bad])
print("|pos| of bad nodes in ref: min", np.abs(a["pos"][bad]).max(1).min(), " in new: min", np.abs(b["pos"][bad]).max(1).min())
print("e values of bad nodes ref:", np.round(a["e"][bad][:20], 3))
print("sorted e (ref, last 5):", np.sort(a["e"])[:5], " (new):", np.sort(b["e"])[:5])
