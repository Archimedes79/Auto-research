import os, sys, pickle
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
ref = pickle.load(open(os.path.join(ROOT, "temp/e276/predfull_e273_p_s0.pkl"), "rb"))[:24]
new = pickle.load(open(os.path.join(ROOT, "temp/e352/check_s0/predfull_null_s0.pkl"), "rb"))
for i in (0, 4, 6, 1, 2):
    a, b = ref[i], new[i]
    print(i, "row", a["row"], b["row"], "nodes", len(a["e"]), len(b["e"]), "links", len(a["link_p"]), len(b["link_p"]), "idx equal", len(a["idx"]) == len(b["idx"]) and bool(np.array_equal(a["idx"], b["idx"])))
    n = min(len(a["e"]), len(b["e"]))
    d = np.abs(a["pos"][:n] - b["pos"][:n]).max(1)
    print("   pos diff: median", float(np.median(d)), "max", float(d.max()), "n>0.01:", int((d > 0.01).sum()), "e diff max", float(np.abs(a["e"][:n] - b["e"][:n]).max()))
    j = int(np.argmax(d))
    print("   worst node", j, "ref pos", a["pos"][j], "e", a["e"][j], "| new pos", b["pos"][j], "e", b["e"][j], "idx", a["idx"][j], b["idx"][j])
    # set comparison by sorted rounded positions
    sa = {tuple(np.round(p, 2)) for p in a["pos"]}; sb = {tuple(np.round(p, 2)) for p in b["pos"]}
    print("   node position sets: only in ref", len(sa - sb), "only in new", len(sb - sa), "common", len(sa & sb))
