import os, sys, pickle
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import jprune_b as PB
ref = pickle.load(open(os.path.join(ROOT, "temp/e276/predfull_e273_p_s0.pkl"), "rb"))[:24]
new = pickle.load(open(os.path.join(ROOT, "temp/e352/check_s0/predfull_null_s0.pkl"), "rb"))
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for i in (0, 5, 12, 3):
    a, b = ref[i], new[i]
    ca, cb = PB.canon_b(a), PB.canon_b(b)
    ka = {tuple(np.round(p, 2)) + tuple(np.round(d, 2)): (j, e) for j, (p, d, e) in enumerate(zip(ca["nodes"]["pos"], ca["nodes"]["dirv"], ca["nodes"]["e"]))}
    kb = {tuple(np.round(p, 2)) + tuple(np.round(d, 2)): (j, e) for j, (p, d, e) in enumerate(zip(cb["nodes"]["pos"], cb["nodes"]["dirv"], cb["nodes"]["e"]))}
    onlya = [k for k in ka if k not in kb]; onlyb = [k for k in kb if k not in ka]
    print(f"frame {i}: nodes {len(a['e'])}/{len(b['e'])}, only in ref {len(onlya)} (e {[round(ka[k][1], 3) for k in onlya][:6]}), only in new {len(onlyb)} (e {[round(kb[k][1], 3) for k in onlyb][:6]}), duplicates in rounded keys: ref {len(a['e']) - len(ka)} new {len(b['e']) - len(kb)}")
    # nodes equal at the same canonical rank?
    if len(a["e"]) == len(b["e"]):
        d = np.abs(ca["nodes"]["pos"] - cb["nodes"]["pos"]).max(1)
        print("   canonical pos diff: n>0.05:", int((d > 0.05).sum()), "median", float(np.median(d)))
    la = ca["links"]; lb = cb["links"]
    print("   links", len(la["key"]), len(lb["key"]), "common keys", len(np.intersect1d(la["key"], lb["key"])))
