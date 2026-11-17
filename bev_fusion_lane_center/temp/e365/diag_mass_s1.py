"""E365 S1 diagnosis 2 (scratch, IN-SAMPLE on VAL): for the sources with a GT successor and no delivered link: p(none), the mass on the GT group G(i), the largest single probability"""
import os, sys, pickle
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training/data", "training/grid", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_ptok_data as LP
S = int(sys.argv[1])
d = pickle.load(open(os.path.join(ROOT, f"runs/e365_s1_s{S}/token_val.pkl"), "rb"))
fr = d["frames"]
c = fr[0]["cand"]; print("cand keys", {k: getattr(v, "shape", type(v).__name__) for k, v in c.items()})
ds, dl = LP.loader("val", seed=S, bs=8, workers=0, ident="e292_a")
rows = []; i = 0
for b in dl:
    for k in range(b["valid"].shape[0]):
        f = fr[i]; i += 1
        valid = b["valid"][k].numpy(); yt = b["ytype"][k].numpy(); y = b["y"][k].numpy() > 0
        ass = (yt > 0) & valid
        c = f["cand"]
        src, dst, p, cls = np.asarray(c["src"]), np.asarray(c["dst"]), np.asarray(c["p"]), np.asarray(c["long"]).astype(int)
        r = f["rec"]; L = np.zeros_like(y); L[np.asarray(r["src"]), np.asarray(r["dst"])] = True
        for s in np.where(y.any(1) & ass)[0]:
            for cl in (0, 1):
                m = (src == s) & (cls == cl) if cls is not None else (src == s)
                if not m.any(): continue
                inG = y[s][dst[m]]
                if not inG.any(): continue
                rows.append((cl, f["none_cls"][cl][s], p[m][inG].sum(), p[m].max(), L[s].any(), f["keep"][s], f["keep"][dst[m][inG]].any()))
a = np.array(rows, float)
print("sources x class with a GT successor among candidates:", len(a))
for cl in (0, 1):
    q = a[a[:, 0] == cl]
    if len(q) == 0: continue
    print("class", cl, "n", len(q), "delivered link %.3f" % q[:, 4].mean(), "mass on G >= 0.5: %.3f" % (q[:, 2] >= 0.5).mean(), "max single p >= 0.3: %.3f" % (q[:, 3] >= 0.3).mean(), "p_none >= 0.5: %.3f" % (q[:, 1] >= 0.5).mean(), "src kept %.3f dst-in-G kept %.3f" % (q[:, 5].mean(), q[:, 6].mean()))
    miss = q[q[:, 4] == 0]
    print("   not delivered: n %d, mass on G >= 0.5: %.3f, p_none >= 0.5: %.3f, source not kept %.3f, no G token kept %.3f" % (len(miss), (miss[:, 2] >= 0.5).mean(), (miss[:, 1] >= 0.5).mean(), 1 - miss[:, 5].mean(), 1 - miss[:, 6].mean()))
