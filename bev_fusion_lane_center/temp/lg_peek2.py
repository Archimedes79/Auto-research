import gzip, pickle, glob, os, collections
import numpy as np
d = "/mnt/c/Auto-research/data/autogt/lane/3576fa1f7e"
fs = sorted(glob.glob(d + "/*.pkl.gz"))
seen = {}
for f in fs[::7][:300]:
    g = pickle.load(gzip.open(f, "rb"))
    for e in g["events"]:
        if e["type"] not in seen:
            seen[e["type"]] = (f.split("/")[-1][:8], e, g)
for t, (f, e, g) in seen.items():
    print("==", t, f)
    print({k: (np.round(v, 2).tolist() if isinstance(v, np.ndarray) else v) for k, v in e.items()})
ids = {p["id"]: p for p in g["pieces"]}
print("piece ids in the last graph", list(ids)[:6])
# do succ points coincide with the end point of the predecessor part?
d_end = []
for f in fs[::20][:50]:
    g = pickle.load(gzip.open(f, "rb"))
    P = {p["id"]: p for p in g["pieces"]}
    for p in g["pieces"]:
        for q in p["succ"]:
            if q in P:
                d_end.append(float(np.linalg.norm(p["pts"][-1] - P[q]["pts"][0])))
d_end = np.array(d_end)
print("succ joint distance: n", len(d_end), "median", np.median(d_end), "max", d_end.max())
print("pieces per frame", np.mean([len(pickle.load(gzip.open(f, "rb"))["pieces"]) for f in fs[::50][:40]]))
