import json, os, glob, sys
root = "data/autogt"
for d in sorted(os.listdir(root)):
    p = os.path.join(root, d)
    if not os.path.isdir(p) or not os.path.exists(os.path.join(p, "index.json")): continue
    idx = json.load(open(os.path.join(p, "index.json")))
    fs = os.listdir(p)
    npz = [f for f in fs if f.endswith(".npz") and ".tmp" not in f]
    tmp = [f for f in fs if ".tmp" in f]
    other = [f for f in fs if not f.endswith(".npz") and f != "index.json"]
    print(d, {k: (v if k != "scenes" else len(v)) for k, v in idx.items()}, "npz", len(npz), "tmp", tmp[:3], "other", other[:3])
