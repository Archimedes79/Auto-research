import glob, json, os, numpy as np
for root in ("data/cache/cam_feats", "data/cache/cam_feats_bevfusion"):
    idx = sorted(glob.glob(root + "/*.index.npz"))
    n = sum(len(np.load(f)["tokens"]) for f in idx)
    print(root, len(idx), "shards", n, "tokens")
small = json.load(open("splits/small.json"))
print(len(small["train"]), len(small["val"]), small["train"][:2])
def jpg(t): return os.path.exists("data/cache/img_front/%s/%s.jpg" % (t[:2], t))
for name in ("small",):
    d = json.load(open("splits/%s.json" % name))
    for part in ("train", "val"):
        nk = {"d": 0, "b": 0, "jpg": 0, "all3": 0}
        for sc in d[part]:
            td = [str(t) for t in np.load("data/cache/cam_feats/%s.index.npz" % sc)["tokens"]]
            tb = [str(t) for t in np.load("data/cache/cam_feats_bevfusion/%s.index.npz" % sc)["tokens"]]
            sb, sd = set(tb), set(td)
            j = [t for t in sd if jpg(t)]
            nk["d"] += len(td); nk["b"] += len(tb); nk["jpg"] += len(j); nk["all3"] += len([t for t in j if t in sb])
        print(name, part, nk)
