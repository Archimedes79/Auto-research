"""throw-away probe: can the argmax decode be rebuilt from a stored dump's candidates (p >= 0.05) and none probabilities? frames of runs/<run>/token_val.pkl whose rebuilt argmax at keep 0.5 differs from the stored one"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "evaluation"))
import jtoken_eval as E  # noqa: E402

run = sys.argv[1]
d = pickle.load(open(os.path.join(E.RUNS, run, "token_val.pkl"), "rb"))
ds = E.open_cache("val", int(run[-1]))
bad = links = 0
for i, f in enumerate(d["frames"]):
    v = np.asarray(ds.c.f["valid"][d["meta"]["idx"][i]], bool)
    c = f["cand"]
    keep = v & (f["pnone"] < np.float32(0.5))
    src, dst, cls = c["src"].astype(int), c["dst"].astype(int), c["long"].astype(int)
    order = np.lexsort((dst, -c["p"], cls, src))
    first = np.ones(len(order), bool)
    first[1:] = (src[order][1:] != src[order][:-1]) | (cls[order][1:] != cls[order][:-1])
    w = order[first]
    w = np.sort(w[(c["p"][w] >= f["none_cls"][cls[w], src[w]]) & keep[src[w]] & keep[dst[w]]])
    got = set(zip(src[w].tolist(), dst[w].tolist()))
    want = set(zip(f["argmax"]["src"].tolist(), f["argmax"]["dst"].tolist()))
    links += len(want)
    bad += got != want
    if got != want and bad < 4:
        print(i, "missing", sorted(want - got)[:5], "extra", sorted(got - want)[:5])
print(f"{run}: {bad} of {len(d['frames'])} frames differ, {links} stored argmax links")
