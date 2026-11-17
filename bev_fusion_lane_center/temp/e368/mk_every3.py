"""E368 read-out (scratch): the TRAIN dump of `jpast_eval.py dump --split train` holds the 873 evenly spaced frames; the sweep of the card reads the 291-frame sample (every third of them, as E366 / E375): write token_train_every3.pkl = frames [::3] with the meta of
`jtoken_eval.dump --every 3`. usage: python mk_every3.py <run> [<run> ...]"""
import os
import pickle
import sys

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import jtoken_eval as JE  # noqa: E402

for run in sys.argv[1:]:
    src = os.path.join(ROOT, "runs", run, "token_train.pkl")
    dst = os.path.join(ROOT, "runs", run, "token_train_every3.pkl")
    assert os.path.exists(src) and not os.path.exists(dst), (src, dst)
    d = pickle.load(open(src, "rb"))
    meta, fr = d["meta"], d["frames"]
    n = len(meta["idx"])
    assert n == len(fr), (n, len(fr))
    m3 = dict(meta)
    for k, v in meta.items():
        if isinstance(v, list) and len(v) == n and k not in ("use",):
            m3[k] = v[::3]
    if isinstance(meta.get("use"), dict):
        m3["use"] = {k: (v[::3] if isinstance(v, list) and len(v) == n else v) for k, v in meta["use"].items()}
    m3["every"] = 3
    # the sample of the read-out tools for every = 3
    import argparse
    a = argparse.Namespace(every=3)
    want = JE.sample(int(os.environ.get("NTRAIN", "8414")), "train", 3)
    assert m3["idx"] == want.tolist(), "the every-3 sample of the dump is not the one of jtoken_eval.sample"
    pickle.dump(dict(d, meta=m3, frames=fr[::3]), open(dst, "wb"), protocol=4)
    print(run, "->", dst, len(fr[::3]), "frames")
