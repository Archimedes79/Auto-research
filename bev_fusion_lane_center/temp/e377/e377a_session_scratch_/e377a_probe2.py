"""throw-away probe: p_none of the untrained floor head on the 3 cache frames of test_anatomy, kept none tokens at several keep thresholds, deg3 of the decode"""
import os
import sys
import tempfile

ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "evaluation", "tests"))
import numpy as np  # noqa: E402
import test_jtoken_eval as T  # noqa: E402

E = T.E
with tempfile.TemporaryDirectory() as tmp:
    runs = os.path.join(tmp, "runs")
    T._floor_run(runs)
    E.dump(E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--max", "3"]), runs)
    import pickle
    d = pickle.load(open(os.path.join(runs, "h", "token_val_n3.pkl"), "rb"))
ds = E.open_cache("val", 0)
pn = np.concatenate([f["pnone"][ds[i]["tokens_np"]["valid"] & (ds[i]["ytype"] == 0)] for i, f in enumerate(d["frames"])])
print("valid none tokens", len(pn), "p_none quantiles", np.quantile(pn, [0, 0.1, 0.5, 0.9, 1]).round(4))
for kt in (0.3, 0.5, 0.55, 0.6, 0.65, 0.7, 1.0):
    print("kept at", kt, int((pn < np.float32(kt)).sum()))
for kt in (0.5, 0.6):
    deg = 0
    for i, f in enumerate(d["frames"]):
        c = E.decode_variant(f, ds[i]["tokens_np"]["valid"], 0.3, kt, False, 0.05)["rec"]
        deg += int(((np.bincount(c["src"].astype(int), minlength=ds.K) + np.bincount(c["dst"].astype(int), minlength=ds.K)) >= 3).sum())
    print("deg3 at keep", kt, deg)
