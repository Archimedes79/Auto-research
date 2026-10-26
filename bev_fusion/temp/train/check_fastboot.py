"""One-off: the fast bootstrap AP equals det_ap (full re-matching) on the baseline's val records, for the full set and a resample."""
import pickle
import sys

import numpy as np

sys.path.insert(0, ".")
from evaluation import evaluate_run as ev  # noqa: E402

rec = pickle.load(open("runs/t1_r0/eval/scene_records.pkl", "rb"))
f = {s: r[0.0] for s, r in rec["det"].items() if 0.0 in r}
sc = sorted(f)
rng = np.random.default_rng(1)
for cls in ("vehicle", "pedestrian"):
    fl = ev.scene_flags(f, sc, cls)
    for ix in (list(range(len(sc))), list(rng.integers(0, len(sc), len(sc)))):
        s_ix = [sc[i] for i in ix]
        a, b = ev.ap_from_flags(fl, s_ix), ev.det_ap(f, s_ix, cls)
        print(cls, f"fast {a:.6f} slow {b:.6f} diff {abs(a - b):.2e}")
        assert abs(a - b) < 1e-9
print("fast bootstrap AP equals det_ap")
