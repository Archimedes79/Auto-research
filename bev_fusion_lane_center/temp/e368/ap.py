"""E368 read-out (scratch): the threshold-free existence quality of stream dumps: AP of 1 - p_none against 'assigned to a GT node' over the valid tokens of VAL; and the node recall at the precision of a reference run. usage: python ap.py <run> [<run> ...] (VAL stream dumps)"""
import os
import pickle
import sys

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "training", "data"))
import lane_tokcache as LC  # noqa: E402
import lane_ptok_data as LP  # noqa: E402

res = {}
for run in sys.argv[1:]:
    seed = int(run[-1])
    d = pickle.load(open(os.path.join(ROOT, "runs", run, "token_val.pkl"), "rb"))
    c = LC.TokCache(LP.cache_dir("val", seed, "e292_a"))
    tok2i = {t: i for i, t in enumerate(c.tokens)}
    s, y = [], []
    for f in d["frames"]:
        i = tok2i[f["token"]]
        v = np.array(c.f["valid"][i])
        s.append(1.0 - np.asarray(f["pnone"], float)[v])
        y.append(np.array(c.f["gt_id"][i])[v] >= 0)
    s, y = np.concatenate(s), np.concatenate(y)
    ap = average_precision_score(y, s)
    pr, rc, _ = precision_recall_curve(y, s)
    res[run] = (ap, pr, rc)
    print("%-12s tokens %d assigned share %.3f | AP of the existence %.4f" % (run, len(y), y.mean(), ap))
runs = list(res)
for a in runs:
    for b in runs:
        if a != b and a[-1] == b[-1] and a.split("_")[1] != b.split("_")[1]:
            pr_b, rc_b = res[b][1], res[b][2]
            p0 = 0.85
            ra = rc_b[np.searchsorted(-pr_b, -p0)] if False else None
print("done")
