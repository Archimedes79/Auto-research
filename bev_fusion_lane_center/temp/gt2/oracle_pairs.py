"""headroom of the pair decision: an ORACLE pair classifier on the keypoints of a pair-head run (logit +20 for a candidate whose two keypoints are strictly matched to the two ends of a GT pair, -20 for every other candidate), delivered through the scorer of record.
The number is the upper bound of everything that can be won by a better pair head on THESE keypoints (an oracle is a sanity check, not a bound of the system). Scratch, CPU; the run's own cards are not touched (the file paths of the scorer are redirected).
usage: oracle_pairs.py <run> <seed> [<variant>]   variant: all (default) = every GT pair whose ends are found; ord = additionally only the pairs of the order rule are no different (same)"""
import builtins
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("BFLC_ROOT", ROOT)
for sub in ("evaluation", "training", "training/grid", "training/data", "tools/viz", ".."):
    sys.path.insert(0, os.path.join(ROOT, sub))
import jgt as J
import jrule_control as JR
import qends_eval as QE
import qgraph_view as QV

run, seed = sys.argv[1], int(sys.argv[2])
SNAP = len(sys.argv) > 3 and sys.argv[3] == "snap"
src = os.path.join(ROOT, "runs", run, "ends_val.pkl")
frames = pickle.load(open(src, "rb"))
tok2gt = {}
out = []
n_gt = n_ok = 0
for e in frames:
    jg = J.junction_gt(QV.load_gt(e["token"]))
    e = dict(e)
    K = e["logit"].shape[0]
    lg = np.full_like(e["logit"], -20.0)
    if jg is not None:
        ve = e["valid"][0] & (e["kp_p"][:K] >= 0.5)
        vx = e["valid"][1] & (e["kp_p"][K:] >= 0.5)
        ma, mb = JR._match(e["kp_pos"][:K], e["kp_dir"][:K], ve, jg["ent"]), JR._match(e["kp_pos"][K:], e["kp_dir"][K:], vx, jg["ext"])
        gt = {(int(p[0]), int(p[1])) for p in jg["pairs"]}
        n_gt += len(gt)
        for a, b in gt:
            if ma[a] >= 0 and mb[b] >= 0 and e["allowed"][ma[a], mb[b]]:
                lg[ma[a], mb[b]] = 20.0
                n_ok += 1
    e["logit"] = lg
    out.append(e)
print(f"oracle: {n_ok} of {n_gt} GT pairs have both ends found and are allowed ({n_ok / n_gt:.3f})", flush=True)
tmp = os.path.join(ROOT, "temp", "gt2", f"oracle_{run}.pkl")
pickle.dump(out, open(tmp, "wb"))
target = os.path.join(ROOT, "temp", "gt2", "thr", f"ORACLE_{run}_{'snap_' if SNAP else ''}thr0.5.json")
os.makedirs(os.path.dirname(target), exist_ok=True)


def safe_open(path, mode="r", *a, **k):
    p = str(path)
    if "ends_card" in p and "w" in mode:
        return builtins.open(target, mode, *a, **k)
    if p.endswith("ends_val.pkl") and run in p:
        return builtins.open(tmp, mode, *a, **k)
    return builtins.open(path, mode, *a, **k)


QE.open = safe_open
QE.cmd_score(SimpleNamespace(run=run, seed=seed, thr=0.5, full=None, base=None, skip_base_check=True, chord=False, snap=SNAP))
