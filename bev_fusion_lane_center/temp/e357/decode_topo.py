"""E357 T0, diagnosis requested by the lane lead: the graph numbers of the standard scorer when the stored pair probabilities are decoded STRUCTURALLY instead of by a threshold on every candidate:
short links (chord <= 8 m): the highest-p successor of each source, kept when p >= thr; long links: mutual best (highest of its source AND of its destination), kept when p >= thr.
A DIAGNOSIS next to the verdict (IN-SAMPLE on VAL, a new output selection that the card did not write: D88, never a result of the card; the section-free read-out of record is `sec0`).
usage (from the problem folder): wsl bash tools/env/_py14.sh temp/e357/decode_topo.py <run> <seed> [thr]"""
import os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evaluation"))
import jtok_eval as JE  # noqa: E402

run, seed = sys.argv[1], int(sys.argv[2])
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
with open(os.path.join(JE.ROOT, "runs", run, "tok_val.pkl"), "rb") as fh:
    preds = pickle.load(fh)
frames, table = JE.val_tables(2.5)
assert len(preds) == len(frames) == 873


def structured(pred, t):
    lk = pred["links"]
    src, dst, pp = np.asarray(lk["src"], int), np.asarray(lk["dst"], int), np.asarray(lk["p"], float)
    pos = t["pos"].astype(float)
    ch = np.linalg.norm(pos[src] - pos[dst], axis=1)
    keep = np.zeros(len(src), bool)
    for long_ in (False, True):
        idx = np.flatnonzero((ch > JE.JT.SHORT) == long_)
        bs, bd = {}, {}
        for i in idx:
            s, d, q = int(src[i]), int(dst[i]), pp[i]
            if s not in bs or q > pp[bs[s]]:
                bs[s] = i
            if d not in bd or q > pp[bd[d]]:
                bd[d] = i
        for i in idx:
            if pp[i] >= thr and bs[int(src[i])] == i and (not long_ or bd[int(dst[i])] == i):
                keep[i] = True
    new = dict(pred)
    new["links"] = {k: (np.asarray(v)[keep] if hasattr(v, "__len__") and not isinstance(v, str) and len(v) == len(src) else v) for k, v in lk.items()}
    return new


with JE.JT.lattice(2.5):
    res = JE.score_arms(frames, lambda f: (table(f.k), {"p": structured(preds[f.k], table(f.k))}), 2.5, thr)["p"]
print(f"run {run}: structured decode (short: highest-p successor per source; long: mutual best), p >= {thr:g}; IN-SAMPLE on VAL, a diagnosis")
for r in ("sec0", "gtsec"):
    m = res[r]
    print(r, {k: round(m[k], 4) for k in ("topo_F1", "topo_P", "topo_R", "junction_F1", "lanes_found", "F1@1.0", "dup_frac", "vert100_dp", "link_short_P", "link_short_R", "link_junction_P", "link_junction_R", "link_junction_F1") if k in m})
