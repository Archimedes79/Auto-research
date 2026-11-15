"""E323 scratch: how many delivered connectors (p >= 0.1) have a near-duplicate partner in the same frame (both ends within R m of the ends of another delivered connector), for the keypoint ends and the attached ends of a dump (no scoring)."""
import os, pickle, sys
import numpy as np
ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
def share(run, attach, R):
    fr = pickle.load(open(os.path.join(ROOT, "runs", run, "ends_val.pkl"), "rb"))
    n = d = 0
    for f in fr:
        k = f["kp_pos"].shape[0] // 2
        p = 1.0 / (1.0 + np.exp(-f["logit"])) * f["allowed"]
        ia, ib = np.nonzero(p >= 0.1)
        if len(ia) == 0:
            continue
        pos = f["att_pos"] if attach else f["kp_pos"]
        A, B = pos[ia], pos[k + ib]
        da = np.linalg.norm(A[:, None] - A[None], axis=-1)
        db = np.linalg.norm(B[:, None] - B[None], axis=-1)
        m = (da <= R) & (db <= R)
        np.fill_diagonal(m, False)
        n += len(ia)
        d += int(m.any(1).sum())
    return n, d
for S in (0, 1):
    for R in (1.0, 2.0):
        row = []
        for run, att, name in ((f"e312_p_s{S}", False, "E312 keypoints"), (f"e320_a_s{S}", False, "E320 keypoints"), (f"e320_a_s{S}", True, "E320 attached")):
            n, d = share(run, att, R)
            row.append(f"{name} {100 * d / n:.1f} % of {n}")
        print(f"seed {S} R {R} m: " + "; ".join(row))
