"""Scratch: host time of the group / rank bookkeeping of the pair head (`jends.JEnds.rank_matrix`) on the stored VAL keypoints (E312 seed 0). CPU, one thread.
usage (project folder): wsl bash -c "nice bash tools/env/_py14.sh temp/_session_scratch_s292/time_groups.py" """
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training/grid", "training", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import jgt as JG  # noqa: E402

d = pickle.load(open(f"{ROOT}/runs/e312_p_s0/ends_val.pkl", "rb"))
K = 32
t_g = t_r = 0.0
n_valid = []
for f in d:
    for kind in (0, 1):
        v = np.asarray(f["valid"])[kind]
        ids = np.where(v)[0]
        if not len(ids):
            continue
        E = np.concatenate([f["kp_pos"][kind * K + ids], f["kp_dir"][kind * K + ids]], 1).astype(float)
        n_valid.append(len(ids))
        t0 = time.perf_counter()
        grp = JG._groups(E)
        t1 = time.perf_counter()
        JG._ranks(E, grp)
        t2 = time.perf_counter()
        t_g += t1 - t0
        t_r += t2 - t1
print(f"frames {len(d)}, mean valid keypoints per kind {np.mean(n_valid):.1f} (max {max(n_valid)})")
print(f"_groups {1000 * t_g / len(d):.2f} ms per frame (both kinds), _ranks {1000 * t_r / len(d):.2f} ms per frame")
