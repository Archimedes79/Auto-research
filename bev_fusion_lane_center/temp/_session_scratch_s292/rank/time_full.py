import os, sys, time, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import vec
JG = vec.JG
ROOT = os.environ["BFLC_ROOT"]
fr = pickle.load(open(os.path.join(ROOT, "runs", "e312_p_s0", "ends_val.pkl"), "rb"))
sets = []
for f in fr:
    pos, D, val = np.asarray(f["kp_pos"], np.float32), np.asarray(f["kp_dir"], np.float32), np.asarray(f["valid"]).reshape(2, 32)
    for kind in (0, 1):
        ids = np.where(val[kind])[0]
        if len(ids):
            sets.append(np.concatenate([pos[kind * 32 + ids], D[kind * 32 + ids]], 1))
def run(fg):
    t0 = time.perf_counter()
    for E in sets:
        grp = fg(E)
        rk = JG._ranks(E, grp)
        cnt = np.bincount(grp)
    return (time.perf_counter() - t0) / len(sets) * 1e3
def only_ranks():
    grps = [JG._groups(E) for E in sets]
    t0 = time.perf_counter()
    for E, g in zip(sets, grps):
        JG._ranks(E, g)
    return (time.perf_counter() - t0) / len(sets) * 1e3
print(f"{len(sets)} sets; old groups+ranks {run(JG._groups):.3f} ms per set; vec groups + old ranks {run(vec.groups_vec):.3f} ms per set; ranks alone {only_ranks():.3f} ms per set")
