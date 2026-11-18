"""equivalence of the pair-list chord_long with the dense ylong of PTokData on real VAL frames (the current code computes the dense one): usage python perf_check.py [n]"""
import os, sys, time
import numpy as np
R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import lane_ptok_data as LP
import qtargets as QT
def chord_long(y, pos):
    yl = np.zeros(y.shape, bool)
    i, j = np.nonzero(y > 0)
    yl[i, j] = (np.linalg.norm(pos[j] - pos[i], axis=-1) > QT.SHORT)
    return yl
ds = LP.PTokData("val", 0)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
idx = np.linspace(0, len(ds) - 1, n).astype(int)
bad, t_dense, t_new = 0, 0.0, 0.0
for i in idx:
    o = ds[int(i)]
    y, pos = o["y"], np.asarray(o["tokens_np"]["pos"], np.float64)
    t0 = time.perf_counter(); d = (y > 0) & (np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=-1) > QT.SHORT); t1 = time.perf_counter(); c = chord_long(y, pos); t2 = time.perf_counter()
    t_dense += t1 - t0; t_new += t2 - t1
    bad += int(not np.array_equal(d, c)) + int(not np.array_equal(d, o["ylong"]))
print(f"{n} VAL frames: differing {bad}; dense {t_dense / n * 1000:.2f} ms, pair list {t_new / n * 1000:.3f} ms per frame")
