import os, sys, time
R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
t0 = time.time()
import torch; print("torch", time.time() - t0); t0 = time.time()
import lane_ptok_data; print("ptok", time.time() - t0); t0 = time.time()
import lane_seq_data; print("seq", time.time() - t0); t0 = time.time()
import scipy.spatial; print("scipy", time.time() - t0)
