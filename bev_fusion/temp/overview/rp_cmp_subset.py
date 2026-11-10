"""scratch: the outputs with the full devkit tables (rp_out_fulltables_test) against the restricted tables (rp_out): must be identical"""
import os
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for i in range(1, 6):
    a = np.load(os.path.join(HERE, "rp_out_fulltables_test", f"{i}.npz"))
    b = np.load(os.path.join(HERE, "rp_out", f"{i}.npz"))
    d = {k: float(np.abs(a[k].astype(np.float32) - b[k].astype(np.float32)).max()) for k in a.files}
    ca = pickle.load(open(os.path.join(HERE, "rp_out_fulltables_test", f"{i}.counts.pkl"), "rb"))
    cb = pickle.load(open(os.path.join(HERE, "rp_out", f"{i}.counts.pkl"), "rb"))
    same = all(np.array_equal(ca[k], cb[k]) for k in ca)
    print(i, "max abs diff of the maps", d, "| seg counts identical:", same)
