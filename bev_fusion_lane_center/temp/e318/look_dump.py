"""E318 scratch: the smoke dump has the position scale per keypoint"""
import os
import pickle

import numpy as np

fr = pickle.load(open(os.path.join(os.environ["BFLC_ROOT"], "runs", "e318_smoke_cpu_s0", "ends_val.pkl"), "rb"))
f = fr[0]
print(len(fr), "frames; keys", sorted(f), "; logsig", f["logsig"].shape, float(np.abs(f["logsig"]).max()), "; scale", f["scale"].shape)
