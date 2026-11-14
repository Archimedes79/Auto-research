"""scratch: a test dump = the E312 seed-0 dump + a planted logsig (follows the range and the keypoint probability with noise), to run the real-scale path of sigma_check.py on real data"""
import os, pickle
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
fr = pickle.load(open(os.path.join(ROOT, "runs", "e312_p_s0", "ends_val.pkl"), "rb"))
r = np.random.RandomState(0)
for f in fr:
    rng = np.linalg.norm(f["kp_pos"], axis=1)
    f["logsig"] = np.clip(-1.0 + rng / 30.0 - 1.0 * f["kp_p"] + 0.3 * r.randn(len(rng)), -3.0, 3.0).astype(np.float32)
pickle.dump(fr, open(os.path.join(ROOT, "temp", "e318", "planted_dump.pkl"), "wb"))
print("frames", len(fr))
