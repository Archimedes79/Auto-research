import os, pickle, numpy as np, sys
R = os.environ["BFLC_ROOT"]
sys.path.insert(0, R + "/evaluation")
import jwire_data as JD
live = pickle.load(open(R + "/temp/jwire/val_clean_records_s0_max24.pkl", "rb"))
ref = {r["row"]: r for r in pickle.load(open(R + "/temp/jwire/val_records_s0.pkl", "rb"))}
bad = 0
for r in live:
    q = ref[r["row"]]
    for k in ("ends", "attrs", "pairs", "kp_p", "gts"):
        a, b = np.asarray(r[k], float), np.asarray(q[k], float)
        ok = a.shape == b.shape and np.allclose(a, b, atol=2e-3, equal_nan=True)
        if not ok:
            bad += 1
            print("row", r["row"], k, a.shape, b.shape, (np.nanmax(np.abs(a - b)) if a.shape == b.shape and a.size else None))
print("records compared:", len(live), "fields differing:", bad)
