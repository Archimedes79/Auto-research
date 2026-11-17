import os, pickle, sys
import numpy as np
R = os.environ["BFLC_ROOT"]
sys.path.insert(0, R + "/evaluation")
import jwire_data as JD
tr = pickle.load(open(R + "/temp/jwire/train_records_s0_max200.pkl", "rb"))
for s in (0.0, 0.3, 0.5, 0.7, 0.9, 1.1):
    jit = JD.make_jitter(s, 0, 4.0)
    err, tot = [], {}
    for r in tr:
        j2 = JD.make_jitter(s, 0, 4.0)
        for n in range(len(r["pairs"])):
            for side, sl in ((0, slice(0, 2)), (1, slice(4, 6))):
                g = r["gts"][n, side]
                if not np.isnan(g).any():
                    p, _ = j2(r["pairs"][n, sl], np.array([0.0, 1.0]))
                    err.append(np.linalg.norm(p - g[:2]))
        for k, v in JD.agreement(JD.instances_of(r, jit)).items():
            tot[k] = tot.get(k, 0) + v
    err = np.array(err)
    print(f"scale {s:.1f}: kp error mean {err.mean():.2f} median {np.median(err):.2f} p90 {np.percentile(err, 90):.2f} | agreement overall {tot['same'] / tot['oracle']:.3f} real {tot['multi_same'] / tot['multi']:.3f} (VAL: 1.36 / 1.16 / 2.61 | 0.856 / 0.757)")
