"""Loop 17: perturbed TRAIN features per failure type and their out-of-fold dense heads (e37_fold0 - 2): sizes and overlap."""
import os

import numpy as np

H = os.path.expanduser("~")
OUT = "/mnt/c/ML-Experiment/out/loop2"
folds = {f: set(int(x) for x in np.load(f"{OUT}/e37_fold{f}/ids_oof.npy")) for f in range(3)}
print("oof folds", {f: len(v) for f, v in folds.items()})
for p in ("dark", "blur", "cam_front3_off", "fog", "radar_off", "extr3", "cam_front_off"):
    d = f"{H}/exp_gcar_b123/pert/{p}"
    done = os.path.exists(f"{d}/DONE_train")
    if not done:
        print(p, "train features: not done")
        continue
    rows = np.load(f"{d}/train_rows.npy")
    cnt = [sum(int(r) in folds[f] for r in rows) for f in range(3)]
    print(f"{p:16s} train rows {len(rows)} in folds {cnt} (sum {sum(cnt)}), dec32 shape {np.load(f'{d}/train_dec32.npy', mmap_mode='r').shape}")
