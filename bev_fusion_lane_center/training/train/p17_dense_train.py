"""Loop 17 / E139: dense maps (heat + direction) of the perturbed TRAIN rows (all seven failure types, 842 rows each), made by the cross-fitted dense head of the fold that did NOT train on the row (e37_fold0 - 2, as the clean training frames),
so the builder is trained on real failure frames with maps that look like those of unseen scenes. Writes runs/e17/dense_trainoof/<pert>_{ids,prob,dir}.npy (order of pert/<pert>/train_rows.npy).
Usage (gcar, GPU): p17_dense_train.py [--perts dark,blur,...]"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_pert_val as EV  # noqa: E402

HOME = os.path.expanduser("~")
OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
ap = argparse.ArgumentParser()
ap.add_argument("--perts", default="dark,blur,cam_front3_off,fog,radar_off,extr3,cam_front_off")
a = ap.parse_args()
od = f"{OUT}/e17/dense_trainoof"
os.makedirs(od, exist_ok=True)
folds = [set(int(x) for x in np.load(f"{OUT}/e37_fold{f}/ids_oof.npy")) for f in range(3)]
heads = [EV.load_net(f"e37_fold{f}") for f in range(3)]
for pn in a.perts.split(","):
    d = f"{HOME}/exp_gcar_b123/pert/{pn}"
    rows = np.load(f"{d}/train_rows.npy")
    dec, lg = np.load(f"{d}/train_dec32.npy", mmap_mode="r"), np.load(f"{d}/train_logit7.npy", mmap_mode="r")
    fo = np.array([next(f for f in range(3) if int(r) in folds[f]) for r in rows])
    prob = np.zeros((len(rows), 200, 200), np.float16)
    dirf = np.zeros((len(rows), 2, 200, 200), np.float16)
    for f in range(3):
        idx = np.where(fo == f)[0]
        net, mean, std = heads[f]
        for i in range(0, len(idx), 64):
            sel = idx[i : i + 64]
            p, dd = EV.infer(net, mean, std, dec[sel], lg[sel])
            prob[sel], dirf[sel] = p.numpy(), dd.numpy()
    np.save(f"{od}/{pn}_ids.npy", rows)
    np.save(f"{od}/{pn}_prob.npy", prob)
    np.save(f"{od}/{pn}_dir.npy", dirf)
    print(pn, prob.shape, flush=True)
print("DONE")
