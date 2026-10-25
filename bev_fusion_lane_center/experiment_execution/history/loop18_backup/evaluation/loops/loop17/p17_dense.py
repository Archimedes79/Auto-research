"""Loop 17: dense maps (heat + direction) of one dense head on the clean VAL features and on the perturbed VAL series, for the builder input (p17_series.py --dense <tag>).
Writes runs/e17/dense_<tag>/clean_{ids,prob,dir}.npy (all VAL rows) and <pert>_{ids,prob,dir}.npy (the 291 rows of the series).
Usage (gcar, GPU): p17_dense.py --head e65_ftall --tag ftall --perts dark,blur,cam_front3_off"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_pert_val as EV  # noqa: E402

HOME = os.path.expanduser("~")
E17 = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e17"

ap = argparse.ArgumentParser()
ap.add_argument("--head", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--perts", default="dark,blur,cam_front3_off")
a = ap.parse_args()
out = f"{E17}/dense_{a.tag}"
os.makedirs(out, exist_ok=True)
net, mean, std = EV.load_net(a.head)
rows = np.array(json.load(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/data/splits.json"))["val"])
d = f"{HOME}/exp_gcar_b123/feats"
dec, lg = np.load(f"{d}/val_dec32.npy", mmap_mode="r")[rows], np.load(f"{d}/val_logit7.npy", mmap_mode="r")[rows]
prob, dirf = EV.infer(net, mean, std, dec, lg)
np.save(f"{out}/clean_ids.npy", rows)
np.save(f"{out}/clean_prob.npy", prob.numpy())
np.save(f"{out}/clean_dir.npy", dirf.numpy())
print("clean", prob.shape, flush=True)
for pn in a.perts.split(","):
    dd = f"{HOME}/exp_gcar_b123/pert/{pn}"
    pr = np.load(f"{dd}/val_rows.npy")
    prob, dirf = EV.infer(net, mean, std, np.load(f"{dd}/val_dec32.npy", mmap_mode="r"), np.load(f"{dd}/val_logit7.npy", mmap_mode="r"))
    np.save(f"{out}/{pn}_ids.npy", pr)
    np.save(f"{out}/{pn}_prob.npy", prob.numpy())
    np.save(f"{out}/{pn}_dir.npy", dirf.numpy())
    print(pn, prob.shape, flush=True)
print("DONE")
