"""E80: previous-frame outputs of the SD head for all TRAIN rows (clean features, SD map with a scene-constant pose bias sigma 1 m): (map-free lane L0, lane L, route R) as uint8 [N,3,200,200]
-> ~/exp_gcar_b123/prior_train_<model>.npy. The temporal head is trained with these (corrupted) outputs of the previous row as its memory input.
Usage (gcar): prior_precompute.py --model e78_sd_a"""
import argparse, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import temporal_eval as TE
import route_head3 as RH3

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="e78_sd_a")
ap.add_argument("--sigma", type=float, default=1.0)
a = ap.parse_args()
D = RH3.D
P = dict(np.load(f"{D}/poses_train.npz"))
SDm = RH3.sd_maps("train")
net, mean, std = TE.load_model(a.model, "sd")
n = len(P["prev"])
t0 = time.time()
out = TE.infer_rows(net, mean, std, "clean", list(range(n)), P, SDm, a.sigma, "sd", part="train")
arr = np.zeros((n, 3, 200, 200), np.uint8)
for i, (L0, L, R, ag) in out.items():
    arr[i] = (torch.stack([L0, L, R]).float() * 255).round().byte().numpy()
np.save(f"{D}/prior_train_{a.model}.npy", arr)
print("saved", arr.shape, f"{time.time() - t0:.0f} s")
