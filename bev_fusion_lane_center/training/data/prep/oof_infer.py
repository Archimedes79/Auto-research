"""Nur Inferenz des Verfeinerers auf OOF-Mengen (ohne Kennzahlen): schreibt out/loop2/<verfeinerer>_oof/refined_oof<k>.pkl.   Aufruf (gcar): oof_infer.py <verfeinerer=e39_main> <k> [<k> ...]"""
import os, pickle, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refiner as R
import temporal_b as TB

ref = sys.argv[1]
ks = [int(k) for k in sys.argv[2:]]
model, ma = TB.load_model(ref)
mean, std = R.channel_stats()
mean, std = mean.cuda(), std.cuda()
os.makedirs(f"{R.OUT}/{ref}_oof", exist_ok=True)
for k in ks:
    t0 = time.time()
    ds = R.RData([dict(run=f"e37_fold{k}", st="oof", part="train")], "relax", train=False)
    r = R.infer(model, ds, mean, std, bs=16, workers=3)
    pickle.dump(r, open(f"{R.OUT}/{ref}_oof/refined_oof{k}.pkl", "wb"))
    print(f"oof{k}: {len(r)} Samples ({time.time() - t0:.0f} s)", flush=True)
