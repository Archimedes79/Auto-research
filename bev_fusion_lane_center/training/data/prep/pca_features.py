"""PCA-Kompression der gespeicherten BEV-Features (512 -> k Kanaele) fuer einen kleineren Zwischenspeicher.

Basis wird aus den TRAININGS-Features geschaetzt (zufaellige Pixel), Training und Validierung werden damit projiziert und als
{split}_feats{k}.npy (float16) gespeichert; die Basis liegt in pca{k}_basis.npz (fuer neue Daten wiederverwendbar).
Aufruf: python pca_features.py ~/exp/feats 64 32
"""
import os
import sys

import numpy as np
import torch

fdir = os.path.expanduser(sys.argv[1])
ks = [int(k) for k in sys.argv[2:]]
dev = "cuda"

tr = np.load(f"{fdir}/train_feats.npy", mmap_mode="r")
S = tr.shape[0]
rng = np.random.RandomState(0)
# 400 000 zufaellige Pixel aus dem Training
idx_s = rng.randint(0, S, 400000)
idx_y = rng.randint(0, 128, 400000)
idx_x = rng.randint(0, 128, 400000)
X = np.empty((400000, 512), np.float32)
for s in np.unique(idx_s):
    m = idx_s == s
    f = np.asarray(tr[s], np.float32)  # [512,128,128]
    X[m] = f[:, idx_y[m], idx_x[m]].T
X = torch.from_numpy(X).to(dev)
mean = X.mean(0)
Xc = X - mean
cov = Xc.T @ Xc / (Xc.shape[0] - 1)
ev, evec = torch.linalg.eigh(cov)
ev, evec = ev.flip(0), evec.flip(1)  # absteigend
tot = ev.sum()
for k in ks:
    print(f"k={k}: erklaerte Varianz {float(ev[:k].sum() / tot) * 100:.1f} %", flush=True)
    basis = evec[:, :k].contiguous()
    np.savez(f"{fdir}/pca{k}_basis.npz", mean=mean.cpu().numpy(), basis=basis.cpu().numpy())
    for split in ("train", "val"):
        f = np.load(f"{fdir}/{split}_feats.npy", mmap_mode="r")
        out = np.lib.format.open_memmap(f"{fdir}/{split}_feats{k}.npy", mode="w+", dtype=np.float16, shape=(f.shape[0], k, 128, 128))
        for i in range(f.shape[0]):
            x = torch.from_numpy(np.asarray(f[i], np.float32)).to(dev)  # [512,H,W]
            y = torch.einsum("chw,ck->khw", x - mean[:, None, None], basis)
            out[i] = y.half().cpu().numpy()
        out.flush()
    print(f"k={k}: gespeichert", flush=True)
