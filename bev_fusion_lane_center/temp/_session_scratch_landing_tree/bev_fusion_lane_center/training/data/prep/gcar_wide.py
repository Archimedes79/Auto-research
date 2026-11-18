"""Loop 4 / E49 (bessere Grundlage): breitere GaussianCaR-Merkmale. Die Decoder-Merkmale (256 Kanaele) wurden bisher auf 32 Hauptkomponenten (PCA) verdichtet; hier: 96 Komponenten.
  gcar_wide.py fit                         PCA (96) aus 300 Trainingszeilen anpassen -> ~/exp_gcar_b123/pca_gcar96.npz (+ erklaerte Varianz bei 32 und 96 Komponenten)
  gcar_wide.py extract --part train --start 0 --stride 4 [--end N]     dec96 fuer die Zeilen range(start, end, stride) in ~/exp_gcar_b123/feats/<part>_dec96.npy (fp16, wie dec32 orientiert)
Aufruf in der Umgebung 'gcar' (run_gcar.sh gcar_wide.py ...)."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gcar_tools as GT  # noqa: E402

HOME = GT.HOME
K = 96
FD = f"{HOME}/exp_gcar_b123/feats"
PCA = f"{HOME}/exp_gcar_b123/pca_gcar96.npz"


def make_ds(cfg, part, rows):
    from omegaconf import OmegaConf

    tokens = json.load(open(f"{FD}/{part}_tokens.json"))
    img_params = OmegaConf.to_container(cfg.data.img_params, resolve=True)
    bev_conf = OmegaConf.to_container(cfg.data.data_config.bev_aug_conf, resolve=True)
    return GT.GcarData("b123", [tokens[i] for i in rows], img_params, bev_conf), len(tokens)


def fit(module, cfg, n=300):
    tokens = json.load(open(f"{FD}/train_tokens.json"))
    rows = list(np.linspace(0, len(tokens) - 1, n).astype(int))
    ds, _ = make_ds(cfg, "train", rows)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=3, collate_fn=GT.collate)
    S, Q, N = torch.zeros(256, dtype=torch.float64, device="cuda"), torch.zeros(256, 256, dtype=torch.float64, device="cuda"), 0
    t0 = time.time()
    for k, b in enumerate(dl):
        o = GT.run(module, GT.to_dev(b))
        x = o["dec"][0].float().reshape(256, -1)[:, ::3].double()  # jedes 3. Pixel
        S += x.sum(1)
        Q += x @ x.T
        N += x.shape[1]
        if (k + 1) % 50 == 0:
            print(f"fit {k + 1}/{n} ({time.time() - t0:.0f} s)", flush=True)
    mean = S / N
    C = Q / N - torch.outer(mean, mean)
    ev, U = torch.linalg.eigh(C)
    ev, U = ev.flip(0), U.flip(1)
    ratio = (ev / ev.sum()).cpu().numpy()
    print("erklaerte Varianz: 32 Komponenten %.4f, 96 Komponenten %.4f, 256: 1.0" % (ratio[:32].sum(), ratio[:96].sum()), flush=True)
    np.savez(PCA, dec_mean=mean.float().cpu().numpy(), dec_basis=U[:, :K].float().cpu().numpy(), ratio=ratio)
    print("gespeichert:", PCA)


def extract(module, cfg, a):
    tokens = json.load(open(f"{FD}/{a.part}_tokens.json"))
    end = a.end or len(tokens)
    rows = list(range(a.start, end, a.stride))
    f = f"{FD}/{a.part}_dec96.npy"
    if not os.path.exists(f):
        try:
            np.lib.format.open_memmap(f, mode="w+", dtype=np.float16, shape=(len(tokens), K, 200, 200))
            print("angelegt:", f, flush=True)
        except FileExistsError:
            pass
    mm = np.load(f, mmap_mode="r+")
    z = np.load(PCA)
    mean, basis = torch.from_numpy(z["dec_mean"]).cuda(), torch.from_numpy(z["dec_basis"]).cuda()
    ds, _ = make_ds(cfg, a.part, rows)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=a.workers, collate_fn=GT.collate)
    t0 = time.time()
    for k, b in enumerate(dl):
        o = GT.run(module, GT.to_dev(b))
        v = GT.orient_t(GT.project(o["dec"][0].float(), mean, basis)).half().cpu().numpy()
        mm[rows[k]] = v
        if (k + 1) % 100 == 0 or k + 1 == len(rows):
            mm.flush()
            dt = time.time() - t0
            print(f"{a.part} [{a.start}:{end}:{a.stride}] {k + 1}/{len(rows)} ({dt:.0f} s, {dt / (k + 1):.2f} s/Sample)", flush=True)
    open(f"{FD}/{a.part}_dec96_{a.start}_{end}_s{a.stride}.done", "w").write("ok")
    print("FERTIG", a.part, a.start, end, a.stride, f"{time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "extract"])
    ap.add_argument("--part", default="train")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    module, cfg = GT.load_module(f"{HOME}/work/gaussiancar_ckpt/nuscenes_map.ckpt")
    if a.cmd == "fit":
        fit(module, cfg)
    else:
        extract(module, cfg, a)


if __name__ == "__main__":
    main()
