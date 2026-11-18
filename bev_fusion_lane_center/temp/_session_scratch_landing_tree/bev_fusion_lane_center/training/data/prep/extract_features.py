"""Frisst alle Mini-Samples (train + val) durch das eingefrorene BEVFusion-Seg-Modell und speichert

  feats  [N, 512, 128, 128] float16 : Eingabe des Kartenkopfes (BEV-Features, Lidar-System, +-51.2 m, 0.8 m Zellen;
                                      512 = SECONDFPN-Ausgabe, 2 x 256 Kanaele)
  sem    [N, 6, 200, 200]   float16 : Ausgabe des Kartenkopfes (Wahrscheinlichkeiten, +-50 m, 0.5 m Zellen)
  tokens [N]                        : sample-Tokens (gleiche Reihenfolge)

Zeilen = x_lidar, Spalten = y_lidar. Zellmitte i: -half + (i + 0.5) * res.
Aufruf (in WSL, Umgebung 'bev', Arbeitsverzeichnis ~/work/bevfusion):
    python <problem folder>/training/data/prep/extract_features.py ~/exp/feats
"""
import json
import os
import sys

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval

out_dir = os.path.expanduser(sys.argv[1])
os.makedirs(out_dir, exist_ok=True)
cfg_path = "configs/nuscenes/seg/fusion-bev256d2-lss-infer.yaml"
# Umgebungsvariablen: DATA = Datenordner relativ zum Repo (Standard data/nuscenes), SPLITS = "train,val",
# PCA_BASIS = npz mit mean/basis (512 -> K Kanaele, spart Platz; Ausgabedatei heisst dann {split}_feats{K}.npy)
DATA = os.environ.get("DATA", "data/nuscenes").rstrip("/")
SPLITS = os.environ.get("SPLITS", "train,val").split(",")
pca = os.environ.get("PCA_BASIS")
if pca:
    _z = np.load(pca)
    pca_mean = torch.from_numpy(_z["mean"]).cuda()
    pca_basis = torch.from_numpy(_z["basis"]).cuda()
    K = pca_basis.shape[1]
else:
    K = 512

configs.load(cfg_path, recursive=True)
cfg = Config(recursive_eval(configs), filename=cfg_path)
cfg.model.pretrained = None
cfg.model.train_cfg = None

model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
if cfg.get("fp16", None) is not None:
    wrap_fp16_model(model)
load_checkpoint(model, "pretrained/bevfusion-seg.pth", map_location="cpu")
model = MMDataParallel(model.cuda(), device_ids=[0]).eval()

cap = {}
head = model.module.heads["map"]
head.register_forward_pre_hook(
    lambda m, inp: cap.__setitem__("x", (inp[0][0] if isinstance(inp[0], (list, tuple)) else inp[0]).detach())
)
head.register_forward_hook(lambda m, inp, out: cap.__setitem__("sem", out.detach()))

for split in SPLITS:
    ds_cfg = cfg.data.test
    ds_cfg.ann_file = f"{DATA}/nuscenes_infos_{split}.pkl"
    ds_cfg.dataset_root = DATA + "/"
    for _st in ds_cfg.pipeline:
        if "dataset_root" in _st:
            _st["dataset_root"] = DATA + "/"
    ds_cfg.test_mode = True
    dataset = build_dataset(ds_cfg)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    n = len(dataset)
    fname = f"{split}_feats{K}.npy" if pca else f"{split}_feats.npy"
    feats = np.lib.format.open_memmap(f"{out_dir}/{fname}", mode="w+", dtype=np.float16, shape=(n, K, 128, 128))
    sem = np.lib.format.open_memmap(f"{out_dir}/{split}_sem.npy", mode="w+", dtype=np.float16, shape=(n, 6, 200, 200))
    tokens = [dataset.data_infos[i]["token"] for i in range(n)]
    with torch.no_grad():
        for i, data in enumerate(loader):
            model(return_loss=False, rescale=True, **data)
            x = cap["x"][0]
            if pca:  # PCA-Projektion auf der GPU: (x - Mittelwert) @ Basis
                x = torch.einsum("chw,ck->khw", x.float() - pca_mean[:, None, None], pca_basis)
            feats[i] = x.half().cpu().numpy()
            sem[i] = cap["sem"][0].half().cpu().numpy()
            if i % 50 == 0:
                print(f"{split} {i}/{n}", flush=True)
    feats.flush()
    sem.flush()
    json.dump(tokens, open(f"{out_dir}/{split}_tokens.json", "w"))
    print(f"{split}: {n} Samples gespeichert", flush=True)
