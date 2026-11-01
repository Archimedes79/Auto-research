"""Schleife 5 / E57: Robustheit gegen die reale Welt - Merkmale des eingefrorenen Fundaments (GaussianCaR) unter kuenstlichen Sensorstoerungen auf den TEST-Zeilen.
Je Stoerung ~/exp_gcar_b123/pert/<name>/test_{dec32,logit7}.npy (fp16, wie die normalen Merkmale: PCA 32 + 7 Karten-Logits, Layout des Solls) und test_rows.npy (Zeilen im Teil 'val').
Stoerungen (--pert):
  cam_front_off      Frontkamera schwarz           cam_front3_off  die drei nach vorn gerichteten Kameras schwarz
  dark               Nacht: Bilder x 0,15 + Rauschen   fog   Schleier (Kontrast x 0,45 + Licht)   blur  Gauss-Unschaerfe sigma 3 px   noise  Rauschen sigma 0,12
  radar_drop90       90 % der Radarpunkte zufaellig weg   radar_off  nur 3 Radarpunkte uebrig
  extr1 / extr3      Kamera-Extrinsik: zufaellige Drehung (sigma 1 bzw. 3 Grad je Achse, je Kamera und Sample)
  night_radar        dark + radar_drop90
Aufruf (Umgebung gcar):  gcar_pert.py --pert dark [--workers 3]"""
import bflc_paths as BP  # project paths from the location of this checkout
import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gcar_tools as G  # noqa: E402

HOME = G.HOME
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
FRONT = G.CAMERAS.index("CAM_FRONT")
FRONT3 = [G.CAMERAS.index(c) for c in ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT")]


def seed_of(token, salt=0):
    return (zlib.crc32(token.encode()) + 7919 * salt) % (2 ** 32)


def euler_rot(deg):
    a, b, c = np.deg2rad(deg)
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


def gauss_blur(x, sigma):
    k = int(2 * round(3 * sigma) + 1)
    ax = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    n, c, h, w = x.shape
    x = F.conv2d(F.pad(x.reshape(n * c, 1, h, w), (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))
    return x.reshape(n, c, h, w)


class PertData(G.GcarData):
    def __init__(self, split, tokens, img_params, bev_conf, pert):
        super().__init__(split, tokens, img_params, bev_conf)
        self.pert = pert

    def sample_dict(self, token):
        d = super().sample_dict(token)
        if self.pert in ("extr1", "extr3"):
            sig = 1.0 if self.pert == "extr1" else 3.0
            rng = np.random.default_rng(seed_of(token, 1))
            ex = []
            for e in d["extrinsics"]:
                E4 = np.eye(4)
                E4[:3, :3] = euler_rot(rng.normal(0.0, sig, 3))
                ex.append((E4 @ np.asarray(e, np.float64)).tolist())
            d["extrinsics"] = ex
        return d

    def __getitem__(self, i, alt=0):
        token = self.tokens[i]
        d = self.sample_dict(token)
        for t in self.tf:
            d = t(d)
        p = self.pert
        img = d["image"]
        raw = img * STD[None] + MEAN[None]
        g = torch.Generator().manual_seed(seed_of(token, 2))
        if p == "cam_front_off":
            raw[FRONT] = 0.0
        elif p == "cam_front3_off":
            raw[FRONT3] = 0.0
        elif p in ("dark", "night_radar"):
            raw = raw * 0.15 + 0.02 * torch.randn(raw.shape, generator=g)
        elif p == "fog":
            raw = raw * 0.45 + 0.55 * 0.78 + 0.01 * torch.randn(raw.shape, generator=g)
        elif p == "blur":
            raw = gauss_blur(raw, 3.0)
        elif p == "noise":
            raw = raw + 0.12 * torch.randn(raw.shape, generator=g)
        d["image"] = (raw.clamp(0.0, 1.0) - MEAN[None]) / STD[None]
        rp = d["radar_points"]
        if p in ("radar_drop90", "night_radar"):
            keep = torch.rand(len(rp), generator=g) < 0.10
            rp = rp[keep] if keep.sum() >= 3 else rp[:3]
        elif p == "radar_off":
            rp = rp[torch.randperm(len(rp), generator=g)[: 3 + 3 * alt]]  # alt > 0: Wiederholung mit mehr Punkten (PTv3-Hilbert-Kodierung scheitert, wenn alle Punkte in derselben Gitterzelle liegen)
        return dict(image=d["image"], lidar2img=d["lidar2img"], radar_points=rp, index=i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pert", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n", type=int, default=0, help="nur die ersten n TEST-Zeilen (Probelauf)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--resume", type=int, default=0, help="1: vorhandene Merkmalsdateien weiterverwenden und nur leere Zeilen nachrechnen (nach einem Abbruch)")
    ap.add_argument("--part", default="test", choices=["test", "train", "val"], help="test = TEST-Zeilen (splits.json); train = Trainingszeilen arange(0, n, stride) fuer das robuste Training (E65); val = VAL-Zeilen (splits.json, jede stride-te) fuer die Modellwahl ohne TEST (E73)")
    a = ap.parse_args()
    if os.path.exists(f"{HOME}/exp_gcar_b123/pert/SKIP_{a.pert}"):  # uebersprungene Stoerungen (Rechenzeit): sofort beenden
        print("SKIP", a.pert, flush=True)
        os._exit(0)
    if a.stride == 1 and a.part == "test" and a.pert in ("radar_off", "blur", "extr3", "radar_drop90", "night_radar", "noise", "extr1"):
        a.stride = 2  # zweite Runde: jede zweite TEST-Zeile (halbe Rechenzeit)
    from omegaconf import OmegaConf

    module, cfg = G.load_module(f"{HOME}/work/gaussiancar_ckpt/nuscenes_map.ckpt")
    d = G.feat_dir("b123")
    if a.part == "train":
        tokens = json.load(open(f"{d}/train_tokens.json"))
        rows = np.arange(0, len(tokens), max(a.stride, 1))
    else:
        tokens = json.load(open(f"{d}/val_tokens.json"))
        rows = np.array(json.load(open(BP.ROOT + "/data/splits.json"))["val" if a.part == "val" else "test"])[:: a.stride]
    if a.n:
        rows = rows[: a.n]
    pca = np.load(f"{HOME}/exp_gcar_b123/pca_gcar.npz")
    dm, db = torch.from_numpy(pca["dec_mean"]).cuda(), torch.from_numpy(pca["dec_basis"]).cuda()
    img_params = OmegaConf.to_container(cfg.data.img_params, resolve=True)
    bev_conf = OmegaConf.to_container(cfg.data.data_config.bev_aug_conf, resolve=True)
    pre = a.part
    out_dir = f"{HOME}/exp_gcar_b123/pert/{a.pert}"
    os.makedirs(out_dir, exist_ok=True)
    N = len(rows)
    resume = bool(a.resume) and os.path.exists(f"{out_dir}/{pre}_logit7.npy") and os.path.exists(f"{out_dir}/{pre}_rows.npy") and (np.load(f"{out_dir}/{pre}_rows.npy") == rows).all()
    if resume:  # nach einem Abbruch (Speichermangel): nur die noch leeren Zeilen nachrechnen
        dec = np.load(f"{out_dir}/{pre}_dec32.npy", mmap_mode="r+")
        lg = np.load(f"{out_dir}/{pre}_logit7.npy", mmap_mode="r+")
        todo = [k for k in range(N) if float(np.abs(np.asarray(lg[k], np.float32)).max()) == 0.0]
        print(f"{a.pert}: resuming, {len(todo)} of {N} rows are missing", flush=True)
    else:
        dec = np.lib.format.open_memmap(f"{out_dir}/{pre}_dec32.npy", mode="w+", dtype=np.float16, shape=(N, 32, 200, 200))
        lg = np.lib.format.open_memmap(f"{out_dir}/{pre}_logit7.npy", mode="w+", dtype=np.float16, shape=(N, 7, 200, 200))
        np.save(f"{out_dir}/{pre}_rows.npy", rows)
        todo = list(range(N))
    ds = PertData("b123", [tokens[int(rows[k])] for k in todo], img_params, bev_conf, a.pert)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=a.workers, collate_fn=G.collate)
    t0 = time.time()
    for j, b in enumerate(dl):
        k = todo[j]
        try:
            o = G.run(module, G.to_dev(b))
        except RuntimeError:
            o = None
            for alt in (1, 2, 3, 4):
                try:
                    o = G.run(module, G.to_dev(G.collate([ds.__getitem__(j, alt=alt)])))
                    print(f"{a.pert}: Zeile {k}: Wiederholung mit {3 + 3 * alt} Radarpunkten", flush=True)
                    break
                except RuntimeError:
                    continue
            if o is None:
                raise
        dec[k] = G.orient_t(G.project(o["dec"][0].float(), dm, db)).half().cpu().numpy()
        lg[k] = G.orient_t(o["logits"][0].float()).half().cpu().numpy()
        if (j + 1) % 100 == 0 or j + 1 == len(todo):
            dec.flush()
            lg.flush()
            print(f"{a.pert} {j + 1}/{len(todo)} ({time.time() - t0:.0f} s, {(time.time() - t0) / (j + 1):.2f} s/Sample)", flush=True)
    open(f"{out_dir}/DONE" + ("" if a.part == "test" else "_" + a.part), "w").write("ok")
    print("FERTIG", a.pert, f"{time.time() - t0:.0f} s", flush=True)
    sys.stdout.flush()
    os._exit(0)  # Beenden ohne Aufraeumen (der normale Abbau blieb haengen)


if __name__ == "__main__":
    main()
