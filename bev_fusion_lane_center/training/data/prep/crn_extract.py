"""BEV-Features des Kamera-Radar-Netzes CRN (eingefroren) extrahieren: gleiche Ablage wie bei BEVFusion.

Ausgabe in --out: {split}_feats32.npy  fp16 [S, 32, 128, 128]  (PCA 128 -> 32 Kanaele, Zeilen so gedreht, dass das Layout dem
                  bisherigen Lidar-System entspricht: Zeile = x nach rechts, Spalte = y nach vorn),
                  {split}_tokens.json (Reihenfolge = Zeitstempel, aus --tokens uebernommen oder aus den Infos berechnet),
                  pca32_basis_crn.npz.
Frame: Das BEV-Gitter von CRN liegt im Ego-System (Hinterachse, x vorn, y links) zum Kamera-Zeitpunkt; Zeile = y_ego, Spalte = x_ego.
       Mit x_L = -y_ego und y_L = x_ego ist Zeile_L = 127 - Zeile_CRN. (centerline_gt.py mit GT_FRAME=ego liefert dazu passende Soll-Daten.)
Es werden weder Lidar-Punktwolken noch Tiefen-Labels (aus Lidar) gelesen: return_depth=False.

Aufruf (Umgebung crn, im Ordner ~/work/CRN):
    python crn_extract.py --root ~/data/nuscenes_full --infos train:nuscenes_infos_train_crn.pkl,val:nuscenes_infos_val_crn.pkl \
        --tokens ~/exp_b123/feats --out ~/exp_crn_b123/feats --ckpt ckpt/CRN_r50_256x704_128x128_4key.pth --exp r50
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

CRN = os.path.expanduser("~/work/CRN")
sys.path.insert(0, CRN)
os.chdir(CRN)

from functools import partial  # noqa: E402

from datasets.nusc_det_dataset import NuscDatasetRadarDet, collate_fn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--infos", required=True, help="split:pkl,split:pkl (pkl relativ zu --root)")
ap.add_argument("--tokens", default="", help="Ordner mit {split}_tokens.json (Reihenfolge der Samples); sonst alle Infos nach Zeitstempel")
ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--exp", default="r50", choices=["r18", "r50"])
ap.add_argument("--pca_samples", type=int, default=300)
ap.add_argument("--k", type=int, default=32)
ap.add_argument("--source", default="fused", choices=["fused", "neck", "img", "pts", "pre"],
                help="fused: Ausgabe der Fusion (128 Kanaele); neck: 256 Kanaele hinter dem BEV-Hals; img/pts/pre: vor der Fusion, nur aktueller Frame (Kamera 80 / Radar 80 / beide 160)")
ap.add_argument("--tag", default="", help="Suffix fuer die Ausgabedateien (z. B. n -> train_feats32n.npy, pca32n_basis_crn.npz)")
ap.add_argument("--limit", type=int, default=0, help="nur die ersten N Samples je Split (Test)")
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--workers", type=int, default=4)
args = ap.parse_args()
root, out = os.path.expanduser(args.root), os.path.expanduser(args.out)
os.makedirs(out, exist_ok=True)

if args.exp == "r50":
    from exps.det.CRN_r50_256x704_128x128_4key import CRNLightningModel
else:
    from exps.det.CRN_r18_256x704_128x128_4key import CRNLightningModel

lm = CRNLightningModel(gpus=1, data_root=root, batch_size_per_device=args.bs)
sd = torch.load(os.path.expanduser(args.ckpt), map_location="cpu")
sd = sd.get("state_dict", sd)
missing = lm.load_state_dict(sd, strict=False)
print("Gewichte geladen; fehlend:", len(missing.missing_keys), "unerwartet:", len(missing.unexpected_keys), flush=True)
lm.return_depth = False
model = lm.model.cuda().eval()
CAP = {}
model.head.neck.register_forward_hook(lambda m, i, o: CAP.__setitem__("n", o[0] if isinstance(o, (list, tuple)) else o))


def make_loader(pkl, tokens):
    ds = NuscDatasetRadarDet(
        ida_aug_conf=lm.ida_aug_conf, bda_aug_conf=lm.bda_aug_conf, rda_aug_conf=lm.rda_aug_conf, img_backbone_conf=lm.backbone_img_conf,
        classes=lm.class_names, data_root=root, info_paths=os.path.join(root, pkl), is_train=False, img_conf=lm.img_conf,
        load_interval=1, num_sweeps=lm.num_sweeps, sweep_idxes=lm.sweep_idxes, key_idxes=lm.key_idxes, return_image=True,
        return_depth=False, return_radar_pv=True, remove_z_axis=lm.remove_z_axis, radar_pv_path="radar_pv_filter")
    by_tok = {i["sample_token"]: k for k, i in enumerate(ds.infos)}
    if tokens is None:
        tokens = [ds.infos[k]["sample_token"] for k in sorted(range(len(ds.infos)), key=lambda k: ds.infos[k]["timestamp"])]
    tokens = [t for t in tokens if t in by_tok]
    if args.limit:
        tokens = tokens[: args.limit]
    sub = torch.utils.data.Subset(ds, [by_tok[t] for t in tokens])
    dl = torch.utils.data.DataLoader(sub, batch_size=args.bs, num_workers=args.workers, shuffle=False,
                                     collate_fn=partial(collate_fn, is_return_image=True, is_return_depth=False, is_return_radar_pv=True))
    return tokens, dl


@torch.no_grad()
def fused_feats(batch):
    sweep_imgs, mats, _, _, _, _, _, pts_pv = batch
    sweep_imgs = sweep_imgs.cuda()
    mats = {k: v.cuda() for k, v in mats.items()}
    with torch.cuda.amp.autocast():  # der CRN-Code ist fuer fp16-Autocast geschrieben (Trainer precision=16)
        ptss_context, ptss_occupancy, _ = model.backbone_pts(pts_pv.cuda(), times=None)
        feats, _ = model.backbone_img(sweep_imgs, mats, ptss_context, ptss_occupancy, times=None)
        if args.source in ("img", "pts", "pre"):  # vor der Fusion, nur aktueller Frame: Kamera-BEV (80), Radar-BEV (80) oder beides (160)
            f0 = feats[:, 0].float()
            return {"img": f0[:, :80], "pts": f0[:, 80:160], "pre": f0[:, :160]}[args.source]
        fused, _ = model.fuser(feats, times=None)
        if args.source == "neck":  # 256 Kanaele hinter dem BEV-Hals der Detektionsköpfe (reichere, aufgabennahe Merkmale)
            model.head(fused, times=None)
            return CAP["n"].float()
    return fused.float()  # [B, 128, 128(y_ego), 128(x_ego)]


splits = dict(p.split(":") for p in args.infos.split(","))
toks_given = {s: json.load(open(os.path.join(os.path.expanduser(args.tokens), f"{s}_tokens.json"))) if args.tokens else None for s in splits}

# 1) PCA-Basis aus Trainings-Samples (gleichmaessig verteilt)
basis_path = os.path.join(out, f"pca{args.k}{args.tag}_basis_crn.npz")
if os.path.exists(basis_path):
    z = np.load(basis_path)
    mu, W = z["mu"], z["W"]
else:
    tok_all = toks_given["train"] if toks_given["train"] is not None else make_loader(splits["train"], None)[0]
    pick = [tok_all[i] for i in np.linspace(0, len(tok_all) - 1, min(args.pca_samples, len(tok_all))).astype(int)]
    tokens, dl = make_loader(splits["train"], pick)  # nur die ausgewaehlten Samples rechnen (frueher: alle durchlaufen)
    X = []
    rng = np.random.RandomState(0)
    for batch in dl:
        f = fused_feats(batch).float()
        for b in range(f.shape[0]):
            v = f[b].reshape(f.shape[1], -1).T  # [16384, C]
            X.append(v[torch.from_numpy(rng.choice(v.shape[0], 3000, replace=False)).cuda()].cpu().numpy())
    X = np.concatenate(X)
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    W = Vt[: args.k].T.astype(np.float32)  # [128, k]
    ev = (S[: args.k] ** 2).sum() / (S**2).sum()
    print(f"PCA {X.shape[1]} -> {args.k} Kanaele, erklaerte Varianz {ev:.3f} (aus {len(X)} Pixelvektoren)", flush=True)
    np.savez(basis_path, mu=mu.astype(np.float32), W=W, explained=ev)
mu_t, W_t = torch.from_numpy(mu).cuda(), torch.from_numpy(W).cuda()

# 2) alle Splits extrahieren
for split, pkl in splits.items():
    tokens, dl = make_loader(pkl, toks_given[split])
    arr = np.lib.format.open_memmap(os.path.join(out, f"{split}_feats{args.k}{args.tag}.npy"), mode="w+", dtype=np.float16, shape=(len(tokens), args.k, 128, 128))
    pos = 0
    for batch in dl:
        f = fused_feats(batch).float()
        z = torch.einsum("bchw,ck->bkhw", f - mu_t.view(1, -1, 1, 1), W_t)
        z = z.flip(2)  # Zeile_L = 127 - Zeile_CRN (x_L = -y_ego)
        arr[pos: pos + z.shape[0]] = z.half().cpu().numpy()
        pos += z.shape[0]
        if (pos // args.bs) % 50 == 0:
            print(f"{split}: {pos}/{len(tokens)}", flush=True)
    arr.flush()
    json.dump(tokens, open(os.path.join(out, f"{split}_tokens.json"), "w"))
    print(f"{split}: {len(tokens)} Samples -> {out}", flush=True)
print("EXTRAKTION FERTIG")
