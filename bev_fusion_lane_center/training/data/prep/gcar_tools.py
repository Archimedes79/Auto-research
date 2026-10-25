"""GaussianCaR (ICRA 2026, Kamera+Radar, BEV-Karten-Segmentierung) als eingefrorenes Fundament: Laden, Daten im Fahrzeugrahmen unseres Solls, Probelauf, Extraktion.

Rahmen: das Netz rechnet im Fahrzeug-System (flach, nur Gierwinkel) zum Zeitpunkt des Lidar-Sweeps. Hier wird stattdessen die Hinterachse zum Zeitpunkt von CAM_FRONT
verwendet (wie GT_FRAME=ego der Kamera-Radar-Experimente): Kamera-Extrinsiken und Radar-Punkte werden in diesen Rahmen gelegt; es werden keine Lidar-Dateien gelesen.
Aufruf (Umgebung 'gcar'):
  gcar_tools.py probe   --split mini --n 8              Modell + Mini-Samples, Formen, Zeit, Orientierungspruefung gegen das Soll
  gcar_tools.py extract --split mini|b123 ...            Merkmale fuer alle Samples (siehe unten)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.expanduser("~/work/gaussiancar")
sys.path.insert(0, REPO)
HOME = os.path.expanduser("~")
CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
SPLITS = {
    "mini": dict(root=f"{HOME}/data/nuscenes", version="v1.0-mini", exp=f"{HOME}/exp_crn_mini"),
    "b123": dict(root=f"{HOME}/data/nuscenes_full", version="v1.0-trainval", exp=f"{HOME}/exp_crn_b123"),
}


# ------------------------------------------------------------------ Modell


def load_module(ckpt, task="map", device="cuda"):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from gaussiancar.utils.config import register_new_resolvers

    register_new_resolvers()
    with initialize_config_dir(config_dir=f"{REPO}/configs", version_base="1.3"):
        cfg = compose(config_name="inference", overrides=[f"task={task}", "module.model.image_encoder.pretrained=false", "debug_mode=false"])
    # PTv3 ohne flash-attn (gleiche Gewichte, Aufmerksamkeit in fp32 statt fp16)
    import gaussiancar.ops.ptv3.model as P

    orig = P.PointTransformerV3.__init__

    def init(self, *a, **k):
        k["enable_flash"] = False
        orig(self, *a, **k)

    P.PointTransformerV3.__init__ = init
    module = instantiate(cfg.module, cfg=cfg)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]  # nur Tensoren/Grundtypen, keine Objekte aus der Pickle-Datei
    res = module.load_state_dict(state, strict=False)
    print(f"Gewichte geladen: {len(state)} Eintraege, fehlend {len(res.missing_keys)}, unerwartet {len(res.unexpected_keys)}", flush=True)
    if res.missing_keys:
        print("  fehlend (Auszug):", res.missing_keys[:6])
    if res.unexpected_keys:
        print("  unerwartet (Auszug):", res.unexpected_keys[:6])
    module.to(device).eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module, cfg


# ------------------------------------------------------------------ Daten


def get_pose(rotation, translation, inv=False, flat=False):
    from gaussiancar.data.common import get_pose as gp

    return gp(rotation, translation, inv=inv, flat=flat)


class GcarData(torch.utils.data.Dataset):
    """Ein Sample = 6 Kamerabilder (448 x 800), Kamera-Projektionsmatrizen, 7 Radar-Sweeps; BEV-Rahmen = Hinterachse zum Zeitpunkt von CAM_FRONT."""

    def __init__(self, split, tokens, img_params, bev_conf, nusc=None):
        from nuscenes.nuscenes import NuScenes

        from gaussiancar.data.transforms.augmentations import RandomTransformBev, RandomTransformImage
        from gaussiancar.data.transforms.nuscenes.loading import ImageDataLoader, RadarDataLoader

        s = SPLITS[split]
        self.root = s["root"]
        self.tokens = tokens
        self.nusc = nusc or NuScenes(version=s["version"], dataroot=self.root, verbose=False)
        self.tf = [
            RandomTransformImage(img_params, training=False, orig_img_size=(img_params["W"], img_params["H"])),
            RandomTransformBev(bev_conf, training=False),
            ImageDataLoader(img_params, self.root),
            RadarDataLoader(self.nusc, num_sweeps=7, lidar_name="CAM_FRONT"),
        ]

    def __len__(self):
        return len(self.tokens)

    def sample_dict(self, token):
        n = self.nusc
        rec = n.get("sample", token)
        ref = n.get("sample_data", rec["data"]["CAM_FRONT"])
        egoref = n.get("ego_pose", ref["ego_pose_token"])
        world_from_ego = get_pose(egoref["rotation"], egoref["translation"], flat=True)
        images, intr, extr = [], [], []
        for ch in CAMERAS:
            sd = n.get("sample_data", rec["data"][ch])
            egocam = n.get("ego_pose", sd["ego_pose_token"])
            cs = n.get("calibrated_sensor", sd["calibrated_sensor_token"])
            cam_from_egocam = get_pose(cs["rotation"], cs["translation"], inv=True)
            egocam_from_world = get_pose(egocam["rotation"], egocam["translation"], inv=True)
            extr.append((cam_from_egocam @ egocam_from_world @ world_from_ego).tolist())
            intr.append(cs["camera_intrinsic"])
            images.append(os.path.relpath(n.get_sample_data_path(sd["token"]), n.dataroot))
        return dict(token=token, images=images, intrinsics=intr, extrinsics=extr)

    def __getitem__(self, i):
        d = self.sample_dict(self.tokens[i])
        for t in self.tf:
            d = t(d)
        return dict(image=d["image"], lidar2img=d["lidar2img"], radar_points=d["radar_points"], index=i)


def collate(batch):
    return dict(image=torch.stack([b["image"] for b in batch]), lidar2img=torch.stack([b["lidar2img"] for b in batch]),
                radar_points=[b["radar_points"] for b in batch], index=[b["index"] for b in batch])


def to_dev(batch, dev="cuda"):
    return dict(image=batch["image"].to(dev), lidar2img=batch["lidar2img"].to(dev), radar_points=[r.to(dev) for r in batch["radar_points"]])


@torch.no_grad()
def run(module, batch):
    """-> Merkmale: dec (256 x 200 x 200, Eingang des Karten-Kopfes), fused (128 x ... fusionierte BEV-Merkmale, feinste Stufe), logits (7 Karten-Schichten)."""
    m = module.model
    cam = m.forward_features_camera(batch)
    rad = m.forward_features_radar(batch)
    cam_bev, _ = m.gs_render_image(cam["features"], cam["centers"], cam["covariances"], cam["opacities"])
    rad_bev, _ = m.gs_render_radar(rad["features"], rad["centers"], rad["covariances"], rad["opacities"])
    fused = m.fuser(cam_bev, rad_bev)
    dec = m.decoder(fused)
    out = m.head(dec)
    keys = ["lane", "road_segment", "road_divider", "lane_divider", "ped_crossing", "walkway", "carpark_area"]
    logits = torch.cat([out[k] for k in keys], 1)
    f0 = fused[0] if isinstance(fused, (list, tuple)) else fused
    return dict(dec=dec, fused=f0, logits=logits, cam=cam_bev, rad=rad_bev)


# ------------------------------------------------------------------ Orientierung gegen das Soll

D4 = [lambda a: a, lambda a: a[..., ::-1, :], lambda a: a[..., :, ::-1], lambda a: a[..., ::-1, ::-1],
      lambda a: np.swapaxes(a, -1, -2), lambda a: np.swapaxes(a, -1, -2)[..., ::-1, :], lambda a: np.swapaxes(a, -1, -2)[..., :, ::-1], lambda a: np.swapaxes(a, -1, -2)[..., ::-1, ::-1]]
D4_NAME = ["identisch", "Zeilen spiegeln", "Spalten spiegeln", "beide spiegeln", "transponiert", "transp.+Zeilen", "transp.+Spalten", "transp.+beide"]


def orientation_scores(prob, gt):
    """prob [N,200,200] (z. B. Karten-Schicht 'lane' / 'road_segment' als Wahrscheinlichkeit), gt [N,200,200] bool (Soll-Mittellinien): mittlere Wahrscheinlichkeit auf den Soll-Pixeln je Spiegelung/Transposition."""
    out = []
    for f in D4:
        p = np.ascontiguousarray(f(prob))
        out.append(float((p * gt).sum() / max(gt.sum(), 1)))
    return out


def orient_t(x):
    """Netz-Ausgabe [..., H, W] -> Layout unseres Solls (Zeile = x nach rechts, Spalte = y nach vorn): D4-Nr. 6 = transponieren, dann Spalten spiegeln."""
    return x.transpose(-1, -2).flip(-1)


def feat_dir(split):
    return f"{HOME}/exp_gcar_{split}/feats"


def project(x, mean, basis):
    """x [C,H,W] (GPU) -> PCA-Merkmale [K,H,W]: (x - mean) @ basis je Pixel."""
    return torch.einsum("chw,ck->khw", x - mean[:, None, None], basis)


def prepare(split):
    """Leere fp16-Memmaps train/val x (dec32, fused32, logit7) anlegen (falls nicht vorhanden) und die Token-Listen kopieren."""
    s = SPLITS[split]
    d = feat_dir(split)
    os.makedirs(d, exist_ok=True)
    for part in ("train", "val"):
        toks = json.load(open(f"{s['exp']}/feats/{part}_tokens.json"))
        json.dump(toks, open(f"{d}/{part}_tokens.json", "w"))
        for g, c in (("dec32", 32), ("fused32", 32), ("logit7", 7)):
            f = f"{d}/{part}_{g}.npy"
            if not os.path.exists(f):
                np.lib.format.open_memmap(f, mode="w+", dtype=np.float16, shape=(len(toks), c, 200, 200))
                print("angelegt:", f, flush=True)


def extract(a, module, cfg):
    """Merkmale fuer die Zeilen range(start, end, stride) des Teils: dec (256 -> PCA 32), fused (128 -> PCA 32), 7 Karten-Logits; orientiert wie unser Soll, fp16-Memmaps.
    --check: nichts schreiben, nur gegen vorhandene Zeilen vergleichen."""
    from omegaconf import OmegaConf

    s = SPLITS[a.split]
    d = feat_dir(a.split)
    tokens = json.load(open(f"{d}/{a.part}_tokens.json"))
    end = a.end or len(tokens)
    gidx = list(range(a.start, end, a.stride))
    if a.n and a.cmd == "extract" and a.check:
        gidx = gidx[: a.n]
    pca = np.load(f"{HOME}/exp_gcar_{a.split}/pca_gcar.npz")
    dm, db = torch.from_numpy(pca["dec_mean"]).cuda(), torch.from_numpy(pca["dec_basis"]).cuda()
    fm, fb = torch.from_numpy(pca["fused_mean"]).cuda(), torch.from_numpy(pca["fused_basis"]).cuda()
    img_params = OmegaConf.to_container(cfg.data.img_params, resolve=True)
    bev_conf = OmegaConf.to_container(cfg.data.data_config.bev_aug_conf, resolve=True)
    t0 = time.time()
    ds = GcarData(a.split, [tokens[i] for i in gidx], img_params, bev_conf)
    print(f"nuScenes geladen ({time.time() - t0:.0f} s), Samples {len(ds)}", flush=True)
    mm = {g: np.load(f"{d}/{a.part}_{g}.npy", mmap_mode="r" if a.check else "r+") for g in ("dec32", "fused32", "logit7")}
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=a.workers, collate_fn=collate)
    t0 = time.time()
    errs = {g: [] for g in mm}
    for k, b in enumerate(dl):
        o = run(module, to_dev(b))
        row = gidx[k]
        out = {
            "dec32": orient_t(project(o["dec"][0].float(), dm, db)),
            "fused32": orient_t(project(o["fused"][0].float(), fm, fb)),
            "logit7": orient_t(o["logits"][0].float()),
        }
        for g, v in out.items():
            v = v.half().cpu().numpy()
            if a.check:
                ref = np.asarray(mm[g][row], np.float32)
                errs[g].append((float(np.abs(v.astype(np.float32) - ref).max()), float(np.corrcoef(v.astype(np.float32).ravel(), ref.ravel())[0, 1])))
            else:
                mm[g][row] = v
        if (k + 1) % 100 == 0 or k + 1 == len(gidx):
            if not a.check:
                for g in mm:
                    mm[g].flush()
            dt = time.time() - t0
            print(f"{a.split} {a.part} [{a.start}:{end}] {k + 1}/{len(gidx)} ({dt:.0f} s, {dt / (k + 1):.2f} s/Sample)", flush=True)
    if a.check:
        for g, e in errs.items():
            print(f"Pruefung {g}: max |Differenz| je Zeile {[round(x[0], 3) for x in e]}, Korrelation {[round(x[1], 5) for x in e]}")
        return
    open(f"{d}/{a.part}_{a.start}_{end}_s{a.stride}.done", "w").write("ok")
    print(f"FERTIG {a.split} {a.part} {a.start} {end} {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "prepare", "extract"])
    ap.add_argument("--split", default="mini", choices=list(SPLITS))
    ap.add_argument("--part", default="val", choices=["train", "val"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ckpt", default=f"{HOME}/work/gaussiancar_ckpt/nuscenes_map.ckpt")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--check", type=int, default=0, help="extract: nichts schreiben, die ersten n Zeilen gegen vorhandene Merkmale vergleichen")
    a = ap.parse_args()
    s = SPLITS[a.split]
    if a.cmd == "prepare":
        prepare(a.split)
        return
    tokens = json.load(open(f"{s['exp']}/feats/{a.part}_tokens.json"))
    module, cfg = load_module(a.ckpt)
    if a.cmd == "extract":
        extract(a, module, cfg)
        return
    from omegaconf import OmegaConf

    img_params = OmegaConf.to_container(cfg.data.img_params, resolve=True)
    bev_conf = OmegaConf.to_container(cfg.data.data_config.bev_aug_conf, resolve=True)
    t0 = time.time()
    ds = GcarData(a.split, tokens[: a.n], img_params, bev_conf)
    print(f"nuScenes geladen ({time.time() - t0:.0f} s), Samples {len(ds)}", flush=True)
    probe(module, ds, a, s)


def probe(module, ds, a, s):
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=a.workers, collate_fn=collate)
    gt = np.load(f"{s['exp']}/feats/{a.part}_gt_raster.npy", mmap_mode="r")
    probs, raws = [], []
    t0 = time.time()
    for k, b in enumerate(dl):
        t1 = time.time()
        o = run(module, to_dev(b))
        torch.cuda.synchronize()
        if k == 0:
            print({kk: tuple(v.shape) for kk, v in o.items()}, "radar_points", tuple(b["radar_points"][0].shape), flush=True)
            print("GPU-Speicher max (GB):", torch.cuda.max_memory_allocated() / 1e9, flush=True)
        pr = torch.sigmoid(o["logits"][0]).float().cpu().numpy()  # [7,200,200]
        probs.append(pr)
        if k < 8:  # Rohmerkmale nur fuer die ersten Samples speichern (Platz)
            raws.append(dict(dec=o["dec"][0].half().cpu().numpy(), fused=o["fused"][0].half().cpu().numpy()))
        print(f"Sample {b['index'][0]}: {time.time() - t1:.2f} s (mit Laden insgesamt {time.time() - t0:.1f} s)", flush=True)
    P = np.stack(probs)
    n = len(P)
    G = np.asarray(gt[:n]).astype(bool)
    names = ["lane", "road_segment", "road_divider", "lane_divider", "ped_crossing", "walkway", "carpark_area"]
    print("\nOrientierung: mittlere Wahrscheinlichkeit der Schicht auf den Soll-Mittellinien-Pixeln (hoeher = besser)")
    best = {}
    for li in (0, 1):
        sc = orientation_scores(P[:, li], G)
        print(f"  {names[li]:13s}", " ".join(f"{x:.3f}" for x in sc), "->", D4_NAME[int(np.argmax(sc))])
        best[names[li]] = int(np.argmax(sc))
    print("Mittel der Schichten ueber alle Pixel:", {nm: round(float(P[:, i].mean()), 3) for i, nm in enumerate(names)})
    os.makedirs(f"{HOME}/work/gcar_probe", exist_ok=True)
    np.savez_compressed(f"{HOME}/work/gcar_probe/probe_{a.split}_{a.part}.npz", probs=P.astype(np.float16), dec=np.stack([r["dec"] for r in raws]), fused=np.stack([r["fused"] for r in raws]))
    print("gespeichert:", f"{HOME}/work/gcar_probe/probe_{a.split}_{a.part}.npz")


if __name__ == "__main__":
    main()
