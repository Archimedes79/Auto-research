"""Dichter Kopf fuer die Spurmitte (E27 ff.): kleines U-Netz auf eingefrorenen BEV-Merkmalen -> Mittellinien-Heatmap (weich, Gauss um die Soll-Linie), optional Richtung.
Auswertung: Schwelle -> Ausduennen -> Punkte -> Punkt-F1/1:1-F1 bei 0,5/1/1,5 m (Geometrie wie E19) fuer TRAIN_EVAL, VAL, TEST.
Aufruf (Umgebung gcar):  dense_head.py --name e27_gcar --fg gcar --feats dec32,logit7 --steps 6000 --loss fp --lam 3
  --fg crn: CRN-Merkmale (PCA 32, 128 x 128, auf 200 x 200 hochgerechnet), --fg gcar: GaussianCaR-Gruppen aus --feats (dec32, fused32, logit7).
Ergebnisse: out/loop2/<name>/{model.pt, metrics.json, log.txt}
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402

HOME = os.path.expanduser("~")
OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
SIGMA_M = 0.75  # Gauss-Breite des Soll-Heatmaps (m)


# ------------------------------------------------------------------ Daten


class Data(torch.utils.data.Dataset):
    """Merkmale und weiches Soll aus Memory-Maps; indices: Zeilen der Token-Liste (train bzw. val)."""

    def __init__(self, fg, groups, part, indices, split="b123", target=True, with_dir=False):
        self.fg, self.groups, self.part, self.idx, self.split, self.target = fg, groups, part, np.asarray(indices), split, target
        self.with_dir = with_dir and target
        self.mm = None

    def _open(self):
        e = f"{HOME}/exp_gcar_{self.split}/feats" if self.fg in ("gcar", "both") else f"{HOME}/exp_crn_{self.split}/feats"
        if self.fg in ("gcar", "both"):
            self.mm = [np.load(f"{e}/{self.part}_{g}.npy", mmap_mode="r") for g in self.groups]
            if self.fg == "both":  # Loop 4 / E52: Kombination zweier eingefrorener Grundlagen: GaussianCaR-Gruppen + CRN-Merkmale (PCA 32, 128 x 128, auf 200 x 200 hochgerechnet)
                self.mm.append(np.load(f"{HOME}/exp_crn_{self.split}/feats/{self.part}_feats32.npy", mmap_mode="r"))
        else:
            self.mm = [np.load(f"{e}/{self.part}_feats32.npy", mmap_mode="r")]
        if self.target:
            self.gt = np.load(f"{HOME}/exp_crn_{self.split}/feats/{self.part}_gt_dist_exact.npy", mmap_mode="r")
        if self.with_dir:
            self.dr = np.load(f"{HOME}/exp_gcar_{self.split}/dir_{self.part}.npy", mmap_mode="r")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        if self.mm is None:
            self._open()
        i = int(self.idx[k])
        xs = [torch.from_numpy(np.asarray(m[i], np.float32)) for m in self.mm]
        if self.fg == "both":
            xs[-1] = F.interpolate(xs[-1][None], size=(200, 200), mode="bilinear", align_corners=False)[0]
        x = torch.cat(xs, 0)
        if self.fg == "crn":
            x = F.interpolate(x[None], size=(200, 200), mode="bilinear", align_corners=False)[0]
        if not self.target:
            return x, i
        d = torch.from_numpy(np.asarray(self.gt[i], np.float32)) * 0.5  # Pixel -> Meter, bei 4 m gekappt
        y = torch.exp(-(d ** 2) / (2 * SIGMA_M ** 2))
        if self.with_dir:
            return x, y, torch.from_numpy(np.asarray(self.dr[i], np.float32)), i
        return x, y, i


def channel_stats(ds, n=200):
    sel = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    xs = torch.stack([ds[int(k)][0] for k in sel])
    return xs.mean((0, 2, 3)), xs.std((0, 2, 3)).clamp_min(1e-4)


# ------------------------------------------------------------------ Modell


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, cin, base=48, out=1, drop=0.0):
        super().__init__()
        c1, c2, c3, c4 = base, 2 * base, 4 * base, 4 * base
        self.stem = block(cin, c1)
        self.d1 = nn.Sequential(nn.MaxPool2d(2), block(c1, c2))
        self.d2 = nn.Sequential(nn.MaxPool2d(2), block(c2, c3))
        self.d3 = nn.Sequential(nn.MaxPool2d(2), block(c3, c4))
        self.u2 = block(c4 + c3, c3)
        self.u1 = block(c3 + c2, c2)
        self.u0 = block(c2 + c1, c1)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.head = nn.Conv2d(c1, out, 1)

    def forward(self, x):
        x0 = self.stem(self.drop(x))
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        up = lambda a, b: torch.cat([F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1)
        y = self.u2(up(x3, x2))
        y = self.u1(up(y, x1))
        y = self.u0(up(y, x0))
        return self.head(y)


# ------------------------------------------------------------------ Verlust


def loss_fn(logit, y, lam=1.0, wpos=1.0, gamma=2.0):
    """Fokale gewichtete BCE mit weichen Zielen. lam = Gewicht der Falschalarm-Seite (Negative, 1-y), wpos = Gewicht der Verpasst-Seite (y)."""
    p = torch.sigmoid(logit)
    pos = wpos * y * ((1 - p) ** gamma) * F.softplus(-logit)
    neg = lam * (1 - y) * (p ** gamma) * F.softplus(logit)
    return (pos + neg).mean() * 100.0


def faug(x, p=0.5):
    """Loop 5 / E65: Augmentierung im Merkmalsraum gegen Sensorausfaelle (x: normierte Merkmale [B,C,H,W], H = lateral, W = vorwaerts, 0,5 m). Je Sample mit Wahrscheinlichkeit p eine zufaellige Stoerung:
    0 Sektor-Ausfall (Keil 40-120 Grad, Merkmale = Kanalmittel), 1 Kanal-Verstaerkung/-Verschiebung (Beleuchtung), 2 Rauschen, 3 Unschaerfe (5x5), 4 Karten-Logits (letzte 7 Kanaele) aus."""
    B, C, H, W = x.shape
    xs = (torch.arange(H, device=x.device).float() - H / 2 + 0.5) * 0.5
    ys = (torch.arange(W, device=x.device).float() - W / 2 + 0.5) * 0.5
    XX, YY = torch.meshgrid(xs, ys, indexing="ij")
    ang = torch.atan2(XX, YY)
    out = x.clone()
    for b in range(B):
        if float(torch.rand(1)) >= p:
            continue
        kind = int(torch.randint(0, 5, (1,)))
        if kind == 0:
            c = float(torch.rand(1)) * 2 * np.pi - np.pi
            w = np.deg2rad(40 + 80 * float(torch.rand(1)))
            m = (torch.remainder(ang - c + np.pi, 2 * np.pi) - np.pi).abs() < w / 2
            out[b][:, m] = 0.0
        elif kind == 1:
            out[b] = out[b] * (1 + 0.3 * torch.randn(C, 1, 1, device=x.device)) + 0.3 * torch.randn(C, 1, 1, device=x.device)
        elif kind == 2:
            out[b] = out[b] + 0.4 * torch.randn_like(out[b])
        elif kind == 3:
            out[b] = F.avg_pool2d(out[b][None], 5, 1, 2)[0]
        else:
            out[b, -7:] = 0.0
    return out


def dir_loss(pred, gt):
    """Richtung (dx, dy) nur auf Spurmitten (|gt| > 0.5): 1 - cos zwischen normierter Vorhersage und Soll."""
    m = (gt.pow(2).sum(1) > 0.25).float()
    pn = pred / pred.pow(2).sum(1, keepdim=True).sqrt().clamp_min(1e-3)
    cos = (pn * gt).sum(1)
    return ((1 - cos) * m).sum() / m.sum().clamp_min(1.0)


# ------------------------------------------------------------------ Auswertung


from grid_eval import eval_probs, geom_counts, load_gt_points, skeleton_points, thin_batch  # noqa: E402,F401  (moved to evaluation/lane_grid)


@torch.no_grad()
def predict(net, ds, mean, std, bs=16, dev="cuda", workers=4, want_dir=False, inorm=0):
    net.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers)
    outs, dirs, ids = [], [], []
    for b in dl:
        x = b[0].to(dev)
        x = (x - mean[None, :, None, None]) / std[None, :, None, None]
        if inorm:
            x = (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 1e-2)
        with torch.autocast("cuda", dtype=torch.float16):
            o = net(x).float()
        outs.append(torch.sigmoid(o[:, 0]).half().cpu())
        if o.shape[1] > 1:
            dirs.append(o[:, 1:3].half().cpu())
        ids.append(b[-1])
    if want_dir:
        return torch.cat(outs), torch.cat(ids).numpy(), (torch.cat(dirs) if dirs else None)
    return torch.cat(outs), torch.cat(ids).numpy()


# ------------------------------------------------------------------ Hauptprogramm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--fg", default="gcar", choices=["gcar", "crn", "both"])
    ap.add_argument("--feats", default="dec32,logit7")
    ap.add_argument("--split", default="b123", choices=["b123", "mini"])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=1.0, help="Gewicht der Falschalarm-Seite")
    ap.add_argument("--wpos", type=float, default=1.0, help="Gewicht der Verpasst-Seite")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--scene_sub", type=int, default=1, help="Kontrolle: nur Trainings-Szenen mit Nummer % k == 0 verwenden (k=19: 11 Szenen)")
    ap.add_argument("--train_n", type=int, default=0, help="nur die ersten n Trainings-Samples (Obergrenzen-Test)")
    ap.add_argument("--train_stride", type=int, default=1, help="nur jedes k-te Trainings-Sample (gleichmaessig ueber die Szenen)")
    ap.add_argument("--dir", type=float, default=0.0, help="Gewicht des Richtungsverlusts (0 = aus; sonst +2 Ausgabekanaele)")
    ap.add_argument("--taus", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--save_prob", type=int, default=1, help="volle Wahrscheinlichkeitskarten der Auswertungsmengen speichern")
    ap.add_argument("--amp", type=int, default=1, help="gemischte Genauigkeit (fp16) im Training")
    ap.add_argument("--train_part", default="train", choices=["train", "val"], help="'val': Training nur auf VAL-A (Szenen mit gerader Nr.; vom Grundlage-Netz ungesehen), Auswertung auf VAL-B + TEST")
    ap.add_argument("--fold", type=int, default=-1, help="Kreuz-Anpassung (Szenen-Folds auf TRAIN): Training auf Szenen mit (Szenen-Nr. % nfolds) != fold, Auswertungsmenge 'oof' = die zurueckgehaltenen Szenen")
    ap.add_argument("--nfolds", type=int, default=3)
    ap.add_argument("--inorm", type=int, default=0, help="Loop 4 / E50: Instanz-Normierung der Eingabe (je Sample und Kanal Mittel/Streuung ueber das Raster) nach der globalen Normierung")
    ap.add_argument("--train_loc", default="", help="Loop 5 / E58: nur Trainingszeilen dieser Orte (Indizes 0 Boston, 1 One-North, 2 Holland Village, 3 Queenstown; Komma-getrennt)")
    ap.add_argument("--train_rows", type=int, default=0, help="Loop 5 / E58: danach zufaellige Teilmenge dieser Zeilenzahl (Seed 123)")
    ap.add_argument("--init_from", default="", help="Loop 5 / E58b: Lauf, dessen Gewichte und Normierung als Start dienen (Feinabstimmung auf einer kleinen Zielmenge)")
    ap.add_argument("--faug", type=float, default=0.0, help="Loop 5 / E65: Wahrscheinlichkeit je Sample einer Merkmalsraum-Stoerung (Sektor-Ausfall, Verstaerkung, Rauschen, Unschaerfe, Karten-Logits aus)")
    ap.add_argument("--flip", type=int, default=0, help="Spiegel-Augmentierung (x -> -x) mit Wahrscheinlichkeit 0,5 je Batch (Merkmale, Soll, Richtung)")
    ap.add_argument("--eval", default="train_eval,val", help="train_eval,val,test (test nur fuer Endwerte)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-8, help="AdamW-eps (groesser = daempft Schritte von Parametern mit winzigen Gradienten)")
    ap.add_argument("--min_scale", type=float, default=32.0, help="Untergrenze der Verlust-Skala (fp16)")
    ap.add_argument("--beta2", type=float, default=0.999, help="Adam beta2 (kleiner = zweites Moment passt sich schneller an; gegen Sprung-Instabilitaet)")
    ap.add_argument("--guard", type=int, default=1, help="Absicherung gegen Entgleisen: bei Verlustsprung (> guard_ratio x bestes 200-Schritt-Mittel) letzten gesunden Stand laden, Lernrate halbieren")
    ap.add_argument("--guard_ratio", type=float, default=1.2)
    ap.add_argument("--clip", type=float, default=3.0, help="adaptives Beschneiden der Gradientennorm (Vielfaches des gleitenden Mittels; 0 = aus, wie in E27-E29)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(f"{OUT}/{a.name}", exist_ok=True)
    log = open(f"{OUT}/{a.name}/log.txt", "a")

    def P(*s):
        t = " ".join(str(x) for x in s)
        print(t, flush=True)
        log.write(t + "\n")
        log.flush()

    P("==", time.strftime("%H:%M:%S"), vars(a))
    groups = a.feats.split(",")
    if a.split == "b123":
        sp = json.load(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/data/splits.json"))
        tr_idx = np.arange(sp["n"]["train"])
        sets = {"train_eval": ("train", sp["train_eval"]), "val": ("val", sp["val"]), "test": ("val", sp["test"])}
    else:  # Mini: Erprobung
        n_tr, n_va = 323, 81
        tr_idx = np.arange(n_tr)
        sets = {"train_eval": ("train", list(range(0, n_tr, 4))), "val": ("val", list(range(n_va)))}
    if a.split == "b123":
        sc = np.load(f"{HOME}/exp_gcar_b123/poses_val.npz")["scene"]
        uniq_ = sorted({int(x) for x in sc[np.array(sp["val"])]})  # VAL = ungerade Szenen-Nummern (+ Szene 42): VAL-A / VAL-B nach dem RANG der Szene in VAL aufteilen (je 11 Szenen)
        rank_ = {x: k for k, x in enumerate(uniq_)}
        valA = np.array([i for i in sp["val"] if rank_[int(sc[i])] % 2 == 0])
        valB = [int(i) for i in sp["val"] if rank_[int(sc[i])] % 2 == 1]
        sets["valB"] = ("val", valB)
        if a.train_part == "val":
            tr_idx = valA
    if a.split == "b123" and a.fold >= 0:
        sc_tr = np.load(f"{HOME}/exp_gcar_b123/poses_train.npz")["scene"]
        rows = np.arange(len(sc_tr))
        tr_idx = rows[sc_tr % a.nfolds != a.fold]
        sets["oof"] = ("train", [int(i) for i in rows[sc_tr % a.nfolds == a.fold]])
    if a.scene_sub > 1 and a.train_part == "train":  # nur Szenen mit Nummer % scene_sub == 0 (alle ihre Zeilen): gleiche Szenenzahl wie VAL-A (11) bei gleicher Zeilenzahl
        sc_all = np.load(f"{HOME}/exp_gcar_b123/poses_train.npz")["scene"]
        tr_idx = np.array([i for i in tr_idx if sc_all[i] % a.scene_sub == 0])
    if a.split == "b123" and a.train_loc and a.train_part == "train":
        loc_tr = np.load(f"{HOME}/exp_gcar_b123/poses_train.npz")["loc"]
        tr_idx = np.array([i for i in tr_idx if int(loc_tr[i]) in {int(x) for x in a.train_loc.split(",")}])
    if a.train_rows and a.train_rows < len(tr_idx):
        tr_idx = np.sort(np.random.RandomState(123).choice(tr_idx, a.train_rows, replace=False))
    if a.split == "b123":
        loc_va = np.load(f"{HOME}/exp_gcar_b123/poses_val.npz")["loc"]
        for nm_, src_ in (("val", sp["val"]), ("test", sp["test"])):
            sets[nm_ + "_b"] = ("val", [int(i) for i in src_ if loc_va[int(i)] == 0])
            sets[nm_ + "_s"] = ("val", [int(i) for i in src_ if loc_va[int(i)] != 0])
    if a.train_stride > 1:
        tr_idx = tr_idx[:: a.train_stride]
    if a.train_n:
        tr_idx = tr_idx[: a.train_n]
    sets["train_own"] = (a.train_part, [int(i) for i in tr_idx[:: max(len(tr_idx) // 400, 1)]])  # Loop 5: Zeilen, auf denen dieser Kopf trainiert wurde (Anpassung an das Training)
    if a.split == "b123" and a.train_part == "train":
        loc_tr_ = np.load(f"{HOME}/exp_gcar_b123/poses_train.npz")["loc"]
        sets["train_eval_b"] = ("train", [int(i) for i in sp["train_eval"] if loc_tr_[int(i)] == 0])
        sets["train_eval_s"] = ("train", [int(i) for i in sp["train_eval"] if loc_tr_[int(i)] != 0])
    ds = Data(a.fg, groups, a.train_part, tr_idx, a.split, with_dir=a.dir > 0)
    mean, std = channel_stats(ds)
    cin = int(mean.shape[0])
    net = UNet(cin, a.base, 3 if a.dir > 0 else 1, a.drop).cuda()
    if a.init_from:  # Loop 5 / E58b: Weiter-Training eines vorhandenen Kopfes (Gewichte und Normierung uebernehmen)
        ck0 = torch.load(f"{OUT}/{a.init_from}/model.pt", map_location="cuda", weights_only=False)
        net.load_state_dict(ck0["state"])
        mean, std = ck0["mean"].float(), ck0["std"].float()
    P("Eingangskanaele", cin, "Parameter", sum(p.numel() for p in net.parameters()), "Trainings-Samples", len(ds))
    mean, std = mean.cuda(), std.cuda()
    step, t0, run = 0, time.time(), []
    curve = []
    gema, n_clip, n_skip = None, 0, 0
    best_m, snap, n_roll, lr_scale, warm_start = None, None, 0, 1.0, None
    if a.steps > 0:
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd, eps=a.eps, betas=(0.9, a.beta2))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
        scaler = torch.cuda.amp.GradScaler(enabled=(a.amp == 1), init_scale=1024.0, growth_interval=200)  # amp 1 = fp16 mit Skalierer, amp 2 = bf16 (kein Skalierer noetig)
        torch.backends.cudnn.benchmark = True
        dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=True, prefetch_factor=4)
        net.train()
    while step < a.steps:
        for bt in dl:
            x, y = bt[0], bt[1]
            x = (x.cuda(non_blocking=True) - mean[None, :, None, None]) / std[None, :, None, None]
            if a.inorm:
                x = (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 1e-2)
            if a.faug > 0:
                x = faug(x, a.faug)
            y = y.cuda(non_blocking=True)
            dg = bt[2].cuda(non_blocking=True) if a.dir > 0 else None
            if a.flip and np.random.rand() < 0.5:  # Zeile = x (rechts): Spiegelung an der Fahrzeug-Laengsachse; Richtungs-x wechselt das Vorzeichen
                x, y = x.flip(2), y.flip(1)
                if dg is not None:
                    dg = dg.flip(2) * torch.tensor([-1.0, 1.0], device=dg.device).view(1, 2, 1, 1)
            with torch.autocast("cuda", dtype=(torch.bfloat16 if a.amp == 2 else torch.float16), enabled=bool(a.amp)):
                o = net(x)
            o = o.float()
            loss = loss_fn(o[:, 0], y, a.lam, a.wpos, a.gamma)
            if a.dir > 0:
                loss = loss + a.dir * dir_loss(o[:, 1:3], dg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if a.clip > 0:  # adaptives Beschneiden: Gradienten ueber dem a.clip-fachen des gleitenden Mittels der Norm werden darauf zurueckgesetzt (faengt Spruenge ab, die den Lauf e30_s4 zerstoert haben)
                scaler.unscale_(opt)
                gn = float(nn.utils.clip_grad_norm_(net.parameters(), float("inf")))
                if np.isfinite(gn):
                    gema = gn if gema is None else gema
                    lim = a.clip * gema
                    if gn > lim:
                        for p_ in net.parameters():
                            if p_.grad is not None:
                                p_.grad.mul_(lim / gn)
                        n_clip += 1
                    gema = 0.98 * gema + 0.02 * min(gn, lim)
            s0_ = scaler.get_scale() if scaler.is_enabled() else 0.0
            scaler.step(opt)
            scaler.update()
            if scaler.is_enabled() and scaler.get_scale() < a.min_scale:
                scaler.update(a.min_scale)  # Skala nicht unter die Untergrenze fallen lassen (Gradienten-Unterlauf in fp16)
            if scaler.is_enabled() and scaler.get_scale() < s0_:
                n_skip += 1  # Schritt wegen nicht endlicher Gradienten uebersprungen
            sched.step()
            mult = lr_scale * (min(1.0, 0.1 + 0.9 * (step - warm_start) / 200.0) if warm_start is not None else 1.0)
            if mult != 1.0:
                for g_ in opt.param_groups:
                    g_["lr"] *= mult
            run.append(float(loss))
            step += 1
            if step % 200 == 0:
                curve.append((step, float(np.mean(run[-200:]))))
                P(f"Schritt {step}/{a.steps} Verlust {curve[-1][1]:.4f} (max {max(run[-200:]):.1f}, median {float(np.median(run[-200:])):.2f}) ({(time.time() - t0) / step:.2f} s/Schritt) geklippt {n_clip} uebersprungen {n_skip} skala {scaler.get_scale() if scaler.is_enabled() else 0:.0f}")
                if a.guard and step >= 400:  # Absicherung: Verlust springt (Entgleisen der Decoder-Aktivierungen) -> letzten gesunden Stand wiederherstellen, ADAM-ZUSTAND NEU (frische Momente), Lernrate 200 Schritte neu hochfahren, danach x0.7
                    cur = curve[-1][1]
                    best_m = cur if best_m is None else min(best_m, cur)
                    if cur > a.guard_ratio * best_m and snap is not None and n_roll < 6:
                        net.load_state_dict(snap[0])
                        opt.state.clear()
                        scaler.load_state_dict(snap[2])
                        lr_scale *= 0.7
                        warm_start = step
                        n_roll += 1
                        P(f"ROLLBACK bei Schritt {step}: Verlust {cur:.3f} > {a.guard_ratio} x bestes {best_m:.3f}; frische Adam-Momente, Lernrate-Faktor jetzt {lr_scale:.2f}, 200 Schritte Anlauf")
                    elif cur <= 1.05 * best_m:
                        snap = (copy.deepcopy(net.state_dict()), None, copy.deepcopy(scaler.state_dict()))
            if step >= a.steps:
                break
    if a.steps == 0:  # nur auswerten: gespeichertes Modell laden, vorhandene Kennzahlen ergaenzen
        ck = torch.load(f"{OUT}/{a.name}/model.pt", map_location="cuda", weights_only=False)
        net.load_state_dict(ck["state"])
        mean, std = ck["mean"].cuda(), ck["std"].cuda()
        mp = f"{OUT}/{a.name}/metrics.json"
        res = json.load(open(mp)) if os.path.exists(mp) else {}
    else:
        torch.save(dict(state=net.state_dict(), mean=mean.cpu(), std=std.cpu(), args=vars(a)), f"{OUT}/{a.name}/model.pt")
        res = dict(args=vars(a), curve=curve, train_s=time.time() - t0, params=sum(p.numel() for p in net.parameters()), clipped_steps=n_clip, rollbacks=n_roll)
        if len(curve) > 5 and curve[-1][1] > 1.25 * min(c[1] for c in curve[3:]):
            res["diverged"] = True
            P("DIVERGIERT: Endverlust", curve[-1][1], "> 1,25 x Minimum", min(c[1] for c in curve[3:]))
    for nm in a.eval.split(","):
        if nm not in sets:
            continue
        part, idx = sets[nm]
        dse = Data(a.fg, groups, part, idx, a.split, target=False)
        prob, ids, dp = predict(net, dse, mean, std, workers=a.workers, want_dir=True, inorm=a.inorm)
        gts = load_gt_points(a.split, part, ids)
        res[nm] = eval_probs(prob, gts, taus=tuple(float(t) for t in a.taus.split(",")))
        if a.save_prob:
            np.save(f"{OUT}/{a.name}/prob_{nm}.npy", prob.numpy())
            np.save(f"{OUT}/{a.name}/ids_{nm}.npy", ids)
            if dp is not None:
                np.save(f"{OUT}/{a.name}/dir_{nm}.npy", dp.numpy())
        best = max(res[nm].items(), key=lambda kv: kv[1]["F1@1.0"])
        P(f"[{nm}] {len(ids)} Samples; bester Schwellwert {best[0]}: F1 @0,5/1/1,5 = {best[1]['F1@0.5']:.3f} / {best[1]['F1@1.0']:.3f} / {best[1]['F1@1.5']:.3f}, P/R@1 {best[1]['P@1.0']:.3f}/{best[1]['R@1.0']:.3f}, 1:1 {best[1]['F1o2o@1.0']:.3f}")
    json.dump(res, open(f"{OUT}/{a.name}/metrics.json", "w"), indent=1)
    P("FERTIG", time.strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()
