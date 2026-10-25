"""v2: Mittellinien-Heatmap aus eingefrorenen BEV-Features + Ausrichtung der Karte durch Suche.

1) HeatNet: kleines vollfaltendes Netz (ca. 0.1 M Parameter, verschiebungsgleich) sagt aus BEV-Features (512 Kanaele)
   und der vorhergesagten Karte (6 Kanaele) eine Mittellinien-Heatmap (200x200, 0.5 m) vorher. Gleichzeitig die
   Baseline 'Mittellinien aus der Wahrnehmung' (Raster-Kopf).
2) Ausrichtung der (kuenstlich verdorbenen) Karten-Knoten an der Heatmap:
   a) starr: Gittersuche ueber Verschiebung/Drehung, grob -> fein (maximiert die mittlere Heatmap an den Knoten)
   b) nicht-starr, iterativ: jeder Knoten tastet ein 2D-Fenster der Heatmap ab (Deformable-Sampling), schaetzt einen
      Versatz (Softmax-gewichtet), tauscht ihn mit Nachbarknoten (< sigma, gewichtet nach Konfidenz) aus, Update,
      wiederholen.
   Dasselbe zum Vergleich mit der perfekten Heatmap (Soll-Raster = Obergrenze des Verfahrens).
"""
import bflc_paths as BP  # project paths from the location of this checkout
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import align_train as A  # noqa: E402

dev = "cuda"
HALF = 50.0

# ------------------------------------------------------------------ Daten


def load_gt(split, fdir, root=None):
    root = root or os.environ.get("GT_ROOT", BP.REPO + "/out/centerline_gt")
    cache = f"{fdir}/{split}_gt_raster.npy"
    if not os.path.exists(cache):
        tokens = json.load(open(f"{fdir}/{split}_tokens.json"))
        np.save(cache, np.stack([np.load(f"{root}/{t}.npz")["centerline"] for t in tokens]))
    return torch.from_numpy(np.load(cache)).to(dev)  # bool [S,200,200]


def load_dist(split, fdir, clip_px=8.0, exact=False):
    """Abstand (in Pixeln, bei clip_px gekappt) zur naechsten Mittellinie.
    exact=True: exakter Abstand zur echten Linienfuehrung (make_dist_gt.py); sonst aus dem dick gezeichneten Raster."""
    import cv2

    if exact:
        return torch.from_numpy(np.load(f"{fdir}/{split}_gt_dist_exact.npy")).to(dev).float()
    cache = f"{fdir}/{split}_gt_dist.npy"
    if not os.path.exists(cache):
        gt = np.load(f"{fdir}/{split}_gt_raster.npy")
        d = np.stack([cv2.distanceTransform((~g).astype(np.uint8), cv2.DIST_L2, 5) for g in gt])
        np.save(cache, np.minimum(d, clip_px).astype(np.float16))
    return torch.from_numpy(np.load(cache)).to(dev).float()  # [S,200,200]


LEVELS_X = {"null": (0.0, 0.0, 0.0), **A.LEVELS}  # 'null' = bereits perfekt ausgerichtet (Schaden durch die Suche)

# ------------------------------------------------------------------ HeatNet


class HeatNet(nn.Module):
    def __init__(self, cin=512, c=32, w=48, drop=0.2, use_feat=True, use_sem=True, out_ch=1, coord=False):
        super().__init__()
        self.use_feat, self.use_sem, self.coord = use_feat, use_sem, coord
        self.red = nn.Sequential(nn.Conv2d(cin, c, 1), nn.GroupNorm(8, c), nn.ReLU())
        self.drop = nn.Dropout2d(drop)
        v = (-HALF + (torch.arange(200).float() + 0.5) * 0.5) / 51.2  # Zellmitten, normiert auf +-51.2 (Feature-Ausdehnung)
        grid = torch.stack([v[None, :].expand(200, 200), v[:, None].expand(200, 200)], -1)  # (Spalte=y, Zeile=x)
        self.register_buffer("grid", grid[None])
        # Koordinatenkanaele (E25): x, y und Radius vom Ursprung (normiert), damit das Netz die entfernungsabhaengige Unschaerfe kalibrieren kann
        self.register_buffer("coordmap", torch.cat([grid.permute(2, 0, 1), torch.sqrt((grid**2).sum(-1))[None]], 0)[None])
        layers, ch = [], (c if use_feat else 0) + (6 if use_sem else 0) + (3 if coord else 0)
        for dil in (1, 2, 4, 8, 1):
            layers += [nn.Conv2d(ch, w, 3, padding=dil, dilation=dil), nn.GroupNorm(8, w), nn.ReLU()]
            ch = w
        layers += [nn.Conv2d(w, out_ch, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, feats, sem, full=False):
        parts = []
        if self.use_feat:
            x = self.red(feats.float())
            x = F.grid_sample(x, self.grid.expand(x.shape[0], -1, -1, -1), mode="bilinear", align_corners=False)
            parts.append(self.drop(x))
        if self.use_sem:
            parts.append(sem.float())
        if self.coord:
            parts.append(self.coordmap.expand(parts[0].shape[0], -1, -1, -1))
        o = self.net(torch.cat(parts, 1))
        return o if full else o[:, 0]  # full=True: alle Ausgabekanaele (Test H: Abstand, cos2t, sin2t)


class UNetHeat(nn.Module):
    """U-Netz: Encoder-Decoder mit Skip-Verbindungen auf dem 200x200-Raster (0,5 m). Gleiche Ein-/Ausgabe wie HeatNet."""

    def __init__(self, cin=512, c=32, base=48, drop=0.2, use_feat=True, use_sem=True, out_ch=1):
        super().__init__()
        self.use_feat, self.use_sem = use_feat, use_sem
        self.red = nn.Sequential(nn.Conv2d(cin, c, 1), nn.GroupNorm(8, c), nn.ReLU())
        self.drop = nn.Dropout2d(drop)
        v = (-HALF + (torch.arange(200).float() + 0.5) * 0.5) / 51.2
        grid = torch.stack([v[None, :].expand(200, 200), v[:, None].expand(200, 200)], -1)
        self.register_buffer("grid", grid[None])
        chin = (c if use_feat else 0) + (6 if use_sem else 0)

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.ReLU())

        b = base
        self.e1, self.e2, self.e3, self.e4 = blk(chin, b), blk(b, 2 * b), blk(2 * b, 4 * b), blk(4 * b, 4 * b)
        self.d3, self.d2, self.d1 = blk(8 * b, 2 * b), blk(4 * b, b), blk(2 * b, b)
        self.out = nn.Conv2d(b, out_ch, 1)

    def forward(self, feats, sem, full=False):
        parts = []
        if self.use_feat:
            x = self.red(feats.float())
            x = F.grid_sample(x, self.grid.expand(x.shape[0], -1, -1, -1), mode="bilinear", align_corners=False)
            parts.append(self.drop(x))
        if self.use_sem:
            parts.append(sem.float())
        x = torch.cat(parts, 1)
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        u3 = self.d3(torch.cat([up(e4), e3], 1))
        u2 = self.d2(torch.cat([up(u3), e2], 1))
        u1 = self.d1(torch.cat([up(u2), e1], 1))
        o = self.out(u1)
        return o if full else o[:, 0]


def make_net(arch="small", cin=512, drop=0.2, c=32, w=48, base=48, use_feat=True, use_sem=True, out_ch=1, coord=None):
    if coord is None:
        coord = os.environ.get("NET_COORD", "0") == "1"  # Koordinatenkanaele nur fuer das FCN (E25); per Umgebungsvariable, damit Auswertungsskripte dasselbe Netz bauen
    if arch == "small" and coord:
        return HeatNet(cin=cin, drop=drop, c=c, w=w, use_feat=use_feat, use_sem=use_sem, out_ch=out_ch, coord=True)
    if arch == "cnt":  # ConvNeXt-V2-Encoder + Transformer-Flaschenhals (models_cnt.py), nur BEV-Merkmale
        from models_cnt import CNTNet

        return CNTNet(cin=cin, c=c, base=base, drop=drop, use_feat=use_feat, use_sem=False, out_ch=out_ch)
    if arch == "unet":
        return UNetHeat(cin=cin, c=c, base=base, drop=drop, use_feat=use_feat, use_sem=use_sem, out_ch=out_ch)
    return HeatNet(cin=cin, drop=drop, c=c, w=w, use_feat=use_feat, use_sem=use_sem, out_ch=out_ch)


def heat_metrics(prob, gt, thr=0.5, tol=2):
    """IoU (exakt) und Praezision/Ausbeute mit Toleranz tol Pixel (1 px = 0.5 m)."""
    pred = prob > thr
    inter, union = (pred & gt).sum().item(), (pred | gt).sum().item()
    k = 2 * tol + 1
    dil = lambda m: F.max_pool2d(m.float()[:, None], k, 1, tol)[:, 0] > 0
    prec = (pred & dil(gt)).sum().item() / max(pred.sum().item(), 1)
    rec = (gt & dil(pred)).sum().item() / max(gt.sum().item(), 1)
    return dict(iou=inter / max(union, 1), prec=prec, rec=rec, f1=2 * prec * rec / max(prec + rec, 1e-9))


# ------------------------------------------------------------------ Ausrichtung


def gauss_blur(H, sigma_px):
    """H [B,200,200] -> geglaettet, Spitzenhoehe bleibt (Kern auf Maximum 1 normiert, dann auf Summe der Eingabe skaliert)."""
    k = int(math.ceil(3 * sigma_px))
    x = torch.arange(-k, k + 1, device=H.device).float()
    g = torch.exp(-x * x / (2 * sigma_px**2))
    g = g / g.sum()
    h = H[:, None]
    h = F.conv2d(F.pad(h, (k, k, 0, 0), mode="replicate"), g.view(1, 1, 1, -1))
    h = F.conv2d(F.pad(h, (0, 0, k, k), mode="replicate"), g.view(1, 1, -1, 1))
    return h[:, 0]


def samp(H, pts):
    """H [B,200,200], pts [B,...,2] (x,y in m) -> [B,...] (bilinear, ausserhalb 0)."""
    B = H.shape[0]
    shp = pts.shape[1:-1]
    g = torch.stack([pts[..., 1], pts[..., 0]], -1).reshape(B, 1, -1, 2) / HALF
    out = F.grid_sample(H[:, None], g, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.reshape(B, *shp)


def rigid_search(p, tg, valid, Hb, trange, tstep, deg, dstep, lam_t=0.0, lam_th=0.0, margin=0.0):
    """Gittersuche; mit lam_t [1/m^2], lam_th [1/Grad^2] wird grosse Korrektur bestraft (Vorwissen: Drift klein)."""
    B, N, _ = p.shape
    ts = torch.arange(-trange, trange + 1e-6, tstep, device=dev)
    ths = torch.arange(-deg, deg + 1e-6, dstep, device=dev) * math.pi / 180
    ty, tx, th = torch.meshgrid(ts, ts, ths, indexing="ij")
    t = torch.stack([tx.reshape(-1), ty.reshape(-1)], -1)  # [Hh,2]
    R = A.rot(th.reshape(-1))  # [Hh,2,2]
    q = torch.einsum("hij,bnj->bhni", R, p) + t[None, :, None]  # [B,Hh,N,2]
    s = samp(Hb, q)  # [B,Hh,N]
    v = valid[:, None].float()
    score = (s * v).sum(-1) / v.sum(-1)
    if lam_t or lam_th:
        pen = lam_t * (t**2).sum(-1) + lam_th * (th.reshape(-1) * 180 / math.pi) ** 2  # [Hh]
        score = score - pen[None]
    best = score.argmax(1)
    if margin > 0:  # 'nicht schaden': Identitaet behalten, wenn der Gewinn die Schwelle nicht ueberschreitet
        ident = ((t.abs().sum(-1) == 0) & (th.reshape(-1) == 0)).nonzero()[0, 0]
        gain = score.gather(1, best[:, None])[:, 0] - score[:, ident]
        best = torch.where(gain < margin, ident.expand_as(best), best)
    Rb, tb = R[best], t[best]
    p2 = torch.einsum("bij,bnj->bni", Rb, p) + tb[:, None]
    tg2 = torch.einsum("bij,bnj->bni", Rb, tg)
    return p2, tg2


def refine(p, tg, valid, Hb, iters, tau, gamma, sig, dn=2.0, dnstep=0.25, dt=0.0, dtstep=0.5, track=None, sigp=0.0):
    B, N, _ = p.shape
    a = torch.arange(-dn, dn + 1e-6, dnstep, device=dev)
    b = torch.arange(-dt, dt + 1e-6, dtstep, device=dev) if dt > 0 else torch.zeros(1, device=dev)  # dt=0: nur quer
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    aa, bb = aa.reshape(-1), bb.reshape(-1)  # [K]
    v = valid.float()
    eye = torch.eye(N, dtype=torch.bool, device=dev)
    for it in range(iters):
        nrm = torch.stack([-tg[..., 1], tg[..., 0]], -1)  # Normale [B,N,2]
        off = aa[None, None, :, None] * nrm[:, :, None] + bb[None, None, :, None] * tg[:, :, None]  # [B,N,K,2]
        s = samp(Hb, p[:, :, None] + off)  # [B,N,K]
        # Vorwissen: kleine Verschiebungen sind wahrscheinlicher (Gauss-Prior mit sigp [m])
        logit = s / tau - (aa**2 + bb**2)[None, None] / (2 * sigp**2) if sigp > 0 else s / tau
        w = torch.softmax(logit, -1)
        shift = (w[..., None] * off).sum(2)  # [B,N,2]
        conf = s.max(-1).values * v
        # Nachrichtenaustausch: Nachbarn gewichtet nach Abstand und Konfidenz
        far = (1e4 + 100.0 * torch.arange(N, device=dev).float())[None, :, None].expand(B, N, 2)
        q = torch.where(valid[..., None], p, far)
        d2 = ((q[:, :, None] - q[:, None]) ** 2).sum(-1)
        wt = torch.exp(-d2 / (2 * sig**2)) * conf[:, None, :]
        shift_s = torch.einsum("bij,bjc->bic", wt, shift) / (wt.sum(-1, keepdim=True) + 1e-6)
        p = p + gamma * shift_s * v[..., None]
        if track is not None:
            track.append(p.clone())
    return p


def align(p, tg, valid, H, cfg, track=None):
    """H: Heatmap-Wahrscheinlichkeiten [B,200,200]. Gibt (nach starr, nach Verfeinerung) zurueck."""
    p1, tg1 = rigid_search(p, tg, valid, gauss_blur(H, cfg["s1"]), 4.0, 0.5, 4.0, 1.0, cfg["lam_t"], cfg["lam_th"], cfg["m1"])
    p1, tg1 = rigid_search(p1, tg1, valid, gauss_blur(H, cfg["s2"]), 0.5, 0.125, 0.5, 0.25, 0.0, 0.0, cfg["m2"])
    if cfg["iters"] == 0:
        return p1, p1
    p2 = refine(p1, tg1, valid, gauss_blur(H, cfg["s3"]), cfg["iters"], cfg["tau"], cfg["gamma"], cfg["sig"], dt=cfg["dt"], track=track, sigp=cfg["sigp"])
    return p1, p2


DEF = dict(s1=2.0, s2=1.0, s3=1.0, iters=0, tau=0.05, gamma=0.3, sig=10.0, lam_t=0.03, lam_th=0.01, dt=0.0, sigp=0.0, m1=0.0, m2=0.0)


@torch.no_grad()
def eval_align(Hfn, data, cfg, bs=8, seeds=3, levels=A.LEVELS):
    res = {}
    S = data["nodes"].shape[0]
    for lname, (amp, deg, nr) in levels.items():
        acc, cnt = None, 0
        for sd in range(seeds):
            gen = torch.Generator(device=dev)
            gen.manual_seed(1000 + sd)
            for i in range(0, S, bs):
                sl = slice(i, min(i + bs, S))
                gt, tg0, valid = data["nodes"][sl], data["tangent"][sl], data["mask"][sl]
                p, tg = A.perturb(gt, tg0, amp, deg, nr, gen)
                p1, p2 = align(p, tg, valid, Hfn(sl), cfg)
                rows = [("verdorben", A.err_stats(p, gt, tg0, valid)), ("nach starr", A.err_stats(p1, gt, tg0, valid)), ("nach Verfeinerung", A.err_stats(p2, gt, tg0, valid))]
                w = sl.stop - sl.start
                if acc is None:
                    acc = {k: {m: 0.0 for m in v} for k, v in rows}
                for k, v in rows:
                    for m in v:
                        acc[k][m] += v[m] * w
                cnt += w
        res[lname] = {k: {m: x / cnt for m, x in v.items()} for k, v in acc.items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default=os.path.expanduser("~/exp/feats"))
    ap.add_argument("--graph", default=os.path.expanduser("~/exp/graph"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg", default="{}", help="JSON mit Ueberschreibungen der Ausrichtungs-Parameter")
    ap.add_argument("--mode", choices=["bce", "dist"], default="bce", help="bce: Mittellinien-Heatmap; dist: Abstand zur Mittellinie")
    ap.add_argument("--eval_val_only", action="store_true", help="Ausrichtungs-Auswertung nur auf der Validierung (nicht auf dem Training)")
    ap.add_argument("--multi", action="store_true", help="Test H: zusaetzlich Spurrichtung (cos 2t, sin 2t) vorhersagen (braucht make_dir_gt.py)")
    ap.add_argument("--arch", choices=["small", "unet", "cnt"], default="small", help="small: FCN mit Dilatationen (0,12 Mio.); unet: Encoder-Decoder; cnt: ConvNeXt-V2 + Transformer (models_cnt.py)")
    ap.add_argument("--amp", action="store_true", help="Training mit gemischter Genauigkeit (fp16-Autocast)")
    ap.add_argument("--base", type=int, default=48, help="Basisbreite des U-Netzes")
    ap.add_argument("--cin", type=int, default=512, help="Eingangskanaele der Features (512, oder 64/32 nach PCA; FEATS_NAME setzen)")
    ap.add_argument("--c", type=int, default=32, help="Kanaele nach der 1x1-Reduktion der Features")
    ap.add_argument("--w", type=int, default=48, help="Breite der Faltungsschichten")
    ap.add_argument("--no_feat", action="store_true", help="nur vorhergesagte Karte (6 Kanaele) als Eingabe")
    ap.add_argument("--no_sem", action="store_true", help="nur BEV-Features als Eingabe")
    ap.add_argument("--flip", action="store_true", help="Augmentierung: links/rechts spiegeln (Zeilenachse)")
    ap.add_argument("--exact_dist", action="store_true", help="Abstands-Ziel exakt zur echten Linienfuehrung statt aus dem dicken Raster")
    ap.add_argument("--train_only", action="store_true", help="nur trainieren und speichern (fuer Ensemble)")
    ap.add_argument("--load", default="", help="trainiertes heatnet.pt laden statt zu trainieren (Komma = Ensemble)")
    ap.add_argument("--sweep", default="", help="JSON-Gitter von Ausrichtungs-Parametern (nur Validierung)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    cfg = {**DEF, **json.loads(args.cfg)}

    tr, va = A.load_split("train", args.feats, args.graph), A.load_split("val", args.feats, args.graph)
    gt_tr, gt_va = load_gt("train", args.feats), load_gt("val", args.feats)
    if args.mode == "dist":
        dist_tr = load_dist("train", args.feats, exact=args.exact_dist)
        dist_va = load_dist("val", args.feats, exact=args.exact_dist)

    out_ch = 3 if args.multi else 1
    if args.multi:
        assert args.mode == "dist" and not args.flip, "--multi braucht --mode dist und kein --flip"
        dir_tr = torch.from_numpy(np.load(f"{args.feats}/train_dir.npy")).to(dev)  # fp16 [S,2,200,200]
    net = make_net(args.arch, args.cin, args.drop, args.c, args.w, args.base, not args.no_feat, not args.no_sem, out_ch).to(dev)
    print("HeatNet Parameter:", sum(p.numel() for p in net.parameters()) / 1e3, "k")
    nets_extra = []
    if args.load:  # mehrere Pfade mit Komma = Ensemble (Mittelwert der Ausgaben)
        paths = args.load.split(",")
        net.load_state_dict(torch.load(paths[0]))
        for pth in paths[1:]:
            n2 = make_net(args.arch, args.cin, args.drop, args.c, args.w, args.base, not args.no_feat, not args.no_sem, out_ch).to(dev)
            n2.load_state_dict(torch.load(pth))
            n2.eval()
            nets_extra.append(n2)
        args.steps = 0
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(args.steps, 1), pct_start=0.1)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)
    S = tr["nodes"].shape[0]
    t0 = time.time()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    for step in range(1, args.steps + 1):
        net.train()
        idx = torch.randint(0, S, (args.bs,), device=dev, generator=gen)
        fb, sb = tr["feats"][idx], tr["sem"][idx]
        y = gt_tr[idx].float()
        d_b = dist_tr[idx] if args.mode == "dist" else None
        if args.flip:  # links/rechts spiegeln: im Lidar-System ist x = rechts = Zeilenachse (-2)
            fl = torch.rand(args.bs, device=dev, generator=gen) < 0.5
            m4 = fl[:, None, None, None]
            fb = torch.where(m4, fb.flip(-2), fb)
            sb = torch.where(m4, sb.flip(-2), sb)
            y = torch.where(fl[:, None, None], y.flip(-2), y)
            if d_b is not None:
                d_b = torch.where(fl[:, None, None], d_b.flip(-2), d_b)
        with torch.cuda.amp.autocast(enabled=args.amp):
            if args.multi:
                out = net(fb, sb, full=True).float()
                logit = out[:, 0]
            else:
                logit = net(fb, sb).float()
        if args.mode == "dist":
            d = d_b / 8.0  # 0..1
            wgt = 1.0 + 4.0 * (d < 0.25).float()  # Naehe der Linie staerker gewichten
            loss = (wgt * (torch.sigmoid(logit) - d).abs()).mean()
            if args.multi:  # Spurrichtung (Doppelwinkel) nur nahe der Mittellinie (Abstand <= 1 m)
                m = (d_b <= 2.0).float()[:, None]
                tdir = dir_tr[idx].float()
                loss = loss + 0.5 * (m * (torch.tanh(out[:, 1:3]) - tdir).abs()).sum() / (2 * m.sum() + 1)
        else:
            bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(8.0, device=dev))
            pr = torch.sigmoid(logit)
            dice = 1 - (2 * (pr * y).sum() + 1) / (pr.sum() + y.sum() + 1)
            loss = bce + dice
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 250 == 0 or step == 1 or os.environ.get("STEP_LOG"):
            print(f"step {step:5d} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)

    net.eval()
    if args.train_only:
        torch.save(net.state_dict(), f"{args.out}/heatnet.pt")
        return

    @torch.no_grad()
    def probs(data, sl):
        o = torch.stack([torch.sigmoid(n(data["feats"][sl], data["sem"][sl])) for n in [net] + nets_extra]).mean(0)
        return 1.0 - o if args.mode == "dist" else o  # 'hoeher = naeher an der Mittellinie'

    with torch.no_grad():
        for name, data, gtr in (("Validierung", va, gt_va), ("Training", tr, gt_tr)):
            P = torch.cat([probs(data, slice(i, min(i + 16, gtr.shape[0]))) for i in range(0, gtr.shape[0], 16)])
            if args.mode == "dist":
                P = ((1.0 - P) * 8.0 <= 1.5).float()  # Abstand <= 1.5 px als Maske, nur fuer die Heatmap-Metrik
            m = heat_metrics(P, gtr)
            print(f"== Heatmap-Qualitaet ({name}, Schwelle 0.5): IoU {m['iou']:.3f} | Praezision(1 m) {m['prec']:.3f} | Ausbeute(1 m) {m['rec']:.3f} | F1(1 m) {m['f1']:.3f}")
            if name == "Validierung":
                heat_val = m

    if args.sweep:
        import itertools

        grid = json.loads(args.sweep)  # z.B. {"lam_t":[0,0.03],"dt":[0,2]}
        keys = list(grid)
        print("== Sweep (Validierung); je Zeile mittlerer Fehler [m] nach starr / nach Verfeinerung, Stufen klein | mittel | gross")
        for vals in itertools.product(*[grid[k] for k in keys]):
            c = {**cfg, **dict(zip(keys, vals))}
            r = eval_align(lambda sl: probs(va, sl), va, c, levels=LEVELS_X)
            cells = " | ".join(f"{r[l]['nach starr']['mean']:.2f} / {r[l]['nach Verfeinerung']['mean']:.2f}" for l in r)
            print(f"  {dict(zip(keys, vals))}: {cells}", flush=True)
        return

    out = dict(heat_val=heat_val, cfg=cfg, args=vars(args))
    for name, data, gtr in (("Validierung", va, gt_va),) + (() if args.eval_val_only else (("Training", tr, gt_tr),)):
        r_pred = eval_align(lambda sl: probs(data, sl), data, cfg, levels=LEVELS_X)
        r_orac = eval_align(lambda sl: gtr[sl].float(), data, cfg, levels=LEVELS_X)
        A.print_res(r_pred, f"{name}: Ausrichtung mit vorhergesagter Heatmap")
        A.print_res(r_orac, f"{name}: Ausrichtung mit perfekter Heatmap (Obergrenze des Verfahrens)")
        out[name] = dict(pred=r_pred, oracle=r_orac)
    json.dump(out, open(f"{args.out}/results.json", "w"), indent=1)
    torch.save(net.state_dict(), f"{args.out}/heatnet.pt")


if __name__ == "__main__":
    main()
