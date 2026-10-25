"""Karten-Ausrichtung: Mittellinien-Knoten der Karte iterativ mit der BEV-Wahrnehmung ausrichten.

Ablauf pro Sample (alles im Lidar-System, Zeile = x, Spalte = y):
  1. Die korrekt ausgerichtete Karte (Knoten alle 3 m auf den Mittellinien) wird KUENSTLICH verdorben:
     starre Verschiebung + Drehung (Lokalisierungsdrift) plus glatte, nicht-starre Verzerrung.
  2. Grobe Ausrichtung: ein CNN sieht die verdorbenen Knoten und die vorhergesagte Karte (sem) und schaetzt
     EINE starre Korrektur (tx, ty, theta) fuer den ganzen Graphen.
  3. Feine Ausrichtung, L Runden: jeder Knoten tastet per Deformable-Sampling K gelernte Punkte um sich herum
     in den BEV-Features (512 Kanaele, eingefroren) und in der vorhergesagten Karte ab, tauscht sich per
     Attention mit Nachbarknoten (< 12 m) aus und schaetzt eine Restverschiebung. Punkte verschieben, wiederholen.
  4. Ziel ist die Original-Lage der Knoten (Deep Supervision ueber alle Runden).

Das BEV-Netz ist eingefroren und seine Features vorab gespeichert (extract_features.py), nur dieser kleine
Decoder wird trainiert.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

dev = "cuda"
HALF_F, HALF_S = 51.2, 50.0  # halbe Kantenlaenge Features (0.8 m Zellen) / Vorhersage (0.5 m Zellen)

# ------------------------------------------------------------------ Daten


class CpuStore:
    """Grosse Feature-/Vorhersage-Arrays bleiben im RAM; Indizierung liefert die Auswahl auf der GPU
    (Aufrufstellen wie data['feats'][idx] bleiben unveraendert)."""

    def __init__(self, t):
        self.t = t
        self.shape = t.shape

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu()
        return self.t[idx].to(dev)

    def __len__(self):
        return self.t.shape[0]


class MmapStore:
    """Merkmale bleiben in der Datei (np.memmap); mehrere Prozesse teilen sich den Seiten-Cache des Betriebssystems statt je 9 GB RAM
    zu belegen. Indizierung wie bei CpuStore (Auswahl landet auf der GPU)."""

    def __init__(self, a):
        self.a = a
        self.shape = a.shape

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu().numpy()
        out = self.a[idx]
        return torch.from_numpy(np.ascontiguousarray(out)).to(dev)

    def __len__(self):
        return self.a.shape[0]


def load_split(split, fdir, gdir):
    on_cpu = os.environ.get("DATA_ON_CPU") == "1"  # fuer grosse Datensaetze (GPU hat nur 12 GB)
    feats = np.load(f"{fdir}/{split}_{os.environ.get('FEATS_NAME', 'feats')}.npy", mmap_mode="r")
    if os.environ.get("FEATS_MMAP") == "1":  # nicht in den RAM kopieren (geteilter Seiten-Cache)
        feats = MmapStore(feats)
    else:
        feats = torch.from_numpy(np.ascontiguousarray(feats))  # fp16 [S,C,128,128]
    if os.environ.get("NO_SEM") == "1":  # Kamera-Radar-Variante: keine Vorhersage des Lidar-Netzes; winziger Platzhalter (nie benutzt)
        sem = torch.zeros(feats.shape[0], 1, 1, 1, dtype=torch.float16)
    else:
        sem = torch.from_numpy(np.load(f"{fdir}/{split}_sem.npy"))  # fp16 [S,6,200,200]
    if isinstance(feats, MmapStore):  # Memory-Map: Merkmale bleiben in der Datei; sem (falls vorhanden) wie sonst
        sem = CpuStore(sem) if on_cpu else sem.to(dev)
    else:
        feats, sem = (CpuStore(feats), CpuStore(sem)) if on_cpu else (feats.to(dev), sem.to(dev))
    g = np.load(f"{gdir}/{split}_graph.npz")
    tj = f"{fdir}/{split}_tokens.json"
    if "tokens" not in g.files:
        raise RuntimeError(f"{gdir}/{split}_graph.npz ohne 'tokens': mit der aktuellen prep_graph.py neu erzeugen (Reihenfolge!)")
    if os.path.exists(tj) and list(g["tokens"]) != json.load(open(tj)):
        raise RuntimeError(f"Reihenfolge von Graph und Features verschieden ({split}): Knoten gehoeren zu falschen Samples")
    return dict(
        feats=feats,
        sem=sem,
        nodes=torch.from_numpy(g["nodes"]).to(dev),
        tangent=torch.from_numpy(g["tangent"]).to(dev),
        mask=torch.from_numpy(g["mask"]).to(dev),
    )


def rot(th):
    c, s = torch.cos(th), torch.sin(th)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)  # [...,2,2]


def perturb(nodes, tangent, amp, deg, nr, gen, scale=None):
    """Verdirbt die Karte. amp [m], deg [Grad], nr [m] sind Maximalwerte; scale [B] in 0..1 skaliert sie."""
    B = nodes.shape[0]
    u = lambda *s: torch.rand(*s, device=dev, generator=gen) * 2 - 1
    if scale is None:
        scale = torch.ones(B, device=dev)
    t = u(B, 2) * amp * scale[:, None]
    th = u(B) * deg * math.pi / 180 * scale
    R = rot(th)
    p = torch.einsum("bij,bnj->bni", R, nodes) + t[:, None]
    tg = torch.einsum("bij,bnj->bni", R, tangent)
    # glatte nicht-starre Verzerrung: 3 Wellen, Wellenlaenge 20-60 m
    for _ in range(3):
        ang = (u(B) * math.pi)[:, None]
        d = torch.cat([torch.cos(ang), torch.sin(ang)], 1)  # Wellenrichtung [B,2]
        L = (30 + 20 * u(B))[:, None]
        ph = (u(B) * math.pi)[:, None]
        a = (u(B, 2) * nr * scale[:, None])  # Vektoramplitude
        w = torch.sin(2 * math.pi * (p * d[:, None]).sum(-1) / L + ph)  # [B,N]
        p = p + w[..., None] * a[:, None]
    return p, tg


def sample(fmap, pts, half):
    """fmap [B,C,H,W] (Zeile = x, Spalte = y), pts [B,N,K,2] in m -> [B,N,K,C]."""
    grid = torch.stack([pts[..., 1] / half, pts[..., 0] / half], -1)
    out = F.grid_sample(fmap, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.permute(0, 2, 3, 1)


def raster_nodes(p, mask):
    """Knoten als 200x200-Karte (0.5 m), mit 5x5-Verdickung."""
    B, N, _ = p.shape
    idx = torch.floor((p + HALF_S) / 0.5).long()
    ok = mask & (idx >= 0).all(-1) & (idx < 200).all(-1)
    img = torch.zeros(B, 200 * 200, device=p.device)
    flat = (idx[..., 0].clamp(0, 199) * 200 + idx[..., 1].clamp(0, 199))
    img.scatter_(1, flat, ok.float())
    img = img.view(B, 1, 200, 200)
    return F.max_pool2d(img, 5, 1, 2)


# ------------------------------------------------------------------ Modell


def fourier(p, nf=6):
    f = (2.0 ** torch.arange(nf, device=p.device)) * math.pi / 64.0
    x = p[..., None] * f
    return torch.cat([x.sin(), x.cos()], -1).flatten(-2)


class Coarse(nn.Module):
    """Eine starre Korrektur (tx, ty, theta) aus verdorbener Karte + vorhergesagter Karte."""

    def __init__(self, cin=6 + 1 + 2, tmax=4.0, deg=6.0):
        super().__init__()
        ch = [cin, 32, 64, 64, 128]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 5 if i == 0 else 3, 2, 2 if i == 0 else 1), nn.GroupNorm(8, ch[i + 1]), nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * 13 * 13, 256), nn.ReLU(), nn.Linear(256, 3))
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)
        self.tmax, self.deg = tmax, deg
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 200), torch.linspace(-1, 1, 200), indexing="ij")
        self.register_buffer("coord", torch.stack([xx, yy])[None])

    def forward(self, p, mask, sem):
        x = torch.cat([sem, raster_nodes(p, mask), self.coord.expand(p.shape[0], -1, -1, -1)], 1)
        o = self.fc(self.net(x))
        t = torch.tanh(o[:, :2]) * self.tmax
        th = torch.tanh(o[:, 2]) * self.deg * math.pi / 180
        return t, th


class Layer(nn.Module):
    def __init__(self, d, c, K, semc=6, radius=12.0):
        super().__init__()
        self.K, self.radius = K, radius
        self.off = nn.Linear(d, K * 2)
        self.att = nn.Linear(d, K)
        self.val = nn.Linear(c + semc, d)
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.mha = nn.MultiheadAttention(d, 4, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.delta = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, h, p, valid, Fr, S, use_feat, use_sem, use_att):
        B, N, _ = h.shape
        off = torch.tanh(self.off(h)).view(B, N, self.K, 2) * 3.0
        w = torch.softmax(self.att(h), -1)
        pts = p.detach()[:, :, None] + off
        sf = sample(Fr, pts, HALF_F) if use_feat else torch.zeros(B, N, self.K, Fr.shape[1], device=h.device)
        ss = sample(S, pts, HALF_S) if use_sem else torch.zeros(B, N, self.K, S.shape[1], device=h.device)
        v = self.val(torch.cat([sf, ss], -1))
        h = self.n1(h + (w[..., None] * v).sum(2))
        if use_att:
            # Nachbarn < radius; ungueltige Knoten weit weg und voneinander getrennt -> sehen nur sich selbst
            q = p.detach().clone()
            far = (1e4 + 100.0 * torch.arange(N, device=p.device).float())[None, :, None].expand(B, N, 2)
            q = torch.where(valid[..., None], q, far)
            d2 = ((q[:, :, None] - q[:, None]) ** 2).sum(-1)  # direkt statt cdist (numerisch exakt)
            m = (d2 > self.radius**2) & ~torch.eye(N, dtype=torch.bool, device=p.device)  # True = nicht erlaubt
            m = m.repeat_interleave(4, 0)
            a, _ = self.mha(h, h, h, attn_mask=m, need_weights=False)
            h = self.n2(h + a)
        h = self.n3(h + self.ffn(h))
        return h, self.delta(h)


class Align(nn.Module):
    def __init__(self, cin=512, c=64, d=128, L=3, K=4, use_feat=True, use_sem=True, use_att=True, use_coarse=True):
        super().__init__()
        self.use_feat, self.use_sem, self.use_att, self.use_coarse = use_feat, use_sem, use_att, use_coarse
        self.red = nn.Sequential(nn.Conv2d(cin, c, 1), nn.GroupNorm(8, c), nn.ReLU())
        self.coarse = Coarse()
        self.inp = nn.Sequential(nn.Linear(c + 6 + 24 + 2, d), nn.ReLU(), nn.Linear(d, d))
        self.layers = nn.ModuleList([Layer(d, c, K) for _ in range(L)])

    def forward(self, feats, sem, p, tg, valid):
        """p: verdorbene Knoten [B,N,2]. Gibt Liste der Positionen nach grob und nach jeder Runde zurueck."""
        Fr = self.red(feats.float())
        S = sem.float()
        outs = []
        if self.use_coarse:
            t, th = self.coarse(p, valid, S)
            R = rot(th)
            p = torch.einsum("bij,bnj->bni", R, p) + t[:, None]
            tg = torch.einsum("bij,bnj->bni", R, tg)
        outs.append(p)
        p0 = p[:, :, None]
        sf = sample(Fr, p0, HALF_F)[:, :, 0] if self.use_feat else torch.zeros(*p.shape[:2], Fr.shape[1], device=p.device)
        ss = sample(S, p0, HALF_S)[:, :, 0] if self.use_sem else torch.zeros(*p.shape[:2], 6, device=p.device)
        h = self.inp(torch.cat([sf, ss, fourier(p), tg], -1))
        for layer in self.layers:
            h, dp = layer(h, p, valid, Fr, S, self.use_feat, self.use_sem, self.use_att)
            p = p.detach() + dp
            outs.append(p)
        return outs


# ------------------------------------------------------------------ Auswertung


def err_stats(p, gt, tg, valid):
    d = p - gt
    e = d.norm(dim=-1)
    lon = (d * tg).sum(-1).abs()
    lat = (d - (d * tg).sum(-1, keepdim=True) * tg).norm(dim=-1)
    v = valid.float()
    n = v.sum()
    return dict(
        mean=float((e * v).sum() / n),
        lat=float((lat * v).sum() / n),
        lon=float((lon * v).sum() / n),
        ok50=float(((e < 0.5).float() * v).sum() / n),
        ok100=float(((e < 1.0).float() * v).sum() / n),
    )


LEVELS = {"klein": (0.5, 1.0, 0.15), "mittel": (1.0, 2.0, 0.30), "gross": (2.0, 3.0, 0.50)}


@torch.no_grad()
def evaluate(model, data, bs=8, seeds=3):
    model.eval()
    res = {}
    S = data["nodes"].shape[0]
    for lname, (amp, deg, nr) in LEVELS.items():
        acc = None
        for sd in range(seeds):
            gen = torch.Generator(device=dev)
            gen.manual_seed(1000 + sd)
            for i in range(0, S, bs):
                sl = slice(i, min(i + bs, S))
                gt, tg0, valid = data["nodes"][sl], data["tangent"][sl], data["mask"][sl]
                p, tg = perturb(gt, tg0, amp, deg, nr, gen)
                outs = model(data["feats"][sl], data["sem"][sl], p, tg, valid)
                rows = [("verdorben", err_stats(p, gt, tg0, valid))]
                rows.append(("nach grob", err_stats(outs[0], gt, tg0, valid)))
                for li, o in enumerate(outs[1:]):
                    rows.append((f"nach Runde {li + 1}", err_stats(o, gt, tg0, valid)))
                w = (sl.stop - sl.start)
                if acc is None:
                    acc = {k: {m: 0.0 for m in v} for k, v in rows}
                    cnt = 0
                for k, v in rows:
                    for m in v:
                        acc[k][m] += v[m] * w
                cnt += w
        res[lname] = {k: {m: x / cnt for m, x in v.items()} for k, v in acc.items()}
    return res


def print_res(res, title):
    print(f"== {title}")
    for lname, rows in res.items():
        print(f"  Stoerung {lname}:")
        for k, v in rows.items():
            print(f"    {k:15s} Fehler {v['mean']:.2f} m (quer {v['lat']:.2f}, laengs {v['lon']:.2f}) | <0.5 m: {v['ok50']*100:4.1f} %  <1 m: {v['ok100']*100:4.1f} %")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default=os.path.expanduser("~/exp/feats"))
    ap.add_argument("--graph", default=os.path.expanduser("~/exp/graph"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--no_feat", action="store_true")
    ap.add_argument("--no_sem", action="store_true")
    ap.add_argument("--no_att", action="store_true")
    ap.add_argument("--no_coarse", action="store_true")
    ap.add_argument("--amp", type=float, default=2.0)
    ap.add_argument("--deg", type=float, default=3.0)
    ap.add_argument("--nr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    tr, va = load_split("train", args.feats, args.graph), load_split("val", args.feats, args.graph)
    model = Align(L=args.layers, use_feat=not args.no_feat, use_sem=not args.no_sem, use_att=not args.no_att, use_coarse=not args.no_coarse).to(dev)
    print("Parameter:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)
    S = tr["nodes"].shape[0]
    t0 = time.time()
    hist = []
    for step in range(1, args.steps + 1):
        model.train()
        idx = torch.randint(0, S, (args.bs,), device=dev, generator=gen)
        gt, tg0, valid = tr["nodes"][idx], tr["tangent"][idx], tr["mask"][idx]
        scale = torch.rand(args.bs, device=dev, generator=gen)
        p, tg = perturb(gt, tg0, args.amp, args.deg, args.nr, gen, scale)
        outs = model(tr["feats"][idx], tr["sem"][idx], p, tg, valid)
        v = valid.float()
        loss = sum(((o - gt).norm(dim=-1) * v).sum() / v.sum() for o in outs)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == 1:
            last = float(((outs[-1] - gt).norm(dim=-1) * v).sum() / v.sum())
            print(f"step {step:5d} loss {float(loss):.3f}  Fehler letzte Runde {last:.2f} m  ({time.time() - t0:.0f}s)", flush=True)
            hist.append((step, float(loss), last))
    res = evaluate(model, va)
    print_res(res, "Validierung (81 Samples, 3 Zufallssaaten je Stufe)")
    res_tr = evaluate(model, {k: v[:81] for k, v in tr.items()})
    print_res(res_tr, "Training (erste 81 Samples) zum Vergleich")
    json.dump(dict(val=res, train=res_tr, args=vars(args), hist=hist), open(f"{args.out}/results.json", "w"), indent=1)
    torch.save(model.state_dict(), f"{args.out}/model.pt")


if __name__ == "__main__":
    main()
