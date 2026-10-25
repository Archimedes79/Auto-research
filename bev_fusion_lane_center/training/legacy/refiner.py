"""Loop 3: Kaskaden-Verfeinerer auf Gitter-Vorschlaegen ('Pooling + Neuschaetzung, Bloecke hintereinander').
Eingabe: Vorschlags-Polylinien (proposals.py, Meter, Fahrtrichtung, 1 m Abstand), eingefrorene BEV-Merkmale (dec32 + logit7) + Heatmap/Richtung des dichten Kopfes (Gitter).
Block: je Punkt quer zur Linie (Normale) die Merkmalskarten an 7 seitlichen Versaetzen (+/-2,5 m) und grob an 5 Versaetzen (+/-6 m) abtasten ('Pooling'),
  Merkmale -> Einbettung -> 1-D-Faltungen laengs der Linie (Kontext) -> Neuschaetzung: seitlicher Versatz, Laengsversatz, Existenz je Punkt; Punkte verschieben, naechster Block tastet neu ab.
Verlust (ohne Zuordnung): Soll-Abstandsfeld an den verschobenen Punkten abtasten (differenzierbar) = Abstand zur wahren Mittellinie; Existenz (wahr, wenn Punkt <= 1 m vom Soll) mit Falschalarm-Gewicht lam_fp;
  Orientierung je Stueck (Soll-Richtungsfeld); Abstands-/Glattheits-Regularisierung. Aufruf: siehe main().
Inferenz -> verfeinerte Stuecke -> Graph (Orientierung, Schnitt bei geringer Existenz, Verbindungen) -> graph_eval."""
import bflc_paths as BP  # project paths from the location of this checkout
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402
import graph_from_dense as GD  # noqa: E402

HOME = os.path.expanduser("~")
OUT = BP.RUNS
PMAX, NMAX = int(os.environ.get("REF_PMAX", 48)), 100
OFFS = (-2.5, -1.5, -0.75, 0.0, 0.75, 1.5, 2.5)
COFFS = (-6.0, -3.0, 0.0, 3.0, 6.0)


# ------------------------------------------------------------------ Daten


class RData(torch.utils.data.Dataset):
    """sources: Liste dict(run, st, part): dichter Lauf <run> mit Menge <st> (prob_/heat_, dir_, ids_, props_<tag>_<st>.pkl), part = 'train' | 'val' (Merkmale/Soll)."""

    def __init__(self, sources, tag="relax", groups=("dec32", "logit7"), train=True, flip_aug=True, split="b123", jitter=0.0):
        self.items = []  # (source index, lokale Zeile)
        self.S = []
        for si, s in enumerate(sources):
            d = f"{OUT}/{s['run']}"
            pr = f"{d}/prob_{s['st']}.npy" if os.path.exists(f"{d}/prob_{s['st']}.npy") else f"{d}/heat_{s['st']}.npy"
            ids = np.load(f"{d}/ids_{s['st']}.npy")
            props = pickle.load(open(f"{d}/props_{tag}_{s['st']}.pkl", "rb"))
            self.S.append(dict(prob=pr, dirf=f"{d}/dir_{s['st']}.npy", ids=ids, props=props, part=s["part"], mm=None))
            self.items += [(si, k) for k in range(len(ids))]
        self.groups, self.train, self.flip_aug, self.split, self.jitter = groups, train, flip_aug, split, jitter
        self.feat = {}
        self.gt = {}

    def __len__(self):
        return len(self.items)

    def order_of(self, n):
        si, k = self.items[n]
        pts = self.S[si]["props"][k]["pts"]
        return sorted(range(len(pts)), key=lambda q: -len(pts[q]))[:PMAX]

    def edges_of(self, n):
        """Kanten des Vorschlagsgraphen in der sortierten Stueck-Nummerierung (Kanten zu abgeschnittenen Stuecken entfallen)."""
        si, k = self.items[n]
        inv = {o: a for a, o in enumerate(self.order_of(n))}
        return [(inv[i], inv[j]) for i, j in self.S[si]["props"][k]["edges"] if i in inv and j in inv]

    def _open(self, si):
        s = self.S[si]
        if s["mm"] is None:
            part = s["part"]
            if part not in self.feat:
                self.feat[part] = [np.load(f"{HOME}/exp_gcar_{self.split}/feats/{part}_{g}.npy", mmap_mode="r") for g in self.groups]
                if self.train:
                    self.gt[part] = (np.load(f"{HOME}/exp_crn_{self.split}/feats/{part}_gt_dist_exact.npy", mmap_mode="r"), np.load(f"{HOME}/exp_gcar_{self.split}/dir_{part}.npy", mmap_mode="r"))
            s["mm"] = (np.load(s["prob"], mmap_mode="r"), np.load(s["dirf"], mmap_mode="r"))
        return s

    def __getitem__(self, n):
        si, k = self.items[n]
        s = self._open(si)
        i = int(s["ids"][k])
        part = s["part"]
        x = np.concatenate([np.asarray(m[i], np.float32) for m in self.feat[part]], 0)
        o = dict(x=torch.from_numpy(x), heat=torch.from_numpy(np.asarray(s["mm"][0][k], np.float32)), dirf=torch.from_numpy(np.asarray(s["mm"][1][k], np.float32)))
        pcs = [s["props"][k]["pts"][q] for q in self.order_of(n)]
        pts = np.zeros((PMAX, NMAX, 2), np.float32)
        mask = np.zeros((PMAX, NMAX), bool)
        for a, q in enumerate(pcs):
            q = np.asarray(q, np.float32)[:NMAX]
            if self.train and self.flip_aug and np.random.rand() < 0.5:
                q = q[::-1]
            if self.train and self.jitter > 0 and len(q) >= 4 and np.random.rand() < 0.5:
                w = np.convolve(np.random.randn(len(q) + 14), np.ones(15) / 15, mode="valid")[: len(q)] * self.jitter * 3.0
                t = np.gradient(q, axis=0)
                t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-6)
                q = (q + w[:, None] * np.stack([-t[:, 1], t[:, 0]], 1)).astype(np.float32)
            pts[a, : len(q)] = q
            pts[a, len(q) :] = q[-1]
            mask[a, : len(q)] = True
        o.update(pts=torch.from_numpy(pts), mask=torch.from_numpy(mask), n=n, row=i)
        if self.train:
            gd, gr = self.gt[part]
            o["gdist"] = torch.from_numpy(np.asarray(gd[i], np.float32) * 0.5)
            o["gdir"] = torch.from_numpy(np.asarray(gr[i], np.float32))
        return o


def channel_stats(split="b123", groups=("dec32", "logit7"), n=300):
    f = f"{OUT}/refiner_stats_{'_'.join(groups)}.npz"
    if os.path.exists(f):
        z = np.load(f)
        return torch.from_numpy(z["mean"]), torch.from_numpy(z["std"])
    mm = [np.load(f"{HOME}/exp_gcar_{split}/feats/train_{g}.npy", mmap_mode="r") for g in groups]
    rows = np.linspace(0, mm[0].shape[0] - 1, n).astype(int)
    rows = rows[rows % 2 == 0]
    xs = np.stack([np.concatenate([np.asarray(m[i], np.float32) for m in mm], 0) for i in rows])
    mean, std = xs.mean((0, 2, 3)), np.maximum(xs.std((0, 2, 3)), 1e-4)
    np.savez(f, mean=mean, std=std)
    return torch.from_numpy(mean), torch.from_numpy(std)


# ------------------------------------------------------------------ Hilfen


def tangents(pts):
    prev = torch.cat([pts[:, :, :1], pts[:, :, :-1]], 2)
    nxt = torch.cat([pts[:, :, 1:], pts[:, :, -1:]], 2)
    t = nxt - prev
    n = t.norm(dim=-1, keepdim=True)
    t = torch.where(n > 1e-4, t / n.clamp_min(1e-6), torch.tensor([1.0, 0.0], device=pts.device).expand_as(t))
    return t


def sample_map(maps, pts, mode="bilinear"):
    """maps [B,C,200,200]; pts [B,P,N,2] (x = Zeile, y = Spalte, m) -> [B,P,N,C]"""
    B, P, N, _ = pts.shape
    grid = torch.stack([pts[..., 1] / 50.0, pts[..., 0] / 50.0], -1).reshape(B, P * N, 1, 2)
    s = F.grid_sample(maps, grid, mode=mode, padding_mode="border", align_corners=False)  # [B,C,P*N,1]
    return s[..., 0].permute(0, 2, 1).reshape(B, P, N, -1)


def pool(maps, pts, nrm, offs):
    B, P, N, _ = pts.shape
    off = torch.tensor(offs, device=pts.device, dtype=pts.dtype).view(1, 1, 1, -1, 1)
    q = pts[:, :, :, None, :] + off * nrm[:, :, :, None, :]
    grid = torch.stack([q[..., 1] / 50.0, q[..., 0] / 50.0], -1).reshape(B, P * N, len(offs), 2)
    s = F.grid_sample(maps, grid, mode="bilinear", padding_mode="zeros", align_corners=False)  # [B,C,P*N,O]
    return s.permute(0, 2, 3, 1).reshape(B, P, N, -1)


class ResConv1d(nn.Module):
    def __init__(self, D, k=5, dil=1):
        super().__init__()
        self.c = nn.Conv1d(D, D, k, padding=dil * (k // 2), dilation=dil)
        self.n = nn.GroupNorm(8, D)

    def forward(self, x, m):
        return x + F.relu(self.n(self.c(x * m))) * m


class Block(nn.Module):
    def __init__(self, cf, cc, D, offs=OFFS, coffs=COFFS):
        super().__init__()
        self.offs, self.coffs = offs, coffs
        self.inp = nn.Sequential(nn.Linear(len(offs) * cf + len(coffs) * cc + D, D), nn.LayerNorm(D), nn.ReLU())
        self.convs = nn.ModuleList([ResConv1d(D, 5, d) for d in (1, 2, 4, 1)])
        self.head = nn.Linear(D, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, mf, mc, pts, mask, emb):
        B, P, N, _ = pts.shape
        t = tangents(pts)
        nrm = torch.stack([-t[..., 1], t[..., 0]], -1)
        feat = torch.cat([pool(mf, pts, nrm, self.offs), pool(mc, pts, nrm, self.coffs), emb], -1)
        h = self.inp(feat)
        D = h.shape[-1]
        m = mask.reshape(B * P, 1, N).to(h.dtype)
        x = h.reshape(B * P, N, D).transpose(1, 2)
        for c in self.convs:
            x = c(x, m)
        h = x.transpose(1, 2).reshape(B, P, N, D)
        o = self.head(h)
        lat = torch.tanh(o[..., 0]) * 2.0
        lng = torch.tanh(o[..., 1]) * 0.6
        new = pts + lat[..., None] * nrm + lng[..., None] * t
        new = torch.where(mask[..., None], new, pts)
        return new, o[..., 2], h


class Refiner(nn.Module):
    def __init__(self, cin=39, c=32, D=96, blocks=3, extra=3):
        super().__init__()
        cin = cin + extra
        self.enc = nn.Sequential(nn.Conv2d(cin, 48, 3, padding=1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True), nn.Conv2d(48, 48, 3, padding=1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.fine = nn.Conv2d(48, c, 1)
        self.coarse = nn.Sequential(nn.AvgPool2d(4), nn.Conv2d(48, c, 3, padding=1), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([Block(c + extra, c, D) for _ in range(blocks)])
        self.orient = nn.Linear(D, 1)
        self.D = D

    def forward(self, x, heat, dirf, pts, mask):
        inp = torch.cat([x, heat[:, None], dirf], 1)
        e = self.enc(inp)
        mf = torch.cat([self.fine(e), heat[:, None], dirf], 1)
        mc = self.coarse(e)
        B, P, N, _ = pts.shape
        emb = torch.zeros(B, P, N, self.D, device=x.device, dtype=x.dtype)
        outs, p = [], pts
        for blk in self.blocks:
            p, ex, emb = blk(mf, mc, p, mask, emb)
            outs.append((p, ex))
        m = mask[..., None].to(emb.dtype)
        ori = self.orient((emb * m).sum(2) / m.sum(2).clamp_min(1.0))[..., 0]
        return outs, ori


# ------------------------------------------------------------------ Verlust


def losses(outs, ori, pts0, mask, gdist, gdir, a):
    gd = gdist[:, None]
    D0 = sample_map(gd, pts0)[..., 0]
    true0 = (D0 <= 3.0) & mask  # Vorschlagspunkt nahe genug an einer wahren Linie, um dorthin gezogen zu werden
    tot, parts = 0.0, {}
    w = [0.5, 0.75, 1.0][-len(outs):] if len(outs) <= 3 else [1.0] * len(outs)
    for bi, ((p, ex), wb) in enumerate(zip(outs, w)):
        D = sample_map(gd, p)[..., 0]
        geo = (F.smooth_l1_loss(D, torch.zeros_like(D), reduction="none", beta=1.0) * true0).sum() / true0.sum().clamp_min(1)
        d1 = p[:, :, 1:] - p[:, :, :-1]
        seg = (mask[:, :, 1:] & mask[:, :, :-1]).float()
        sp = ((d1.norm(dim=-1) - a.spacing).abs() * seg).sum() / seg.sum().clamp_min(1)
        d2 = p[:, :, 2:] - 2 * p[:, :, 1:-1] + p[:, :, :-2]
        seg2 = (mask[:, :, 2:] & mask[:, :, 1:-1] & mask[:, :, :-2]).float()
        sm = (d2.norm(dim=-1) * seg2).sum() / seg2.sum().clamp_min(1)
        y = ((D.detach() <= a.ex_tol) & mask).float()
        bce = F.binary_cross_entropy_with_logits(ex, y, reduction="none")
        wt = torch.where(y > 0.5, torch.ones_like(y), torch.full_like(y, a.lam_fp))
        exl = (bce * wt * mask).sum() / mask.sum().clamp_min(1)
        tot = tot + wb * (geo + a.w_sp * sp + a.w_sm * sm + a.w_ex * exl)
        if bi == len(outs) - 1:
            parts.update(geo=float(geo), sp=float(sp), sm=float(sm), ex=float(exl))
    # Orientierung (letzter Block): Soll-Richtung an den verschobenen Punkten
    p = outs[-1][0].detach()
    t = tangents(p)
    g = sample_map(gdir, p, mode="nearest")  # [B,P,N,2]
    Dn = sample_map(gd, p)[..., 0]
    ok = mask & (Dn <= 1.5) & (g.norm(dim=-1) > 0.5)
    dot = (t * g).sum(-1) * ok
    cnt = ok.sum(2)
    mean = dot.sum(2) / cnt.clamp_min(1)
    valid = (cnt >= 3) & (mean.abs() > 0.2)
    lab = (mean < 0).float()
    ol = (F.binary_cross_entropy_with_logits(ori, lab, reduction="none") * valid).sum() / valid.sum().clamp_min(1)
    tot = tot + a.w_or * ol
    parts["ori"] = float(ol)
    return tot, parts


# ------------------------------------------------------------------ Inferenz und Graph


@torch.no_grad()
def infer(model, ds, mean, std, bs=16, workers=4, dev="cuda"):
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers)
    res = []
    for b in dl:
        x = (b["x"].to(dev) - mean[None, :, None, None]) / std[None, :, None, None]
        heat, dirf, pts, mask = b["heat"].to(dev), b["dirf"].to(dev), b["pts"].to(dev), b["mask"].to(dev)
        outs, ori = model(x, heat, dirf, pts, mask)
        p, ex = outs[-1]
        # einmal je Batch auf die CPU holen (frueher: je Stueck ein .cpu() = tausende Synchronisierungen, unter Last sehr langsam)
        pts_c, p_c, ex_c, ori_c, m_c = pts.cpu().numpy(), p.cpu().numpy(), torch.sigmoid(ex).cpu().numpy(), torch.sigmoid(ori).cpu().numpy(), mask.cpu().numpy()
        for j in range(p.shape[0]):
            m = m_c[j]
            res.append(dict(edges0=ds.edges_of(int(b["n"][j])), p0=[pts_c[j, a][m[a]] for a in range(PMAX) if m[a].any()], p=[p_c[j, a][m[a]] for a in range(PMAX) if m[a].any()],
                            ex=[ex_c[j, a][m[a]] for a in range(PMAX) if m[a].any()], ori=[float(ori_c[j, a]) for a in range(PMAX) if m[a].any()]))
    return res


def suppress_duplicates(pieces, subs, exs, frac=0.7, tol=0.75):
    """Nach der Verfeinerung koennen zwei Vorschlaege auf dieselbe Soll-Linie gezogen werden: kuerzere Stuecke, die zu >= frac innerhalb tol m eines laengeren behaltenen liegen, entfallen."""
    from scipy.spatial import cKDTree

    if len(pieces) < 2 or frac <= 0:
        return pieces, subs, exs
    ln = [float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()) for q in pieces]
    keep, bank = [], np.zeros((0, 2))
    for k in np.argsort(ln)[::-1]:
        a = GE.resample(pieces[k], 0.5)
        if len(bank) and (cKDTree(bank).query(a)[0] <= tol).mean() >= frac:
            continue
        keep.append(int(k))
        bank = np.vstack([bank, a])
    keep.sort()
    remap = {o: n for n, o in enumerate(keep)}
    subs2 = {}
    for a_, lst in subs.items():
        l2 = [remap[x] for x in lst if x in remap]
        if l2:
            subs2[a_] = l2
    return [pieces[k] for k in keep], subs2, [exs[k] for k in keep]


def pieces_from_refined(r, tau_e=0.5, tau_o=0.5, min_pts=3, use_orient=True, dedupe=0.7):
    """verfeinerte Stuecke -> (Stuecke in Meter, subs: Stueck-Nr. -> Liste der Teilstuecke, flipped: umorientierte Stuecke, ex: Existenz je Teilstueck-Punkt)."""
    pieces, subs, flipped, exs = [], {}, set(), []
    for a, (p, ex, o) in enumerate(zip(r["p"], r["ex"], r["ori"])):
        if use_orient and o > tau_o:
            p, ex = p[::-1], ex[::-1]
            flipped.add(a)
        k = ex >= tau_e
        if not k.any():
            continue
        k = k.copy()
        for i in range(1, len(k) - 1):
            if not k[i] and k[i - 1] and k[i + 1]:
                k[i] = True
        i = 0
        while i < len(k):
            if k[i]:
                j = i
                while j < len(k) and k[j]:
                    j += 1
                if j - i >= min_pts:
                    subs.setdefault(a, []).append(len(pieces))
                    pieces.append(p[i:j].astype(np.float64))
                    exs.append(ex[i:j])
                i = j
            else:
                i += 1
    pieces, subs, exs = suppress_duplicates(pieces, subs, exs, dedupe)
    return pieces, subs, flipped, exs


def reduce_edges(edges):
    """transitive Reduktion: Kante (i, j) entfaellt, wenn es k mit (i, k) und (k, j) gibt (Abkuerzung ueber ein kurzes Zwischenstueck)."""
    succ = {}
    for i, j in edges:
        succ.setdefault(i, set()).add(j)
    out = []
    for i, j in edges:
        if any(k != j and j in succ.get(k, ()) for k in succ.get(i, ())):
            continue
        out.append((i, j))
    return sorted(set(out))


def graph_from_refined(r, tau_e=0.5, tau_o=0.5, min_pts=3, link=dict(link_m=1.5, gap_m=6.0, gap_deg=30.0, max_turn=100.0), use_orient=True, inherit=True, dedupe=0.7, reduce=False):
    """verfeinerte Stuecke -> Graph (Meter): Orientierung, Schnitt bei Existenz < tau_e, Kanten = vom Gitter-Graphen vererbte (Abzweige/Zusammenfuehrungen mit Ortskonvention) + Abstands-/Kurs-Verbindungen."""
    pieces, subs, flipped, _ = pieces_from_refined(r, tau_e, tau_o, min_pts, use_orient, dedupe)
    edges = set()
    if len(pieces) > 1:
        edges |= set(GD.link_edges([q / GD.PX for q in pieces], link["link_m"] / GD.PX, link["gap_m"] / GD.PX, link["gap_deg"], link["max_turn"]))
    if inherit:
        for i, j in r.get("edges0", []):
            if i in subs and j in subs and i not in flipped and j not in flipped:
                edges.add((subs[i][-1], subs[j][0]))
    return dict(pts=pieces, edges=reduce_edges(sorted(edges)) if reduce else sorted(edges))


def evaluate_refined(res, gts, with_route=True, workers=6, **kw):
    graphs = [graph_from_refined(r, **kw) for r in res]
    return GD.evaluate_parallel(graphs, gts, workers, with_route), graphs


# ------------------------------------------------------------------ Hauptprogramm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", default="", help="Quellen run:menge:teil[,..] z.B. e27_fold0:oof:train,...  (teil = train|val)")
    ap.add_argument("--eval", default="", help="Quellen wie --train, Name=run:menge:teil z.B. val=e27_gcar:val:val")
    ap.add_argument("--tag", default="relax")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--D", type=int, default=96)
    ap.add_argument("--lam_fp", type=float, default=3.0)
    ap.add_argument("--ex_tol", type=float, default=1.0)
    ap.add_argument("--w_ex", type=float, default=0.5)
    ap.add_argument("--w_sp", type=float, default=0.3)
    ap.add_argument("--w_sm", type=float, default=0.2)
    ap.add_argument("--w_or", type=float, default=0.2)
    ap.add_argument("--spacing", type=float, default=1.0)
    ap.add_argument("--tau_e", type=float, default=0.3, help="Existenz-Schwelle fuer die Auswertung im Lauf (genauer Sweep auf VAL danach)")
    ap.add_argument("--in_drop", type=float, default=0.0, help="Kanal-Ausfall (Dropout2d) auf den 39 Grundlage-Merkmalen im Training")
    ap.add_argument("--in_noise", type=float, default=0.0, help="Gauss-Rauschen (Std in normierten Einheiten) auf den Grundlage-Merkmalen im Training")
    ap.add_argument("--jitter", type=float, default=0.0, help="Std der seitlichen Zufallsverschiebung der Vorschlaege im Training (m)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init_from", default="")
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
    mean, std = channel_stats()
    mean, std = mean.cuda(), std.cuda()
    model = Refiner(39, 32, a.D, a.blocks).cuda()
    P("Parameter", sum(p.numel() for p in model.parameters()))
    if a.init_from:
        ck = torch.load(f"{OUT}/{a.init_from}/model.pt", map_location="cuda", weights_only=False)
        model.load_state_dict(ck["state"])
    step, t0, curve = 0, time.time(), []
    if a.steps > 0 and a.train:
        srcs = [dict(run=x.split(":")[0], st=x.split(":")[1], part=x.split(":")[2]) for x in a.train.split(",")]
        ds = RData(srcs, a.tag, train=True, jitter=a.jitter)
        P("Trainings-Samples", len(ds))
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=True, prefetch_factor=3)
        model.train()
        run = []
        while step < a.steps:
            for b in dl:
                x = (b["x"].cuda(non_blocking=True) - mean[None, :, None, None]) / std[None, :, None, None]
                if a.in_drop > 0:  # Kanalausfall auf den Stiftmerkmalen (Regularisierung gegen Anpassung an die Trainingsszenen)
                    x = F.dropout2d(x, a.in_drop, training=True)
                if a.in_noise > 0:
                    x = x + a.in_noise * torch.randn_like(x)
                heat, dirf, pts, mask = b["heat"].cuda(), b["dirf"].cuda(), b["pts"].cuda(), b["mask"].cuda()
                outs, ori = model(x, heat, dirf, pts, mask)
                loss, parts = losses(outs, ori, pts, mask, b["gdist"].cuda(), b["gdir"].cuda(), a)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                sched.step()
                run.append([float(loss)] + [parts[k] for k in ("geo", "sp", "sm", "ex", "ori")])
                step += 1
                if step % 100 == 0:
                    r = np.mean(run[-100:], 0)
                    curve.append((step, *[float(v) for v in r]))
                    P(f"Schritt {step}/{a.steps} Verlust {r[0]:.3f} (Abstand {r[1]:.3f}, Abst.-Reg {r[2]:.3f}, Glatt {r[3]:.3f}, Existenz {r[4]:.3f}, Orient. {r[5]:.3f}) {(time.time() - t0) / step:.2f} s/Schritt")
                if step >= a.steps:
                    break
        torch.save(dict(state=model.state_dict(), args=vars(a)), f"{OUT}/{a.name}/model.pt")
    res = dict(args=vars(a), curve=curve, params=sum(p.numel() for p in model.parameters()))
    for item in [x for x in a.eval.split(",") if x]:
        nm, src = item.split("=")
        run_, st_, part_ = src.split(":")
        ds = RData([dict(run=run_, st=st_, part=part_)], a.tag, train=False)
        t1 = time.time()
        r = infer(model, ds, mean, std)
        ids = np.load(f"{OUT}/{run_}/ids_{st_}.npy")
        gts = GD.load_gt_graphs(part_, ids)
        pickle.dump(r, open(f"{OUT}/{a.name}/refined_{nm}.pkl", "wb"))
        m, _ = evaluate_refined(r, gts, tau_e=a.tau_e)
        # Vergleich: Gitter-Vorschlaege unveraendert (Stuecke wie sie sind, gleiche Verbindungslogik)
        base = [dict(edges0=x["edges0"], p=x["p0"], ex=[np.ones(len(q)) for q in x["p0"]], ori=[0.0] * len(x["p0"])) for x in r]
        mb, _ = evaluate_refined(base, gts, tau_e=a.tau_e)
        res[nm] = dict(refined=m, proposals=mb)
        P(f"[{nm}] {len(ids)} Samples ({time.time() - t1:.0f} s): Vorschlaege F1@1 {mb['F1@1.0']:.3f} (P {mb['P@1.0']:.3f} R {mb['R@1.0']:.3f}) route F {mb['route_f']:.3f} -> verfeinert F1@1 {m['F1@1.0']:.3f} (P {m['P@1.0']:.3f} R {m['R@1.0']:.3f}) F1@0.5 {mb['F1@0.5']:.3f} -> {m['F1@0.5']:.3f} route F {m['route_f']:.3f}")
    json.dump(res, open(f"{OUT}/{a.name}/metrics.json", "w"), indent=1)
    P("FERTIG", time.strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()
