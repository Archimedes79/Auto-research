"""E32 ff.: duenner Spurgraph-Extraktor ('LaneFormer'). Eingefrorene BEV-Merkmale -> U-Netz-Rumpf (mit dichtem Hilfskopf) -> Deformable-Attention-Decoder
mit M Spur-Abfragen, je K Punkte (Polylinie in Fahrtrichtung) + Existenz + Nachfolger-Matrix (Splits/Merges explizit: Zeile i hat >= 2 Nachfolger / Spalte j >= 2 Vorgaenger).
Training: Hungarian-Zuordnung je Decoder-Schicht (Kosten: Existenz + L1 der Punkte), fokale Existenzverlust mit Falschalarm-Gewicht lam_fp (> 1 = Falschalarme teurer als Verpasstes),
L1 auf Punkten, BCE auf der Nachfolger-Matrix (nur Paare zugeordneter Abfragen), dichter Hilfskopf (Heatmap + Richtung, wie E27/E31) am Rumpf.
Soll: gtK_<part>.npz aus make_gt_arrays.py (Spuren in <= 20 m lange Teilstuecke, 11 Punkte je Stueck, Verkettung als Kanten).
Aufruf (Umgebung gcar):  lane_former.py --name e32_main --steps 12000 --lam_fp 3        Mini-Probe: --split mini --steps 300
Ergebnisse: out/loop2/<name>/{model.pt, metrics.json, log.txt, pred_<set>.npz}
"""
import bflc_paths as BP  # project paths from the location of this checkout
import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import align_tools as AT  # noqa: E402
import dense_head as DH  # noqa: E402
import graph_eval as GE  # noqa: E402
import lf_inputs as LI  # noqa: E402

HOME = os.path.expanduser("~")
OUT = BP.RUNS
K, MMAX = 11, 96


def inv_sig(x, eps=1e-4):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


# ------------------------------------------------------------------ Daten


class SData(torch.utils.data.Dataset):
    """Merkmale + Soll-Stuecke (+ dichte Ziele zum Trainieren). Schluessel: x, idx, pts [M,K,2] m, valid [M], adj [M,M]; train: heat, dirf;
    prior (PriorProvider): prior [3,200,200], noise [3]; pose (PoseTable): xprev [C,200,200] (Merkmale des Vorgaengers, Null wenn keiner), Q [4], s [2], has [1]."""

    def __init__(self, groups, part, indices, split="b123", train=True, prior=None, pose=None):
        self.base = DH.Data("gcar", groups, part, indices, split, target=train, with_dir=train)
        g = np.load(f"{HOME}/exp_gcar_{split}/gtK_{part}.npz")
        self.pts, self.valid, self.adj = g["pts"], g["valid"], g["adj"]
        self.train = train
        self.prior = prior
        self.pose = pose
        self.part = part

    def __len__(self):
        return len(self.base)

    def __getitem__(self, k):
        it = self.base[k]
        i = int(self.base.idx[k])
        x = it[0]
        o = dict(x=x, idx=i, pts=torch.from_numpy(self.pts[i]), valid=torch.from_numpy(self.valid[i]), adj=torch.from_numpy(self.adj[i]))
        if self.train:
            o["heat"], o["dirf"] = it[1], it[2]
        if self.prior is not None:
            o["prior"], o["noise"] = self.prior.get(self.part, i)
        if self.pose is not None:
            Q, s_, has = self.pose.rel(self.part, i)
            j = self.pose.prev(self.part, i)
            xp = torch.cat([torch.from_numpy(np.asarray(m[j], np.float32)) for m in self.base.mm], 0) if j >= 0 else torch.zeros_like(x)
            o.update(xprev=xp, Q=torch.from_numpy(Q), s=torch.from_numpy(s_), has=torch.from_numpy(has))
        return o


def prep_input(b, mean, std, cfg, dev="cuda"):
    """Rohbatch -> (x [B,C,H,W] normiert, extra [B,Ce,H,W] oder None, info). cfg: align (bool), logit_off (Index des ersten logit7-Kanals), temporal ('A' oder '')."""
    raw = b["x"].to(dev, non_blocking=True)
    x = (raw - mean[None, :, None, None]) / std[None, :, None, None]
    ex, info = [], {}
    if "prior" in b:
        pr = b["prior"].to(dev, non_blocking=True)
        if cfg.get("align"):
            off = cfg["logit_off"]
            pr, est = AT.align(pr, LI.obs_from_logits(raw[:, off : off + 7]))
            info["est"] = est
        ex.append(pr)
    if "xprev" in b and cfg.get("temporal") == "A":
        xp = b["xprev"].to(dev, non_blocking=True)
        xpw, valid = LI.fold_prev(xp, b["Q"].to(dev), b["s"].to(dev))
        xpw = (xpw - mean[None, :, None, None]) / std[None, :, None, None]
        has = b["has"].to(dev).view(-1, 1, 1, 1)
        ex += [xpw * valid * has, valid * has]
    return x, (torch.cat(ex, 1) if ex else None), info


# ------------------------------------------------------------------ Bausteine


def pos2emb(xy, dim=128, temp=10000.0):
    """xy [...,2] in [0,1] -> Sinus-Einbettung [..., dim]."""
    nf = dim // 4
    f = temp ** (torch.arange(nf, device=xy.device, dtype=xy.dtype) / nf)
    a = xy[..., None] * 2 * math.pi / f  # [...,2,nf]
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


class MSDA(nn.Module):
    """Mehrskalen-Deformable-Attention in reinem PyTorch (grid_sample). query [B,Q,d], ref [B,Q,2] (x_norm = Zeile, y_norm = Spalte, [0,1]), mem Liste [B,c_l,h_l,w_l] (eigene Kanalzahl je Ebene,
    c_l durch H teilbar; je Ebene eine Ausgabe-Projektion c_l -> d, damit die grosse Ebene (200 x 200) schmal bleibt)."""

    def __init__(self, d=128, chs=(32, 64, 128), H=8, P=4):
        super().__init__()
        self.d, self.H, self.L, self.P = d, H, len(chs), P
        L = self.L
        self.off = nn.Linear(d, H * L * P * 2)
        self.wts = nn.Linear(d, H * L * P)
        self.out = nn.ModuleList([nn.Linear(c, d) for c in chs])
        nn.init.zeros_(self.off.weight)
        th = torch.arange(H, dtype=torch.float32) * (2 * math.pi / H)
        grid = torch.stack([th.cos(), th.sin()], -1)
        grid = (grid / grid.abs().max(-1, keepdim=True)[0]).view(H, 1, 1, 2).repeat(1, L, P, 1)
        for p in range(P):
            grid[:, :, p] *= p + 1
        with torch.no_grad():
            self.off.bias.copy_(grid.flatten())
        nn.init.zeros_(self.wts.weight)
        nn.init.zeros_(self.wts.bias)

    def forward(self, query, ref, mem):
        B, Q, d = query.shape
        H, L, P = self.H, self.L, self.P
        off = self.off(query).view(B, Q, H, L, P, 2)
        w = self.wts(query).view(B, Q, H, L * P).softmax(-1).view(B, Q, H, L, P)
        out = 0
        for l in range(L):
            _, c, h_l, w_l = mem[l].shape
            v = mem[l].reshape(B * H, c // H, h_l, w_l)
            loc = ref[:, :, None, None, :] + off[:, :, :, l] / torch.tensor([h_l, w_l], device=query.device, dtype=query.dtype)  # [B,Q,H,P,2] (x_norm, y_norm)
            grid = torch.stack([loc[..., 1], loc[..., 0]], -1) * 2 - 1  # grid_sample: (Spalte, Zeile)
            grid = grid.permute(0, 2, 1, 3, 4).reshape(B * H, Q, P, 2)
            s = F.grid_sample(v, grid.to(v.dtype), mode="bilinear", padding_mode="zeros", align_corners=False)  # [B*H, c/H, Q, P]
            ww = w[:, :, :, l].permute(0, 2, 1, 3).reshape(B * H, 1, Q, P).to(s.dtype)
            o = (s * ww).sum(-1).reshape(B, c, Q).transpose(1, 2)
            out = out + self.out[l](o)
        return out


class DecLayer(nn.Module):
    def __init__(self, d=128, H=8, chs=(32, 64, 128), P=4, ff=512, drop=0.0):
        super().__init__()
        self.pt_attn = nn.MultiheadAttention(d, H, dropout=drop, batch_first=True)
        self.inst_attn = nn.MultiheadAttention(d, H, dropout=drop, batch_first=True)
        self.msda = MSDA(d, chs, H, P)
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.ReLU(inplace=True), nn.Dropout(drop), nn.Linear(ff, d))
        self.n1, self.n2, self.n3, self.n4 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, tok, pos, ref, mem):
        B, M, Kp, d = tok.shape
        t = tok.reshape(B * M, Kp, d)
        p = pos.reshape(B * M, Kp, d)
        q = t + p
        t = self.n1(t + self.pt_attn(q, q, t, need_weights=False)[0])
        inst = t.view(B, M, Kp, d).mean(2)
        pi = pos.mean(2)
        qi = inst + pi
        upd = self.inst_attn(qi, qi, inst, need_weights=False)[0]
        t = self.n2(t + upd.reshape(B * M, 1, d))
        c = self.msda((t + p).view(B, M * Kp, d), ref.reshape(B, M * Kp, 2), mem).reshape(B * M, Kp, d)
        t = self.n3(t + c)
        t = self.n4(t + self.ffn(t))
        return t.view(B, M, Kp, d)


class AdjHead(nn.Module):
    """Nachfolger-Matrix i -> j aus Instanz-Einbettungen und Geometrie (Ende von i gegen Anfang von j)."""

    def __init__(self, d=128, dh=64):
        super().__init__()
        self.a, self.b = nn.Linear(d, dh), nn.Linear(d, dh)
        self.mlp = nn.Sequential(nn.Linear(2 * dh + 6, 128), nn.ReLU(inplace=True), nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, e, pts):
        pts = pts.detach()
        end, start = pts[:, :, -1], pts[:, :, 0]
        te = F.normalize(pts[:, :, -1] - pts[:, :, -3], dim=-1, eps=1e-3)
        ts = F.normalize(pts[:, :, 2] - pts[:, :, 0], dim=-1, eps=1e-3)
        delta = start[:, None, :, :] - end[:, :, None, :]  # [B,i,j,2]
        dist = delta.norm(dim=-1, keepdim=True)
        dn = delta / (dist + 1e-3)
        cos = (te[:, :, None] * ts[:, None]).sum(-1, keepdim=True)
        ce = (te[:, :, None] * dn).sum(-1, keepdim=True)
        cs = (dn * ts[:, None]).sum(-1, keepdim=True)
        geo = torch.cat([delta / 10.0, dist / 10.0, cos, ce, cs], -1)
        M = e.shape[1]
        a, b = self.a(e), self.b(e)
        x = torch.cat([a[:, :, None].expand(-1, -1, M, -1), b[:, None].expand(-1, M, -1, -1), geo], -1)
        return self.mlp(x)[..., 0]


class Trunk(nn.Module):
    """U-Netz wie dense_head.UNet, gibt Karten auf 200, 100 und 50 Punkte Kantenlaenge (c, 2c, 4c Kanaele) zurueck."""

    def __init__(self, cin, base=32):
        super().__init__()
        c1, c2, c3, c4 = base, 2 * base, 4 * base, 4 * base
        self.stem = DH.block(cin, c1)
        self.d1 = nn.Sequential(nn.MaxPool2d(2), DH.block(c1, c2))
        self.d2 = nn.Sequential(nn.MaxPool2d(2), DH.block(c2, c3))
        self.d3 = nn.Sequential(nn.MaxPool2d(2), DH.block(c3, c4))
        self.u2 = DH.block(c4 + c3, c3)
        self.u1 = DH.block(c3 + c2, c2)
        self.u0 = DH.block(c2 + c1, c1)
        self.chs = (c1, c2, c3)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        up = lambda a, b: torch.cat([F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1)
        y2 = self.u2(up(x3, x2))
        y1 = self.u1(up(y2, x1))
        y0 = self.u0(up(y1, x0))
        return y0, y1, y2


class LaneFormer(nn.Module):
    def __init__(self, cin, d=128, M=96, Kp=K, n_layers=4, H=8, P=4, base=32, anchors=None, drop=0.0, cin_extra=0):
        super().__init__()
        self.M, self.Kp, self.d, self.cin_extra = M, Kp, d, cin_extra
        self.trunk = Trunk(cin + 2 + cin_extra, base)
        c1, c2, c3 = self.trunk.chs
        self.dense = nn.Conv2d(c1, 3, 1)  # Heatmap + Richtung (Hilfskopf = Raster-Ansatz auf demselben Rumpf)
        self.proj = nn.ModuleList([nn.Sequential(nn.Conv2d(c, c, 1), nn.GroupNorm(8, c)) for c in (c1, c2, c3)])
        self.layers = nn.ModuleList([DecLayer(d, H, (c1, c2, c3), P, 4 * d, drop) for _ in range(n_layers)])
        self.query = nn.Embedding(M, d)
        self.point_embed = nn.Parameter(torch.randn(Kp, d) * 0.02)
        if anchors is None:
            anchors = torch.rand(M, Kp, 2)
        self.anchor = nn.Parameter(inv_sig(anchors))  # [M,K,2] (x_norm, y_norm), lernbar
        self.pos_mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, d))
        self.reg = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, 2)) for _ in range(n_layers)])
        self.cls = nn.ModuleList([nn.Linear(d, 1) for _ in range(n_layers)])
        self.adj = AdjHead(d)
        self.track_proj = nn.Linear(d, d)  # Zeitverfolgung B: Einbettung des Vorgaengers -> Inhalt der Verfolger-Abfrage
        nn.init.constant_(self.cls[0].bias, -2.0)
        for c in self.cls:
            nn.init.constant_(c.bias, -2.0)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 200), torch.linspace(-1, 1, 200), indexing="ij")
        self.register_buffer("coord", torch.stack([yy, xx])[None], persistent=False)  # Zeile = x, Spalte = y

    def anchors(self):
        return torch.sigmoid(self.anchor)

    def forward(self, x, extra=None, init_ref=None, init_content=None):
        B = x.shape[0]
        parts = [x, self.coord.expand(B, -1, -1, -1).to(x.dtype)]
        if extra is not None:
            parts.append(extra.to(x.dtype))
        y0, y1, y2 = self.trunk(torch.cat(parts, 1))
        dense = self.dense(y0)
        mem = [self.proj[0](y0), self.proj[1](y1), self.proj[2](y2)]
        q = self.query.weight[None].expand(B, -1, -1)
        ref = torch.sigmoid(self.anchor)[None].expand(B, -1, -1, -1)
        if init_ref is not None:
            ref = init_ref
        if init_content is not None:
            q = q + init_content
        tok = q[:, :, None] + self.point_embed[None, None]
        outs = []
        for li, layer in enumerate(self.layers):
            pos = self.pos_mlp(pos2emb(ref, self.d))
            tok = layer(tok, pos, ref, mem)
            delta = self.reg[li](tok)
            ref = torch.sigmoid(inv_sig(ref.detach().float()) + delta.float())
            inst = tok.mean(2)
            outs.append(dict(cls=self.cls[li](inst)[..., 0], pts=ref * 100.0 - 50.0, emb=inst))
        adj = self.adj(outs[-1]["emb"], outs[-1]["pts"])
        return outs, adj, dense


# ------------------------------------------------------------------ Zuordnung und Verlust


@torch.no_grad()
def hungarian(cls_logit, pts, gt_pts, gt_valid, w_c=1.0, w_p=5.0):
    p = cls_logit.sigmoid()
    cost_pts = (pts[:, :, None] - gt_pts[:, None]).abs().sum(-1).mean(-1) / 5.0  # [B,M,G] (m / 5)
    cost = (w_c * (-p[:, :, None]) + w_p * cost_pts).cpu().numpy()
    res = []
    for b in range(cls_logit.shape[0]):
        g = int(gt_valid[b].sum())
        if g == 0:
            res.append((np.zeros(0, int), np.zeros(0, int)))
            continue
        r, c = linear_sum_assignment(cost[b][:, :g])
        res.append((r, c))
    return res


def set_loss(out, gt_pts, gt_valid, lam_fp, w_pts, gamma=2.0, wpos=1.0):
    cls, pts = out["cls"], out["pts"]
    match = hungarian(cls, pts, gt_pts, gt_valid)
    t = torch.zeros_like(cls)
    bi, qi, gi = [], [], []
    for b, (r, c) in enumerate(match):
        t[b, r] = 1.0
        bi += [b] * len(r)
        qi += list(r)
        gi += list(c)
    p = torch.sigmoid(cls)
    pos = wpos * t * ((1 - p) ** gamma) * F.softplus(-cls)
    neg = lam_fp * (1 - t) * (p ** gamma) * F.softplus(cls)
    n_gt = max(int(gt_valid.sum()), 1)
    l_cls = (pos + neg).sum() / n_gt
    if len(bi):
        bi_t, qi_t, gi_t = torch.tensor(bi, device=cls.device), torch.tensor(qi, device=cls.device), torch.tensor(gi, device=cls.device)
        l_pts = ((pts[bi_t, qi_t] - gt_pts[bi_t, gi_t]).abs().sum(-1).mean(-1) / 5.0).mean()
    else:
        l_pts = pts.sum() * 0
    return l_cls + w_pts * l_pts, l_cls.detach(), l_pts.detach(), match


def adj_loss(adj_logit, match, gt_adj, w_pos):
    B, M, _ = adj_logit.shape
    tgt = torch.zeros_like(adj_logit)
    msk = torch.zeros_like(adj_logit)
    for b, (r, c) in enumerate(match):
        if len(r) == 0:
            continue
        rr, cc = torch.as_tensor(r, device=adj_logit.device), torch.as_tensor(c, device=adj_logit.device)
        sub = gt_adj[b][cc][:, cc].float()  # [n,n] Soll-Nachfolger zwischen den zugeordneten Stuecken
        tgt[b][rr[:, None], rr[None]] = sub
        msk[b][rr[:, None], rr[None]] = 1.0
    w = msk * (1 + (w_pos - 1) * tgt)
    return (F.binary_cross_entropy_with_logits(adj_logit, tgt, reduction="none") * w).sum() / msk.sum().clamp_min(1.0)


# ------------------------------------------------------------------ Vorhersage -> Graph


def to_graph(cls_prob, pts, adj_prob, tau_c=0.5, tau_a=0.5, dedupe=0.0, max_gap=4.0):
    keep = np.nonzero(cls_prob >= tau_c)[0]
    keep = keep[np.argsort(-cls_prob[keep])]
    if dedupe > 0 and len(keep) > 1:
        from scipy.spatial import cKDTree

        kept, bank = [], np.zeros((0, 2))
        for k in keep:
            a = GE.resample(pts[k], 0.5)
            if len(bank) and (cKDTree(bank).query(a)[0] <= 1.0).mean() >= dedupe:
                continue
            kept.append(k)
            bank = np.vstack([bank, a])
        keep = np.array(sorted(kept), int)
    else:
        keep = np.sort(keep)
    pos = {int(k): n for n, k in enumerate(keep)}
    edges = [(pos[int(a)], pos[int(b)]) for a in keep for b in keep if a != b and adj_prob[a, b] >= tau_a and (max_gap <= 0 or np.linalg.norm(pts[a][-1] - pts[b][0]) <= max_gap)]
    return dict(pts=[pts[k].astype(np.float64) for k in keep], edges=edges)


def snap_polyline(P, heat, iters=3, half=2.0, step=0.25, damp=0.7, thr=0.3):
    """Raster-Einrasten der Polylinie (m): jeder Punkt wandert quer zur Linie zum gewichteten Schwerpunkt des Heatmap-Kamms (aktive Kontur), Endpunkte bleiben laengs fest. heat [200,200] (Zeile = x, Spalte = y)."""
    from scipy.ndimage import map_coordinates

    P = np.asarray(P, np.float64).copy()
    offs = np.arange(-half, half + 1e-6, step)
    for _ in range(iters):
        t = np.gradient(P, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-6)
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        cand = P[:, None, :] + offs[None, :, None] * nrm[:, None, :]  # [K,S,2]
        rc = (cand + 50.0) / 0.5 - 0.5
        h = map_coordinates(np.asarray(heat, np.float32), [rc[..., 0].ravel(), rc[..., 1].ravel()], order=1, mode="nearest").reshape(len(P), len(offs))
        w = np.where(h >= thr, h**2, 0.0)
        ok = w.sum(1) > 1e-6
        shift = np.zeros(len(P))
        shift[ok] = (w[ok] * offs[None]).sum(1) / w[ok].sum(1)
        P = P + damp * shift[:, None] * nrm
    return P


def snap_graph(g, heat, **kw):
    return dict(pts=[snap_polyline(p, heat, **kw) if len(p) >= 3 else p for p in g["pts"]], edges=g["edges"])


@torch.no_grad()
def infer(model, ds, bs=16, workers=4, dev="cuda", mean=None, std=None, cfg=None):
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers)
    R = dict(cls=[], pts=[], adj=[], heat=[], dirf=[], idx=[], est=[], noise=[])
    for b in dl:
        x, ex, info = prep_input(b, mean, std, cfg or {}, dev)
        outs, adj, dense = model(x, ex)
        if "est" in info:
            R["est"].append(info["est"].cpu())
        if "noise" in b:
            R["noise"].append(b["noise"])
        R["cls"].append(outs[-1]["cls"].sigmoid().cpu())
        R["pts"].append(outs[-1]["pts"].cpu())
        R["adj"].append(adj.sigmoid().cpu())
        R["heat"].append(dense[:, 0].sigmoid().half().cpu())
        R["dirf"].append(dense[:, 1:3].half().cpu())
        R["idx"].append(b["idx"])
    return {k: torch.cat(v).numpy() for k, v in R.items() if len(v)}


def graphs_from(R, tau_c, tau_a, dedupe=0.0, max_gap=4.0):
    return [to_graph(R["cls"][i], R["pts"][i], R["adj"][i], tau_c, tau_a, dedupe, max_gap) for i in range(len(R["idx"]))]


def load_gt_graphs(part, ids, split="b123"):
    G = pickle.load(open(f"{HOME}/exp_gcar_{split}/graph_gt_{part}.pkl", "rb"))
    return [G[int(i)] for i in ids]


def quick_eval(R, gts, taus=(0.3, 0.5, 0.7), tau_a=0.5, with_route=False, n=None, workers=1):
    out = {}
    n = n or len(R["idx"])
    for tc in taus:
        gr = graphs_from(R, tc, tau_a)[:n]
        if workers > 1:
            import graph_from_dense as GD

            out[str(tc)] = GD.evaluate_parallel(gr, gts[:n], workers, with_route)
        else:
            out[str(tc)] = GE.evaluate(gr, gts[:n], with_route=with_route)
    return out


# ------------------------------------------------------------------ Hauptprogramm


def init_anchors(sd, M, seed=0):
    rng = np.random.RandomState(seed)
    cand = np.argwhere(sd.valid)
    sel = cand[rng.choice(len(cand), M, replace=False)]
    pts = np.stack([sd.pts[i, j] for i, j in sel])  # [M,K,2] m
    return torch.from_numpy((pts + 50.0) / 100.0).float()


def load_partial(model, state):
    """Gewichte laden; abweichende Eingangskanalzahl der ersten Faltung: alte Kanaele kopieren, neue mit 0 starten (Modell = altes Modell am Anfang)."""
    own = model.state_dict()
    n_ok = n_part = n_skip = 0
    for k, v in state.items():
        if k not in own:
            n_skip += 1
        elif own[k].shape == v.shape:
            own[k] = v
            n_ok += 1
        elif v.dim() == 4 and own[k].shape[0] == v.shape[0] and own[k].shape[1] > v.shape[1] and own[k].shape[2:] == v.shape[2:]:
            nw = torch.zeros_like(own[k])
            nw[:, : v.shape[1]] = v
            own[k] = nw
            n_part += 1
        else:
            n_skip += 1
    model.load_state_dict(own)
    return n_ok, n_part, n_skip


def propagate(model, out_p, Q, s, has, n_track):
    """Zeitverfolgung B: die n_track sichersten Abfragen des Vorgaengers (Polylinie nach der Eigenbewegung ins aktuelle System, Einbettung) belegen die ersten Abfrage-Plaetze."""
    B = Q.shape[0]
    top = out_p["cls"].detach().sigmoid().topk(n_track, dim=1).indices  # [B,N]
    g = lambda t: torch.gather(t, 1, top.view(B, n_track, *([1] * (t.dim() - 2))).expand(-1, -1, *t.shape[2:]))
    pts = g(out_p["pts"].detach().float())  # [B,N,K,2] m (Vorgaenger-System)
    emb = g(out_p["emb"].detach().float())
    pt = torch.einsum("bij,bnkj->bnki", Q.view(B, 2, 2), pts) + s.view(B, 1, 1, 2)
    ref_t = ((pt + 50.0) / 100.0).clamp(0.02, 0.98)
    anc = model.anchors()[None].expand(B, -1, -1, -1).clone()
    h = has.view(B, 1, 1, 1)
    init_ref = anc.clone()
    init_ref[:, :n_track] = torch.where(h > 0, ref_t, anc[:, :n_track])
    init_content = torch.zeros(B, model.M, model.d, device=Q.device)
    init_content[:, :n_track] = model.track_proj(emb) * has.view(B, 1, 1)
    return init_ref, init_content


@torch.no_grad()
def infer_seq(model, ds, mean, std, cfg, n_track, dev="cuda", workers=2):
    """Auswertung mit Zeitverfolgung B: Bilder der Reihe nach, Vorhersage des Vorgaengers wird weitergereicht (nur wenn das Vorgaenger-Sample unmittelbar davor verarbeitet wurde)."""
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=workers)
    R = dict(cls=[], pts=[], adj=[], heat=[], dirf=[], idx=[])
    store = {}
    for b in dl:
        x, ex, info = prep_input(b, mean, std, cfg, dev)
        i = int(b["idx"][0])
        j = ds.pose.prev(ds.part, i)
        if j in store and n_track > 0:
            ir, ic = propagate(model, store[j], b["Q"].to(dev), b["s"].to(dev), torch.ones(1, device=dev), n_track)
        else:
            ir = ic = None
        outs, adj, dense = model(x, ex, ir, ic)
        store[i] = dict(cls=outs[-1]["cls"], pts=outs[-1]["pts"], emb=outs[-1]["emb"])
        store.pop(i - 6, None)
        R["cls"].append(outs[-1]["cls"].sigmoid().cpu())
        R["pts"].append(outs[-1]["pts"].cpu())
        R["adj"].append(adj.sigmoid().cpu())
        R["heat"].append(dense[:, 0].sigmoid().half().cpu())
        R["dirf"].append(dense[:, 1:3].half().cpu())
        R["idx"].append(b["idx"])
    return {k: torch.cat(v).numpy() for k, v in R.items() if len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--feats", default="dec32,logit7")
    ap.add_argument("--split", default="b123", choices=["b123", "mini"])
    ap.add_argument("--steps", type=int, default=12000, help="0 = nur auswerten (mit --init_from)")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lr_trunk", type=float, default=1.0, help="Faktor auf die Lernrate des Rumpfs (+ dichter Kopf), z.B. 0.4 bei Warmstart")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--M", type=int, default=96)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--lam_fp", type=float, default=1.0, help="Gewicht der Falschalarm-Seite im Existenzverlust (Negative)")
    ap.add_argument("--wpos", type=float, default=1.0)
    ap.add_argument("--w_pts", type=float, default=5.0)
    ap.add_argument("--w_adj", type=float, default=1.0)
    ap.add_argument("--adj_pos", type=float, default=3.0)
    ap.add_argument("--w_dense", type=float, default=1.0)
    ap.add_argument("--dense_lam", type=float, default=1.0)
    ap.add_argument("--train_stride", type=int, default=1)
    ap.add_argument("--eval", default="train_eval,val")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", type=int, default=0)
    ap.add_argument("--mon", type=int, default=1500, help="Zwischenauswertung alle n Schritte (Geometrie, 120 Samples)")
    ap.add_argument("--init_from", default="", help="Lauf, dessen Gewichte geladen werden (Feinabstimmung)")
    ap.add_argument("--init_dense", default="", help="dichter Lauf (dense_head.py, gleiche --base, mit --dir), dessen Rumpf + Kopf als Start dient")
    ap.add_argument("--prior", default="none", choices=["none", "map", "gt_oracle", "map_wrong"])
    ap.add_argument("--prior_xy", type=float, default=0.0, help="Pose-Stoerung Training: Std Verschiebung (m)")
    ap.add_argument("--prior_deg", type=float, default=0.0, help="Pose-Stoerung Training: Std Drehung (Grad)")
    ap.add_argument("--prior_degrade", type=float, default=0.0, help="Anteil geloeschter Kartenbloecke (veraltete Karte), Training")
    ap.add_argument("--align", type=int, default=0, help="Vorwissen vor der Nutzung an die Beobachtung ausrichten (Training und Auswertung)")
    ap.add_argument("--eval_prior_xy", type=float, default=-1.0, help="Pose-Stoerung bei der Auswertung (-1 = wie Training)")
    ap.add_argument("--eval_prior_deg", type=float, default=-1.0)
    ap.add_argument("--eval_degrade", type=float, default=-1.0)
    ap.add_argument("--temporal", default="", choices=["", "A", "B"], help="A: Merkmale des Vorgaengers gefaltet als Zusatzkanaele; B: Abfragen des Vorgaengers weiterreichen")
    ap.add_argument("--n_track", type=int, default=40)
    ap.add_argument("--t_gap", type=int, default=2, help="Zeitverfolgung: Vorgaenger = t_gap Schluessel-Bilder zurueck (2: nur gerade Zeilen sind extrahiert)")
    ap.add_argument("--save_arrays", type=int, default=1)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(f"{OUT}/{a.name}", exist_ok=True)
    log = open(f"{OUT}/{a.name}/log.txt", "a")

    def P(*s):
        t = " ".join(str(x) for x in s)
        print(t, flush=True)
        log.write(t + "\n")
        log.flush()

    P("==", time.strftime("%H:%M:%S"), vars(a))
    groups = a.feats.split(",")
    sizes = dict(dec32=32, fused32=32, logit7=7)
    logit_off = sum(sizes[g] for g in groups[: groups.index("logit7")]) if "logit7" in groups else 0
    cfg = dict(align=bool(a.align), logit_off=logit_off, temporal=a.temporal if a.temporal == "A" else "")
    if a.split == "b123":
        sp = json.load(open(BP.ROOT + "/_to_delete/recon/sb/loop2/splits.json"))
        tr_idx = np.arange(sp["n"]["train"])
        sets = {"train_eval": ("train", sp["train_eval"]), "val": ("val", sp["val"]), "test": ("val", sp["test"])}
    else:
        n_tr, n_va = 323, 81
        tr_idx = np.arange(n_tr)
        sets = {"train_eval": ("train", list(range(0, n_tr, 4))), "val": ("val", list(range(n_va)))}
    if a.train_stride > 1:
        tr_idx = tr_idx[:: a.train_stride]
    ex_xy = a.prior_xy if a.eval_prior_xy < 0 else a.eval_prior_xy
    ex_deg = a.prior_deg if a.eval_prior_deg < 0 else a.eval_prior_deg
    ex_dg = a.prior_degrade if a.eval_degrade < 0 else a.eval_degrade
    prior_tr = LI.PriorProvider(a.split, a.prior, a.prior_xy, a.prior_deg, a.prior_degrade, True) if a.prior != "none" else None
    prior_ev = LI.PriorProvider(a.split, a.prior, ex_xy, ex_deg, ex_dg, False) if a.prior != "none" else None
    pose = LI.PoseTable(a.split, a.t_gap) if a.temporal else None
    ds = SData(groups, "train", tr_idx, a.split, True, prior_tr, pose)
    mean, std = DH.channel_stats(ds.base)
    cin = int(mean.shape[0])
    n_extra = (3 if a.prior != "none" else 0) + (cin + 1 if a.temporal == "A" else 0)
    model = LaneFormer(cin, a.d, a.M, K, a.layers, 8, 4, a.base, init_anchors(ds, a.M, a.seed), cin_extra=n_extra).cuda()
    if a.init_from:
        ck = torch.load(f"{OUT}/{a.init_from}/model.pt", map_location="cpu", weights_only=False)
        P("Gewichte aus", a.init_from, "(unveraendert/teilweise/uebersprungen):", load_partial(model, ck["state"]))
        mean, std = ck["mean"][:cin], ck["std"][:cin]
        if a.temporal == "B":  # Verfolger-Inhalt startet bei 0: das Modell beginnt als das Einzelbild-Modell
            nn.init.zeros_(model.track_proj.weight)
            nn.init.zeros_(model.track_proj.bias)
    if a.init_dense:
        ck = torch.load(f"{OUT}/{a.init_dense}/model.pt", map_location="cpu", weights_only=False)
        st = {}
        for k, v in ck["state"].items():
            if k.startswith("head."):
                st["dense." + k[5:]] = v
            else:
                st["trunk." + k] = v
        P("Rumpf + dichter Kopf aus", a.init_dense, "(unveraendert/teilweise/uebersprungen):", load_partial(model, st))
        mean, std = ck["mean"][:cin], ck["std"][:cin]
    P("Eingangskanaele", cin, "+ Zusatz", n_extra, "Parameter", sum(p.numel() for p in model.parameters()), "Trainings-Samples", len(ds))
    mean, std = mean.cuda(), std.cuda()
    step, t0, run, curve, mon = 0, time.time(), [], [], []
    if a.steps > 0:
        tp = [p_ for n_, p_ in model.named_parameters() if n_.startswith(("trunk.", "dense."))]
        op = [p_ for n_, p_ in model.named_parameters() if not n_.startswith(("trunk.", "dense."))]
        opt = torch.optim.AdamW([dict(params=tp, lr=a.lr * a.lr_trunk), dict(params=op, lr=a.lr)], weight_decay=a.wd)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr * a.lr_trunk, a.lr], total_steps=a.steps, pct_start=0.05)
        scaler = torch.cuda.amp.GradScaler(enabled=bool(a.amp))
        dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=True, prefetch_factor=4)
        mon_sets = {}
        for nm in ("train_eval", "val"):
            part, idx = sets[nm]
            sel = idx[:120] if a.temporal == "B" else idx[:: max(len(idx) // 120, 1)][:120]
            if a.temporal == "B" and nm == "train_eval":
                continue
            mon_sets[nm] = (SData(groups, part, sel, a.split, False, prior_ev, pose), load_gt_graphs(part, sel, a.split))
        model.train()
        while step < a.steps:
            for b in dl:
                x, ex, _ = prep_input(b, mean, std, cfg)
                gp, gv, ga = b["pts"].cuda(), b["valid"].cuda(), b["adj"].cuda()
                ir = ic = None
                if a.temporal == "B":
                    with torch.no_grad():
                        model.eval()
                        xp = (b["xprev"].cuda() - mean[None, :, None, None]) / std[None, :, None, None]
                        with torch.autocast("cuda", dtype=torch.float16, enabled=bool(a.amp)):
                            op, _, _ = model(xp, None)
                        model.train()
                    ir, ic = propagate(model, op[-1], b["Q"].cuda(), b["s"].cuda(), b["has"].cuda().view(-1), a.n_track)
                with torch.autocast("cuda", dtype=torch.float16, enabled=bool(a.amp)):
                    outs, adj, dense = model(x, ex, ir, ic)
                tot = 0
                for li, o in enumerate(outs):
                    o = dict(cls=o["cls"].float(), pts=o["pts"].float())
                    l, c_, p_, match = set_loss(o, gp, gv, a.lam_fp, a.w_pts, wpos=a.wpos)
                    tot = tot + l
                lc, lp = float(c_), float(p_)
                l_adj = adj_loss(adj.float(), match, ga, a.adj_pos)
                l_dense = DH.loss_fn(dense[:, 0].float(), b["heat"].cuda(), a.dense_lam, 1.0, 2.0) + DH.dir_loss(dense[:, 1:3].float(), b["dirf"].cuda())
                loss = tot + a.w_adj * l_adj + a.w_dense * l_dense
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                run.append([float(loss), lc, lp, float(l_adj), float(l_dense)])
                step += 1
                if step % 100 == 0:
                    r = np.mean(run[-100:], 0)
                    curve.append((step, *[float(v) for v in r]))
                    P(f"Schritt {step}/{a.steps} Verlust {r[0]:.3f} (Existenz {r[1]:.3f}, Punkte {r[2]:.3f}, Nachfolger {r[3]:.3f}, dicht {r[4]:.3f}) {(time.time() - t0) / step:.2f} s/Schritt")
                if a.mon and step % a.mon == 0 and step < a.steps:
                    rec = {}
                    for nm, (dss, gts) in mon_sets.items():
                        R = infer_seq(model, dss, mean, std, cfg, a.n_track, workers=2) if a.temporal == "B" else infer(model, dss, workers=2, mean=mean, std=std, cfg=cfg)
                        q = quick_eval(R, gts, taus=(0.5,), with_route=False)["0.5"]
                        rec[nm] = (q["F1@1.0"], q["P@1.0"], q["R@1.0"])
                        model.train()
                    mon.append((step, rec))
                    P(f"  [Zwischenstand {step}] " + "  ".join(f"{k}: F1 {v[0]:.3f} (P {v[1]:.3f}, R {v[2]:.3f})" for k, v in rec.items()))
                    torch.save(dict(state=model.state_dict(), mean=mean.cpu(), std=std.cpu(), args=vars(a)), f"{OUT}/{a.name}/model.pt")
                if step >= a.steps:
                    break
        torch.save(dict(state=model.state_dict(), mean=mean.cpu(), std=std.cpu(), args=vars(a)), f"{OUT}/{a.name}/model.pt")
    res = dict(args=vars(a), curve=curve, mon=mon, train_s=time.time() - t0, params=sum(p.numel() for p in model.parameters()))
    for nm in a.eval.split(","):
        if nm not in sets:
            continue
        part, idx = sets[nm]
        dse = SData(groups, part, idx, a.split, False, prior_ev, pose)
        R = infer_seq(model, dse, mean, std, cfg, a.n_track, workers=a.workers) if a.temporal == "B" else infer(model, dse, workers=a.workers, mean=mean, std=std, cfg=cfg)
        gts = load_gt_graphs(part, R["idx"], a.split)
        if a.save_arrays:
            np.savez_compressed(f"{OUT}/{a.name}/pred_{nm}.npz", **{k: (v.astype(np.float16) if k in ("cls", "adj") else v) for k, v in R.items() if k not in ("heat", "dirf")})
            np.save(f"{OUT}/{a.name}/heat_{nm}.npy", R["heat"])
            np.save(f"{OUT}/{a.name}/dir_{nm}.npy", R["dirf"])
            np.save(f"{OUT}/{a.name}/ids_{nm}.npy", R["idx"])
        res[nm] = quick_eval(R, gts, taus=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8), with_route=False, workers=6)
        if "est" in R:
            err = R["est"] - R["noise"]
            res[nm + "_align"] = dict(err_median=np.median(np.abs(err), 0).tolist(), pos_err_median=float(np.median(np.linalg.norm(err[:, :2], axis=1))), pos_err_before_median=float(np.median(np.linalg.norm(R["noise"][:, :2], axis=1))))
        best = max(res[nm].items(), key=lambda kv: kv[1]["F1@1.0"])
        P(f"[{nm}] {len(gts)} Samples; bester Existenz-Schwellwert {best[0]}: F1 @0,5/1/1,5 = {best[1]['F1@0.5']:.3f} / {best[1]['F1@1.0']:.3f} / {best[1]['F1@1.5']:.3f}, P/R@1 {best[1]['P@1.0']:.3f}/{best[1]['R@1.0']:.3f}, 1:1 {best[1]['F1o2o@1.0']:.3f}, Splits F1 {best[1]['split_F1']:.3f}")
    json.dump(res, open(f"{OUT}/{a.name}/metrics.json", "w"), indent=1)
    P("FERTIG", time.strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()
