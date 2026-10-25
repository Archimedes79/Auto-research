"""Loop 15 (E120): differentiable position refinement of the trace-network graph.
The wavefront of the trace network (E94) places vertices by argmax over heading bins / step classes plus small residuals; training is teacher-forced, so no loss ever sees the finished graph.
Here the finished graph (vertices, edges, step tokens of E94) is the START of a MapTR-style decoder: L layers, each layer samples the BEV features at the CURRENT vertex positions (bilinear, differentiable),
mixes the vertices with a graph-biased self-attention and predicts a continuous displacement (iterative reference-point refinement). The loss sits on the finished polylines
(point-to-polyline distance in both directions, no hard assignment) and therefore on the positions; downstream losses can reach the positions through the same path.
Usage (gcar, GPU): p15_refine.py --name e120_a [--epochs 8] [--detach 0] [--layers 3] [--stem init|scratch|frozen] [--limit 0]
Writes out/loop2/<name>/{model.pt, log.txt, result.json}"""
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402
import p10_infer as PI  # noqa: E402
import p10_net as PN  # noqa: E402

HOME = os.path.expanduser("~")
TOKDIR = f"{HOME}/exp_gcar_b123/p10"
OUT = PI.OUT
DU = (-2.0, 0.0, 2.0)
DV = (-2.5, -1.5, -0.75, 0.0, 0.75, 1.5, 2.5)
NPAT = len(DU) * len(DV)
BUCKETS = torch.tensor([1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 11.0, 15.0, 20.0])


# ------------------------------------------------------------------ data
class Frames(torch.utils.data.Dataset):
    def __init__(self, specs, limit=0, train=True):
        """specs: list of (token file stem, EvalSet split)"""
        self.items, self.es, self.src = [], {}, []
        for name, split in specs:
            z = pickle.load(open(f"{TOKDIR}/{name}.pkl", "rb"))
            self.es[split] = PI.EvalSet(split)
            its = z["items"]
            if limit:
                its = its[:limit]
            for x in its:
                self.items.append(x)
                self.src.append(split)
        self.train = train

    def __len__(self):
        return len(self.items)

    def get_x(self, es, k):
        row = es.rows[k]
        x = np.concatenate([np.asarray(m[row], np.float32) for m in es.mm], 0)
        x = (x - es.mean[:, None, None]) / es.std[:, None, None]
        x = np.concatenate([x, np.asarray(es.prob[k], np.float32)[None], np.asarray(es.dirf[k], np.float32)], 0)
        return torch.from_numpy(x.astype(np.float16))

    def __getitem__(self, i):
        it = self.items[i]
        es = self.es[self.src[i]]
        x = self.get_x(es, it["k"])
        gt = es.gt[it["row"]]
        G, _, _ = GE.dense_points(gt, 0.5)
        G = np.asarray(G, np.float32).reshape(-1, 2)
        e = np.array(it["edges"], np.int64).reshape(-1, 2)
        return dict(x=x, xy=torch.from_numpy(np.asarray(it["xy"], np.float32)), tok=torch.from_numpy(np.asarray(it["tok"], np.float16)), has=torch.from_numpy(np.asarray(it["has"], np.float32)),
                    edges=torch.from_numpy(e), gt=torch.from_numpy(G), idx=i)


def collate(b):
    B = len(b)
    N = max(1, max(len(o["xy"]) for o in b))
    E = max(1, max(len(o["edges"]) for o in b))
    Gm = max(1, max(len(o["gt"]) for o in b))
    o = dict(x=torch.stack([q["x"] for q in b]), xy=torch.zeros(B, N, 2), tok=torch.zeros(B, N, 128), has=torch.zeros(B, N), vm=torch.zeros(B, N), edges=torch.zeros(B, E, 2, dtype=torch.long), em=torch.zeros(B, E),
             gt=torch.zeros(B, Gm, 2), gm=torch.zeros(B, Gm), idx=torch.tensor([q["idx"] for q in b]))
    for i, q in enumerate(b):
        n, e, g = len(q["xy"]), len(q["edges"]), len(q["gt"])
        o["xy"][i, :n], o["tok"][i, :n], o["has"][i, :n], o["vm"][i, :n] = q["xy"], q["tok"].float(), q["has"], 1
        if e:
            o["edges"][i, :e], o["em"][i, :e] = q["edges"], 1
        if g:
            o["gt"][i, :g], o["gm"][i, :g] = q["gt"], 1
    return o


# ------------------------------------------------------------------ model
def tangents(p, edges, em):
    """unit tangent per vertex from the incident edges (differentiable in p). p [B,N,2], edges [B,E,2]"""
    a, b = edges[..., 0], edges[..., 1]
    pa = torch.gather(p, 1, a[..., None].expand(-1, -1, 2))
    pb = torch.gather(p, 1, b[..., None].expand(-1, -1, 2))
    d = (pb - pa) * em[..., None]
    d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-3) * em[..., None]
    acc = torch.zeros_like(p)
    acc = acc.scatter_add(1, a[..., None].expand(-1, -1, 2), d).scatter_add(1, b[..., None].expand(-1, -1, 2), d)
    n = acc.norm(dim=-1, keepdim=True)
    t = torch.where(n > 1e-3, acc / n.clamp(min=1e-3), torch.tensor([1.0, 0.0], device=p.device).expand_as(acc))
    return t


class RLayer(nn.Module):
    def __init__(self, d=128, H=4, c0=67, c1=96, cp=24, du_max=1.5):
        super().__init__()
        self.H = H
        self.du_max = du_max
        self.p0, self.p1 = nn.Conv2d(c0, cp, 1), nn.Conv2d(c1, cp, 1)
        self.samp = nn.Sequential(nn.Linear(NPAT * 2 * cp, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Sequential(nn.Linear(6, 64), nn.GELU(), nn.Linear(64, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.bias = nn.Embedding(len(BUCKETS) + 1, H)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.delta = nn.Linear(d, 2)
        self.keep = nn.Linear(d, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        self.register_buffer("du", torch.tensor(DU))
        self.register_buffer("dv", torch.tensor(DV))
        self.register_buffer("bk", BUCKETS.clone())

    def forward(self, h, p, t, f0, f1, vm):
        B, N, _ = p.shape
        nrm = torch.stack([-t[..., 1], t[..., 0]], -1)
        off = self.du[:, None, None] * 0 + 0  # placeholder to keep the shape logic explicit
        uu = self.du[:, None].expand(len(DU), len(DV)).reshape(-1)
        vv = self.dv[None, :].expand(len(DU), len(DV)).reshape(-1)
        q = p[:, :, None, :] + uu[None, None, :, None] * t[:, :, None, :] + vv[None, None, :, None] * nrm[:, :, None, :]  # [B,N,P,2]
        g = torch.cat([PN.smp(self.p0(f0), q), PN.smp(self.p1(f1), q)], -1)  # [B,N,P,2cp]
        pf = torch.cat([p / 50.0, t, (p.norm(dim=-1, keepdim=True) / 50.0), torch.zeros_like(p[..., :1])], -1)
        h = h + self.samp(g.flatten(2)) + self.pos(pf)
        qkv = self.qkv(self.n1(h)).view(B, N, 3, self.H, -1).permute(2, 0, 3, 1, 4)
        dist = torch.cdist(p, p)
        bi = torch.bucketize(dist, self.bk)
        bias = self.bias(bi).permute(0, 3, 1, 2)  # [B,H,N,N]
        mask = (vm[:, None, :, None] > 0) & (vm[:, None, None, :] > 0)
        bias = bias.masked_fill(~mask, -1e4)
        a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=bias).transpose(1, 2).reshape(B, N, -1)
        h = h + self.o(a)
        h = h + self.ff(self.n2(h))
        d2 = torch.tanh(self.delta(h)) * torch.tensor([self.du_max, 2.5], device=h.device)  # (along, across) in metres
        return h, d2, self.keep(h)[..., 0]


class RefineNet(nn.Module):
    def __init__(self, layers=3, d=128, detach=False, stem="init", stem_from="e94_a", du_max=1.5):
        super().__init__()
        self.detach = detach
        self.stem = PN.Stem()
        if stem in ("init", "frozen"):
            sd = torch.load(f"{OUT}/{stem_from}/model.pt", map_location="cpu")
            self.stem.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("stem.")})
        self.freeze_stem = stem == "frozen"
        self.tok = nn.Sequential(nn.Linear(128, d), nn.GELU(), nn.Linear(d, d))
        self.has_emb = nn.Parameter(torch.zeros(d))
        self.layers = nn.ModuleList([RLayer(d, du_max=du_max) for _ in range(layers)])

    def forward(self, x, xy, tok, has, vm, edges, em):
        self.stem.eval()
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_stem):
            with torch.autocast("cuda", dtype=torch.float16):
                feats, _, _ = self.stem(x)
            f0, f1 = feats[0].float(), feats[1].float()
        h = self.tok(tok) * has[..., None] + self.has_emb * (1 - has[..., None])
        p = xy
        ps, ks = [], []
        for li, layer in enumerate(self.layers):
            pin = p.detach() if (self.detach and li > 0) else p
            t = tangents(pin, edges, em)
            h, d2, kp = layer(h, pin, t, f0, f1, vm)
            nrm = torch.stack([-t[..., 1], t[..., 0]], -1)
            p = pin + (d2[..., 0:1] * t + d2[..., 1:2] * nrm) * vm[..., None]
            ps.append(p)
            ks.append(kp)
        return ps, ks


# ------------------------------------------------------------------ losses
def seg_dist(g, a, b):
    """distance of points g [m,2] to the segments (a,b) [E,2]: min over segments -> [m]"""
    ab = b - a
    L2 = (ab * ab).sum(-1).clamp(min=1e-6)
    ga = g[:, None, :] - a[None]
    t = ((ga * ab[None]).sum(-1) / L2[None]).clamp(0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return (g[:, None, :] - proj).norm(dim=-1).min(1)[0]


def frame_loss(p, edges, em, G, gm, p_in, cap=3.0, beta=0.25):
    """p [N,2] positions of one frame, edges [E,2], G [m,2] ground-truth points. Returns precision loss, recall loss, and mean distances for logging"""
    e = edges[em > 0]
    if len(e) == 0 or gm.sum() == 0:
        z = p.sum() * 0
        return z, z
    Gv = G[gm > 0]
    a, b = p[e[:, 0]], p[e[:, 1]]
    fr = torch.tensor([0.0, 1 / 3, 2 / 3, 1.0], device=p.device)[:, None, None]
    Q = (a[None] + fr * (b - a)[None]).reshape(-1, 2)
    with torch.no_grad():
        a0, b0 = p_in[e[:, 0]], p_in[e[:, 1]]
        Q0 = (a0[None] + fr * (b0 - a0)[None]).reshape(-1, 2)
        mq = torch.cdist(Q0, Gv).min(1)[0] <= cap
        mg = seg_dist(Gv, a0, b0) <= cap
    dq = torch.cdist(Q, Gv).min(1)[0].clamp(max=cap)
    dg = seg_dist(Gv, a, b).clamp(max=cap)
    lp = F.smooth_l1_loss(dq[mq], torch.zeros_like(dq[mq]), beta=beta) if mq.any() else p.sum() * 0
    lr = F.smooth_l1_loss(dg[mg], torch.zeros_like(dg[mg]), beta=beta) if mg.any() else p.sum() * 0
    return lp, lr


def keep_target(p, G, gm, thr=1.0):
    Gv = G[gm > 0]
    if len(Gv) == 0:
        return torch.zeros(len(p), device=p.device)
    return (torch.cdist(p.detach(), Gv).min(1)[0] <= thr).float()


def batch_loss(model, bt, dev, w=(0.3, 0.5, 1.0), w_keep=0.2):
    x, xy, tok, has, vm = (bt[k].to(dev) for k in ("x", "xy", "tok", "has", "vm"))
    edges, em = bt["edges"].to(dev), bt["em"].to(dev)
    ps, ks = model(x, xy, tok, has, vm, edges, em)
    return loss_from(ps, ks, bt, dev, w, w_keep)


def loss_from(ps, ks, bt, dev, w=(0.3, 0.5, 1.0), w_keep=0.2):
    xy, vm = bt["xy"].to(dev), bt["vm"].to(dev)
    edges, em, gt, gm = (bt[k].to(dev) for k in ("edges", "em", "gt", "gm"))
    L = len(ps)
    ww = w if len(w) == L else [1.0] * L
    tot, lp_s, lr_s, kb_s = 0.0, 0.0, 0.0, 0.0
    B = xy.shape[0]
    for b in range(B):
        n = int(vm[b].sum())
        if n == 0:
            continue
        for li in range(L):
            lp, lr = frame_loss(ps[li][b, :n], edges[b], em[b], gt[b], gm[b], xy[b, :n])
            tot = tot + ww[li] * (lp + lr) / B
            if li == L - 1:
                lp_s += float(lp) / B
                lr_s += float(lr) / B
        tgt = keep_target(ps[-1][b, :n], gt[b], gm[b])
        kl = F.binary_cross_entropy_with_logits(ks[-1][b, :n], tgt)
        tot = tot + w_keep * kl / B
        kb_s += float(kl) / B
    return tot, dict(prec=lp_s, rec=lr_s, keep=kb_s)


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def refine_split(model, ds, dev, bs=8, workers=4, n_layers=None):
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, collate_fn=collate)
    model.eval()
    out = {}
    for bt in dl:
        x, xy, tok, has, vm = (bt[k].to(dev) for k in ("x", "xy", "tok", "has", "vm"))
        edges, em = bt["edges"].to(dev), bt["em"].to(dev)
        ps, ks = model(x, xy, tok, has, vm, edges, em)
        pf, kf = ps[-1].cpu().numpy(), ks[-1].sigmoid().cpu().numpy()
        for b, i in enumerate(bt["idx"].tolist()):
            n = len(ds.items[i]["xy"])
            out[i] = (pf[b, :n].astype(np.float64), kf[b, :n])
    return out


def to_graph(item, xy):
    pts = [np.array([xy[v] for v in p], np.float64) for p in item["pieces"]]
    return dict(pts=pts, edges=[], nodes=item["pieces"])


def evaluate_split(ds, ref, split_es, extra=True):
    """ref: dict idx -> (xy_refined, keep). Returns metrics of the refined and the input graphs on the same frames"""
    gts, gin, gref = [], [], []
    for i, it in enumerate(ds.items):
        gts.append(split_es[ds.src[i]].gt[it["row"]])
        gin.append(to_graph(it, np.asarray(it["xy"], np.float64)))
        gref.append(to_graph(it, ref[i][0]))
    res = {}
    for nm, g in (("input", gin), ("refined", gref)):
        r = GE.evaluate(g, gts, with_route=False)
        res[nm] = {k: float(v) for k, v in r.items() if k in ("P@0.5", "R@0.5", "F1@0.5", "P@1.0", "R@1.0", "F1@1.0", "P@1.5", "R@1.5", "F1@1.5", "dir_acc", "lanes_found", "false_len_frac", "split6_F1", "merge6_F1")}
    # vertex error to the ground truth (vertices within 2 m of it)
    from scipy.spatial import cKDTree

    d_in, d_rf = [], []
    for i, it in enumerate(ds.items):
        G, _, _ = GE.dense_points(gts[i], 0.5)
        if len(G) == 0 or len(it["xy"]) == 0:
            continue
        t = cKDTree(G)
        a = t.query(np.asarray(it["xy"]))[0]
        b = t.query(ref[i][0])[0]
        m = a <= 2.0
        d_in.append(a[m])
        d_rf.append(b[m])
    d_in, d_rf = np.concatenate(d_in), np.concatenate(d_rf)
    res["vertex_err"] = dict(input_mean=float(d_in.mean()), refined_mean=float(d_rf.mean()), input_median=float(np.median(d_in)), refined_median=float(np.median(d_rf)),
                             within05_in=float((d_in <= 0.5).mean()), within05_ref=float((d_rf <= 0.5).mean()), n=int(len(d_in)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--detach", type=int, default=0)
    ap.add_argument("--stem", default="init")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--du_max", type=float, default=1.5, help="largest displacement ALONG the lane per layer (m); 0 = lateral refinement only")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda"
    od = f"{OUT}/{a.name}"
    os.makedirs(od, exist_ok=True)
    log = open(f"{od}/log.txt", "a")

    def P(*s):
        m = " ".join(str(x) for x in s)
        print(m, flush=True)
        log.write(m + "\n")
        log.flush()

    model = RefineNet(layers=a.layers, detach=bool(a.detach), stem=a.stem, du_max=a.du_max).to(dev)
    P("params", sum(p.numel() for p in model.parameters()), "args", vars(a))
    val_ds = Frames([("tok_e94_a_val_clean", "val_clean")], limit=a.limit, train=False)
    test_ds = Frames([("tok_e94_a_test_clean", "test_clean")], limit=a.limit, train=False)
    if not a.eval_only:
        tr = Frames([("tok_e94_a_oof0", "oof0"), ("tok_e94_a_oof1", "oof1"), ("tok_e94_a_oof2", "oof2")], limit=a.limit)
        P("train frames", len(tr), "val", len(val_ds), "test", len(test_ds))
        dl = torch.utils.data.DataLoader(tr, batch_size=a.bs, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True, persistent_workers=a.workers > 0)
        stem_params = list(model.stem.parameters())
        rest = [p for n, p in model.named_parameters() if not n.startswith("stem.")]
        opt = torch.optim.AdamW([dict(params=stem_params, lr=a.lr * 0.3), dict(params=rest, lr=a.lr)], weight_decay=1e-4)
        total = a.epochs * len(dl)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr * 0.3, a.lr], total_steps=total, pct_start=0.1)
        step, t0, best = 0, time.time(), 1e9
        val_small = Frames([("tok_e94_a_val_clean", "val_clean")], limit=0, train=False)
        val_small.items = val_small.items[::4]
        val_small.src = val_small.src[::4]
        for ep in range(a.epochs):
            model.train()
            model.stem.eval()
            agg = dict(loss=0.0, prec=0.0, rec=0.0, keep=0.0)
            nb = 0
            for bt in dl:
                loss, st = batch_loss(model, bt, dev)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                sched.step()
                step += 1
                nb += 1
                agg["loss"] += float(loss)
                for k_ in st:
                    agg[k_] += st[k_]
                if step % 100 == 0:
                    P(f"ep {ep} step {step}/{total} loss {agg['loss'] / nb:.4f} prec {agg['prec'] / nb:.4f} rec {agg['rec'] / nb:.4f} keep {agg['keep'] / nb:.4f} {time.time() - t0:.0f} s")
            ref = refine_split(model, val_small, dev, workers=a.workers)
            r = evaluate_split(val_small, ref, val_small.es)
            P(f"== epoch {ep} val(1/4) F1@1 input {r['input']['F1@1.0']:.4f} refined {r['refined']['F1@1.0']:.4f}  F1@0.5 {r['input']['F1@0.5']:.4f} -> {r['refined']['F1@0.5']:.4f}  vertex err {r['vertex_err']['input_mean']:.3f} -> {r['vertex_err']['refined_mean']:.3f}")
            score = -(r["refined"]["F1@0.5"] + r["refined"]["F1@1.0"])
            if score < best:
                best = score
                torch.save(model.state_dict(), f"{od}/model.pt")
                P("   saved best")
    sd = torch.load(f"{od}/model.pt", map_location=dev)
    model.load_state_dict(sd)
    res = {}
    for nm, ds in (("val", val_ds), ("test", test_ds)):
        ref = refine_split(model, ds, dev, workers=a.workers)
        res[nm] = evaluate_split(ds, ref, ds.es)
        pickle.dump({i: (ref[i][0].astype(np.float32), ref[i][1].astype(np.float32)) for i in ref}, open(f"{od}/refined_{nm}.pkl", "wb"))
        P(nm, json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in res[nm].items()}))
    json.dump(dict(res=res, args=vars(a)), open(f"{od}/result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
