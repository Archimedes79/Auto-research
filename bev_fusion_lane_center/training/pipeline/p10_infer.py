"""Loop 10: wavefront inference of the learned trace network and evaluation against the rule pipeline (same rows, same protocol).
Usage (gcar, GPU): p10_infer.py --name e94_a --split val_clean [--n 200] [--tau_seed 0.3] [--tau_on 0.5] [--route] [--tag a]"""
import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402
import p8_lib as L  # noqa: E402
import p10_expert as EX  # noqa: E402
import p10_net as PN  # noqa: E402
import p10_wave as W  # noqa: E402

OUT = L.OUT
HOME = os.path.expanduser("~")
BANDS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
M_CV = 160


def load_maps(split):
    """heat / direction maps of the frozen dense head for a split of the cache (val_clean / test_clean) -> (prob, dir, ids)"""
    if split == "val_clean":
        d = f"{OUT}/e37_final"
        return np.load(f"{d}/prob_val.npy", mmap_mode="r"), np.load(f"{d}/dir_val.npy", mmap_mode="r"), np.load(f"{d}/ids_val.npy")
    d = f"{OUT}/e57_stored"
    return np.load(f"{d}/prob_test.npy", mmap_mode="r"), np.load(f"{d}/dir_test.npy", mmap_mode="r"), np.load(f"{d}/ids_test.npy")


class EvalSet:
    def __init__(self, split):
        if split.startswith("oof"):  # out-of-fold rows of the train part (fold f held out of the trace-network training), dense maps of the cross-fitted dense head f
            f = int(split[3:])
            full = L.load_src("train_oof")
            n0 = sum(len(np.load(f"{OUT}/e37_fold{q}/ids_oof.npy")) for q in range(f))
            ids = np.load(f"{OUT}/e37_fold{f}/ids_oof.npy")
            sel = list(range(n0, n0 + len(ids)))
            assert [int(i) for i in full["ids"][n0 : n0 + len(ids)]] == [int(i) for i in ids]
            src = dict(part="train", ids=[int(i) for i in ids], graphs=[full["graphs"][k] for k in sel], pprob=[full["pprob"][k] for k in sel], pdir=[full["pdir"][k] for k in sel])
            self.prob, self.dirf = np.load(f"{OUT}/e37_fold{f}/prob_oof.npy", mmap_mode="r"), np.load(f"{OUT}/e37_fold{f}/dir_oof.npy", mmap_mode="r")
        else:
            src = L.load_src(split)
            self.prob, self.dirf, ids = load_maps(split)
            assert [int(i) for i in ids] == [int(i) for i in src["ids"]]
        self.src, self.part = src, src["part"]
        self.rows = [int(r) for r in src["ids"]]
        self.mm = [np.load(f"{L.D}/feats/{self.part}_{g}.npy", mmap_mode="r") for g in ("dec32", "logit7")]
        z = np.load(f"{OUT}/refiner_stats_dec32_logit7.npz")
        self.mean, self.std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
        self.gt = L.cache("gt", self.part)

    def batch(self, ks, dev="cuda"):
        xs = []
        for k in ks:
            x = np.concatenate([np.asarray(m[self.rows[k]], np.float32) for m in self.mm], 0)
            x = (x - self.mean[:, None, None]) / self.std[:, None, None]
            xs.append(np.concatenate([x, np.asarray(self.prob[k], np.float32)[None], np.asarray(self.dirf[k], np.float32)], 0))
        return torch.from_numpy(np.stack(xs)).to(dev)


@torch.no_grad()
def make_seeds(seed_lg, dirh, tau=0.3, k_max=48, nms=7):
    p = torch.sigmoid(seed_lg.float())
    mx = F.max_pool2d(p[:, None], nms, 1, nms // 2)[:, 0]
    keep = (p == mx) & (p > tau)
    out = []
    dn = F.normalize(dirh.float(), dim=1)
    for b in range(p.shape[0]):
        idx = keep[b].nonzero()
        if len(idx) == 0:
            out.append([])
            continue
        sc = p[b][idx[:, 0], idx[:, 1]]
        o = sc.argsort(descending=True)[:k_max]
        idx, sc = idx[o], sc[o]
        pos = (idx.float() + 0.5) * 0.5 - 50.0
        d = dn[b][:, idx[:, 0], idx[:, 1]].T
        th = torch.atan2(d[:, 1], d[:, 0])
        out.append([(pos[j].cpu().numpy().astype(np.float64), float(th[j]), float(sc[j])) for j in range(len(idx))])
    return out


class NetPolicy:
    def __init__(self, net, feats, tau_on=0.5, tau_slope=0.0, dev="cuda", own_guard=True, ev_bias=None):
        self.net, self.feats, self.dev = net, feats, dev
        self.tau_on, self.tau_slope = tau_on, tau_slope
        self.ev_bias = ev_bias

    @torch.no_grad()
    def decide(self, items, fr):
        B = self.feats[0].shape[0]
        by = [[] for _ in range(B)]
        for j, it in enumerate(items):
            by[it["b"]].append(j)
        N = max(len(x) for x in by)
        dev = self.dev
        pos = torch.zeros(B, N, 2)
        th = torch.zeros(B, N)
        flag = torch.ones(B, N)
        hist = torch.zeros(B, N, 6, 4)
        valid = torch.zeros(B, N, dtype=torch.bool)
        for b, js in enumerate(by):
            for n, j in enumerate(js):
                it = items[j]
                pos[b, n] = torch.from_numpy(np.asarray(it["pos"], np.float32))
                th[b, n] = it["theta"]
                flag[b, n] = it["flag"]
                valid[b, n] = True
                hh = it["hist"]
                cur = np.asarray(it["pos"])
                prev = [np.asarray(x[:2]) for x in hh]
                # vertices before the current position, nearest first
                prev = prev[:-1][::-1] if len(prev) and np.linalg.norm(prev[-1] - cur) < 0.05 else prev[::-1]
                last = cur
                u_, v_ = np.array([math.cos(it["theta"]), math.sin(it["theta"])]), np.array([-math.sin(it["theta"]), math.cos(it["theta"])])
                for h, pv in enumerate(prev[:6]):
                    rel = pv - cur
                    hist[b, n, h] = torch.tensor([float(rel @ u_) / 10.0, float(rel @ v_) / 10.0, min(float(np.linalg.norm(pv - last)) / 6.0, 1.0), 1.0])
                    last = pv
        # context points around every head from the frame clouds
        cap = max(max(f.nc for f in fr), 1)
        cloud = torch.zeros(B, cap, 6)
        nc = torch.zeros(B, dtype=torch.long)
        for b, f in enumerate(fr):
            if f.nc:
                cloud[b, : f.nc] = torch.from_numpy(f.cloud[: f.nc])
            nc[b] = f.nc
        cloud, nc, pos, th, flag, hist, valid = [t.to(dev) for t in (cloud, nc, pos, th, flag, hist, valid)]
        d = (cloud[:, None, :, 0:2] - pos[:, :, None, :]).norm(dim=-1)  # [B,N,cap]
        ok = (torch.arange(cap, device=dev)[None, None, :] < nc[:, None, None]) & (d <= 16.0)
        d = d.masked_fill(~ok, 1e9)
        k = min(M_CV, cap)
        dv, ix = d.topk(k, dim=-1, largest=False)
        cv = torch.gather(cloud[:, None].expand(B, N, cap, 6), 2, ix[..., None].expand(B, N, k, 6))
        cvalid = (dv < 1e8).float()
        cv = torch.cat([cv[..., :5], cvalid[..., None]], -1)
        if k < M_CV:
            cv = F.pad(cv, (0, 0, 0, M_CV - k))
            ix = F.pad(ix, (0, M_CV - k))
        S = dict(pos=pos, theta=th, flag=flag, hist=hist, cv=cv)
        with torch.autocast("cuda", dtype=torch.float16):
            out = self.net(self.feats, S, qc=None)
        qp = out["qc_used"].float() + out["off"].float()[..., None] * out["nrm"].float()
        r = pos.norm(dim=-1)
        thr = self.tau_on + self.tau_slope * r / 50.0
        on = torch.sigmoid(out["on"].float()) >= thr
        evl = out["ev"].float()
        if self.ev_bias is not None:
            evl = evl + torch.tensor(self.ev_bias, device=dev)
        ev = evl.argmax(-1)
        cb = out["step"].argmax(-1)
        ptr = out["ptr"].float().argmax(-1)
        node_of = cv[..., 5]
        psi = out["psi"].float()
        tok = cps = None
        if getattr(self.net, "nav", False):
            tok, cps = out["h2"].float().cpu().numpy().astype(np.float16), out["cp"].float().cpu().numpy()
        qp, on, ev, cb, ptr, psi = [t.cpu().numpy() for t in (qp, on, ev, cb, ptr, psi)]
        cvn = torch.gather(cloud[..., 5][:, None].expand(B, N, cap), 2, ix[..., :M_CV] if ix.shape[-1] >= M_CV else ix).cpu().numpy() if False else None
        ix_c = ix.cpu().numpy()
        cloud_node = cloud[..., 5].cpu().numpy()
        res = [None] * len(items)
        for b, js in enumerate(by):
            for n, j in enumerate(js):
                e = int(ev[b, n])
                tgt = None
                if e == EX.MERGE:
                    pi = int(ptr[b, n])
                    if pi < M_CV and pi < ix_c.shape[-1]:
                        nd = int(cloud_node[b, ix_c[b, n, pi]])
                        own = set(fr[b].traces[items[j]["trace"]]["nodes"][-30:])
                        if nd >= 0 and nd not in own:
                            tgt = nd
                    if tgt is None:
                        e = EX.CONT
                res[j] = dict(q=qp[b, n].astype(np.float64), on=bool(on[b, n]), ev=e, psi=float(psi[b, n]), ptr=tgt, cls=int(cb[b, n]))
                if tok is not None:
                    res[j]["tok"], res[j]["cp"] = tok[b, n], cps[b, n]
        return res


def band_counts(preds, gts):
    cnt = {b: np.zeros(6) for b in BANDS}  # tp_p, n_p, tp_g, n_g, tp_g25, 0
    for g, gt in zip(preds, gts):
        P, _, _ = GE.dense_points(g, 0.5)
        G, _, _ = GE.dense_points(gt, 0.5)
        if len(G) == 0:
            continue
        rP, rG = (np.linalg.norm(P, axis=1) if len(P) else np.zeros(0)), np.linalg.norm(G, axis=1)
        dG = cKDTree(P).query(G)[0] if len(P) else np.full(len(G), 99.0)
        dP = cKDTree(G).query(P)[0] if len(P) else np.zeros(0)
        for b in BANDS:
            mg = (rG >= b[0]) & (rG < b[1])
            cnt[b][3] += mg.sum()
            cnt[b][2] += (dG[mg] <= 1.0).sum()
            cnt[b][4] += (dG[mg] <= 2.5).sum()
            if len(P):
                mp = (rP >= b[0]) & (rP < b[1])
                cnt[b][1] += mp.sum()
                cnt[b][0] += (dP[mp] <= 1.0).sum()
    out = {}
    for b in BANDS:
        c = cnt[b]
        p_, r_ = c[0] / max(c[1], 1), c[2] / max(c[3], 1)
        out[f"{b[0]}-{b[1]}"] = dict(P=float(p_), R=float(r_), F1=float(2 * p_ * r_ / max(p_ + r_, 1e-9)), R25=float(c[4] / max(c[3], 1)))
    return out


def run_eval(net, es, ks, bs=16, tau_seed=0.3, tau_on=0.5, tau_slope=0.0, wave_kw=None, verbose=True, ev_bias=None, dev="cuda", return_frames=False, gt_seeds=False, k_seed=192, nms=5, orient=False):
    net.eval()
    preds, stats, frames = [], dict(heads=0, verts=0, forks=0, merges=0, seeds=0), []
    t0 = time.time()
    for s in range(0, len(ks), bs):
        kk = ks[s : s + bs]
        x = es.batch(kk, dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            feats, seed_lg, dirh = net.encode(x)
        feats = [f.float() for f in feats]
        seeds = make_seeds(seed_lg, dirh, tau_seed, k_seed, nms)
        if gt_seeds:  # diagnostic: oracle seeds (one random point per ground-truth piece, exact travel heading) instead of the seed head
            rs = np.random.RandomState(7)
            seeds = [W.gt_seeds(es.gt[es.rows[k]], rs, 1) for k in kk]
        stats["seeds"] += sum(len(x_) for x_ in seeds)
        pol = NetPolicy(net, feats, tau_on, tau_slope, dev, ev_bias=ev_bias)
        fr = W.run(pol, len(kk), seeds, **(wave_kw or {}))
        dmaps = F.normalize(dirh.float(), dim=1).cpu().numpy() if orient else None
        for bi, f in enumerate(fr):
            if orient:
                g_, nfl = W.orient_by_field(f, dmaps[bi])
                preds.append(g_)
                stats["flipped"] = stats.get("flipped", 0) + nfl
            else:
                preds.append(f.assemble())
            stats["heads"] += len(f.heads)
            stats["verts"] += len(f.xy)
            stats["forks"] += f.n_fork
            stats["merges"] += f.n_merge
            if return_frames:
                f.cloud = None
                frames.append(f)
        if verbose:
            print(f"{s + len(kk)}/{len(ks)} frames {time.time() - t0:.0f} s", flush=True)
    return (preds, stats, frames) if return_frames else (preds, stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val_clean")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--tau_seed", type=float, default=0.3)
    ap.add_argument("--tau_on", type=float, default=0.5)
    ap.add_argument("--tau_slope", type=float, default=0.0)
    ap.add_argument("--route", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no_cov", action="store_true")
    ap.add_argument("--no_ptr", action="store_true")
    ap.add_argument("--no_range", action="store_true")
    ap.add_argument("--r_cov", type=float, default=1.2)
    ap.add_argument("--k_seed", type=int, default=192)
    ap.add_argument("--nms", type=int, default=5)
    ap.add_argument("--order", default="near", help="near: rings 12.5/25/37.5/50 m, seeds near first; far: one stage, seeds far first (E96)")
    ap.add_argument("--merge_bias", type=float, default=0.0, help="added to the merge event logit (negative: fewer merges)")
    ap.add_argument("--orient", action="store_true", help="optional orientation by the network's direction head (E99c)")
    ap.add_argument("--four", action="store_true", help="spawn four heads per seed (both flags x both headings); needs a network trained with wrong-direction states")
    a = ap.parse_args()
    dev = "cuda"
    net = PN.TraceNet(use_cov=not a.no_cov, use_ptr=not a.no_ptr, use_range=not a.no_range).to(dev)
    net.load_state_dict(torch.load(f"{OUT}/{a.name}/model.pt", map_location=dev))
    es = EvalSet(a.split)
    ks = list(range(len(es.rows)))
    if a.n:
        ks = list(np.random.RandomState(0).choice(len(ks), min(a.n, len(ks)), replace=False))
        ks.sort()
    preds, stats = run_eval(net, es, ks, k_seed=a.k_seed, nms=a.nms, tau_seed=a.tau_seed, tau_on=a.tau_on, tau_slope=a.tau_slope, wave_kw=dict(r_cov=a.r_cov, r_head=a.r_cov, four_heads=a.four, order=a.order, **(dict(rings=(70.0,)) if a.order == "far" else {})), ev_bias=[0.0, 0.0, 0.0, a.merge_bias, 0.0] if a.merge_bias else None, orient=a.orient)
    gts = [es.gt[es.rows[k]] for k in ks]
    base = [es.src["graphs"][k] for k in ks]
    res = {}
    for nm, g in (("learned", preds), ("rule_loose", base)):
        r = GE.evaluate(g, gts, with_route=a.route)
        r["bands"] = band_counts(g, gts)
        res[nm] = r
        print(nm, json.dumps({k: round(v, 4) for k, v in r.items() if k in ("P@1.0", "R@1.0", "F1@1.0", "split6_F1", "merge6_F1", "lanes_found", "false_len_frac", "dir_acc", "route_f")}), flush=True)
        print("  bands F1", {b: round(v["F1"], 3) for b, v in r["bands"].items()}, flush=True)
    print("stats", {k: v / len(ks) for k, v in stats.items()})
    d = f"{OUT}/{a.name}"
    tag = a.tag or a.split
    json.dump(dict(res=res, stats=stats, n=len(ks), args=vars(a)), open(f"{d}/eval_{tag}.json", "w"), indent=1)
    pickle.dump(dict(preds=preds, ks=ks), open(f"{d}/graphs_{tag}.pkl", "wb"))


if __name__ == "__main__":
    main()
