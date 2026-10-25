"""Loop 10: training data of the learned trace network. One item = one frame: the 39 frozen BEV channels, the (out-of-fold) heat / direction map of the dense head, the ground-truth
distance and direction fields (seed / auxiliary targets) and NS training states drawn from the ground-truth graph with the corrective expert (noisy positions / headings, ring-limited context).
Usage: imported by p10_train.py."""
import math
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402
import p10_expert as EX  # noqa: E402

HOME = os.path.expanduser("~")
D = f"{HOME}/exp_gcar_b123"
OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
M_CV = 160
NS = 32
RING = 12.5


class FrameGT:
    """per-frame ground truth in the two orientations, flat sample arrays in forward orientation"""

    def __init__(self, gt):
        self.gt = gt
        self.G = {1: EX.ExpertGraph(gt, False), -1: EX.ExpertGraph(gt, True)}
        Gf = self.G[1]
        self.off = np.zeros(Gf.n + 1, int)
        for i, p in enumerate(Gf.pts):
            self.off[i + 1] = self.off[i] + len(p)
        self.N = int(self.off[-1])
        if self.N:
            self.xy = np.concatenate([p for p in Gf.pts if len(p)])
            self.tan = np.concatenate([t for t in Gf.tan if len(t)])
            self.piece = np.concatenate([np.full(len(p), i) for i, p in enumerate(Gf.pts) if len(p)])
            self.ixf = np.concatenate([np.arange(len(p)) for p in Gf.pts if len(p)])
            self.r = np.linalg.norm(self.xy, axis=1)
            self.tree = cKDTree(self.xy)

    def em_lists(self, em_flat, flag):
        Gf = self.G[1]
        out = []
        for i, p in enumerate(Gf.pts):
            v = em_flat[self.off[i] : self.off[i + 1]]
            out.append(v if flag > 0 else v[::-1])
        return out


def cv_from_flat(F, em_flat, p, rad=16.0):
    """context point set around p: [M_CV, 6] = x, y, travel dx, dy, is_vertex, valid (1 m spacing, vertices every 2 m)"""
    cv = np.zeros((M_CV, 6), np.float32)
    ref = np.zeros((M_CV, 2), np.int32)
    d = np.linalg.norm(F.xy - p[None], axis=1)
    m = em_flat & (d <= rad) & (F.ixf % 2 == 0)
    idx = np.nonzero(m)[0]
    if len(idx) > M_CV:
        idx = idx[np.argsort(d[idx])[:M_CV]]
    n = len(idx)
    if n:
        cv[:n, 0:2] = F.xy[idx]
        cv[:n, 2:4] = F.tan[idx]
        cv[:n, 4] = (F.ixf[idx] % 4 == 0)
        cv[:n, 5] = 1.0
        ref[:n, 0] = F.piece[idx]
        ref[:n, 1] = F.ixf[idx]
    return cv, ref


def make_states(F, rng, ns=NS, neg_pts=None, p_neg=0.2, p_dup=0.12, p_drop=0.1, sig_lat=0.5, sig_ang=10.0, max_range=None, route_fn=None, p_wd=0.0):
    """ns states of one frame. Returns dict of numpy arrays (fixed shapes)."""
    S = dict(pos=np.zeros((ns, 2), np.float32), theta=np.zeros(ns, np.float32), flag=np.ones(ns, np.float32), hist=np.zeros((ns, 6, 4), np.float32), cv=np.zeros((ns, M_CV, 6), np.float32),
             qc=np.zeros((ns, 2), np.float32), k=np.zeros(ns, np.int64), c=np.zeros(ns, np.int64), res_phi=np.zeros(ns, np.float32), res_len=np.zeros(ns, np.float32), ev=np.ones(ns, np.int64),
             psi=np.zeros(ns, np.float32), off=np.zeros(ns, np.float32), ptr=np.full(ns, -1, np.int64), on=np.zeros(ns, np.float32), valid=np.zeros(ns, np.float32), geo=np.zeros(ns, np.float32), rng=np.zeros(ns, np.float32), route=np.zeros(ns, np.float32))
    if F.N == 0:
        return S
    Gf = F.G[1]
    r_all = F.r if max_range is None else F.r
    bands = [(0, 17.0), (17.0, 33.0), (33.0, 1e9)]
    bidx = [np.nonzero((r_all >= a) & (r_all < b))[0] for a, b in bands]
    bw = np.array([0.25, 0.25, 0.5]) * np.array([len(x) > 0 for x in bidx])
    bw = bw / bw.sum() if bw.sum() > 0 else bw
    for s in range(ns):
        neg = rng.rand() < p_neg
        flag = 1 if rng.rand() < 0.5 else -1
        G = F.G[flag]
        if not neg:
            b = rng.choice(3, p=bw)
            j = int(bidx[b][rng.randint(len(bidx[b]))])
            i, ixf = int(F.piece[j]), int(F.ixf[j])
            n_i = len(G.pts[i])
            ix = ixf if flag > 0 else n_i - 1 - ixf
            if n_i < 4:
                continue
            th_t = EX.ang(G.tan[i][ix])
            lat = float(np.clip(rng.randn() * sig_lat, -1.5, 1.5))
            nrm = np.array([-math.sin(th_t), math.cos(th_t)])
            p = G.pts[i][ix] + lat * nrm
            theta = th_t + float(np.clip(rng.randn() * math.radians(sig_ang), -math.radians(25), math.radians(25)))
            wd = rng.rand() < p_wd  # wrong-direction head: sits on a lane but faces against the travel direction implied by its flag -> must stop (the seeds spawn both hypotheses)
            if wd:
                theta += math.pi
        else:
            wd = False
            p = None
            if neg_pts is not None and len(neg_pts[0]) and rng.rand() < 0.7:
                q = rng.randint(len(neg_pts[0]))
                p, theta = neg_pts[0][q].copy(), float(neg_pts[1][q]) + (math.pi if flag < 0 else 0.0)
            else:
                for _ in range(10):
                    pp = (rng.rand(2) - 0.5) * 90.0
                    if F.tree.query(pp)[0] > 3.0:
                        p, theta = pp, rng.rand() * 2 * math.pi
                        break
            if p is None:
                continue
            i, ix = 0, 0
        r = float(np.linalg.norm(p))
        stage = int(min(r // RING, 5))
        l_ctx = RING * stage + rng.rand() * RING
        em_flat = F.r <= l_ctx
        if not neg:
            dup = rng.rand() < p_dup
            length = rng.uniform(3.0, 10.0) if dup else 60.0
            for pc, a, bb in G.own_future(i, ix, length):
                n = len(G.pts[pc])
                if flag > 0:
                    em_flat[F.off[pc] + a : F.off[pc] + bb] = False
                else:
                    em_flat[F.off[pc] + n - bb : F.off[pc] + n - a] = False
            for pc in range(Gf.n):
                if pc != i and len(Gf.pts[pc]) and rng.rand() < p_drop:
                    em_flat[F.off[pc] : F.off[pc + 1]] = False
        S["pos"][s], S["theta"][s], S["flag"][s], S["rng"][s] = p, theta, flag, r
        cv, ref = cv_from_flat(F, em_flat, p)
        S["cv"][s] = cv
        S["valid"][s] = 1.0
        if neg or wd:
            S["ev"][s], S["on"][s], S["geo"][s] = EX.END, 0.0, 0.0
            S["qc"][s] = p + 2.0 * EX.unit(theta)
            S["hist"][s] = 0.0
            continue
        em = F.em_lists(em_flat, flag)
        nl = float(np.clip(rng.randn() * 0.4, -1.2, 1.2))
        no = float(np.clip(rng.randn() * 0.3, -0.8, 0.8))
        t = G.targets(p, theta, em, noise_lat=nl, noise_lon=no)
        if t is None:
            S["valid"][s] = 0.0
            continue
        ev = t["ev"]
        ptr = -1
        if ev == EX.MERGE and t["ptr_ref"] is not None:
            pc, ixg = t["ptr_ref"]
            n = len(G.pts[pc])
            ixf2 = ixg if flag > 0 else n - 1 - ixg
            qpos = Gf.pts[pc][min(ixf2, len(Gf.pts[pc]) - 1)]
            vm = (cv[:, 4] > 0.5) & (cv[:, 5] > 0.5)
            if vm.any():
                dd = np.linalg.norm(cv[:, :2] - qpos[None], axis=1)
                dd[~vm] = 1e9
                jn = int(dd.argmin())
                if dd[jn] <= 3.0:
                    ptr = jn
            if ptr < 0:
                ev = EX.CONT
        S["k"][s], S["c"][s], S["res_phi"][s], S["res_len"][s] = t["k"], t["c"], t["res_phi"], t["res_len"]
        S["ev"][s], S["psi"][s], S["off"][s], S["ptr"][s] = ev, t["psi"], t["off"], ptr
        S["qc"][s] = t["qc"]
        if route_fn is not None:
            S["route"][s] = route_fn(t["q_star"])
        S["on"][s] = 1.0
        S["geo"][s] = 1.0
        # history: 6 earlier vertices behind the projection, random step classes
        cum = 0.0
        pi_, ixi = t["piece"], t["ix"]
        for h in range(6):
            cl = rng.choice(3, p=[0.3, 0.3, 0.4])
            cum += EX.STEPS[cl]
            bp = G.back_path(pi_, ixi, cum)
            if bp is None:
                break
            bp = bp + rng.randn(2) * 0.2
            rel = bp - p
            u, v = float(rel @ EX.unit(theta)), float(rel @ np.array([-math.sin(theta), math.cos(theta)]))
            S["hist"][s, h] = (u / 10.0, v / 10.0, EX.STEPS[cl] / 6.0, 1.0)
    return S


class TraceData(torch.utils.data.Dataset):
    def __init__(self, split="train", rows=None, ns=NS, train=True, seed=0, heat_run=None, p_drop=0.1, nav=False, p_wrong=0.05, p_none=0.05, p_bias=0.2, p_wd=0.0):
        self.split, self.ns, self.train, self.seed, self.p_drop, self.p_wd = split, ns, train, seed, p_drop, p_wd
        self.nav, self.p_wrong, self.p_none, self.p_bias = nav, p_wrong, p_none, p_bias
        self.gt = pickle.load(open(f"{D}/graph_gt_{split}.pkl", "rb"))
        self.rows = np.arange(len(self.gt)) if rows is None else np.asarray(rows)
        self.groups = ("dec32", "logit7")
        self.mm = None
        z = np.load(f"{OUT}/refiner_stats_dec32_logit7.npz")
        self.mean, self.std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
        self.oof = None
        self.neg = None

    def __len__(self):
        return len(self.rows)

    def _open(self):
        if self.mm is not None:
            return
        self.mm = [np.load(f"{D}/feats/{self.split}_{g}.npy", mmap_mode="r") for g in self.groups]
        self.gd = np.load(f"{HOME}/exp_crn_b123/feats/{self.split}_gt_dist_exact.npy", mmap_mode="r")
        self.gdir = np.load(f"{D}/dir_{self.split}.npy", mmap_mode="r")
        if self.split == "train":
            self.oof = {}
            self.row2 = {}
            for f in range(3):
                d = f"{OUT}/e37_fold{f}"
                ids = np.load(f"{d}/ids_oof.npy")
                self.oof[f] = (np.load(f"{d}/prob_oof.npy", mmap_mode="r"), np.load(f"{d}/dir_oof.npy", mmap_mode="r"))
                for k, r in enumerate(ids):
                    self.row2[int(r)] = (f, k)
            src = pickle.load(open(f"{D}/p8/train_oof.pkl", "rb"))
            self.neg_graphs = {int(r): g for r, g in zip(src["ids"], src["graphs"])}

    def neg_points(self, row, F):
        g = self.neg_graphs.get(int(row))
        if g is None or F.N == 0:
            return None
        P, Dd, _ = GE.dense_points(g, 1.0)
        if len(P) == 0:
            return None
        dist = F.tree.query(P)[0]
        m = dist > 2.0
        if not m.any():
            return None
        return P[m], np.arctan2(Dd[m, 1], Dd[m, 0])

    def __getitem__(self, n):
        self._open()
        row = int(self.rows[n])
        rng = np.random.RandomState((self.seed * 1000003 + n * 7919 + (np.random.randint(1 << 30) if self.train else 0)) % (2 ** 31))
        x = np.concatenate([np.asarray(m[row], np.float32) for m in self.mm], 0)
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        f, k = self.row2[row]
        heat = np.asarray(self.oof[f][0][k], np.float32)
        dirf = np.asarray(self.oof[f][1][k], np.float32)
        x = np.concatenate([x, heat[None], dirf], 0).astype(np.float16)
        F = FrameGT(self.gt[row])
        route_fn, raster = None, None
        if self.nav:
            import p8_lib as L
            import sd_lib as SL

            if not hasattr(self, "sdl"):
                self.sdl = L.cache("sd", self.split)
                self.rh = np.load(f"{D}/route_heat_{self.split}.npy", mmap_mode="r")
            u = rng.rand()
            kind = "w" if u < self.p_wrong else ("0" if u < self.p_wrong + self.p_none else "n")
            sd0 = self.sdl[row] if kind != "w" else self.sdl[(row + 397) % len(self.sdl)]
            if sd0 is None:
                kind = "0"
            if kind == "0":
                raster = np.zeros((5, 200, 200), np.float32)
            else:
                sigma = float(rng.choice([0.0, 1.0, 2.0, 3.0]))
                drop = 0.3 if rng.rand() < 0.3 else 0.0
                kw = dict(bias=float(rng.choice([1.0, 2.0]))) if (kind == "n" and rng.rand() < self.p_bias) else {}
                sdm, _, _ = L.draw_sd(sd0, row, sigma, drop, int(rng.randint(1, 1 << 20)), **kw)
                raster = SL.raster_sd(sdm)
            heat = np.asarray(self.rh[row])

            def route_fn(q, kind=kind, heat=heat):
                if kind != "n":
                    return 0.0
                r_, c_ = int(np.clip(round((q[0] + 50.0) / 0.5 - 0.5), 0, 199)), int(np.clip(round((q[1] + 50.0) / 0.5 - 0.5), 0, 199))
                return float(heat[r_, c_] >= 128)

        S = make_states(F, rng, self.ns, neg_pts=self.neg_points(row, F), p_drop=self.p_drop, route_fn=route_fn, p_wd=self.p_wd)
        o = dict(x=torch.from_numpy(x), gd=torch.from_numpy(np.asarray(self.gd[row], np.float32) * 0.5), gdir=torch.from_numpy(np.asarray(self.gdir[row], np.float32)), row=row)
        o.update({k_: torch.from_numpy(v) for k_, v in S.items()})
        if raster is not None:
            o["sd"] = torch.from_numpy(raster.astype(np.float16))
        return o
