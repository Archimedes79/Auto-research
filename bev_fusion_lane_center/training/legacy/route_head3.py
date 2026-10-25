"""Loop 7 / E78: route_head2.py plus the SD-map conditioning (--maps sd | sd_nocnt: ADASIS-like road map with sparse shape points, lane count, pose error; sd_lib.py, sd_precompute.py).
Loop 6 / E73: systematic fine-tuning of the end-to-end lane + route head (E60 -> E65d) with a clean validation protocol.

What is new compared with route_head.py (E60 / E65d):
  * fine-tuning factors that can be varied one at a time: perturbation set (--pert_train), perturbation mass, learning rate, steps, trainable layer groups (--train),
    EMA of the weights (--ema), L2-SP penalty towards the initial weights (--l2sp), loss weights, map-noise schedule (--p_exact/--sig_lo/--sig_hi/--p_wrong/--p_none);
  * a map-trust output (--trust 1): a scalar head on the U-Net bottleneck that predicts whether the given map is usable (pose error <= 1 m at 20 m lever arm, not a map of another place);
  * a DEV protocol that never touches the TEST scenes: VAL scenes with clean features (every 2nd row) and the seven synthetic perturbations
    (features of every 3rd VAL row, pert/<name>/val_*.npy from gcar_pert.py --part val), several map conditions evaluated in ONE pass over the features;
  * a TEST protocol (--test 1) with the same cells as the E65d evaluation (clean features 7 map conditions, 7 perturbations x 4 map conditions);
  * weight interpolation / soup of saved models (--soup name:weight,name:weight) as an evaluation-only mode (WiSE-FT, model soups).
Call (environment gcar):  run_gcar.sh route_head2.py --name e73_x --init_from e60_main --steps 1500 --lr 2e-4 --pert_train dark,cam_front_off,fog --dev 1
Output: out/loop2/<name>/{model.pt, [model_ema.pt], dev.json, [test.json], log.txt}
"""
import argparse
import copy
import json
import os
import pickle
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc_lib as AL  # noqa: E402
import dense_head as DH  # noqa: E402
import route_head as RH  # noqa: E402
import sd_lib as SL  # noqa: E402

HOME, OUT = DH.HOME, DH.OUT
D = f"{HOME}/exp_gcar_b123"
N_MAP = 4
SD_CFG = dict(eps_lo=0.5, eps_hi=2.0, p_cnt=0.1)  # ADASIS-like quality of the simulated SD map (sparsification tolerance m, lane-count error probability)


def n_map(mode):
    """number of map channels: HD map raster 4 (route, lanes, dx, dy); SD map raster 5 (route, roads, dx, dy, lane count)"""
    return 5 if str(mode).startswith("sd") else 4


_SDM = {}


def sd_maps(part):
    if part not in _SDM:
        _SDM[part] = pickle.load(open(f"{D}/sd_{part}.pkl", "rb"))
    return _SDM[part]
PERTS = ["dark", "cam_front_off", "fog", "cam_front3_off", "blur", "extr3", "radar_off"]
SP = json.load(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/data/splits.json"))
LEVER_M = 20.0  # lever arm of a yaw error for the map-trust label
TRUST_ERR_M = 1.0  # a map is 'usable' when the displacement of its points at 20 m ahead is <= 1 m


# ------------------------------------------------------------------ model


class TrustUNet(DH.UNet):
    """DH.UNet (same parameter names, so E60 / E65d checkpoints load) plus an optional scalar map-trust head on the bottleneck (global mean + max pooling)."""

    def __init__(self, cin, base=48, out=4, trust=False):
        super().__init__(cin, base, out, 0.0)
        self.trust_head = nn.Sequential(nn.Linear(8 * base, 64), nn.ReLU(inplace=True), nn.Linear(64, 1)) if trust else None

    def forward(self, x, with_trust=False):
        x0 = self.stem(self.drop(x))
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        up = lambda a, b: torch.cat([F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1)
        y = self.u2(up(x3, x2))
        y = self.u1(up(y, x1))
        y = self.u0(up(y, x0))
        y = self.head(y)
        if with_trust:
            if self.trust_head is None:
                return y, None
            return y, self.trust_head(torch.cat([x3.mean((2, 3)), x3.amax((2, 3))], 1))[:, 0]
        return y


def make_net(init_from, trust=False, base=48, nmap=4):
    cin = 39 + nmap
    net = TrustUNet(cin, base, 4, trust).cuda()
    mean, std = torch.zeros(cin), torch.ones(cin)
    if init_from:
        ck = torch.load(f"{OUT}/{init_from}/model.pt", map_location="cuda", weights_only=False)
        own = net.state_dict()
        for k, v in ck["state"].items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
            elif k in own and v.ndim == 4 and k.startswith("stem") and own[k].shape[1] > v.shape[1] and own[k].shape[0] == v.shape[0]:  # extra map channels: new input weights start at zero
                w = torch.zeros_like(own[k])
                w[:, : v.shape[1]] = v
                own[k] = w
        net.load_state_dict(own)
        mean, std = ck["mean"].cpu().clone(), ck["std"].cpu().clone()
        if len(mean) < cin:
            mean = torch.cat([mean, torch.zeros(cin - len(mean))])
            std = torch.cat([std, torch.ones(cin - len(std))])
    return net, mean.cuda(), std.cuda()


GROUPS = ["stem", "d1", "d2", "d3", "u2", "u1", "u0", "head", "trust_head"]


def set_trainable(net, spec):
    """spec: 'all' | comma list of block names (stem,d1,d2,d3,u2,u1,u0; the 1x1 head and the trust head are always trained) | 'bn' (BatchNorm affine parameters + heads)
    | 'trust_only' (only the trust head; everything else frozen incl. BatchNorm statistics, i.e. the lane / route outputs stay identical).
    Returns the list of frozen modules (kept in eval mode so that their BatchNorm statistics do not move)."""
    frozen = []
    if spec == "all":
        return frozen
    for p in net.parameters():
        p.requires_grad = False
    train_groups = {"head", "trust_head"}
    if spec == "bn":
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = True
    elif spec == "trust_only":
        train_groups = {"trust_head"}
    else:
        train_groups |= set(spec.split(","))
    for g in GROUPS:
        mod = getattr(net, g, None)
        if mod is None:
            continue
        if g in train_groups:
            for p in mod.parameters():
                p.requires_grad = True
        elif spec != "bn":
            frozen.append(mod)
    return frozen


class EMA:
    def __init__(self, net, decay):
        self.decay = decay
        self.sd = {k: v.detach().clone().float() if v.dtype.is_floating_point else v.detach().clone() for k, v in net.state_dict().items()}

    @torch.no_grad()
    def update(self, net):
        for k, v in net.state_dict().items():
            if v.dtype.is_floating_point:
                self.sd[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.sd[k].copy_(v)


# ------------------------------------------------------------------ data


def pose_err(R, t, lever=LEVER_M):
    """displacement (m) of a point `lever` m ahead of the vehicle under the map transform x -> R x + t (the effective pose error of the map)."""
    x0 = np.array([0.0, lever])
    return float(np.linalg.norm(R @ x0 + t - x0))


def build_maps_sd(SDm, i, n_all, rng, sxy, drop, wrong, none_, maps_mode, reg=None):
    """SD map raster [5,200,200] (route, roads, dx, dy, lane count/4) of sample i: ADASIS-like degradation (sparse shape points, lane-count errors, missing side roads `drop`), then the pose error of the map
    (translation sigma sxy per axis, yaw 2 x sxy degrees), a wrong map (another sample) or none. reg as in build_maps."""
    maps = np.zeros((5, 200, 200), np.float32)
    src = i if not wrong else (i + n_all // 2) % n_all
    sd = SDm[src]
    if none_ or sd is None or maps_mode == "none":
        return maps, 0.0, False
    d = rng.randn(2) * sxy
    yaw = rng.randn() * 2.0 * sxy
    R = AL.rot2(yaw)
    dg = SL.degrade_sd(sd, rng, SD_CFG["eps_lo"], SD_CFG["eps_hi"], SD_CFG["p_cnt"], drop)
    if reg is None:
        Rt, dt, Rres, tres = R, d, R, d
    else:
        Rr, tr = reg
        Rt, dt = Rr.T @ R, (d - tr) @ Rr
        Rres, tres = Rt, Rr.T @ (d - tr)
    maps = SL.raster_sd(dg, Rt, dt)
    if maps_mode == "sd_nocnt":
        maps[4] = 0.0
    return maps, pose_err(Rres, tres), True


def build_maps(G, RT, i, n_all, rng, sxy, drop, wrong, none_, maps_mode="route+lanes", reg=None, SDm=None):
    """Map raster [4,200,200] (route, lanes, direction dx, dy) of sample i with pose error (translation sigma sxy m per axis, yaw 2 x sxy degrees), dropped lanes, wrong map (another sample) or none.
    reg = (R, t): optional registration x_map = R x_ex + t found by graph-ICP against the extracted lane graph; the map is then brought back into the vehicle frame (E74 registration front end).
    Same sequence of random draws as RouteData.__getitem__ (so that the E65d evaluation is reproduced). Returns maps, effective pose error at 20 m ahead (m), map-present flag."""
    if str(maps_mode).startswith("sd"):
        return build_maps_sd(SDm, i, n_all, rng, sxy, drop, wrong, none_, maps_mode, reg)
    maps = np.zeros((N_MAP, 200, 200), np.float32)
    src = i if not wrong else (i + n_all // 2) % n_all
    g, rt = G[src], RT[src]
    err, ok = 0.0, False
    if not none_ and rt is not None and maps_mode != "none":
        d = rng.randn(2) * sxy
        yaw = rng.randn() * 2.0 * sxy
        R = AL.rot2(yaw)
        if reg is None:
            tf = lambda p: np.asarray(p, np.float64) @ R.T + d
            Rres, tres = R, d
        else:
            Rr, tr = reg
            tf = lambda p: (np.asarray(p, np.float64) @ R.T + d - tr) @ Rr
            Rres, tres = Rr.T @ R, Rr.T @ (d - tr)
        rpieces = [tf(g["pts"][p]) for p in rt["route"] if len(g["pts"][p]) >= 2]
        m_r, _ = RH.raster(rpieces)
        maps[0] = m_r
        if maps_mode == "route+lanes":
            lp = [tf(p) for p in g["pts"] if len(p) >= 2 and (rng.rand() >= drop)]
            m_l, dr_l = RH.raster(lp)
            maps[1], maps[2:4] = m_l, dr_l
        err = pose_err(Rres, tres)
        ok = True
    return maps, err, ok


class TrainData(RH.RouteData):
    """RouteData for training with a configurable map-noise schedule and the map-trust label."""

    def __getitem__(self, k):
        if self.rh is None:
            self.rh = np.load(f"{D}/route_heat_{self.part}.npy", mmap_mode="r")
        it = self.base[k]
        i = int(self.base.idx[k])
        c = self.cfg
        rng = self._rng(i)
        if self.pt_names:
            if self.pt is None:
                self._open_pert()
            cands = [p for p in self.pt_names if i in self.pt[p][2]]
            if cands and rng.rand() < c.get("p_use_pert", 0.85):
                p = cands[int(rng.randint(len(cands)))]
                j = self.pt[p][2][i]
                xp = torch.cat([torch.from_numpy(np.asarray(self.pt[p][0][j], np.float32)), torch.from_numpy(np.asarray(self.pt[p][1][j], np.float32))], 0)
                it = (xp,) + tuple(it[1:])
        u = rng.rand()
        sxy = 0.0 if u < c["p_exact"] else rng.uniform(c["sig_lo"], c["sig_hi"]) * (1.0 if c["noise_train"] else 0.0)
        drop = rng.uniform(0.0, 0.4) if rng.rand() < 0.5 else 0.0
        wrong = rng.rand() < c["p_wrong"]
        none_ = rng.rand() < c["p_none"]
        if str(c["maps"]).startswith("sd") and getattr(self, "SDm", None) is None:
            self.SDm = sd_maps(self.part)
        maps, err, ok = build_maps(self.G, self.RT, i, self.n_all, rng, sxy, drop, wrong, none_, c["maps"], SDm=getattr(self, "SDm", None))
        tl = 1.0 if (ok and not wrong and err <= TRUST_ERR_M) else 0.0
        return dict(x=it[0], maps=torch.from_numpy(maps), idx=i, y=it[1], dirf=it[2], yr=torch.from_numpy(np.asarray(self.rh[i], np.float32) / 255.0), tlabel=torch.tensor(tl), tvalid=torch.tensor(1.0 if ok else 0.0))


class EvalSet(torch.utils.data.Dataset):
    """Features (clean or perturbed) of a list of rows plus the map rasters of several conditions at once (cond = (name, sxy, drop, wrong, none))."""

    def __init__(self, part, idx, feat, conds, maps_mode="route+lanes", pert_prefix="val"):
        self.part, self.idx, self.feat, self.conds, self.maps_mode, self.pert_prefix = part, np.asarray(idx), feat, conds, maps_mode, pert_prefix
        self.G = pickle.load(open(f"{D}/graph_gt_{part}.pkl", "rb"))
        self.RT = pickle.load(open(f"{D}/route_gt_{part}.pkl", "rb"))
        self.n_all = len(self.G)
        self.SDm = sd_maps(part) if str(maps_mode).startswith("sd") else None
        self.base = DH.Data("gcar", ["dec32", "logit7"], part, self.idx, "b123", target=False) if feat == "clean" else None
        self.pf, self.row_of = None, None

    def _open_pert(self):
        d_ = f"{D}/pert/{self.feat}"
        rows = np.load(f"{d_}/{self.pert_prefix}_rows.npy")
        self.pf = (np.load(f"{d_}/{self.pert_prefix}_dec32.npy", mmap_mode="r"), np.load(f"{d_}/{self.pert_prefix}_logit7.npy", mmap_mode="r"))
        self.row_of = {int(r): j for j, r in enumerate(rows)}

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        if self.feat == "clean":
            x = self.base[k][0]
        else:
            if self.pf is None:
                self._open_pert()
            j = self.row_of[i]
            x = torch.cat([torch.from_numpy(np.asarray(self.pf[0][j], np.float32)), torch.from_numpy(np.asarray(self.pf[1][j], np.float32))], 0)
        maps, tl, tv = [], [], []
        for (_, sxy, drop, wrong, none_, use_reg) in self.conds:
            rng = np.random.RandomState((zlib.crc32(f"{self.part}{i}".encode()) + 17) % (2 ** 32))
            reg = None
            if use_reg:  # E74: registration (R, t) of the disturbed map to the extracted graph, precomputed by reg_precompute.py
                rg = registrations(self.part, str(self.maps_mode).startswith("sd"))[sxy].get(i)
                reg = (rg[0], rg[1]) if rg is not None else (np.eye(2), np.zeros(2))
            m, err, ok = build_maps(self.G, self.RT, i, self.n_all, rng, sxy, drop, wrong, none_, self.maps_mode, reg, SDm=self.SDm)
            maps.append(m)
            tl.append(1.0 if (ok and not wrong and err <= TRUST_ERR_M) else 0.0)
            tv.append(1.0 if ok else 0.0)
        return dict(x=x, maps=torch.from_numpy(np.stack(maps)), idx=i, tlabel=torch.tensor(tl), tvalid=torch.tensor(tv))


def parse_conds(s):
    """'name:sigma_m:drop[:flag]' with flag w = wrong map, n = no map, r = disturbed map registered to the extracted graph (E74)."""
    out = []
    for cond in s.split(","):
        f = cond.split(":")
        flag = f[3] if len(f) > 3 else ""
        out.append((f[0], float(f[1]), float(f[2]), flag == "w", flag == "n", flag == "r"))
    return out


_REG = {}


def registrations(part, sd=False):
    """{sigma: {row: (R, t, score)}} from reg_precompute.py (HD map) or sd_reg_precompute.py (SD map); rows of the part 'val' = VAL + TEST rows."""
    key = (part, sd)
    if key not in _REG:
        _REG[key] = pickle.load(open(f"{D}/{'reg_sd' if sd else 'reg'}_{part}.pkl", "rb"))
    return _REG[key]


# ------------------------------------------------------------------ evaluation

_GT = {}
_RP = {}
TRUST_THR = 0.5  # gate threshold of the learned trust score (set from --trust_thr; chosen on DEV)
AGREE_THR = 0.4  # gate threshold of the label-free map-agreement score (set from --agree_thr; chosen on DEV)
KEEP_ARRAYS = False  # store per-sample arrays (agreement, trust, route success, label) in the result (final models only)


def gt_points(part):
    if part not in _GT:
        _GT[part] = np.load(f"{HOME}/exp_crn_b123/feats/{part}_gt_pts_exact.npy", allow_pickle=True)
    return _GT[part]


def route_pts(part):
    if part not in _RP:
        _RP[part] = pickle.load(open(f"{D}/route_pts_{part}.pkl", "rb"))
    return _RP[part]


def lane_prf(prob, gts, tau=0.5, dev="cuda"):
    """pooled point precision / recall / F1 at 1 m (same counting as DH.eval_probs 'F1@1.0')."""
    tpp = npp = tpg = ng = 0
    for s in range(0, len(prob), 32):
        sk = DH.skeleton_points(prob[s : s + 32].float().to(dev), tau)
        for kk, P in enumerate(sk):
            G = gts[s + kk]
            npp += len(P)
            ng += len(G)
            if len(P) and len(G):
                tpp += int((cKDTree(G).query(P)[0] <= 1.0).sum())
                tpg += int((cKDTree(P).query(G)[0] <= 1.0).sum())
    pr, rc = tpp / max(npp, 1), tpg / max(ng, 1)
    return dict(lane_P=pr, lane_R=rc, lane_F1=2 * pr * rc / max(pr + rc, 1e-9))


def route_eval(rprob, ids, part, tau=0.5, dev="cuda", pts=None):
    """route heat map -> points; coverage / precision at 1.5 m, success (both >= 0.8, like assoc_eval), point F1 at 1 m (same as RH.route_metrics for one tau)."""
    pts = route_pts(part) if pts is None else pts
    cov = n_gt = prec = n_pr = 0
    succ = []
    tp = fp = fn = 0
    for s in range(0, len(ids), 32):
        sk = DH.skeleton_points(rprob[s : s + 32].float().to(dev), tau)
        for kk, P in enumerate(sk):
            g = pts[int(ids[s + kk])]
            if g is None:
                continue
            P = np.asarray(P, np.float64)
            if len(P) == 0:
                n_gt += len(g)
                succ.append(0)
                fn += len(g)
                continue
            d1 = cKDTree(P).query(g)[0]
            d2 = cKDTree(g).query(P)[0]
            c1, c2 = int((d1 <= 1.5).sum()), int((d2 <= 1.5).sum())
            cov, n_gt, prec, n_pr = cov + c1, n_gt + len(g), prec + c2, n_pr + len(P)
            succ.append(int(c1 / len(g) >= 0.8 and c2 / len(P) >= 0.8))
            tp1, fp1, fn1 = int((d2 <= 1.0).sum()), int((d2 > 1.0).sum()), int((d1 > 1.0).sum())
            tp, fp, fn = tp + tp1, fp + fp1, fn + fn1
    return dict(route_success=float(np.mean(succ)) if succ else 0.0, route_cov=cov / max(n_gt, 1), route_prec=prec / max(n_pr, 1), route_F1=2 * tp / max(2 * tp + fp + fn, 1), succ=succ)


def auroc(score, label):
    """Mann-Whitney AUROC; label 1 = positive."""
    score, label = np.asarray(score, np.float64), np.asarray(label)
    n1, n0 = int((label == 1).sum()), int((label == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def eval_cells(net, mean, std, ds, part, bs=16, workers=3, tau=0.5, extra=None):
    """One pass over the features of ds; every map condition of ds.conds is run through the network. Returns {cond: metrics} plus the pooled trust AUROC."""
    net.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers)
    nC = len(ds.conds)
    L = [[] for _ in range(nC)]
    R = [[] for _ in range(nC)]
    T = [[] for _ in range(nC)]
    A = [[] for _ in range(nC)]
    ids, TL, TV = [], [], []
    for b in dl:
        x = b["x"].cuda(non_blocking=True)
        mp = b["maps"].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):  # map-free lane prediction of the same head (zero map input)
            o0 = net((torch.cat([x, torch.zeros_like(mp[:, 0])], 1) - mean[None, :, None, None]) / std[None, :, None, None]).float()
        pn = torch.sigmoid(o0[:, 0])
        for c in range(nC):
            dil = F.max_pool2d((mp[:, c, 1:2] > 0).float(), 5, 1, 2)[:, 0]  # map lanes dilated by 1 m
            A[c].append(((pn * dil).sum((1, 2)) / dil.sum((1, 2)).clamp_min(1.0)).cpu())
            xin = (torch.cat([x, mp[:, c]], 1) - mean[None, :, None, None]) / std[None, :, None, None]
            with torch.autocast("cuda", dtype=torch.float16):
                o, t = net(xin, with_trust=True)
            o = o.float()
            L[c].append(torch.sigmoid(o[:, 0]).half().cpu())
            R[c].append(torch.sigmoid(o[:, 3]).half().cpu())
            T[c].append(torch.sigmoid(t.float()).cpu() if t is not None else torch.zeros(len(x)))
        ids.append(b["idx"])
        TL.append(b["tlabel"])
        TV.append(b["tvalid"])
    ids = torch.cat(ids).numpy()
    TL, TV = torch.cat(TL).numpy(), torch.cat(TV).numpy()
    gt = gt_points(part)
    gts = [gt[int(i)].astype(np.float32) for i in ids]
    res, tr_s, tr_l = {}, [], []
    for c, cond in enumerate(ds.conds):
        prob, rprob = torch.cat(L[c]), torch.cat(R[c])
        re_ = route_eval(rprob, ids, part, tau)
        succ = np.asarray(re_.pop('succ'), float)
        m = dict(n=len(ids), **lane_prf(prob, gts, tau), **re_)
        for nm_, pts_ in (extra or {}).items():  # E75: success against another route (e.g. the driven one while the alternative was given)
            m['route_success@' + nm_] = route_eval(rprob, ids, part, tau, pts=pts_)['route_success']
        t = torch.cat(T[c]).numpy()
        v = TV[:, c] > 0
        m["trust_mean"] = float(t[v].mean()) if v.any() else None
        ag = torch.cat(A[c]).numpy()
        m["agree_mean"] = float(ag[v].mean()) if v.any() else None
        acc_a = (ag >= AGREE_THR) & v  # gate of the agreement score
        m["agr_accept"] = float(acc_a.sum() / max(v.sum(), 1))
        if len(succ) == len(ag):
            m["agr_succ_accepted"] = float(succ[acc_a].mean()) if acc_a.any() else None
            m["agr_succ_rejected"] = float(succ[(~acc_a) & v].mean()) if ((~acc_a) & v).any() else None
        if KEEP_ARRAYS:
            m["_arr"] = dict(agree=[round(float(z), 4) for z in ag[v]], trust=[round(float(z), 4) for z in t[v]], succ=[int(z) for z in succ[v]] if len(succ) == len(ag) else None, tl=[int(z) for z in TL[:, c][v]])
        if net.trust_head is not None and len(succ) == len(t):  # gate at trust 0.5: share of accepted samples and route success among accepted / rejected samples
            acc = (t >= TRUST_THR) & v
            m["gate_accept"] = float(acc.sum() / max(v.sum(), 1))
            m["gate_succ_accepted"] = float(succ[acc].mean()) if acc.any() else None
            m["gate_succ_rejected"] = float(succ[(~acc) & v].mean()) if ((~acc) & v).any() else None
        res[cond[0]] = m
        tr_s.append(t[v])
        tr_l.append(TL[:, c][v])
    s_all, l_all = np.concatenate(tr_s), np.concatenate(tr_l)
    ag_all = np.concatenate([torch.cat(A[c]).numpy()[TV[:, c] > 0] for c in range(nC)])
    res["_agree_auc"] = auroc(ag_all, l_all)
    res["_trust_auc"] = auroc(s_all, l_all) if net.trust_head is not None else None
    # exact vs wrong map only
    names = [c[0] for c in ds.conds]
    if "exact" in names and "wrong" in names:
        ie, iw = names.index("exact"), names.index("wrong")
        ae, aw = torch.cat(A[ie]).numpy(), torch.cat(A[iw]).numpy()
        ve, vw = TV[:, ie] > 0, TV[:, iw] > 0
        res["_agree_auc_exact_vs_wrong"] = auroc(np.concatenate([ae[ve], aw[vw]]), np.concatenate([np.ones(int(ve.sum())), np.zeros(int(vw.sum()))]))
    if net.trust_head is not None and "exact" in names and "wrong" in names:
        ie, iw = names.index("exact"), names.index("wrong")
        se, sw = torch.cat(T[ie]).numpy(), torch.cat(T[iw]).numpy()
        ve, vw = TV[:, ie] > 0, TV[:, iw] > 0
        res["_trust_auc_exact_vs_wrong"] = auroc(np.concatenate([se[ve], sw[vw]]), np.concatenate([np.ones(int(ve.sum())), np.zeros(int(vw.sum()))]))
    return res


DEV_CLEAN = "exact:0:0,s1:1:0,wrong:0:0:w,none:0:0:n"
DEV_PERT = "exact:0:0,wrong:0:0:w"


def dev_eval(net, mean, std, a, P):
    """DEV protocol: VAL scenes only. Clean features (every 2nd labelled row) x 4 map conditions; the 7 perturbations (every 3rd VAL row) x conditions of --dev_pert_conds."""
    perts = [p for p in a.dev_perts.split(",") if p]
    while not all(os.path.exists(f"{D}/pert/{p}/DONE_val") for p in perts):  # the VAL features of a perturbation are only valid after gcar_pert.py --part val has finished
        P("waiting for the VAL features of", [p for p in perts if not os.path.exists(f"{D}/pert/{p}/DONE_val")])
        time.sleep(30)
    t0 = time.time()
    RTv = pickle.load(open(f"{D}/route_gt_val.pkl", "rb"))
    idx_all = [i for i in SP["val"] if RTv[i] is not None]
    res = {}
    cc = parse_conds(DEV_CLEAN)
    res["clean"] = eval_cells(net, mean, std, EvalSet("val", idx_all[:: a.dev_stride], "clean", cc, maps_mode=a.maps), "val", workers=a.workers)
    for p in perts:
        have = set(np.load(f"{D}/pert/{p}/val_rows.npy").tolist())
        idx = [i for i in idx_all if i in have]
        res[p] = eval_cells(net, mean, std, EvalSet("val", idx, p, parse_conds(a.dev_pert_conds), maps_mode=a.maps, pert_prefix="val"), "val", workers=a.workers)
    res["score"] = dev_score(res, perts)
    P(f"DEV {time.time() - t0:.0f} s | score {res['score']['score']:.4f} | clean exact {res['clean']['exact']['lane_F1']:.3f}/{res['clean']['exact']['route_success']:.3f} s1 route {res['clean']['s1']['route_success']:.3f} "
      f"wrong lane {res['clean']['wrong']['lane_F1']:.3f} none lane {res['clean']['none']['lane_F1']:.3f} | pert exact lane {res['score']['pert_exact_lane']:.3f} route {res['score']['pert_exact_route']:.3f} wrong lane {res['score']['pert_wrong_lane']:.3f}"
      + (f" | trust AUC clean {res['clean']['_trust_auc']:.3f}" if res["clean"]["_trust_auc"] is not None else ""))
    return res


def dev_score(res, perts=PERTS):
    pe_l = float(np.mean([res[p]["exact"]["lane_F1"] for p in perts]))
    pe_r = float(np.mean([res[p]["exact"]["route_success"] for p in perts]))
    pw_l = float(np.mean([res[p]["wrong"]["lane_F1"] for p in perts]))
    c = res["clean"]
    comps = [c["exact"]["lane_F1"], c["exact"]["route_success"], c["s1"]["route_success"], c["wrong"]["lane_F1"], c["none"]["lane_F1"], pe_l, pe_r, pw_l]
    return dict(score=float(np.mean(comps)), clean_exact_lane=c["exact"]["lane_F1"], clean_exact_route=c["exact"]["route_success"], clean_s1_route=c["s1"]["route_success"], clean_wrong_lane=c["wrong"]["lane_F1"],
                clean_none_lane=c["none"]["lane_F1"], pert_exact_lane=pe_l, pert_exact_route=pe_r, pert_wrong_lane=pw_l)


TEST_CLEAN = "clean:0:0,s1:1:0,s2:2:0,d30:0:0.3,s1d30:1:0.3,wrong:0:0:w,none:0:0:n"
TEST_PERT = "clean:0:0,s1:1:0,wrong:0:0:w,none:0:0:n"


def test_eval(net, mean, std, a, P):
    """TEST protocol (run once for the finalists): all labelled TEST rows with clean features x 7 map conditions, the 7 perturbations x 4 map conditions (as in the E65d evaluation)."""
    t0 = time.time()
    RTv = pickle.load(open(f"{D}/route_gt_val.pkl", "rb"))
    idx_all = [i for i in SP["test"] if RTv[i] is not None]
    res = {}
    res["clean"] = eval_cells(net, mean, std, EvalSet("val", idx_all, "clean", parse_conds(a.test_clean_conds), maps_mode=a.maps), "val", workers=a.workers)
    for p in [q for q in a.test_perts.split(",") if q]:
        have = set(np.load(f"{D}/pert/{p}/test_rows.npy").tolist())
        idx = [i for i in idx_all if i in have][:: a.test_pert_stride]
        res[p] = eval_cells(net, mean, std, EvalSet("val", idx, p, parse_conds(a.test_pert_conds), maps_mode=a.maps, pert_prefix="test"), "val", workers=a.workers)
    c = res["clean"]
    P(f"TEST {time.time() - t0:.0f} s | clean features (lane F1 / route success): " + " ".join(f"{k} {v['lane_F1']:.3f}/{v['route_success']:.3f}" for k, v in c.items() if not k.startswith("_")))
    return res


def route_points_of(G, entry):
    return np.concatenate([AL.resample(G["pts"][p], 1.0) for p in entry["route"] if len(G["pts"][p]) >= 2]).astype(np.float32)


def alt_eval(net, mean, std, a, P):
    """E75: does the head follow a CHANGED navigation goal? TEST rows whose driven route crosses a split: route_alt_val.pkl = the route rebuilt to follow the OTHER branch at the first split, route_drv_sub_val.pkl = the driven route of the same rows.
    Input: the driven route or the alternative route (exact map pose); success of the route output is measured against the route that was GIVEN (compliance) and against the other one (ignoring the input)."""
    t0 = time.time()
    G = pickle.load(open(f"{D}/graph_gt_val.pkl", "rb"))
    ALT = pickle.load(open(f"{D}/route_alt_val.pkl", "rb"))
    DRV = pickle.load(open(f"{D}/route_drv_sub_val.pkl", "rb"))
    rows = [i for i in SP["test"] if ALT.get(i) is not None and DRV.get(i) is not None and len(ALT[i]["route"]) > 0 and ALT[i]["route"] != DRV[i]["route"]]
    pa = {i: route_points_of(G[i], ALT[i]) for i in rows}
    pd_ = {i: route_points_of(G[i], DRV[i]) for i in rows}
    P(f"alt rows {len(rows)}")
    res = dict(n_rows=len(rows))
    feats = ["clean"] + [q for q in a.test_perts.split(",") if q]
    for feat in feats:
        idx = rows
        if feat != "clean":
            have = set(np.load(f"{D}/pert/{feat}/test_rows.npy").tolist())
            idx = [i for i in rows if i in have]
        out = {}
        for nm, RTx in (("given_drv", DRV), ("given_alt", ALT)):
            ds = EvalSet("val", idx, feat, parse_conds("exact:0:0"), maps_mode=a.maps, pert_prefix="test")
            ds.RT = RTx
            if str(a.maps).startswith("sd"):  # SD map rebuilt from the given route (driven or alternative branch)
                sdl = [None] * len(G)
                for i_ in rows:
                    sdl[i_] = SL.build_sd(G[i_], RTx[i_]["route"])
                ds.SDm = sdl
            out[nm] = eval_cells(net, mean, std, ds, "val", workers=a.workers, extra=dict(drv=pd_, alt=pa))["exact"]
        res[feat] = out
        P(f"  {feat:15s} n {out['given_drv']['n']} | driven given: success vs driven {out['given_drv']['route_success@drv']:.3f}, vs alt {out['given_drv']['route_success@alt']:.3f} | "
          f"alternative given: vs alt {out['given_alt']['route_success@alt']:.3f}, vs driven {out['given_alt']['route_success@drv']:.3f}")
    P(f"ALT {time.time() - t0:.0f} s")
    return res


# ------------------------------------------------------------------ main


def prep(b, mean, std):
    x = torch.cat([b["x"].cuda(non_blocking=True), b["maps"].cuda(non_blocking=True)], 1)
    return (x - mean[None, :, None, None]) / std[None, :, None, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--init_from", default="e60_main")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the false-alarm side of the focal BCE")
    ap.add_argument("--wpos", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--dir", type=float, default=0.5)
    ap.add_argument("--wroute", type=float, default=1.0)
    ap.add_argument("--maps", default="route+lanes", choices=["route+lanes", "route", "none", "sd", "sd_nocnt"])
    ap.add_argument("--noise_train", type=int, default=1)
    ap.add_argument("--p_exact", type=float, default=0.3, help="fraction of training samples with an exact map")
    ap.add_argument("--sig_lo", type=float, default=0.3)
    ap.add_argument("--sig_hi", type=float, default=2.0)
    ap.add_argument("--p_wrong", type=float, default=0.1)
    ap.add_argument("--p_none", type=float, default=0.15)
    ap.add_argument("--pert_train", default="", help="comma list of perturbation types whose TRAIN features are mixed in")
    ap.add_argument("--p_pert_mass", type=float, default=0.55)
    ap.add_argument("--p_use_pert", type=float, default=0.85)
    ap.add_argument("--train", default="all", help="all | block names (stem,d1,d2,d3,u2,u1,u0) | bn | trust_only")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights (0 = off); the EMA model is saved and evaluated as <name>__ema")
    ap.add_argument("--l2sp", type=float, default=0.0, help="L2-SP penalty weight towards the initial weights")
    ap.add_argument("--trust", type=int, default=0, help="1: add the map-trust head")
    ap.add_argument("--wtrust", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev", type=int, default=1)
    ap.add_argument("--dev_pert_conds", default=DEV_PERT)
    ap.add_argument("--dev_perts", default=",".join(PERTS), help="perturbations of the DEV protocol (default: all seven)")
    ap.add_argument("--dev_stride", type=int, default=2, help="every n-th labelled VAL row for the clean DEV cells")
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--test_clean_conds", default=TEST_CLEAN)
    ap.add_argument("--test_pert_conds", default=TEST_PERT)
    ap.add_argument("--test_perts", default=",".join(PERTS))
    ap.add_argument("--agree_thr", type=float, default=0.4, help="gate threshold of the map-agreement score")
    ap.add_argument("--trust_thr", type=float, default=0.5, help="gate threshold of the learned trust score")
    ap.add_argument("--keep_arrays", type=int, default=0, help="store per-sample arrays in the result (final models)")
    ap.add_argument("--alt", type=int, default=0, help="E75: counterfactual-route evaluation on TEST rows with an alternative goal")
    ap.add_argument("--test_pert_stride", type=int, default=1, help="every n-th TEST row for the perturbed cells (cheaper evaluation)")
    ap.add_argument("--sd_eps_lo", type=float, default=0.5)
    ap.add_argument("--sd_eps_hi", type=float, default=2.0)
    ap.add_argument("--sd_pcnt", type=float, default=0.1)
    ap.add_argument("--sd_tr_eps_lo", type=float, default=-1.0, help="SD quality during TRAINING (default: same as the evaluation values --sd_eps_lo/hi/--sd_pcnt)")
    ap.add_argument("--sd_tr_eps_hi", type=float, default=-1.0)
    ap.add_argument("--sd_tr_pcnt", type=float, default=-1.0)
    ap.add_argument("--load", default="", help="evaluation only: a saved run name (out/loop2/<name>/model.pt) or a .pt path; no training")
    ap.add_argument("--soup", default="", help="evaluation only: weight average of saved models 'nameA:wA,nameB:wB' written to <name>/model.pt")
    a = ap.parse_args()
    global AGREE_THR, KEEP_ARRAYS, TRUST_THR
    AGREE_THR, KEEP_ARRAYS, TRUST_THR = a.agree_thr, bool(a.keep_arrays), a.trust_thr
    SD_CFG.update(eps_lo=a.sd_eps_lo if a.sd_tr_eps_lo < 0 else a.sd_tr_eps_lo, eps_hi=a.sd_eps_hi if a.sd_tr_eps_hi < 0 else a.sd_tr_eps_hi, p_cnt=a.sd_pcnt if a.sd_tr_pcnt < 0 else a.sd_tr_pcnt)
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
    res = dict(args=vars(a))
    models = []  # (suffix, state)
    if a.soup:  # weight average of saved models, e.g. WiSE-FT interpolation 'e60_main:0.5,e73_x:0.5'
        parts = [(p.split(":")[0], float(p.split(":")[1])) for p in a.soup.split(",")]
        cks = [torch.load(f"{OUT}/{n}/model.pt", map_location="cpu", weights_only=False) for n, _ in parts]
        st = {}
        allkeys = {}
        for ck in cks:
            for k, v in ck["state"].items():
                allkeys.setdefault(k, v)
        for k, v in allkeys.items():
            if v.dtype.is_floating_point:
                have = [(w, ck) for (_, w), ck in zip(parts, cks) if k in ck["state"]]
                tot = sum(w for w, _ in have)  # keys missing in some checkpoints (e.g. the trust head) are averaged over the models that have them
                st[k] = sum(w / tot * ck["state"][k].float() for w, ck in have)
            else:
                st[k] = v
        torch.save(dict(state=st, mean=cks[0]["mean"], std=cks[0]["std"], args=vars(a)), f"{OUT}/{a.name}/model.pt")
        a.load = a.name
    if a.load:  # evaluation of a saved model: --load <run name> or a path to a .pt file
        path = a.load if a.load.endswith(".pt") else f"{OUT}/{a.load}/model.pt"
        ck = torch.load(path, map_location="cuda", weights_only=False)
        cin_ck = [v.shape[1] for k, v in ck["state"].items() if k.startswith("stem") and v.ndim == 4][0]
        net, mean, std = make_net(None, any(k.startswith("trust_head") for k in ck["state"]), nmap=cin_ck - 39)
        net.load_state_dict(ck["state"])
        mean, std = ck["mean"].cuda(), ck["std"].cuda()
        models.append(("", None))
    else:
        net, mean, std = make_net(a.init_from, bool(a.trust), nmap=n_map(a.maps))
        frozen = set_trainable(net, a.train)
        n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
        P("parameters", sum(p.numel() for p in net.parameters()), "trainable", n_tr)
        RTtr = pickle.load(open(f"{D}/route_gt_train.pkl", "rb"))
        tr_idx = np.array([i for i in range(SP["n"]["train"]) if RTtr[i] is not None])
        cfg = dict(maps=a.maps, noise_train=a.noise_train, p_wrong=a.p_wrong, p_none=a.p_none, p_exact=a.p_exact, sig_lo=a.sig_lo, sig_hi=a.sig_hi, pert=None, pert_train=a.pert_train, p_use_pert=a.p_use_pert)
        ds = TrainData("train", tr_idx, True, cfg)
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd, eps=1e-6, betas=(0.9, 0.98))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
        scaler = torch.cuda.amp.GradScaler(growth_interval=200)
        torch.backends.cudnn.benchmark = True
        sampler = None
        if a.pert_train:
            prow = set()
            for pn in a.pert_train.split(","):
                prow |= {int(r) for r in np.load(f"{D}/pert/{pn}/train_rows.npy")}
            isp = np.array([int(i) in prow for i in tr_idx])
            w = np.where(isp, a.p_pert_mass / max(int(isp.sum()), 1), (1 - a.p_pert_mass) / max(int((~isp).sum()), 1))
            sampler = torch.utils.data.WeightedRandomSampler(torch.from_numpy(w), num_samples=a.steps * a.bs + 64, replacement=True)
            P("perturbed rows", int(isp.sum()), "of", len(isp), "; mass", a.p_pert_mass)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=(sampler is None), sampler=sampler, num_workers=a.workers, drop_last=True, persistent_workers=True, prefetch_factor=4)
        theta0 = {k: p.detach().clone() for k, p in net.named_parameters() if p.requires_grad} if a.l2sp > 0 else None
        ema = EMA(net, a.ema) if a.ema > 0 else None
        use_trust = bool(a.trust)
        net.train()
        for m in frozen:
            m.eval()
        step, t0, run, curve = 0, time.time(), [], []
        gema, n_clip = None, 0
        while step < a.steps:
            for b in dl:
                x = prep(b, mean, std)
                y, dg, yr = b["y"].cuda(non_blocking=True), b["dirf"].cuda(non_blocking=True), b["yr"].cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    o, t = net(x, with_trust=True)
                o = o.float()
                loss = DH.loss_fn(o[:, 0], y, a.lam, a.wpos, a.gamma) + a.dir * DH.dir_loss(o[:, 1:3], dg) + a.wroute * DH.loss_fn(o[:, 3], yr, a.lam, a.wpos, a.gamma)
                if use_trust:
                    tv, tl = b["tvalid"].cuda(), b["tlabel"].cuda()
                    lt = (F.binary_cross_entropy_with_logits(t.float(), tl, reduction="none") * tv).sum() / tv.sum().clamp_min(1.0)
                    loss = loss + a.wtrust * lt * 10.0
                if theta0 is not None:
                    loss = loss + a.l2sp * sum(((p - theta0[k]) ** 2).sum() for k, p in net.named_parameters() if k in theta0)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), float("inf")))
                if np.isfinite(gn):
                    gema = gn if gema is None else gema
                    lim = 3.0 * gema
                    if gn > lim:
                        for p_ in net.parameters():
                            if p_.grad is not None:
                                p_.grad.mul_(lim / gn)
                        n_clip += 1
                    gema = 0.98 * gema + 0.02 * min(gn, lim)
                scaler.step(opt)
                scaler.update()
                sched.step()
                if ema is not None:
                    ema.update(net)
                run.append(float(loss))
                step += 1
                if step % 250 == 0:
                    curve.append((step, float(np.mean(run[-250:]))))
                    P(f"step {step}/{a.steps} loss {curve[-1][1]:.4f} ({(time.time() - t0) / step:.2f} s/step) clipped {n_clip}")
                if step >= a.steps:
                    break
        torch.save(dict(state=net.state_dict(), mean=mean.cpu(), std=std.cpu(), args=vars(a)), f"{OUT}/{a.name}/model.pt")
        models.append(("", None))
        if ema is not None:
            torch.save(dict(state=ema.sd, mean=mean.cpu(), std=std.cpu(), args=vars(a)), f"{OUT}/{a.name}/model_ema.pt")
            models.append(("__ema", ema.sd))
        res.update(curve=curve, train_s=time.time() - t0, clipped_steps=n_clip)
        del dl
    SD_CFG.update(eps_lo=a.sd_eps_lo, eps_hi=a.sd_eps_hi, p_cnt=a.sd_pcnt)  # evaluation always with the evaluation quality of the SD map
    for suffix, sd in models:
        if sd is not None:
            net.load_state_dict(sd)
        P("== evaluating", a.name + suffix)
        out = {}
        if a.dev:
            out["dev"] = dev_eval(net, mean, std, a, P)
        if a.test:
            out["test"] = test_eval(net, mean, std, a, P)
        if a.alt:
            out["alt"] = alt_eval(net, mean, std, a, P)
        res["eval" + suffix] = out
    json.dump(res, open(f"{OUT}/{a.name}/{'result_eval.json' if a.load else 'result.json'}", "w"), indent=1)
    P("DONE", time.strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()
