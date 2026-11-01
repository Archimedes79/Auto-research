"""Loop 7 / E79: temporal lane propagation and fusion of the dense end-to-end head outputs (lane + route probability maps), training-free.
Protocol (as E67/E68, TEST scenes): in every scene the rows are ordered in time; the first half uses clean features, the second half the features of one perturbation type (failure phase, 1 s between frames).
The SD map of every frame carries a SCENE-CONSTANT pose bias (world-frame translation sigma, yaw 2 sigma deg; it appears in the ego frame as R_t^T w), like a real map bias.
Per frame the head gives X_t = (lane, route) with the map and L0_t = lane probability without map (map-independent hypotheses). Fusion variants use the ego-motion-warped result of the previous frame:
  single | ema_p(a) probability EMA | ema_l(a) logit EMA | gate: map-independent verification (a lane/route hypothesis is only kept from the memory where the map-free hypothesis or the current evidence agrees)
  hold_oracle / hold_mon: during (detected) failure frames the warped memory replaces the frame | ema+hold.
Failure monitor (label-free): lane mass of L0 compared with the running reference of the last unflagged frames.
Usage (gcar): temporal_eval.py --model e78_sd_a --maps sd --sigma 1 --out e79_a"""
import argparse
import json
import os
import pickle
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc_lib as AL  # noqa: E402
import dense_head as DH  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from p13_temporal_val import make_grid, warp  # noqa: E402  (the copies of these two functions were identical)
import route_head3 as RH3  # noqa: E402
import sd_lib as SL  # noqa: E402

D, OUT = RH3.D, RH3.OUT
DEV = "cuda"
PERTS = RH3.PERTS


def load_model(name, maps):
    ck = torch.load(f"{OUT}/{name}/model.pt", map_location="cuda", weights_only=False)
    cin_ck = [v.shape[1] for k, v in ck["state"].items() if k.startswith("stem") and v.ndim == 4][0]
    net, mean, std = RH3.make_net(None, any(k.startswith("trust_head") for k in ck["state"]), nmap=cin_ck - 39)
    net.load_state_dict(ck["state"])
    net.eval()
    load_model.args = ck.get("args", {})
    return net, ck["mean"].cuda(), ck["std"].cuda()


def sd_map_for(i, P, SDm, sigma):
    """SD map raster [5,200,200] of row i with the scene-constant world-frame bias (translation sigma, yaw 2 sigma deg)."""
    sc = int(P["scene"][i])
    rs = np.random.RandomState((zlib.crc32(f"scenebias{sc}".encode()) + 3) % (2 ** 32))
    w = rs.randn(2) * sigma
    th = rs.randn() * 2.0 * sigma
    d = P["R"][i].T @ w
    sd = SDm[i] if not WRONG else SDm[(i + len(SDm) // 2) % len(SDm)]  # WRONG: the SD map of another place for the whole sequence
    if sd is None:
        return np.zeros((5, 200, 200), np.float32)
    dg = SL.degrade_sd(sd, np.random.RandomState((zlib.crc32(f"sd{sc}".encode()) + i) % (2 ** 32)))
    return SL.raster_sd(dg, AL.rot2(th), d)


@torch.no_grad()
def infer_rows(net, mean, std, feat, rows, P, SDm, sigma, maps_mode, workers=3, split="test", part="val"):
    """-> {row: (L0, L, R)} half-precision CPU tensors [200,200]; the map of every row carries the scene-constant bias."""
    ds = RH3.EvalSet(part, rows, feat, RH3.parse_conds("exact:0:0"), maps_mode=maps_mode, pert_prefix=split)
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=workers)
    out = {}
    for b in dl:
        x = b["x"].cuda(non_blocking=True)
        mp = torch.from_numpy(np.stack([sd_map_for(i, P, SDm, sigma) for i in b["idx"].tolist()])).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            o0 = net((torch.cat([x, torch.zeros_like(mp)], 1) - mean[None, :, None, None]) / std[None, :, None, None]).float()
            o = net((torch.cat([x, mp], 1) - mean[None, :, None, None]) / std[None, :, None, None]).float()
        p0 = torch.sigmoid(o0[:, 0])
        dil = F.max_pool2d((mp[:, 1:2] > 0).float(), 11, 1, 5)[:, 0]  # SD road corridor (+-2.5 m)
        ag = ((p0 * dil).sum((1, 2)) / dil.sum((1, 2)).clamp_min(1.0)).cpu()
        L0, L, R = p0.half().cpu(), torch.sigmoid(o[:, 0]).half().cpu(), torch.sigmoid(o[:, 3]).half().cpu()
        for k, i in enumerate(b["idx"].tolist()):
            out[i] = (L0[k], L[k], R[k], float(ag[k]))
    return out


def logit(p):
    p = p.clamp(1e-3, 1 - 1e-3)
    return torch.log(p / (1 - p))


def fuse_scene(rows, X, X0, fail_true, P, variant, par, mon=None):
    """rows: time-ordered rows; X[f] = [2,200,200] (lane, route) probability on the GPU; X0[f] = map-free lane [200,200]; fail_true[f] bool; mon[f] bool (flagged) or None.
    Returns list of fused [2,200,200] (GPU)."""
    out, mem, prev = [], None, None
    for f, row in enumerate(rows):
        x = X[f]
        if variant in ("single0", "ema0"):
            x = torch.stack([X0[f], x[1]])
        y = x
        w = valid = None
        if mem is not None:
            Rc_, tc_ = P["R"][row], P["t"][row]
            if ODO[0] > 0 or ODO[1] > 0:  # odometry error of the relative pose: yaw N(0, odo deg * sqrt(dt)), translation N(0, odo_t * distance) per axis (independent per frame pair)
                rg = np.random.RandomState((zlib.crc32(f"odo{row}".encode()) + 5) % (2 ** 32))
                dt_ = max(float(P["ts"][row] - P["ts"][prev]), 0.1)
                dist = float(np.linalg.norm(P["t"][row] - P["t"][prev]))
                Rc_ = Rc_ @ AL.rot2(rg.randn() * ODO[0] * np.sqrt(dt_))
                tc_ = tc_ + rg.randn(2) * ODO[1] * dist
            w, valid = warp(mem, make_grid(P["R"][prev], P["t"][prev], Rc_, tc_))
            if variant == "ema_pLa":  # self-alignment of the memory: small search over yaw / shift of the warped memory for the best normalised correlation of its lane channel with the current lane output
                xl = x[0]
                best, bw, bv = None, w, valid
                for k_, (psi, tx, ty) in enumerate(ALIGN_GRID):
                    if k_ == 0:
                        w_c, v_c = w, valid
                    else:
                        w_c, v_c = warp(mem, make_grid(P["R"][prev], P["t"][prev], Rc_ @ AL.rot2(psi), tc_ + np.array([tx, ty])))
                    m_ = v_c.float()
                    num = (w_c[0] * xl * m_).sum()
                    den = torch.sqrt((w_c[0] ** 2 * m_).sum() * (xl ** 2 * m_).sum()) + 1e-6
                    sc_ = float(num / den) - ALIGN_PEN * (abs(tx) + abs(ty) + 0.5 * abs(psi))
                    if best is None or sc_ > best:
                        best, bw, bv = sc_, w_c, v_c
                w, valid = bw, bv
        flag = bool(fail_true[f]) if variant.endswith("oracle") else bool(mon[f]) if (mon is not None and variant.endswith("mon")) else False
        a = par.get("a", 0.5)
        if variant in ("single", "single0") or w is None:
            y = x
        elif variant == "ema_p":
            y = torch.where(valid[None], a * w + (1 - a) * x, x)
        elif variant in ("rsel", "rselh"):  # lane: EMA (a); route: keep only the route components that continue the previous (warped) route, optionally hold the previous route when the frame has none
            lane = torch.where(valid, a * w[0] + (1 - a) * x[0], x[0])
            r = x[1].clone()
            if w is not None:
                from scipy import ndimage as ndi
                rb = (r >= 0.5).cpu().numpy()
                pm = (w[1] >= 0.5).float()
                pmd = (F.max_pool2d(pm[None, None], 7, 1, 3)[0, 0] > 0).cpu().numpy()
                lab, nl = ndi.label(rb)
                keep = np.zeros_like(rb)
                for c_ in range(1, nl + 1):
                    m_ = lab == c_
                    if (m_ & pmd).sum() >= 0.3 * m_.sum():
                        keep |= m_
                if nl > 0 and keep.any():
                    r = r * torch.from_numpy(keep.astype(np.float32)).to(r.device)
                elif variant == "rselh" and (nl == 0 or not rb.any()) and pm.sum() > 0:
                    r = 0.6 * pm  # hold: the previous route (warped) replaces an empty route output
            y = torch.stack([lane, r])
        elif variant == "ema0":  # EMA of the map-free lane hypothesis (the route channel of the frame is kept)
            y = torch.where(valid[None], a * w + (1 - a) * x, x)
            y = torch.stack([y[0], x[1]])
        elif variant in ("ema_pL", "ema_pLa"):  # lane only; the route output of the frame is kept (ema_pLa: with the self-alignment above)
            y = torch.where(valid[None], a * w + (1 - a) * x, x)
            y = torch.stack([y[0], x[1]])
        elif variant == "ema_l":
            y = torch.where(valid[None], torch.sigmoid(a * logit(w) + (1 - a) * logit(x)), x)
        elif variant == "gate":
            # map-independent verification: memory counts only where the current map-free hypothesis (dilated) or the current map-conditioned lane evidence supports it
            sup = F.max_pool2d(torch.maximum(X0[f], x[0])[None, None], 7, 1, 3)[0, 0] > par.get("thr", 0.3)
            wv = torch.where(sup[None] & valid[None], w, torch.zeros_like(w))
            y = torch.maximum(x, par.get("g", 0.8) * wv)
        elif variant in ("hold_oracle", "hold_mon"):
            y = torch.where(valid[None], w, x) if flag else x
        elif variant in ("ema_hold_oracle", "ema_hold_mon"):
            y = torch.where(valid[None], w, x) if flag else torch.where(valid[None], a * w + (1 - a) * x, x)
        out.append(y)
        mem, prev = y, row
    return out


def mon_flags(rows, AG):
    """label-free failure monitor: agreement of the map-free lane hypothesis with the SD road corridor below par_thr x the median of the last 4 unflagged frames (the first 3 frames of a scene are never flagged)."""
    mass = list(AG)
    ref, flags = [], []
    for f, m in enumerate(mass):
        if par_abs > 0:
            fl = m < par_abs
        else:
            r = np.median(ref[-4:]) if len(ref) >= 3 else None
            fl = r is not None and m < par_thr * r
        flags.append(fl)
        if not fl:
            ref.append(m)
    return flags, mass


par_thr = 0.6
WRONG = False
ODO = [0.0, 0.0]
ALIGN_PEN = 0.01
ALIGN_GRID = [(0.0, 0.0, 0.0)] + [(psi, tx, ty) for psi in (-3.0, -1.5, 0.0, 1.5, 3.0) for tx in (-1.5, -0.75, 0.0, 0.75, 1.5) for ty in (-1.5, -0.75, 0.0, 0.75, 1.5) if (psi, tx, ty) != (0.0, 0.0, 0.0)]
par_abs = -1.0


_P = None


def consistency(lane, rows, tau=0.5):
    """temporal match quality of the extracted lane points: between consecutive frames of a scene the points of frame t-1 are warped with the ego motion into frame t; consistency = mean of (share of the current points that lie within 1 m of a warped previous point,
    share of the warped previous points within 1 m of a current point), both restricted to the area seen in both frames."""
    sk = []
    for s_ in range(0, len(rows), 32):
        sk += [np.asarray(x, np.float64) for x in DH.skeleton_points(lane[s_ : s_ + 32].float().to(DEV), tau)]
    vals = []
    for k in range(1, len(rows)):
        r0, r1 = rows[k - 1], rows[k]
        if int(_P["scene"][r0]) != int(_P["scene"][r1]) or len(sk[k]) < 5 or len(sk[k - 1]) < 5:
            continue
        Rp, tp, Rc, tc = _P["R"][r0], _P["t"][r0], _P["R"][r1], _P["t"][r1]
        w = ((sk[k - 1] @ Rp.T) + tp - tc) @ Rc  # previous points in the current frame
        cur_back = ((sk[k] @ Rc.T) + tc - tp) @ Rp  # current points in the previous frame
        w = w[(np.abs(w) < 48).all(1)]
        cb = np.abs(cur_back) < 48
        cb = cb.all(1)
        if len(w) < 5 or cb.sum() < 5:
            continue
        a = float((cKDTree(w).query(sk[k][cb])[0] <= 1.0).mean())
        b = float((cKDTree(sk[k][cb]).query(w)[0] <= 1.0).mean())
        vals.append(0.5 * (a + b))
    return float(np.mean(vals)) if vals else float("nan")


def metrics_for(fused, rows, ids_all, gts_all, tau=0.5):
    """lane F1 / P / R and route success over a list of fused frames (list [2,200,200] GPU)"""
    if not fused:
        return {}
    lane = torch.stack([y[0].half().cpu() for y in fused])
    route = torch.stack([y[1].half().cpu() for y in fused])
    gts = [gts_all[int(r)].astype(np.float32) for r in rows]
    m = RH3.lane_prf(lane, gts, tau)
    re_ = RH3.route_eval(route, np.asarray(rows), "val", tau)
    re_.pop("succ")
    m.update(route_success=re_["route_success"], route_cov=re_["route_cov"], route_prec=re_["route_prec"])
    m["n"] = len(rows)
    if _P is not None:
        m["consistency"] = consistency(lane, rows, tau)
    return m


class FeatReader:
    def __init__(self, feat, rows, split, part="val"):
        self.ds = RH3.EvalSet(part, rows, feat, RH3.parse_conds("exact:0:0"), maps_mode="sd", pert_prefix=split)
        self.k = {int(r): k for k, r in enumerate(rows)}

    def __call__(self, row):
        return self.ds[self.k[row]]["x"]


def filter_t(lane, route, thr, rng_m):
    if thr > 0:
        lane = torch.where(lane >= thr, lane, torch.zeros_like(lane))
        route = torch.where(route >= thr, route, torch.zeros_like(route))
    if rng_m > 0:
        c0 = int((rng_m + 50.0) / 0.5)
        lane, route = lane.clone(), route.clone()
        lane[:, c0:] = 0.0
        route[:, c0:] = 0.0
    return lane, route


@torch.no_grad()
def run_rec(net, mean, std, a, P, SDm, seqs, rd_clean, rd_pert, cfg, gts_all):
    """recurrent evaluation of a temporal head (maps sdt): memory M = (window) EMA of the filtered outputs of the model itself, warped with the ego motion into the current frame (cfg: use, src, thr, rng, hist, a)."""
    fused = dict(fail=[], cl=[])
    rows_o = dict(fail=[], cl=[])
    ag_all, true_all = [], []
    nrm = lambda z: (z - mean[None, :, None, None]) / std[None, :, None, None]
    h, aa = int(cfg.get("hist", 1)), float(cfg.get("a", 0.7))
    a_r = (sum(aa ** j for j in range(1, h)) / sum(aa ** j for j in range(h))) if h > 1 else 0.0
    for rows in seqs:
        n = len(rows)
        f0 = n // 2
        M, prow = None, None  # memory in the coordinates of frame prow
        for f, r in enumerate(rows):
            x = (rd_pert if f >= f0 else rd_clean)(r).cuda().float()[None]
            mp = torch.from_numpy(sd_map_for(r, P, SDm, a.sigma)).cuda()[None]
            pri = torch.zeros(1, 3, 200, 200, device=DEV)
            wm = vm = None
            if M is not None:
                wm, vm = warp(M, make_grid(P["R"][prow], P["t"][prow], P["R"][r], P["t"][r]))
                if cfg["use"]:
                    pri = torch.cat([wm, vm[None].float()], 0)[None]
            with torch.autocast("cuda", dtype=torch.float16):
                o0 = net(nrm(torch.cat([x, torch.zeros_like(mp), pri], 1))).float()
                o1 = net(nrm(torch.cat([x, mp, pri], 1))).float()
            L0, L, Rr = torch.sigmoid(o0[0, 0]), torch.sigmoid(o1[0, 0]), torch.sigmoid(o1[0, 3])
            ph = "fail" if f >= f0 else "cl"
            fused[ph].append(torch.stack([L, Rr]))
            rows_o[ph].append(r)
            dil = F.max_pool2d((mp[:, 1:2] > 0).float(), 11, 1, 5)[0, 0]
            ag_all.append(float((L0 * dil).sum() / dil.sum().clamp_min(1.0)))
            true_all.append(int(f >= f0))
            lane = L0 if cfg["src"] == "free" else L
            route = torch.zeros_like(lane) if cfg["src"] == "free" else Rr
            lane, route = filter_t(lane, route, cfg["thr"], cfg["rng"])
            Y = torch.stack([lane, route])
            if h > 1 and wm is not None:
                Y = torch.where(vm[None], a_r * wm + (1 - a_r) * Y, Y)
            M, prow = Y, r
    return dict(fail=metrics_for(fused["fail"], rows_o["fail"], None, gts_all), clean=metrics_for(fused["cl"], rows_o["cl"], None, gts_all), auroc_agree=float(auroc(-np.array(ag_all), np.array(true_all))))


def auroc(s, l):
    return RH3.auroc(s, l)


def main():
    global par_thr, par_abs, WRONG, ODO
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="e78_sd_a")
    ap.add_argument("--maps", default="sd")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--perts", default=",".join(PERTS))
    ap.add_argument("--min_frames", type=int, default=8)
    ap.add_argument("--mon_thr", type=float, default=0.6)
    ap.add_argument("--mon_abs", type=float, default=-1.0, help="absolute agreement threshold of the failure monitor (chosen on VAL); <= 0: relative rule")
    ap.add_argument("--odo", type=float, default=0.0, help="odometry yaw error of the relative pose, deg per sqrt(s)")
    ap.add_argument("--odo_t", type=float, default=0.0, help="odometry translation error as a fraction of the travelled distance")
    ap.add_argument("--wrong", type=int, default=0, help="1: the SD map is the map of another place (never right)")
    ap.add_argument("--recurrent", type=int, default=0, help="1: evaluate a temporal head (maps sdt) recurrently with its own previous outputs as memory")
    ap.add_argument("--split", default="test", help="test | val (DEV: VAL scenes, perturbed rows every 3rd row)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="single:0,ema_p:0.5,ema_p:0.7,ema_l:0.5,ema_l:0.7,gate:0,hold_oracle:0,hold_mon:0,ema_hold_oracle:0.5,ema_hold_mon:0.5")
    a = ap.parse_args()
    par_thr, par_abs, WRONG = a.mon_thr, a.mon_abs, bool(a.wrong)
    ODO = [a.odo, a.odo_t]
    os.makedirs(f"{OUT}/{a.out}", exist_ok=True)
    P = dict(np.load(f"{D}/poses_val.npz"))
    global _P
    _P = P
    SDm = RH3.sd_maps("val")
    RTv = pickle.load(open(f"{D}/route_gt_val.pkl", "rb"))
    gts_all = RH3.gt_points("val")
    net, mean, std = load_model(a.model, a.maps)
    SP = RH3.SP
    if a.recurrent:
        targs = load_model.args
        cfg0 = dict(src=targs.get("prior_src", "free"), thr=targs.get("prior_thr", 0.0), rng=targs.get("prior_range", 0.0), hist=targs.get("prior_hist", 1), a=targs.get("prior_a", 0.7))
        rows_all = [i for i in SP[a.split] if RTv[i] is not None]
        rd_clean = FeatReader("clean", rows_all, a.split)
        res = dict(args=vars(a), cfg=cfg0, perts={})
        for pn in a.perts.split(","):
            have = set(np.load(f"{D}/pert/{pn}/{a.split}_rows.npy").tolist())
            rows_p = [i for i in rows_all if i in have]
            rd_pert = FeatReader(pn, rows_p, a.split)
            by = {}
            for r in rows_p:
                by.setdefault(int(P["scene"][r]), []).append(r)
            seqs = [sorted(rr, key=lambda i: P["ts"][i]) for rr in by.values() if len(rr) >= a.min_frames]
            out_p = {}
            for key, cfg in (("rec_noprior", dict(cfg0, use=False)), ("rec", dict(cfg0, use=True))):
                out_p[key] = run_rec(net, mean, std, a, P, SDm, seqs, lambda r: rd_clean(r), rd_pert, cfg, gts_all)
                mf, mc = out_p[key]["fail"], out_p[key]["clean"]
                print(f"[{pn}] {key:12s} failure: laneF1 {mf['lane_F1']:.3f} (P {mf['lane_P']:.3f} R {mf['lane_R']:.3f}) route {mf['route_success']:.3f} | clean phase: laneF1 {mc['lane_F1']:.3f} route {mc['route_success']:.3f}", flush=True)
            res["perts"][pn] = out_p
        json.dump(res, open(f"{OUT}/{a.out}/result.json", "w"), indent=1)
        print("DONE", time.strftime("%H:%M:%S"))
        return
    test_rows = [i for i in SP[a.split]]
    t0 = time.time()
    clean_rows = [i for i in test_rows if RTv[i] is not None]
    CL = infer_rows(net, mean, std, "clean", clean_rows, P, SDm, a.sigma, a.maps, split=a.split)
    print(f"clean outputs {len(CL)} rows, {time.time() - t0:.0f} s", flush=True)
    variants = [(v.split(":")[0], float(v.split(":")[1])) for v in a.variants.split(",")]
    res = dict(args=vars(a), perts={})
    examples = {}
    for pn in a.perts.split(","):
        have = np.load(f"{D}/pert/{pn}/{a.split}_rows.npy").tolist()
        rows_p = [i for i in clean_rows if i in set(have)]
        PT = infer_rows(net, mean, std, pn, rows_p, P, SDm, a.sigma, a.maps, split=a.split)
        by = {}
        for r in rows_p:
            by.setdefault(int(P["scene"][r]), []).append(r)
        seqs = [sorted(rr, key=lambda i: P["ts"][i]) for rr in by.values() if len(rr) >= a.min_frames]
        out_p = {}
        for (var, a_par) in variants:
            key = f"{var}:{a_par:g}" if var in ("ema_p", "ema_pL", "ema_pLa", "ema0", "ema_l", "ema_hold_oracle", "ema_hold_mon", "rsel", "rselh") else var
            fr_fused, fr_rows, cl_fused, cl_rows = [], [], [], []
            mon_all, true_all, mass_all = [], [], []
            for rows in seqs:
                n = len(rows)
                f0 = n // 2
                X, X0, AG = [], [], []
                for f, r in enumerate(rows):
                    L0, L, R, ag = (PT if f >= f0 else CL)[r]
                    X.append(torch.stack([L, R]).cuda().float())
                    X0.append(L0.cuda().float())
                    AG.append(ag)
                fail_true = [f >= f0 for f in range(n)]
                fl, mass = mon_flags(rows, AG)
                fused = fuse_scene(rows, X, X0, fail_true, P, var, dict(a=a_par), fl)
                for f, r in enumerate(rows):
                    (fr_fused, fr_rows) if f >= f0 else (cl_fused, cl_rows)
                    if f >= f0:
                        fr_fused.append(fused[f])
                        fr_rows.append(r)
                    else:
                        cl_fused.append(fused[f])
                        cl_rows.append(r)
                mon_all += fl
                true_all += fail_true
                mass_all += mass
            mf = metrics_for(fr_fused, fr_rows, None, gts_all)
            mc = metrics_for(cl_fused, cl_rows, None, gts_all)
            out_p[key] = dict(fail=mf, clean=mc)
            if var == "single":
                out_p["_mon_data"] = dict(ag=[round(float(z), 4) for z in mass_all], fail=[int(z) for z in true_all])
                out_p["_monitor"] = dict(acc=float(np.mean(np.array(mon_all) == np.array(true_all))), flag_fail=float(np.mean(np.array(mon_all)[np.array(true_all)])), flag_clean=float(np.mean(np.array(mon_all)[~np.array(true_all)])), auroc_mass=float(auroc(-np.array(mass_all), np.array(true_all).astype(int))))
            print(f"[{pn}] {key:18s} failure: laneF1 {mf['lane_F1']:.3f} (P {mf['lane_P']:.3f} R {mf['lane_R']:.3f}) route {mf['route_success']:.3f} | clean phase: laneF1 {mc['lane_F1']:.3f} route {mc['route_success']:.3f}", flush=True)
            if var in ("single", "ema_l", "hold_mon") and pn in ("dark", "cam_front3_off", "blur") and a_par in (0.0, 0.5):
                # keep one scene for a figure: lane maps of the last 6 failure frames
                rows = seqs[len(seqs) // 2]
                examples.setdefault(pn, {})[key] = dict(rows=rows[-6:])
        res["perts"][pn] = out_p
        print(f"[{pn}] monitor: {out_p['_monitor']}", flush=True)
    json.dump(res, open(f"{OUT}/{a.out}/result.json", "w"), indent=1)
    print("DONE", time.strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()
