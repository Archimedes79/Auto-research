"""Loop 8: shared library of the perception-first pipeline (perceived lane graph -> rough localisation -> assignment to the SD map -> navigation attributes).
Frames: everything the network sees is in the ego (vehicle) frame, x forward, y left, metres. The SD map carries a rigid pose error (translation d, yaw); the rough localisation estimates the correction,
the corrected SD map is expressed in the ego frame, and the SD-relative piece features are computed there.
"""
import os
import pickle
import sys
import zlib

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc_lib as AL  # noqa: E402
import graph_eval as GE  # noqa: E402
import sd_lib as SL  # noqa: E402

HOME = os.path.expanduser("~")
D = f"{HOME}/exp_gcar_b123"
OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
LANE_W = 3.5
KPT = 6  # sampled points per piece
_C = {}


def seed_of(x, salt=0):
    return (zlib.crc32(str(x).encode()) + 7919 * salt) % (2 ** 32)


# ------------------------------------------------------------------ data


def cache(name, part):
    key = (name, part)
    if key not in _C:
        if name == "gt":
            _C[key] = pickle.load(open(f"{D}/graph_gt_{part}.pkl", "rb"))
        elif name == "rt":
            _C[key] = pickle.load(open(f"{D}/route_gt_{part}.pkl", "rb"))
        elif name == "sd":
            _C[key] = pickle.load(open(f"{D}/sd_{part}.pkl", "rb"))
        elif name == "rpts":
            _C[key] = pickle.load(open(f"{D}/route_pts_{part}.pkl", "rb"))
        elif name == "poses":
            z = np.load(f"{D}/poses_{part}.npz")
            _C[key] = {k: z[k] for k in z.files}
        elif name == "src":
            _C[key] = pickle.load(open(f"{D}/p8/{part}.pkl", "rb"))
    return _C[key]


def load_src(src):
    """perceived-graph cache of p8_build.py"""
    return cache("src", src)


def gt_lane_points(part, row, ds=0.5):
    P, _, _ = GE.dense_points(cache("gt", part)[int(row)], ds)
    return P


# ------------------------------------------------------------------ SD map handling


def draw_sd(sd, row, sigma, drop=0.0, salt=1, eps_lo=0.5, eps_hi=2.0, p_cnt=0.1, bias=0.0):
    """degraded SD map + rigid pose error (translation sigma m, yaw 2 sigma deg); returns (sd_map_frame, d, R) with sd_map_frame = dict(roads, route) of (poly, cnt) in the (wrongly posed) map frame."""
    rng = np.random.RandomState(seed_of(row, salt))
    d = rng.randn(2) * sigma
    R = AL.rot2(rng.randn() * 2.0 * sigma)
    dg = SL.degrade_sd(sd, rng, eps_lo=eps_lo, eps_hi=eps_hi, p_cnt=p_cnt, drop=drop)
    if bias > 0:  # per-road lateral bias (E83 quality study): every road polyline and route piece is shifted sideways by its own N(0, bias) draw
        for key in ("roads", "route"):
            new_ = []
            for p, c in dg[key]:
                p = np.asarray(p, np.float64)
                if len(p) >= 2:
                    t = AL.tangents(p)
                    p = p + rng.randn() * bias * np.stack([-t[:, 1], t[:, 0]], 1)
                new_.append((p, c))
            dg[key] = new_
    out = {k: [(np.asarray(p, np.float64) @ R.T + d, np.asarray(c, np.float64)) for p, c in dg[k]] for k in ("roads", "route")}
    return out, d, R


def sd_vertices(sdm, ds=1.0):
    """SD roads/route -> vertex arrays at ds spacing: dict(V, T, n, rid, s) for roads and for the route (s = arclength along the concatenated route)."""
    out = {}
    for key in ("roads", "route"):
        V, T, N, R, S = [], [], [], [], []
        s0 = 0.0
        for k, (poly, cnt) in enumerate(sdm[key]):
            p = np.asarray(poly, np.float64)
            if len(p) < 2:
                continue
            seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg)])
            if s[-1] < 1e-6:
                continue
            t = np.linspace(0.0, s[-1], max(int(np.ceil(s[-1] / ds)), 1) + 1)
            q = np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)
            V.append(q)
            T.append(AL.tangents(q))
            N.append(np.interp(t, s, np.asarray(cnt, np.float64)))
            R.append(np.full(len(q), k))
            S.append(s0 + t)
            s0 += s[-1]
        if V:
            out[key] = dict(V=np.concatenate(V), T=np.concatenate(T), n=np.concatenate(N), rid=np.concatenate(R), s=np.concatenate(S))
        else:
            out[key] = dict(V=np.zeros((0, 2)), T=np.zeros((0, 2)), n=np.zeros(0), rid=np.zeros(0, int), s=np.zeros(0))
    return out


def template_vertices(sv, w=LANE_W):
    """lane-count-aware matching target: for a road with n lanes the perceived lane centre lines lie at (k - (n-1)/2) w from the SD centre line."""
    V, T = [], []
    for v, t, n in zip(sv["V"], sv["T"], sv["n"]):
        n = int(max(round(n), 1))
        nl = np.array([-t[1], t[0]])
        for k in range(n):
            V.append(v + (k - (n - 1) / 2.0) * w * nl)
            T.append(t)
    if not V:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.asarray(V), np.asarray(T)


def perceived_vertices(g, keep=None):
    pcs = [p for k, p in enumerate(g["pts"]) if keep is None or keep[k]]
    V, T, pid, _ = AL.vertices(g["pts"] if keep is None else [p if keep[k] else np.zeros((0, 2)) for k, p in enumerate(g["pts"])])
    return V, T, pid


def localise(sdm, EV, ET, mode="icp", search=3.0, yaw_search=6.0, w=LANE_W):
    """rough localisation: correction (Rr, tr) with x_map = Rr x_ego + tr. mode: none | icp (SD centre lines) | tmpl (lane-count-aware lane template)."""
    if mode == "none" or len(EV) < 5:
        return np.eye(2), np.zeros(2), 0.0
    sv = sd_vertices(sdm)["roads"]
    if mode == "tmpl":
        MV, MT = template_vertices(sv, w)
    else:
        MV, MT = sv["V"], sv["T"]
    if len(MV) < 5:
        return np.eye(2), np.zeros(2), 0.0
    return AL.register(MV, MT, EV, ET, search=search, yaw_search=yaw_search)


def sd_to_ego(sdm, Rr, tr):
    """express the SD map (map frame) in the ego frame: x_ego = Rr^T (x_map - tr)."""
    return {k: [((np.asarray(p) - tr) @ Rr, c) for p, c in sdm[k]] for k in ("roads", "route")}


def refine_per_road(sde, EV, ET, w=LANE_W, max_shift=3.0, step=0.25, sig=0.7, min_support=10, reach=6.0, lam=0.15, margin=0.0, return_shifts=False):
    """loop 9 (E90): per-road lateral re-registration. After the rigid localisation every SD polyline (road or route piece, ego frame) is shifted sideways by the offset that best aligns its
    lane template (n offset lines at (k - (n-1)/2) w) with the perceived lane vertices near it (heading-compatible vertices within `reach`); kernel score with a prior towards zero shift;
    polylines with fewer than `min_support` supporting vertices keep their position."""
    if len(EV) < min_support:
        return (sde, []) if return_shifts else sde
    tree_e = cKDTree(EV)
    shifts = np.arange(-max_shift, max_shift + 1e-6, step)
    out, used = {}, []
    for key in ("roads", "route"):
        new = []
        for poly, cnt in sde[key]:
            p = np.asarray(poly, np.float64)
            if len(p) < 2:
                new.append((poly, cnt))
                continue
            seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg)])
            if s[-1] < 8.0:
                new.append((poly, cnt))
                continue
            t = np.arange(0.0, s[-1] + 1e-6, 1.0)
            q = np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)
            T = AL.tangents(q)
            Nq = np.stack([-T[:, 1], T[:, 0]], 1)
            nl = int(max(round(float(np.mean(np.interp(t, s, np.asarray(cnt, np.float64)))), 1), 1))
            tq = cKDTree(q)
            near = tree_e.query_ball_point(q, r=reach)
            cand = np.unique(np.concatenate([np.asarray(i, int) for i in near])) if any(len(i) for i in near) else np.zeros(0, int)
            if len(cand) < min_support:
                new.append((poly, cnt))
                continue
            d, j = tq.query(EV[cand])
            ok = (d < reach) & (np.sum(ET[cand] * T[j], axis=1) > 0.85)  # same direction of travel (oncoming lanes belong to another SD road)
            if ok.sum() < min_support:
                new.append((poly, cnt))
                continue
            lat = np.sum((EV[cand][ok] - q[j][ok]) * Nq[j][ok], axis=1)
            offs = (np.arange(nl) - (nl - 1) / 2.0) * w
            sc = np.array([np.exp(-0.5 * ((lat[:, None] - (offs[None, :] + sh)) / sig) ** 2).sum() for sh in shifts]) - lam * len(lat) * np.abs(shifts) / max_shift
            i0_ = int(np.argmin(np.abs(shifts)))
            dl = float(shifts[int(np.argmax(sc))])
            if sc.max() - sc[i0_] < margin * len(lat):  # no clear gain over the unshifted position
                dl = 0.0
            used.append(dl)
            Np = np.stack([-AL.tangents(p)[:, 1], AL.tangents(p)[:, 0]], 1)
            new.append((p + dl * Np, cnt))
        out[key] = new
    return (out, used) if return_shifts else out


# ------------------------------------------------------------------ piece features


def resample_k(p, k=KPT):
    p = np.asarray(p, np.float64)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, max(s[-1], 1e-6), k)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def _nearest_dir(V, T, tree, q, qt, r):
    """nearest same-direction vertex within r: (dist, idx), dist = r_clip when none."""
    cand = tree.query_ball_point(q, r)
    best = (r, -1)
    for c in cand:
        if float(T[c] @ qt) < AL.COS_DIR:
            continue
        d = float(np.linalg.norm(V[c] - q))
        if d < best[0]:
            best = (d, c)
    return best


NODE_GEO = 14
NODE_SD = 9 * KPT + 4
N_SCENE = 12


def scene_sd_features(sv, s0_idx):
    """scene-level navigation features from the (ego-frame) SD map: route available, lane count at the ego, upcoming turn angle (20 / 40 m), distance to the next junction on the route, number of roads."""
    f = np.zeros(N_SCENE, np.float32)
    rt, rd = sv["route"], sv["roads"]
    f[0] = float(len(rt["V"]) > 0)
    f[1] = min(len(rd["V"]) / 200.0, 3.0)
    if len(rt["V"]) < 3:
        return f
    i0 = s0_idx
    s = rt["s"]
    if i0 is None:
        return f
    n0 = rt["n"][i0]
    f[2] = n0 / 4.0

    def ang(sa, sb):
        ia, ib = int(np.argmin(np.abs(s - sa))), int(np.argmin(np.abs(s - sb)))
        ta, tb = rt["T"][ia], rt["T"][ib]
        return float(np.arctan2(ta[0] * tb[1] - ta[1] * tb[0], ta @ tb))

    s0 = s[i0]
    f[3], f[4], f[5] = ang(s0, s0 + 20), ang(s0, s0 + 40), ang(s0 + 20, s0 + 60)
    f[6] = (s.max() - s0) / 100.0
    if len(rd["V"]):
        tree = cKDTree(rd["V"])
        # junction: a road vertex within 3.5 m of a route vertex at s > s0 whose heading differs by > 35 deg
        m = (s > s0 + 3) & (s < s0 + 80)
        jd = 1.0
        for v, t, ss in zip(rt["V"][m][::2], rt["T"][m][::2], s[m][::2]):
            for c in tree.query_ball_point(v, 3.5):
                if abs(float(rd["T"][c] @ t)) < 0.82:
                    jd = (ss - s0) / 80.0
                    break
            if jd < 1.0:
                break
        f[7] = jd
    return f


def row_features(g, pprob, pdir, sdm_ego):
    """per-piece features. g: perceived graph (ego frame), sdm_ego: SD map in the ego frame (or None). -> dict(node [N, NODE_GEO+NODE_SD], rel [N,N,4], scene [N_SCENE], pts [N,KPT,2], length [N])"""
    n = len(g["pts"])
    pts = np.zeros((n, KPT, 2))
    geo = np.zeros((n, NODE_GEO), np.float32)
    ln = np.zeros(n)
    outs = {i: [] for i in range(n)}
    ins = {i: [] for i in range(n)}
    for i, j in g["edges"]:
        if i < n and j < n:
            outs[i].append(j)
            ins[j].append(i)
    for k, p in enumerate(g["pts"]):
        p = np.asarray(p, np.float64)
        if len(p) < 2:
            pts[k] = 0
            continue
        q = resample_k(p)
        pts[k] = q
        L = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        ln[k] = L
        t0, t1 = q[1] - q[0], q[-1] - q[-2]
        t0, t1 = t0 / (np.linalg.norm(t0) + 1e-9), t1 / (np.linalg.norm(t1) + 1e-9)
        pr, pdc = pprob[k], pdir[k]
        d_ego = float(np.linalg.norm(q, axis=1).min())
        geo[k] = [min(L / 50.0, 2.0), pr.mean() if len(pr) else 0, pr.min() if len(pr) else 0, pdc.mean() if len(pdc) else 0, q[0, 0] / 50, q[0, 1] / 50, q[-1, 0] / 50, q[-1, 1] / 50,
                  t0[0], t0[1], t1[0], t1[1], min(d_ego / 20.0, 2.0), min(len(outs[k]), 3) / 3.0 + 0.0 * len(ins[k])]
    rel = np.zeros((n, n, 4), np.float32)
    for i in range(n):
        for j in outs[i]:
            rel[i, j, 0] = 1.0
            rel[j, i, 1] = 1.0
    # parallel (side by side, same direction) neighbours
    for i in range(n):
        if ln[i] < 1:
            continue
        ti = np.gradient(pts[i], axis=0)
        ti /= np.linalg.norm(ti, axis=1, keepdims=True) + 1e-9
        for j in range(n):
            if i == j or ln[j] < 1:
                continue
            mid = pts[i][KPT // 2]
            dj = np.linalg.norm(pts[j] - mid, axis=1)
            c = int(np.argmin(dj))
            if dj[c] > 5.5 or dj[c] < 1.0:
                continue
            tj = pts[j][min(c + 1, KPT - 1)] - pts[j][max(c - 1, 0)]
            tj /= np.linalg.norm(tj) + 1e-9
            if float(tj @ ti[KPT // 2]) < 0.85:
                continue
            v = pts[j][c] - mid
            side = ti[KPT // 2][0] * v[1] - ti[KPT // 2][1] * v[0]  # > 0: j is on the left of i
            rel[i, j, 2 if side > 0 else 3] = 1.0
    sdf = np.zeros((n, NODE_SD), np.float32)
    scene = np.zeros(N_SCENE, np.float32)
    if sdm_ego is not None:
        sv = sd_vertices(sdm_ego)
        rt, rd = sv["route"], sv["roads"]
        ar = AL.vertices([np.vstack([[0.0, 0.0], [1.0, 0.0]])])
        trt = cKDTree(rt["V"]) if len(rt["V"]) else None
        trd = cKDTree(rd["V"]) if len(rd["V"]) else None
        i0 = None
        if trt is not None:
            dd, ii = trt.query([0.0, 0.0])
            i0 = int(ii) if dd < 12 else None
        scene = scene_sd_features(sv, i0)
        s_ego = rt["s"][i0] if i0 is not None else 0.0
        for k in range(n):
            if ln[k] < 1:
                continue
            q = pts[k]
            tq = np.gradient(q, axis=0)
            tq /= np.linalg.norm(tq, axis=1, keepdims=True) + 1e-9
            for m in range(KPT):
                f = np.zeros(9, np.float32)
                f[0] = 1.0  # distances saturate at 15 m
                f[4] = 1.0
                if trt is not None:
                    d, c = _nearest_dir(rt["V"], rt["T"], trt, q[m], tq[m], 15.0)
                    if c >= 0:
                        nl = np.array([-rt["T"][c][1], rt["T"][c][0]])
                        f[0] = d / 15.0
                        f[1] = float((q[m] - rt["V"][c]) @ nl) / LANE_W / 2.0
                        f[2] = float(rt["T"][c] @ tq[m])
                        f[3] = np.clip((rt["s"][c] - s_ego) / 50.0, -2, 2)
                        f[8] = rt["n"][c] / 4.0
                if trd is not None:
                    d, c = _nearest_dir(rd["V"], rd["T"], trd, q[m], tq[m], 15.0)
                    if c >= 0:
                        nl = np.array([-rd["T"][c][1], rd["T"][c][0]])
                        f[4] = d / 15.0
                        f[5] = rd["n"][c] / 4.0
                        f[6] = float((q[m] - rd["V"][c]) @ nl) / LANE_W
                        f[7] = float(rd["T"][c] @ tq[m])
                sdf[k, m * 9 : (m + 1) * 9] = f
            near = sdf[k, 0 : 9 * KPT : 9]
            sdf[k, 9 * KPT + 0] = float((near < 0.3).mean())  # share of points within 4.5 m of the route
            sdf[k, 9 * KPT + 1] = float(near.min())
            sdf[k, 9 * KPT + 2] = float((sdf[k, 4 : 9 * KPT : 9] < 0.3).mean())
            sdf[k, 9 * KPT + 3] = float(len(rt["V"]) > 0)
    return dict(node=np.concatenate([geo, sdf], 1), rel=rel, scene=scene, pts=pts.astype(np.float32), length=ln.astype(np.float32))


# ------------------------------------------------------------------ labels and metrics


def piece_labels(g, lane_pts, route_pts):
    """per piece: share of its 1 m vertices within 1 m of a GT lane point (true) / within 1.5 m of the GT route points (route)."""
    n = len(g["pts"])
    tr_ = np.zeros(n, np.float32)
    rt_ = np.zeros(n, np.float32)
    tl = cKDTree(lane_pts) if lane_pts is not None and len(lane_pts) else None
    trp = cKDTree(route_pts) if route_pts is not None and len(route_pts) else None
    for k, p in enumerate(g["pts"]):
        q = AL.resample(p, 1.0)
        if len(q) < 2:
            continue
        if tl is not None:
            tr_[k] = float((tl.query(q)[0] <= 1.0).mean())
        if trp is not None:
            rt_[k] = float((trp.query(q)[0] <= 1.5).mean())
    return tr_, rt_


def piece_points(g, sel, ds=1.0):
    P = [AL.resample(g["pts"][k], ds) for k in np.nonzero(sel)[0] if len(g["pts"][k]) >= 2]
    P = [q for q in P if len(q) >= 2]
    return np.concatenate(P) if P else np.zeros((0, 2))


def route_success(P, gtr):
    """coverage / precision of the route points P at 1.5 m against the GT route points; success = both >= 0.8 (as E59 / E77 / E78)."""
    if gtr is None or len(gtr) == 0:
        return None
    if len(P) == 0:
        return dict(succ=0, cov=0.0, prec=0.0, n_gt=len(gtr), n_pr=0)
    c = float((cKDTree(P).query(gtr)[0] <= 1.5).mean())
    p = float((cKDTree(gtr).query(P)[0] <= 1.5).mean())
    return dict(succ=int(c >= 0.8 and p >= 0.8), cov=c, prec=p, n_gt=len(gtr), n_pr=len(P))


def lane_pr(P, gt_pts, tol=1.0):
    """point precision / recall at tol of the kept lane points against GT lane points (counts)."""
    if len(P) == 0:
        return dict(tp_p=0, n_p=0, tp_g=0, n_g=len(gt_pts))
    tp_p = int((cKDTree(gt_pts).query(P)[0] <= tol).sum()) if len(gt_pts) else 0
    tp_g = int((cKDTree(P).query(gt_pts)[0] <= tol).sum()) if len(gt_pts) else 0
    return dict(tp_p=tp_p, n_p=len(P), tp_g=tp_g, n_g=len(gt_pts))
