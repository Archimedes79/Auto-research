"""Loop 8: SEGMENT-level representation of the perceived lane graph (sparse graph with short edges).
Finding that motivated it (diag): with whole traced pieces as the unit the oracle route success is 0.45 (a piece follows the route for a while and then goes straight on),
with ~6 m segments it is 0.98 -> the assignment has to act on short edges between nodes every ~6 m.
Token = one segment (about 6 m, 1 m vertices) of a traced piece. Relations: successor / predecessor (along the piece and across the graph edges), left / right parallel neighbour, same parent piece.
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc_lib as AL  # noqa: E402
import p8_lib as L  # noqa: E402

SEG_LS = 6.0
KS = 4
SEG_GEO = 20
SEG_SD = 9 * KS + 4
NSEG_MAX = 128
LSP = 8  # stored vertices per segment (<= SEG_LS + 2)


def segments_of(g, pprob, pdir, ls=SEG_LS):
    segs = []
    first_of, last_of = {}, {}
    out_n = {}
    for i, j in g["edges"]:
        out_n[i] = out_n.get(i, 0) + 1
    for k, p in enumerate(g["pts"]):
        q = AL.resample(p, 1.0)
        m = len(q)
        if m < 2:
            continue
        pp = np.asarray(pprob[k], np.float32)
        pd = np.asarray(pdir[k], np.float32)
        if len(pp) != m:
            x = np.linspace(0, 1, max(len(pp), 2))
            xi = np.linspace(0, 1, m)
            pp = np.interp(xi, x, pp if len(pp) > 1 else np.repeat(pp, 2)) if len(pp) else np.zeros(m, np.float32)
            pd = np.interp(xi, x, pd if len(pd) > 1 else np.repeat(pd, 2)) if len(pd) else np.zeros(m, np.float32)
        ns = max(1, int(round((m - 1) / ls)))
        b = np.linspace(0, m - 1, ns + 1).round().astype(int)
        for a in range(ns):
            lo, hi = b[a], b[a + 1]
            if hi - lo < 1:
                continue
            if a == 0:
                first_of[k] = len(segs)
            last_of[k] = len(segs)
            segs.append(dict(piece=k, a=a, nseg=ns, q=q[lo : hi + 1], pm=float(pp[lo : hi + 1].mean()), pmin=float(pp[lo : hi + 1].min()), dm=float(pd[lo : hi + 1].mean()), plen=float(m - 1),
                             ppm=float(pp.mean()), nout=out_n.get(k, 0)))
    return segs, first_of, last_of


def resample_k(q, k=KS):
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, max(s[-1], 1e-6), k)
    return np.stack([np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])], 1)


def sd_point_feat(rt, rd, trt, trd, s_ego, q, tq):
    f = np.zeros(9, np.float32)
    f[0] = 1.0
    f[4] = 1.0
    if trt is not None:
        d, c = L._nearest_dir(rt["V"], rt["T"], trt, q, tq, 15.0)
        if c >= 0:
            nl = np.array([-rt["T"][c][1], rt["T"][c][0]])
            f[0] = d / 15.0
            f[1] = float((q - rt["V"][c]) @ nl) / L.LANE_W / 2.0
            f[2] = float(rt["T"][c] @ tq)
            f[3] = np.clip((rt["s"][c] - s_ego) / 50.0, -2, 2)
            f[8] = rt["n"][c] / 4.0
    if trd is not None:
        d, c = L._nearest_dir(rd["V"], rd["T"], trd, q, tq, 15.0)
        if c >= 0:
            nl = np.array([-rd["T"][c][1], rd["T"][c][0]])
            f[4] = d / 15.0
            f[5] = rd["n"][c] / 4.0
            f[6] = float((q - rd["V"][c]) @ nl) / L.LANE_W
            f[7] = float(rd["T"][c] @ tq)
    return f


def row_features_seg(g, pprob, pdir, sdm_ego, ls=SEG_LS):
    segs, first_of, last_of = segments_of(g, pprob, pdir, ls)
    n = min(len(segs), NSEG_MAX)
    if len(segs) > NSEG_MAX:  # keep the most reliable segments
        order = np.argsort([-s["pm"] for s in segs])[:NSEG_MAX]
        keep = np.sort(order)
        remap = {int(o): i for i, o in enumerate(keep)}
        segs = [segs[i] for i in keep]
    else:
        remap = {i: i for i in range(len(segs))}
    node = np.zeros((n, SEG_GEO + SEG_SD), np.float32)
    pts4 = np.zeros((n, KS, 2), np.float32)
    spts = np.zeros((n, LSP, 2), np.float32)
    snp = np.zeros(n, np.int16)
    length = np.zeros(n, np.float32)
    piece = np.zeros(n, np.int32)
    rel = np.zeros((n, n, 5), np.uint8)
    sv = None
    trt = trd = None
    s_ego = 0.0
    scene = np.zeros(L.N_SCENE, np.float32)
    if sdm_ego is not None:
        sv = L.sd_vertices(sdm_ego)
        rt, rd = sv["route"], sv["roads"]
        trt = cKDTree(rt["V"]) if len(rt["V"]) else None
        trd = cKDTree(rd["V"]) if len(rd["V"]) else None
        i0 = None
        if trt is not None:
            dd, ii = trt.query([0.0, 0.0])
            i0 = int(ii) if dd < 12 else None
        scene = L.scene_sd_features(sv, i0)
        s_ego = rt["s"][i0] if i0 is not None else 0.0
    for i, s in enumerate(segs):
        q = s["q"]
        m = min(len(q), LSP)
        spts[i, :m] = q[:m]
        snp[i] = m
        length[i] = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
        piece[i] = s["piece"]
        p4 = resample_k(q)
        pts4[i] = p4
        t = np.gradient(p4, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        t0, t1 = t[0], t[-1]
        turn = float(np.arctan2(t0[0] * t1[1] - t0[1] * t1[0], t0 @ t1))
        d_ego = float(np.linalg.norm(p4, axis=1).min())
        node[i, :SEG_GEO] = [min(length[i] / 10.0, 2.0), s["pm"], s["pmin"], s["dm"], p4[0, 0] / 50, p4[0, 1] / 50, p4[-1, 0] / 50, p4[-1, 1] / 50, t0[0], t0[1], t1[0], t1[1], turn,
                             min(s["plen"] / 50.0, 2.0), s["a"] / max(s["nseg"], 1), float(s["a"] == 0), float(s["a"] == s["nseg"] - 1), min(s["nout"], 3) / 3.0, min(d_ego / 20.0, 2.0), s["ppm"]]
        if sv is not None:
            for mm in range(KS):
                node[i, SEG_GEO + mm * 9 : SEG_GEO + (mm + 1) * 9] = sd_point_feat(rt, rd, trt, trd, s_ego, p4[mm], t[mm])
            near = node[i, SEG_GEO : SEG_GEO + 9 * KS : 9]
            b0 = SEG_GEO + 9 * KS
            node[i, b0] = float((near < 0.3).mean())
            node[i, b0 + 1] = float(near.min())
            node[i, b0 + 2] = float((node[i, SEG_GEO + 4 : SEG_GEO + 9 * KS : 9] < 0.3).mean())
            node[i, b0 + 3] = float(len(rt["V"]) > 0)
    # relations
    for i, s in enumerate(segs):
        if i + 1 < n and segs[i + 1]["piece"] == s["piece"] and segs[i + 1]["a"] == s["a"] + 1:
            rel[i, i + 1, 0] = 1
            rel[i + 1, i, 1] = 1
    for a_, b_ in g["edges"]:
        if a_ in last_of and b_ in first_of:
            i, j = remap.get(last_of[a_]), remap.get(first_of[b_])
            if i is not None and j is not None:
                rel[i, j, 0] = 1
                rel[j, i, 1] = 1
    pc = np.array([s["piece"] for s in segs])
    for i in range(n):
        rel[i, pc == pc[i], 4] = 1
        rel[i, i, 4] = 0
    if n > 1:
        mid = pts4[:, KS // 2]
        tree = cKDTree(mid)
        tm = np.gradient(pts4, axis=1)
        tm /= np.linalg.norm(tm, axis=2, keepdims=True) + 1e-9
        for i in range(n):
            for j in tree.query_ball_point(mid[i], 9.0):
                if j == i or pc[j] == pc[i]:
                    continue
                dj = np.linalg.norm(pts4[j] - mid[i], axis=1)
                c = int(np.argmin(dj))
                if dj[c] > 5.5 or dj[c] < 1.0:
                    continue
                if float(tm[j, c] @ tm[i, KS // 2]) < 0.85:
                    continue
                v = pts4[j, c] - mid[i]
                side = tm[i, KS // 2][0] * v[1] - tm[i, KS // 2][1] * v[0]
                rel[i, j, 2 if side > 0 else 3] = 1
    return dict(node=node, rel=rel, scene=scene, pts=pts4, spts=spts, snp=snp, length=length, piece=piece, n=n)


def seg_labels(spts, snp, lane_pts, route_pts):
    n = len(snp)
    tl = np.zeros(n, np.float32)
    rl = np.zeros(n, np.float32)
    tlt = cKDTree(lane_pts) if lane_pts is not None and len(lane_pts) else None
    trp = cKDTree(route_pts) if route_pts is not None and len(route_pts) else None
    for i in range(n):
        q = spts[i, : snp[i]]
        if len(q) < 2:
            continue
        if tlt is not None:
            tl[i] = float((tlt.query(q)[0] <= 1.0).mean())
        if trp is not None:
            rl[i] = float((trp.query(q)[0] <= 1.5).mean())
    return tl, rl
