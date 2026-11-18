import hashlib
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import lane_gt_data as LD
import lane_tok_data as LT
import lane_tokcache as LC
import sd_lib as SD

c = LC.TokCache(os.path.join(R, "runs/e365_cache/be292_a_0_K512_9ad1d4ce30_v1/val"))
ds = LT.TokData("val", res=2.0, feats=False)
lane_gt, _ = LD.lane_store()
RAD, COS = 1.5, 0.5


def sd_samples(g, rng, degrade, shift=None, yaw=0.0):
    sd = SD.build_sd(dict(pts=[np.asarray(p["pts"], float) for p in g["pieces"]]), [])
    if degrade:
        sd = SD.degrade_sd(sd, rng, drop=0.1)
    P, T, C = [], [], []
    for poly, cnt in sd["roads"]:
        poly = np.asarray(poly, float)
        if len(poly) < 2:
            continue
        if degrade:
            cy, sy = np.cos(yaw), np.sin(yaw)
            poly = poly @ np.array([[cy, -sy], [sy, cy]]).T + shift
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        if s[-1] < 1e-6:
            continue
        t = np.linspace(0, s[-1], max(int(np.ceil(s[-1] / 2.0)), 1) + 1)
        q = np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], 1)
        tg = np.gradient(q, axis=0)
        tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
        P.append(q)
        T.append(tg)
        C.append(np.interp(t, s, np.asarray(cnt, float)))
    if not P:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    return np.concatenate(P), np.concatenate(T), np.concatenate(C)


def maps_of(i, degrade):
    g = lane_gt.load(c.tokens[i])
    rng = np.random.RandomState(int(hashlib.md5(c.tokens[i].encode()).hexdigest()[:8], 16))
    ang = rng.uniform(0, 2 * np.pi)
    sh = rng.uniform(1, 2) * np.array([np.cos(ang), np.sin(ang)])
    return sd_samples(g, rng, degrade, sh, np.deg2rad(rng.randn()))


N = len(c)
cls = ("novel", "dup", "false", "deliv_true", "deliv_false")
res = {(mode, k): [] for mode in ("exact", "error", "null_error") for k in cls}
bins = []
nsd = []
for i in range(N):
    t = ds.table(i)
    gpos, gdir = t["pos"], t["dirv"]
    if len(gpos) == 0:
        continue
    pos, dirv, valid, ep = (np.array(c.f[k][i], float) if k != "valid" else np.array(c.f[k][i]) for k in ("pos", "dirv", "valid", "e_p"))
    deliv = valid & (ep >= 0.5)
    weak = valid & ~deliv
    near = (np.linalg.norm(pos[:, None] - gpos[None], axis=2) <= RAD) & (dirv @ gdir.T >= COS)
    miss = ~((near & deliv[:, None]).any(0))
    sel = dict(novel=weak & (near & miss[None]).any(1), dup=weak & near.any(1) & ~(near & miss[None]).any(1), false=weak & ~near.any(1), deliv_true=deliv & near.any(1), deliv_false=deliv & ~near.any(1))
    j = (i + 97) % N
    maps = dict(exact=maps_of(i, False), error=maps_of(i, True), null_error=maps_of(j, True))
    nsd.append(len(maps["exact"][0]))
    for mode, (P, T, C) in maps.items():
        if len(P) == 0:
            sup = np.zeros(len(pos), bool)
            dist = np.full(len(pos), 99.0)
        else:
            d, jj = cKDTree(P).query(pos)
            dist = d
            sup = (d <= 3.0) & (np.abs((dirv * T[jj]).sum(1)) >= 0.8)
        for k in cls:
            res[(mode, k)].append(np.stack([dist[sel[k]], sup[sel[k]]], 1))
        if mode == "error":
            bins.append(np.stack([ep[weak], sup[weak], near.any(1)[weak], (near & miss[None]).any(1)[weak]], 1))
print("VAL seed 0, road-level SD map (sd_lib.build_sd; error = degrade_sd drop 0.1 + shift 1-2 m + yaw 1 degree), samples per frame %.0f" % np.mean(nsd))
print("class | n | within 3 m and aligned (|cos| >= 0.8): exact / error / another scene's error map | median distance exact / error")
for k in cls:
    a = {m: np.concatenate(res[(m, k)]) for m in ("exact", "error", "null_error")}
    print("  %-12s %7d | %.3f / %.3f / %.3f | %.2f / %.2f" % (k, len(a["exact"]), a["exact"][:, 1].mean(), a["error"][:, 1].mean(), a["null_error"][:, 1].mean(), np.median(a["exact"][:, 0]), np.median(a["error"][:, 0])))
B = np.concatenate(bins)
print("weak tokens by existence bin, error map: p range | n | share near a GT node | map-supported if near GT / if not | novel share (supported / unsupported)")
for lo, hi in ((0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)):
    m = (B[:, 0] >= lo) & (B[:, 0] < hi)
    t_ = m & (B[:, 2] > 0.5)
    f_ = m & (B[:, 2] < 0.5)
    s_ = m & (B[:, 1] > 0.5)
    u_ = m & (B[:, 1] < 0.5)
    print("  %.2f - %.2f | %6d | %.3f | %.3f / %.3f | %.4f / %.4f" % (lo, hi, m.sum(), t_.sum() / m.sum(), B[t_, 1].mean(), B[f_, 1].mean(), B[s_, 3].mean(), B[u_, 3].mean()))

# full precision of the same numbers (for the digit-for-digit comparison with `lane_map_data.py potential`)
import json
out = dict(classes={}, by_bin=[], samples_per_frame=float(np.mean(nsd)))
for k in cls:
    a = {m: np.concatenate(res[(m, k)]) for m in ("exact", "error", "null_error")}
    out["classes"][k] = dict(n=len(a["exact"]), supported_exact=float(a["exact"][:, 1].mean()), supported_error=float(a["error"][:, 1].mean()), supported_other_scene=float(a["null_error"][:, 1].mean()),
                             median_dist_exact=float(np.median(a["exact"][:, 0])), median_dist_error=float(np.median(a["error"][:, 0])))
for lo, hi in ((0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)):
    m = (B[:, 0] >= lo) & (B[:, 0] < hi)
    t_, f_, s_, u_ = m & (B[:, 2] > 0.5), m & (B[:, 2] < 0.5), m & (B[:, 1] > 0.5), m & (B[:, 1] < 0.5)
    out["by_bin"].append(dict(lo=lo, hi=hi, n=int(m.sum()), true_share=float(t_.sum() / m.sum()), supported_if_true=float(B[t_, 1].mean()), supported_if_false=float(B[f_, 1].mean()),
                              novel_share_supported=float(B[s_, 3].mean()), novel_share_unsupported=float(B[u_, 3].mean())))
json.dump(out, open(os.path.join(R, "temp/_session_scratch_d113/work3/step0_full.json"), "w"), indent=1)
