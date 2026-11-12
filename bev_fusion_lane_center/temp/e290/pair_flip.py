"""E290 (scratch): pair flip rate between consecutive VAL keyframes of a stored pair head dump, and the effect of averaging with the ego-warped previous logits.
usage: pair_flip.py <run e.g. e284_h2_s0> [weight 0.5]"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, "../../tools/viz")
sys.path.insert(0, "../../training/grid")
import jgt as J
import qgraph_view as QV

run = sys.argv[1]
W = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
R = os.environ["BFLC_ROOT"]
frames = {f["token"]: f for f in pickle.load(open(os.path.join(R, "runs", run, "ends_val.pkl"), "rb"))}
infos = QV._infos()


def lane_frame(tok):
    p = infos[tok]["cam_infos"]["CAM_FRONT"]["ego_pose"]
    w, x, y, z = p["rotation"]
    Re = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z)]])         # the 2 x 2 part of the rotation matrix (yaw)
    A = Re @ np.array([[0.0, 1.0], [-1.0, 0.0]])
    return A, np.asarray(p["translation"][:2], float)


def warp(E, ta, tb):
    """ends (n, 4) x, y, tx, ty of frame ta in the lane frame of frame tb"""
    Aa, ca = lane_frame(ta)
    Ab, cb = lane_frame(tb)
    pos = ((E[:, :2] @ Aa.T + ca) - cb) @ Ab
    d = (E[:, 2:] @ Aa.T) @ Ab
    return np.concatenate([pos, d], 1)


def match(kp, kd, valid, G, pos_tol=4.0, ang=30.0):
    """for every GT end the index of the nearest valid keypoint within pos_tol and ang degrees, -1 none"""
    out = -np.ones(len(G), int)
    for g, e in enumerate(G):
        d = np.linalg.norm(kp - e[:2], axis=1)
        a = np.degrees(np.arccos(np.clip(kd @ e[2:], -1, 1)))
        ok = valid & (d <= pos_tol) & (a <= ang)
        if ok.any():
            out[g] = int(np.argmin(np.where(ok, d, 1e9)))
    return out


def frame_pairs(tok):
    """GT ends and pairs of the frame, the head's logit for every GT pair (-10 when not available)"""
    jg = J.junction_gt(QV.load_gt(tok))
    if jg is None or not len(jg["pairs"]) or tok not in frames:
        return None
    f = frames[tok]
    K = f["logit"].shape[0]
    ia = match(f["kp_pos"][:K], f["kp_dir"][:K], f["valid"][0], jg["ent"])
    ib = match(f["kp_pos"][K:], f["kp_dir"][K:], f["valid"][1], jg["ext"])
    z = {}
    for a, b in jg["pairs"][:, :2].astype(int):
        z[(a, b)] = float(f["logit"][ia[a], ib[b]]) if ia[a] >= 0 and ib[b] >= 0 and f["allowed"][ia[a], ib[b]] else -10.0
    return dict(ent=jg["ent"], ext=jg["ext"], z=z, pairs=[tuple(p) for p in jg["pairs"][:, :2].astype(int)])


def counterpart(Ecur, Eprev_warped, tol=3.0, ang=30.0):
    """index in prev of every end of cur (-1 none)"""
    return match(Eprev_warped[:, :2], Eprev_warped[:, 2:], np.ones(len(Eprev_warped), bool), Ecur, tol, ang)


stat = dict(frames=0, pairs_both=0, flip_c=0, flip_w=0, deliv_c_both=0, deliv_c_none=0, deliv_w_both=0, deliv_w_none=0, shuffled_flip=0)
rng = np.random.RandomState(0)
for sid, (name, toks) in QV.val_scenes().items():
    cache = {}
    for t in toks:
        cache[t] = frame_pairs(t)
    # arm w: the logit of a pair at frame t = (1 - W) z(t) + W z(t-1 warped), when the pair has a counterpart in the previous frame
    wz = {}
    for i, t in enumerate(toks):
        cur = cache[t]
        if cur is None:
            continue
        zz = dict(cur["z"])
        if i > 0 and cache[toks[i - 1]] is not None:
            prev = cache[toks[i - 1]]
            ea = counterpart(cur["ent"], warp(prev["ent"], toks[i - 1], t))
            eb = counterpart(cur["ext"], warp(prev["ext"], toks[i - 1], t))
            for (a, b) in cur["pairs"]:
                if ea[a] >= 0 and eb[b] >= 0 and (ea[a], eb[b]) in prev["z"]:
                    zz[(a, b)] = (1 - W) * cur["z"][(a, b)] + W * prev["z"][(ea[a], eb[b])]
        wz[t] = zz
    for i in range(1, len(toks)):
        t0, t1 = toks[i - 1], toks[i]
        c0, c1 = cache[t0], cache[t1]
        if c0 is None or c1 is None:
            continue
        stat["frames"] += 1
        ea = counterpart(c1["ent"], warp(c0["ent"], t0, t1))
        eb = counterpart(c1["ext"], warp(c0["ext"], t0, t1))
        for (a, b) in c1["pairs"]:
            if ea[a] >= 0 and eb[b] >= 0 and (ea[a], eb[b]) in c0["z"]:
                a0, b0 = ea[a], eb[b]
                stat["pairs_both"] += 1
                d0, d1 = c0["z"][(a0, b0)] >= 0.0, c1["z"][(a, b)] >= 0.0                       # logit >= 0 = p >= 0.5
                stat["flip_c"] += int(d0 != d1)
                stat["deliv_c_both"] += int(d0 and d1)
                stat["deliv_c_none"] += int(not d0 and not d1)
                w0, w1 = wz[t0][(a0, b0)] >= 0.0, wz[t1][(a, b)] >= 0.0
                stat["flip_w"] += int(w0 != w1)
                stat["deliv_w_both"] += int(w0 and w1)
                stat["deliv_w_none"] += int(not w0 and not w1)
n = max(stat["pairs_both"], 1)
res = dict(run=run, weight=W, **stat, flip_rate_c=stat["flip_c"] / n, flip_rate_w=stat["flip_w"] / n,
           relative_change=(stat["flip_w"] - stat["flip_c"]) / max(stat["flip_c"], 1), delivered_in_neither_c=stat["deliv_c_none"] / n, delivered_in_neither_w=stat["deliv_w_none"] / n)
print(json.dumps(res, indent=1))
os.makedirs(os.path.join(R, "runs", "e290"), exist_ok=True)
json.dump(res, open(os.path.join(R, "runs", "e290", f"pair_flip_{run}.json"), "w"), indent=1)
