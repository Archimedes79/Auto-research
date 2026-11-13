"""Step 0 of E299 (time for the junction head): how much can ONE ego-warped previous keyframe add to the pair head of E312? CPU, VAL, no training, no GPU.

    wsl bash tools/env/_py14.sh jtime.py ceiling --seed 0 [--run e312_p_s0]            the ceiling: the TRUE pairs delivered at t-1 (p >= 0.1) and not delivered at t, split into LOGIT-ONLY flips (both ends have a counterpart keypoint at t, the pair
                                                                                        probability stays below 0.1) and END-MISS flips (an end has no counterpart keypoint at t: a re-scoring layer cannot fix those)
    wsl bash tools/env/_py14.sh jtime.py replay --seed 0 [--w 0.5] [--run e312_p_s0]   the zero-parameter replay: the pair logit of a pair with a counterpart pair at t-1 becomes (1 - w) x logit(t) + w x logit(t-1); pair precision at the recall of the
                                                                                        original (logit 0 and -2.2, `jpair_pr` definition) and the snapped topo F1 of the E312 read-out with the replayed logits
    wsl bash tools/env/_py14.sh jtime.py selftest

The association of keypoints across the two frames uses the PREDICTED keypoints only (the keypoints of t-1 warped by the relative ego pose into the frame of t; mutual nearest neighbours within 3 m and 30 degrees); the GT enters only the truth label of a pair (`jwiring.match` and the GT pairs of
the frame) in the statistics and the precision, never the replayed logit. Frames are taken in time order per scene (`qgraph_view.val_scenes`, poses from `_infos`, the CAM_FRONT ego pose in the lane frame as in `temp/e290/pair_flip.py`). The weight is PRE-REGISTERED at 0.5 in the card
(E299 section 1); any other weight is an in-sample sweep on VAL and is labelled so in the output. Needs the GT ends of the VAL frames (`jwiring.py gt-ends`). Writes temp/jtime/<run>_<mode>.json.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training", "training/grid", "tools/viz", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))

THR = 0.1                      # the delivered read-out: p >= 0.1 (logit -2.197)
TOL, ANG = 3.0, 30.0           # the association of keypoints across two frames


def lane_frame(info):
    """(A, c) of a keyframe: lane-frame coordinates p_lane = (R p_ego_world ...) as in temp/e290/pair_flip.py: the 2 x 2 yaw rotation of the CAM_FRONT ego pose composed with the lane-frame axes, and the ego translation"""
    p = info["cam_infos"]["CAM_FRONT"]["ego_pose"]
    w, x, y, z = p["rotation"]
    Re = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z)]])
    return Re @ np.array([[0.0, 1.0], [-1.0, 0.0]]), np.asarray(p["translation"][:2], float)


def warp(E, fa, fb):
    """ends E (n, 4: x, y, tx, ty) of the lane frame fa = lane_frame(info a) in the lane frame fb"""
    (Aa, ca), (Ab, cb) = fa, fb
    pos = ((E[:, :2] @ Aa.T + ca) - cb) @ Ab
    d = (E[:, 2:] @ Aa.T) @ Ab
    return np.concatenate([pos, d], 1)


def counterpart(prev_w, cur, valid_prev, valid_cur, tol=TOL, ang=ANG):
    """prev_w (n, 4) keypoints of t-1 in the frame of t, cur (m, 4) keypoints of t -> (n,) index of the counterpart in cur (mutual nearest within tol m and ang degrees), -1 none"""
    n, m = len(prev_w), len(cur)
    out = -np.ones(n, int)
    if not n or not m:
        return out
    d = np.linalg.norm(prev_w[:, None, :2] - cur[None, :, :2], axis=-1)
    a = np.degrees(np.arccos(np.clip((prev_w[:, None, 2:] * cur[None, :, 2:]).sum(-1), -1, 1)))
    ok = (d <= tol) & (a <= ang) & valid_prev[:, None] & valid_cur[None, :]
    d = np.where(ok, d, 1e9)
    i_best = d.argmin(1)
    j_best = d.argmin(0)
    for i in range(n):
        if ok[i, i_best[i]] and j_best[i_best[i]] == i:
            out[i] = i_best[i]
    return out


def _prob(e):
    return 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]


def _truth(e, g):
    """(K, K) bool: the candidate pair is a TRUE pair (both ends match the two ends of one GT pair, `jwiring.match`) - labels only"""
    import jwiring as JW
    k = e["kp_pos"].shape[0] // 2
    valid = np.asarray(e["valid"], bool)
    m_in = JW.match(e["kp_pos"][:k], e["kp_dir"][:k], valid[0], g["ent"], g["ent_m"])
    m_out = JW.match(e["kp_pos"][k:], e["kp_dir"][k:], valid[1], g["ext"], g["ext_m"])
    gtp = {(int(x), int(y)) for x, y in g["pairs"] if x >= 0}
    T = np.zeros((k, k), bool)
    for i in range(k):
        for j in range(k):
            if m_in[i] >= 0 and m_out[j] >= 0 and (int(m_in[i]), int(m_out[j])) in gtp:
                T[i, j] = True
    return T


def _setup(a):
    import qgraph_eval as QE
    import qgraph_view as QV
    import jwiring as JW
    run = a.run or f"e312_p_s{a.seed}"
    ends = {f["token"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", run, "ends_val.pkl"), "rb"))}
    gt = pickle.load(open(JW.GT_PATH, "rb"))
    frames = {f.token: f for f in QE.val_frames()}
    scenes = QV.val_scenes()
    infos = QV._infos()
    return run, ends, gt, frames, scenes, infos


def _pairs_of_frames(scenes, ends):
    """[(t0, t1)] consecutive keyframes of one scene that both have a dump entry"""
    out = []
    for _, (_, toks) in scenes.items():
        for k in range(1, len(toks)):
            if toks[k - 1] in ends and toks[k] in ends:
                out.append((toks[k - 1], toks[k]))
    return out


def _associate(e0, e1, f0, f1):
    """counterparts of the entry and the exit keypoints of t-1 in t -> (cp_in (K,), cp_out (K,))"""
    k = e0["kp_pos"].shape[0] // 2
    v0, v1 = np.asarray(e0["valid"], bool), np.asarray(e1["valid"], bool)
    E0 = np.concatenate([e0["kp_pos"], e0["kp_dir"]], 1)
    E1 = np.concatenate([e1["kp_pos"], e1["kp_dir"]], 1)
    W0 = warp(E0, f0, f1)
    return counterpart(W0[:k], E1[:k], v0[0], v1[0]), counterpart(W0[k:], E1[k:], v0[1], v1[1])


def ceiling(a):
    run, ends, gt, frames, scenes, infos = _setup(a)
    st = dict(true_pairs_t0_delivered=0, delivered_both=0, logit_only=0, end_miss=0, flips_total_either=0, flips_one_only=0)
    n = 0
    for t0, t1 in _pairs_of_frames(scenes, ends):
        e0, e1 = ends[t0], ends[t1]
        g0, g1 = gt[frames[t0].row], gt[frames[t1].row]
        p0, p1 = _prob(e0), _prob(e1)
        T0, T1 = _truth(e0, g0), _truth(e1, g1)
        cpi, cpo = _associate(e0, e1, lane_frame(infos[t0]), lane_frame(infos[t1]))
        n += 1
        for i, j in zip(*np.nonzero((p0 >= THR) & T0)):
            st["true_pairs_t0_delivered"] += 1
            ci, cj = cpi[i], cpo[j]
            if ci < 0 or cj < 0:
                st["end_miss"] += 1
            elif p1[ci, cj] >= THR:
                st["delivered_both"] += 1
            else:
                st["logit_only"] += 1
        # the flip rate of E290: true pairs delivered in exactly one of two consecutive frames (counterparts by the keypoints)
        for i, j in zip(*np.nonzero(T0)):
            ci, cj = cpi[i], cpo[j]
            d0 = p0[i, j] >= THR
            d1 = bool(ci >= 0 and cj >= 0 and p1[ci, cj] >= THR and T1[ci, cj])
            if d0 or d1:
                st["flips_total_either"] += 1
                st["flips_one_only"] += int(d0 != d1)
    missed = st["end_miss"] + st["logit_only"]
    out = dict(run=run, consecutive_pairs_of_frames=n, **st, logit_only_share_of_missed=st["logit_only"] / max(missed, 1), flip_rate=st["flips_one_only"] / max(st["flips_total_either"], 1))
    print(json.dumps(out, indent=1))
    os.makedirs(os.path.join(ROOT, "temp", "jtime"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "temp", "jtime", f"{run}_ceiling.json"), "w"), indent=1)


def replay(a):
    import jpair_pr as JP
    import qgraph_eval as QE
    import qgraph_readout as QR
    from qends_eval import add_connectors, make_pairs
    run, ends, gt, frames, scenes, infos = _setup(a)
    w = a.w
    prev_of = {t1: t0 for t0, t1 in _pairs_of_frames(scenes, ends)}
    dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{a.seed}.pkl"), "rb"))
    order = [f.token for f in QE.val_frames()]
    assert [d["row"] for d in dump] == [frames[t].row for t in order], "the full dump is not in the order of the VAL frames"
    z_new = {}
    for t1, t0 in prev_of.items():                                                                                        # the replayed logits of every frame that has a previous frame
        e0, e1 = ends[t0], ends[t1]
        cpi, cpo = _associate(e0, e1, lane_frame(infos[t0]), lane_frame(infos[t1]))
        z = e1["logit"].astype(float).copy()
        al0 = e0["allowed"].astype(bool)
        for i in range(len(cpi)):
            for j in range(len(cpo)):
                ci, cj = cpi[i], cpo[j]
                if ci >= 0 and cj >= 0 and al0[i, j] and e1["allowed"][ci, cj]:
                    z[ci, cj] = (1 - w) * z[ci, cj] + w * float(e0["logit"][i, j])
        z_new[t1] = z
    # pair precision at the recall of the original operating point (jpair_pr: precision at the smallest number of delivered pairs whose recall reaches the control's)
    labels, s_old, s_new, n_gt = [], [], [], 0
    for t in order:
        e = ends[t]
        T = _truth(e, gt[frames[t].row])
        al = e["allowed"].astype(bool)
        labels.append(T[al])
        s_old.append(e["logit"][al].astype(float))
        s_new.append(z_new.get(t, e["logit"].astype(float))[al])
        n_gt += int((gt[frames[t].row]["pairs"][:, 0] >= 0).sum())
    labels, s_old, s_new = np.concatenate(labels), np.concatenate(s_old), np.concatenate(s_new)
    out = dict(run=run, w=w, label=("PRE-REGISTERED weight" if w == 0.5 else "IN-SAMPLE sweep on VAL"), n_gt=n_gt)
    for name, thr in (("logit_0", 0.0), ("logit_m2.2", -2.197)):
        sel = s_old >= thr
        rec = float(labels[sel].sum() / max(n_gt, 1))
        prec_old = float(labels[sel].mean())
        prec_new = JP.precision_at_recall_exact(s_new, labels, n_gt, rec)
        out[name] = dict(recall_of_original=rec, precision_original=prec_old, precision_replayed_at_that_recall=prec_new, gain=None if prec_new is None else prec_new - prec_old)
    # the snapped topo F1 of the E312 read-out with the original and the replayed logits (connectors at p >= 0.1)
    spec = QR.parse("close:1.5@0.5")
    rec_ = dict(old=[], new=[])
    for b, t in enumerate(order):
        d, f = dump[b], frames[t]
        pr = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                           links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), spec)
        graph, bk = QE.readout(pr, 0.5, 0.5)
        e = ends[t]
        k = e["kp_pos"].shape[0] // 2
        for name, lg in (("old", e["logit"].astype(float)), ("new", z_new.get(t, e["logit"].astype(float)))):
            p = 1.0 / (1.0 + np.exp(-lg)) * e["allowed"]
            ia, ib = np.nonzero(p >= THR)
            pairs = make_pairs(e, ia, ib, k)
            raw, _, _ = add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], np.asarray(pr["links"]["long"], bool), pairs, snap=True)
            rec_[name].append(QE.graph_counts(raw, f))
        f._lane = f._ref = None
    num = {v: QE.graph_numbers(rec_[v]) for v in rec_}
    out["snapped"] = {v: {kk: float(num[v][kk]) for kk in ("topo_F1", "dup_frac", "junction_F1", "F1@1.0", "lanes_found")} for v in num}
    out["snapped_difference"] = {kk: out["snapped"]["new"][kk] - out["snapped"]["old"][kk] for kk in out["snapped"]["old"]}
    print(json.dumps(out, indent=1))
    os.makedirs(os.path.join(ROOT, "temp", "jtime"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "temp", "jtime", f"{run}_replay_w{w}.json"), "w"), indent=1)


def selftest():
    # the same pose: the warp is the identity; a pure translation of the ego moves the ends the other way
    info = lambda yaw, t: {"cam_infos": {"CAM_FRONT": {"ego_pose": {"rotation": [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], "translation": [t[0], t[1], 0.0]}}}}
    f0, f1 = lane_frame(info(0.3, (10.0, 5.0))), lane_frame(info(0.3, (10.0, 5.0)))
    E = np.array([[1.0, 2.0, 0.0, 1.0], [-3.0, 4.0, 1.0, 0.0]])
    assert np.allclose(warp(E, f0, f1), E, atol=1e-9), "identical poses: the warp is the identity"
    fa, fb = lane_frame(info(0.0, (0.0, 0.0))), lane_frame(info(0.7, (2.0, -1.0)))
    Wp = warp(np.array([[0.0, 10.0, 0.0, 1.0]]), fa, fb)
    assert abs(np.linalg.norm(Wp[0, 2:]) - 1.0) < 1e-9 and np.linalg.norm(Wp[0, :2] - np.array([0.0, 10.0])) > 0.5, Wp     # a rotated and moved ego: the heading stays a unit vector, the position changes
    # round trip: a -> b -> a is the identity
    back = warp(warp(E, fa, fb), fb, fa)
    assert np.allclose(back, E, atol=1e-9), (back, E)
    # the association: mutual nearest within 3 m and 30 degrees; a keypoint with no partner, a duplicate and a wrong heading
    prev = np.array([[0.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, 1.0], [20.0, 0.0, 0.0, 1.0], [30.0, 0.0, 1.0, 0.0]])
    cur = np.array([[0.5, 0.0, 0.0, 1.0], [10.2, 0.0, 0.0, 1.0], [10.8, 0.0, 0.0, 1.0], [30.0, 0.0, 0.0, 1.0]])
    cp = counterpart(prev, cur, np.ones(4, bool), np.ones(4, bool))
    assert cp.tolist() == [0, 1, -1, -1], cp                                                                  # 0 -> 0; 1 -> 1 (the nearer of two); 2 has no partner within 3 m; 3 has the wrong heading
    # the probability and the truth helpers on a tiny dump
    e = dict(logit=np.array([[0.0, -9.0], [-9.0, 2.0]]), allowed=np.ones((2, 2), bool))
    assert np.allclose(_prob(e)[0, 0], 0.5) and _prob(e)[1, 1] > 0.85
    print("jtime selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("ceiling", "replay"):
        s = sub.add_parser(c)
        s.add_argument("--seed", type=int, required=True)
        s.add_argument("--run", default=None)
        if c == "replay":
            s.add_argument("--w", type=float, default=0.5)
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"ceiling": ceiling, "replay": replay, "selftest": lambda _: selftest()}[a.cmd](a)
