"""temporal_union.py - scratch (post-hoc probe, no verdict): would the single-frame lane_divider maps of EARLIER keyframes, warped into the current ego frame, bring back the lines the network misses?
t20_base_s0 (last checkpoint, EMA weights, CPU, memory EMPTY for every frame = single-frame predictions, dt 0), the previous N keyframes (N = 1..4, 0.5 s apart) of the same scene are warped
into the current frame with the ego poses and combined with the current map; the current frame's GT is the reference for every variant.

    temporal_union.py plan | infer | eval | all        (CPU only; run through rp_py.sh: nice 15, 2 threads, CUDA_VISIBLE_DEVICES empty)
 plan   picks the current frames from the 246-frame subset (frames with >= 4 predecessors in the dataset, preferring those whose predecessors are already cached) within an extra-inference budget
 infer  runs the missing predecessor frames through lane_readout.infer (same model, same code, memory empty); extra frames are stored as ids 1000+ in the same maps folder
 eval   warps, combines, scores (hand count of the metric as in lane_readout), writes temp/overview/temporal_union_result.txt / .json
Poses: the GT cache's meta['T_e2g'] = the ego pose (4x4, ego -> global) of the CAM_FRONT sample_data of the keyframe, which is the pose the GT of that frame is registered to (autogt keyframe_gt:
T_e2g = ego_pose(CAM_FRONT sample_data)); the maps are in that ego frame. cell centre (x, y, z = 0) of the current grid -> global with T_cur -> previous ego frame with inv(T_prev) -> previous cell.
Valid = inside the previous grid AND inside the previous frame's camera (fov) mask of its GT cache (cells the camera cannot see contribute nothing; they are masked, not zero evidence).
Combination rules (fixed before the first look): (a) max of current and valid warped maps; (b) noisy-or 1 - prod(1 - p); (c) mean over the maps that see the cell (the current map always does).
Threshold p > 0.5 as the metric. Warp: nearest cell (primary); bilinear for N = 1 as the second case. Controls (N = 1 and 4): the previous pose shifted 3 m sideways (current ego lateral axis) and the
relative ego motion of ANOTHER scene's pair applied to this pair (a wrong pose of plausible size)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lane_readout as LR  # noqa: E402

RUN = "t20_base_s0"
PLAN = os.path.join(HERE, "tu_plan.json")
BUDGET = 220                  # extra inference frames (the coordinator's cap is about 200)
NMAX = 4
SIGFILE = "_0.5_4_6a7aa7864a.npz"
GTDIR = os.path.join(LR.ROOT, "data", "cache", "gt")


def progress(msg):
    with open(os.path.join(HERE, "night_progress.txt"), "a") as f:
        f.write(f"[{time.strftime('%H:%M')}] JOB 5 {msg}\n")


def scene_chains():
    """scene name -> list of keyframe tokens in time order (nuScenes sample.json next chain)"""
    root = LR.FROOT
    scenes = json.load(open(os.path.join(root, "v1.0-trainval", "scene.json")))
    samples = {s["token"]: s for s in json.load(open(os.path.join(root, "v1.0-trainval", "sample.json")))}
    out = {}
    for sc in scenes:
        chain, t = [], sc["first_sample_token"]
        while t:
            chain.append(t)
            t = samples[t]["next"]
        out[sc["name"]] = chain
    return out


def do_plan():
    frames = LR.load_frames()
    cp = LR.resolve_ckpt(RUN, None)
    tag = LR.tag_of(RUN, cp)
    maps_dir = os.path.join(LR.MAPS_DIR, tag)
    chains = scene_chains()
    cached = {f["token"]: f["i"] for f in frames if os.path.exists(os.path.join(maps_dir, f"{f['i']}.npz"))}
    pos = {}
    for sc, ch in chains.items():
        for m, t in enumerate(ch):
            pos[t] = (sc, m)
    cand = []
    n_with = {n: 0 for n in range(1, NMAX + 1)}
    for f in frames:
        if f["i"] not in cached.values():
            continue
        sc, m = pos[f["token"]]
        ch = chains[sc]
        for n in range(1, NMAX + 1):
            if m - n >= 1 and all(os.path.exists(os.path.join(GTDIR, ch[m - k] + SIGFILE)) for k in range(1, n + 1)):
                n_with[n] += 1
        if m - NMAX < 1:
            continue
        preds = [ch[m - k] for k in range(1, NMAX + 1)]
        if not all(os.path.exists(os.path.join(GTDIR, t + SIGFILE)) for t in preds):
            continue
        cost = sum(1 for t in preds if t not in cached)
        cand.append(dict(i=f["i"], token=f["token"], scene=f["scene"], m=m, preds=preds, cost=cost))
    # how many of the 246 frames have the predecessors already cached
    have = {n: 0 for n in range(1, NMAX + 1)}
    for f in frames:
        if f["i"] not in cached.values():
            continue
        sc, m = pos[f["token"]]
        ch = chains[sc]
        for n in range(1, NMAX + 1):
            if m - n >= 1 and all(ch[m - k] in cached for k in range(1, n + 1)):
                have[n] += 1
    # greedy: cheapest first, at most 2 per scene, within the budget; extra frames are counted once
    cand.sort(key=lambda c: (c["cost"], c["scene"]))
    chosen, extra, per_scene = [], set(), {}
    for c in cand:
        if per_scene.get(c["scene"], 0) >= 2:
            continue
        new = {t for t in c["preds"] if t not in cached}
        if len(extra | new) > BUDGET:
            continue
        chosen.append(c)
        extra |= new
        per_scene[c["scene"]] = per_scene.get(c["scene"], 0) + 1
    ext = [dict(i=1000 + k, token=t, scene=pos[t][0], file=t + SIGFILE) for k, t in enumerate(sorted(extra))]
    ext_id = {e["token"]: e["i"] for e in ext}
    for c in chosen:
        c["pred_ids"] = [cached[t] if t in cached else ext_id[t] for t in c["preds"]]
    plan = dict(currents=chosen, ext=ext, n_cached_with_pred=have, n_candidates=len(cand), n_subset_frames=len(cached))
    json.dump(plan, open(PLAN, "w"), indent=1)
    msg = (f"plan: {len(cached)} cached subset frames; frames whose N previous keyframes are already cached: N=1 {have[1]}, N=2 {have[2]}, N=3 {have[3]}, N=4 {have[4]} (fewer than 150: extra inference needed); "
           f"{len(cand)} subset frames have 4 predecessors in the dataset; chosen {len(chosen)} current frames of {len({c['scene'] for c in chosen})} scenes with {len(ext)} EXTRA frames to infer (budget {BUDGET}, about 12 min CPU)")
    print(msg)
    progress(msg)
    return plan


def do_infer():
    plan = json.load(open(PLAN))
    cp = LR.resolve_ckpt(RUN, None)
    tag = LR.tag_of(RUN, cp)
    LR.infer(RUN, cp, tag, plan["ext"], 0, "")
    progress(f"infer: extra predecessor frames done ({len(plan['ext'])} frames, maps in readout_maps/{tag}/1000+)")


# ---------------------------------------------------------------------------------------------------------------- warp + combine
def load_gt(token_file):
    g = np.load(os.path.join(GTDIR, token_file), allow_pickle=True)["gt"].item()["cur"]
    return g


def warp(prob, cam_prev, T_cur, T_prev, mode="nn"):
    """previous map (X, Y) and its camera mask -> the current grid: (values, valid)"""
    Tc = np.asarray(T_cur, float)
    Tp = np.asarray(T_prev, float)
    P = np.c_[LR.XC.ravel(), LR.YC.ravel(), np.zeros(LR.XC.size), np.ones(LR.XC.size)]
    q = (np.linalg.inv(Tp) @ Tc @ P.T).T                                           # cell centres in the previous ego frame
    fi, fj = (q[:, 0] - LR.X0) / LR.RES, (q[:, 1] - LR.Y0) / LR.RES                # continuous cell coordinates (cell k spans [k, k + 1))
    ii, jj = np.floor(fi).astype(int), np.floor(fj).astype(int)
    inside = (ii >= 0) & (ii < LR.NX) & (jj >= 0) & (jj < LR.NY)
    ic, jc = np.clip(ii, 0, LR.NX - 1), np.clip(jj, 0, LR.NY - 1)
    valid = inside & cam_prev[ic, jc]
    if mode == "nn":
        val = prob[ic, jc]
    else:                                                                          # bilinear between cell centres (centre of cell k at k + 0.5)
        u, v = fi - 0.5, fj - 0.5
        i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
        du, dv = u - i0, v - j0
        ok = (i0 >= 0) & (i0 + 1 < LR.NX) & (j0 >= 0) & (j0 + 1 < LR.NY)
        i0c, j0c = np.clip(i0, 0, LR.NX - 2), np.clip(j0, 0, LR.NY - 2)
        val = (prob[i0c, j0c] * (1 - du) * (1 - dv) + prob[i0c + 1, j0c] * du * (1 - dv) + prob[i0c, j0c + 1] * (1 - du) * dv + prob[i0c + 1, j0c + 1] * du * dv)
        valid = valid & ok
    return val.reshape(LR.NX, LR.NY).astype(np.float32), valid.reshape(LR.NX, LR.NY)


def combine(cur, warped, rule):
    """cur (X, Y); warped = list of (values, valid); rule a / b / c"""
    if not warped:
        return cur
    if rule == "a":
        out = cur.copy()
        for v, m in warped:
            out = np.maximum(out, np.where(m, v, 0.0))
        return out
    if rule == "b":
        keep = (1.0 - cur).astype(np.float64)
        for v, m in warped:
            keep = keep * np.where(m, 1.0 - v, 1.0)
        return (1.0 - keep).astype(np.float32)
    s, n = cur.astype(np.float64).copy(), np.ones(cur.shape)
    for v, m in warped:
        s += np.where(m, v, 0.0)
        n += m
    return (s / n).astype(np.float32)


def shifted_pose(T_prev, T_cur, d=3.0):
    T = np.array(T_prev, float)
    T[:3, 3] += np.asarray(T_cur, float)[:3, 1] * d                                # 3 m along the current ego lateral axis (global frame)
    return T


def eval_all():
    from scipy import ndimage  # noqa: F401
    sys.path.insert(0, os.path.join(LR.ROOT, "temp", "lane_tests"))
    import paint_vs_label as PV
    plan = json.load(open(PLAN))
    frames = LR.load_frames()
    cp = LR.resolve_ckpt(RUN, None)
    tag = LR.tag_of(RUN, cp)
    maps_dir = os.path.join(LR.MAPS_DIR, tag)
    st = LR.get_static(frames)
    cur = plan["currents"]
    ext = {e["i"]: e for e in plan["ext"]}
    scene_ids = {s: k for k, s in enumerate(sorted({c["scene"] for c in cur}))}
    S = len(scene_ids)
    sc = np.array([scene_ids[c["scene"]] for c in cur])
    # variants
    V = [("base", 0, None, "nn", "")]
    for n in range(1, NMAX + 1):
        for r in "abc":
            V.append((f"N{n}{r}", n, r, "nn", ""))
    for r in "abc":
        V.append((f"N1{r}-bilinear", 1, r, "bil", ""))
    for n in (1, 4):
        for r in "abc":
            V.append((f"N{n}{r}-shift3m", n, r, "nn", "shift"))
            V.append((f"N{n}{r}-wrongmotion", n, r, "nn", "wrong"))
    names = [v[0] for v in V]
    F = len(cur)
    cnt = np.zeros((F, len(V), 2, 3, 4))
    # label points of the current frame (outside intersections, in fov, 5-60 m): class arrays and found flags
    pts_cls, pts_found, seg_stats = [], [[] for _ in V], np.zeros((F, len(V), 3))
    galign = np.zeros((F, NMAX, 3, 3, 4))
    gts = {}
    for k, c in enumerate(cur):
        g = load_gt(c["token"] + SIGFILE)
        gts[k] = g
    for k, c in enumerate(cur):
        g = gts[k]
        i = c["i"]
        Tc = np.asarray(g["meta"]["T_e2g"], float)
        P_cur = np.load(os.path.join(maps_dir, f"{i}.npz"))["lane_divider"].astype(np.float32)
        G, cam, I, occ = st["lane"][i], st["cam"][i], st["inter"][i], st["occ"][i]
        nG_all = [LR.dilate(G, r) for r in LR.CELLS]
        Gj = G & ~I
        nG_out = [LR.dilate(Gj, r) for r in LR.CELLS]
        # previous frames
        prev, prev_gt = [], []
        for n, (tok, pid) in enumerate(zip(c["preds"], c["pred_ids"]), start=1):
            pmap = np.load(os.path.join(maps_dir, f"{pid}.npz"))["lane_divider"].astype(np.float32)
            gp = load_gt(tok + SIGFILE)
            prev.append((pmap, np.asarray(gp["camera"], np.float32) > 0, np.asarray(gp["meta"]["T_e2g"], float)))
            prev_gt.append(np.asarray(gp["lane_divider"], np.float32))
        ko = (k + 7) % F                                                            # another scene's pair for the wrong-motion control (a different scene; checked below)
        while cur[ko]["scene"] == c["scene"]:
            ko = (ko + 1) % F
        other = cur[ko]
        To = np.asarray(gts[ko]["meta"]["T_e2g"], float)
        Tpo_list = [np.asarray(load_gt(t + SIGFILE)["meta"]["T_e2g"], float) for t in other["preds"]]       # the other pair's previous poses n = 1..4
        # polylines of the current frame: points (grid, fov, outside intersection) per segment
        segs = []
        for pl in g["meta"]["vectors"].get("lane_divider", []):
            pts, _ = PV.densify(pl)
            if len(pts) < 2:
                continue
            cx, cy = np.floor((pts[:, 0] - LR.X0) / LR.RES).astype(int), np.floor((pts[:, 1] - LR.Y0) / LR.RES).astype(int)
            ok = (cx >= 0) & (cx < LR.NX) & (cy >= 0) & (cy < LR.NY)
            ok[ok] = cam[cx[ok], cy[ok]] & ~I[cx[ok], cy[ok]]
            idx = np.nonzero(ok)[0]
            for part in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1) if len(idx) else []:
                if len(part) >= 2:
                    segs.append((cx[part], cy[part], pts[part]))
        pcell = [(cx, cy, p) for cx, cy, p in segs]
        # points for the class table: all points of the segments with 5 <= x < 60
        px = np.concatenate([p[:, 0] for _, _, p in pcell]) if pcell else np.zeros(0)
        py = np.concatenate([p[:, 1] for _, _, p in pcell]) if pcell else np.zeros(0)
        pcx = np.concatenate([cx for cx, _, _ in pcell]) if pcell else np.zeros(0, int)
        pcy = np.concatenate([cy for _, cy, _ in pcell]) if pcell else np.zeros(0, int)
        sel = (px >= 5) & (px < 60)
        pts_cls.append(np.c_[np.full(sel.sum(), k), px[sel], np.abs(py[sel]), occ[pcx[sel], pcy[sel]]])
        for vi, (name, n, r, mode, ctl) in enumerate(V):
            warped = []
            for (pmap, pcam, Tp) in prev[:n]:
                T_use = Tp
                if ctl == "shift":
                    T_use = shifted_pose(Tp, Tc)
                elif ctl == "wrong":                                                # the relative ego motion (current -> previous keyframe n) of ANOTHER scene's pair, applied to this pair
                    T_use = Tc @ np.linalg.inv(To) @ Tpo_list[len(warped)]
                warped.append(warp(pmap, pcam, Tc, T_use, mode))
            M = combine(P_cur, warped, r) if r else P_cur
            Pb = M > 0.5
            nP = [LR.dilate(Pb, rr) for rr in LR.CELLS]
            cnt[k, vi, 0] = LR.tol_counts(Pb, G, cam, nG_all, nP)
            Pj = Pb & ~I
            cnt[k, vi, 1] = LR.tol_counts(Pj, Gj, cam & ~I, nG_out, [LR.dilate(Pj, rr) for rr in LR.CELLS])
            found = nP[1][pcx, pcy] if len(pcx) else np.zeros(0, bool)
            pts_found[vi].append(found[sel])
            # segments found nowhere / >= 50 %
            nw, g50, tot = 0, 0, 0
            for cx, cy, _ in pcell:
                fr = nP[1][cx, cy].mean()
                tot += 1
                nw += fr == 0
                g50 += fr >= 0.5
            seg_stats[k, vi] = [tot, nw, g50]
        # pose check without the model: the previous frames' GT lane cells warped into the current frame against the current GT (valid = previous grid and camera mask, and the current camera mask)
        for n in range(1, NMAX + 1):
            for ci_, ctl in enumerate(("", "shift", "wrong")):
                Tp = prev[n - 1][2]
                T_use = Tp if ctl == "" else shifted_pose(Tp, Tc) if ctl == "shift" else Tc @ np.linalg.inv(To) @ Tpo_list[n - 1]
                val, ok = warp(prev_gt[n - 1], prev[n - 1][1], Tc, T_use, "nn")
                Pb = (val > 0.5) & ok
                cm = ok & cam
                galign[k, n - 1, ci_] = LR.tol_counts(Pb, G, cm, nG_all, [LR.dilate(Pb, rr) for rr in LR.CELLS])
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{F}", flush=True)
    cls = np.concatenate(pts_cls)
    found_all = np.stack([np.concatenate(pf) for pf in pts_found])                   # (V, Npts)
    np.savez(os.path.join(HERE, "tu_arrays.npz"), cnt=cnt, cls=cls, found=found_all, seg=seg_stats, sc=sc, names=np.array(names), galign=galign)
    return names, V, cnt, cls, found_all, seg_stats, sc, S, cur, plan, galign


def report(names, V, cnt, cls, found_all, seg_stats, sc, S, cur, plan, galign):
    bt = LR.Boot(S)
    iou, prec, rec, ci = LR.iou, LR.prec, LR.rec, LR.ci
    L, R = [], {"n_current": len(cur), "n_scenes": S, "n_extra_frames": len(plan["ext"]), "variants": names}
    L.append(f"==== temporal union probe: t20_base_s0 single-frame lane_divider maps of the previous N keyframes warped into the current ego frame; {len(cur)} current frames of {S} scenes, "
             f"{len(plan['ext'])} extra predecessor frames inferred (CPU, memory empty) ====")
    L += ["", "CAVEATS (read first):",
          " - a post-hoc aggregation of single-frame predictions (the network ran with an EMPTY memory for every frame); an upper-ish probe of the idea of a longer line memory, not a trained memory and not what a trained memory would do",
          " - the current frame's GT is the reference for all variants; vehicles move but lines do not, so no motion compensation of objects; cells outside a previous map's grid or camera (fov) mask are masked (no evidence)",
          " - poses: the GT cache meta['T_e2g'] (CAM_FRONT sample_data ego pose of each keyframe, the pose the GT is registered to); a pose error shows up in the controls below",
          f" - {len(cur)} current frames (frames of the 246-frame subset with >= 4 predecessors in the dataset, chosen to fit the extra-inference budget), not the 246; the baseline row is the same frames",
          " - the rules and the threshold 0.5 were fixed before looking; nothing was tuned on IoU; the intervals are paired scene bootstraps (2000 resamples) and exclude the choice of the 'best' row (there is none)"]
    tot = cnt.sum(0)                                                                  # (V, 2, 3, 4)
    bi = names.index("base")
    boots = [bt.resample(bt.scenes(cnt[:, v], sc)) for v in range(len(names))]
    L += ["", "-- IoU at tolerance 0 / 0.5 / 1.0 m (all cells | outside intersections), precision / recall at 0.5 m (all cells); d = paired difference to base [95% CI]",
          f"{'variant':20s} | {'IoU@0':>6s} {'IoU@0.5':>8s} {'d [95% CI]':>22s} {'IoU@1':>6s} | {'outside@0.5':>11s} {'d [95% CI]':>22s} | {'prec':>6s} {'rec':>6s}"]
    R["rows"] = {}
    for vi, nm in enumerate(names):
        row = []
        for v in (0, 1):
            d = iou(boots[vi][:, v, 1]) - iou(boots[bi][:, v, 1])
            row.append((iou(tot[vi, v, 1]), iou(tot[vi, v, 1]) - iou(tot[bi, v, 1]), ci(d)))
        L.append(f"{nm:20s} | {iou(tot[vi, 0, 0]):6.4f} {row[0][0]:8.4f} {row[0][1]:+8.4f} [{row[0][2][0]:+.3f},{row[0][2][1]:+.3f}] {iou(tot[vi, 0, 2]):6.4f} | {row[1][0]:11.4f} {row[1][1]:+8.4f} [{row[1][2][0]:+.3f},{row[1][2][1]:+.3f}] | "
                 f"{prec(tot[vi, 0, 1]):6.3f} {rec(tot[vi, 0, 1]):6.3f}")
        R["rows"][nm] = dict(iou0=float(iou(tot[vi, 0, 0])), iou05=float(iou(tot[vi, 0, 1])), iou1=float(iou(tot[vi, 0, 2])), iou05_out=float(iou(tot[vi, 1, 1])), d05=row[0][1], d05_ci=row[0][2], d05_out=row[1][1], d05_out_ci=row[1][2],
                             prec05=float(prec(tot[vi, 0, 1])), rec05=float(rec(tot[vi, 0, 1])), iou1_out=float(iou(tot[vi, 1, 2])), iou0_out=float(iou(tot[vi, 1, 0])))
    # precision / recall differences with intervals
    L += ["", "-- precision / recall at 0.5 m (all cells) and false predicted cells; paired differences to base [95% CI]"]
    for vi, nm in enumerate(names):
        if vi == bi:
            continue
        dp = prec(boots[vi][:, 0, 1]) - prec(boots[bi][:, 0, 1])
        dr = rec(boots[vi][:, 0, 1]) - rec(boots[bi][:, 0, 1])
        fp_ratio = tot[vi, 0, 1, 1] / tot[bi, 0, 1, 1]
        L.append(f"{nm:20s} precision {prec(tot[vi, 0, 1]) - prec(tot[bi, 0, 1]):+.3f} [{ci(dp)[0]:+.3f},{ci(dp)[1]:+.3f}]   recall {rec(tot[vi, 0, 1]) - rec(tot[bi, 0, 1]):+.3f} [{ci(dr)[0]:+.3f},{ci(dr)[1]:+.3f}]   false predicted cells x{fp_ratio:.2f}   "
                 f"(outside: prec {prec(tot[vi, 1, 1]):.3f}, rec {rec(tot[vi, 1, 1]):.3f}; base {prec(tot[bi, 1, 1]):.3f} / {rec(tot[bi, 1, 1]):.3f})")
    ga = galign.sum(0)                                                                # (N, ctl, tol, 4)
    L += ["", "-- pose check WITHOUT the model: the previous keyframes' GT lane cells warped into the current frame with the poses, scored against the current GT (cells valid in the previous grid and camera mask and in the current camera mask); "
          "a correct pose gives a high score, the controls must be low",
          f"{'N':>3s} {'pose':12s} | {'IoU@0':>6s} {'IoU@0.5':>8s} {'IoU@1':>6s} | {'prec@0.5':>8s} {'rec@0.5':>8s}"]
    R["pose_check"] = {}
    for n in range(NMAX):
        for ci_, cn in enumerate(("poses", "shift 3 m", "wrong motion")):
            L.append(f"{n + 1:3d} {cn:12s} | {iou(ga[n, ci_, 0]):6.3f} {iou(ga[n, ci_, 1]):8.3f} {iou(ga[n, ci_, 2]):6.3f} | {prec(ga[n, ci_, 1]):8.3f} {rec(ga[n, ci_, 1]):8.3f}")
            R["pose_check"][f"N{n + 1} {cn}"] = dict(iou0=float(iou(ga[n, ci_, 0])), iou05=float(iou(ga[n, ci_, 1])), iou1=float(iou(ga[n, ci_, 2])), prec05=float(prec(ga[n, ci_, 1])), rec05=float(rec(ga[n, ci_, 1])))
    # classes of the label points
    k_pt = cls[:, 0].astype(int)
    x, ay, occ = cls[:, 1], cls[:, 2], cls[:, 3] > 0
    classes = ([("all points", np.ones(len(x), bool)), ("not occluded", ~occ), ("occluded", occ)] + [(f"|y| {lo}-{hi} m", (ay >= lo) & (ay < hi)) for lo, hi in ((0, 2.5), (2.5, 5), (5, 8), (8, 12), (12, 30))]
               + [(f"range {lo}-{hi} m", (x >= lo) & (x < hi)) for lo, hi in ((5, 15), (15, 30), (30, 45), (45, 60))])
    pts_sc = sc[k_pt]
    L += ["", f"-- recall of the GT label points (outside intersections, in fov, 5-60 m ahead; {len(x)} points; found = a predicted cell within 0.5 m): base recall and the share of the MISSED points that come back, "
          "(recall - base) / (1 - base), per class; N = 1 / 2 / 3 / 4 for the three rules"]
    R["classes"] = {}
    sel_v = [names.index(f"N{n}{r}") for r in "abc" for n in (1, 2, 3, 4)]
    L.append(f"{'class':18s} {'n':>6s} {'base recall':>11s} | " + " | ".join(f"rule {r}: N=1 N=2 N=3 N=4" for r in "abc"))
    for cn, m in classes:
        base_r = found_all[bi][m].mean()
        cells = []
        for r in "abc":
            vals = []
            for n in (1, 2, 3, 4):
                vi = names.index(f"N{n}{r}")
                rec_v = found_all[vi][m].mean()
                vals.append((rec_v - base_r) / (1 - base_r) if base_r < 1 else float("nan"))
            cells.append("  ".join(f"{v:5.2f}" for v in vals))
        L.append(f"{cn:18s} {int(m.sum()):6d} {base_r:11.3f} | " + " | ".join(f"{c:>26s}" for c in cells))
        R["classes"][cn] = dict(n=int(m.sum()), base=float(base_r), back={f"N{n}{r}": float((found_all[names.index(f'N{n}{r}')][m].mean() - base_r) / (1 - base_r)) for r in "abc" for n in (1, 2, 3, 4)})
    # recall differences with intervals for a few key variants (paired over scenes)
    L += ["", "-- recall difference to base per class with paired scene-bootstrap 95% intervals (rule a = max, N = 1 / 4; rule b N = 4; rule c N = 4)"]
    for vname in ("N1a", "N4a", "N4b", "N4c", "N4a-shift3m", "N4a-wrongmotion"):
        vi = names.index(vname)
        for cn, m in classes:
            num_b, den = np.bincount(pts_sc, found_all[bi] * m, S), np.bincount(pts_sc, m.astype(float), S)
            num_v = np.bincount(pts_sc, found_all[vi] * m, S)
            with np.errstate(divide="ignore", invalid="ignore"):
                d = (bt.W @ num_v) / (bt.W @ den) - (bt.W @ num_b) / (bt.W @ den)
            L.append(f"{vname:16s} {cn:16s} n {int(den.sum()):6d}  {num_v.sum() / den.sum() - num_b.sum() / den.sum():+.4f} [{ci(d)[0]:+.4f}, {ci(d)[1]:+.4f}]")
    # segments
    L += ["", "-- whole GT line segments (polyline parts in grid, fov, outside intersections): found nowhere / found >= 50 % (found = within 0.5 m of a predicted cell)"]
    s = seg_stats.sum(0)
    for vi, nm in enumerate(names):
        L.append(f"{nm:20s} segments {int(s[vi, 0])}   found nowhere {int(s[vi, 1])} ({s[vi, 1] / s[vi, 0]:.3f})   found >= 50 % {int(s[vi, 2])} ({s[vi, 2] / s[vi, 0]:.3f})")
    R["segments"] = {nm: dict(total=int(s[vi, 0]), nowhere=int(s[vi, 1]), ge50=int(s[vi, 2])) for vi, nm in enumerate(names)}
    txt = "\n".join(L) + "\n"
    open(os.path.join(HERE, "temporal_union_result.txt"), "w").write(txt)
    json.dump(R, open(os.path.join(HERE, "temporal_union_result.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["plan", "infer", "eval", "all"])
    a = ap.parse_args()
    if a.cmd in ("plan", "all"):
        do_plan()
    if a.cmd in ("infer", "all"):
        do_infer()
    if a.cmd in ("eval", "all"):
        out = eval_all()
        report(*out)
        progress("eval done: temporal_union_result.txt / .json")


if __name__ == "__main__":
    main()
