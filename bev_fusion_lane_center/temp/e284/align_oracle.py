"""scratch (continuation of offset_origin.py: the 1 - 2.5 m offsets are NOT a graph-building error - the head's own centre map has its ridge where the graph is, 94 % of the offset points on plain road have a real ridge but only 14 % of them have it on the GT):
is the offset a PER-FRAME RIGID shift (a registration / pose error of the frame, or a global localisation error of the network) or local? ORACLE diagnostic, CPU, no model: baseline B seed s (full dump, hard close-node step 1.5 m, read-out 0.5), VAL 873.
Per frame the delivered polylines (all kinds, 0.5 m samples) are moved by the translation (tx, ty) in +-4 m (step 0.5) and a rotation of +-2 deg (step 1 deg) about the car that maximise the mean of
  P1 = share of the delivered points within 1.0 m of a GT lane point and R1 = share of the GT points (|x|, |y| < 48) within 1.0 m of a delivered point.
Reported (pooled over the points of all frames, and by zone: plain road = farther than 8 m from a GT connector point inside an intersection): P1, R1, F1 before and after, the distribution of |t| and of the angle, the share of the frames with |t| > 1 m,
and how much of the missed GT length (R1 loss) the per-frame rigid motion removes. The GT is used to find the motion: this says how much a PERFECT per-frame registration would give, not what a method can reach.
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 bash tools/env/_py14.sh temp/e284/align_oracle.py <seed> [--rows N]"""
import json
import os
import pickle
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qgraph_readout as QR  # noqa: E402

SEED = int(sys.argv[1])
N_ROWS = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 0
SPEC = QR.parse("close:1.5@0.5")
frames = QE.val_frames()
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{SEED}.pkl"), "rb"))
if N_ROWS:
    frames, dump = frames[:N_ROWS], dump[:N_ROWS]
TS = np.arange(-4.0, 4.01, 0.5)
ANG = np.radians([-2.0, -1.0, 0.0, 1.0, 2.0])
acc = {z: np.zeros(8) for z in ("all", "plain road")}      # P hit before, n delivered, P hit after, R hit before, n GT, R hit after, (unused), (unused)
per_frame = []
t0 = time.time()
for n, (f, d) in enumerate(zip(frames, dump)):
    g = f.lane()
    P, J = [], []
    for pc in g.get("pieces", []):
        q = np.asarray(pc["pts"], float).reshape(-1, 2)
        if len(q) < 2:
            continue
        inter = np.zeros(len(q), bool)
        for s in pc["segments"]:
            if s["kind"] == "connector" and s["in_intersection"]:
                inter[s["i0"]: s["i1"] + 1] = True
        P.append(q)
        J.append(inter)
    P, J = np.concatenate(P), np.concatenate(J)
    P = P[(np.abs(P) < 48.0).all(1)] if False else P
    inb = (np.abs(P) < 48.0).all(1)
    jt = cKDTree(P[J]) if J.any() else None
    zroad = (jt.query(P)[0] > 8.0) if jt is not None else np.ones(len(P), bool)
    pred = dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"]))
    graph, bk = QE.readout(QR.apply(pred, SPEC), 0.5, 0.5)
    if not graph.lines:
        f._lane = f._ref = None
        continue
    D = np.concatenate([np.asarray(l, float) for l in graph.lines])
    D = D[(np.abs(D) < 48.0).all(1)]
    if not len(D):
        continue
    gt_tree = cKDTree(P[inb])
    Pin, zin = P[inb], zroad[inb]
    zD = (jt.query(D)[0] > 8.0) if jt is not None else np.ones(len(D), bool)

    def score(Dm, tree_d=None):
        td = cKDTree(Dm)
        p_hit = gt_tree.query(Dm)[0] <= 1.0
        r_hit = td.query(Pin)[0] <= 1.0
        return p_hit, r_hit

    p0, r0 = score(D)
    best = (0.5 * (p0.mean() + r0.mean()), 0.0, 0.0, 0.0)
    Ds = D[::2]
    td_all = None
    for a in ANG:
        c, s_ = np.cos(a), np.sin(a)
        Dr = Ds @ np.array([[c, s_], [-s_, c]])
        for tx in TS:
            for ty in TS:
                Dm = Dr + (tx, ty)
                p_ = (gt_tree.query(Dm)[0] <= 1.0).mean()
                if p_ + 0.0 < best[0] - 0.5:
                    continue
                r_ = (cKDTree(Dm).query(Pin)[0] <= 1.0).mean()
                sc = 0.5 * (p_ + r_)
                if sc > best[0] + 1e-9:
                    best = (sc, float(tx), float(ty), float(np.degrees(a)))
    _, tx, ty, ad = best
    a = np.radians(ad)
    c, s_ = np.cos(a), np.sin(a)
    D2 = D @ np.array([[c, s_], [-s_, c]]) + (tx, ty)
    p1, r1 = score(D2)
    for z, mD, mG in (("all", np.ones(len(D), bool), np.ones(len(Pin), bool)), ("plain road", zD, zin)):
        acc[z] += np.array([p0[mD].sum(), mD.sum(), p1[mD].sum(), r0[mG].sum(), mG.sum(), r1[mG].sum(), 0, 0])
    per_frame.append((float(np.hypot(tx, ty)), float(ad), float(p0.mean()), float(r0.mean()), float(p1.mean()), float(r1.mean())))
    f._lane = f._ref = None
    if (n + 1) % 100 == 0:
        print(f"  frame {n + 1}/{len(frames)}  {time.time() - t0:.0f} s", flush=True)
PF = np.array(per_frame)
res = dict(seed=SEED, frames=len(PF))
for z, a in acc.items():
    P0_, P1_, R0_, R1_ = a[0] / a[1], a[2] / a[1], a[3] / a[4], a[5] / a[4]
    res[z] = dict(P_before=P0_, P_after=P1_, R_before=R0_, R_after=R1_, F1_before=2 * P0_ * R0_ / (P0_ + R0_), F1_after=2 * P1_ * R1_ / (P1_ + R1_), missed_gt_before=1 - R0_, missed_gt_after=1 - R1_, share_of_miss_removed=1 - (1 - R1_) / (1 - R0_))
res["shift"] = dict(median_t=float(np.median(PF[:, 0])), p90_t=float(np.percentile(PF[:, 0], 90)), share_frames_t_over_1m=float((PF[:, 0] > 1.0).mean()), share_frames_t_over_2m=float((PF[:, 0] > 2.0).mean()),
                    median_abs_angle=float(np.median(np.abs(PF[:, 1]))), share_frames_angle_nonzero=float((PF[:, 1] != 0).mean()), share_frames_at_grid_edge=float((PF[:, 0] >= 4.5).mean()))
res["per_frame_gain_median"] = dict(P=float(np.median(PF[:, 4] - PF[:, 2])), R=float(np.median(PF[:, 5] - PF[:, 3])))
json.dump(res, open(os.path.join(ROOT, "temp", "e284", f"align_oracle_s{SEED}{'_rows' + str(N_ROWS) if N_ROWS else ''}.json"), "w"), indent=1)
print(f"{len(PF)} frames, seed {SEED}")
for z in ("all", "plain road"):
    r = res[z]
    print(f"  {z:11s} P1 {r['P_before']:.3f} -> {r['P_after']:.3f}   R1 {r['R_before']:.3f} -> {r['R_after']:.3f}   F1 {r['F1_before']:.3f} -> {r['F1_after']:.3f}   share of the GT length missed {100 * r['missed_gt_before']:.1f} % -> {100 * r['missed_gt_after']:.1f} % ({100 * r['share_of_miss_removed']:.0f} % of the loss removed)")
print("  per-frame motion:", json.dumps({k: round(v, 3) for k, v in res["shift"].items()}))
print("  median per-frame gain:", json.dumps({k: round(v, 3) for k, v in res["per_frame_gain_median"].items()}))
