"""scratch (Mirko 2026-11-10 'die Punkte sind instabil'; E283 follow-up): how unstable is the delivered lane graph from one keyframe to the next, MEASURED? CPU, VAL 873, baseline B + hard step 1.5 m, one process per seed (argument 0 / 1).
For every pair (previous keyframe a, keyframe b; 0.5 s apart; 851 of the 873 frames have a previous one inside VAL) the points of a (polylines resampled every 0.5 m) are warped into the ego frame of b with the poses (the convention is
chosen on the GT: the one with the smallest residual of the GT of a against the GT of b), and the share of them that lie within 0.5 / 1.0 m of the delivered graph of b is counted; the same for the lane GT (a warped against b). Region: points that
lie in the metric box (|x|, |y| < 40 m) of b AFTER the warp and of a BEFORE it (the part of the road both frames see). The GT share is the natural change between the frames (parked-car occlusion and lanes that enter or leave the 0.5 s step are in
both numbers), the delivered share minus the GT share is the INSTABILITY of the network (a point that was there and is gone or has moved although the road has not changed), reported both ways (a to b and b to a) and split into plain
lane points and connector points (the GT l4 `kind` of the nearest GT point of b), by distance ahead of the ego.
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 bash tools/env/_py14.sh temp/e284/p23_jitter.py <seed>
READING (written before the run; a diagnostic, no verdict on a model): the delivered graph is temporally LESS consistent than the GT when the share within 0.5 m is lower than the GT's by more than 0.05 on both seeds; then the 'unstable points'
are a measured weakness and a temporal-smoothing card (E281 tried a long-link memory: CUT) may be written with this number as its baseline; at 0.05 or less the complaint is not a matter of the keyframe-to-keyframe consistency (it may be the frame-internal
jitter of the vertex placement: `vert100_dp` 19 against the GT's 11) and the next measure is that."""
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

HEAD = open(os.path.join(os.environ["BFLC_ROOT"], "temp", "e284", "pair_geometry.py")).read().split("# ---- TRAIN set")[0]
exec(compile(HEAD, "pair_geometry_head", "exec"))          # numpy, cKDTree, LD, QE, QT, ROOT
import qgraph_readout as QR  # noqa: E402

SEED = int(sys.argv[1])
SPEC = QR.parse("close:1.5@0.5")
frames = QE.val_frames()
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{SEED}.pkl"), "rb"))
POSE = np.load(os.path.expanduser("~/exp_gcar_b123/poses_val.npz"))
T, R, PREV = POSE["t"], POSE["R"], POSE["prev"]
rows = [f.row for f in frames]
by_row = {r: i for i, r in enumerate(rows)}
pairs = [(by_row[int(PREV[r])], by_row[r]) for r in rows if int(PREV[r]) >= 0 and int(PREV[r]) in by_row]
print("pairs", len(pairs), flush=True)


def lane2ego(p):
    return np.c_[p[:, 1], -p[:, 0]]


def ego2lane(e):
    return np.c_[-e[:, 1], e[:, 0]]


CONV = {"ego pose, lane frame converted": lambda p, ra, ta, rb, tb: ego2lane((lane2ego(p) @ ra.T + ta - tb) @ rb),
        "pose acts on the lane frame": lambda p, ra, ta, rb, tb: (p @ ra.T + ta - tb) @ rb,
        "ego pose, R transposed": lambda p, ra, ta, rb, tb: ego2lane((lane2ego(p) @ ra + ta - tb) @ rb.T)}


def resample(lines, step=0.5):
    out = []
    for P in lines:
        P = np.asarray(P, float).reshape(-1, 2)
        if len(P) < 2:
            continue
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.r_[0, np.cumsum(seg)]
        if s[-1] < 1e-6:
            continue
        q = np.arange(0, s[-1], step)
        out.append(np.c_[np.interp(q, s, P[:, 0]), np.interp(q, s, P[:, 1])])
    return np.concatenate(out) if out else np.zeros((0, 2))


def gt_points(f):
    g = f.lane()
    pts, conn = [], []
    for pc in g.get("pieces", []):
        q = np.asarray(pc["pts"], float).reshape(-1, 2)
        if len(q) < 2:
            continue
        c = np.zeros(len(q), bool)
        for s in pc["segments"]:
            if s["kind"] == "connector":
                c[s["i0"]: s["i1"] + 1] = True
        pts.append(q)
        conn.append(c)
    f._lane = f._ref = None
    return (np.concatenate(pts), np.concatenate(conn)) if pts else (np.zeros((0, 2)), np.zeros(0, bool))


# delivered graph of every frame, resampled; GT points with the connector flag
DEL, GTP = [], []
for b, (f, d) in enumerate(zip(frames, dump)):
    pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                         links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), SPEC)
    graph, bk = QE.readout(pred, 0.5, 0.5)
    DEL.append(resample(graph.lines))
    GTP.append(gt_points(f))
    if (b + 1) % 200 == 0:
        print(f"  read {b + 1}/{len(frames)}", flush=True)

# convention from the GT
best, res = None, {}
for name, fn in CONV.items():
    r = []
    for a, b in pairs[::max(1, len(pairs) // 40)][:40]:
        pa, pb = GTP[a][0], GTP[b][0]
        if len(pa) and len(pb):
            w = fn(pa, R[rows[a]], T[rows[a]], R[rows[b]], T[rows[b]])
            ok = (np.abs(w) < 40).all(1)
            if ok.any():
                r.append(float(np.median(cKDTree(pb).query(w[ok])[0])))
    res[name] = float(np.mean(r)) if r else 9e9
best = min(res, key=res.get)
print("pose convention (mean median residual of the GT of a against the GT of b, m):", {k: round(v, 3) for k, v in res.items()}, "->", best, flush=True)
fn = CONV[best]
BOXM = 40.0
rows_out = {k: [] for k in ("del05", "del10", "gt05", "gt10", "del05_conn", "gt05_conn", "del05_plain", "gt05_plain", "n_del", "n_gt", "n_conn", "n_plain")}
by_dist = {"0-15": [[], []], "15-30": [[], []], "30-40": [[], []]}                      # share within 0.5 m of the delivered graph / of the GT, by |y| of the point in b
for direction in ("a->b", "b->a"):
    for a0, b0 in pairs:
        a, b = (a0, b0) if direction == "a->b" else (b0, a0)
        # points of frame a, in the box of a, warped into b and kept when inside b's box
        for kind, srcpts, dstpts in (("del", DEL[a], DEL[b]), ("gt", GTP[a][0], GTP[b][0])):
            if not len(srcpts) or not len(dstpts):
                continue
            w = fn(srcpts, R[rows[a]], T[rows[a]], R[rows[b]], T[rows[b]])
            ok = (np.abs(srcpts) < BOXM).all(1) & (np.abs(w) < BOXM).all(1)
            if not ok.any():
                continue
            d = cKDTree(dstpts).query(w[ok])[0]
            if kind == "del" and len(GTP[a][0]):                                    # refinement (the GT baseline of the first reading is 1 by construction: the lane GT of a is inside the lane GT of b wherever the boxes overlap)
                corr = cKDTree(GTP[a][0]).query(srcpts[ok])[0] <= 1.0                  # the point of a is correct (within 1 m of the GT of a)
                if corr.any():
                    rows_out.setdefault("persist10", []).append(float((d[corr] <= 1.0).mean()))
                    rows_out.setdefault("persist05", []).append(float((d[corr] <= 0.5).mean()))
                    both = d[corr][d[corr] <= 1.0]
                    if len(both):
                        rows_out.setdefault("jit_med", []).append(float(np.median(both)))
                        rows_out.setdefault("jit_p90", []).append(float(np.percentile(both, 90)))
                    rows_out.setdefault("n_correct", []).append(int(corr.sum()))
            if kind == "gt":                                                      # b's coverage of the GT points that both frames see = b's recall in the shared region
                d2 = cKDTree(DEL[b]).query(w[ok])[0] if len(DEL[b]) else np.full(int(ok.sum()), 1e9)
                rows_out.setdefault("cover_b10", []).append(float((d2 <= 1.0).mean()))
            rows_out[f"{kind}05"].append(float((d <= 0.5).mean()))
            rows_out[f"{kind}10"].append(float((d <= 1.0).mean()))
            rows_out[f"n_{kind}"].append(int(ok.sum()))
            if kind == "gt":
                cf = GTP[a][1][ok]
                for tag, m in (("conn", cf), ("plain", ~cf)):
                    if m.any():
                        rows_out[f"gt05_{tag}"].append(float((d[m] <= 0.5).mean()))
                        rows_out[f"n_{tag}"].append(int(m.sum()))
            else:
                # connector points of the delivered graph: the nearest GT point of a is a connector point
                ta = cKDTree(GTP[a][0]).query(srcpts[ok])[1] if len(GTP[a][0]) else None
                if ta is not None:
                    cf = GTP[a][1][ta]
                    for tag, m in (("conn", cf), ("plain", ~cf)):
                        if m.any():
                            rows_out[f"del05_{tag}"].append(float((d[m] <= 0.5).mean()))
            for k_, (lo, hi) in {"0-15": (0, 15), "15-30": (15, 30), "30-40": (30, 40)}.items():
                m = (np.abs(w[ok][:, 1]) >= lo) & (np.abs(w[ok][:, 1]) < hi)
                if m.any():
                    by_dist[k_][0 if kind == "del" else 1].append(float((d[m] <= 0.5).mean()))
out = {k: (float(np.mean(v)) if v and not k.startswith("n_") else (float(np.sum(v)) if v else 0.0)) for k, v in rows_out.items()}
out["by_distance_ahead_share_within_05"] = {k: dict(delivered=float(np.mean(v[0])) if v[0] else None, gt=float(np.mean(v[1])) if v[1] else None) for k, v in by_dist.items()}
out.update(seed=SEED, pose_convention=best, pose_residual=res, pairs=len(pairs))
out["instability_05"] = out["gt05"] - out["del05"]
out["instability_10"] = out["gt10"] - out["del10"]
json.dump(out, open(os.path.join(ROOT, "temp", "e284", f"p23_jitter_s{SEED}.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
print(f"FIRST READING (rule of the docstring) seed {SEED}: delivered share within 0.5 m {out['del05']:.3f}, GT {out['gt05']:.3f}, instability {out['instability_05']:+.3f}; within 1.0 m {out['del10']:.3f} / {out['gt10']:.3f}; "
      f"graph less consistent than the GT by more than 0.05: {out['instability_05'] > 0.05}  [NOT VALID AS A TEST: the GT share is 1 by construction, so the rule only says that the delivered graph is not the GT]")
print(f"REFINED (by hand, written after the first run): of the correct delivered points of a (within 1 m of the GT) {out.get('persist10', 0):.3f} have a delivered point of b within 1.0 m and {out.get('persist05', 0):.3f} within 0.5 m; "
      f"b covers {out.get('cover_b10', 0):.3f} of the shared GT points (its recall there); among the persistent ones the position jumps by median {out.get('jit_med', 0):.2f} m, p90 {out.get('jit_p90', 0):.2f} m")
