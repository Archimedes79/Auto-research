"""scratch (Mirko's OK of 2026-11-08 'Ja, messen'): how much of the junction target follows from CLEAN ENTRIES AND EXITS? An oracle on the 88 VAL frames (GT only, no network):
  GT     the exact points of every piece (sanity: must reproduce T)
  A      the lane stretches exact; every connector run (a contiguous run of connector points of a piece) replaced by ONE cubic Hermite curve from its true start to its true end with the true headings at both ends and the
         tangent length k x chord (k = 1.0 fixed, 0.8 / 1.2 as a sensitivity): the clean ends AND the true pairs are known, only the SHAPE between them is generic
  B      the lane stretches exact; a Hermite curve (k = 1.0) for EVERY geometrically possible pair of an entry (start of a connector run) and an exit (end of a connector run): clean ends, pairs unknown
         (a pair is possible when the heading changes by <= 130 deg, the chord is >= 2 m and the exit lies ahead of the entry along its heading and the entry behind the exit along the exit's heading)
The oracle map T' = exp(-d^2 / 2) of the nearest oracle point (points with |x|, |y| < 48, as the training target). Against the real target T over the junction cells (< 8 m from the junction points, |x|, |y| <= 47, T > 0.05 or the
compared map > 0.05): correlation, mean |X - T|, IoU of X >= 0.5 against T >= 0.5, and the fill of the empty cells (T < 0.1, within 6 m of a lane point; share with X > 0.3). The same numbers for the E279 output p (seed 0).
Also: the number of candidate pairs of B per true pair and the recall of the true pairs by the rule.
Reads temp/e284/target_vs_pred.npz. CPU, no network.
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh temp/e284/oracle_junction.py"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation", "tools/viz"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qtargets as QT  # noqa: E402

TOK = "5dd954d0ac0242a990ba35eef950c937"
z = np.load(os.path.join(ROOT, "temp", "e284", "target_vs_pred.npz"))
toks = [str(t) for t in z["tokens"]]
n_stat = 88
P0, T0, DJ0, DG0 = (z[k].astype(np.float32) for k in ("p", "t", "dj", "dg"))
by_tok = {f.token: f for f in QE.val_frames()}
c = -50.0 + (np.arange(100) + 0.5)
CX, CY = np.meshgrid(c, c, indexing="ij")
cells = np.stack([CX.ravel(), CY.ravel()], 1)
inbox = (np.abs(cells) <= 47.0).all(1)


def hermite(a, ta, b, tb, k=1.0, step=0.25):
    L = float(np.linalg.norm(b - a))
    n = max(8, int(1.6 * L / step))
    s = np.linspace(0, 1, n)[:, None]
    h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    return h00 * a + h10 * (k * L * ta) + h01 * b + h11 * (k * L * tb)


def cluster(items, tol_pos=1.5, tol_ang=40.0):
    """greedy clusters of (pos, tan): -> list of (pos, tan), id per item"""
    reps, ids = [], []
    for pos, tan in items:
        for j, (rp, rt) in enumerate(reps):
            if np.linalg.norm(pos - rp) <= tol_pos and np.degrees(np.arccos(np.clip(tan @ rt, -1, 1))) <= tol_ang:
                ids.append(j)
                break
        else:
            reps.append((pos, tan))
            ids.append(len(reps) - 1)
    return reps, ids


def heat_of(pts):
    pts = np.concatenate(pts) if isinstance(pts, list) else pts
    pts = pts[(np.abs(pts) < 48.0).all(1)]
    d = cKDTree(pts).query(cells)[0]
    return np.exp(-d * d / 2.0).astype(np.float32)


def allowed(a, ta, b, tb):
    ch = b - a
    ang = np.degrees(np.arccos(np.clip(ta @ tb, -1, 1)))
    return ang <= 130.0 and np.linalg.norm(ch) >= 2.0 and (ch @ ta) > 0 and (ch @ tb) > 0


VARS = ("GT", "A", "A k=0.8", "A k=1.2", "B")
maps = {v: [] for v in VARS}
pair_stats = dict(true_pairs=0, candidates=0, recalled=0, frames=0, short_runs=0, runs=0)
curves_picture = None
for i in range(n_stat + 1 if TOK not in toks[:n_stat] else n_stat):
    f = by_tok[toks[i]]
    g = f.lane()
    lane_pts, exact_all, runs = [], [], []
    for pc in g.get("pieces", []):
        q = np.asarray(pc["pts"], float).reshape(-1, 2)
        if len(q) < 2:
            continue
        tan = QT._tangents(q)
        conn = np.zeros(len(q), bool)
        for s in pc["segments"]:
            if s["kind"] == "connector":
                conn[s["i0"]: s["i1"] + 1] = True
        exact_all.append(q)
        lane_pts.append(q[~conn])
        k = 0
        while k < len(q):
            if conn[k]:
                j = k
                while j + 1 < len(q) and conn[j + 1]:
                    j += 1
                runs.append((q[k: j + 1], q[k], tan[k], q[j], tan[j]))
                k = j + 1
            else:
                k += 1
    lane_pts = [a for a in lane_pts if len(a)]
    long_runs = [r for r in runs if len(r[0]) >= 3]
    short_runs = [r[0] for r in runs if len(r[0]) < 3]
    pair_stats["runs"] += len(runs)
    pair_stats["short_runs"] += len(short_runs)
    base = lane_pts + short_runs
    maps["GT"].append(heat_of(exact_all))

    def curves(k):
        return [hermite(r[1], r[2], r[3], r[4], k=k) for r in long_runs]
    maps["A"].append(heat_of(base + curves(1.0)))
    maps["A k=0.8"].append(heat_of(base + curves(0.8)))
    maps["A k=1.2"].append(heat_of(base + curves(1.2)))
    ent, eid = cluster([(r[1], r[2]) for r in long_runs])
    ext, xid = cluster([(r[3], r[4]) for r in long_runs])
    true_pairs = set(zip(eid, xid))
    cand = [(a, b) for a in range(len(ent)) for b in range(len(ext)) if allowed(ent[a][0], ent[a][1], ext[b][0], ext[b][1])]
    pair_stats["true_pairs"] += len(true_pairs)
    pair_stats["candidates"] += len(cand)
    pair_stats["recalled"] += len(true_pairs & set(cand))
    pair_stats["frames"] += 1
    maps["B"].append(heat_of(base + [hermite(ent[a][0], ent[a][1], ext[b][0], ext[b][1]) for a, b in cand]))
    if toks[i] == TOK:
        curves_picture = (g, long_runs, [(ent[a], ext[b]) for a, b in cand])
    f._lane = f._ref = None
    if (i + 1) % 20 == 0:
        print(i + 1, "frames", flush=True)

pick = list(range(n_stat))
fi_pic = toks.index(TOK)
ALL = {v: np.stack(maps[v]) for v in VARS}
ALL["E279 output p"] = P0[: len(ALL["GT"])]
Tn = T0[: len(ALL["GT"])]
DJn, DGn = DJ0[: len(Tn)], DG0[: len(Tn)]


def measure(X, rows):
    a = dict(n=0, sx=0.0, st=0.0, sxx=0.0, stt=0.0, sxt=0.0, ab=0.0, hit=0, onx=0, ont=0, gap=0, gap_hi=0)
    for i in rows:
        m = (DJn[i] < 8.0) & inbox & ((X[i] > 0.05) | (Tn[i] > 0.05))
        mg = (DJn[i] < 8.0) & inbox & (Tn[i] < 0.1) & (DGn[i] < 6.0)
        mi = (DJn[i] < 8.0) & inbox
        a["n"] += int(m.sum())
        a["sx"] += float(X[i][m].sum())
        a["st"] += float(Tn[i][m].sum())
        a["sxx"] += float((X[i][m] ** 2).sum())
        a["stt"] += float((Tn[i][m] ** 2).sum())
        a["sxt"] += float((X[i][m] * Tn[i][m]).sum())
        a["ab"] += float(np.abs(X[i][m] - Tn[i][m]).sum())
        ox, ot = (X[i] >= 0.5) & mi, (Tn[i] >= 0.5) & mi
        a["hit"] += int((ox & ot).sum())
        a["onx"] += int(ox.sum())
        a["ont"] += int(ot.sum())
        a["gap"] += int(mg.sum())
        a["gap_hi"] += int((X[i][mg] > 0.3).sum())
    n = max(a["n"], 1)
    cov = a["sxt"] / n - a["sx"] / n * a["st"] / n
    vx, vt = a["sxx"] / n - (a["sx"] / n) ** 2, a["stt"] / n - (a["st"] / n) ** 2
    return dict(corr=float(cov / np.sqrt(max(vx * vt, 1e-12))), mean_abs=a["ab"] / n, iou=a["hit"] / max(a["onx"] + a["ont"] - a["hit"], 1), gap_fill=a["gap_hi"] / max(a["gap"], 1), cells=a["n"])


res = {k: measure(v, pick) for k, v in ALL.items()}
res["pairs"] = dict(frames=pair_stats["frames"], true_pairs_per_frame=pair_stats["true_pairs"] / pair_stats["frames"], candidates_per_frame=pair_stats["candidates"] / pair_stats["frames"],
                    candidates_per_true=pair_stats["candidates"] / max(pair_stats["true_pairs"], 1), recall_of_true_pairs=pair_stats["recalled"] / max(pair_stats["true_pairs"], 1), runs=pair_stats["runs"], short_runs=pair_stats["short_runs"])
json.dump(res, open(os.path.join(ROOT, "temp", "e284", "oracle_junction.json"), "w"), indent=1)
print("junction cells (< 8 m), 88 VAL frames, against the real target T")
for k in list(ALL):
    v = res[k]
    print(f"  {k:14s} corr {v['corr']:.3f}  mean |X - T| {v['mean_abs']:.3f}  IoU(0.5) {100 * v['iou']:5.1f} %  empty cells filled (> 0.3) {100 * v['gap_fill']:5.1f} %  (cells {v['cells']})")
print("pairs:", json.dumps({a: round(b, 2) for a, b in res["pairs"].items()}))

# ---- the picture of the frame: T, A, B (zoom)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

g, long_runs, cand_pic = curves_picture
fig, ax = plt.subplots(1, 3, figsize=(30, 7.6), dpi=55)
for a, M, ttl in ((ax[0], ALL["GT"][fi_pic], "real target T (exact GT points)"), (ax[1], ALL["A"][fi_pic], "A: true ends + true pairs, ONE generic Hermite curve each"),
                  (ax[2], ALL["B"][fi_pic], "B: true ends, ALL possible pairs (pairs unknown)")):
    a.imshow(M.reshape(100, 100).T, origin="lower", extent=(-50, 50, -50, 50), cmap="gray_r", vmin=0, vmax=1)
    a.set_xlim(-12, 52)
    a.set_ylim(-12, 28)
    a.set_aspect("equal")
    a.set_title(ttl)
    a.set_xlabel("x right (m)")
    a.set_ylabel("y forward (m)")
for pc in g["pieces"]:
    q = np.asarray(pc["pts"], float)
    ax[0].plot(q[:, 0], q[:, 1], "-", color="tab:red", lw=0.6, alpha=0.7)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "temp", "e284", "oracle_junction.png"), dpi=55)
print("written oracle_junction.png")
