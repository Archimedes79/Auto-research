"""scratch (Mirko's remark of 2026-11-08: the middle picture - the training target - is what the map should give, and before the junction the prediction deviates massively): the E279 centre map p (probability) against the
TRAINING TARGET T = exp(-d^2 / 2), d = cell centre to the nearest lane point (g05 of qraster.centre_points: all points of all pieces with |x|, |y| < 48 - exactly what grid_loss uses), as heat maps (not thresholded),
and the deviation by the distance of the cell to the junction (the connector points that the lane GT marks as inside an intersection, as in the earlier E283 / E284 scripts).
CPU, every 10th VAL frame. Writes temp/e284/target_vs_pred.npz (p, T, distance to the junction, distance to the lane, per frame), temp/e284/target_vs_pred.json and temp/e284/target_vs_pred.png (one frame, three panels).
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh temp/e284/target_vs_pred.py [--step 10] [--token <token>]"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation", "tools/viz"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qpredict as QP  # noqa: E402

STEP = int(sys.argv[sys.argv.index("--step") + 1]) if "--step" in sys.argv else 10
TOK = sys.argv[sys.argv.index("--token") + 1] if "--token" in sys.argv else "5dd954d0ac0242a990ba35eef950c937"
BINS = (("0-8 m (in the junction)", 0.0, 8.0), ("8-16 m", 8.0, 16.0), ("16-24 m", 16.0, 24.0), ("24-35 m", 24.0, 35.0), ("35 m and more", 35.0, 1e9))
c = -50.0 + (np.arange(100) + 0.5)
CX, CY = np.meshgrid(c, c, indexing="ij")
cells = np.stack([CX.ravel(), CY.ravel()], 1)
fn = QP.make_pred_fn("e279_w_s0", "exact", "cpu", keep_maps=True)
frames = QE.val_frames()
pick = frames[::STEP]
if TOK not in [f.token for f in pick]:
    pick = pick + [f for f in frames if f.token == TOK]


def target_of(f):
    g = f.lane()
    P = [np.asarray(p["pts"], float).reshape(-1, 2) for p in g.get("pieces", []) if len(p["pts"]) >= 2]
    P = np.concatenate(P) if P else np.zeros((0, 2))
    P = P[(np.abs(P) < 48.0).all(1)]
    d = cKDTree(P).query(cells)[0] if len(P) else np.full(len(cells), 1e9)
    return d, P


Ps, Ts, DJ, DG, toks = [], [], [], [], []
for k, f in enumerate(pick):
    pred = fn(f.token)
    p = np.asarray(pred["maps"]["centre1"][0], float).reshape(-1)
    d, _ = target_of(f)
    nt = f.ref().nt
    sp = nt["pts"][nt["supp"]]
    dj = cKDTree(sp).query(cells)[0] if len(sp) else np.full(len(cells), 1e9)
    Ps.append(p.astype(np.float16))
    Ts.append(np.exp(-d * d / 2.0).astype(np.float16))
    DJ.append(np.minimum(dj, 1e4).astype(np.float16))
    DG.append(np.minimum(d, 1e4).astype(np.float16))
    toks.append(f.token)
    f._lane = f._ref = None
    if (k + 1) % 20 == 0:
        print(k + 1, "frames", flush=True)
Ps, Ts, DJ, DG = [np.stack(a).astype(np.float32) for a in (Ps, Ts, DJ, DG)]
np.savez_compressed(os.path.join(ROOT, "temp", "e284", "target_vs_pred.npz"), p=Ps.astype(np.float16), t=Ts.astype(np.float16), dj=DJ.astype(np.float16), dg=DG.astype(np.float16), tokens=np.array(toks))

# ---- statistics: the frames of the step only (not the extra picture frame), by the distance of the cell to the junction
n_stat = len(frames[::STEP])
inbox = (np.abs(cells) <= 48.0).all(1)
res = {}
rows = [("no junction in the frame", None)] + [(b[0], b[1:]) for b in BINS]
for name, lim in rows:
    a = dict(hit=0.0, miss=0.0, false=0.0, abs_err=0.0, n_sig=0.0, far15=0.0, pred=0.0, gt_len=0.0, mean_t_on_pred=0.0)
    for i in range(n_stat):
        p, T, dj, dg = Ps[i], Ts[i], DJ[i], DG[i]
        if lim is None:
            m = (dj > 1e3) & inbox
        else:
            m = (dj >= lim[0]) & (dj < lim[1]) & inbox
        on_p, on_t = p >= 0.5, T >= 0.5
        a["hit"] += float((on_p & on_t & m).sum())
        a["miss"] += float((~on_p & on_t & m).sum())
        a["false"] += float((on_p & ~on_t & m).sum())
        sig = m & ((T > 0.05) | (p > 0.05))
        a["abs_err"] += float(np.abs(p - T)[sig].sum())
        a["n_sig"] += float(sig.sum())
        a["far15"] += float((on_p & m & (dg > 1.5)).sum())
        a["pred"] += float((on_p & m).sum())
        a["gt_len"] += 0.5 * float((m & (dg <= 0.5)).sum()) / 1.0     # about one 1 m cell per metre of lane: cells within 0.5 m of a lane point, per metre -> normalised below
    res[name] = dict(recall=a["hit"] / max(a["hit"] + a["miss"], 1), precision=a["hit"] / max(a["hit"] + a["false"], 1), iou=a["hit"] / max(a["hit"] + a["miss"] + a["false"], 1),
                     miss_cells=a["miss"], false_cells=a["false"], hit_cells=a["hit"], mean_abs_err=a["abs_err"] / max(a["n_sig"], 1), share_pred_farther_1_5=a["far15"] / max(a["pred"], 1),
                     pred_cells=a["pred"], target_cells=a["hit"] + a["miss"])
json.dump(dict(frames=n_stat, step=STEP, bins=res), open(os.path.join(ROOT, "temp", "e284", "target_vs_pred.json"), "w"), indent=1)
print("frames", n_stat)
for k, v in res.items():
    print(f"{k:26s} target cells {v['target_cells']:8.0f}  recall {100 * v['recall']:5.1f} %  precision {100 * v['precision']:5.1f} %  IoU {100 * v['iou']:5.1f} %  missing {v['miss_cells']:7.0f}  too many {v['false_cells']:7.0f}  mean |p - T| {v['mean_abs_err']:.3f}  predicted > 1.5 m from a lane point {100 * v['share_pred_farther_1_5']:.1f} %")

# ---- the picture of one frame
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

i = toks.index(TOK)
p, T, dj = Ps[i].reshape(100, 100), Ts[i].reshape(100, 100), DJ[i].reshape(100, 100)
f = [x for x in frames if x.token == TOK][0]
nt = f.ref().nt
sp = nt["pts"][nt["supp"]]
g = f.lane()
fig, ax = plt.subplots(1, 3, figsize=(24, 8.2), dpi=60)
for a, im, ttl, cm, vm in ((ax[0], T, "TARGET T = exp(-d^2 / 2)  (what the map should give)", "gray_r", (0, 1)), (ax[1], p, "E279 output p (probability, not thresholded)", "gray_r", (0, 1)),
                           (ax[2], p - T, "p - T   (red: too much, blue: missing)", "bwr", (-1, 1))):
    a.imshow(im.T, origin="lower", extent=(-50, 50, -50, 50), cmap=cm, vmin=vm[0], vmax=vm[1])
    a.set_xlim(-50, 50)
    a.set_ylim(-50, 50)
    a.set_aspect("equal")
    a.set_title(ttl)
    a.set_xlabel("x right (m)")
    a.set_ylabel("y forward (m)")
    if len(sp):
        a.plot(sp[:, 0], sp[:, 1], ".", color="tab:green", ms=1.5, alpha=0.5)
for pc in g["pieces"]:
    q = np.asarray(pc["pts"], float)
    ax[2].plot(q[:, 0], q[:, 1], "-", color="k", lw=0.5, alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "temp", "e284", "target_vs_pred.png"), dpi=60)
print("written target_vs_pred.png / .json / .npz")
