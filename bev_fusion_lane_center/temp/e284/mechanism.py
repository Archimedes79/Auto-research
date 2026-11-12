"""scratch (Mirko's question of 2026-11-08: how do the smear and the twist inside the junction come about?). Three tests on the E279 centre map p against the training target T, junction cells (< 8 m from the junction points):
  range    does the misfit grow with the range of the cell from the ego (the BEV features get worse far away) or stay the same (the target itself is ambiguous)?
  train    the same measures on 88 TRAIN frames (seed 0, no augmentation): is the misfit there too (the head cannot fit it: capacity / information / ambiguity) or only on VAL (it does not generalise)?
  seeds    seed 1 on the same VAL frames: do the two seeds fill the same empty cells (systematic: something both learned) or different ones (noise)?
Measures: corr(p, T) over the cells with p > 0.05 or T > 0.05; the 'empty-cell fill' = share of cells with p > 0.3 among the cells the target leaves empty (T < 0.1) within 6 m of a lane point; mean |p - T|.
Reads temp/e284/target_vs_pred.npz (VAL p of seed 0, T, distances), runs seed 1 on VAL and seed 0 on TRAIN on the CPU, writes temp/e284/mechanism.npz and .json.
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh temp/e284/mechanism.py"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation", "tools/viz"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_gt_data as LD  # noqa: E402
import qpredict as QP  # noqa: E402
import qtargets as QT  # noqa: E402

c = -50.0 + (np.arange(100) + 0.5)
CX, CY = np.meshgrid(c, c, indexing="ij")
cells = np.stack([CX.ravel(), CY.ravel()], 1)
RANGE = np.linalg.norm(cells, axis=1)
inbox = (np.abs(cells) <= 47.0).all(1)

z = np.load(os.path.join(ROOT, "temp", "e284", "target_vs_pred.npz"))
toks = [str(t) for t in z["tokens"]][:88]
P0, T0, DJ0, DG0 = (z[k].astype(np.float32)[:88] for k in ("p", "t", "dj", "dg"))


def gt_arrays(g):
    pcs = [np.asarray(p["pts"], float).reshape(-1, 2) for p in g.get("pieces", []) if len(p["pts"]) >= 2]
    pts = np.concatenate(pcs)
    pts = pts[(np.abs(pts) < 48.0).all(1)]
    d = cKDTree(pts).query(cells)[0]
    nt = QT.node_table(g, sparse=None)
    sp = nt["pts"][nt["supp"]]
    dj = cKDTree(sp).query(cells)[0] if len(sp) else np.full(len(cells), 1e4)
    return np.exp(-d * d / 2.0).astype(np.float32), np.minimum(d, 1e4).astype(np.float32), np.minimum(dj, 1e4).astype(np.float32)


def stats(p, T, DG, DJ, sel=None):
    """pooled over frames; sel(cells mask per frame) optional extra mask (n_cells,)"""
    a = dict(n=0, sp=0.0, st=0.0, spp=0.0, stt=0.0, spt=0.0, abs=0.0, gap=0, gap_hi=0)
    for i in range(len(p)):
        m = (DJ[i] < 8.0) & inbox & ((p[i] > 0.05) | (T[i] > 0.05))
        mg = (DJ[i] < 8.0) & inbox & (T[i] < 0.1) & (DG[i] < 6.0)
        if sel is not None:
            m &= sel
            mg &= sel
        a["n"] += int(m.sum())
        a["sp"] += float(p[i][m].sum())
        a["st"] += float(T[i][m].sum())
        a["spp"] += float((p[i][m] ** 2).sum())
        a["stt"] += float((T[i][m] ** 2).sum())
        a["spt"] += float((p[i][m] * T[i][m]).sum())
        a["abs"] += float(np.abs(p[i][m] - T[i][m]).sum())
        a["gap"] += int(mg.sum())
        a["gap_hi"] += int((p[i][mg] > 0.3).sum())
    n = max(a["n"], 1)
    cov = a["spt"] / n - a["sp"] / n * a["st"] / n
    vp, vt = a["spp"] / n - (a["sp"] / n) ** 2, a["stt"] / n - (a["st"] / n) ** 2
    return dict(cells=a["n"], corr=float(cov / np.sqrt(max(vp * vt, 1e-12))), mean_abs=a["abs"] / n, gap_cells=a["gap"], gap_fill=a["gap_hi"] / max(a["gap"], 1))


res = {}
# ---- range (VAL, seed 0)
res["range_val_s0"] = {}
for name, a, b in (("0-16 m", 0, 16), ("16-32 m", 16, 32), ("32-47 m", 32, 47.01)):
    res["range_val_s0"][name] = stats(P0, T0, DG0, DJ0, sel=(RANGE >= a) & (RANGE < b))
res["val_s0_all"] = stats(P0, T0, DG0, DJ0)
print("VAL seed 0 all:", json.dumps(res["val_s0_all"]))
for k, v in res["range_val_s0"].items():
    print("  range", k, json.dumps(v))

# ---- seed 1 on VAL
fn1 = QP.make_pred_fn("e279_w_s1", "exact", "cpu", keep_maps=True)
P1 = np.stack([np.asarray(fn1(t)["maps"]["centre1"][0], np.float32).reshape(-1) for t in toks])
print("seed 1 on VAL done", flush=True)
res["val_s1_all"] = stats(P1, T0, DG0, DJ0)
print("VAL seed 1 all:", json.dumps(res["val_s1_all"]))
# agreement between the seeds in the junction cells
m_all = [(DJ0[i] < 8.0) & inbox & ((P0[i] > 0.05) | (P1[i] > 0.05) | (T0[i] > 0.05)) for i in range(88)]
a0 = np.concatenate([P0[i][m_all[i]] for i in range(88)])
a1 = np.concatenate([P1[i][m_all[i]] for i in range(88)])
at = np.concatenate([T0[i][m_all[i]] for i in range(88)])
gap_all = [(DJ0[i] < 8.0) & inbox & (T0[i] < 0.1) & (DG0[i] < 6.0) for i in range(88)]
g0 = np.concatenate([P0[i][gap_all[i]] for i in range(88)])
g1 = np.concatenate([P1[i][gap_all[i]] for i in range(88)])
res["seed_agreement_junction"] = dict(corr_s0_s1=float(np.corrcoef(a0, a1)[0, 1]), corr_s0_T=float(np.corrcoef(a0, at)[0, 1]), corr_s1_T=float(np.corrcoef(a1, at)[0, 1]),
                                      gap_fill_s0=float((g0 > 0.3).mean()), gap_fill_s1=float((g1 > 0.3).mean()), gap_s1_hi_given_s0_hi=float((g1[g0 > 0.3] > 0.3).mean()), gap_corr_s0_s1=float(np.corrcoef(g0, g1)[0, 1]),
                                      mean_abs_s0_s1=float(np.abs(a0 - a1).mean()), mean_abs_s0_T=float(np.abs(a0 - at).mean()))
print("seeds:", json.dumps(res["seed_agreement_junction"]))

# ---- seed 0 on TRAIN (no augmentation: the loader reads stored features, nothing is random here)
m0 = QP.load_model(os.path.join(ROOT, "runs", "e279_w_s0"), "cpu")
ds = LD.QData("train", map_mode="exact", pixels=False)
step = max(1, len(ds.ks) // 88)
idx = list(range(0, len(ds.ks), step))[:88]
Pt, Tt, DGt, DJt = [], [], [], []
for k, i in enumerate(idx):
    item = ds[i]
    b = LD.collate([item])
    pred = QP.predict(m0, b, keep_maps=True)[0]
    Pt.append(np.asarray(pred["maps"]["centre1"][0], np.float32).reshape(-1))
    t, dg, dj = gt_arrays(ds.lg.load(item["token"]))
    Tt.append(t)
    DGt.append(dg)
    DJt.append(dj)
    if (k + 1) % 20 == 0:
        print(k + 1, "train frames", flush=True)
Pt, Tt, DGt, DJt = (np.stack(a) for a in (Pt, Tt, DGt, DJt))
res["train_s0_all"] = stats(Pt, Tt, DGt, DJt)
res["train_frames"] = len(idx)
print("TRAIN seed 0 all:", json.dumps(res["train_s0_all"]))
np.savez_compressed(os.path.join(ROOT, "temp", "e284", "mechanism.npz"), p1=P1.astype(np.float16), pt=Pt.astype(np.float16), tt=Tt.astype(np.float16), dgt=DGt.astype(np.float16), djt=DJt.astype(np.float16))
json.dump(res, open(os.path.join(ROOT, "temp", "e284", "mechanism.json"), "w"), indent=1)
print("written mechanism.json")
