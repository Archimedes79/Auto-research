"""Metrics for GRID heads (dense lane-centre-line output on the 200 x 200 BEV raster, 0.5 m cells): prediction map -> thin skeleton -> points in metres -> point P/R/F1.

Pipeline (one definition for every dense head of the project):
  prob map [N,200,200]   (heat map in 0..1)         -> skeleton_points(prob, tau)         threshold, Zhang-Suen thinning, drop components < 30 px
  distance field [S,200,200] in metres (E19 ff.)   -> skeletons(D, thr, blur, low, ...)   threshold on the distance, optional blur / hysteresis / range-dependent threshold
  points P vs dense ground-truth points G (m)      -> geom_counts(P, G)                  counts for P/R/F1 at 0.5/1.0/1.5 m + greedy 1:1 matches (graph_eval.match_1to1)
  eval_probs(prob, gts, taus)  = threshold sweep on a probability map;  est_eval(D, gts, thr, ...) = the same on distance fields (prf counts, no 1:1)
The cut-out functions are unchanged copies of scripts/dense_head.py, scripts/field_tools.py, scripts/heat_align.py and scripts/centerline_estimator.py
(the originals import them from here after the verified patch). Coordinates: row = x, column = y, x = -50 + (row + 0.5) * 0.5 m.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

import graph_eval as GE

HOME = __import__("os").path.expanduser("~")
dev = "cuda"
HALF = 50.0


def gauss_blur(H, sigma_px):
    """H [B,200,200] -> separable Gaussian blur (sigma in pixels, replicate padding). The kernel is normalised to sum 1, so the mean of H is kept
    and a narrow peak gets lower (a constant field is unchanged); the old docstring said the peak height is kept, which the code never did."""
    k = int(math.ceil(3 * sigma_px))
    x = torch.arange(-k, k + 1, device=H.device).float()
    g = torch.exp(-x * x / (2 * sigma_px**2))
    g = g / g.sum()
    h = H[:, None]
    h = F.conv2d(F.pad(h, (k, k, 0, 0), mode="replicate"), g.view(1, 1, 1, -1))
    h = F.conv2d(F.pad(h, (0, 0, k, k), mode="replicate"), g.view(1, 1, -1, 1))
    return h[:, 0]


def prf(P, G, taus=(0.5, 1.0, 1.5)):
    """Zaehler fuer Praezision/Ausbeute bei Toleranzen. P,G: Punktmengen [n,2] (m)."""
    if len(P) == 0 or len(G) == 0:
        return {t: (0, len(P), 0, len(G)) for t in taus}
    D = torch.cdist(torch.from_numpy(P).float(), torch.from_numpy(G).float())
    dp, dg = D.min(1).values, D.min(0).values
    return {t: (int((dp <= t).sum()), len(P), int((dg <= t).sum()), len(G)) for t in taus}


def thin_batch(mask):
    """Zhang-Suen fuer bool [B,H,W] (GPU), wie field_tools.thin_batch."""
    img = mask.float()

    def step(img, first):
        p = F.pad(img, (1, 1, 1, 1))
        P2, P3, P4, P5 = p[:, :-2, 1:-1], p[:, :-2, 2:], p[:, 1:-1, 2:], p[:, 2:, 2:]
        P6, P7, P8, P9 = p[:, 2:, 1:-1], p[:, 2:, :-2], p[:, 1:-1, :-2], p[:, :-2, :-2]
        seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
        Bn = sum(seq[:8])
        An = sum(((seq[i] == 0) & (seq[i + 1] == 1)).float() for i in range(8))
        c = (Bn >= 2) & (Bn <= 6) & (An == 1)
        c = c & (((P2 * P4 * P6) == 0) & ((P4 * P6 * P8) == 0)) if first else c & (((P2 * P4 * P8) == 0) & ((P2 * P6 * P8) == 0))
        return img * (~(img.bool() & c)).float()

    for _ in range(60):
        new = step(step(img, True), False)
        if torch.equal(new, img):
            break
        img = new
    return img.bool()


def skeleton_points(prob, tau, min_comp=30):
    """prob [B,200,200] (GPU) -> Liste von Punktmengen (m, Fahrzeug-System) je Sample."""
    from scipy import ndimage as ndi

    sk = thin_batch(prob >= tau).cpu().numpy()
    out = []
    for b in range(sk.shape[0]):
        s = sk[b]
        if min_comp > 0:
            lab, n = ndi.label(s, structure=np.ones((3, 3)))
            if n:
                sizes = ndi.sum(s, lab, range(1, n + 1))
                s = np.isin(lab, 1 + np.nonzero(sizes >= min_comp)[0])
        P = np.argwhere(s).astype(np.float32)
        P = np.stack([-50 + (P[:, 0] + 0.5) * 0.5, -50 + (P[:, 1] + 0.5) * 0.5], 1)
        out.append(P[(np.abs(P) < 48).all(1)])
    return out


def geom_counts(P, G):
    from scipy.spatial import cKDTree

    c = np.zeros((3, 4))
    m1 = np.zeros(3)
    if len(P) and len(G):
        tG = cKDTree(G)
        dp = tG.query(P)[0]
        dg = cKDTree(P).query(G)[0]
        for k, t in enumerate(GE.TAUS):
            c[k] = (np.sum(dp <= t), len(P), np.sum(dg <= t), len(G))
            m1[k] = GE.match_1to1(P, G, tG, t)
    else:
        c[:, 1], c[:, 3] = len(P), len(G)
    return c, m1


def eval_probs(prob, gts, taus=(0.3, 0.4, 0.5, 0.6, 0.7), dev="cuda"):
    """prob [N,200,200] (cpu half) gegen dichte Soll-Punkte gts (Liste) -> {tau: {P,R,F1,F1o2o}} bei 1 m plus F1 bei 0,5/1,5 m."""
    res = {}
    for tau in taus:
        C, M1 = np.zeros((3, 4)), np.zeros(3)
        for i in range(0, len(prob), 32):
            pts = skeleton_points(prob[i : i + 32].float().to(dev), tau)
            for k, P in enumerate(pts):
                c, m = geom_counts(P, gts[i + k])
                C += c
                M1 += m
        r = {}
        for k, t in enumerate(GE.TAUS):
            pr, rc = C[k, 0] / max(C[k, 1], 1), C[k, 2] / max(C[k, 3], 1)
            r[f"P@{t}"], r[f"R@{t}"], r[f"F1@{t}"] = float(pr), float(rc), float(2 * pr * rc / max(pr + rc, 1e-9))
            r[f"F1o2o@{t}"] = float(2 * M1[k] / max(C[k, 1] + C[k, 3], 1))
        r["F0.5@1.0"] = float(1.25 * r["P@1.0"] * r["R@1.0"] / max(0.25 * r["P@1.0"] + r["R@1.0"], 1e-9))
        res[str(tau)] = r
    return res


def load_gt_points(split, part, idx):
    """dichte exakte Soll-Punkte (Fahrzeug-System) der Samples idx."""
    a = np.load(f"{HOME}/exp_crn_{split}/feats/{part}_gt_pts_exact.npy", allow_pickle=True)
    return [a[int(i)].astype(np.float32) for i in idx]


def skeletons(D, thr=0.5, blur=0.0, low=None, min_comp=0, bs=16, rg=0.0):
    """Felder (m) -> duenne Mittellinien-Pixel bool [S,200,200] (cpu numpy). rg: Entfernungsanstieg der Schwelle thr * (1 + rg * r / 48 m)
    (Felder sind in der Ferne unschaerfer; r = Abstand vom Raster-Ursprung)."""
    from scipy import ndimage as ndi

    Dt = torch.as_tensor(D)
    out = np.zeros((Dt.shape[0], 200, 200), bool)
    c = -HALF + (torch.arange(200, device=dev).float() + 0.5) * 0.5
    rad = torch.sqrt(c[:, None] ** 2 + c[None, :] ** 2)  # Zeile = x, Spalte = y
    thr_map = thr * (1.0 + rg * rad / 48.0)
    for i in range(0, Dt.shape[0], bs):
        d = Dt[i : i + bs].float().to(dev)
        db = gauss_blur(d, blur) if blur > 0 else d
        blob = db <= thr_map[None]
        if low is not None and low > thr:
            lowm = db <= torch.clamp(thr_map, min=low)[None]  # grow mask >= seed threshold per pixel, so a far seed (thr_map > low when rg > 0) always survives (review bug L3, D24)
            m = blob
            for _ in range(80):
                m2 = (F.max_pool2d(m.float()[:, None], 3, 1, 1)[:, 0] > 0) & lowm
                if torch.equal(m2, m):
                    break
                m = m2
            blob = m
        sk = thin_batch(blob).cpu().numpy()
        if min_comp > 0:
            for b in range(sk.shape[0]):
                lab, n = ndi.label(sk[b], structure=np.ones((3, 3)))
                if n:
                    sizes = ndi.sum(sk[b], lab, range(1, n + 1))
                    keep = np.isin(lab, 1 + np.nonzero(sizes >= min_comp)[0])
                    sk[b] = keep
        out[i : i + bs] = sk
    return out


def est_eval(D, gts, thr=0.5, blur=0.0, low=None, min_comp=0, sub=1, rg=0.0):
    """Praezision/Ausbeute/F1 bei 0,5/1,0/1,5 m (Summen ueber Samples wie CenterlineEstimator-Auswertung). sub: nur jedes sub-te Sample."""
    idx = np.arange(0, len(gts), sub)
    sk = skeletons(np.asarray(D)[idx] if not torch.is_tensor(D) else D[idx], thr, blur, low, min_comp, rg=rg)
    tot = {t: np.zeros(4) for t in (0.5, 1.0, 1.5)}
    for b, i in enumerate(idx):
        P = np.argwhere(sk[b]).astype(np.float32)
        P = np.stack([-50 + (P[:, 0] + 0.5) * 0.5, -50 + (P[:, 1] + 0.5) * 0.5], 1)
        P = P[(np.abs(P) < 48).all(1)]
        for t, v in prf(P, gts[i]).items():
            tot[t] += np.array(v, float)
    res = {}
    for t, (tp_p, n_p, tp_g, n_g) in tot.items():
        pr, rc = tp_p / max(n_p, 1), tp_g / max(n_g, 1)
        res[t] = (pr, rc, 2 * pr * rc / max(pr + rc, 1e-9))
    return res
