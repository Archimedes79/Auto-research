"""Werkzeuge fuer Auswertungen auf vorhergesagten Abstandsfeldern (Meter, [S,200,200]) ohne Netz im Speicher.

 - predict_fields: Netz -> Abstandsfeld je Validierungs-Sample (gecacht)
 - load_poses:     Ego-Posen im Raster-System (Hinterachse, Achsen wie Lidar-System: x rechts, y vorn), Vorgaenger-Frames, Zukunftspfad
 - warp / fuse:    Felder fruehere Frames per Ego-Bewegung in den aktuellen Frame abbilden und gewichtet mitteln (kausal)
 - thin_batch:     Zhang-Suen-Ausduennen fuer ganze Batches (gleiches Ergebnis wie CenterlineEstimator.thin, viel schneller)
 - est_eval:       Schaetzer-Auswertung (Praezision/Ausbeute/F1 bei 0,5/1,0/1,5 m) auf (fusionierten) Feldern
 - route_labels:   Knoten auf dem tatsaechlich gefahrenen Pfad der naechsten Sekunden ("Navigationsziel"-Ersatz)
 - lane_assoc:     Spurzuordnung der ausgerichteten Karten-Knoten (liegt der Knoten auf seiner eigenen Spur?)
"""
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from pyquaternion import Quaternion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import centerline_estimator as CE  # noqa: E402
import heat_align as H  # noqa: E402

dev = "cuda"
HALF = 50.0


def load_graph(gdir, split="val"):
    g = np.load(f"{gdir}/{split}_graph.npz")
    return {k: g[k] for k in g.files}


@torch.no_grad()
def predict_fields(net, feats_path, bs=16, cache=None):
    """Abstand zur naechsten Mittellinie in m (bei 4 m gekappt) fuer alle Samples; fp16 [S,200,200]."""
    if cache and os.path.exists(cache):
        return np.load(cache)
    feats = np.load(feats_path, mmap_mode="r")
    out = np.zeros((feats.shape[0], 200, 200), np.float16)
    net.eval()
    for i in range(0, feats.shape[0], bs):
        f = torch.from_numpy(np.ascontiguousarray(feats[i : i + bs])).to(dev)
        out[i : i + bs] = (torch.sigmoid(net(f, None)).float() * 4.0).cpu().numpy().astype(np.float16)  # sigmoid = Abstand/8 px; 8 px = 4 m
    if cache:
        np.save(cache, out)
    return out


# ------------------------------------------------------------------ Posen, Zeit

R_L = np.array([[0.0, 1.0], [-1.0, 0.0]])


def load_poses(infos_pkl, tokens, n_prev=4, n_future=6, frame="ego"):
    """Posen im Raster-System. Rueckgabe dict: T [S,2], R [S,2,2] (p_L = (p - T) @ R), prev [S,n_prev] (Zeile in 'tokens' oder -1),
    fut [S,n_future,2] (Zukunftspositionen im Raster-System des Samples), fut_ok [S,n_future].
    frame='ego': Hinterachse zum Kamera-Zeitpunkt (Kamera-Radar); frame='lidar': Lidar-Sensor (Vergleichsrechnung mit BEVFusion; nur Metadaten)."""
    # Zwischenspeicher: die CRN-Infos belegen beim Laden mehrere GB RAM; die kleinen Pose-Arrays werden einmal berechnet und wiederverwendet
    cache = f"/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/_loop1_crn/poses_{frame}_{len(tokens)}_{tokens[0][:8]}_{tokens[-1][:8]}_{n_prev}_{n_future}.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    infos = pickle.load(open(infos_pkl, "rb"))
    row = {t: i for i, t in enumerate(tokens)}
    kidx = {inf["sample_token"]: k for k, inf in enumerate(infos)}
    S = len(tokens)
    T, R = np.zeros((S, 2)), np.zeros((S, 2, 2))
    for i, t in enumerate(tokens):
        if frame == "lidar":
            li = infos[kidx[t]]["lidar_infos"]["LIDAR_TOP"]
            ep, cs = li["ego_pose"], li["calibrated_sensor"]
            e2g = np.eye(4)
            e2g[:3, :3], e2g[:3, 3] = Quaternion(ep["rotation"]).rotation_matrix, ep["translation"]
            l2e = np.eye(4)
            l2e[:3, :3], l2e[:3, 3] = Quaternion(cs["rotation"]).rotation_matrix, cs["translation"]
            l2g = e2g @ l2e
            T[i], R[i] = l2g[:2, 3], l2g[:2, :2]
            continue
        ep = infos[kidx[t]]["cam_infos"]["CAM_FRONT"]["ego_pose"]
        T[i] = ep["translation"][:2]
        R[i] = Quaternion(ep["rotation"]).rotation_matrix[:2, :2] @ R_L
    prev = -np.ones((S, n_prev), np.int64)
    fut = np.zeros((S, n_future, 2))
    fut_ok = np.zeros((S, n_future), bool)
    for i, t in enumerate(tokens):
        k = kidx[t]
        sc = infos[k]["scene_token"]
        for m in range(1, n_prev + 1):
            if k - m >= 0 and infos[k - m]["scene_token"] == sc and infos[k - m]["sample_token"] in row:
                prev[i, m - 1] = row[infos[k - m]["sample_token"]]
        for m in range(1, n_future + 1):
            if k + m < len(infos) and infos[k + m]["scene_token"] == sc and infos[k + m]["sample_token"] in row:
                j = row[infos[k + m]["sample_token"]]
                fut[i, m - 1] = (T[j] - T[i]) @ R[i]
                fut_ok[i, m - 1] = True
    out = dict(T=T, R=R, prev=prev, fut=fut, fut_ok=fut_ok)
    np.savez(cache, **out)
    del infos
    return out


_CEN = None


def _grid_pts():
    global _CEN
    if _CEN is None:
        c = -HALF + (torch.arange(200, device=dev).float() + 0.5) * 0.5
        gx, gy = torch.meshgrid(c, c, indexing="ij")  # Zeile = x, Spalte = y
        _CEN = torch.stack([gx, gy], -1).reshape(-1, 2)
    return _CEN


@torch.no_grad()
def warp_fields(Dj, Ti, Ri, Tj, Rj):
    """Felder Dj [B,200,200] (Frame j) in den Frame i abbilden. T [B,2], R [B,2,2] (p_L = (p - T) @ R). -> (Feld, gueltig)."""
    pL = _grid_pts()[None].expand(Dj.shape[0], -1, -1)  # [B,40000,2]
    pg = Ti[:, None] + torch.einsum("bij,bnj->bni", Ri, pL)  # Welt
    pj = torch.einsum("bni,bij->bnj", pg - Tj[:, None], Rj)  # im Frame j
    grid = torch.stack([pj[..., 1], pj[..., 0]], -1).reshape(-1, 200, 200, 2) / HALF  # (Spalte = y, Zeile = x)
    w = F.grid_sample(Dj[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]
    ok = F.grid_sample(torch.ones_like(Dj)[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0] > 0.999
    return w, ok


@torch.no_grad()
def fuse_fields(D, poses, K, decay=0.6, bs=16, tau=None):
    """Kausale Zeitfusion: gewichtetes Mittel des aktuellen und der K vorherigen Felder (Gewicht decay^k), jeweils per Ego-Bewegung abgebildet.
    tau (m): statt des Mittels ein gewichtetes Soft-Minimum  -tau * log(sum w exp(-d/tau) / sum w)  (tau -> 0: Minimum, tau -> inf: Mittel);
    das Minimum bewahrt eine Linie, die nur in einem Frame sichtbar war (Ausbeute), das Mittel verwischt sie.
    D [S,200,200] (np/torch), Rueckgabe torch cpu float16."""
    Dt = torch.as_tensor(D)
    S = Dt.shape[0]
    T = torch.from_numpy(poses["T"]).float().to(dev)
    R = torch.from_numpy(poses["R"]).float().to(dev)
    out = torch.zeros(S, 200, 200, dtype=torch.float16)
    tr = (lambda d: torch.exp(-d / tau)) if tau else (lambda d: d)
    for i0 in range(0, S, bs):
        idx = torch.arange(i0, min(i0 + bs, S))
        cur = Dt[idx].float().to(dev)
        num, den = tr(cur).clone(), torch.ones_like(cur)
        for k in range(1, K + 1):
            j = torch.from_numpy(poses["prev"][idx.numpy(), k - 1])
            has = j >= 0
            if not has.any():
                continue
            jj = j.clamp(min=0)
            w, ok = warp_fields(Dt[jj].float().to(dev), T[idx], R[idx], T[jj], R[jj])
            wt = (decay**k) * (ok & has.to(dev)[:, None, None]).float()
            num += wt * tr(w)
            den += wt
        m = num / den
        out[idx] = ((-tau * torch.log(m.clamp(min=1e-12))) if tau else m).clamp(0, 4).half().cpu()
    return out


# ------------------------------------------------------------------ Ausduennen, Schaetzer-Auswertung


from grid_eval import est_eval, skeletons, thin_batch  # noqa: E402,F401  (moved to evaluation/lane_grid)


# ------------------------------------------------------------------ Route, Spurzuordnung


def route_labels(graph, poses, half_width=2.0, ahead=-2.0):
    """Knoten (wahre Lage) auf dem gefahrenen Pfad der naechsten Samples (Ersatz fuer ein Navigationsziel aus der Karte). bool [S,N]."""
    nodes, mask = graph["nodes"], graph["mask"]
    S, N, _ = nodes.shape
    out = np.zeros((S, N), bool)
    for i in range(S):
        pts = [np.zeros(2)] + [poses["fut"][i, m] for m in range(poses["fut"].shape[1]) if poses["fut_ok"][i, m]]
        if len(pts) < 2:
            continue
        path = []
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(int(np.linalg.norm(b - a) / 0.5), 1)
            path.append(a + (b - a) * (np.arange(n)[:, None] / n))
        path = np.concatenate(path + [pts[-1][None]])
        d = np.linalg.norm(nodes[i][:, None] - path[None], axis=-1).min(1)
        out[i] = mask[i] & (d <= half_width) & (nodes[i][:, 1] >= ahead)
    return out


@torch.no_grad()
def lane_assoc(q, nodes, poly, mask, route, tie=0.25):
    """q: ausgerichtete (oder verdorbene) Knoten [B,N,2]; nodes: wahre Lage; poly [B,N] Spur-Index; mask, route [B,N] (bool; torch, GPU).
    Ein Knoten gilt als richtig zugeordnet, wenn seine naechste wahre Spur seine eigene ist (oder die eigene nur um <= tie m schlechter).
    -> (Genauigkeit alle, Genauigkeit Routenknoten, Anzahl alle, Anzahl Route)."""
    a, b = nodes[:, :-1], nodes[:, 1:]
    seg_ok = (poly[:, :-1] == poly[:, 1:]) & mask[:, :-1] & mask[:, 1:] & ((a - b).norm(dim=-1) < 6.0)
    seg_lane = poly[:, :-1]
    ab = b - a
    ap = q[:, :, None] - a[:, None]  # [B,N,M,2]
    tt = ((ap * ab[:, None]).sum(-1) / (ab * ab).sum(-1)[:, None].clamp(min=1e-6)).clamp(0, 1)
    proj = a[:, None] + tt[..., None] * ab[:, None]
    d = (q[:, :, None] - proj).norm(dim=-1)
    d = torch.where(seg_ok[:, None], d, torch.full_like(d, 1e9))
    dmin, am = d.min(-1)
    lane_pred = seg_lane.gather(1, am)
    own = (seg_lane[:, None] == poly[:, :, None]) & seg_ok[:, None]
    d_own = torch.where(own, d, torch.full_like(d, 1e9)).min(-1).values
    usable = mask & (d_own < 1e8)
    correct = ((lane_pred == poly) | (d_own <= dmin + tie)) & usable
    ra = usable & route
    return correct.sum().item(), correct[route & usable].sum().item(), usable.sum().item(), ra.sum().item()
