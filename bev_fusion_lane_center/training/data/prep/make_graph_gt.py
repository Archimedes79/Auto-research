"""Soll-Graph der Spurmitten je Sample (E27 ff.): Spur-Polylinien im Fahrzeug-System (Hinterachse zum CAM_FRONT-Zeitpunkt; x nach rechts, y nach vorn, +-50 m) mit Verbindungen.

Quelle: nuScenes-Karte (Spuren 'lane' + Spurverbinder 'lane_connector', Mittellinien als Kreisbogen-Pfade, auf 0,5 m abgetastet) und deren Konnektivitaet ('outgoing'/'incoming' je Spur).
Je Spur werden die Stuecke innerhalb des Fensters behalten (Reihenfolge = Fahrtrichtung); Kanten i -> j nur, wenn das Ende von i und der Anfang von j im Fenster liegen
(offene Enden am Fensterrand tragen keine Kante). Splits = Spur mit >= 2 Nachfolgern, Merges = Spur mit >= 2 Vorgaengern.
Ausgabe: <exp>/graph_gt_<part>.pkl (Liste je Sample von dict(tokens, kind, pts, start_open, end_open, edges)); Aufruf: make_graph_gt.py <b123|mini>
"""
import json
import os
import pickle
import sys
import time

import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

HOME = os.path.expanduser("~")
CFG = {"b123": dict(root=f"{HOME}/data/nuscenes_full", version="v1.0-trainval", exp=f"{HOME}/exp_crn_b123", out=f"{HOME}/exp_gcar_b123"),
       "mini": dict(root=f"{HOME}/data/nuscenes", version="v1.0-mini", exp=f"{HOME}/exp_crn_mini", out=f"{HOME}/exp_gcar_mini")}
STEP, HALF, MARGIN = 0.5, 50.0, 75.0


def load_map(root, loc):
    m = NuScenesMap(dataroot=root, map_name=loc)
    toks = [r["token"] for r in m.lane] + [r["token"] for r in m.lane_connector]
    kind = {r["token"]: "lane" for r in m.lane}
    kind.update({r["token"]: "connector" for r in m.lane_connector})
    disc = m.discretize_lanes(toks, STEP)  # token -> [(x, y, yaw)] in Fahrtrichtung
    polys = {t: np.asarray(v, np.float64)[:, :2] for t, v in disc.items() if len(v) >= 2}
    tl = list(polys)
    bb = np.array([[polys[t][:, 0].min(), polys[t][:, 1].min(), polys[t][:, 0].max(), polys[t][:, 1].max()] for t in tl])
    conn = {t: (list(m.connectivity.get(t, {}).get("incoming", [])), list(m.connectivity.get(t, {}).get("outgoing", []))) for t in tl}
    return dict(polys=polys, tokens=tl, bbox=bb, conn=conn, kind=kind)


def ego_pose(nusc, sample):
    sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    Re = Quaternion(ep["rotation"]).rotation_matrix[:2, :2]
    return np.asarray(ep["translation"][:2], np.float64), Re @ np.array([[0.0, 1.0], [-1.0, 0.0]])


def sample_graph(M, t, R):
    near = (M["bbox"][:, 2] > t[0] - MARGIN) & (M["bbox"][:, 0] < t[0] + MARGIN) & (M["bbox"][:, 3] > t[1] - MARGIN) & (M["bbox"][:, 1] < t[1] + MARGIN)
    toks, kinds, pts, so, eo, owner = [], [], [], [], [], []
    first_piece, last_piece = {}, {}
    for k in np.nonzero(near)[0]:
        tk = M["tokens"][k]
        p = (M["polys"][tk] - t) @ R
        inside = (np.abs(p[:, 0]) <= HALF) & (np.abs(p[:, 1]) <= HALF)
        if not inside.any():
            continue
        idx = np.nonzero(inside)[0]
        runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        for r in runs:
            if len(r) < 4:  # < 2 m
                continue
            piece = p[r[0] : r[-1] + 1].astype(np.float32)
            s_open, e_open = r[0] > 0, r[-1] < len(p) - 1
            owner.append(tk)
            toks.append(tk)
            kinds.append(M["kind"][tk])
            pts.append(piece)
            so.append(bool(s_open))
            eo.append(bool(e_open))
            n = len(pts) - 1
            if not s_open:
                first_piece[tk] = n
            if not e_open:
                last_piece[tk] = n
    edges = []
    for tk, i in last_piece.items():
        for succ in M["conn"][tk][1]:
            j = first_piece.get(succ)
            if j is not None and j != i:
                edges.append((i, j))
    return dict(tokens=toks, kind=kinds, pts=pts, start_open=so, end_open=eo, edges=edges)
def main(tag):
    c = CFG[tag]
    t0 = time.time()
    nusc = NuScenes(version=c["version"], dataroot=c["root"], verbose=False)
    print(f"nuScenes geladen ({time.time() - t0:.0f} s)", flush=True)
    maps = {}
    for part in ("train", "val"):
        toks = json.load(open(f"{c['exp']}/feats/{part}_tokens.json"))
        out = []
        for i, tk in enumerate(toks):
            s = nusc.get("sample", tk)
            loc = nusc.get("log", nusc.get("scene", s["scene_token"])["log_token"])["location"]
            if loc not in maps:
                maps[loc] = load_map(c["root"], loc)
                print("Karte", loc, len(maps[loc]["tokens"]), "Spuren/Verbinder", flush=True)
            t, R = ego_pose(nusc, s)
            out.append(sample_graph(maps[loc], t, R))
            if i % 1000 == 0:
                print(part, i, len(toks), f"{time.time() - t0:.0f} s", flush=True)
        os.makedirs(c["out"], exist_ok=True)
        pickle.dump(out, open(f"{c['out']}/graph_gt_{part}.pkl", "wb"))
        n_l = np.mean([len(g["pts"]) for g in out])
        n_e = np.mean([len(g["edges"]) for g in out])
        spl = np.mean([sum(1 for x in np.bincount([e[0] for e in g["edges"]], minlength=len(g["pts"])) if x >= 2) for g in out])
        mrg = np.mean([sum(1 for x in np.bincount([e[1] for e in g["edges"]], minlength=len(g["pts"])) if x >= 2) for g in out])
        print(f"{part}: {len(out)} Samples, Spuren-Stuecke je Sample {n_l:.1f}, Kanten {n_e:.1f}, Splits {spl:.2f}, Merges {mrg:.2f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
