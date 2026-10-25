"""Ego-Posen je Sample fuer Karten-Vorwissen (E34) und Zeitverfolgung (E35).
Je Teil (train/val) in ~/exp_gcar_<tag>/poses_<part>.npz: t [N,2] globale Lage der Hinterachse zum CAM_FRONT-Zeitpunkt (m), R [N,2,2] mit  p_global = t + R @ p_fahrzeug  (Fahrzeug-System: x rechts, y vorn),
loc [N] Index in LOCS, scene [N] laufende Szenen-Nr. im Teil, prev [N] Zeile des Vorgaenger-Samples derselben Szene im selben Teil (-1 = keins), dt [N] Zeitabstand dazu (s), ts [N] Zeitstempel (s).
Aufruf: make_poses.py <b123|mini>   (Umgebung bev)"""
import json
import os
import sys

import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

HOME = os.path.expanduser("~")
CFG = {"b123": dict(root=f"{HOME}/data/nuscenes_full", version="v1.0-trainval", exp=f"{HOME}/exp_crn_b123", out=f"{HOME}/exp_gcar_b123"),
       "mini": dict(root=f"{HOME}/data/nuscenes", version="v1.0-mini", exp=f"{HOME}/exp_crn_mini", out=f"{HOME}/exp_gcar_mini")}
LOCS = ["boston-seaport", "singapore-onenorth", "singapore-hollandvillage", "singapore-queenstown"]


def main(tag):
    c = CFG[tag]
    nusc = NuScenes(version=c["version"], dataroot=c["root"], verbose=False)
    for part in ("train", "val"):
        toks = json.load(open(f"{c['exp']}/feats/{part}_tokens.json"))
        pos = {t: i for i, t in enumerate(toks)}
        N = len(toks)
        T, R = np.zeros((N, 2)), np.zeros((N, 2, 2))
        loc, scene, prev, dt, ts = np.zeros(N, int), np.zeros(N, int), -np.ones(N, int), np.zeros(N), np.zeros(N)
        sc_ids = {}
        for i, tk in enumerate(toks):
            s = nusc.get("sample", tk)
            sd = nusc.get("sample_data", s["data"]["CAM_FRONT"])
            ep = nusc.get("ego_pose", sd["ego_pose_token"])
            Re = Quaternion(ep["rotation"]).rotation_matrix[:2, :2]
            T[i] = ep["translation"][:2]
            R[i] = Re @ np.array([[0.0, 1.0], [-1.0, 0.0]])  # Fahrzeug (x rechts, y vorn) -> global
            sc = nusc.get("scene", s["scene_token"])
            loc[i] = LOCS.index(nusc.get("log", sc["log_token"])["location"])
            scene[i] = sc_ids.setdefault(s["scene_token"], len(sc_ids))
            ts[i] = sd["timestamp"] * 1e-6
        for i, tk in enumerate(toks):
            p = nusc.get("sample", tk)["prev"]
            if p and p in pos:
                prev[i] = pos[p]
                dt[i] = ts[i] - ts[prev[i]]
        np.savez(f"{c['out']}/poses_{part}.npz", t=T, R=R, loc=loc, scene=scene, prev=prev, dt=dt, ts=ts)
        print(part, N, "mit Vorgaenger", int((prev >= 0).sum()), "Szenen", len(sc_ids), "dt-Median", float(np.median(dt[prev >= 0])), flush=True)
        # Selbsttest: Soll-Graph-Punkt im Fahrzeug-System -> global -> zurueck
    print("fertig")


if __name__ == "__main__":
    main(sys.argv[1])
