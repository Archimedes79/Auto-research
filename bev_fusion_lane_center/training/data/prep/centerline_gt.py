"""Spurmitten (Mittellinien) aus der nuScenes-Karte als BEV-Raster erzeugen.

Das ist die "Soll-Datenbasis" (Ground Truth) fuer das Experiment: Fuer jeden
Keyframe wird die Mittellinie aller Spuren und Spurverbinder in das BEV-Gitter
um den LIDAR_TOP-Sensor gezeichnet (Standard: +-50 m, 0.5 m Zellen = 200x200, wie bei BEVFusion).

Konvention des Rasters (= BEVFusion-Labels, mit orient_check.py gegen die Original-Routine
LoadBEVSegmentation gemessen: 100 % der Mittellinien liegen in dessen 'drivable_area'):
                         Zeile  = (x_lidar + half)/res
                         Spalte = (y_lidar + half)/res
x_lidar zeigt nach rechts vom Auto, y_lidar nach vorn (Lidar-System, ca. 0.94 m vor der Hinterachse).

Aufruf in WSL (Conda-Umgebung mit nuscenes-devkit, opencv-python, pyquaternion):
    python centerline_gt.py --dataroot <problem folder>/data/nuscenes \
        --version v1.0-mini --out <repository folder>/out/centerline_gt --debug 8

STATUS: lauffaehig (Mini: 404 Samples), Orientierung geprueft. Die --debug-Bilder zeigen die roten
Mittellinien in der grauen befahrbaren Flaeche.
"""
import argparse
import os

import cv2
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


def build_lane_polylines(nusc_map, step_m):
    """Alle Spuren + Spurverbinder einer Karte als Liste von Nx2-Polylinien (global)."""
    tokens = [r["token"] for r in nusc_map.lane] + [r["token"] for r in nusc_map.lane_connector]
    discretized = nusc_map.discretize_lanes(tokens, step_m)  # token -> [(x, y, yaw), ...]
    polys = [np.asarray(v, dtype=np.float64)[:, :2] for v in discretized.values() if len(v) >= 2]
    bboxes = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in polys])
    return polys, bboxes


def lidar_pose(nusc, sample):
    """Position (2,) und Rotation (2x2, Lidar->Welt) des LIDAR_TOP-Sensors im Kartensystem.

    Wie BEVFusion: lidar2global = ego2global @ lidar2ego. Das Lidar sitzt ~0.94 m vor der Hinterachse und
    ist um -90 Grad gegen das Auto gedreht; das Raster haengt am Lidar, nicht am Ego-System.

    GT_FRAME=ego (Kamera-Radar-Netz CRN, kein Lidar): Ursprung = Hinterachse (ego_pose von CAM_FRONT zum Kamera-Zeitpunkt),
    Achsen wie das Lidar-System (x nach rechts, y nach vorn), also R = R_ego @ Rz(-90 Grad). Damit bleibt das Rasterlayout
    (Zeile = x, Spalte = y) fuer alle nachgelagerten Skripte gleich. Es wird nur der Pose-Eintrag der Metadaten gelesen.
    """
    if os.environ.get("GT_FRAME") == "ego":
        sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        ep = nusc.get("ego_pose", sd["ego_pose_token"])
        Re = Quaternion(ep["rotation"]).rotation_matrix[:2, :2]
        return np.asarray(ep["translation"][:2], dtype=np.float64), Re @ np.array([[0.0, 1.0], [-1.0, 0.0]])
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(ep["rotation"]).rotation_matrix
    ego2global[:3, 3] = ep["translation"]
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(cs["rotation"]).rotation_matrix
    lidar2ego[:3, 3] = cs["translation"]
    lidar2global = ego2global @ lidar2ego
    return lidar2global[:2, 3], lidar2global[:2, :2]  # Welt->Lidar: (p - t) @ R


def rasterize(polys_ego, half, res, thickness):
    n = int(round(2 * half / res))
    img = np.zeros((n, n), np.uint8)
    for p in polys_ego:
        pts = np.stack([(p[:, 1] + half) / res, (p[:, 0] + half) / res], axis=1)  # (Spalte, Zeile)
        # int-Abschneiden wie das Devkit (calib_lines.py: IoU 0.942 gegen das Original-Label; Runden: 0.70)
        cv2.polylines(img, [np.floor(pts).astype(np.int32).reshape(-1, 1, 2)], False, 255, thickness)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", required=True)
    ap.add_argument("--half", type=float, default=50.0, help="halbe Kantenlaenge in m")
    ap.add_argument("--res", type=float, default=0.5, help="Zellgroesse in m")
    ap.add_argument("--step", type=float, default=0.5, help="Abtastabstand der Mittellinie in m")
    ap.add_argument("--thickness", type=int, default=2, help="Linienbreite in Zellen (2 wie das Devkit)")
    ap.add_argument("--debug", type=int, default=0, help="Anzahl Kontrollbilder (PNG)")
    ap.add_argument("--infos", default="", help="Komma-Liste von infos-pkl: nur deren Samples rastern (bei Teil-Datensaetzen)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    if args.infos:
        import pickle

        keep = {i["token"] for p in args.infos.split(",") for i in pickle.load(open(p, "rb"))["infos"]}
        nusc.sample = [s for s in nusc.sample if s["token"] in keep]
        print(f"{len(nusc.sample)} Samples aus den infos")
    maps = {}  # location -> (NuScenesMap, polys, bboxes)
    margin = args.half * 1.5
    n_dbg = 0

    for i, sample in enumerate(nusc.sample):
        scene = nusc.get("scene", sample["scene_token"])
        loc = nusc.get("log", scene["log_token"])["location"]
        if loc not in maps:
            m = NuScenesMap(dataroot=args.dataroot, map_name=loc)
            maps[loc] = (m, *build_lane_polylines(m, args.step))
        nusc_map, polys, bboxes = maps[loc]

        t, R = lidar_pose(nusc, sample)  # Raster haengt am Lidar (wie BEVFusion), nicht am Ego-System
        yaw = np.arctan2(R[1, 0], R[0, 0])  # Richtung der Lidar-x-Achse im Kartensystem

        near = (
            (bboxes[:, 2] > t[0] - margin) & (bboxes[:, 0] < t[0] + margin)
            & (bboxes[:, 3] > t[1] - margin) & (bboxes[:, 1] < t[1] + margin)
        )
        polys_ego = [(polys[k] - t) @ R for k in np.nonzero(near)[0]]
        mask = rasterize(polys_ego, args.half, args.res, args.thickness)
        np.savez_compressed(os.path.join(args.out, sample["token"] + ".npz"), centerline=mask > 0)

        if n_dbg < args.debug:
            n = mask.shape[0]
            # befahrbare Flaeche wie das BEVFusion-Label 'drivable_area' (Lidar-Pose, transponiert)
            drv = nusc_map.get_map_mask(
                (t[0], t[1], 2 * args.half, 2 * args.half), np.degrees(yaw), ["drivable_area"], (n, n)
            ).transpose(0, 2, 1).astype(bool).any(axis=0)
            vis = np.full((n, n, 3), 255, np.uint8)
            vis[drv] = (170, 170, 170)
            vis[mask > 0] = (0, 0, 255)  # BGR: rot
            cv2.imwrite(os.path.join(args.out, f"debug_{i:04d}.png"), vis)
            n_dbg += 1

    print(f"{len(nusc.sample)} Samples -> {args.out}")


if __name__ == "__main__":
    main()
