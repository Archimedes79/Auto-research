"""Karten-Vorwissen (E34): globale Raster je Karte, aus Schichten, die NICHT die Soll-Quelle sind.
Soll-Quelle (E27 ff.): nuScenes 'lane' + 'lane_connector' (Mittellinien-Pfade + Konnektivitaet). Vorwissen hier: drivable_area (befahrbare Flaeche), road_divider + lane_divider (Trennlinien), ped_crossing (Zebrastreifen).
Kanaele: 0 befahrbar, 1 Trennlinien (Strassen- und Spurtrenner, 1 m breit), 2 Zebrastreifen. 0,5 m je Pixel, Zeile = globales y, Spalte = globales x.
Ausgabe ~/exp_gcar_<tag>/prior_<karte>.npy (uint8 [3,H,W]) + prior_meta.json. Aufruf: make_prior.py <b123|mini>   (Umgebung bev)
Dazu eine Pruefung: keine der verwendeten Schicht-Eintraege ist ein lane/lane_connector-Eintrag (Token-Menge disjunkt)."""
import json
import os
import sys

import cv2
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap

HOME = os.path.expanduser("~")
CFG = {"b123": dict(root=f"{HOME}/data/nuscenes_full", out=f"{HOME}/exp_gcar_b123"), "mini": dict(root=f"{HOME}/data/nuscenes", out=f"{HOME}/exp_gcar_mini")}
LOCS = ["boston-seaport", "singapore-onenorth", "singapore-hollandvillage", "singapore-queenstown"]
RES = 0.5


def poly_pts(m, token):
    p = m.extract_polygon(token)
    ext = np.array(p.exterior.coords)
    holes = [np.array(h.coords) for h in p.interiors]
    return ext, holes


def main(tag):
    c = CFG[tag]
    meta = {}
    for loc in LOCS:
        try:
            m = NuScenesMap(dataroot=c["root"], map_name=loc)
        except Exception as e:
            print("Karte fehlt:", loc, str(e)[:80])
            continue
        W, H = m.canvas_edge  # Meter
        w, h = int(np.ceil(W / RES)) + 1, int(np.ceil(H / RES)) + 1
        img = np.zeros((3, h, w), np.uint8)

        def px(a):
            return np.round(np.stack([a[:, 0] / RES, a[:, 1] / RES], 1)).astype(np.int32)

        # 0: befahrbare Flaeche (Polygone mit Loechern)
        n_dr = 0
        for rec in m.drivable_area:
            for pt in rec["polygon_tokens"]:
                ext, holes = poly_pts(m, pt)
                cv2.fillPoly(img[0], [px(ext)], 1)
                for hl in holes:
                    cv2.fillPoly(img[0], [px(hl)], 0)
                n_dr += 1
        # 1: Trennlinien
        n_ln = 0
        for layer in ("road_divider", "lane_divider"):
            for rec in getattr(m, layer):
                ln = m.extract_line(rec["line_token"])
                a = np.array(ln.coords)
                if len(a) >= 2:
                    cv2.polylines(img[1], [px(a).reshape(-1, 1, 2)], False, 1, 2)
                    n_ln += 1
        # 2: Zebrastreifen
        n_pc = 0
        for rec in m.ped_crossing:
            ext, _ = poly_pts(m, rec["polygon_token"])
            cv2.fillPoly(img[2], [px(ext)], 1)
            n_pc += 1
        used = {rec["token"] for layer in ("drivable_area", "road_divider", "lane_divider", "ped_crossing") for rec in getattr(m, layer)}
        gt_src = {r["token"] for r in m.lane} | {r["token"] for r in m.lane_connector}
        assert not (used & gt_src), "Vorwissen enthaelt Soll-Eintraege!"
        np.save(f"{c['out']}/prior_{loc}.npy", img)
        meta[loc] = dict(W=float(W), H=float(H), shape=list(img.shape), res=RES, n_drivable=n_dr, n_lines=n_ln, n_ped=n_pc, frac=[float(img[k].mean()) for k in range(3)])
        print(loc, meta[loc], "| Token-Menge disjunkt zur Soll-Quelle: ok", flush=True)
    json.dump(meta, open(f"{c['out']}/prior_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
