"""Pictures of a lane graph (`autogt.lane_graph`): a top view and the graph projected into the six cameras (the review picture of the lane GT).

    python -m autogt lanegraph SAMPLE_TOKEN [--version v1.0-mini] [--dataroot DIR] [--out lane_graph.png]

The pieces are drawn in one colour per piece id, the events as markers (split orange up-triangle, merge blue down-triangle, stop red square, exit yellow square, start green
circle, end grey circle, window edge white tick). Points lie on the ground plane of the ego frame (z = 0), as the other autogt layers do; every camera
uses its own ego pose and calibration, so the motion between the six exposures is accounted for.
"""
import os
import zlib

import cv2
import numpy as np

from . import lane_graph as LG
from .nuscenes_adapter import pose_matrix

CAMERAS = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")
LINE = {"lane_divider": (255, 255, 255), "road_divider": (0, 220, 255), "boundary": (255, 255, 0)}                 # BGR: white, yellow, cyan
MARK = {"split": ((0, 165, 255), "^"), "merge": ((255, 128, 0), "v"), "cross": ((255, 0, 255), "d"), "stop": ((0, 0, 255), "s"), "exit": ((0, 255, 255), "s"),
        "start": ((0, 200, 0), "o"), "end": ((128, 128, 128), "o"), "window": ((255, 255, 255), "|"), "corner": ((255, 255, 0), "d")}


def _colour(key):
    h = (zlib.crc32(str(key).encode()) * 2654435761) % 180
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def _marker(img, uv, kind, size, label=None):
    colour, shape = MARK[kind]
    u, v = int(round(uv[0])), int(round(uv[1]))
    s = size
    if shape == "^":
        cv2.fillConvexPoly(img, np.array([[u, v - s], [u - s, v + s], [u + s, v + s]]), colour)
    elif shape == "v":
        cv2.fillConvexPoly(img, np.array([[u, v + s], [u - s, v - s], [u + s, v - s]]), colour)
    elif shape == "d":
        cv2.fillConvexPoly(img, np.array([[u, v - s], [u - s, v], [u, v + s], [u + s, v]]), colour)
    elif shape == "s":
        cv2.rectangle(img, (u - s, v - s), (u + s, v + s), colour, -1)
    elif shape == "o":
        cv2.circle(img, (u, v), s, colour, -1)
    else:
        cv2.line(img, (u - s, v), (u + s, v), colour, 2)
    cv2.rectangle(img, (u - s - 1, v - s - 1), (u + s + 1, v + s + 1), (0, 0, 0), 1) if shape != "|" else None
    if label:
        cv2.putText(img, label, (u + s + 2, v - s), cv2.FONT_HERSHEY_SIMPLEX, max(0.35, s / 14), (255, 255, 255), 1, cv2.LINE_AA)


def bev_image(graph, ppm=8, title=None, labels=True):
    """the top view of a graph in the lane frame (x right, y forward), window |x|, |y| <= half; ego at the centre"""
    half = graph["half"]
    n = int(2 * half * ppm)
    img = np.full((n, n, 3), 28, np.uint8)
    for g in range(-int(half), int(half) + 1, 10):
        c = (60, 60, 60)
        cv2.line(img, (int((g + half) * ppm), 0), (int((g + half) * ppm), n), c, 1)
        cv2.line(img, (0, int((half - g) * ppm)), (n, int((half - g) * ppm)), c, 1)

    def px(p):
        return (p[..., 0] + half) * ppm, (half - p[..., 1]) * ppm
    for kind, fill in (("road", (60, 60, 60)), ("intersection", (40, 70, 110))):             # the drivable area and the intersection areas, under everything else
        layer = img.copy()
        for poly in graph.get("areas", {}).get(kind, []):
            u, v = px(poly["exterior"])
            cv2.fillPoly(layer, [np.c_[u, v].astype(np.int32)], fill)
            for h in poly["holes"]:
                u, v = px(h)
                cv2.fillPoly(layer, [np.c_[u, v].astype(np.int32)], (28, 28, 28))
        img = cv2.addWeighted(layer, 0.8, img, 0.2, 0) if kind == "road" else cv2.addWeighted(layer, 0.6, img, 0.4, 0)
    for part in graph["pieces"]:
        u, v = px(part["pts"])
        pts = np.c_[u, v].astype(np.int32)
        cv2.polylines(img, [pts], False, _colour(part["piece"]), 2, cv2.LINE_AA)
        if len(pts) > 6:                                     # the direction of travel
            m = len(pts) // 2
            cv2.arrowedLine(img, tuple(pts[m - 1]), tuple(pts[m + 1]), _colour(part["piece"]), 2, tipLength=1.2)
    for kind, lines in graph.get("lines", {}).items():
        for ln in lines:
            u, v = px(ln["pts"])
            cv2.polylines(img, [np.c_[u, v].astype(np.int32)], False, LINE[kind], 1, cv2.LINE_AA)
            for c in ln["corners"]:
                _marker(img, px(c["pos"]), "corner", 3, None)
    for part in graph["pieces"]:                                 # lane count and index of every lane piece, at its middle
        if part["kind"] == "lane" and part.get("lane_count") and len(part["pts"]) > 2:
            u, v = px(part["pts"][len(part["pts"]) // 2])
            cv2.putText(img, f"{part['lane_index'] + 1}/{part['lane_count']}", (int(u) + 4, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    for e in graph["events"]:
        u, v = px(e["pos"])
        _marker(img, (u, v), e["type"], 5 if e["type"] != "window" else 7, None)
    ex, ey = px(np.array([0.0, 0.0]))
    cv2.rectangle(img, (int(ex - 1.0 * ppm), int(ey - 3.1 * ppm)), (int(ex + 1.0 * ppm), int(ey + 1.0 * ppm)), (255, 255, 255), 1)
    if title and labels:
        cv2.putText(img, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _lane_to_ego(p):
    """lane frame (x right, y forward) -> ego frame (x forward, y left)"""
    return np.c_[p[..., 1], -p[..., 0]]


def camera_image(gt, sample, graph, channel, T_e2g_front, scale=0.5, step=0.5):
    """the camera image of `channel` with the graph projected on the ground plane"""
    sd = gt.nusc.get("sample_data", sample["data"][channel])
    cs = gt.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    K = np.asarray(cs["camera_intrinsic"], float)
    T_c2e = pose_matrix(cs)
    T_e2g = pose_matrix(gt.nusc.get("ego_pose", sd["ego_pose_token"]))
    img = cv2.imread(os.path.join(gt.dataroot, sd["filename"]))
    if img is None:                                          # parts 7 - 10 of nuScenes have only CAM_FRONT unpacked here
        img = np.full((900, 1600, 3), 40, np.uint8)
        cv2.putText(img, "image not on disk", (500, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 3, cv2.LINE_AA)
    T_front2cam = np.linalg.inv(T_c2e) @ np.linalg.inv(T_e2g) @ T_e2g_front        # ego(front time) -> this camera

    def project(p_lane, dz=0.0):
        e = _lane_to_ego(np.asarray(p_lane, float).reshape(-1, 2))
        h = np.c_[e, np.full(len(e), dz), np.ones(len(e))] @ T_front2cam.T
        z = h[:, 2]
        uv = (K @ h[:, :3].T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = uv[:, :2] / uv[:, 2:3]
        return uv, z
    for part in graph["pieces"]:
        P = part["pts"].astype(float)
        if len(P) < 2:
            continue
        # the 0.5 m samples are too coarse for a line close to the camera: densify to 0.1 m
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        u = np.arange(0, s[-1], 0.1)
        D = np.c_[np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1])]
        uv, z = project(D)
        ok = z > 0.5
        pts = np.round(uv).astype(np.int32)
        for run in np.split(np.nonzero(ok)[0], np.nonzero(np.diff(np.nonzero(ok)[0]) > 1)[0] + 1):
            if len(run) > 1:
                cv2.polylines(img, [pts[run]], False, _colour(part["piece"]), 4, cv2.LINE_AA)
    for kind, lines in graph.get("lines", {}).items():
        for ln in lines:
            P = ln["pts"].astype(float)
            if len(P) < 2:
                continue
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            u = np.arange(0, s[-1], 0.2)
            D = np.c_[np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1])] if len(u) else P[:1]
            uv, z = project(D)
            ok = z > 0.5
            pts = np.round(uv).astype(np.int32)
            for run in np.split(np.nonzero(ok)[0], np.nonzero(np.diff(np.nonzero(ok)[0]) > 1)[0] + 1):
                if len(run) > 1:
                    cv2.polylines(img, [pts[run]], False, LINE[kind], 2, cv2.LINE_AA)
            for c in ln["corners"]:
                uv, z = project(c["pos"][None])
                if z[0] > 0.5 and 0 <= uv[0, 0] < img.shape[1] and 0 <= uv[0, 1] < img.shape[0]:
                    _marker(img, uv[0], "corner", 9, None)
    for e in graph["events"]:
        uv, z = project(e["pos"][None])
        if z[0] > 0.5 and 0 <= uv[0, 0] < img.shape[1] and 0 <= uv[0, 1] < img.shape[0]:
            _marker(img, uv[0], e["type"], 12 if e["type"] != "window" else 14, e["type"] if e["type"] not in ("window",) else None)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.putText(img, channel, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def has_cameras(gt, sample_token):
    """True when all six camera images of the keyframe are on disk"""
    sample = gt.nusc.get("sample", sample_token)
    return all(os.path.exists(os.path.join(gt.dataroot, gt.nusc.get("sample_data", sample["data"][ch])["filename"])) for ch in CAMERAS)


def mosaic(gt, sample_token, graph, title=None, scale=0.5):
    """the six cameras (2 x 3) and the top view side by side"""
    sample = gt.nusc.get("sample", sample_token)
    T_front = gt.ego_pose(gt.cam(sample_token))
    tiles = [camera_image(gt, sample, graph, ch, T_front, scale) for ch in CAMERAS]
    h, w = tiles[0].shape[:2]
    grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    bev = bev_image(graph, ppm=max(1, int(round(2 * h / (2 * graph["half"])))), title=title)
    bev = cv2.resize(bev, (2 * h, 2 * h), interpolation=cv2.INTER_AREA)
    return np.hstack([grid, bev])


def legend_text(graph):
    cnt = {}
    for e in graph["events"]:
        cnt[e["type"]] = cnt.get(e["type"], 0) + 1
    n_lines = {k: len(v) for k, v in graph.get("lines", {}).items()}
    n_corners = sum(len(ln["corners"]) for v in graph.get("lines", {}).values() for ln in v)
    return (f"{len(graph['pieces'])} pieces, {len(graph.get('roads', []))} roads/intersections, events " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items()))
            + (f"; lines {n_lines}, {n_corners} corners" if n_lines else ""))
