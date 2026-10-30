"""Video: driving through the GT of one nuScenes scene at the 12 Hz front-camera rate (Mirko, 2026-10-30).

Left the camera image with the GT projected (map lines: lane divider yellow, road divider orange, centre-line cyan; 3D
boxes: vehicle red, pedestrian magenta, two-wheeler orange, white rim = stationary, grey = no lidar/radar point);
right the BEV GT in the ego frame at that frame (semantic layers of training/gt_bev.py, dimmed outside the camera view,
boxes as footprints with heading and speed arrows).
- map layers: the exact generator of training/gt_bev.py at the pose of EVERY camera frame (keyframes and sweeps)
- boxes: annotated at 2 Hz only; between two annotations their world pose is linearly interpolated (yaw by the shortest
  angle) at the camera time; stationary instances use their single fixed world pose (rule of 2026-10-26); an instance seen
  in one annotation only is shown within 0.3 s of it
The script does not change gt_bev.py (a subclass picks the camera frame).
Run from the bev_fusion folder:
    wsl ~/miniconda3/envs/gcar/bin/python temp/checks/video_gt_drive.py <scene name> <out mp4> [max seconds]
"""
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

sys.path.insert(0, "training")
sys.path.insert(0, "temp/checks")
from gt_bev import NuScenesGT, category_group, pose_matrix  # noqa: E402
from lift_pd import Grid  # noqa: E402
from view_gt_observable import densify  # noqa: E402
from view_gt_samples import semantic_rgb  # noqa: E402

G = Grid()
COL = {"vehicle": (50, 50, 240), "pedestrian": (230, 60, 230), "two_wheeler": (30, 150, 255)}        # BGR
LCOL = {"lane_divider": (25, 230, 255), "road_divider": (0, 128, 255), "centreline": (230, 230, 25)}   # BGR
PXM = 9.0                                                                                           # BEV pixels per metre
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


class SweepGT(NuScenesGT):
    """keyframe_gt at the pose of any CAM_FRONT sample_data of the scene (the map layers do not depend on the keyframe)."""
    sd_override = None

    def cam(self, sample_token, channel="CAM_FRONT"):
        return self.sd_override if self.sd_override is not None else super().cam(sample_token, channel)


def keyframe_tracks(src, scene):
    """per keyframe: time (us), {instance: dict(world pose, size, group, observed)}"""
    out, tok = [], scene["first_sample_token"]
    fixed = src.stationary_instances(scene["token"])
    while tok:
        s = src.nusc.get("sample", tok)
        d = {}
        for atok in s["anns"]:
            a = src.nusc.get("sample_annotation", atok)
            grp = category_group(a["category_name"])
            if grp is None:
                continue
            inst = a["instance_token"]
            if inst in fixed:
                tr, yaw, size = fixed[inst]
            else:
                tr, yaw, size = np.array(a["translation"]), Quaternion(a["rotation"]).yaw_pitch_roll[0], np.array(a["size"])
            d[inst] = dict(t=np.asarray(tr, float), yaw=float(yaw), size=np.asarray(size, float), group=grp,
                           observed=a["num_lidar_pts"] + a["num_radar_pts"] > 0, stationary=inst in fixed)
        out.append((s["timestamp"], d))
        tok = s["next"]
    return out


def boxes_at(tracks, t):
    """interpolated boxes at time t (us): list of dicts with world pose; speed in m/s"""
    ts = np.array([x[0] for x in tracks])
    i = int(np.searchsorted(ts, t) - 1)
    i = min(max(i, 0), len(ts) - 2)
    (t0, d0), (t1, d1) = tracks[i], tracks[i + 1]
    w = float(np.clip((t - t0) / max(t1 - t0, 1), 0.0, 1.0))
    out = []
    for inst in set(d0) | set(d1):
        a, b = d0.get(inst), d1.get(inst)
        if a and b:
            if a["stationary"]:
                pose, yaw, speed = a["t"], a["yaw"], 0.0
            else:
                pose = (1 - w) * a["t"] + w * b["t"]
                dy = (b["yaw"] - a["yaw"] + math.pi) % (2 * math.pi) - math.pi
                yaw = a["yaw"] + w * dy
                speed = float(np.hypot(*(b["t"][:2] - a["t"][:2])) / max((t1 - t0) * 1e-6, 1e-3))
            out.append(dict(a, t=pose, yaw=yaw, speed=speed, observed=a["observed"] and b["observed"]))
        else:
            c, dt = (a, (t - t0) * 1e-6) if a else (b, (t1 - t) * 1e-6)
            if abs(dt) <= 0.3:
                out.append(dict(c, speed=0.0))
    return out


def corners_world(b):
    c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
    w, l, h = b["size"]
    loc = np.array([[sx * l / 2, sy * w / 2, sz * h / 2] for sz in (-1, 1) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    return np.stack([b["t"][0] + c * loc[:, 0] - s * loc[:, 1], b["t"][1] + s * loc[:, 0] + c * loc[:, 1], b["t"][2] + loc[:, 2]], 1)


def seg_cam(img, K, a, b, color, thick, near=0.5):
    if a[2] < near and b[2] < near:
        return
    if a[2] < near or b[2] < near:
        t = (near - a[2]) / (b[2] - a[2])
        c = a + t * (b - a)
        a, b = (c, b) if a[2] < near else (a, c)
    uv = [(int(round(K[0, 0] * p[0] / p[2] + K[0, 2])), int(round(K[1, 1] * p[1] / p[2] + K[1, 2]))) for p in (a, b)]
    cv2.line(img, uv[0], uv[1], color, thick, cv2.LINE_AA)


def to_px(x, y):
    return int(round((26.0 - y) * PXM)), int(round((60.0 - x) * PXM))


def render(src, scene, sd, tracks, kf_token, header):
    src.sd_override = sd
    gt = src.keyframe_gt(kf_token, G)
    meta = gt["meta"]
    T_g2e = np.linalg.inv(meta["T_e2g"])
    K, T_e2c = meta["K"], np.linalg.inv(meta["T_c2e"])
    boxes = boxes_at(tracks, sd["timestamp"])
    # camera panel (drawn at full resolution)
    img = cv2.cvtColor(np.asarray(Image.open(os.path.join(src.dataroot, sd["filename"])).convert("RGB")), cv2.COLOR_RGB2BGR)
    for layer, pls in meta["vectors"].items():
        for pl in pls:
            d = densify(pl, 1.0)
            keep = (d[:, 0] > -5) & (d[:, 0] < 70) & (np.abs(d[:, 1]) < 30)
            P = (np.c_[d, np.zeros(len(d)), np.ones(len(d))] @ T_e2c.T)[:, :3]
            for i in range(len(d) - 1):
                if keep[i] and keep[i + 1]:
                    seg_cam(img, K, P[i], P[i + 1], LCOL[layer], 2 if layer != "centreline" else 1)
    ego_boxes = []
    for b in boxes:
        cw = corners_world(b)
        ce = (np.c_[cw, np.ones(8)] @ T_g2e.T)[:, :3]
        ego_boxes.append((b, ce))
        if ce[:, 0].max() < 1 or ce[:, 0].min() > 80:
            continue
        P = (np.c_[ce, np.ones(8)] @ T_e2c.T)[:, :3]
        col = COL[b["group"]] if b["observed"] else (170, 170, 170)
        for i, j in EDGES:
            if b["stationary"]:
                seg_cam(img, K, P[i], P[j], (255, 255, 255), 4)
            seg_cam(img, K, P[i], P[j], col, 2)
    cam_panel = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
    # BEV panel
    rgb = semantic_rgb(gt)
    rgb = rgb * (1 - 0.55 * (1 - gt["camera"]))[..., None]
    bev = np.flip(np.flip(rgb, 0), 1)
    bev = cv2.resize((np.clip(bev, 0, 1) * 255).astype(np.uint8)[..., ::-1], (int(52 * PXM), int(60 * PXM)),
                     interpolation=cv2.INTER_NEAREST)
    for b, ce in ego_boxes:
        fp = ce[:4, :2]
        if fp[:, 0].max() < 0 or fp[:, 0].min() > 60 or abs(fp[:, 1]).min() > 27:
            continue
        pts = np.array([to_px(x, y) for x, y in fp], np.int32)
        col = COL[b["group"]] if b["observed"] else (150, 150, 150)
        cv2.fillPoly(bev, [pts], col)
        cv2.polylines(bev, [pts], True, (255, 255, 255) if b["stationary"] else (20, 20, 20), 1, cv2.LINE_AA)
        cx, cy = fp.mean(0)
        front = fp[:2].mean(0)
        cv2.line(bev, to_px(cx, cy), to_px(front[0], front[1]), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(bev, to_px(0, 0), 5, (255, 255, 255), -1)
    out = np.full((540 + 44, 960 + bev.shape[1], 3), 255, np.uint8)
    out[44:, :960] = cam_panel
    out[44:44 + bev.shape[0], 960:] = bev
    cv2.putText(out, header[0], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(out, header[1], (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    return out[:, :out.shape[1] - out.shape[1] % 2][:out.shape[0] - out.shape[0] % 2]


def main(name, out_path, max_s):
    root = os.path.expanduser("~/data/nuscenes")
    src = SweepGT(root, "v1.0-mini")
    scene = next(s for s in src.nusc.scene if s["name"] == name)
    tracks = keyframe_tracks(src, scene)
    t_lo, t_hi = tracks[0][0], tracks[-1][0]
    first = scene["first_sample_token"]
    track = src.camera_track(first)
    frames = [sd for sd in track if t_lo <= sd["timestamp"] <= t_hi]
    if max_s:
        frames = frames[:int(max_s * 12)]
    key_ts = {t for t, _ in tracks}
    writer = None
    for n, sd in enumerate(frames):
        k = int(np.argmin([abs(t - sd["timestamp"]) for t in key_ts]))
        kf_token = scene["first_sample_token"]
        for _ in range(k):
            kf_token = src.nusc.get("sample", kf_token)["next"]
        i = track.index(sd)
        sp = 0.0
        if i + 1 < len(track):
            rel = np.linalg.inv(src.ego_pose(sd)) @ src.ego_pose(track[i + 1])
            sp = rel[0, 3] / max((track[i + 1]["timestamp"] - sd["timestamp"]) * 1e-6, 1e-3)
        head = (f"{name}: {scene['description'][:60]} | t = {(sd['timestamp'] - t_lo) * 1e-6:5.1f} s | ego {sp * 3.6:3.0f} km/h | "
                f"{'KEYFRAME' if sd['is_key_frame'] else 'sweep'}",
                "map layers exact per frame (training/gt_bev.py); boxes: 2 Hz annotations, interpolated in between; white rim = stationary")
        frame = render(src, scene, sd, tracks, kf_token, head)
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 12, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        if n % 24 == 0:
            print(f"{n}/{len(frames)} frames", flush=True)
    writer.release()
    print("saved", out_path, f"{len(frames)} frames, {len(frames) / 12:.1f} s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0)
