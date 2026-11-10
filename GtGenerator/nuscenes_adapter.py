"""nuScenes adapter of the flow box (flow.py): turns devkit records into Track / Ego / RadarFrame.

    scene = NuScenesScene(nusc, dataroot, "scene-0061")
    fg = scene.flow(sample_token)                              # FlowGT (ego motion + objects) at the camera time of that keyframe
    fg = scene.flow(sample_token, at="annotation")             # the same at the annotation time

All inputs are rebuilt from time stamps: tracks from the 2 Hz annotations (box centre, yaw, size, observed = at least one
lidar or radar point), the ego from the 12 Hz CAM_FRONT poses of the whole scene, radar from the five radars' samples and
sweeps within +-`radar_window` s of the frame time (position and ego-compensated radial Doppler in the world frame).
Needs only the metadata and the radar files - no lidar, no maps.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pyquaternion import Quaternion

from .flow import Ego, RadarFrame, Track, estimate_flow

RADARS = ("RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT")


def category_group(name):
    """nuScenes category -> GtGenerator group (None = not a GT class)."""
    if name.startswith(("vehicle.bicycle", "vehicle.motorcycle")):
        return "two_wheeler"
    if name.startswith("vehicle."):
        return "vehicle"
    if name.startswith("human.pedestrian"):
        return "pedestrian"
    return None


def pose_matrix(rec):
    T = np.eye(4)
    T[:3, :3] = Quaternion(rec["rotation"]).rotation_matrix
    T[:3, 3] = rec["translation"]
    return T


class NuScenesScene:
    def __init__(self, nusc, dataroot, scene_token_or_name, radar_window=0.1, config=None):
        self.nusc, self.root, self.radar_window, self.config = nusc, dataroot, radar_window, config
        self.scene = next(s for s in nusc.scene if scene_token_or_name in (s["name"], s["token"]))
        self.tracks = self._tracks()
        self.ego = self._ego()
        self._points = {}                                  # radar files read ahead by `_prefetch`, taken by `_radar_frame`
        self._sample_times = None

    # ---- builders
    def _tracks(self):
        nusc, per = self.nusc, {}
        tok = self.scene["first_sample_token"]
        while tok:
            s = nusc.get("sample", tok)
            for atok in s["anns"]:
                a = nusc.get("sample_annotation", atok)
                g = category_group(a["category_name"])
                if g is None:
                    continue
                per.setdefault(a["instance_token"], []).append(
                    (s["timestamp"] * 1e-6, a["translation"], Quaternion(a["rotation"]).yaw_pitch_roll[0], a["size"], g,
                     a["num_lidar_pts"] + a["num_radar_pts"] > 0, atok))
            tok = s["next"]
        out = []
        for inst, rows in per.items():
            trans = np.array([r[1] for r in rows], float)
            out.append(Track(inst, [r[0] for r in rows], trans[:, :2], [r[2] for r in rows], rows[0][3], rows[0][4],
                             [r[5] for r in rows], extra={"atoks": [r[6] for r in rows], "trans": trans}))
        return out

    def _ego(self):
        nusc = self.nusc
        sd = nusc.get("sample_data", nusc.get("sample", self.scene["first_sample_token"])["data"]["CAM_FRONT"])
        while sd["prev"]:
            sd = nusc.get("sample_data", sd["prev"])
        t, x, y, yaw = [], [], [], []
        while True:
            p = nusc.get("ego_pose", sd["ego_pose_token"])
            t.append(sd["timestamp"] * 1e-6)
            x.append(p["translation"][0])
            y.append(p["translation"][1])
            yaw.append(Quaternion(p["rotation"]).yaw_pitch_roll[0])
            if not sd["next"]:
                break
            sd = nusc.get("sample_data", sd["next"])
        return Ego(t, x, y, yaw)

    # ---- radar
    def radar(self, t):
        """RadarFrames of the five radars within +-radar_window s of t."""
        from nuscenes.utils.data_classes import RadarPointCloud
        nusc = self.nusc
        sample = self._nearest_sample(t)
        want = []
        for ch in RADARS:
            sd0 = nusc.get("sample_data", sample["data"][ch])
            cands = [sd0]
            for direction in ("prev", "next"):
                sd = sd0
                while sd[direction]:
                    sd = nusc.get("sample_data", sd[direction])
                    if abs(sd["timestamp"] * 1e-6 - t) > self.radar_window:
                        break
                    cands.append(sd)
            want += [(sd, ch) for sd in cands if abs(sd["timestamp"] * 1e-6 - t) <= self.radar_window]
        self._prefetch([sd for sd, _ in want], RadarPointCloud)
        frames = []
        for sd, ch in want:
            fr = self._radar_frame(sd, ch, RadarPointCloud)
            if fr is not None:
                frames.append(fr)
        return frames

    def _prefetch(self, sds, RadarPointCloud):
        """read the radar files of a frame in parallel: opening a file on the raw-data disk takes about 50 ms, a keyframe needs
        10 - 15 of them, and the frames are made from the points in the same order as before (the result does not change)"""
        todo = [sd for sd in sds if sd["token"] not in self._points]
        if len(todo) < 2:
            return
        with ThreadPoolExecutor(max_workers=min(8, len(todo))) as ex:
            for sd, pts in zip(todo, ex.map(lambda s: RadarPointCloud.from_file(os.path.join(self.root, s["filename"])).points, todo)):
                self._points[sd["token"]] = pts

    def _radar_frame(self, sd, ch, RadarPointCloud):
        key = sd["token"]
        nusc = self.nusc
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        Tg = pose_matrix(nusc.get("ego_pose", sd["ego_pose_token"])) @ pose_matrix(cs)       # world <- sensor
        pts = self._points.pop(key, None)
        if pts is None:
            pts = RadarPointCloud.from_file(os.path.join(self.root, sd["filename"])).points
        if pts.shape[1] == 0:
            fr = None
        else:
            P = (Tg[:3, :3] @ pts[:3]).T + Tg[:3, 3]
            V = (Tg[:3, :3] @ np.vstack([pts[8], pts[9], np.zeros(pts.shape[1])])).T        # vx_comp, vy_comp: ego compensated
            r = P[:, :2] - Tg[:2, 3]
            d = np.einsum("ij,ij->i", V[:, :2], r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-6))
            fr = RadarFrame(sd["timestamp"] * 1e-6, ch, Tg[:2, 3].copy(), P[:, :2], d)
        return fr

    def _nearest_sample(self, t):
        if self._sample_times is None:
            rows, tok = [], self.scene["first_sample_token"]
            while tok:
                s = self.nusc.get("sample", tok)
                rows.append((s["timestamp"] * 1e-6, s))
                tok = s["next"]
            self._sample_times = rows
        return min(self._sample_times, key=lambda r: abs(r[0] - t))[1]

    # ---- the one call
    def time_of(self, sample_token, at="camera"):
        s = self.nusc.get("sample", sample_token)
        return (self.nusc.get("sample_data", s["data"]["CAM_FRONT"])["timestamp"] if at == "camera" else s["timestamp"]) * 1e-6

    def flow(self, sample_token, at="camera", use_radar=True, config=None):
        """FlowGT for a keyframe. at = "camera" (the CAM_FRONT time of the keyframe: the ego frame of the GT) or
        "annotation" (the annotation / lidar time)."""
        t = self.time_of(sample_token, at)
        return estimate_flow(self.tracks, self.ego, t, self.radar(t) if use_radar else (), config or self.config)
