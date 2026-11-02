"""nuScenes samples for the whole chain (plan.md sections 0-9): front camera history + all radars + ego motion + GT.

One sample = keyframe k (it needs a previous and a next keyframe, for the GT at about -0.5 / 0 / +0.5 s):
- camera: the CAM_FRONT image at k and n_cam - 1 older 12 Hz frames (every `cam_stride`-th sweep), each resized and
  cropped like DualBEV's test pipeline (1600 x 900 -> 704 x 256, `camera_encoder.nuscenes_front_ida`) and normalised;
  per frame its intrinsics K, T_cam_car, the image transform ida, the 27-value camera vector of the depth head and the
  ego steps (dt, v, omega) from that frame to the current one (for `splat.splat_window` / `chain_positions`)
- radar: every sweep of the five radars within `radar_window` seconds before the current camera time (and up to it);
  each detection moved into the ego frame of the current camera time by the EXACT poses (static content exact) plus,
  optionally, its measured radial motion vr * u * age (the only object motion a radar measures); per detection:
  b (sample index after collation), x, y (m), r (range from its sensor), psi (ray direction in the current frame), vr
  (ego-compensated radial velocity, m/s), rcs, dt (age, s), sensor (0..4) - the format of training/radar_bev.py
- GT: `autogt.NuScenesGT.training_sample` (cur / prev / next with their ego steps), optionally cached as compressed
  float16 npz in a cache folder of this project (derived data, not in git; written atomically, a broken file is rebuilt)
- meta: tokens, timestamps, the ego speed and yaw rate of the last camera frame
Raw data is read through the WSL views (~/data/nuscenes_full, ~/data/nuscenes for mini; datasets/NuScenes/README.md).
Splits: official nuScenes train / val scene names restricted to the scenes on disk (`available_scenes`).
Run the self-test (nuScenes mini): python training/nuscenes_dataset.py
"""
import math
import os
import sys

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

_BEV_FUSION = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
while not os.path.isdir(os.path.join(_BEV_FUSION, "autogt")) and os.path.dirname(_BEV_FUSION) != _BEV_FUSION:
    _BEV_FUSION = os.path.dirname(_BEV_FUSION)       # a run snapshot (runs/<name>/code/) sits three levels below the bev_fusion folder
if _BEV_FUSION not in sys.path:
    sys.path.append(_BEV_FUSION)                     # the GT generator: the standalone suite bev_fusion/autogt/
from autogt.nuscenes_adapter import pose_matrix  # noqa: E402
from autogt.nuscenes_gt import NuScenesGT  # noqa: E402

try:
    from .camera_encoder import IMG_MEAN, IMG_STD, camera_vector, nuscenes_front_ida
    from .lift_pd import Grid
except ImportError:                                   # run as a script
    from camera_encoder import IMG_MEAN, IMG_STD, camera_vector, nuscenes_front_ida
    from lift_pd import Grid

RADARS = ("RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT")
DST_HW = (256, 704)


def available_scenes(nusc, split):
    """official split ('train' / 'val' / 'mini_train' / 'mini_val') restricted to scenes whose CAM_FRONT keyframes exist."""
    from nuscenes.utils.splits import create_splits_scenes
    names = set(create_splits_scenes()[split])
    out = []
    for sc in nusc.scene:
        if sc["name"] not in names:
            continue
        s = nusc.get("sample", sc["first_sample_token"])
        sd = nusc.get("sample_data", s["data"]["CAM_FRONT"])
        if os.path.exists(os.path.join(nusc.dataroot, sd["filename"])):
            out.append(sc["token"])
    return out


class FrontRadarDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, version, scenes=None, grid=None, n_cam=3, cam_stride=2, radar_window=0.5,
                 radar_doppler_shift=True, with_gt=True, gt_ss=4, cache_dir=None, nusc=None):
        self.gt = NuScenesGT(dataroot, version, nusc)
        self.nusc, self.root = self.gt.nusc, dataroot
        self.grid = grid or Grid()
        self.n_cam, self.cam_stride, self.radar_window = n_cam, cam_stride, radar_window
        self.doppler_shift, self.with_gt, self.gt_ss, self.cache_dir = radar_doppler_shift, with_gt, gt_ss, cache_dir
        scenes = set(scenes) if scenes is not None else {s["token"] for s in self.nusc.scene}
        self.samples = [s["token"] for s in self.nusc.sample if s["scene_token"] in scenes and s["prev"] and s["next"]]
        self.ida = nuscenes_front_ida(dst_hw=DST_HW)

    def __len__(self):
        return len(self.samples)

    # ---- camera
    def _image_uint8(self, sd):
        """(H, W, 3) uint8 RGB after the resize + crop of self.ida (also what the lane-branch image cache stores)."""
        img = Image.open(os.path.join(self.root, sd["filename"])).convert("RGB")
        s = self.ida[0, 0].item()
        w, h = int(round(img.width * s)), int(round(img.height * s))
        top = h - DST_HW[0]
        img = img.resize((w, h), Image.BILINEAR).crop((0, top, DST_HW[1], h))
        return np.asarray(img)

    def _image(self, sd):
        x = (torch.tensor(self._image_uint8(sd), dtype=torch.float32) - torch.tensor(IMG_MEAN)) / torch.tensor(IMG_STD)
        return x.permute(2, 0, 1)

    def _cam_frames(self, token):
        track = self.gt.camera_track(token)
        cur = self.gt.cam(token)
        i0 = [t["token"] for t in track].index(cur["token"])
        idx = [max(0, i0 - k * self.cam_stride) for k in range(self.n_cam)]
        frames = []
        for i in idx:
            sd = track[i]
            cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            K = torch.tensor(cs["camera_intrinsic"], dtype=torch.float32)
            T_c2e = torch.tensor(pose_matrix(cs), dtype=torch.float32)
            steps = self.gt.ego_steps(sd["timestamp"], cur["timestamp"], track) if i != i0 else []
            cv = camera_vector(K[None], self.ida[None, :2, :2], self.ida[None, :2, 2], T_c2e[None])[0]
            frames.append({"img": self._image(sd), "K": K, "T_cam_car": torch.linalg.inv(T_c2e), "cam_vec": cv,
                           "steps": steps, "timestamp": sd["timestamp"], "token": sd["token"]})
        return frames, track, cur

    # ---- radar
    def _radar(self, sample, cur_sd):
        from nuscenes.utils.data_classes import RadarPointCloud
        T_cur = pose_matrix(self.nusc.get("ego_pose", cur_sd["ego_pose_token"]))
        T_cur_inv = np.linalg.inv(T_cur)
        t_cur = cur_sd["timestamp"]
        cols = {k: [] for k in ("x", "y", "r", "psi", "vr", "rcs", "dt", "sensor")}
        for si, ch in enumerate(RADARS):
            sd = self.nusc.get("sample_data", sample["data"][ch])
            while sd["next"] and self.nusc.get("sample_data", sd["next"])["timestamp"] <= t_cur:
                sd = self.nusc.get("sample_data", sd["next"])          # newest sweep not after the camera time
            while sd is not None and (t_cur - sd["timestamp"]) * 1e-6 <= self.radar_window:
                if sd["timestamp"] <= t_cur:
                    pc = RadarPointCloud.from_file(os.path.join(self.root, sd["filename"]))
                    P = pc.points
                    cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
                    T_s2e = pose_matrix(cs)
                    T_e2g = pose_matrix(self.nusc.get("ego_pose", sd["ego_pose_token"]))
                    T = T_cur_inv @ T_e2g @ T_s2e                           # sensor -> current ego frame (exact)
                    xs, ys = P[0], P[1]
                    phi = np.arctan2(ys, xs)                                # ray in the sensor frame
                    vr = P[8] * np.cos(phi) + P[9] * np.sin(phi)            # vx_comp, vy_comp: the radial vector
                    age = (t_cur - sd["timestamp"]) * 1e-6
                    if self.doppler_shift:
                        rng = np.hypot(xs, ys)
                        scale = (rng + vr * age) / np.maximum(rng, 1e-6)    # radial motion along the ray
                        xs, ys = xs * scale, ys * scale
                    pts = np.stack([xs, ys, np.zeros_like(xs), np.ones_like(xs)])
                    q = T @ pts
                    sensor_now = T @ np.array([0.0, 0.0, 0.0, 1.0])
                    cols["x"].append(q[0])
                    cols["y"].append(q[1])
                    cols["r"].append(np.hypot(q[0] - sensor_now[0], q[1] - sensor_now[1]))
                    cols["psi"].append(np.arctan2(q[1] - sensor_now[1], q[0] - sensor_now[0]))
                    cols["vr"].append(vr)
                    cols["rcs"].append(P[5])
                    cols["dt"].append(np.full(len(xs), age))
                    cols["sensor"].append(np.full(len(xs), si))
                sd = self.nusc.get("sample_data", sd["prev"]) if sd["prev"] else None
        out = {k: torch.tensor(np.concatenate(v) if v else np.zeros(0), dtype=torch.float32) for k, v in cols.items()}
        out["sensor"] = out["sensor"].long()
        return out

    # ---- GT (optionally cached)
    def gt_signature(self):
        """identity of the GT this dataset reads: '<res>_<ss>_<generator signature>' (autogt source + flags), the suffix of its GT
        cache files; recorded by train.py (runs/<name>/gt.json) and by the evaluation (metrics.json, the suite's manifest)"""
        return f"{self.grid.res}_{self.gt_ss}_{self.gt.signature()}"

    def _gt(self, token):
        # the generator signature is part of the key: GT cached by an older generator is never reused
        path = os.path.join(self.cache_dir, f"{token}_{self.gt_signature()}.npz") if self.cache_dir else None
        if path and os.path.exists(path):
            try:
                with np.load(path, allow_pickle=True) as z:
                    return z["gt"].item()
            except Exception as e:                    # noqa: BLE001  an empty / truncated file of a killed writer: built again
                if not getattr(self, "_told_rebuild", False):     # said once per process: a cache that never reads (permissions) shows
                    print(f"GT cache: {os.path.basename(path)} unreadable ({type(e).__name__}), built again", flush=True)
                    self._told_rebuild = True
        g = self.gt.training_sample(token, self.grid, self.gt_ss)
        if path:
            os.makedirs(self.cache_dir, exist_ok=True)
            small = {k: {n: (a.astype(np.float16) if isinstance(a, np.ndarray) and a.dtype == np.float32 else a)
                         for n, a in v.items()} for k, v in g.items()}
            tmp = f"{path}.tmp.{os.getpid()}"         # a file object: savez would append .npz to a name without it
            try:
                with open(tmp, "wb") as f:
                    np.savez_compressed(f, gt=np.array(small, dtype=object))
                os.replace(tmp, path)                 # atomic: a reader (or a killed writer) never leaves a partial file
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        return g

    def __getitem__(self, i):
        token = self.samples[i]
        sample = self.nusc.get("sample", token)
        frames, track, cur = self._cam_frames(token)
        radar = self._radar(sample, cur)
        last = self.gt.ego_steps(track[max(0, [t["token"] for t in track].index(cur["token"]) - 1)]["timestamp"],
                                 cur["timestamp"], track) or [(1 / 12, 0.0, 0.0)]
        item = {"token": token, "frames": frames, "radar": radar, "ida": self.ida,
                "ego": {"v": last[-1][1], "omega": last[-1][2]}, "timestamp": cur["timestamp"]}
        if self.with_gt:
            item["gt"] = self._gt(token)
        return item


def collate(items):
    """Batch: images (B, F, 3, H, W) and per-frame rig tensors (B, F, ...); per frame the ego steps as tensors (B,) per
    step (the same number of steps for every sample of a frame index: frames with fewer steps - the scene start -
    are padded with zero steps); radar detections concatenated with a sample index b; GT kept as a list."""
    B, Fn = len(items), len(items[0]["frames"])
    out = {"img": torch.stack([torch.stack([f["img"] for f in it["frames"]]) for it in items]),
           "K": torch.stack([torch.stack([f["K"] for f in it["frames"]]) for it in items]),
           "T_cam_car": torch.stack([torch.stack([f["T_cam_car"] for f in it["frames"]]) for it in items]),
           "cam_vec": torch.stack([torch.stack([f["cam_vec"] for f in it["frames"]]) for it in items]),
           "ida": items[0]["ida"], "tokens": [it["token"] for it in items],
           "v": torch.tensor([it["ego"]["v"] for it in items]), "omega": torch.tensor([it["ego"]["omega"] for it in items])}
    steps = []
    for k in range(Fn):
        n = max(len(it["frames"][k]["steps"]) for it in items)
        per = []
        for j in range(n):
            col = [it["frames"][k]["steps"][j] if j < len(it["frames"][k]["steps"]) else (0.0, 0.0, 0.0) for it in items]
            per.append(tuple(torch.tensor([c[m] for c in col], dtype=torch.float32) for m in range(3)))
        steps.append(per)
    out["steps"] = steps
    out["steps_raw"] = [[f["steps"] for f in it["frames"]] for it in items]      # per sample, python floats
    rad = {k: torch.cat([it["radar"][k] for it in items]) for k in items[0]["radar"]}
    rad["b"] = torch.cat([torch.full((len(it["radar"]["x"]),), b, dtype=torch.long) for b, it in enumerate(items)])
    out["radar"] = rad
    if "gt" in items[0]:
        out["gt"] = [it["gt"] for it in items]
    return out


# ----------------------------------------------------------------------------------------------- self-test
def _selftest():
    root = os.path.expanduser("~/data/nuscenes")
    ds = FrontRadarDataset(root, "v1.0-mini", grid=Grid(), n_cam=3, cam_stride=2, with_gt=True)
    assert len(ds) > 300, len(ds)
    it = ds[10]
    fr = it["frames"]
    assert len(fr) == 3 and fr[0]["img"].shape == (3, 256, 704) and fr[0]["steps"] == []
    assert len(fr[1]["steps"]) == 2 and len(fr[2]["steps"]) == 4, [len(f["steps"]) for f in fr]
    dts = [sum(s[0] for s in f["steps"]) for f in fr]
    assert all(0.1 < d < 0.25 for d in [dts[1]]) and 0.25 < dts[2] < 0.45, dts
    assert fr[0]["timestamp"] > fr[1]["timestamp"] > fr[2]["timestamp"]
    assert fr[0]["cam_vec"].shape == (27,)
    rad = it["radar"]
    assert len(rad["x"]) > 50 and float(rad["dt"].max()) <= ds.radar_window + 1e-6 and float(rad["dt"].min()) >= 0
    assert set(rad["sensor"].unique().tolist()) <= set(range(5)) and len(rad["sensor"].unique()) >= 3
    # static compensation is exact: a static radar point of an older sweep, transformed by the poses, is consistent
    # with the steps chain of splat.py (first-order translation per frame) to a few centimetres
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from splat import chain_positions
    p = torch.tensor([[[30.0, 5.0]]], dtype=torch.float64)
    T_old = pose_matrix(ds.nusc.get("ego_pose", ds.nusc.get("sample_data", fr[2]["token"])["ego_pose_token"]))
    T_cur = pose_matrix(ds.nusc.get("ego_pose", ds.nusc.get("sample_data", fr[0]["token"])["ego_pose_token"]))
    q = np.linalg.inv(T_cur) @ T_old @ np.array([30.0, 5.0, 0.0, 1.0])
    c = chain_positions(p, fr[2]["steps"])[0, 0]
    assert math.hypot(c[0] - q[0], c[1] - q[1]) < 0.1, (c, q[:2])
    # GT present with the three keyframes and their steps
    g = it["gt"]
    assert set(g) == {"cur", "prev", "next"} and abs(g["next"]["meta"]["dt"] - 0.5) < 0.1
    assert g["cur"]["drivable_area"].shape == (120, 104)
    # collation of two samples
    b = collate([ds[10], ds[11]])
    assert b["img"].shape == (2, 3, 3, 256, 704) and b["K"].shape == (2, 3, 3, 3)
    assert len(b["steps"]) == 3 and len(b["steps"][2]) == 4 and b["steps"][2][0][0].shape == (2,)
    assert int(b["radar"]["b"].max()) == 1 and len(b["radar"]["b"]) == len(b["radar"]["x"])
    print(f"nuscenes_dataset selftest ok: {len(ds)} mini samples; 3 camera frames 0 / {dts[1]:.3f} / {dts[2]:.3f} s old "
          f"with 0 / 2 / 4 ego steps; {len(rad['x'])} radar detections from {len(rad['sensor'].unique())} radars, age <= "
          f"{float(rad['dt'].max()):.3f} s; pose transform = steps chain within {math.hypot(c[0] - q[0], c[1] - q[1]):.3f} m; "
          f"GT cur / prev / next; collation")


if __name__ == "__main__":
    _selftest()
