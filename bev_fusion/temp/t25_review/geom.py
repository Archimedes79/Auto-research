import os
import sys

import numpy as np

ROOT = "/mnt/c/Auto-research/bev_fusion"
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(1, "/mnt/c/Auto-research")
os.chdir(ROOT)
from nuscenes_dataset import FrontRadarDataset  # noqa: E402
from nuscenes.utils.data_classes import Quaternion  # noqa: E402

ds = FrontRadarDataset(os.path.expanduser("~/data/nuscenes"), "v1.0-mini", n_cam=3, cam_stride=2, with_gt=False, radar_window=0.09, radar_per_frame=True)


def pose(tok):
    p = ds.nusc.get("ego_pose", ds.nusc.get("sample_data", tok)["ego_pose_token"])
    T = np.eye(4)
    T[:3, :3] = Quaternion(p["rotation"]).rotation_matrix
    T[:3, 3] = p["translation"]
    return T


def world(rad, T, static=True):
    keep = np.abs(rad["vr"].numpy()) < 0.3 if static else slice(None)
    pts = np.stack([rad["x"].numpy()[keep], rad["y"].numpy()[keep], np.zeros(int(np.sum(keep))), np.ones(int(np.sum(keep)))])
    return (T @ pts)[:2].T


def nn_med(a, b):
    d = np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1))
    return float(np.median(d.min(1)))


res = []
for i in range(0, len(ds), 25):
    it = ds[i]
    Ts = [pose(f["token"]) for f in it["frames"]]
    dt_ego = float(np.linalg.norm(Ts[0][:2, 3] - Ts[2][:2, 3]))
    if dt_ego < 1.5:
        continue
    w0, w2 = world(it["radar_seq"][0], Ts[0]), world(it["radar_seq"][2], Ts[2])
    w2_wrong = world(it["radar_seq"][2], Ts[0])
    if len(w0) < 5 or len(w2) < 5:
        continue
    res.append((dt_ego, nn_med(w2, w0), nn_med(w2_wrong, w0)))
res = np.array(res)
print("samples", len(res), "ego displacement frame0-frame2 median %.2f m" % np.median(res[:, 0]))
print("median NN distance of static returns, frame2 -> frame0 (world): right pose %.2f m, frame-0 pose used for frame 2 %.2f m" % (np.median(res[:, 1]), np.median(res[:, 2])))
