"""Velocity / flow GT per nuScenes keyframe, always in metres per SECOND (Mirko, 2026-10-30: "proportional to time, not
frame differences").

Nothing here is a per-frame displacement: every velocity comes from time stamps (annotation times, camera times), and a
displacement over any dt is flow * dt (`displace`). It builds on training/gt_bev.py (which is not changed here, so the GT
cache signature stays valid) and gives, in the ego frame at the keyframe (x forward, y left, the grid of gt_bev):
- `flow_ego`  (2, nx, ny): apparent velocity of a STATIC world point at that cell, caused by the ego motion alone:
                -(v_ego + omega x r) = (-vx + omega * y, -vy - omega * x)  (v_ego, omega from the neighbouring 12 Hz camera poses)
- `flow_obj`  (2, nx, ny): the world velocity of the annotated object that covers the cell, in ego axes (0 elsewhere)
- `flow`      = flow_ego + flow_obj: how the content of a cell moves in the ego frame (first order; the splat block uses the
                exact chain of ego steps instead, this is the quantity its velocity head has to find)
- `weight`    (nx, ny): 1 where the flow is known, 0 on the footprints of boxes without lidar / radar points (velocity
                unreliable) and on cells no sensor covers
Object velocity estimators (world frame, from the 2 Hz annotations of one instance, time-normalised):
  "dev"   nuScenes devkit box_velocity (difference of the neighbouring annotations)
  "fit1"  least-squares line through all annotations within +-1.2 s  (slope = velocity)
  "fit2"  least-squares parabola through them (slope at the annotation time); needs 4 points, else fit1
  suffix "+h": vehicles faster than 2 m/s move along their heading (no sideways slip): project the velocity on it when
  it is within 35 degrees of the heading
Stationary instances (gt_bev rule) have velocity 0. Which estimator is used is decided by the radar Doppler check
(`temp/checks/check_flow_radar.py`, mini, 1789 moving observed boxes, 4269 radar points inside): radial error against the
measured Doppler, RMSE / median / within 0.5 m/s - null (v = 0) 6.62 / 3.76 / 14 %, dev 1.54 / 0.21 / 78 %,
fit1 1.38 / 0.21 / 79 %, fit2 1.38 / 0.20 / 80 %, fit1+h 1.36 / 0.21 / 78 %: the fits beat the devkit by 10 % in RMSE, the
parabola and the heading projection add nothing measurable -> default "fit1". The radar sees only the radial component.
Run the self-test (nuScenes mini): python training/gt_flow.py
"""
import math
import os

import numpy as np
from pyquaternion import Quaternion

from gt_bev import NuScenesGT, cell_centres, category_group, pose_matrix

WINDOW = 1.2          # seconds, annotation window of the fits
MIN_SPEED_H = 2.0     # m/s, heading projection only above this speed
MAX_ANGLE_H = math.radians(35.0)


def instance_tracks(src, scene_token):
    """instance token -> list of dict(t [s], xy world, yaw, atok), time ordered (annotation time = keyframe time)."""
    nusc = src.nusc
    out = {}
    tok = nusc.get("scene", scene_token)["first_sample_token"]
    while tok:
        s = nusc.get("sample", tok)
        for atok in s["anns"]:
            a = nusc.get("sample_annotation", atok)
            out.setdefault(a["instance_token"], []).append(
                dict(t=s["timestamp"] * 1e-6, xy=np.array(a["translation"][:2], float),
                     yaw=Quaternion(a["rotation"]).yaw_pitch_roll[0], atok=atok))
        tok = s["next"]
    return out


def fit_velocity(track, j, order):
    """velocity [m/s] at annotation j of a track from a least-squares polynomial of the window; None if < 3 points."""
    t0 = track[j]["t"]
    idx = [i for i, r in enumerate(track) if abs(r["t"] - t0) <= WINDOW]
    if len(idx) < 3:
        return None
    tau = np.array([track[i]["t"] - t0 for i in idx])
    P = np.stack([track[i]["xy"] for i in idx])
    deg = 2 if (order == 2 and len(idx) >= 4) else 1
    A = np.vander(tau, deg + 1, increasing=True)            # columns 1, tau, tau^2
    coef, *_ = np.linalg.lstsq(A, P, rcond=None)
    return coef[1]


def heading_project(v, yaw, group):
    sp = float(np.hypot(*v))
    if group != "vehicle" or sp < MIN_SPEED_H:
        return v
    h = np.array([math.cos(yaw), math.sin(yaw)])
    ang = math.acos(np.clip(abs(float(v @ h)) / sp, 0, 1))
    return float(v @ h) * h if ang < MAX_ANGLE_H else v


def object_velocity(src, tracks, inst, j, method, group="vehicle", stationary=False):
    """world velocity (2,) m/s of the instance at its j-th annotation, or None when unknown."""
    if stationary:
        return np.zeros(2)
    tr = tracks[inst]
    base = method.split("+")[0]
    if base == "dev":
        v = np.asarray(src.nusc.box_velocity(tr[j]["atok"]), float)[:2]
        v = None if not np.isfinite(v).all() else v
    else:
        v = fit_velocity(tr, j, 2 if base == "fit2" else 1)
        if v is None:                                         # short track: devkit's one-sided difference
            v = np.asarray(src.nusc.box_velocity(tr[j]["atok"]), float)[:2]
            v = None if not np.isfinite(v).all() else v
    if v is not None and method.endswith("+h"):
        v = heading_project(v, tr[j]["yaw"], group)
    return v


def ego_motion(src, sample_token, half=1):
    """(vx, vy, omega) of the ego car in its own frame at the keyframe's front-camera time [m/s, m/s, rad/s], from the
    poses `half` camera frames before and after (central difference over the real time stamps)."""
    track = src.camera_track(sample_token)
    sd = src.cam(sample_token)
    i = [s["token"] for s in track].index(sd["token"])
    a, b = track[max(i - half, 0)], track[min(i + half, len(track) - 1)]
    dt = (b["timestamp"] - a["timestamp"]) * 1e-6
    Ta, Tb, Tm = src.ego_pose(a), src.ego_pose(b), src.ego_pose(sd)
    v = Tm[:3, :3].T @ (Tb[:3, 3] - Ta[:3, 3]) / dt            # world displacement over the real time, in the middle frame
    rel = np.linalg.inv(Ta) @ Tb
    return float(v[0]), float(v[1]), float(math.atan2(rel[1, 0], rel[0, 0]) / dt)


def flow_ego_field(grid, vx, vy, omega):
    """apparent velocity (2, nx, ny) of static world points in the ego frame: -(v + omega x r)."""
    gx, gy = cell_centres(grid)
    return np.stack([-vx + omega * gy, -vy - omega * gx]).astype(np.float32)


def displace(flow, dt):
    """displacement over any time dt [s] (first order): flow [m/s] * dt."""
    return flow * dt


def keyframe_flow(src, gt, sample_token, tracks=None, method="fit1"):
    """flow layers for a keyframe GT dict `gt` from NuScenesGT.keyframe_gt (same sample). Returns a dict with flow_ego,
    flow_obj, flow, weight, ego (vx, vy, omega) and per box velocity in `box_vel` (world, m/s; None = unknown)."""
    sample = src.nusc.get("sample", sample_token)
    tracks = tracks or instance_tracks(src, sample["scene_token"])
    T_e2g = gt["meta"]["T_e2g"]
    R_g2e = np.linalg.inv(T_e2g)[:2, :2]
    grid = Grid_from(gt)
    vx, vy, om = ego_motion(src, sample_token)
    f_ego = flow_ego_field(grid, vx, vy, om)
    f_obj = np.zeros_like(f_ego)
    weight = (1 - gt["ignore"].clip(0, 1) * (gt["ignore"] > 0)).astype(np.float32)
    box_vel = []
    idx_of = {}
    for tr_inst, tr in tracks.items():
        for j, r in enumerate(tr):
            idx_of[r["atok"]] = (tr_inst, j)
    for bi, b in enumerate(gt["meta"]["boxes"]):
        inst, j = idx_of[b["token"]]
        v = object_velocity(src, tracks, inst, j, method, b["group"], b["stationary"])
        box_vel.append(v)
        cells = gt["instance"] == bi
        if v is None:
            weight[cells] = 0.0
            continue
        ve = R_g2e @ v
        f_obj[0][cells], f_obj[1][cells] = ve[0], ve[1]
    return dict(flow_ego=f_ego, flow_obj=f_obj, flow=f_ego + f_obj, weight=weight, ego=(vx, vy, om), box_vel=box_vel)


def Grid_from(gt):
    from lift_pd import Grid
    res = float(gt["meta"]["res"])
    return Grid(res=res)


# ----------------------------------------------------------------------------------------------- self-test
def _selftest():
    from lift_pd import Grid
    g = Grid()
    src = NuScenesGT(os.path.expanduser("~/data/nuscenes"), "v1.0-mini")
    scene = next(s for s in src.nusc.scene if s["name"] == "scene-0061")
    tok = scene["first_sample_token"]
    for _ in range(8):
        tok = src.nusc.get("sample", tok)["next"]
    gt = src.keyframe_gt(tok, g)
    tracks = instance_tracks(src, scene["token"])
    fl = keyframe_flow(src, gt, tok, tracks)
    vx, vy, om = fl["ego"]
    # 1 flow scales with time: ego velocity from +-1 and +-3 camera frames agree (no frame-count dependence)
    a, b = ego_motion(src, tok, 1), ego_motion(src, tok, 3)
    assert abs(a[0] - b[0]) < 0.6 and abs(a[2] - b[2]) < 0.05, (a, b)
    # 2 ego flow against the exact ego pose: a static point displaced by flow * dt = the exact pose change, at dt = one
    #    camera frame (first-order error < 1 cm at 60 m) - and the error at 0.5 s, which is why the splat uses the exact chain
    track = src.camera_track(tok)
    sd = src.cam(tok)
    i = [s["token"] for s in track].index(sd["token"])
    nxt = track[i + 1]
    dt = (nxt["timestamp"] - sd["timestamp"]) * 1e-6
    T = np.linalg.inv(src.ego_pose(nxt)) @ src.ego_pose(sd)    # point in ego(sd) -> ego(nxt)
    P = np.array([[59.0, 20.0, 0, 1], [30.0, -10.0, 0, 1], [10.0, 5.0, 0, 1]])
    exact = (P @ T.T)[:, :2]
    ix = [int((p[0] - g.x_min) / g.res) for p in P]
    iy = [int((p[1] - g.y_min) / g.res) for p in P]
    f = np.array([[fl["flow_ego"][0, a_, b_], fl["flow_ego"][1, a_, b_]] for a_, b_ in zip(ix, iy)])
    gx, gy = cell_centres(g)
    cc = np.array([[gx[a_, b_], gy[a_, b_]] for a_, b_ in zip(ix, iy)])
    err = np.abs((cc + f * dt) - (np.c_[cc, np.zeros(3), np.ones(3)] @ T.T)[:, :2]).max()
    assert err < 0.05, err                                      # cell-centre points, 1 camera frame: first order + pose noise
    # 3 stationary objects have zero flow_obj; moving ones nonzero; weight 0 on ignored footprints
    bx = gt["meta"]["boxes"]
    for bi, b in enumerate(bx):
        cells = gt["instance"] == bi
        if cells.any() and b["stationary"]:
            assert np.abs(fl["flow_obj"][:, cells]).max() == 0
    moving = [bi for bi, v in enumerate(fl["box_vel"]) if v is not None and np.hypot(*v) > 1.0]
    assert moving, "scene-0061 keyframe 8 has moving vehicles"
    assert fl["weight"].min() == 0 and fl["weight"].max() == 1
    # 4 displacement = flow * dt for any dt: velocity of a moving box times 0.5 s lands near its next annotation
    errs = []
    for bi in moving:
        b = bx[bi]
        nxt_ann = src.nusc.get("sample_annotation", b["token"])["next"]
        if not nxt_ann:
            continue
        a2 = src.nusc.get("sample_annotation", nxt_ann)
        dt2 = (src.nusc.get("sample", a2["sample_token"])["timestamp"] - src.nusc.get("sample", gt["meta"]["sample"])["timestamp"]) * 1e-6
        pred = b["world"][:2] + fl["box_vel"][bi] * dt2
        errs.append(np.linalg.norm(pred - np.array(a2["translation"][:2])))
    assert errs and np.median(errs) < 1.0, errs
    print(f"gt_flow selftest ok: ego v {vx:.1f} m/s, omega {om:+.3f} rad/s from +-1 and +-3 frames alike; ego flow vs exact pose "
          f"within {err * 100:.1f} cm per camera frame; {len(moving)} moving boxes, velocity x dt lands {np.median(errs):.2f} m (median) "
          f"from the next annotation after {dt2:.2f} s; stationary flow 0; weight 0 on {int((fl['weight'] == 0).sum())} cells")


if __name__ == "__main__":
    _selftest()
