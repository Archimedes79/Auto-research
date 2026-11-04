"""Self-test of the whole autoGT suite (nuScenes mini needed for the real-data part): python -m autogt selftest

1 flow estimator on a synthetic world with known truth (autogt/flow.py)
2 rasteriser: exact areas (also a big clipped polygon with a hole), line distance, occlusion
3 real keyframes: a road under the ego car in every map location, 0.25 m pooled = 0.5 m, static layers of the next keyframe =
  the current ones moved by the exact ego motion, instance layer consistent, stationary boxes fixed in the world,
  ego steps add up to the keyframe gap
4 velocity / flow layers: flow = flow_ego + velocity, ego flow = the exact pose change per camera frame, stationary velocity 0,
  moving boxes' speeds close to the devkit's, unknown velocity never a silent zero (weight 0), weights 0 on ignored boxes,
  a 12 Hz sweep frame (sd=) gives a ground-consistent frame, moving boxes at the camera time (centre = annotation + v * dt,
  stationary / unknown-velocity boxes unchanged, boxes_at="annotation" restores the annotations)
5 the build command writes and reloads a cache that is keyed by the generator signature; `load` / `current_signature` of the
  top-level API; AGENTS.md names every layer, meta entry and box field of a real keyframe
"""
import json
import math
import os
import tempfile

import cv2
import numpy as np

from . import flow as flowmod
from .grid import GridSpec
from .layers import Raster, cell_centres, occlusion
from .nuscenes_gt import NuScenesGT


def _same_gt(a, b):
    """two keyframes (dict of layers + meta) agree in every array byte and every meta entry (checks._feed, the digest of `autogt digest`)"""
    import hashlib
    from .checks import _feed
    ha, hb = hashlib.sha1(), hashlib.sha1()
    _feed(ha, a)
    _feed(hb, b)
    return ha.hexdigest() == hb.hexdigest()


def run(root=None, verbose=True):
    import torch
    import torch.nn.functional as F
    flowmod._selftest()                                                                               # 1
    # 1b ego velocity at the ends of a sequence (v6): a drive at 10 m/s along world x with a yaw rate of 0.2 rad/s, poses every 0.1 s over 1 s; the
    #    first and the last pose give the same speed as the middle (the clamped window of v5 gave half of it)
    tt_ = np.arange(0, 1.01, 0.1)
    ego_ = flowmod.Ego(t=tt_, x=10.0 * tt_, y=np.zeros(11), yaw=0.2 * tt_)
    for t_ in (0.0, 0.03, 0.5, 0.97, 1.0):
        v_ = flowmod.ego_velocity(ego_, t_)
        assert abs(np.hypot(v_[0], v_[1]) - 10.0) < 1e-3 and abs(v_[2] - 0.2) < 1e-6, (t_, v_)       # speed (the car drives along world x, the ego frame turns)
    assert flowmod.ego_velocity(flowmod.Ego(t=[0.0], x=[0.0], y=[0.0], yaw=[0.0]), 0.0) == (0.0, 0.0, 0.0)
    g = GridSpec()
    R = Raster(g, 4)                                                                                  # 2
    sq = np.array([[10.0, -1.0], [12.0, -1.0], [12.0, 1.0], [10.0, 1.0]])
    cov = R.coverage(R.fill(R.canvas(), [(sq, [])]))
    assert abs(cov.sum() * g.res ** 2 - 4.0) < 0.15, cov.sum() * g.res ** 2
    assert cov[20:24, 50:54].min() > 0.99 and cov[:19].max() == 0
    from shapely.geometry import Polygon as ShPolygon, box as sh_box
    th = np.linspace(0, 2 * np.pi, 201)[:-1]
    ring = np.c_[30 + 80 * np.cos(th), 80 * np.sin(th)]
    hole = np.c_[30 + 10 * np.cos(th[::2]), 10 * np.sin(th[::2])][::-1]
    covb = R.coverage(R.fill(R.canvas(), [(ring, [hole])]))
    want = ShPolygon(ring, [hole]).intersection(sh_box(g.x_min, g.y_min, g.x_max, g.y_max)).area
    assert abs(covb.sum() * g.res ** 2 - want) / want < 0.003, (covb.sum() * g.res ** 2, want)
    img = R.lines(R.canvas(), [np.array([[5.0, 0.25], [50.0, 0.25]])])
    dist = np.minimum(R.at_centres(cv2.distanceTransform((img == 0).astype(np.uint8), cv2.DIST_L2, 5) * R.s), 2)
    assert dist[40, 52] < 0.1 and abs(dist[40, 54] - 1.0) < 0.15, (dist[40, 52], dist[40, 54])
    gx, gy = cell_centres(g)
    occ = occlusion(gx, gy, (0.0, 0.0), [np.array([[20.0, -1], [24, -1], [24, 1], [20, 1]])])
    assert occ[60, 52] and not occ[60, 70] and not occ[44, 52]

    root = os.path.expanduser(root or "~/data/nuscenes")                                              # 3
    gt = NuScenesGT(root, "v1.0-mini")
    scene = gt.nusc.scene[0]
    s1 = gt.nusc.get("sample", scene["first_sample_token"])["next"]
    ts = gt.training_sample(s1, g)
    cur, nxt = ts["cur"], ts["next"]
    assert cur["drivable_area"].mean() > 0.05 and cur["camera"].mean() > 0.2
    seen = {}
    for sc in gt.nusc.scene:                      # a road under the ego car in every location (null-token road polygon)
        loc = gt.nusc.get("log", sc["log_token"])["location"]
        if loc not in seen:
            k0 = gt.keyframe_gt(sc["first_sample_token"], g)
            seen[loc] = float(k0["drivable_area"][0:6, 50:54].mean())
            assert seen[loc] > 0.5, (loc, seen[loc])
    fine = gt.keyframe_gt(s1, GridSpec(res=0.25))
    for layer in ("drivable_area", "walkway", "vehicle"):
        pooled = fine[layer].reshape(g.nx, 2, g.ny, 2).mean((1, 3))
        assert np.abs(pooled - cur[layer]).mean() < 0.01, layer
    T = np.linalg.inv(cur["meta"]["T_e2g"]) @ nxt["meta"]["T_e2g"]
    P = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size), np.ones(gx.size)], 1) @ T.T
    gxn = (P[:, 1] - g.y_min) / (g.y_max - g.y_min) * 2 - 1
    gyn = (P[:, 0] - g.x_min) / (g.x_max - g.x_min) * 2 - 1
    grid_s = torch.tensor(np.stack([gxn, gyn], 1), dtype=torch.float32).view(1, g.nx, g.ny, 2)
    moved = F.grid_sample(torch.tensor(cur["drivable_area"])[None, None], grid_s, align_corners=False)[0, 0].numpy()
    inside = (np.abs(gxn) < 0.98) & (np.abs(gyn) < 0.98)
    a, b = moved.reshape(-1)[inside] > 0.5, nxt["drivable_area"].reshape(-1)[inside] > 0.5
    iou = (a & b).sum() / max(1, (a | b).sum())
    assert iou > 0.95, iou
    bxs, ii = cur["meta"]["boxes"], cur["instance"]
    assert (ii >= 0).any()
    for i in np.unique(ii[ii >= 0]):
        assert bxs[i]["observed"] and cur[bxs[i]["group"]][ii == i].min() > 0, i
    fixed = gt.stationary_instances(scene["token"])
    both = [(b_, c) for b_ in cur["meta"]["boxes"] for c in nxt["meta"]["boxes"] if b_["instance"] == c["instance"] and b_["stationary"]]
    assert both and all(np.allclose(b_["world"], c["world"]) for b_, c in both), "stationary boxes must not move"
    assert abs(sum(s[0] for s in ts["next"]["meta"]["steps"]) - ts["next"]["meta"]["dt"]) < 1e-6
    assert abs(sum(s[0] for s in ts["prev"]["meta"]["steps"]) - ts["prev"]["meta"]["dt"]) < 1e-6

    # 4 velocity / flow layers on a scene with moving traffic (scene-0061, keyframe 8)
    sc61 = next(s for s in gt.nusc.scene if s["name"] == "scene-0061")
    tok = sc61["first_sample_token"]
    for _ in range(8):
        tok = gt.nusc.get("sample", tok)["next"]
    k = gt.keyframe_gt(tok, g)
    assert np.allclose(k["flow"], k["flow_ego"] + k["velocity"], atol=1e-5)
    bx = k["meta"]["boxes"]
    moving = [i for i, b_ in enumerate(bx) if b_["observed"] and b_["vel_known"] and not b_["stationary"] and np.hypot(*b_["vel"]) > 1.0]
    assert moving and all(b_["vel"][0] == 0 and b_["vel"][1] == 0 for b_ in bx if b_["stationary"])
    diffs = [abs(np.hypot(*bx[i]["vel"]) - np.hypot(*np.nan_to_num(gt.nusc.box_velocity(bx[i]["token"]))[:2])) for i in moving]
    assert np.median(diffs) < 0.5, np.median(diffs)                                      # close to the devkit's speeds
    cells = (k["instance"] == moving[0]) & (k[bx[moving[0]]["group"]] > 0)
    assert cells.any() and np.allclose(k["velocity"][:, cells], bx[moving[0]]["vel"][:, None], atol=1e-3)   # edge cells too
    # v6: a cell with a velocity names its box; the sigma of an edge cell is the box's sigma, not a coverage-scaled one
    assert (k["instance"][np.abs(k["velocity"]).sum(0) > 0] >= 0).all(), "velocity without an instance"
    sg_ = bx[moving[0]]["vel_sigma"]
    assert not np.isfinite(sg_) or np.allclose(k["flow_sigma"][cells], sg_, atol=1e-4), (sg_, np.unique(k["flow_sigma"][cells]))
    wide = NuScenesGT(root, "v1.0-mini", nusc=gt.nusc, flow_config=flowmod.Config(ego_half_window=0.3))      # the option reaches the dense ego flow
    assert wide.signature() != gt.signature() and not np.array_equal(wide.keyframe_gt(tok, g)["flow_ego"], k["flow_ego"])
    no_obj = (k["vehicle"] + k["pedestrian"] + k["two_wheeler"]) == 0                 # no observed box in the cell: no velocity
    assert np.abs(k["velocity"][:, no_obj]).max() == 0 and (k["flow_weight"][k["ignore"] > 0.9] == 0).all()
    assert k["flow_weight"].min() == 0 and k["flow_weight"].max() == 1
    # boxes at the camera time: constant velocity over (camera - annotation time); stationary / unknown boxes stay
    ann = NuScenesGT(root, "v1.0-mini", nusc=gt.nusc, boxes_at="annotation")
    assert ann.signature() != gt.signature()
    ka = ann.keyframe_gt(tok, g)
    t_cam = (gt.cam(tok)["timestamp"] - gt.nusc.get("sample", tok)["timestamp"]) * 1e-6
    assert -0.06 < t_cam < -0.01 and k["meta"]["boxes_at"] == "camera" and ka["meta"]["boxes_at"] == "annotation", t_cam
    Re = k["meta"]["T_e2g"][:2, :2]
    n_moved = 0
    for b_, a_ in zip(k["meta"]["boxes"], ka["meta"]["boxes"]):
        assert b_["token"] == a_["token"] and a_["shift_s"] == 0.0 and np.allclose(a_["world"], b_["world_ann"])
        if b_["stationary"] or not b_["vel_known"]:
            assert b_["shift_s"] == 0.0 and np.allclose(b_["world"], b_["world_ann"]) and np.allclose(b_["center"], a_["center"])
        else:
            n_moved += 1
            assert abs(b_["shift_s"] - t_cam) < 1e-12 and np.allclose(b_["shift"], b_["vel"] * t_cam, atol=1e-9)
            # (the 2 x 2 block of a tilted ego rotation is not exactly orthogonal: millimetres)
            assert np.allclose(b_["world"][:2] - b_["world_ann"][:2], Re @ b_["vel"] * t_cam, atol=3e-3)
            assert b_["world"][2] == b_["world_ann"][2] and b_["yaw"] == a_["yaw"] and list(b_["size"]) == list(a_["size"])
            assert np.allclose(b_["center"][:2] - a_["center"][:2], b_["shift"], atol=1e-9)
    assert n_moved and any(b_["shift_s"] != 0.0 for b_ in k["meta"]["boxes"])
    assert np.array_equal(k["drivable_area"], ka["drivable_area"]) and not np.array_equal(k["vehicle"], ka["vehicle"])
    short = gt.keyframe_gt(sc61["first_sample_token"], g)           # first keyframe: tracks without history
    assert all(b_["vel_known"] or (b_["vel"] == 0).all() for b_ in short["meta"]["boxes"])     # never a silent nonzero
    unk = [b_ for b_ in short["meta"]["boxes"] if not b_["vel_known"]]
    assert unk and all(b_["shift_s"] == 0.0 and np.array_equal(b_["world"], b_["world_ann"]) for b_ in unk), len(unk)   # not moved
    track = gt.camera_track(tok)                                  # ego flow = the exact pose change per camera frame
    sd = gt.cam(tok)
    i = [s["token"] for s in track].index(sd["token"])
    nx_ = track[i + 1]
    dt = (nx_["timestamp"] - sd["timestamp"]) * 1e-6
    Tn = np.linalg.inv(gt.ego_pose(nx_)) @ gt.ego_pose(sd)
    ix, iy = 59, 60
    c = np.array([gx[ix, iy], gy[ix, iy]])
    err = np.abs((c + k["flow_ego"][:, ix, iy] * dt) - (np.r_[c, 0, 1] @ Tn.T)[:2]).max()
    assert err < 0.05, err
    # 4a coverage (v7, autogt/coverage.py): the camera layer is the old ground-plane projection, boxes hide with their height, every box
    #    carries the observed shares of its footprint, and the helpers have known answers
    from . import coverage as C
    from .nuscenes_adapter import pose_matrix
    cs_ = gt.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    K_, Tc_ = np.array(cs_["camera_intrinsic"]), pose_matrix(cs_)
    P_ = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size), np.ones(gx.size)], 1) @ np.linalg.inv(Tc_).T
    old_cam = (P_[:, 2] > 0.5) & (K_[0, 0] * P_[:, 0] / np.maximum(P_[:, 2], 1e-6) + K_[0, 2] >= 0) & (K_[0, 0] * P_[:, 0] / np.maximum(P_[:, 2], 1e-6) + K_[0, 2] < sd["width"]) \
        & (K_[1, 1] * P_[:, 1] / np.maximum(P_[:, 2], 1e-6) + K_[1, 2] >= 0) & (K_[1, 1] * P_[:, 1] / np.maximum(P_[:, 2], 1e-6) + K_[1, 2] < sd["height"])
    assert np.array_equal(k["camera"] > 0, old_cam.reshape(gx.shape)), "the camera layer is the ground-plane projection of CAM_FRONT"
    old_occ = occlusion(gx, gy, Tc_[:2, 3], [b_["corners"] for b_ in bx if b_["observed"]])
    assert ((k["occluded"] > 0) <= old_occ).all() and (k["occluded"] > 0).sum() <= old_occ.sum(), "soft hiding hides a subset of the old rule"
    for b_ in bx:
        s_ = (b_["share_camera"], b_["share_radar"], b_["share_fusion"])
        assert all(0.0 <= x_ <= 1.0 for x_ in s_) and s_[2] <= min(s_[0], s_[1]) + 1e-9, s_
        assert isinstance(b_["num_lidar_pts"], int) and isinstance(b_["num_radar_pts"], int)
        assert (b_["num_lidar_pts"] + b_["num_radar_pts"] > 0) == b_["observed"]
    assert any(0 < b_["share_fusion"] < 1 for b_ in bx) and any(b_["share_fusion"] == 0 for b_ in bx), "shares are not all 0 or 1"
    fus_ = C.fusion_cells(k)
    assert np.array_equal(fus_, (k["camera"] > 0) & (k["occluded"] == 0) & (k["radar"] > 0))
    assert np.array_equal(C.coverage_for(k, "map"), fus_) and np.array_equal(C.coverage_for(k, "radar_aux"), k["radar"] > 0)
    assert np.array_equal(C.coverage_for(k, "camera_aux"), (k["camera"] > 0) & (k["occluded"] == 0))
    cam_ = np.array([1.5, 0.0, 1.6])                                                   # known answers on the synthetic scene
    low = (np.array([[20.0, -1], [24, -1], [24, 1], [20, 1]]), 0.0, 0.5)               # a 0.5 m high box, x 20..24: its shadow ends at
    hid = C.hidden_cells(gx, gy, cam_, [low])                                          # 1.5 + (24 - 1.5) * 1.6 / (1.6 - 0.5) = 34.2 m
    row = gx[:, 52][hid[:, 52]]
    assert 24 < row.min() < 24.6 and 33.5 < row.max() < 34.5, (row.min(), row.max())
    tall = (low[0], 0.0, 50.0)
    assert np.array_equal(C.hidden_cells(gx, gy, cam_, [tall]), C.hidden_cells(gx, gy, cam_, [tall], soft=False)), "infinite height = the old rule"
    assert not C.hidden_cells(gx, gy, cam_, [low])[(gx > 20.3) & (gx < 23.7) & (np.abs(gy) < 0.7)].any(), "a box does not hide its own cells"
    Kt_, Tt_ = np.array([[1260.0, 0, 800], [0, 1260.0, 450], [0, 0, 1]]), np.eye(4)
    Tt_[:3, :3], Tt_[:3, 3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float), cam_
    car_ = (np.array([[10.0, -1], [14, -1], [14, 1], [10, 1]]), 0.0, 1.5)
    ped_ = {"corners": np.array([[17.0, -0.3], [17.6, -0.3], [17.6, 0.3], [17.0, 0.3]]), "size": [0.6, 0.6, 1.7]}
    kid_ = dict(ped_, size=[0.6, 0.6, 1.0])
    assert C.object_shares(ped_, [car_], cam_, Kt_, Tt_, (1600, 900), [], g)["camera"] == 1.0, "a 1.7 m pedestrian behind a 1.5 m car is seen"
    assert C.object_shares(kid_, [car_], cam_, Kt_, Tt_, (1600, 900), [], g)["camera"] == 0.0, "a 1.0 m high object behind it is hidden"
    half_ = np.array([[10.0, -1], [12, -1], [12, 1], [10, 1]])
    assert C.footprint_share(half_, gy > 0, g) == 0.5 and C.footprint_share(np.array([[58.0, -1], [62, -1], [62, 1], [58, 1]]), np.ones((g.nx, g.ny), bool), g) == 0.5
    fan_ = C.radar_cells(np.array([10.0, 10.0, 80.0, -5.0]), np.array([0.0, 20.0, 0.0, 0.0]), [np.eye(4)])
    assert fan_.tolist() == [True, False, False, False], fan_                          # in the fan, 63 deg off axis, beyond 70 m, behind
    rig_ = C.radar_cells(gx, gy, C.rig_radar_poses())                                  # the table of the rig = the fans of a real keyframe
    assert (rig_ != (k["radar"] > 0)).mean() < 0.01 and len(C.rig_radar_poses()) == 5, (rig_ != (k["radar"] > 0)).mean()
    for b_ in bx:                                                                      # one rule for ground truth and detections: the stored
        assert C.shares_in(k, b_["corners"], b_["size"][2])["fusion"] == b_["share_fusion"], b_["token"]   # share is what a detection of that footprint gets
    assert C.shares_in(k, bx[0]["corners"], group="pedestrian")["fusion"] >= 0.0 and set(k["meta"]) >= {"image_size", "radar_poses"}
    # the blocking test against an exact reference (shapely: the segment clipped by the footprint, then the height interval); the cases
    # include long boxes right next to the camera, where a pre-filter by angle once dropped points (found by the review of 2026-11-01)
    from shapely.geometry import LineString, Point
    rng_ = np.random.default_rng(7)

    def ref_blocked(cam, pt, fp, z_hi):
        poly = ShPolygon(fp)
        if fp[:, 0].max() < cam[0] or poly.buffer(1e-9).contains(Point(pt[:2])):
            return False
        inter = LineString([cam[:2], pt[:2]]).intersection(poly)
        if inter.is_empty:
            return False
        ts = [np.hypot(*(np.array(c_) - cam[:2])) / np.hypot(*(pt[:2] - cam[:2])) for c_ in inter.coords]
        zz = [cam[2] + t_ * (pt[2] - cam[2]) for t_ in (min(ts), max(ts))]
        return max(zz) >= 0 and min(zz) <= z_hi
    repro = np.array([[3, 1.2], [3, 3.7], [-5, 3.7], [-5, 1.2]], float)
    assert C.blocked(np.array([[20.0, 8.1, 0.0]]), np.array([0, 0, 1.5]), [(repro, 0.0, 3.5)])[0], "long box next to the camera"
    cam3_, bad_n, total_n = np.array([0.0, 0.0, 1.5]), 0, 0
    for _ in range(120):
        c_ = rng_.uniform([-2, -6], [25, 6])
        L_, W_, yw_ = rng_.uniform(1.5, 12), rng_.uniform(0.6, 2.6), rng_.uniform(0, np.pi)
        R_ = np.array([[np.cos(yw_), -np.sin(yw_)], [np.sin(yw_), np.cos(yw_)]])
        fp_ = c_ + np.array([[L_ / 2, W_ / 2], [L_ / 2, -W_ / 2], [-L_ / 2, -W_ / 2], [-L_ / 2, W_ / 2]]) @ R_.T
        if ShPolygon(fp_).contains(Point(0, 0)):
            continue
        z_hi_ = rng_.uniform(0.5, 3.5)
        pts_ = np.c_[rng_.uniform([0, -15], [40, 15], (40, 2)), rng_.choice([0.0, 0.8, 1.7], 40)]
        got_ = C.blocked(pts_, cam3_, [(fp_, 0.0, z_hi_)])
        want_ = np.array([ref_blocked(cam3_, p_, fp_, z_hi_) for p_ in pts_])
        bad_n, total_n = bad_n + int((got_ != want_).sum()), total_n + len(pts_)
    assert bad_n == 0 and total_n > 3000, (bad_n, total_n)
    ped_pts = [C.footprint_share(np.array([[10.0, -0.3], [10.6, -0.3], [10.6, 0.3], [10.0, 0.3]]) + [0, s_], gy > 0.0, g) for s_ in np.linspace(-0.1, 0.1, 9)]
    exact_ = [(0.3 + s_) / 0.6 for s_ in np.linspace(-0.1, 0.1, 9)]
    assert max(abs(a_ - b_) for a_, b_ in zip(ped_pts, exact_)) < 0.05, "a 0.6 m footprint: share within 0.05 of the exact area share"
    try:
        C.footprint_samples(np.array([[0, 0], [1, 0], [0, 1], [1, 1.0]]))
        raise AssertionError("corners that are not consecutive must be refused")
    except ValueError:
        pass
    sw = gt.keyframe_gt(tok, g, sd=nx_)                           # a 12 Hz sweep frame
    assert abs(sw["meta"]["timestamp"] - nx_["timestamp"]) == 0 and sw["drivable_area"].mean() > 0.05
    # boxes at a sweep 0.2 s after the annotation (sd=): the constant-velocity shift must agree with the straight line between the
    # annotations of this and the next keyframe - independent of the sign of v (a wrong direction would be off by 2 * |shift|)
    nxt_tok = gt.nusc.get("sample", tok)["next"]
    t0a, t1a = (gt.nusc.get("sample", x)["timestamp"] for x in (tok, nxt_tok))
    sd3 = track[i + 3]
    sw3 = gt.keyframe_gt(tok, g, sd=sd3)
    f = (sd3["timestamp"] - t0a) / (t1a - t0a)
    nxt_boxes = {b_["instance"]: b_ for b_ in ann.keyframe_gt(nxt_tok, g)["meta"]["boxes"]}
    errs, wrong = [], []
    for b_ in sw3["meta"]["boxes"]:
        o_ = nxt_boxes.get(b_["instance"])
        if b_["stationary"] or not b_["vel_known"] or o_ is None or np.hypot(*b_["vel"]) < 3:
            continue
        lin = b_["world_ann"][:2] + f * (o_["world"][:2] - b_["world_ann"][:2])
        errs.append(np.linalg.norm(b_["world"][:2] - lin))
        wrong.append(np.linalg.norm(2 * b_["world_ann"][:2] - b_["world"][:2] - lin))            # the opposite sign
        assert abs(b_["shift_s"] - (sd3["timestamp"] - t0a) * 1e-6) < 1e-12
    assert len(errs) >= 3 and np.median(errs) < 0.3 and np.median(wrong) > 2 * np.median(errs), (len(errs), np.median(errs), np.median(wrong))

    # 4b the GT versions (D29): the table is complete and in order, and THIS source has a row: an edit of a signature file without a new
    #    row (and, after the commit, a tag autogt/vN) fails here
    from . import current_version, versions as V
    import ast                                                # every module of the package is part of the signature or listed as not shaping the GT
    from .nuscenes_gt import SIGNATURE_FILES
    here_ = os.path.dirname(os.path.abspath(__file__))
    todo_, reach_ = ["nuscenes_gt"], set()
    while todo_:
        mod_ = todo_.pop()
        if mod_ in reach_:
            continue
        reach_.add(mod_)
        for n_ in ast.walk(ast.parse(open(os.path.join(here_, mod_ + ".py"), encoding="utf-8").read())):
            if isinstance(n_, ast.ImportFrom) and n_.level == 1:
                todo_ += [n_.module] if n_.module else [a_.name for a_ in n_.names]
    assert {m_ + ".py" for m_ in reach_} == set(SIGNATURE_FILES), (
        f"the modules nuscenes_gt.py imports are {sorted(reach_)} but SIGNATURE_FILES is {SIGNATURE_FILES}: a module that shapes the GT must be hashed")
    loose_ = sorted(set(f_ for f_ in os.listdir(here_) if f_.endswith(".py")) - set(SIGNATURE_FILES) - set(V.NOT_GENERATOR))
    assert not loose_, (
        f"{loose_} is neither in SIGNATURE_FILES (nuscenes_gt.py) nor in versions.NOT_GENERATOR: decide whether it shapes the GT "
        "(then it is hashed and a change needs a new version row) or not (then list it)")
    rows_ = V.table()
    assert [r["version"] for r in rows_] == [f"v{i + 1}" for i in range(len(rows_))] and len({r["signature"] for r in rows_}) == len(rows_)
    assert V.version_of(gt.signature()) is not None, (f"the generator signature {gt.signature()} has no row in autogt/versions.csv: add one, "
                                                      "commit, tag autogt/v<N> (decision D29)")
    store_sig = V.canonical(gt.signature())                       # the folder of the stored GT: the signature itself, or the older one whose GT this code makes
    assert current_version() == V.version_of(store_sig) and current_version(boxes_at="annotation").startswith(current_version() + "+")
    assert store_sig == V.canonical(store_sig) and V.canonical("deadbeef00") == "deadbeef00", "same_gt_as ends at a signature that is its own store"
    assert all(V.row_of(r["same_gt_as"]) for r in rows_ if r.get("same_gt_as")), "same_gt_as names a version of the table"
    assert V.version_of_key("0.5_4_" + gt.signature()) == V.version_of(store_sig) == V.version_of_key("0.5_x0_40_y-20_20_4_" + gt.signature())
    assert V.version_of_key("0.5_4_deadbeef00") is None and V.version_of_key(gt.signature()) == V.version_of(store_sig)
    import subprocess                                       # the older rows are final, so their git tag exists (the last row is tagged after its commit)
    repo_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.exists(os.path.join(repo_, ".git")):
        try:
            tags_ = set(subprocess.run(["git", "-C", repo_, "tag", "-l", "autogt/*"], capture_output=True, text=True, timeout=30).stdout.split())
        except (OSError, subprocess.SubprocessError):
            tags_ = set()
        lost_ = [r["tag"] for r in rows_[:-1] if r["tag"] not in tags_]
        assert not tags_ or not lost_, f"versions.csv names git tags that do not exist: {lost_} (git tag -a autogt/vN <commit>)"
        if tags_ and rows_[-1]["tag"] not in tags_:
            print(f"note: the newest version {rows_[-1]['version']} has no git tag {rows_[-1]['tag']} yet: commit, then tag it")
    from . import suite as suite_mod2                       # the key of a process is computed once: a second call does not hash the files again
    suite_mod2._SIGNATURES.clear()
    first_, gt_sig_ = suite_mod2.current_signature(), store_sig
    real_sig = suite_mod2.NuScenesGT.signature

    def _no_hashing(self):
        raise AssertionError("current_signature hashed the source files a second time")
    suite_mod2.NuScenesGT.signature = _no_hashing
    try:
        assert first_ == gt_sig_ and suite_mod2.current_signature() == first_
    finally:
        suite_mod2.NuScenesGT.signature = real_sig

    # 5 the build command (cache keyed by the signature)
    from .cli import build, load_keyframe
    with tempfile.TemporaryDirectory() as d:
        n = build(root, "v1.0-mini", d, scenes=["scene-0061"], stride=20, workers=1, res=0.5, verbose=False)
        sig_dir = os.path.join(d, store_sig)
        files = sorted(os.listdir(sig_dir))
        assert n == len(files) - 1 and "index.json" in files, files
        tok2 = [f for f in files if f.endswith(".npz")][0][:-4]
        g2 = load_keyframe(sig_dir, tok2)
        assert g2["flow"].shape == (2, g.nx, g.ny) and g2["meta"]["sample"]
        from . import current_signature, load as load_stored                          # the two helpers of the top-level API
        assert current_signature() == store_sig and current_signature(boxes_at="annotation") == ann.signature()
        assert np.array_equal(load_stored(tok2, data=d)["flow"], g2["flow"])
        from . import missing                                                          # the store covers a token that is there, not another
        assert missing([tok2, "0" * 32], data=d) == ["0" * 32] and missing([tok2], data=d) == []
        from .view import stored                                                       # a picture straight from the stored GT
        import matplotlib.pyplot as plt
        fig = stored(tok2, None, data=d)
        assert len(fig.axes) >= 5 and fig.axes[0].images, "stored-GT figure"
        plt.close(fig)
        from .view import motion_arrows                                                # the arrows of the motion panel point the way the car goes
        fl = np.zeros((2, g.nx, g.ny), np.float32)
        fl[0, 40:48, 60:64], fl[1, 40:48, 60:64] = 5.0, 3.0                            # 5 m/s forward and 3 m/s to the left (+y)
        fig_a, ax_a = plt.subplots()
        ax_a.set_xlim(26, -26)                                                         # the panel's axes: y inverted (left = +y), x forward up
        ax_a.set_ylim(0, 60)
        q = motion_arrows(ax_a, fl)
        fig_a.canvas.draw()
        step = ax_a.transData.transform(np.c_[q.X + q.U, q.Y + q.V]) - ax_a.transData.transform(np.c_[q.X, q.Y])
        assert len(step) and (step[:, 0] < 0).all() and (step[:, 1] > 0).all(), f"motion arrows must point up-left on the screen: {step}"
        plt.close(fig_a)
        try:
            load_stored("0" * 32, data=d)
            raise AssertionError("load of an unknown keyframe must fail")
        except FileNotFoundError as e:
            assert store_sig in str(e)
    # 5b generate, the entry point: version found in the folder, groups of sequences with the light devkit give the same GT as the full
    #    devkit, a rerun writes nothing, a wrong scene or a wrong folder is refused with a message
    from . import cli as cli_mod
    assert cli_mod.detect_version(root) == "v1.0-mini"
    with tempfile.TemporaryDirectory() as d:
        old_group, cli_mod.GROUP_SCENES = cli_mod.GROUP_SCENES, 1                      # two groups: the loop over devkit loads is exercised
        try:
            r = cli_mod.generate(root, d, scenes=["scene-0061", "scene-0103"], stride=20, verbose=False)
            from . import suite as suite_mod
            real_gen, suite_mod.AutoGT = suite_mod.AutoGT, None                        # a rerun of finished groups must not load the devkit
            try:
                again = cli_mod.generate(root, d, scenes=["scene-0061", "scene-0103"], stride=20, verbose=False)
            finally:
                suite_mod.AutoGT = real_gen
        finally:
            cli_mod.GROUP_SCENES = old_group
        assert r["signature"] == store_sig and r["out"] == os.path.join(d, store_sig), r
        assert r["version"] == "v1.0-mini" and r["scenes"] == 2 and r["keyframes"] == r["written"] == 4 and again["written"] == 0, (r, again)
        held = json.load(open(os.path.join(r["out"], "index.json")))
        assert held["scenes"] == ["scene-0061", "scene-0103"] and held["keyframes"] == 4 and held["complete"] is True, held
        assert held["gt_version"] == r["gt_version"] == current_version(), (held, r)                      # the version key is stamped
        mine = load_keyframe(r["out"], tok2)                                           # the keyframe `build` wrote above, from the full devkit
        assert _same_gt(mine, g2), "light == full: every layer and every meta entry"
        for bad, why in ((dict(scenes=["scene-9999"]), "unknown scene"), (dict(version="v1.0-trainval"), "version")):
            try:
                cli_mod.generate(root, d, verbose=False, **bad)
                raise AssertionError(f"generate must refuse: {why}")
            except (ValueError, OSError):
                pass
        try:
            cli_mod.detect_version(d)
            raise AssertionError("a folder that is not a nuScenes folder must be refused")
        except ValueError as e:
            assert "not a nuScenes folder" in str(e)
    # 5c scene sets and the manifest of a store: small is a part of full, a set is complete when every keyframe has a file, the index.json
    #    says gt_version / commit / dirty / sets_complete and whether the folder is a release
    import contextlib
    import io
    from . import sets
    from .cli import write_index
    from .suite import AutoGT as _Gen2
    assert sets.names() == ["full", "small"], sets.names()
    small, full_ = sets.scenes("small"), sets.scenes("full")
    assert len(small) == 10 and len(full_) == 850 == len(set(full_)) and set(small) <= set(full_), (len(small), len(full_))
    small_tokens = sets.tokens("small", "v1.0-mini", root)
    assert len(small_tokens) > 300 and len(set(small_tokens)) == len(small_tokens), len(small_tokens)
    for call, why in ((lambda: sets.tokens("full", "v1.0-mini", root), "full is not in mini"), (lambda: sets.scenes("nope"), "unknown set"),
                      (lambda: cli_mod.generate(root, "x", set_name="small", scenes=["scene-0061"]), "set and scenes together")):
        try:
            call()
            raise AssertionError(f"must refuse: {why}")
        except ValueError:
            pass
    cs = V.code_state()
    assert set(cs) == {"commit", "dirty"} and (cs["commit"] is None) == (cs["dirty"] is None), cs
    with tempfile.TemporaryDirectory() as d:
        sd5 = os.path.join(d, store_sig)
        os.makedirs(sd5)
        for t in small_tokens[1:]:
            open(os.path.join(sd5, t + ".npz"), "wb").write(b"x")
        st = sets.status("small", data=d, version="v1.0-mini", dataroot=root)
        assert st["have"] == len(small_tokens) - 1 and st["total"] == len(small_tokens) and not st["complete"], st
        assert sets.complete_in(sd5, "v1.0-mini", root) == []
        open(os.path.join(sd5, small_tokens[0] + ".npz"), "wb").write(b"x")
        assert sets.status("small", data=d, version="v1.0-mini", dataroot=root)["complete"]
        assert sets.complete_in(sd5, "v1.0-mini", root) == ["small"]                  # full needs trainval: skipped, not an error
        from .suite import scene_tokens as tokens_of                                   # `have`: the check before a training
        want_ = sets.scenes("small")[:2]
        every = sets.stored_for(want_, data=d, version="v1.0-mini", dataroot=root)
        assert every["missing"] == [] and every["have"] == every["total"] > 0 and every["scenes"] == 2, every
        gone = tokens_of(want_[0], "v1.0-mini", root)[0]
        os.remove(os.path.join(sd5, gone + ".npz"))
        gap = sets.stored_for(want_, data=d, version="v1.0-mini", dataroot=root)
        assert gap["missing"] == [gone] and gap["have"] == gap["total"] - 1, gap
        open(os.path.join(sd5, gone + ".npz"), "wb").write(b"x")
        assert sets.stored_for(want_, data=d, version="v1.0-mini", dataroot=root, stride=1000)["total"] == 2, "stride"
        gen5 = _Gen2(root, "v1.0-mini", ss=4, light=True, scenes=["scene-0061"])

        def manifest(run, complete=True):
            write_index(sd5, gen5, "v1.0-mini", ["scene-0061"], 1, 1, {}, complete=complete, run=run)
            return json.load(open(os.path.join(sd5, "index.json")))
        ok = {"set": "full", "commit": "abc1234", "dirty": False, "sets_complete": ["full"]}
        m = manifest(ok)
        assert m["release"] is True and m["gt_version"] == current_version() and m["commit"] == "abc1234" and m["sets_complete"] == ["full"], m
        assert manifest({**ok, "dirty": True})["release"] is False, "a dirty tree is no release"
        assert manifest({**ok, "dirty": None})["release"] is False, "an unknown tree state is no release"
        assert manifest(ok, complete=False)["release"] is False, "an unfinished run is no release"
        assert manifest({**ok, "sets_complete": ["small"]})["sets_complete"] == ["full", "small"], "sets that were complete stay listed"
        m = manifest({**ok, "sets_complete": ["small"]})
        assert m["release"] is True, m                                                # full came from the earlier run on this folder
        m = manifest(None)                                                             # a report / build run keeps the manifest keys
        assert m["release"] is True and m["commit"] == "abc1234" and m["sets_complete"] == ["full", "small"], m
        assert manifest(None, complete=False)["release"] is False, "the provisional index of a running job is no release"
        # review of 2026-11-02: the sets a run could check are taken from ITS list (a deleted keyframe must remove `full`), a set that this
        # nuScenes version cannot check keeps its old entry
        assert sets.checkable("v1.0-mini", root) == ["small"], sets.checkable("v1.0-mini", root)
        m = manifest({**ok, "sets_complete": ["small"], "sets_checked": ["small"]})
        assert m["sets_complete"] == ["full", "small"] and m["release"] is True, "mini cannot check full: its entry stays"
        m = manifest({**ok, "sets_complete": [], "sets_checked": ["full", "small"]})
        assert m.get("sets_complete", []) == [] and m["release"] is False, "a hole in full takes the set out and the release with it"
        m = manifest({**ok, "sets_complete": ["full", "small"], "sets_checked": ["full", "small"], "environment": cli_mod._environment()})
        assert m["sets_complete"] == ["full", "small"] and m["release"] is True and set(m["environment"]) >= {"numpy", "shapely"}, m
        # a job that runs for hours refuses a generator that changed meanwhile (a pool worker imports the source tree afresh)
        cli_mod._same_generator(gen5, gen5.signature())
        try:
            cli_mod._same_generator(gen5, "deadbeef00")
            raise AssertionError("a changed generator must be refused")
        except RuntimeError as e:
            assert "changed while this job runs" in str(e)
        os.remove(os.path.join(sd5, "index.json"))
        assert manifest({"set": None, "commit": "abc1234", "dirty": False, "sets_complete": ["small"]})["release"] is False, "no full set"
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as sets_dir:
        with open(os.path.join(sets_dir, "tiny.txt"), "w") as f:
            f.write("# two sequences\nscene-0061\nscene-0103  # trailing comment\n\n")
        old_dir, sets.DIR = sets.DIR, sets_dir                                         # a tiny set instead of the real ones
        try:
            assert sets.names() == ["tiny"] and sets.scenes("tiny") == ["scene-0061", "scene-0103"]
            r = cli_mod.generate(root, d, set_name="tiny", stride=20, verbose=False)
        finally:
            sets.DIR = old_dir
        held = json.load(open(os.path.join(r["out"], "index.json")))
        assert held["set"] == "tiny" and held["scenes"] == ["scene-0061", "scene-0103"] and r["keyframes"] == 4, (held, r)
        assert held["complete"] is True and held["release"] is False and r["release"] is False and "commit" in held and "dirty" in held, held
        calls = []                                                                       # a rerun with nothing to do does not flag the store incomplete
        real_write = cli_mod.write_index
        cli_mod.write_index = lambda *a, **k: (calls.append(k.get("complete", True)), real_write(*a, **k))[1]
        sets.DIR = sets_dir
        try:
            r2 = cli_mod.generate(root, d, set_name="tiny", stride=20, verbose=False)
        finally:
            cli_mod.write_index, sets.DIR = real_write, old_dir
        assert r2["written"] == 0 and calls == [True], calls
        r_opt = cli_mod.generate(root, d, scenes=["scene-0061"], stride=40, use_flow=False, verbose=False)      # the folder name without the devkit
        assert r_opt["signature"] == _Gen2(root, "v1.0-mini", ss=4, light=True, scenes=["scene-0061"], use_flow=False).signature() != r["signature"]
    # 5d the index of a folder holds what the folder holds (review of 2026-11-01): a smaller or a repeated run neither shrinks the counts
    #    nor marks an interrupted run complete; strides that differ are said so
    with tempfile.TemporaryDirectory() as d:
        sd6 = os.path.join(d, store_sig)
        os.makedirs(sd6)
        for i in range(5):
            open(os.path.join(sd6, f"t{i}.npz"), "wb").write(b"x")
        open(os.path.join(sd6, "t9.npz.1.tmp.npz"), "wb").write(b"x")                  # a leftover temporary file is no keyframe
        open(os.path.join(sd6, "t8.npz"), "wb").close()                                # nor is an empty file
        gen6 = _Gen2(root, "v1.0-mini", ss=4, light=True, scenes=["scene-0061"])
        big = [f"scene-{i:04d}" for i in range(19)] + ["scene-0061"]                  # small is a part of big

        def idx(names, stride=1, complete=True):
            write_index(sd6, gen6, "v1.0-trainval" if len(names) > 5 else "v1.0-mini", names, stride, 99, {}, complete=complete)
            return json.load(open(os.path.join(sd6, "index.json")))
        m = idx(big, complete=False)                                                   # an interrupted big run ...
        assert m["complete"] is False and m["keyframes"] == 99, m
        m = idx(["scene-0061"])                                                        # ... then a finished small one
        assert m["complete"] is False and m["scenes"] == sorted(big) and m["version"] == "v1.0-trainval", m
        m = idx(big)                                                                   # the big run resumed and finished
        assert m["complete"] is True and m["keyframes"] == 5 and m["stride"] == 1, m
        m = idx(big, stride=2)                                                         # the same scenes again with another stride
        assert m["keyframes"] == 5 and m["stride"] == "mixed" and m["complete"] is True, m
        m = idx(["scene-0061"], stride=2)                                              # a subset after a full store
        assert m["scenes"] == sorted(big) and m["complete"] is True and m["stride"] == "mixed", m
    # 5e a rerun after damage writes exactly the damage; two worker processes (spawn) give the same GT as one; wrong arguments are refused
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as d2:
        kw = dict(scenes=["scene-0061", "scene-0103"], stride=20, verbose=False)
        r = cli_mod.generate(root, d, **kw)
        stored = sorted(f for f in os.listdir(r["out"]) if f.endswith(".npz"))
        os.remove(os.path.join(r["out"], stored[0]))                                   # a lost file,
        open(os.path.join(r["out"], stored[1]), "wb").close()                          # an empty one (full disk),
        open(os.path.join(r["out"], stored[2] + ".1.tmp.npz"), "wb").write(b"half")    # a leftover of a killed writer next to a good file
        again = cli_mod.generate(root, d, **kw)
        assert again["written"] == 2 and len(sets.held(r["out"])) == 4, again
        assert load_keyframe(r["out"], stored[1][:-4])["drivable_area"].shape == (120, 104)
        two = cli_mod.generate(root, d2, workers=2, **kw)                              # the spawn pool: each worker builds its own generator
        assert two["written"] == 4 and two["signature"] == r["signature"], two
        for t in (stored[0][:-4], stored[3][:-4]):
            a_, b_ = load_keyframe(r["out"], t), load_keyframe(two["out"], t)
            assert _same_gt(a_, b_), "workers=2 == workers=1: every layer and every meta entry"
        for bad, why in ((dict(scenes=[]), "no scene is not all scenes"), (dict(scenes=["scene-0061"], stride=0), "stride 0"),
                         (dict(scenes=["scene-0061"], res=0.25), "another resolution in a used folder")):
            try:
                cli_mod.generate(root, d, **{**dict(stride=20, verbose=False), **bad})
                raise AssertionError(f"generate must refuse: {why}")
            except ValueError:
                pass
        dup = cli_mod.generate(root, d, scenes=["scene-0061", "scene-0061"], stride=20, verbose=False)
        assert dup["scenes"] == 1 and dup["keyframes"] == 2, dup                       # a name twice is one sequence
    # 5f the raw-data check: complete on mini, and a data root without files is reported (every listed file, the map expansion)
    ok_ = sets.check_raw(root, "v1.0-mini", scenes=["scene-0061"])
    assert ok_["missing"] == 0 and not ok_["maps"] and ok_["camera"] > 100 and ok_["radar"] > 500, ok_
    with tempfile.TemporaryDirectory() as sd:                                           # a named set: the parameter `scenes` hid the function (a TypeError before 2026-11-02)
        with open(os.path.join(sd, "one.txt"), "w") as f:
            f.write("scene-0061\n")
        old_dir, sets.DIR = sets.DIR, sd
        try:
            assert sets.check_raw(root, "v1.0-mini", set_name="one") == ok_
        finally:
            sets.DIR = old_dir
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "v1.0-mini"))
        for t in ("scene", "sample", "log", "sample_data"):
            os.symlink(os.path.join(root, "v1.0-mini", t + ".json"), os.path.join(d, "v1.0-mini", t + ".json"))
        bad_ = sets.check_raw(d, scenes=["scene-0061"])                                 # tables only: no image, no radar file, no map
        assert bad_["missing"] == ok_["camera"] + ok_["radar"] and len(bad_["examples"]) == 10 and bad_["maps"], bad_
        assert "missing" in sets.report_raw(bad_) and "all on disk" in sets.report_raw(ok_)
    # 5g the digests that show two generators make the same GT (`same_gt_as`): a digest per frame, and the compare finds a changed one
    import argparse
    from . import checks as checks_mod
    dg1 = checks_mod.digests(root, "v1.0-mini", stride=40, sweeps=("scene-0061",))
    assert len(dg1["digests"]) >= 11 and dg1["signature"] == gt.signature(), dg1["signature"]
    with tempfile.TemporaryDirectory() as d:
        pa, pb = os.path.join(d, "a.json"), os.path.join(d, "b.json")
        json.dump(dg1, open(pa, "w"))
        json.dump(dg1, open(pb, "w"))
        cli_mod._cmd_digest(argparse.Namespace(compare=[pa, pb], out=None, stride=2))               # identical: no exit
        json.dump({**dg1, "digests": {**dg1["digests"], next(iter(dg1["digests"])): "0" * 40}}, open(pb, "w"))
        try:
            cli_mod._cmd_digest(argparse.Namespace(compare=[pa, pb], out=None, stride=2))
            raise AssertionError("a changed frame must be found")
        except SystemExit as e_:
            assert e_.code == 1
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli_mod._print_sets()
    assert "small" in buf.getvalue() and "full" in buf.getvalue(), buf.getvalue()
    # the agent guide names every layer, meta entry and box field the generator produces (so it cannot drift from the code)
    guide = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "AGENTS.md"), encoding="utf-8").read()
    keys = [k for k in g2 if k != "meta"] + list(g2["meta"]) + list(g2["meta"]["boxes"][0]) + ["dt", "steps"]
    missing = [k for k in keys if f"`{k}`" not in guide]
    assert not missing, f"autogt/AGENTS.md does not mention (in backticks): {missing}"
    from .report import report                                                        # 7 the report: one PDF, no picture files
    with tempfile.TemporaryDirectory() as d:
        rep = report(root, "v1.0-mini", os.path.join(d, "r.pdf"), os.path.join(d, "data"), scenes=["scene-0061", "scene-1077"], stride=20,
                     samples=2, verbose=False)
        left = sorted(os.listdir(d))
        assert left == ["data", "r.pdf"] and os.path.getsize(os.path.join(d, "r.pdf")) > 20000 and len(rep["samples"]) == 2, left
        assert not [f for _, _, fs in os.walk(d) for f in fs if f.lower().endswith((".png", ".jpg"))], "no picture files"
        stored = os.listdir(os.path.join(d, "data", rep["signature"]))
        assert "index.json" in stored and rep["written"] == len([f for f in stored if f.endswith(".npz")]), rep
        from .cli import write_index                                                   # a smaller run must not shrink the folder's index
        from .suite import AutoGT as _Gen
        sd_ = os.path.join(d, "data", rep["signature"])
        write_index(sd_, _Gen(root, "v1.0-mini", ss=4, light=True), "v1.0-mini", ["scene-0061"], 20, 1, {})
        held = json.load(open(os.path.join(sd_, "index.json")))
        assert set(held["scenes"]) == {"scene-0061", "scene-1077"} and held["keyframes"] == len([f for f in stored if f.endswith(".npz")]), held
    # 7b the memory-light devkit gives the identical GT, and a parallel run of the report equals a serial one and resumes
    from .suite import AutoGT
    full = AutoGT(root, "v1.0-mini", ss=4)
    light = AutoGT(root, "v1.0-mini", ss=4, light=True)
    tk = full.keyframes("scene-0061")[8]
    from . import scene_tokens
    assert scene_tokens("scene-0061", "v1.0-mini", root) == full.keyframes("scene-0061")        # the devkit-free list is the same
    a, b2 = full.keyframe(tk), light.keyframe(tk)
    assert _same_gt(a, b2), "light devkit must give identical GT (every layer and every meta entry)"
    assert len(light.nusc.sample_data) < 0.7 * len(full.nusc.sample_data)
    with tempfile.TemporaryDirectory() as d:
        kw = dict(scenes=["scene-0061", "scene-0103", "scene-1077"], stride=20, samples=3, verbose=False)
        r1 = report(root, "v1.0-mini", os.path.join(d, "a.pdf"), os.path.join(d, "d1"), workers=1, **kw)
        r2 = report(root, "v1.0-mini", os.path.join(d, "b.pdf"), os.path.join(d, "d2"), workers=2, light=True, **kw)
        assert r1["keyframes"] == r2["keyframes"] and r1["samples"] == r2["samples"] and not r2["failed"], (r1, r2)
        sd2 = os.path.join(d, "d2", r2["signature"])
        assert len([f for f in os.listdir(sd2) if f.startswith("_scene_")]) == 3
        r3 = report(root, "v1.0-mini", os.path.join(d, "c.pdf"), os.path.join(d, "d2"), workers=2, light=True, **kw)       # resumes
        assert r3["keyframes"] == r1["keyframes"] and r3["written"] == 0, r3
        r4 = report(root, "v1.0-mini", os.path.join(d, "e.pdf"), os.path.join(d, "d2"), workers=1, scenes=kw["scenes"], stride=10,
                    samples=3, verbose=False)                                       # another stride must NOT reuse the saved sequences
        assert r4["keyframes"] > r1["keyframes"] and r4["written"] > 0, r4
    from . import clean as cleanmod                                                  # the cleanup lists, moves, deletes only its own
    cleanmod.selftest()
    # 5h the lane graph (lane_graph.py, road_graph.py, lane_lines.py): a made-up map for the logic, mini for the checks against the map
    from .lane_graph_selftest import run as lane_graph_selftest
    lane_graph_msg = lane_graph_selftest(root, verbose=False)
    # 5i one release, one command: the standard GT and the lane GT of the same sequences come out of one `generate`, into two stores of one release
    from . import lane_gt as lane_gt_mod
    from .suite import scene_tokens as tokens_of_scene
    assert V.current_release() is not None, (f"this source (front {V.version_of(gt.signature())}, lane {lane_gt_mod.version_of(lane_gt_mod.signature())}) has no row in "
                                             "autogt/releases.csv: add one (the release is the front version, with a point when only the lane GT changed)")
    with tempfile.TemporaryDirectory() as d:
        toks_ = tokens_of_scene("scene-0061", "v1.0-mini", root)
        both = cli_mod.generate(root, d, scenes=["scene-0061"], verbose=False)
        assert both["written"] == len(toks_) and both["lane"] and both["lane"]["written"] == len(toks_), both
        assert os.path.isdir(os.path.join(d, "lane", both["lane"]["signature"])) and lane_gt_mod.missing(toks_, data=os.path.join(d, "lane")) == []
        assert json.load(open(os.path.join(both["out"], "index.json")))["suite_release"] == V.current_release()
        only = cli_mod.generate(root, d, scenes=["scene-0061"], verbose=False, lane=False)
        assert only["lane"] is None and only["written"] == 0, only
        assert cli_mod.generate(root, d, scenes=["scene-0061"], stride=20, verbose=False)["lane"] is None, "the lane GT is made for every keyframe: nothing at stride 20"
    # 6 a one-second video encodes to H.264 and plays back (read-back check inside encode)
    from .video import run as make_video
    with tempfile.TemporaryDirectory() as d:
        r = make_video("scene-0061", os.path.join(d, "t.mp4"), 1.0, dataroot=root)
        assert r["frames"] == 12 and r["codec"] in ("h264", "gif"), r
    if verbose:
        print(f"autogt selftest ok: signature {gt.signature()}; {len(seen)} locations have a road under the ego car; "
              f"static layers move with the ego (IoU {iou:.3f}); {len(fixed)} stationary instances fixed in the world; "
              f"{len(moving)} moving boxes in scene-0061 kf 8 (median speed difference to the devkit {np.median(diffs):.2f} m/s); "
              f"ego flow vs exact pose {err * 100:.1f} cm per camera frame; build + reload of {n} keyframes ok; 1 s video {r['codec']} read back")
        print(lane_graph_msg)


if __name__ == "__main__":
    run()
