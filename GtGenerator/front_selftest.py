"""Self-test cases of the front generator (nuscenes_gt.py, flow.py, layers.py, coverage.py) and of what draws or stores its keyframes; run by
`python -m GtGenerator selftest` (selftest.py holds the table of cases). A case without `root` needs no data, a case with `root` needs nuScenes mini.

    ego_velocity     the ego velocity at the ends of a sequence on a synthetic drive (v6)
    raster           the rasteriser: exact areas (also a big clipped polygon with a hole), line distance, occlusion
    coverage_rules   the coverage helpers (coverage.py) against known answers and an exact shapely reference
    motion_arrows    the arrows of the motion panel point the way the car goes
    real_keyframes   a road under the ego car in every map location, 0.25 m pooled = 0.5 m, the static layers of the next keyframe = the current
                     ones moved by the exact ego motion, instance consistent, stationary boxes fixed in the world, ego steps add up to the gap
    velocity_layers  flow = flow_ego + velocity, ego flow = the exact pose change per camera frame, stationary velocity 0, speeds close to the
                     devkit's, unknown velocity never a silent zero, boxes at the camera time, 12 Hz sweep frames
    coverage         the masks of a real keyframe: the camera layer, soft hiding, the shares of every box, the radar fans of the rig
    light_devkit     the memory-light devkit gives the identical GT
    report           one PDF without picture files; a parallel run equals a serial one, resumes, and another stride is not reused
    light_cache      the light devkit's cache: equal to the scan, invalidated by a changed table, a vanished file falls back to the scan
    video            a one-second video encodes to H.264 and plays back
"""
import functools
import json
import os
import tempfile

import numpy as np

from .grid import GridSpec
from .layers import cell_centres, occlusion


def _same_gt(a, b):
    """two keyframes (dict of layers + meta) agree in every array byte and every meta entry (checks._feed, the digest of `GtGenerator digest`)"""
    import hashlib
    from .checks import _feed
    ha, hb = hashlib.sha1(), hashlib.sha1()
    _feed(ha, a)
    _feed(hb, b)
    return ha.hexdigest() == hb.hexdigest()


@functools.lru_cache(maxsize=1)
def devkit(root):
    """the front generator on the full devkit of mini, loaded once per process for every case that needs it"""
    from .nuscenes_gt import NuScenesGT
    return NuScenesGT(root, "v1.0-mini")


@functools.lru_cache(maxsize=1)
def keyframe_61(root):
    """(token, keyframe) of scene-0061, keyframe 8: a scene with moving traffic"""
    gt = devkit(root)
    tok = next(s for s in gt.nusc.scene if s["name"] == "scene-0061")["first_sample_token"]
    for _ in range(8):
        tok = gt.nusc.get("sample", tok)["next"]
    return tok, gt.keyframe_gt(tok, GridSpec())


# ---------------------------------------------------------------------------------------------------------------- no data
def ego_velocity():
    """a drive at 10 m/s along world x with a yaw rate of 0.2 rad/s, poses every 0.1 s over 1 s: the first and the last pose give the same
    speed as the middle (the clamped window of v5 gave half of it)"""
    from . import flow as flowmod
    tt_ = np.arange(0, 1.01, 0.1)
    ego_ = flowmod.Ego(t=tt_, x=10.0 * tt_, y=np.zeros(11), yaw=0.2 * tt_)
    for t_ in (0.0, 0.03, 0.5, 0.97, 1.0):
        v_ = flowmod.ego_velocity(ego_, t_)
        assert abs(np.hypot(v_[0], v_[1]) - 10.0) < 1e-3 and abs(v_[2] - 0.2) < 1e-6, (t_, v_)       # speed (the car drives along world x, the ego frame turns)
    assert flowmod.ego_velocity(flowmod.Ego(t=[0.0], x=[0.0], y=[0.0], yaw=[0.0]), 0.0) == (0.0, 0.0, 0.0)


def raster():
    import cv2
    from shapely.geometry import Polygon as ShPolygon, box as sh_box
    from .layers import Raster
    g = GridSpec()
    R = Raster(g, 4)
    sq = np.array([[10.0, -1.0], [12.0, -1.0], [12.0, 1.0], [10.0, 1.0]])
    cov = R.coverage(R.fill(R.canvas(), [(sq, [])]))
    assert abs(cov.sum() * g.res ** 2 - 4.0) < 0.15, cov.sum() * g.res ** 2
    assert cov[20:24, 50:54].min() > 0.99 and cov[:19].max() == 0
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


def coverage_rules():
    """the helpers of coverage.py (v7) on a synthetic scene with known answers, and the blocking test against an exact reference (shapely: the
    segment clipped by the footprint, then the height interval); the cases include long boxes right next to the camera, where a pre-filter by
    angle once dropped points (found by the review of 2026-11-01)"""
    from shapely.geometry import LineString, Point, Polygon as ShPolygon
    from . import coverage as C
    g = GridSpec()
    gx, gy = cell_centres(g)
    cam_ = np.array([1.5, 0.0, 1.6])
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
    edge_ = np.array([[58.0, -1], [62, -1], [62, 1], [58, 1]])                         # half of it beyond the grid
    assert C.footprint_share(half_, gy > 0, g) == 0.5 and C.footprint_share(edge_, np.ones((g.nx, g.ny), bool), g) == 0.5
    fan_ = C.radar_cells(np.array([10.0, 10.0, 80.0, -5.0]), np.array([0.0, 20.0, 0.0, 0.0]), [np.eye(4)])
    assert fan_.tolist() == [True, False, False, False], fan_                          # in the fan, 63 deg off axis, beyond 70 m, behind
    a59, a61 = np.radians(59.0), np.radians(61.0)                                     # the edges of the fan: 70 m range, 60 deg half angle
    edge_x, edge_y = np.array([69.9, 70.1, 20 * np.cos(a59), 20 * np.cos(a61)]), np.array([0.0, 0.0, 20 * np.sin(a59), -20 * np.sin(a61)])
    assert C.radar_cells(edge_x, edge_y, [np.eye(4)]).tolist() == [True, False, True, False], C.RADAR_FOV
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


def motion_arrows():
    """the arrows of the motion panel point the way the car goes: 5 m/s forward and 3 m/s to the left (+y) is up-left on the screen"""
    from .view import motion_arrows as arrows, plt                                 # view.py chooses the Agg backend
    g = GridSpec()
    fl = np.zeros((2, g.nx, g.ny), np.float32)
    fl[0, 40:48, 60:64], fl[1, 40:48, 60:64] = 5.0, 3.0
    fig_a, ax_a = plt.subplots()
    ax_a.set_xlim(26, -26)                                                         # the panel's axes: y inverted (left = +y), x forward up
    ax_a.set_ylim(0, 60)
    q = arrows(ax_a, fl)
    fig_a.canvas.draw()
    step = ax_a.transData.transform(np.c_[q.X + q.U, q.Y + q.V]) - ax_a.transData.transform(np.c_[q.X, q.Y])
    assert len(step) and (step[:, 0] < 0).all() and (step[:, 1] > 0).all(), f"motion arrows must point up-left on the screen: {step}"
    plt.close(fig_a)


# ---------------------------------------------------------------------------------------------------------------- nuScenes mini
def real_keyframes(root):
    import torch
    import torch.nn.functional as F
    g = GridSpec()
    gx, gy = cell_centres(g)
    gt = devkit(root)
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
        # 0.25 m pooled to 0.5 m against 0.5 m: measured 0.0006 for the vehicle layer (mean 0.017), so the bound is relative to the layer
        diff = np.abs(pooled - cur[layer])
        assert diff.mean() < 0.002 + 0.05 * cur[layer].mean() and diff.sum() < 0.1 * cur[layer].sum() + 1.0, (layer, diff.mean())
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
    return (f"signature {gt.signature()}; {len(seen)} locations have a road under the ego car; static layers move with the ego (IoU {iou:.3f}); "
            f"{len(fixed)} stationary instances fixed in the world")


def velocity_layers(root):
    """velocity / flow on scene-0061 keyframe 8; the boxes at the camera time (constant velocity over camera - annotation time, stationary /
    unknown boxes stay); 12 Hz sweep frames"""
    from . import flow as flowmod
    from .nuscenes_gt import NuScenesGT
    g = GridSpec()
    gx, gy = cell_centres(g)
    gt = devkit(root)
    tok, k = keyframe_61(root)
    sc61 = next(s for s in gt.nusc.scene if s["name"] == "scene-0061")
    assert np.allclose(k["flow"], k["flow_ego"] + k["velocity"], atol=1e-5)
    bx = k["meta"]["boxes"]
    moving = [i for i, b_ in enumerate(bx) if b_["observed"] and b_["vel_known"] and not b_["stationary"] and np.hypot(*b_["vel"]) > 1.0]
    assert moving and all(b_["vel"][0] == 0 and b_["vel"][1] == 0 for b_ in bx if b_["stationary"])
    diffs = [abs(np.hypot(*bx[i]["vel"]) - np.hypot(*np.nan_to_num(gt.nusc.box_velocity(bx[i]["token"]))[:2])) for i in moving]
    assert np.median(diffs) < 0.1, np.median(diffs)                                    # measured 0.02: close to the devkit's speeds
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
    return (f"{len(moving)} moving boxes in scene-0061 kf 8 (median speed difference to the devkit {np.median(diffs):.2f} m/s); "
            f"ego flow vs exact pose {err * 100:.1f} cm per camera frame")


def coverage(root):
    """the masks of a real keyframe (v7): the camera layer is the old ground-plane projection, boxes hide with their height, every box carries
    the observed shares of its footprint, the rig table of the radars gives the fans of the keyframe"""
    from . import coverage as C
    from .nuscenes_adapter import pose_matrix
    g = GridSpec()
    gx, gy = cell_centres(g)
    gt = devkit(root)
    tok, k = keyframe_61(root)
    bx = k["meta"]["boxes"]
    sd = gt.cam(tok)
    cs_ = gt.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    K_, Tc_ = np.array(cs_["camera_intrinsic"]), pose_matrix(cs_)
    P_ = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size), np.ones(gx.size)], 1) @ np.linalg.inv(Tc_).T
    u_, v_ = K_[0, 0] * P_[:, 0] / np.maximum(P_[:, 2], 1e-6) + K_[0, 2], K_[1, 1] * P_[:, 1] / np.maximum(P_[:, 2], 1e-6) + K_[1, 2]
    old_cam = (P_[:, 2] > 0.5) & (u_ >= 0) & (u_ < sd["width"]) & (v_ >= 0) & (v_ < sd["height"])
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
    rig_ = C.radar_cells(gx, gy, C.rig_radar_poses())                                  # the table of the rig = the fans of a real keyframe
    assert (rig_ != (k["radar"] > 0)).mean() < 0.002 and len(C.rig_radar_poses()) == 5, (rig_ != (k["radar"] > 0)).mean()
    for b_ in bx:                                                                      # one rule for ground truth and detections: the stored
        assert C.shares_in(k, b_["corners"], b_["size"][2])["fusion"] == b_["share_fusion"], b_["token"]   # share is what a detection of that footprint gets
    assert C.shares_in(k, bx[0]["corners"], group="pedestrian")["fusion"] >= 0.0 and set(k["meta"]) >= {"image_size", "radar_poses"}


def light_devkit(root):
    from .suite import GtGenerator, scene_tokens
    full = GtGenerator(root, "v1.0-mini", ss=4, nusc=devkit(root).nusc)
    light = GtGenerator(root, "v1.0-mini", ss=4, light=True)
    tk = full.keyframes("scene-0061")[8]
    assert scene_tokens("scene-0061", "v1.0-mini", root) == full.keyframes("scene-0061")        # the devkit-free list is the same
    assert _same_gt(full.keyframe(tk), light.keyframe(tk)), "light devkit must give identical GT (every layer and every meta entry)"
    assert len(light.nusc.sample_data) < 0.7 * len(full.nusc.sample_data)


def report(root):
    from .report import report as make_report
    from .store import write_index
    from .suite import GtGenerator
    with tempfile.TemporaryDirectory() as d:
        rep = make_report(root, "v1.0-mini", os.path.join(d, "r.pdf"), os.path.join(d, "data"), scenes=["scene-0061", "scene-1077"], stride=20,
                          samples=2, verbose=False)
        left = sorted(os.listdir(d))
        assert left == ["data", "r.pdf"] and os.path.getsize(os.path.join(d, "r.pdf")) > 20000 and len(rep["samples"]) == 2, left
        assert not [f for _, _, fs in os.walk(d) for f in fs if f.lower().endswith((".png", ".jpg"))], "no picture files"
        stored = os.listdir(os.path.join(d, "data", rep["signature"]))
        assert "index.json" in stored and rep["written"] == len([f for f in stored if f.endswith(".npz")]), rep
        sd_ = os.path.join(d, "data", rep["signature"])                                  # a smaller run must not shrink the folder's index
        write_index(sd_, GtGenerator(root, "v1.0-mini", ss=4, light=True), "v1.0-mini", ["scene-0061"], 20, 1, {})
        held = json.load(open(os.path.join(sd_, "index.json")))
        assert set(held["scenes"]) == {"scene-0061", "scene-1077"} and held["keyframes"] == len([f for f in stored if f.endswith(".npz")]), held
    with tempfile.TemporaryDirectory() as d:                                             # a parallel run equals a serial one and resumes
        kw = dict(scenes=["scene-0061", "scene-0103", "scene-1077"], stride=20, samples=3, verbose=False)
        r1 = make_report(root, "v1.0-mini", os.path.join(d, "a.pdf"), os.path.join(d, "d1"), workers=1, **kw)
        r2 = make_report(root, "v1.0-mini", os.path.join(d, "b.pdf"), os.path.join(d, "d2"), workers=2, light=True, **kw)
        assert r1["keyframes"] == r2["keyframes"] and r1["samples"] == r2["samples"] and not r2["failed"], (r1, r2)
        sd2 = os.path.join(d, "d2", r2["signature"])
        assert len([f for f in os.listdir(sd2) if f.startswith("_scene_")]) == 3
        r3 = make_report(root, "v1.0-mini", os.path.join(d, "c.pdf"), os.path.join(d, "d2"), workers=2, light=True, **kw)       # resumes
        assert r3["keyframes"] == r1["keyframes"] and r3["written"] == 0, r3
        r4 = make_report(root, "v1.0-mini", os.path.join(d, "e.pdf"), os.path.join(d, "d2"), workers=1, scenes=kw["scenes"], stride=10,
                         samples=3, verbose=False)                                       # another stride must NOT reuse the saved sequences
        assert r4["keyframes"] > r1["keyframes"] and r4["written"] > 0, r4


def light_cache(root):
    from .light_nusc import cache_valid, ensure_cache, light_nuscenes, scene_names
    with tempfile.TemporaryDirectory() as d:
        assert not cache_valid(root, "v1.0-mini", d)
        ensure_cache(root, "v1.0-mini", d, verbose=False)
        assert cache_valid(root, "v1.0-mini", d)
        names_ = ["scene-0061", "scene-0553"]
        fast, scan = light_nuscenes(root, "v1.0-mini", names_, cache_base=d), light_nuscenes(root, "v1.0-mini", names_, cache_base=os.path.join(d, "none"))
        for tab_ in ("sample_data", "ego_pose", "sample_annotation", "sample", "scene"):
            ta, tb = {r["token"]: r for r in getattr(fast, tab_)}, {r["token"]: r for r in getattr(scan, tab_)}
            assert ta == tb and ta, f"the cached {tab_} equals the scanned one"
        assert {s["token"]: s["data"] for s in fast.sample} == {s["token"]: s["data"] for s in scan.sample}, "the reverse index (sample['data']) is the same"
        assert all("RADAR" not in r["filename"] for r in light_nuscenes(root, "v1.0-mini", names_, radars=False, cache_base=d).sample_data)
        assert any("CAM_BACK" in r["filename"] for r in light_nuscenes(root, "v1.0-mini", names_, cameras=("CAM_FRONT", "CAM_BACK"), cache_base=d).sample_data)
        folder_ = os.path.join(d, "light_v1.0-mini")
        victim_ = [f_ for f_ in os.listdir(folder_) if f_.endswith(".pkl.gz")][0]
        os.rename(os.path.join(folder_, victim_), os.path.join(d, "aside.pkl.gz"))
        # a cache file is gone although the index is valid (a swap, a damaged copy): the scan answers
        gone = light_nuscenes(root, "v1.0-mini", [n_ for n_ in scene_names(root, "v1.0-mini")], cache_base=d)
        assert len(gone.scene) == len(scene_names(root, "v1.0-mini")) and len(gone.sample_data) > 0, "a vanished cache file falls back to the scan"
        os.rename(os.path.join(d, "aside.pkl.gz"), os.path.join(folder_, victim_))
        ensure_cache(root, "v1.0-mini", d, verbose=False)                         # a valid cache is left alone
        os.remove(os.path.join(folder_, "index.json"))
        # a rebuild swaps the folder: the old one is renamed aside and removed, no .tmp / .old folder is left
        ensure_cache(root, "v1.0-mini", d, verbose=False)
        assert cache_valid(root, "v1.0-mini", d) and os.listdir(d) == ["light_v1.0-mini"], os.listdir(d)
        idx_path = os.path.join(d, "light_v1.0-mini", "index.json")
        idx_ = json.load(open(idx_path))
        idx_["sources"]["ego_pose"][0] += 1                                  # the table on disk is no longer the one the cache was made from
        json.dump(idx_, open(idx_path, "w"))
        assert not cache_valid(root, "v1.0-mini", d)


def video(root):
    from .video import run as make_video
    with tempfile.TemporaryDirectory() as d:
        r = make_video("scene-0061", os.path.join(d, "t.mp4"), 1.0, dataroot=root)
        assert r["frames"] == 12 and r["codec"] in ("h264", "gif"), r
    return f"1 s video {r['codec']} read back"
