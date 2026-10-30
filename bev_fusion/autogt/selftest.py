"""Self-test of the whole autoGT suite (nuScenes mini needed for the real-data part): python -m autogt selftest

1 flow estimator on a synthetic world with known truth (autogt/flow.py)
2 rasteriser: exact areas (also a big clipped polygon with a hole), line distance, occlusion
3 real keyframes: a road under the ego car in every map location, 0.25 m pooled = 0.5 m, static layers of the next keyframe =
  the current ones moved by the exact ego motion, instance layer consistent, stationary boxes fixed in the world,
  ego steps add up to the keyframe gap
4 velocity / flow layers: flow = flow_ego + velocity, ego flow = the exact pose change per camera frame, stationary velocity 0,
  moving boxes' speeds close to the devkit's, unknown velocity never a silent zero (weight 0), weights 0 on ignored boxes,
  a 12 Hz sweep frame (sd=) gives a ground-consistent frame
5 the build command writes and reloads a cache that is keyed by the generator signature
"""
import math
import os
import tempfile

import cv2
import numpy as np

from . import flow as flowmod
from .grid import GridSpec
from .layers import Raster, cell_centres, occlusion
from .nuscenes_gt import NuScenesGT


def run(root=None, verbose=True):
    import torch
    import torch.nn.functional as F
    flowmod._selftest()                                                                               # 1
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
    no_obj = (k["vehicle"] + k["pedestrian"] + k["two_wheeler"]) == 0                 # no observed box in the cell: no velocity
    assert np.abs(k["velocity"][:, no_obj]).max() == 0 and (k["flow_weight"][k["ignore"] > 0.9] == 0).all()
    assert k["flow_weight"].min() == 0 and k["flow_weight"].max() == 1
    short = gt.keyframe_gt(sc61["first_sample_token"], g)           # first keyframe: tracks without history
    assert all(b_["vel_known"] or (b_["vel"] == 0).all() for b_ in short["meta"]["boxes"])     # never a silent nonzero
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

    # 5 the build command (cache keyed by the signature)
    from .cli import build, load_keyframe
    with tempfile.TemporaryDirectory() as d:
        n = build(root, "v1.0-mini", d, scenes=["scene-0061"], stride=20, workers=1, res=0.5, verbose=False)
        sig_dir = os.path.join(d, gt.signature())
        files = sorted(os.listdir(sig_dir))
        assert n == len(files) - 1 and "index.json" in files, files
        g2 = load_keyframe(sig_dir, [f for f in files if f.endswith(".npz")][0][:-4])
        assert g2["flow"].shape == (2, g.nx, g.ny) and g2["meta"]["sample"]
    from .report import report                                                        # 7 the report: one PDF, no picture files
    with tempfile.TemporaryDirectory() as d:
        rep = report(root, "v1.0-mini", os.path.join(d, "r.pdf"), os.path.join(d, "data"), scenes=["scene-0061", "scene-1077"], stride=20,
                     samples=2, verbose=False)
        left = sorted(os.listdir(d))
        assert left == ["data", "r.pdf"] and os.path.getsize(os.path.join(d, "r.pdf")) > 20000 and len(rep["samples"]) == 2, left
        assert not [f for _, _, fs in os.walk(d) for f in fs if f.lower().endswith((".png", ".jpg"))], "no picture files"
        assert rep["written"] == len(os.listdir(os.path.join(d, "data", rep["signature"]))) - 0, rep
    from . import clean as cleanmod                                                  # the cleanup lists, moves, deletes only its own
    cleanmod.selftest()
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


if __name__ == "__main__":
    run()
