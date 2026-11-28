"""Tests of net/prep/augment.py (its former self-test).

    python tests/test_augment.py      (from the problem folder, with its module path)
"""
import math

import numpy as np
import torch

from net import netpath
from net.prep.augment import flip_batch, flip_cam


def _principal_axis(xs, ys):
    c = np.cov(np.stack([xs, ys]))
    w, v = np.linalg.eigh(c)
    return math.atan2(v[1, 1], v[0, 1]), w[1] / max(w[0], 1e-9)


def _selftest():
    import copy
    import os
    from net import netpath
    os.chdir(netpath.SITE)                                       # the data folder: the default weights are named relative to it
    from net.loss.losses import BEVLoss, sample_targets
    from net.model.model import BEVFusionModel, ModelConfig
    from net.prep.nuscenes_dataset import FrontRadarDataset, collate
    from net.model.radar_bev import rasterize
    from net.model.temporal_fusion import advance_pose
    from net.prep.batches import gap_ego_motion
    torch.manual_seed(0)
    ds = FrontRadarDataset(os.path.expanduser("~/data/nuscenes"), "v1.0-mini", n_cam=3, cam_stride=2, with_gt=True)
    batch = collate([ds[10], ds[200]])
    orig = copy.deepcopy(batch)

    def eq(x, y):
        if isinstance(x, dict):
            return set(x) == set(y) and all(eq(x[k], y[k]) for k in x)
        if isinstance(x, (list, tuple)) and not isinstance(x, np.ndarray):
            return len(x) == len(y) and all(eq(p, q) for p, q in zip(x, y))
        if isinstance(x, (np.ndarray, float, int)) and not isinstance(x, bool):     # equal_nan: boxes of unknown velocity carry
            return bool(np.allclose(np.asarray(x, float), np.asarray(y, float), atol=1e-9, equal_nan=True))   # vel_sigma = NaN
        return x == y

    def same_gt(a, b):
        assert set(a) == set(b)
        for k in a:
            if k == "meta":
                assert eq(a[k]["boxes"], b[k]["boxes"]) and eq(a[k].get("steps", []), b[k].get("steps", [])), k
            else:
                assert np.array_equal(a[k], b[k]), k

    # 1 flipping twice is the identity, a sample that is not flipped is untouched, the input arrays are never modified
    g0 = batch["gt"][0]["cur"]
    snap = copy.deepcopy(g0)
    mirrored = flip_batch(copy.deepcopy(batch), [True, False])
    same_gt(batch["gt"][0]["cur"], snap)                                              # (the copy was flipped, not the input)
    again = flip_batch(mirrored, [True, False])
    for key in ("cur", "prev", "next"):
        for i in (0, 1):
            same_gt(again["gt"][i][key], orig["gt"][i][key])
    assert torch.equal(again["omega"], orig["omega"]) and torch.equal(again["radar"]["y"], orig["radar"]["y"])
    assert torch.equal(again["radar"]["psi"], orig["radar"]["psi"])
    assert all(torch.equal(w1, w0) for f1, f0 in zip(again["steps"], orig["steps"]) for (_, _, w1), (_, _, w0) in zip(f1, f0))
    one = flip_batch(copy.deepcopy(batch), [True, False])
    assert torch.equal(one["radar"]["y"][one["radar"]["b"] == 1], orig["radar"]["y"][orig["radar"]["b"] == 1])
    assert torch.equal(one["radar"]["y"][one["radar"]["b"] == 0], -orig["radar"]["y"][orig["radar"]["b"] == 0])
    same_gt(one["gt"][1]["cur"], orig["gt"][1]["cur"])
    # 2 the mirrored dense rasters agree with the mirrored boxes: centroid, axis and velocity of the cells of every box
    checked = 0
    for which, g in (("original", orig["gt"][0]["cur"]), ("mirrored", one["gt"][0]["cur"])):
        res = 0.5
        for i, bx in enumerate(g["meta"]["boxes"]):
            cells = np.argwhere(g["instance"] == i)
            if len(cells) < 25:
                continue
            xs, ys = -0.0 + (cells[:, 0] + 0.5) * res, -26.0 + (cells[:, 1] + 0.5) * res
            if xs.min() < 1 or xs.max() > 59 or abs(ys).max() > 25:
                continue                                                               # cut by the grid border
            assert abs(xs.mean() - bx["center"][0]) < 0.6 and abs(ys.mean() - bx["center"][1]) < 0.6, (which, i)
            ang, ratio = _principal_axis(xs, ys)
            if ratio > 3.0:
                assert abs(math.remainder(ang - bx["yaw"], math.pi)) < 0.25, (which, i, ang, bx["yaw"])
            sp = math.hypot(*bx["vel"])
            if sp > 1.0:
                v = np.median(g["velocity"][:, cells[:, 0], cells[:, 1]].astype(float), axis=1)   # box edge cells are partial
                assert np.allclose(v, bx["vel"], atol=0.05), (which, v, bx["vel"])
            checked += 1
    assert checked >= 6, checked
    # 3 the radar raster of the mirrored detections is the mirrored raster (sin psi and the sub-cell offset change sign)
    B = 2
    R0 = rasterize({k: v for k, v in orig["radar"].items() if k in ("x", "y", "r", "psi", "vr", "rcs", "dt", "b")}, B)
    R1 = rasterize({k: v for k, v in one["radar"].items() if k in ("x", "y", "r", "psi", "vr", "rcs", "dt", "b")}, B)
    want = R0[0].flip(-1).clone()
    want[3] = -want[3]                                                                 # channel 3 = sin psi
    want[-1] = -want[-1]                                                               # last channel = dy
    assert torch.allclose(R1[0], want, atol=1e-5) and torch.allclose(R1[1], R0[1], atol=1e-5)    # not torch.equal: scatter sums on
                                                                                              # several CPU threads are not bit-exact
    # 4 the ego steps: the gap-mean motion mirrors (omega -> -omega), a chain of steps ends at the mirrored pose
    st = orig["gt"][0]["next"]["meta"]["steps"]
    a, b = gap_ego_motion(st), gap_ego_motion(one["gt"][0]["next"]["meta"]["steps"])
    assert abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-6 and abs(a[2] + b[2]) < 1e-9, (a, b)
    assert orig["gap_steps"][0]["next"] == st and one["gap_steps"][0]["next"] == one["gt"][0]["next"]["meta"]["steps"] and \
        one["gap_steps"][1] == orig["gap_steps"][1], "the batch's gap steps mirror with the GT meta steps (and only for the flipped sample)"
    p0 = p1 = torch.zeros(1, 3)
    for (dt, v, w), (_, _, wm) in zip(st, one["gt"][0]["next"]["meta"]["steps"]):
        p0, p1 = advance_pose(p0, dt, v, w), advance_pose(p1, dt, v, wm)
    assert torch.allclose(p1, p0 * torch.tensor([1.0, -1.0, -1.0]), atol=1e-5), (p0, p1)
    # 5 the model: the flip mask mirrors the camera BEV and its visibility, a mask of zeros changes nothing, the loss on the
    #   mirrored GT is finite
    cam = torch.randn(2, 3, 4, 120, 104)
    obs = torch.rand(2, 1, 120, 104)
    c2, o2 = flip_cam(cam, obs, torch.tensor([True, False]))
    assert torch.equal(c2[0], cam[0].flip(-1)) and torch.equal(c2[1], cam[1]) and torch.equal(o2[0], obs[0].flip(-1))
    model = BEVFusionModel(ModelConfig(pretrained="", aux_heads=False)).eval()
    with torch.no_grad():                                      # (the untrained heads output almost constants: compare the trunk)
        y0 = model(copy.deepcopy(orig), dts=(0.0,))["features"]
        b_zero = flip_batch(copy.deepcopy(orig), [False, False])
        y1 = model(b_zero, dts=(0.0,))["features"]
        y2 = model(copy.deepcopy(one), dts=(0.0,))["features"]
    assert torch.allclose(y0, y1, atol=1e-6) and torch.isfinite(y2).all() and (y2[1] - y0[1]).abs().max() < 1e-5
    assert not torch.allclose(y2[0], y0[0], atol=1e-6), float((y2[0] - y0[0]).abs().max())      # the mirrored sample differs
    gts = sample_targets(one["gt"])
    model.train()
    res = model(copy.deepcopy(one), dts=tuple(sorted(gts)))
    L = BEVLoss(grid=model.grid)(res["out"], gts, step_frac=1.0)
    assert torch.isfinite(L["total"])
    print(f"augment selftest ok: double flip = identity, input never modified, {checked} boxes agree with their mirrored rasters "
          f"(centroid, axis, velocity), radar raster of mirrored detections = mirrored raster, ego step / pose mirror, camera BEV "
          f"and visibility flip per sample, zero mask = no change, loss finite on a mirrored batch")


if __name__ == "__main__":
    _selftest()
