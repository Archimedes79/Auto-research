"""Tests of evaluation/bev_views.py (its former self-test).

    python -P -m bev_fusion.tests.test_bev_views      (from the problem folder, with its module path)
"""
from __future__ import annotations
import math

import numpy as np

from bev_fusion.evaluation.bev_views import BLUE, DET_COLOR, GREEN, GT_OVERLAY, NOCARE_COLOR, OUTSIDE_COLOR, bce_map, box_shapes, error_box_shapes, loss_overlay, moving_layers, past_box_shapes, radar_points, speed_dir_rgb, tinted


# ------------------------------------------------------------------------------------------------ self-test of the pure picture helpers
def selftest():
    # radar points: opacity >= 50 % and rising with the strength, colour by the sign of the Doppler
    rad = {"x": np.array([10.0, 20.0, 30.0]), "y": np.array([0.0, 1.0, 2.0]), "vr": np.array([-15.0, 0.0, 15.0]), "rcs": np.array([0.0, 10.0, 20.0])}
    c = radar_points(rad).color
    assert c.shape == (3, 4) and abs(c[0, 3] - 0.5) < 1e-6 and abs(c[2, 3] - 1.0) < 1e-6 and 0.5 < c[1, 3] < 1.0, c
    assert c[0, 2] > c[0, 0] and c[2, 0] > c[2, 2], "approaching = cool (blue), receding = warm (red)"
    one = radar_points({"x": np.array([5.0]), "y": np.array([0.0]), "vr": np.array([1.0]), "rcs": np.array([3.0])}).color
    assert one.shape == (1, 4) and one[0, 3] >= 0.5, "a single return"
    # speed and direction: hue = direction, brightness = speed
    v = np.zeros((2, 2, 3), np.float32)
    v[:, 0, 0], v[:, 0, 1], v[:, 0, 2] = (10.0, 0.0), (0.0, 5.0), (-5.0, 0.0)
    img = speed_dir_rgb(v, 10.0)
    assert np.allclose(img[0, 0], (1, 0, 0)) and np.allclose(img[1, 1], 0), "ahead at full speed = red, standing = black"
    assert abs(img[0, 1].max() - 0.5) < 1e-6 and img[0, 1].argmax() == 1, "left at half speed = half-bright green side"
    assert abs(img[0, 2].max() - 0.5) < 1e-6 and img[0, 2][0] < img[0, 2][1] and img[0, 2][1] == img[0, 2][2], "behind = cyan"
    # moving objects as one layer
    prob = {"vehicle": np.array([[1.0, 1.0], [0.0, 0.0]]), "pedestrian": np.zeros((2, 2)), "drivable_area": np.ones((2, 2))}
    gprob = {"vehicle": np.array([[1.0, 1.0], [1.0, 0.0]]), "pedestrian": np.zeros((2, 2)), "drivable_area": np.ones((2, 2))}
    vel = np.zeros((2, 2, 2))
    vel[0, 0, 0] = 3.0                                                   # predicted: cell (0,0) moves
    gvel = np.zeros((2, 2, 2))
    gvel[0, 0, 0], gvel[0, 1, 0] = 3.0, 2.0                               # ground truth: cells (0,0) and (1,0) move
    pm, gm, ps, gs = moving_layers(prob, vel, gprob, gvel)
    assert pm.tolist() == [[True, False], [False, False]] and gm.tolist() == [[True, False], [True, False]]
    assert ps.tolist() == [[False, True], [False, False]] and gs.tolist() == [[False, True], [False, False]]
    # past boxes: the car drove 5 m forward since, so a box 10 m ahead then is 5 m ahead now, faded
    T_then, T_now = np.eye(4), np.eye(4)
    T_now[0, 3] = 5.0
    sh = past_box_shapes([{"group": "vehicle", "cx": 10.0, "cy": 0.0, "length": 4.0, "width": 2.0, "yaw": 0.0}], {"T_e2g": T_then}, {"T_e2g": T_now})
    poly = next(x for x in sh if x.kind == "polygon")
    assert abs(np.mean([p[1] for p in poly.xy]) - 5.0) < 1e-6 and abs(poly.color[3] - 0.30) < 1e-9
    T_now2 = np.eye(4)
    T_now2[:2, :2] = [[0, -1], [1, 0]]                                    # the car turned left by 90 degrees: a box straight ahead is now on its right
    sh2 = next(x for x in past_box_shapes([{"group": "vehicle", "cx": 10.0, "cy": 0.0, "length": 4.0, "width": 2.0, "yaw": 0.0}], {"T_e2g": T_then}, {"T_e2g": T_now2}) if x.kind == "polygon")
    assert abs(np.mean([p[0] for p in sh2.xy]) + 10.0) < 1e-6 and abs(np.mean([p[1] for p in sh2.xy])) < 1e-6, "turned frame: y = -10, x = 0"
    # errors in colour: a box right, one false, one missed, one 'no care' on each side
    area = np.ones((120, 104), bool)
    area[60:] = False                                                    # x >= 30 m is not observed
    def box(x, y, g="vehicle", s=None):
        b = {"group": g, "cx": float(x), "cy": float(y), "length": 4.0, "width": 2.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0}
        if s is not None:
            b["score"] = s
        return b
    gts = [box(10, 0), box(20, 5), box(45, 0)]                             # right, missed, no care (x = 45 m, outside the observed area)
    shares = [1.0, 1.0, 0.0]
    dets = [box(10.2, 0, s=0.9), box(15, -8, s=0.8), box(46, 0, s=0.9), box(12, 12, s=0.1)]       # right, false, on the no-care box, under the score threshold
    shapes, cnt = error_box_shapes(dets, gts, shares, area)
    assert cnt == {"tp": 1, "fp": 1, "fn": 1, "nocare": 2}, cnt
    def poly_at(cx, cy):                                                   # the polygon of the box centred at (cx, cy)
        return next(x for x in shapes if x.kind == "polygon" and abs(np.mean([q[1] for q in x.xy]) - cx) < 0.05 and abs(np.mean([q[0] for q in x.xy]) - cy) < 0.05)
    rgb = lambda x: tuple(np.round(np.asarray(x.color, float), 3))        # noqa: E731
    fp_, fn_, ok_, nc_ = poly_at(15, -8), poly_at(20, 5), poly_at(10.2, 0), poly_at(45, 0)
    assert rgb(fp_) == tuple(np.round(tinted(DET_COLOR, GREEN), 3)) and fp_.ls == "-" and fp_.lw == 2.0, "a false detection: amber + green, solid, thick"
    assert rgb(fn_) == tuple(np.round(tinted(GT_OVERLAY, BLUE, 0.6), 3)) and fn_.ls == "--" and fn_.lw == 2.0, "a missed box: white + blue, dashed, thick"
    assert rgb(ok_) == tuple(np.round(DET_COLOR, 3)) and ok_.ls == "-" and ok_.lw == 1.4, "the right detection keeps the plain amber"
    assert rgb(nc_) == tuple(np.round(NOCARE_COLOR, 3)), "a no-care box is violet"
    assert rgb(poly_at(10, 0)) == tuple(np.round(GT_OVERLAY, 3)) and poly_at(10, 0).ls == "--", "the GT box of a right detection: white, dashed"
    # the scored window: a false detection 56 deg off the axis and a GT box beyond 60 m are neither counted nor coloured as errors
    dets2 = dets + [box(10, 15, s=0.9)]
    shapes2, cnt2 = error_box_shapes(dets2, gts + [box(70, 0), box(10, -15)], shares + [1.0, 1.0], area)
    assert cnt2 == cnt, (cnt2, cnt)
    # a ground-truth box just beyond the window (60.8 m) neutralises a detection inside it, as in the metric (the detection is ignored, not a false alarm)
    shapes3, cnt3 = error_box_shapes(dets + [box(59.4, 0, s=0.9)], gts + [box(60.8, 0)], shares + [1.0], area)
    assert cnt3 == dict(cnt, nocare=cnt["nocare"] + 1), (cnt3, cnt)
    far = [x for x in shapes2 if x.kind == "polygon" and abs(np.mean([q[1] for q in x.xy]) - 70) < 0.3][0]
    assert abs(np.asarray(far.color, float)[3] - OUTSIDE_COLOR[3]) < 1e-9, "outside the window: faint violet"
    # robustness: a NaN Doppler / RCS, a model without object layers, no radar strength at all
    nanrad = radar_points({"x": np.array([1.0, 2.0, 3.0]), "y": np.zeros(3), "vr": np.array([np.nan, 5.0, -5.0]), "rcs": np.array([1.0, np.nan, 3.0])}).color
    assert np.isfinite(nanrad).all() and (nanrad[:, 3] >= 0.5).all(), nanrad
    assert np.isfinite(radar_points({"x": np.array([1.0]), "y": np.zeros(1), "vr": np.zeros(1)}).color).all(), "no rcs key"
    pm0, gm0, ps0, gs0 = moving_layers({"drivable_area": np.ones((2, 2))}, None, {"drivable_area": np.ones((2, 2))}, np.ones((2, 2, 2)))
    assert not (pm0.any() or gm0.any() or ps0.any() or gs0.any()), "no object layers: nothing moves"
    # past boxes keep their heading through the pose: a box with yaw 0.5 seen from a car that turned by 90 degrees has yaw 0.5 - 90 deg
    b_then = {"group": "vehicle", "cx": 10.0, "cy": 0.0, "length": 4.0, "width": 2.0, "yaw": 0.5}
    got = next(x for x in past_box_shapes([b_then], {"T_e2g": np.eye(4)}, {"T_e2g": T_now2}) if x.kind == "polygon")
    want = next(x for x in box_shapes([dict(b_then, cx=0.0, cy=-10.0, yaw=0.5 - math.pi / 2)], (0, 0, 0, 0.3), ":", 0.9) if x.kind == "polygon")
    assert np.allclose(got.xy, want.xy, atol=1e-6), (got.xy, want.xy)
    # the heat of a loss has no red tint (hue yellow to white): green is at least 0.85 of red at every opacity
    for loss_v in (0.3, 1.0, 1.7, 2.0):
        px = loss_overlay(np.zeros((1, 1, 3), np.float32), np.array([[loss_v]]), vmax=2.0, strength=1.0)[0, 0]
        assert px[1] >= 0.85 * px[0], (loss_v, px)
    # radar: standing = grey, approaching = blue, receding = orange
    rr = radar_points({"x": np.array([1.0, 2.0, 3.0]), "y": np.zeros(3), "vr": np.array([-8.0, 0.0, 8.0]), "rcs": np.array([1.0, 2.0, 3.0])}).color
    assert rr[1, 0] == rr[1, 1] == rr[1, 2] and rr[0, 2] > rr[0, 0] and rr[2, 0] > rr[2, 2], rr
    # the loss heat: the BCE of known cells and the overlay (no loss = the picture itself)
    lm = bce_map(np.array([[0.5, 0.99], [0.01, 0.5]]), np.array([[1.0, 1.0], [1.0, 0.0]]))
    assert abs(lm[0, 0] - math.log(2)) < 1e-9 and abs(lm[0, 1] + math.log(0.99)) < 1e-9 and abs(lm[1, 0] + math.log(0.01)) < 1e-9 and abs(lm[1, 1] - math.log(2)) < 1e-9
    base = np.full((2, 2, 3), 0.3, np.float32)
    out = loss_overlay(base, np.array([[0.0, 0.0], [0.0, 4.0]]), vmax=2.0)
    assert np.allclose(out[0, 0], 0.3) and out[1, 1].mean() > 0.6, out
    print("bev_views selftest ok: radar opacity / direction, speed-direction colours, moving layer, faded past boxes through the poses, error colours of the boxes")


if __name__ == "__main__":
    selftest()
