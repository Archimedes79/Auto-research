p = "evaluation/bev_views.py"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''    cols = {tuple(np.round(np.asarray(x.color, float), 3)) for x in shapes if x.kind == "polygon"}
    assert tuple(np.round(tinted(DET_COLOR, GREEN), 3)) in cols and tuple(np.round(tinted(GT_OVERLAY, BLUE, 0.6), 3)) in cols and tuple(np.round(NOCARE_COLOR, 3)) in cols
    assert tuple(np.round(DET_COLOR, 3)) in cols, "the right detection keeps the plain amber"''', '''    def poly_at(cx, cy):                                                   # the polygon of the box centred at (cx, cy)
        return next(x for x in shapes if x.kind == "polygon" and abs(np.mean([q[1] for q in x.xy]) - cx) < 0.3 and abs(np.mean([q[0] for q in x.xy]) - cy) < 0.3)
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
        assert px[1] >= 0.85 * px[0], (loss_v, px)''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
