p = "evaluation/bev_views.py"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


# radar colours: grey = about standing, saturated cool / warm = approaching / receding
rep('''    import matplotlib.cm as cm
    vr = np.asarray(rad["vr"], float)
    rgba = cm.coolwarm((np.clip(vr, -15, 15) + 15) / 30.0).copy()
    rcs = np.asarray(rad["rcs"], float) if "rcs" in rad else np.zeros_like(vr)''', '''    vr = np.asarray(rad["vr"], float)
    k = np.clip(np.abs(vr) / 6.0, 0, 1)[:, None]                                      # 0 = standing (grey), 1 = 6 m/s and more (saturated)
    base = np.where((vr < 0)[:, None], np.asarray((0.15, 0.65, 1.0)), np.asarray((1.0, 0.50, 0.10)))
    rgba = np.ones((len(vr), 4))
    rgba[:, :3] = np.asarray((0.55, 0.55, 0.55)) * (1 - k) + base * k
    rcs = np.asarray(rad["rcs"], float) if "rcs" in rad else np.zeros_like(vr)''')
rep('''    """the radar detections of the window as ONE point layer: colour = the direction of the radial velocity (cool = approaching, warm = receding, grey = about standing still
    relative to the car), transparency''', '''    """the radar detections of the window as ONE point layer: colour = the direction of the radial velocity (cool blue = approaching, warm orange = receding, grey = about
    standing still; saturation grows with |vr| up to 6 m/s), transparency''')

# loss over the GT
rep('''def moving_layers(prob, vel, gt_prob, gt_vel, groups=OBJECT_LAYERS):''', '''def bce_map(p, g, eps: float = 1e-4) -> np.ndarray:
    """the binary cross-entropy of every cell of a probability map against the ground truth (what the layer's training loss is made of, per cell)"""
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    g = np.clip(np.asarray(g, np.float64), 0.0, 1.0)
    return -(g * np.log(p) + (1 - g) * np.log(1 - p))


def loss_overlay(base_rgb, loss, vmax: float = 2.0, strength: float = 0.85) -> np.ndarray:
    """a loss map as a transparent heat (black -> yellow -> white, opacity = the loss relative to `vmax`) over a picture: where the loss sits, on the ground truth it belongs to"""
    import matplotlib.cm as cm
    a = (np.clip(np.asarray(loss, np.float32) / vmax, 0, 1) * strength)[..., None]
    heat = cm.afmhot(np.clip(np.asarray(loss, np.float32) / vmax, 0, 1))[..., :3]
    return (np.asarray(base_rgb, np.float32) * (1 - a) + heat.astype(np.float32) * a).astype(np.float32)


def moving_layers(prob, vel, gt_prob, gt_vel, groups=OBJECT_LAYERS):''')
rep('''    return [bev(grid, "rgb", "25 moving objects, ground truth (velocity above 0.5 m/s)",''', '''    heat = []
    for name, title in (("drivable_area", "29 drivable area"), ("vehicle", "30 vehicle footprint")):
        if name in o0["prob"] and name in gt_prob:
            lm = bce_map(o0["prob"][name], gt_prob[name])
            v_ = m["fov"] if name == "drivable_area" else valid
            heat.append(bev(grid, "rgb", f"{title}: loss per cell (BCE) as a heat over the GT; mean on the evaluated cells {float(lm[v_].mean()) if v_.any() else float('nan'):.3f}",
                            loss_overlay(composite(gt_prob, gt_dist, observed=obs), np.where(v_, lm, 0.0))))
    if vel is not None:
        gp = np.hypot(gvel[0], gvel[1])
        lv = np.hypot(vel[0] - gvel[0], vel[1] - gvel[1]) * (np.any([np.asarray(gt_prob[n]) > 0.5 for n in OBJECT_LAYERS if n in gt_prob], axis=0) if any(n in gt_prob for n in OBJECT_LAYERS) else 0)
        heat.append(bev(grid, "rgb", f"31 velocity error on the object cells (|v_pred - v_GT|, white = {4.0:.0f} m/s) as a heat over the GT",
                        loss_overlay(composite(gt_prob, gt_dist, observed=obs), lv, vmax=4.0)))
    return [bev(grid, "rgb", "25 moving objects, ground truth (velocity above 0.5 m/s)",''')
rep('''            bev(grid, "rgb", "28 stationary world, system", composite(stat_pred, o0["dist"], observed=obs))]''', '''            bev(grid, "rgb", "28 stationary world, system", composite(stat_pred, o0["dist"], observed=obs))] + heat''')

# selftest additions
rep('''    print("bev_views selftest ok: radar opacity / direction,''', '''    # radar: standing = grey, approaching = blue, receding = orange
    rr = radar_points({"x": np.array([1.0, 2.0, 3.0]), "y": np.zeros(3), "vr": np.array([-8.0, 0.0, 8.0]), "rcs": np.array([1.0, 2.0, 3.0])}).color
    assert rr[1, 0] == rr[1, 1] == rr[1, 2] and rr[0, 2] > rr[0, 0] and rr[2, 0] > rr[2, 2], rr
    # the loss heat: the BCE of known cells and the overlay (no loss = the picture itself)
    lm = bce_map(np.array([[0.5, 0.99], [0.01, 0.5]]), np.array([[1.0, 1.0], [1.0, 0.0]]))
    assert abs(lm[0, 0] - math.log(2)) < 1e-9 and abs(lm[0, 1] + math.log(0.99)) < 1e-9 and abs(lm[1, 0] + math.log(0.01)) < 1e-9 and abs(lm[1, 1] - math.log(2)) < 1e-9
    base = np.full((2, 2, 3), 0.3, np.float32)
    out = loss_overlay(base, np.array([[0.0, 0.0], [0.0, 4.0]]), vmax=2.0)
    assert np.allclose(out[0, 0], 0.3) and out[1, 1].mean() > 0.6, out
    print("bev_views selftest ok: radar opacity / direction,''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
