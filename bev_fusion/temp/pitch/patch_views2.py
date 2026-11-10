p = "evaluation/bev_views.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:90])
    s = s.replace(old, new)


# --- shared helper in BevView
rep('''    def _preds(self, pred: Prediction, dt=0.0):
        return pred.outputs["dt"][dt]
''', '''    def _preds(self, pred: Prediction, dt=0.0):
        return pred.outputs["dt"][dt]

    def pred_boxes(self, sample: Sample, o, dt=0.0, past: bool = False):
        """the boxes of a prediction panel with the errors in colour (error_box_shapes), the ground truth of the frame behind them and, for dt 0, the ground-truth boxes of the
        frame half a second earlier faded (the history of the scene) -> (shapes, counts)"""
        from evaluation.bev_fusion_area import fusion_area, gt_with_shares
        g = sample.reference[dt]
        gts, shares = gt_with_shares(g, self.t.groups)
        shapes, c = error_box_shapes(o["dets"], gts, shares, fusion_area(g))
        if past and dt == 0.0 and -0.5 in sample.reference:
            try:
                shapes = past_box_shapes(gt_boxes(sample.reference[-0.5], self.t.groups), sample.reference[-0.5]["meta"], g["meta"]) + shapes
            except KeyError:                                          # a GT without poses in its meta
                pass
        return shapes, c
''')

# --- Overview: the prediction panel
rep('''                  bev(grid, "rgb", "prediction (amber boxes: score >= 0.3, dashed: GT)", composite(o["prob"], o["dist"], observed=observed_area(g0)),
                      box_shapes(gtb, GT_OVERLAY, "--", 0.7) + box_shapes(dets, DET_COLOR)),''',
    '''                  bev(grid, "rgb", f"prediction: boxes right {bx['tp']} (amber), false {bx['fp']} (amber + green), missed {bx['fn']} (dashed + blue), no care {bx['nocare']} (violet); faded = 0.5 s ago",
                      composite(o["prob"], o["dist"], observed=observed_area(g0)), bshapes),''')
rep('''        dets = [d for d in o["dets"] if d["score"] >= 0.3]
        P, G = o["prob"]["drivable_area"] > 0.5, np.asarray(g0["drivable_area"], np.float32) > 0.5
        e_drv = _err(P, G, m["fov"], "drivable_area")''', '''        bshapes, bx = self.pred_boxes(sample, o, 0.0, past=True)
        P, G = o["prob"]["drivable_area"] > 0.5, np.asarray(g0["drivable_area"], np.float32) > 0.5
        e_drv = _err(P, G, m["fov"], "drivable_area")''')

# --- Outputs: panel 14, dt +-0.5, and the new layers
rep('''                  bev(grid, "rgb", "14 prediction, dt 0 (amber boxes: score >= 0.3, dashed white: GT)", composite(o0["prob"], o0["dist"], observed=observed_area(g0)),
                      box_shapes(gtb, GT_OVERLAY, "--", 0.7) + box_shapes([d for d in o0["dets"] if d["score"] >= 0.3], DET_COLOR)),''',
    '''                  bev(grid, "rgb", f"14 prediction, dt 0: boxes right {bx0['tp']}, false {bx0['fp']} (green tint), missed {bx0['fn']} (blue tint, dashed), no care {bx0['nocare']} (violet)",
                      composite(o0["prob"], o0["dist"], observed=observed_area(g0)), bs0),''')
rep('''        P, G = o0["prob"]["drivable_area"] > 0.5, np.asarray(g0["drivable_area"], np.float32) > 0.5
        pv, gv = o0["prob"]["vehicle"] > 0.5, np.asarray(g0["vehicle"], np.float32) > 0.5
        panels = [bev(grid, "rgb", "13 ground truth''', '''        bs0, bx0 = self.pred_boxes(sample, o0, 0.0, past=True)
        P, G = o0["prob"]["drivable_area"] > 0.5, np.asarray(g0["drivable_area"], np.float32) > 0.5
        pv, gv = o0["prob"]["vehicle"] > 0.5, np.asarray(g0["vehicle"], np.float32) > 0.5
        panels = [bev(grid, "rgb", "13 ground truth''')
rep('''            panels.append(bev(grid, "rgb", f"prediction at dt {dt:+.1f} s; brown = unknown cells (splat mass < 0.5); dashed = GT then", img,
                              box_shapes(gt_boxes(sample.reference[dt], t.groups), GT_OVERLAY, "--", 0.7) +
                              box_shapes([d for d in od["dets"] if d["score"] >= 0.3], DET_COLOR)))''',
    '''            bsd, bxd = self.pred_boxes(sample, od, dt)
            panels.append(bev(grid, "rgb", f"prediction at dt {dt:+.1f} s; brown = unknown cells (splat mass < 0.5); boxes right {bxd['tp']}, false {bxd['fp']}, missed {bxd['fn']}", img, bsd))''')

# new panels before the speed block: moving / stationary layers, speed + direction
rep('''        vel, gvel = o0.get("vhat"), (np.asarray(g0["velocity"], np.float32) if "velocity" in g0 else None)
        veh = np.asarray(g0["vehicle"], np.float32) > 0.5''', '''        vel, gvel = o0.get("vhat"), (np.asarray(g0["velocity"], np.float32) if "velocity" in g0 else None)
        if gvel is not None:
            panels += self.layer_panels(grid, o0, g0, gt_prob, gt_dist, m, vel, gvel)
        veh = np.asarray(g0["vehicle"], np.float32) > 0.5''')
rep('''        if gsp is not None:
            panels.append(bev(grid, "raster", "21 GT speed |v| of the vehicle cells (m/s)", np.where(veh, gsp, 0),
                              box_shapes(gtb, "c", "--", 0.7), cmap="viridis", vmax=vmax, colorbar=True))
        if vel is not None:
            sp = np.hypot(vel[0], vel[1])
            panels.append(bev(grid, "raster", "22 predicted speed |v| on the same cells (same colour scale)", np.where(veh, sp, 0),
                              box_shapes(gtb, "c", "--", 0.7), cmap="viridis", vmax=vmax, colorbar=True))
            panels.append(bev(grid, "raster", f"23 predicted speed everywhere (scale to its own max {sp.max():.2f} m/s)", sp,
                              box_shapes(gtb, "c", "--", 0.7), cmap="viridis", colorbar=True))''',
    '''        wheel = "hue = direction of travel (red ahead, green to the left, cyan behind, violet to the right), brightness = speed"
        if gsp is not None:
            panels.append(bev(grid, "rgb", f"21 GT velocity of the vehicle cells: {wheel} (full colour = {vmax:.1f} m/s)", speed_dir_rgb(np.where(veh, 1.0, 0.0) * gvel, vmax),
                              box_shapes(gtb, "0.55", "--", 0.7)))
        if vel is not None:
            sp = np.hypot(vel[0], vel[1])
            panels.append(bev(grid, "rgb", "22 predicted velocity on the same cells (same scale and colours)", speed_dir_rgb(np.where(veh, 1.0, 0.0) * vel, vmax),
                              box_shapes(gtb, "0.55", "--", 0.7)))
            panels.append(bev(grid, "rgb", f"23 predicted velocity everywhere (full colour = {vmax:.1f} m/s; the static world should stay black)", speed_dir_rgb(vel, vmax),
                              box_shapes(gtb, "0.55", "--", 0.7)))''')

# the layer panels method
rep('''class StagesView(BevView):''', '''def _layer_err(P, G, valid, color):
    from evalsuite.render.figure import error_rgb
    return error_rgb(P, G, valid, colors=class_colors(color))


def _layer_panels(self, grid, o0, g0, gt_prob, gt_dist, m, vel, gvel):
    """the semantic output as two layers, each ground truth beside the system result: (a) MOVING objects (vehicles, pedestrians, two-wheelers with a velocity above 0.5 m/s) as
    pixels, red, right = red, false = red + green, missed = red + blue; (b) the STATIONARY world (areas, lines, the objects that stand), ground truth and system as pictures"""
    pm, gm, ps, gs = moving_layers(o0["prob"], vel, gt_prob, gvel)
    valid = m["obj_fov"]
    stat_pred = {n: (a if n not in OBJECT_LAYERS else np.where(ps, a, 0.0)) for n, a in o0["prob"].items()}
    stat_gt = {n: (a if n not in OBJECT_LAYERS else np.where(gs, a, 0.0)) for n, a in gt_prob.items()}
    obs = observed_area(g0)
    return [bev(grid, "rgb", "25 moving objects, ground truth (velocity above 0.5 m/s)", np.where(gm[..., None], np.asarray(MOVING_RED, np.float32), np.asarray(BG, np.float32))),
            bev(grid, "rgb", _caption(error_counts(pm, gm, valid), "26 moving objects, system (red right, + green false, + blue missed)"), _layer_err(pm, gm, valid, MOVING_RED)),
            bev(grid, "rgb", "27 stationary world, ground truth (areas, lines, standing objects)", composite(stat_gt, gt_dist, observed=obs)),
            bev(grid, "rgb", "28 stationary world, system", composite(stat_pred, o0["dist"], observed=obs))]


OutputsView.layer_panels = _layer_panels


class StagesView(BevView):''')

# --- Dashboard: prediction now / dt +-0.5
rep('''        panels.append(bev(grid, "rgb", "prediction now (amber boxes >= 0.3, dashed = GT)", composite(o0["prob"], o0["dist"], observed=observed_area(g0)),
                          boxes + box_shapes([d for d in o0["dets"] if d["score"] >= 0.3], DET_COLOR)))''',
    '''        bsn, bxn = self.pred_boxes(sample, o0, 0.0, past=True)
        panels.append(bev(grid, "rgb", f"prediction now: boxes right {bxn['tp']}, false {bxn['fp']} (green), missed {bxn['fn']} (blue), no care {bxn['nocare']} (violet); faded = 0.5 s ago",
                          composite(o0["prob"], o0["dist"], observed=observed_area(g0)), bsn))''')
rep('''                panels.append(bev(grid, "rgb", f"prediction at dt {dt:+.1f} s (dashed = GT then)", composite(od["prob"], od["dist"], observed=observed_area(sample.reference[dt])),
                                  box_shapes(gt_boxes(sample.reference[dt], t.groups), GT_OVERLAY, "--", 0.7) +
                                  box_shapes([d for d in od["dets"] if d["score"] >= 0.3], DET_COLOR)))''',
    '''                bsd, bxd = self.pred_boxes(sample, od, dt)
                panels.append(bev(grid, "rgb", f"prediction at dt {dt:+.1f} s: boxes right {bxd['tp']}, false {bxd['fp']}, missed {bxd['fn']}",
                                  composite(od["prob"], od["dist"], observed=observed_area(sample.reference[dt])), bsd))''')
rep('''            panels.append(bev(grid, "raster", "predicted speed field (m/s)", sp, boxes, cmap="viridis", colorbar=True, vmax=max(2.0, float(sp.max()))))''',
    '''            panels.append(bev(grid, "rgb", "predicted velocity: hue = direction, brightness = speed (full colour = 10 m/s)", speed_dir_rgb(o0["vhat"], 10.0), boxes))''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
