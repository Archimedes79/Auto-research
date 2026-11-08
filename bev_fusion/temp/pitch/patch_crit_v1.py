p = "evaluation/bev_views.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:90])
    s = s.replace(old, new)


rep("from evaluation.metrics import FRONT_HALF_FOV, angle_error, dilate, gt_boxes_from_meta, line_cells",
    "from evaluation.metrics import FRONT_HALF_FOV, MOVING_SPEED, angle_error, dilate, gt_boxes_from_meta, in_front, line_cells")
rep('''NOCARE_COLOR, MOVING_RED = (0.66, 0.58, 0.95), (0.95, 0.20, 0.20)
MOVING_SPEED = 0.5                                                     # m/s: the evaluation's moving / static split (metrics.MOVING_SPEED)


def _outline(b, color, ls, lw) -> List[Shape]:
    return box_shapes([b], color, ls, lw)
''', '''NOCARE_COLOR, MOVING_RED = (0.66, 0.58, 0.95), (0.95, 0.20, 0.20)
OUTSIDE_COLOR = (0.66, 0.58, 0.95, 0.35)                                # boxes outside the scored window: faint violet, not counted
''')

# ---- error_box_shapes: the scored window
a = s.index("def error_box_shapes(")
b = s.index("def past_box_shapes(")
s = s[:a] + '''def error_box_shapes(dets, gts_all, shares, area, thr: float = 2.0, min_score: float = 0.3):
    """Boxes of a prediction panel with the errors told apart by colour (the convention of the maps: right = the class colour, false = tinted green, missed = tinted blue):
      detection that matches a ground-truth box (within `thr` m)     amber, solid      its ground-truth box: white, dashed, thin
      false detection                                                 amber tinted green, solid, thick
      missed ground-truth box                                         white tinted blue, dashed, thick
      'no care' (D38: less than half of the footprint in the area the rig observes: such a box is neither a hit nor a miss nor a false alarm): violet, thin
      outside the scored window of the numbers (x < 60 m, +-32.3 deg): faint violet, not counted
    -> (shapes, counts {"tp","fp","fn","nocare"}): the counts are those of the window, as the numbers count. gts_all: EVERY annotated box of the scored classes with its footprint
    share (bev_fusion_area.gt_with_shares); the detections are drawn from score `min_score` on (what the pictures always did)."""
    from evaluation.bev_errors import classify_frame
    from evaluation.bev_fusion_area import prepare
    shown = [d for d in dets if d.get("score", 1.0) >= min_score]
    inside_d = [d for d in shown if in_front(d)]
    pairs = [(g, f) for g, f in zip(gts_all, shares) if in_front(g)]
    gts_in, shares_in = [g for g, _ in pairs], [f for _, f in pairs]
    kept, scored = prepare(inside_d, gts_in, shares_in, area, include_partial=False)
    neutral_d = [d for d in inside_d if not any(d is k for k in kept)]
    neutral_g = [g for g in gts_in if not any(g is k for k in scored)]
    r = classify_frame(kept, scored, sorted({b["group"] for b in list(kept) + list(scored)}), thr, 0.0)
    out: List[Shape] = box_shapes([b for b in list(shown) + list(gts_all) if not in_front(b)], OUTSIDE_COLOR, "-", 0.6)
    out += box_shapes(neutral_g + neutral_d, NOCARE_COLOR, "-", 0.8)
    for d, g in r["tp"]:
        out += box_shapes([g], GT_OVERLAY, "--", 0.7) + box_shapes([d], DET_COLOR, "-", 1.4)
    for d, _kind, _other in r["fp"]:
        out += box_shapes([d], tinted(DET_COLOR, GREEN), "-", 2.0)
    for g, _kind, _other in r["fn"]:
        out += box_shapes([g], tinted(GT_OVERLAY, BLUE, 0.6), "--", 2.0)
    return out, {"tp": len(r["tp"]), "fp": len(r["fp"]), "fn": len(r["fn"]), "nocare": len(neutral_g) + len(neutral_d)}


def box_caption(c) -> str:
    """the legend sentence of a prediction panel's boxes from the counts of error_box_shapes (None: the plain boxes of the fallback)"""
    if c is None:
        return "boxes: amber = detections (score >= 0.3), dashed = GT (error colours not available for this frame)"
    return f"boxes right {c['tp']} (amber), false {c['fp']} (green tint), missed {c['fn']} (blue tint, dashed), no care {c['nocare']} (violet), scored window"


''' + s[b:]

# ---- the radar layer: NaN safe
rep('''    vr = np.asarray(rad["vr"], float)
    k = np.clip(np.abs(vr) / 6.0, 0, 1)[:, None] ''', '''    vr = np.nan_to_num(np.asarray(rad["vr"], float), nan=0.0)                         # a Doppler that was not measured counts as standing
    k = np.clip(np.abs(vr) / 6.0, 0, 1)[:, None] ''')
rep('''    rcs = np.asarray(rad["rcs"], float) if "rcs" in rad else np.zeros_like(vr)
    lo, hi = (np.percentile(rcs, 5), np.percentile(rcs, 95)) if len(rcs) > 1 else (0.0, 1.0)
    rgba[:, 3] = 0.5 + 0.5 * np.clip((rcs - lo) / max(hi - lo, 1e-6), 0, 1)''', '''    rcs = np.asarray(rad["rcs"], float) if "rcs" in rad else np.zeros_like(vr)
    ok = np.isfinite(rcs)
    lo, hi = (np.percentile(rcs[ok], 5), np.percentile(rcs[ok], 95)) if ok.sum() > 1 else (0.0, 1.0)
    rgba[:, 3] = 0.5 + 0.5 * np.clip(np.where(ok, (rcs - lo) / max(hi - lo, 1e-6), 0.0), 0, 1)''')

# ---- loss heat without red
rep('''    """a loss map as a transparent heat (black -> yellow -> white, opacity = the loss relative to `vmax`) over a picture: where the loss sits, on the ground truth it belongs to"""
    import matplotlib.cm as cm
    a = (np.clip(np.asarray(loss, np.float32) / vmax, 0, 1) * strength)[..., None]
    heat = cm.afmhot(np.clip(np.asarray(loss, np.float32) / vmax, 0, 1))[..., :3]
    return (np.asarray(base_rgb, np.float32) * (1 - a) + heat.astype(np.float32) * a).astype(np.float32)''',
    '''    """a loss map as a transparent heat (a ramp without red: yellow, then white; opacity = the loss relative to `vmax`) over a picture: where the loss sits, on the ground truth it
    belongs to"""
    t = np.clip(np.asarray(loss, np.float32) / vmax, 0, 1)
    yellow, white = np.asarray((0.95, 0.85, 0.15), np.float32), np.ones(3, np.float32)
    heat = np.where((t < 0.6)[..., None], yellow, yellow + (white - yellow) * ((t - 0.6) / 0.4)[..., None])
    a = (t * strength)[..., None]
    return (np.asarray(base_rgb, np.float32) * (1 - a) + heat * a).astype(np.float32)''')

# ---- moving_layers: no object layers
rep('''    def objs(pr):
        return np.any([np.asarray(pr[n], np.float32) > 0.5 for n in groups if n in pr], axis=0) if any(n in pr for n in groups) else None
    po, go = objs(prob), objs(gt_prob)
    ps = np.hypot(vel[0], vel[1]) > MOVING_SPEED if vel is not None else np.zeros_like(po)
    gs = np.hypot(gt_vel[0], gt_vel[1]) > MOVING_SPEED
    return (po & ps), (go & gs), (po & ~ps), (go & ~gs)''', '''    shape = np.asarray(gt_vel[0]).shape

    def objs(pr):
        have = [np.asarray(pr[n], np.float32) > 0.5 for n in groups if n in pr]
        return np.any(have, axis=0) if have else np.zeros(shape, bool)               # a model / GT without object layers: nothing moves
    po, go = objs(prob), objs(gt_prob)
    ps = np.hypot(vel[0], vel[1]) > MOVING_SPEED if vel is not None else np.zeros(shape, bool)
    gs = np.hypot(gt_vel[0], gt_vel[1]) > MOVING_SPEED
    return (po & ps), (go & gs), (po & ~ps), (go & ~gs)''')

# ---- pred_boxes: fall back to the plain boxes on any failure
rep('''        from evaluation.bev_fusion_area import fusion_area, gt_with_shares
        g = sample.reference[dt]
        gts, shares = gt_with_shares(g, self.t.groups)
        shapes, c = error_box_shapes(o["dets"], gts, shares, fusion_area(g))
        if past and dt == 0.0 and -0.5 in sample.reference:
            try:
                shapes = past_box_shapes(gt_boxes(sample.reference[-0.5], self.t.groups), sample.reference[-0.5]["meta"], g["meta"]) + shapes
            except KeyError:                                          # a GT without poses in its meta
                pass
        return shapes, c''', '''        from evaluation.bev_fusion_area import fusion_area, gt_with_shares
        g = sample.reference[dt]
        try:
            gts, shares = gt_with_shares(g, self.t.groups)
            shapes, c = error_box_shapes(o["dets"], gts, shares, fusion_area(g))
        except Exception:                                             # noqa: BLE001  (a GT without the fusion layers, an odd box: the picture must not cost the evaluation its cards)
            return box_shapes(gt_boxes(g, self.t.groups), GT_OVERLAY, "--", 0.7) + box_shapes([d for d in o["dets"] if d.get("score", 1.0) >= 0.3], DET_COLOR), None
        if past and dt == 0.0 and -0.5 in sample.reference:
            try:
                shapes = past_box_shapes(gt_boxes(sample.reference[-0.5], self.t.groups), sample.reference[-0.5]["meta"], g["meta"]) + shapes
            except Exception:                                         # noqa: BLE001  (no poses in the meta, a singular pose)
                pass
        return shapes, c''')

# ---- titles
rep('''f"prediction: boxes right {bx['tp']} (amber), false {bx['fp']} (amber + green), missed {bx['fn']} (dashed + blue), no care {bx['nocare']} (violet); faded = 0.5 s ago",''',
    '''f"prediction: {box_caption(bx)}; faded = 0.5 s ago",''')
rep('''f"14 prediction, dt 0: boxes right {bx0['tp']}, false {bx0['fp']} (green tint), missed {bx0['fn']} (blue tint, dashed), no care {bx0['nocare']} (violet)",''',
    '''f"14 prediction, dt 0: {box_caption(bx0)}",''')
rep('''f"prediction at dt {dt:+.1f} s; brown = unknown cells (splat mass < 0.5); boxes right {bxd['tp']}, false {bxd['fp']}, missed {bxd['fn']}"''',
    '''f"prediction at dt {dt:+.1f} s; brown = unknown cells (splat mass < 0.5); {box_caption(bxd)}"''')
rep('''f"prediction now: boxes right {bxn['tp']}, false {bxn['fp']} (green), missed {bxn['fn']} (blue), no care {bxn['nocare']} (violet); faded = 0.5 s ago",''',
    '''f"prediction now: {box_caption(bxn)}; faded = 0.5 s ago",''')
rep('''f"prediction at dt {dt:+.1f} s: boxes right {bxd['tp']}, false {bxd['fp']}, missed {bxd['fn']}",''', '''f"prediction at dt {dt:+.1f} s: {box_caption(bxd)}",''')

# ---- layer panels as a method of OutputsView, guarded, unknown velocity masked
a = s.index("def _layer_err(P, G, valid, color):")
b = s.index("class StagesView(BevView):")
s = s[:a] + "def _layer_err(P, G, valid, color):\n    from evalsuite.render.figure import error_rgb\n    return error_rgb(P, G, valid, colors=class_colors(color))\n\n\n\n" + s[b:]
rep('''class OutputsView(BevView):
    name = "outputs"
    description = "detailed results: GT, prediction, errors, dt +-0.5 s, splat mass, scores, speed"
''', '''class OutputsView(BevView):
    name = "outputs"
    description = "detailed results: GT, prediction, errors, dt +-0.5 s, splat mass, scores, speed"

    def layer_panels(self, grid, o0, g0, gt_prob, gt_dist, m, vel, gvel):
        """the semantic output as two layers, each ground truth beside the system result: (a) MOVING objects (vehicles, pedestrians, two-wheelers with a velocity above
        0.5 m/s) as pixels, red, right = red, false = red + green, missed = red + blue; (b) the STATIONARY world (areas, lines, the objects that stand); then the loss of
        the drivable-area / vehicle layers and the velocity error as a heat over the GT. Cells of unknown velocity (flow_weight 0: GT velocity 0 means "not known") are left out
        of the moving / stationary split and of the velocity error. A failure here costs the page these panels, nothing else."""
        try:
            return self._layer_panels(grid, o0, g0, gt_prob, gt_dist, m, vel, gvel)
        except Exception:                                                     # noqa: BLE001
            return [Panel("text", "25 - 31 moving / stationary layers and loss heat", "not available for this frame")]

    def _layer_panels(self, grid, o0, g0, gt_prob, gt_dist, m, vel, gvel):
        known = np.asarray(g0["flow_weight"], np.float32) > 0 if "flow_weight" in g0 else np.ones(np.asarray(gvel[0]).shape, bool)
        pm, gm, ps, gs = (x & known for x in moving_layers(o0["prob"], vel, gt_prob, gvel))
        valid = m["obj_fov"] & known
        stat_pred = {n: (a if n not in OBJECT_LAYERS else np.where(ps, a, 0.0)) for n, a in o0["prob"].items()}
        stat_gt = {n: (a if n not in OBJECT_LAYERS else np.where(gs, a, 0.0)) for n, a in gt_prob.items()}
        obs = observed_area(g0)
        heat = []
        for name, title in (("drivable_area", "29 drivable area"), ("vehicle", "30 vehicle footprint")):
            if name in o0["prob"] and name in gt_prob:
                lm = bce_map(o0["prob"][name], gt_prob[name])
                v_ = m["fov"] if name == "drivable_area" else m["obj_fov"]
                heat.append(bev(grid, "rgb", f"{title}: loss per cell (BCE) as a heat over the GT (yellow, white = high); mean on the evaluated cells ({'camera cells' if name == 'drivable_area' else 'object cells'}) "
                                f"{float(lm[v_].mean()) if v_.any() else float('nan'):.3f}", loss_overlay(composite(gt_prob, gt_dist, observed=obs), np.where(v_, lm, 0.0))))
        if vel is not None:
            objs = np.any([np.asarray(gt_prob[n]) > 0.5 for n in OBJECT_LAYERS if n in gt_prob], axis=0) if any(n in gt_prob for n in OBJECT_LAYERS) else np.zeros(known.shape, bool)
            lv = np.hypot(vel[0] - gvel[0], vel[1] - gvel[1]) * (objs & known)
            heat.append(bev(grid, "rgb", "31 velocity error on the object cells of known velocity (|v_pred - v_GT|, white = 4 m/s) as a heat over the GT",
                            loss_overlay(composite(gt_prob, gt_dist, observed=obs), lv, vmax=4.0)))
        return [bev(grid, "rgb", "25 moving objects, ground truth (velocity above 0.5 m/s, known velocity)", np.where(gm[..., None], np.asarray(MOVING_RED, np.float32), np.asarray(BG, np.float32))),
                bev(grid, "rgb", _caption(error_counts(pm, gm, valid), "26 moving objects, system (red right, + green false, + blue missed)"), _layer_err(pm, gm, valid, MOVING_RED)),
                bev(grid, "rgb", "27 stationary world, ground truth (areas, lines, standing objects)", composite(stat_gt, gt_dist, observed=obs)),
                bev(grid, "rgb", "28 stationary world, system", composite(stat_pred, o0["dist"], observed=obs))] + heat
''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
