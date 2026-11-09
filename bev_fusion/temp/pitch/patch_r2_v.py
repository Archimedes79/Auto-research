p = "evaluation/bev_views.py"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


# 3: prepare gets EVERY ground-truth box (the metric keeps the ones beyond the window as neutral for the detections)
rep('''    shown = [d for d in dets if d.get("score", 1.0) >= min_score]
    inside_d = [d for d in shown if in_front(d)]
    pairs = [(g, f) for g, f in zip(gts_all, shares) if in_front(g)]
    gts_in, shares_in = [g for g, _ in pairs], [f for _, f in pairs]
    kept, scored = prepare(inside_d, gts_in, shares_in, area, include_partial=False)
    neutral_d = [d for d in inside_d if not any(d is k for k in kept)]
    neutral_g = [g for g in gts_in if not any(g is k for k in scored)]
    r = classify_frame(kept, scored, sorted({b["group"] for b in list(kept) + list(scored)}), thr, 0.0)''', '''    shown = [d for d in dets if d.get("score", 1.0) >= min_score]
    inside_d = [d for d in shown if in_front(d)]
    kept, scored = prepare(inside_d, list(gts_all), list(shares), area, include_partial=False)         # as the metric: ALL annotated boxes, the ones beyond the window only neutralise
    neutral_d = [d for d in inside_d if not any(d is k for k in kept)]
    neutral_g = [g for g in gts_all if in_front(g) and not any(g is k for k in scored)]               # counted: the window's no-care boxes''')
rep('''    out: List[Shape] = box_shapes([b for b in list(shown) + list(gts_all) if not in_front(b)], OUTSIDE_COLOR, "-", 0.6)''', '''    r = classify_frame(kept, scored, sorted({b["group"] for b in list(kept) + list(scored)}), thr, 0.0)
    out: List[Shape] = box_shapes([b for b in list(shown) + list(gts_all) if not in_front(b)], OUTSIDE_COLOR, "-", 0.6)''')

# 7: no predicted velocity (a prior subject): no moving / stationary comparison
rep('''        known = np.asarray(g0["flow_weight"], np.float32) > 0 if "flow_weight" in g0 else np.ones(np.asarray(gvel[0]).shape, bool)''', '''        if vel is None:                                                       # this subject predicts no velocity: "moving" and "stationary" cannot be compared
            return [Panel("text", "25 - 28 moving objects / stationary world", "this subject has no velocity output: no moving / stationary comparison"),
                    *self._heat_panels(grid, o0, g0, gt_prob, gt_dist, m, None, gvel)]
        known = np.asarray(g0["flow_weight"], np.float32) > 0 if "flow_weight" in g0 else np.ones(np.asarray(gvel[0]).shape, bool)''')
rep('''        obs = observed_area(g0)
        heat = []
        for name, title in (("drivable_area", "29 drivable area"), ("vehicle", "30 vehicle footprint")):''', '''        obs = observed_area(g0)
        heat = self._heat_panels(grid, o0, g0, gt_prob, gt_dist, m, vel, gvel)
        return [bev(grid, "rgb", "25 moving objects, ground truth (velocity above 0.5 m/s, known velocity)", np.where(gm[..., None], np.asarray(MOVING_RED, np.float32), np.asarray(BG, np.float32))),
                bev(grid, "rgb", _caption(error_counts(pm, gm, valid), "26 moving objects, system (red right, + green false, + blue missed)"), _layer_err(pm, gm, valid, MOVING_RED)),
                bev(grid, "rgb", "27 stationary world, ground truth (areas, lines, standing objects)", composite(stat_gt, gt_dist, observed=obs)),
                bev(grid, "rgb", "28 stationary world, system", composite(stat_pred, o0["dist"], observed=obs))] + heat

    def _heat_panels(self, grid, o0, g0, gt_prob, gt_dist, m, vel, gvel):
        """the loss of the drivable-area / vehicle layers and the velocity error as a heat over the GT"""
        known = np.asarray(g0["flow_weight"], np.float32) > 0 if "flow_weight" in g0 else np.ones(np.asarray(gvel[0]).shape, bool)
        obs = observed_area(g0)
        heat = []
        for name, title in (("drivable_area", "29 drivable area"), ("vehicle", "30 vehicle footprint")):''')
# remove the old tail of the method (the return list that followed the heat loop)
a = s.index('''            heat.append(bev(grid, "rgb", "31 velocity error on the object cells of known velocity''')
b = s.index("    def render(self, sample: Sample, pred: Prediction, ctx: ViewContext) -> Page:", a)
tail = '''            heat.append(bev(grid, "rgb", "31 velocity error on the object cells of known velocity (|v_pred - v_GT|, white = 4 m/s) as a heat over the GT",
                            loss_overlay(composite(gt_prob, gt_dist, observed=obs), lv, vmax=4.0)))
        return heat

'''
s = s[:a] + tail + s[b:]

# panel 23 keeps its own-max scale in the title
rep('''f"23 predicted velocity everywhere (full colour = {vmax:.1f} m/s; the static world should stay black)"''', '''f"23 predicted velocity everywhere (full colour = {vmax:.1f} m/s, the same scale as 21 / 22; the predicted maximum is {sp.max():.1f} m/s; the static world should stay black)"''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
