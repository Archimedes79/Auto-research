p = "evaluation/bev_views.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:80])
    s = s.replace(old, new)


# ---------------------------------------------------------------- new helpers after box_shapes
rep('''def fov_shapes() -> List[Shape]:''', '''NOCARE_COLOR, MOVING_RED = (0.66, 0.58, 0.95), (0.95, 0.20, 0.20)
MOVING_SPEED = 0.5                                                     # m/s: the evaluation's moving / static split (metrics.MOVING_SPEED)


def _outline(b, color, ls, lw) -> List[Shape]:
    return box_shapes([b], color, ls, lw)


def error_box_shapes(dets, gts_all, shares, area, thr: float = 2.0, min_score: float = 0.3):
    """Boxes of a prediction panel with the errors told apart by colour (the convention of the maps: right = the class colour, false = tinted green, missed = tinted blue):
      detection that matches a ground-truth box (within `thr` m)     amber, solid      its ground-truth box: white, dashed, thin
      false detection                                                 amber tinted green, solid, thick
      missed ground-truth box                                         white tinted blue, dashed, thick
      'no care' (D38: less than half of the footprint in the area the rig observes: such a box is neither a hit nor a miss nor a false alarm): violet, thin
    -> (shapes, counts {"tp","fp","fn","nocare"}). gts_all: EVERY annotated box of the scored classes with its footprint share (bev_fusion_area.gt_with_shares)."""
    from evaluation.bev_errors import classify_frame
    from evaluation.bev_fusion_area import prepare
    shown = [d for d in dets if d.get("score", 1.0) >= min_score]
    kept, scored = prepare(shown, gts_all, shares, area, include_partial=False)
    neutral_d = [d for d in shown if not any(d is k for k in kept)]
    neutral_g = [g for g in gts_all if not any(g is k for k in scored)]
    r = classify_frame(kept, scored, sorted({b["group"] for b in list(kept) + list(scored)}), thr, 0.0)
    out: List[Shape] = []
    for b in neutral_g + neutral_d:
        out += _outline(b, NOCARE_COLOR, "-", 0.8)
    for d, g in r["tp"]:
        out += _outline(g, GT_OVERLAY, "--", 0.7) + _outline(d, DET_COLOR, "-", 1.4)
    for d, _kind, _other in r["fp"]:
        out += _outline(d, tinted(DET_COLOR, GREEN), "-", 2.0)
    for g, _kind, _other in r["fn"]:
        out += _outline(g, tinted(GT_OVERLAY, BLUE, 0.6), "--", 2.0)
    return out, {"tp": len(r["tp"]), "fp": len(r["fp"]), "fn": len(r["fn"]), "nocare": len(neutral_g) + len(neutral_d)}


def past_box_shapes(boxes_then, meta_then, meta_now, alpha: float = 0.30, color=GT_OVERLAY) -> List[Shape]:
    """the boxes of an earlier frame (`boxes_then` in the ego frame of `meta_then`) moved into the ego frame of the current frame through the two recorded poses (T_e2g) and
    drawn faded: the history of the scene behind the present. Boxes that moved with the ego car stay where the world kept them."""
    T = np.linalg.inv(np.asarray(meta_now["T_e2g"], float)) @ np.asarray(meta_then["T_e2g"], float)
    dyaw = math.atan2(T[1, 0], T[0, 0])
    moved = []
    for b in boxes_then:
        x, y, *_ = T @ np.array([b["cx"], b["cy"], 0.0, 1.0])
        moved.append(dict(b, cx=float(x), cy=float(y), **({"yaw": b["yaw"] + dyaw} if "yaw" in b else {})))
    return box_shapes(moved, tuple(color) + (alpha,), ":", 0.9)


def radar_points(rad) -> Shape:
    """the radar detections of the window as ONE point layer: colour = the direction of the radial velocity (cool = approaching, warm = receding, grey = about standing still
    relative to the car), transparency = the strength of the return (RCS, relative to this frame: at least 50 % opaque for the weakest)"""
    import matplotlib.cm as cm
    vr = np.asarray(rad["vr"], float)
    rgba = cm.coolwarm((np.clip(vr, -15, 15) + 15) / 30.0).copy()
    rcs = np.asarray(rad["rcs"], float) if "rcs" in rad else np.zeros_like(vr)
    lo, hi = (np.percentile(rcs, 5), np.percentile(rcs, 95)) if len(rcs) > 1 else (0.0, 1.0)
    rgba[:, 3] = 0.5 + 0.5 * np.clip((rcs - lo) / max(hi - lo, 1e-6), 0, 1)
    return Shape("points", np.c_[np.asarray(rad["y"], float), np.asarray(rad["x"], float)], rgba, "-", 0.6)


def speed_dir_rgb(v, vmax: float) -> np.ndarray:
    """a velocity field (2, X, Y) m/s -> RGB (X, Y, 3): hue = the direction of travel (red +x ahead, green-cyan to the left, blue-violet behind, see the wheel note), brightness =
    the speed (black = standing still, full colour = `vmax` and faster)"""
    import matplotlib.colors as mc
    v = np.asarray(v, np.float32)
    sp = np.hypot(v[0], v[1])
    hsv = np.stack([(np.arctan2(v[1], v[0]) / (2 * np.pi)) % 1.0, np.ones_like(sp), np.clip(sp / max(vmax, 1e-6), 0, 1)], -1)
    return mc.hsv_to_rgb(hsv).astype(np.float32)


def moving_layers(prob, vel, gt_prob, gt_vel, groups=OBJECT_LAYERS):
    """the moving objects as one pixel layer: (predicted, ground truth) boolean maps = a cell of any object class (prob > 0.5) whose velocity is above 0.5 m/s. The stationary
    objects are the object cells that are not moving."""
    def objs(pr):
        return np.any([np.asarray(pr[n], np.float32) > 0.5 for n in groups if n in pr], axis=0) if any(n in pr for n in groups) else None
    po, go = objs(prob), objs(gt_prob)
    ps = np.hypot(vel[0], vel[1]) > MOVING_SPEED if vel is not None else np.zeros_like(po)
    gs = np.hypot(gt_vel[0], gt_vel[1]) > MOVING_SPEED
    return (po & ps), (go & gs), (po & ~ps), (go & ~gs)


def fov_shapes() -> List[Shape]:''')

# ---------------------------------------------------------------- the radar points: one layer, direction colour, strength transparency
rep('''        pts = [Shape("points", np.c_[rad["y"].numpy(), rad["x"].numpy()], _doppler_colors(rad["vr"].numpy()), "-", 0.6)]
        panels.append(bev(grid, "rgb", f"4 radar detections of the window ({len(rad['x'])}): colour = Doppler vr (blue -15 .. red +15 m/s)",''',
    '''        pts = [radar_points({k: v.numpy() for k, v in rad.items()})]
        panels.append(bev(grid, "rgb", f"4 radar detections of the window ({len(rad['x'])}): colour = direction of the Doppler (cool = approaching, warm = receding), opacity = strength (RCS)",''')
rep('''        panels.append(bev(grid, "rgb", f"radar detections ({len(rad['x'])}), colour = Doppler", np.zeros((2, 2, 3)),
                          [Shape("points", np.c_[rad["y"].numpy(), rad["x"].numpy()], _doppler_colors(rad["vr"].numpy()), "-", 0.6)] + boxes))''',
    '''        panels.append(bev(grid, "rgb", f"radar detections ({len(rad['x'])}): colour = Doppler direction, opacity = strength", np.zeros((2, 2, 3)),
                          [radar_points({k: v.numpy() for k, v in rad.items()})] + boxes))''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
