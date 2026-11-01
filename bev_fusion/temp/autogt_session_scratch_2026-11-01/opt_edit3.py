p = "autogt/coverage.py"
s = open(p, encoding="utf-8").read()
a = s.index("def object_shares_many(")
b = s.index("def object_shares(box, prisms, cam, K, T_c2e, size, radar_poses, grid, step=STEP):")
new = '''def object_shares_many(boxes, prisms, cam, K, T_c2e, size, radar_poses, grid, step=STEP):
    """the observed shares of every object in `boxes` (dicts with `corners` and `size`; a box may be one of the `prisms`: it does not hide
    its own samples), one pass for all of them: the share of the footprint in the camera's view with the whole height of the object
    counted (the camera sees a footprint sample when it sees the object above it at the bottom, the middle or the top: inside the image
    and not hidden by another box), in the radar fan, and in both. Samples outside the grid count as not observed. -> a list of
    {"camera", "radar", "fusion"}"""
    zero = {"camera": 0.0, "radar": 0.0, "fusion": 0.0}
    xs, zk, spans, start = [], [], [], 0
    for box in boxes:
        fp, z_lo, z_hi = prism_of(box)
        xy = footprint_samples(fp, step)
        ok = _on_grid(xy, grid)
        if not ok.any():                             # entirely outside the grid: nothing to observe
            spans.append(None)
            continue
        n = len(xy)
        xs.append(xy)
        zk.append(np.repeat((z_lo + np.asarray(FRACTIONS) * (z_hi - z_lo))[:, None], n, axis=1))        # (heights, n)
        spans.append((start, n, ok))
        start += n
    if not xs:
        return [dict(zero) for _ in boxes]
    xy_all, z_all = np.concatenate(xs), np.concatenate(zk, axis=1)
    pts = np.concatenate([np.c_[xy_all, z_all[k]] for k in range(len(FRACTIONS))])
    vis = (camera_cells(pts[:, 0], pts[:, 1], K, T_c2e, size, z=pts[:, 2]) & ~blocked(pts, cam, prisms)).reshape(len(FRACTIONS), -1)
    seen_all = vis.any(0)
    fan_all = radar_cells(xy_all[:, 0], xy_all[:, 1], radar_poses)
    out = []
    for sp in spans:
        if sp is None:
            out.append(dict(zero))
            continue
        s0, n, ok = sp
        seen, fan = seen_all[s0:s0 + n], fan_all[s0:s0 + n]
        out.append({"camera": float((ok & seen).mean()), "radar": float((ok & fan).mean()), "fusion": float((ok & seen & fan).mean())})
    return out


'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("coverage rewritten")
