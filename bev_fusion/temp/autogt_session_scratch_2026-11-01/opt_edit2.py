p = "autogt/nuscenes_gt.py"
s = open(p, encoding="utf-8").read()
# 1 the map: a polygon of the map never changes, the devkit builds its shapely object from the nodes on every call
old = """        if loc not in self._maps:
            self._maps[loc] = NuScenesMap(dataroot=self.dataroot, map_name=loc)
        return self._maps[loc]
"""
new = """        if loc not in self._maps:
            nmap = NuScenesMap(dataroot=self.dataroot, map_name=loc)
            # a polygon of the map never changes, but the devkit builds its shapely object from the nodes on every call (11,000 times per
            # keyframe): one object per token, the same one every time (read only)
            inner, done = nmap.explorer.extract_polygon, {}

            def cached(token):
                if token not in done:
                    done[token] = inner(token)
                return done[token]
            nmap.explorer.extract_polygon = cached
            self._maps[loc] = nmap
        return self._maps[loc]
"""
assert old in s
s = s.replace(old, new, 1)
# 2 the centre-line segments: one conversion per arc instead of one per segment
old = """            xy = to_ego(P[:, :2])
            vectors["centreline"].append(xy)
            hd = P[:, 2] - yaw_e
            R.lines(cmask, [xy])
            for i in range(len(xy) - 1):                        # direction per segment
                seg = xy[i:i + 2]
                cv2.polylines(ccos, [R.pts(seg)], False, float(math.cos(hd[i])), 1, cv2.LINE_8, SHIFT)
                cv2.polylines(csin, [R.pts(seg)], False, float(math.sin(hd[i])), 1, cv2.LINE_8, SHIFT)
"""
new = """            xy = to_ego(P[:, :2])
            vectors["centreline"].append(xy)
            hd = P[:, 2] - yaw_e
            R.lines(cmask, [xy])
            pp = R.pts(xy)                                      # the integer points of the arc, once (a segment is two of them)
            for i in range(len(xy) - 1):                        # direction per segment
                cv2.polylines(ccos, [pp[i:i + 2]], False, float(math.cos(hd[i])), 1, cv2.LINE_8, SHIFT)
                cv2.polylines(csin, [pp[i:i + 2]], False, float(math.sin(hd[i])), 1, cv2.LINE_8, SHIFT)
"""
assert old in s
s = s.replace(old, new, 1)
# 3 the shares of all boxes of a keyframe in one pass
old = """        for b in boxes:                                          # per object: the share of its footprint the sensors observe (D38 b)
            sh = cvg.object_shares(b, prisms, cam_xyz, K, T_c2e, img_size, poses, grid)
            b["share_camera"], b["share_radar"], b["share_fusion"] = sh["camera"], sh["radar"], sh["fusion"]
"""
new = """        for b, sh in zip(boxes, cvg.object_shares_many(boxes, prisms, cam_xyz, K, T_c2e, img_size, poses, grid)):
            b["share_camera"], b["share_radar"], b["share_fusion"] = sh["camera"], sh["radar"], sh["fusion"]     # per object, D38 b
"""
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)

p = "autogt/coverage.py"
s = open(p, encoding="utf-8").read()
a = s.index("def object_shares(box, prisms, cam, K, T_c2e, size, radar_poses, grid, step=STEP):")
b = s.index("def shares_in(g, corners, height=None, group=None, grid=None):")
new = '''def object_shares_many(boxes, prisms, cam, K, T_c2e, size, radar_poses, grid, step=STEP):
    """the observed shares of every object in `boxes` (dicts with `corners` and `size`; a box may be one of the `prisms`: it does not hide
    its own samples), one pass for all of them: the share of the footprint in the camera's view with the whole height of the object
    counted (the camera sees a footprint sample when it sees the object above it at the bottom, the middle or the top: inside the image
    and not hidden by another box), in the radar fan, and in both. Samples outside the grid count as not observed. -> a list of
    {"camera", "radar", "fusion"}"""
    zero = {"camera": 0.0, "radar": 0.0, "fusion": 0.0}
    xys, zs, spans = [], [], []
    for box in boxes:
        fp, z_lo, z_hi = prism_of(box)
        xy = footprint_samples(fp, step)
        ok = _on_grid(xy, grid)
        if not ok.any():                             # entirely outside the grid: nothing to observe
            spans.append(None)
            continue
        spans.append((sum(len(x) for x in xys), len(xy), ok))
        xys.append(xy)
        zs.append(z_lo + np.asarray(FRACTIONS) * (z_hi - z_lo))
    if not xys:
        return [dict(zero) for _ in boxes]
    xy_all = np.concatenate(xys)
    pts = np.concatenate([np.c_[xy_all[start:start + n], np.full(n, z[k])] for k in range(len(FRACTIONS))
                          for (start, n, _), z in ((sp, zs[j]) for j, sp in enumerate(s for s in spans if s is not None))])
    vis = camera_cells(pts[:, 0], pts[:, 1], K, T_c2e, size, z=pts[:, 2]) & ~blocked(pts, cam, prisms)
    fan = radar_cells(xy_all[:, 0], xy_all[:, 1], radar_poses)
    out, j, base = [], 0, 0
    per_height = len(xy_all)
    for sp in spans:
        if sp is None:
            out.append(dict(zero))
            continue
        start, n, ok = sp
        seen = np.zeros(n, bool)
        for k in range(len(FRACTIONS)):
            seen |= vis[k * per_height + start:k * per_height + start + n]
        f = fan[start:start + n]
        out.append({"camera": float((ok & seen).mean()), "radar": float((ok & f).mean()), "fusion": float((ok & seen & f).mean())})
    return out


def object_shares(box, prisms, cam, K, T_c2e, size, radar_poses, grid, step=STEP):
    """the observed shares of one object, see `object_shares_many` -> {"camera", "radar", "fusion"}"""
    return object_shares_many([box], prisms, cam, K, T_c2e, size, radar_poses, grid, step)[0]


'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("edited")
