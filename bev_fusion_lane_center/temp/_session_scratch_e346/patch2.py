p = "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e346/e346_lib.py"
s = open(p, encoding="utf-8").read()
a = s.index("def collision_counts(lane):")
b = s.index("# ------------------------------------------------------------------------------------------------------------------------------ one frame")
new = '''def collision_counts(lane):
    """the chain code's cell collisions on the lane GT, connector points kept (no intersection suppression), on the grid in force. Two pieces with points in one cell are ONE node when their closest
    two points there are within COLL_DIST m and COLL_ANG degrees in heading (pieces that run together: a shared entry or exit, a join: consecutive pieces share their end point), else a collision.
    -> [occupied cells, cells with two or more pieces, cells with two pieces whose closest points differ in heading by more than COLL_ANG degrees (crossing), collision cells, connectors (`connector`
    segments of the lane GT), connectors with a collision cell along them (a cell of theirs in which their piece collides), the same two numbers for the connector segments inside an intersection]"""
    cells, keys, aff, pts, tans = {}, [], [], [], []
    for i, p in enumerate(lane["pieces"]):
        q = np.asarray(p["pts"], float).reshape(-1, 2)
        pts.append(q)
        tans.append(QT._tangents(q))
        aff.append(set())
        c = QT.cell_of(q)
        key = c[:, 0] * QT.N + c[:, 1]
        keys.append(key)
        for k in np.unique(key):
            cells.setdefault(int(k), []).append((i, np.flatnonzero(key == k)))
    out = np.zeros(8)
    out[0] = len(cells)
    for key, members in cells.items():
        if len(members) < 2:
            continue
        out[1] += 1
        coll = cross = False
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                (ia, ja), (ib, jb) = members[x], members[y]
                d = np.linalg.norm(pts[ia][ja][:, None] - pts[ib][jb][None], axis=2)
                u, v = np.unravel_index(d.argmin(), d.shape)
                ang = np.degrees(np.arccos(np.clip(float(tans[ia][ja[u]] @ tans[ib][jb[v]]), -1.0, 1.0)))
                bad = d[u, v] > COLL_DIST or ang > COLL_ANG
                cross |= ang > COLL_ANG
                coll |= bad
                if bad:
                    aff[ia].add(key)
                    aff[ib].add(key)
        out[2] += cross
        out[3] += coll
    for i, p in enumerate(lane["pieces"]):
        for s in p["segments"]:
            if s["kind"] != "connector":
                continue
            hit = bool(aff[i] & set(keys[i][s["i0"]: s["i1"] + 1].tolist()))
            out[4] += 1
            out[5] += hit
            if s["in_intersection"]:
                out[6] += 1
                out[7] += hit
    return out


'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)
print("ok")
