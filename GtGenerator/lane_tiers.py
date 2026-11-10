"""Relevance tiers and visibility of the lane GT (lane version l4; decision D59 / D60 of the lane project).

Every point of every lane piece of a keyframe graph gets two labels, stored next to `pts` (arrays of the length of `pts`):
  tier  1 = the ego lane: the path the car really drives. NON-CAUSAL: the CAM_FRONT ego poses `tier_back` m before and `tier_ahead` m after the keyframe are the path (labels
            only, never an input of a network). Where the car stops or the sequence ends before `min_ahead` m, the path goes on along the map (the straightest successor).
        2 = the lanes directly beside a tier-1 piece (the `left` / `right` neighbours of the road graph, same direction) within `neighbour_tol` m of a tier-1 point
        3 = everything else (branches not taken, opposite lanes (`opposite_tier`), far lanes)
  vis   True when at least one of the cameras of the rig (`cameras`: the six of Spurmitte, CAM_FRONT for BEV Fusion, l5) sees the ground point (`rig_masks`, the nominal calibration) and no observed vehicle / pedestrian / two-wheeler
        box of the keyframe hides it (the box with its real height, `coverage` geometry; a box does not hide a point inside itself). Boxes are at the annotation time, not moved to
        the camera time (at most 0.9 m). Static things (buildings, trees) are not in the nuScenes boxes: a point behind a building counts as visible. The sensor is the CAMERA: radar
        and range are not part of it.
  occ   True when the point is in a camera's view but hidden by a box (vis False although covered); vis False and occ False = outside every camera's view.
  tier_src (per point) 0 = not tier 1, 1 = on the driven path, 2 = tier 1 by the map extension.
Events get `tier` (the lowest tier of the pieces they belong to near the event) and `vis`. The graph gets `ego_path` = {"pts": (n, 2) in the frame of the graph, `back_m`, `ahead_m`
(driven metres)} and `tier_info` (lengths in metres per tier and their visible share, the number of hiding boxes, the extension length).

    g = lane_gt.load(token)             # l4: g["pieces"][i]["tier"], ["vis"], ["tier_src"], g["events"][j]["tier"], g["ego_path"]["pts"]
    lane_tiers.selftest()               # toy lane: path, neighbours, opposite lane, crossing connector, extension, hiding box (a few seconds, numpy only)
"""
import math

import numpy as np

from . import lane_graph as LG
from . import rig_masks as RM

STEP = 0.5                                                   # the point spacing of the pieces
MIN_PATH_STEP = 0.3                                          # ego positions closer than this to the last kept one are dropped (a standing car)


# --------------------------------------------------------------------------------------------------------------------------- copies of the front GT's geometry
# The four helpers below are COPIES of `nuscenes_adapter.category_group` and `coverage.camera_cells / _wedge / _clip`: those files belong to the front GT's signature, and a change there must not
# silently change the l4 labels without changing the lane signature (the critic pass of 2026-11-07). The calibration constants come from rig_masks.py, which is hashed into the lane signature.
def category_group(name):
    """nuScenes category -> group (None = not a hiding class)"""
    if name.startswith(("vehicle.bicycle", "vehicle.motorcycle")):
        return "two_wheeler"
    if name.startswith("vehicle."):
        return "vehicle"
    if name.startswith("human.pedestrian"):
        return "pedestrian"
    return None


def camera_cells(gx, gy, K, T_c2e, size, z=0.0, near=0.5):
    """bool array of gx's shape: the point (x, y, z) in the ego frame is in front of the camera (depth > near) and projects into the image of `size` = (width, height)"""
    gx, gy = np.asarray(gx, float), np.asarray(gy, float)
    P = np.stack([gx.ravel(), gy.ravel(), np.broadcast_to(np.asarray(z, float), gx.shape).ravel(), np.ones(gx.size)], 1) @ np.linalg.inv(T_c2e).T
    d = P[:, 2]
    u = K[0, 0] * P[:, 0] / np.maximum(d, 1e-6) + K[0, 2]
    v = K[1, 1] * P[:, 1] / np.maximum(d, 1e-6) + K[1, 2]
    return ((d > near) & (u >= 0) & (u < size[0]) & (v >= 0) & (v < size[1])).reshape(gx.shape)


def _clip(cam_xy, pts, fp):
    """for the horizontal segments cam -> pts (n, 2): the parameter interval [t0, t1] inside the convex quad fp (clipped to [0, 1]; t0 > t1 = the segment misses it); t1 == 1 means that the end
    point is inside the quad"""
    d = pts - cam_xy
    t0, t1 = np.zeros(len(pts)), np.ones(len(pts))
    cen = fp.mean(0)
    for k in range(4):
        a, e = fp[k], fp[(k + 1) % 4] - fp[k]
        n = np.array([e[1], -e[0]])
        if n @ (cen - a) > 0:
            n = -n                                   # outward normal
        num = float(n @ (cam_xy - a))                # > 0: the camera is outside this edge
        den = d @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -num / den
        pos, neg = den > 1e-12, den < -1e-12
        t1 = np.where(pos, np.minimum(t1, t), t1)
        t0 = np.where(neg, np.maximum(t0, t), t0)
        if num > 0:
            t1 = np.where(pos | neg, t1, -1.0)       # parallel to the edge and outside: no hit
    return t0, t1


def _wedge(cam_xy, pts, fp):
    """a cheap pre-filter: the points inside the angular cone that a footprint spans as seen from the camera (all of them when the camera is so close that the cone is wide)"""
    v = fp - cam_xy
    vc = fp.mean(0) - cam_xy
    ang = np.arctan2(vc[0] * v[:, 1] - vc[1] * v[:, 0], vc[0] * v[:, 0] + vc[1] * v[:, 1])
    if ang.max() - ang.min() > 2.6 or (vc @ vc) < 1e-6:
        return np.ones(len(pts), bool)
    lo, hi = v[np.argmin(ang)], v[np.argmax(ang)]
    p = pts - cam_xy
    return ((lo[0] * p[:, 1] - lo[1] * p[:, 0]) >= -1e-9) & ((p[:, 0] * hi[1] - p[:, 1] * hi[0]) >= -1e-9)


# --------------------------------------------------------------------------------------------------------------------------- the driven path
def ego_path(nusc, sample_token, T_e2g, back=25.0, ahead=55.0, frame="lane"):
    """the CAM_FRONT ego positions around the keyframe in `frame`, oldest first, the keyframe position (0, 0) in between -> {"pts", "back_m", "ahead_m"}; the metres are the driven
    ones (0 when the sequence starts or ends there)"""
    sample = nusc.get("sample", sample_token)
    sd0 = nusc.get("sample_data", sample["data"]["CAM_FRONT"])

    def xy(sd):
        return np.asarray(nusc.get("ego_pose", sd["ego_pose_token"])["translation"][:2], float)
    p0 = xy(sd0)

    def walk(key, limit):
        out, sd, dist, last = [], sd0, 0.0, p0
        while sd[key] and dist < limit:
            sd = nusc.get("sample_data", sd[key])
            p = xy(sd)
            dist += float(np.linalg.norm(p - last))
            last = p
            out.append(p)
        return out
    fwd, bwd = walk("next", ahead + 5.0), walk("prev", back + 5.0)
    G = np.array(bwd[::-1] + [p0] + fwd, float)
    P = LG.to_frame(G, T_e2g, frame)
    i0 = len(bwd)                                            # the index of the keyframe position
    keep = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[keep[-1]]) >= MIN_PATH_STEP or i == i0:
            keep.append(i)
    P = P[keep]
    k0 = keep.index(i0)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))] if len(P) > 1 else np.zeros(1)
    return dict(pts=P.astype(np.float32), back_m=float(min(arc[k0] - arc[0], back)), ahead_m=float(min(arc[-1] - arc[k0], ahead)))


def _to_polyline(P, Q, chunk=1500):
    """for the points P (n, 2) and the polyline Q (m, 2): distance to the nearest point of the polyline (n,), its direction (n, 2) and its arc position (n,)"""
    A, B = Q[:-1], Q[1:]
    AB = B - A
    L = np.linalg.norm(AB, axis=1)
    L2 = np.maximum(L * L, 1e-12)
    arc = np.r_[0.0, np.cumsum(L)]
    dist, tang, pos = np.empty(len(P)), np.empty((len(P), 2)), np.empty(len(P))
    for c0 in range(0, len(P), chunk):
        X = P[c0:c0 + chunk]
        t = np.clip(((X[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0.0, 1.0)
        C = A[None] + t[..., None] * AB[None]
        D = np.linalg.norm(X[:, None, :] - C, axis=2)
        j = D.argmin(1)
        r = np.arange(len(X))
        dist[c0:c0 + chunk] = D[r, j]
        tang[c0:c0 + chunk] = AB[j] / np.maximum(L[j], 1e-9)[:, None]
        pos[c0:c0 + chunk] = arc[j] + t[r, j] * L[j]
    return dist, tang, pos


def _tangents(pts):
    """unit direction of travel at every point of a piece (central differences)"""
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return np.zeros((len(pts), 2))
    d = np.gradient(pts, axis=0)
    return d / np.maximum(np.linalg.norm(d, axis=1), 1e-9)[:, None]


# --------------------------------------------------------------------------------------------------------------------------- tiers
def tier_pieces(g, path, p):
    """set `tier`, `tier_src` on every piece of the graph `g` (any frame) from the driven path `path` (dict of ego_path) and the parameters `p` (lane_gt params); -> info dict"""
    pieces = g["pieces"]
    for q in pieces:
        n = len(q["pts"])
        q["tier"], q["tier_src"] = np.full(n, 3, np.uint8), np.zeros(n, np.uint8)
    if not pieces:
        return dict(extension_m=0.0, back_extension_m=0.0)
    byid = {q["id"]: q for q in pieces}
    Q = np.asarray(path["pts"], float)
    real_end = float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum()) if len(Q) > 1 else 0.0      # arc length of the DRIVEN path; points beyond it (the straight reference below) are not driven
    if path["ahead_m"] < 15.0:                                # a standing car or the end of the sequence: the reference goes straight on along the last heading (the car's axis) for 20 m
        fwd = np.array([0.0, 1.0]) if g["frame"] == "lane" else np.array([1.0, 0.0])
        if len(Q) >= 2 and np.linalg.norm(Q[-1] - Q[max(0, len(Q) - 6)]) >= 1.0:
            fwd = (Q[-1] - Q[max(0, len(Q) - 6)]) / np.linalg.norm(Q[-1] - Q[max(0, len(Q) - 6)])
        Q = np.vstack([Q, Q[-1] + fwd * (0.5 * np.arange(1, 41))[:, None]])
    sl, o = [], 0
    for q in pieces:
        sl.append(slice(o, o + len(q["pts"])))
        o += len(q["pts"])
    P = np.concatenate([np.asarray(q["pts"], float) for q in pieces])
    T = np.concatenate([_tangents(q["pts"]) for q in pieces])
    d, tp, s = _to_polyline(P, Q)
    on = (d <= p["path_tol"]) & ((T * tp).sum(1) >= p["path_cos"])
    # the pieces on the path: enough metres of it, and the gap-free span between the first and the last point on it
    for q, a in zip(pieces, sl):
        o_ = on[a]
        if not o_.any():
            continue
        if o_.sum() * STEP >= min(6.0, 0.3 * max(q["length"], STEP)):
            idx = np.nonzero(o_)[0]
            span = slice(idx[0], idx[-1] + 1)
            q["tier"][span] = 1
            q["tier_src"][span] = np.where(s[a][span] > real_end + p["path_tol"], 2, 1)            # 1 = on the driven path, 2 = on the straight reference beyond its end
    # the map extension where the driven path is short (a stopped car, the end of the sequence)
    ext = 0.0
    if p["extend"] and path["ahead_m"] < p["min_ahead"]:
        want = p["tier_ahead"] - path["ahead_m"]
        best = None                                           # the tier-1 point with the largest arc position
        for q, a in zip(pieces, sl):
            m = q["tier"] == 1
            if m.any():
                k = np.nonzero(m)[0][np.argmax(s[a][m])]
                if best is None or s[a][k] > best[0]:
                    best = (float(s[a][k]), q, int(k))
        if best is not None:
            _, q, k = best
            seen = {q["id"]}
            while True:
                n = len(q["pts"])
                rest = slice(k + 1, n)
                grow = q["tier"][rest] != 1
                q["tier"][rest], q["tier_src"][rest] = 1, np.where(grow, 2, q["tier_src"][rest])
                ext += STEP * int(grow.sum())                                                      # only the metres that were not tier 1 before
                if ext >= want or q["end_open"]:
                    break
                tail = _tangents(q["pts"])[-1]
                cand = [byid[i] for i in q["succ"] if i in byid and i not in seen]
                if not cand:
                    break
                nxt = max(cand, key=lambda c: float(_tangents(c["pts"])[0] @ tail))
                seen.add(nxt["id"])
                q, k = nxt, -1
    # the same BEHIND the car where the car has not driven (the start of a log, a car that has just started): the own lane behind is tier 1 whatever the position in the log (critic pass 2026-11-07)
    back_ext = 0.0
    if p["extend"] and path["back_m"] < p["min_back"]:
        want_b = p["tier_back"] - path["back_m"]
        first = None                                          # the tier-1 point with the smallest arc position
        for q, a in zip(pieces, sl):
            m = q["tier"] == 1
            if m.any():
                k = np.nonzero(m)[0][np.argmin(s[a][m])]
                if first is None or s[a][k] < first[0]:
                    first = (float(s[a][k]), q, int(k))
        if first is not None:
            _, q, k = first
            seen = {q["id"]}
            while True:
                head = slice(0, k)
                grow = q["tier"][head] != 1
                q["tier"][head], q["tier_src"][head] = 1, np.where(grow, 2, q["tier_src"][head])
                back_ext += STEP * int(grow.sum())
                if back_ext >= want_b or q["start_open"]:
                    break
                head_t = _tangents(q["pts"])[0]
                cand = [byid[i] for i in q["pred"] if i in byid and i not in seen]
                if not cand:
                    break
                prv = max(cand, key=lambda c: float(_tangents(c["pts"])[-1] @ head_t))
                seen.add(prv["id"])
                q, k = prv, len(prv["pts"])
    # tier 2: the neighbours of the tier-1 pieces within neighbour_tol of a tier-1 point
    one = np.concatenate([np.asarray(q["pts"], float)[q["tier"] == 1] for q in pieces])
    if len(one):
        from scipy.spatial import cKDTree
        tree = cKDTree(one)
        for q in pieces:
            if not (q["tier"] == 1).any():
                continue
            for key, tier in (("left", 2), ("right", 2), ("opposite", int(p["opposite_tier"]))):
                for i in q.get(key, []):
                    nb = byid.get(i)
                    if nb is None:
                        continue
                    near = tree.query(np.asarray(nb["pts"], float))[0] <= p["neighbour_tol"]
                    sel = near & (nb["tier"] == 3)
                    nb["tier"][sel] = np.minimum(nb["tier"][sel], tier)
    return dict(extension_m=float(ext), back_extension_m=float(back_ext))


# --------------------------------------------------------------------------------------------------------------------------- visibility
def box_prisms(nusc, sample_token, T_e2g):
    """the observed vehicle / pedestrian / two-wheeler boxes of the keyframe as prisms (footprint (4, 2) in the EGO frame, z_lo, z_hi): the ones that hide the ground"""
    from pyquaternion import Quaternion
    out = []
    for tok in nusc.get("sample", sample_token)["anns"]:
        a = nusc.get("sample_annotation", tok)
        if a["num_lidar_pts"] + a["num_radar_pts"] <= 0:
            continue
        cat = nusc.get("category", nusc.get("instance", a["instance_token"])["category_token"])["name"]
        if category_group(cat) is None:
            continue
        w, l, h = a["size"]
        yaw = Quaternion(a["rotation"]).yaw_pitch_roll[0]
        c, s = math.cos(yaw), math.sin(yaw)
        corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]]) @ np.array([[c, s], [-s, c]]) + np.asarray(a["translation"][:2], float)
        out.append((LG.to_frame(corners, T_e2g, "ego"), 0.0, float(h)))
    return out


def _blocked(points, cam, prisms):
    """as coverage.blocked, with two differences for LANE points: (1) a box behind the camera hides for a rear camera too (coverage.blocked skips boxes with x < camera x: right for the
    front camera only); (2) the ground UNDER a box is hidden by it (coverage.blocked leaves a point inside a footprint alone, because an object is not hidden by itself: the road under a
    car is not seen by any camera)"""
    out = np.zeros(len(points), bool)
    for fp, z_lo, z_hi in prisms:
        idx = np.nonzero(_wedge(cam[:2], points[:, :2], fp) & ~out)[0]
        if not len(idx):
            continue
        t0, t1 = _clip(cam[:2], points[idx, :2], fp)
        cross = t0 < t1
        za, zb = cam[2] + t0 * (points[idx, 2] - cam[2]), cam[2] + t1 * (points[idx, 2] - cam[2])
        out[idx] |= cross & (np.maximum(za, zb) >= z_lo) & (np.minimum(za, zb) <= z_hi)
    return out


def visibility(pts_ego, prisms, cameras=RM.SIX):
    """(vis, covered), both bool (n,): the ground point (x forward, y left) is in the image of at least one of the `cameras` (default the six of the rig: Spurmitte; BEV fusion
    has the front camera only, lane_gt.RIGS) (covered), and a line to such a camera is free of the prisms (vis). covered & ~vis = hidden by a box"""
    pts_ego = np.asarray(pts_ego, float).reshape(-1, 2)
    vis, covered = np.zeros(len(pts_ego), bool), np.zeros(len(pts_ego), bool)
    if not len(pts_ego):
        return vis, covered
    P3 = np.c_[pts_ego, np.zeros(len(pts_ego))]
    for name in cameras:
        K, T = RM.camera_calibration(name)
        c = camera_cells(pts_ego[:, 0], pts_ego[:, 1], K, T, RM.IMAGE_SIZE)
        covered |= c
        cov = np.nonzero(c & ~vis)[0]
        if not len(cov):
            continue
        hidden = _blocked(P3[cov], T[:3, 3], prisms) if prisms else np.zeros(len(cov), bool)
        vis[cov[~hidden]] = True
    return vis, covered


# --------------------------------------------------------------------------------------------------------------------------- the whole step
def add_tiers(g, nusc, sample_token, T_e2g, p, cameras=RM.SIX):
    """add the labels to the graph `g` (made by lane_graph.crop; any frame) in place; `p` = the lane GT parameters (tier_back, tier_ahead, path_tol, path_cos, neighbour_tol,
    min_ahead, extend, opposite_tier); `cameras` = the cameras of the rig that `vis` / `occ` refer to (lane_gt.RIGS)"""
    path = ego_path(nusc, sample_token, T_e2g, p["tier_back"], p["tier_ahead"], g["frame"])
    info = tier_pieces(g, path, p)
    prisms = box_prisms(nusc, sample_token, T_e2g)
    to_ego = (lambda a: np.c_[a[:, 1], -a[:, 0]]) if g["frame"] == "lane" else (lambda a: a)
    for q in g["pieces"]:
        q["vis"], cov = visibility(to_ego(np.asarray(q["pts"], float)), prisms, cameras)
        q["occ"] = cov & ~q["vis"]                                                                          # in a camera's view but hidden by a box (not: outside every view)
    for e in g["events"]:
        best = (3, False, 3.0)
        for i in list(e["in"]) + list(e["out"]):
            q = next((x for x in g["pieces"] if x["id"] == i), None)
            if q is None or not len(q["pts"]):
                continue
            dd = np.linalg.norm(np.asarray(q["pts"], float) - np.asarray(e["pos"], float), axis=1)
            k = int(dd.argmin())
            if dd[k] <= 3.0 and (int(q["tier"][k]), not bool(q["vis"][k])) < (best[0], not best[1]):
                best = (int(q["tier"][k]), bool(q["vis"][k]), float(dd[k]))
        e["tier"], e["vis"] = best[0], best[1]
    tl = {t: float(sum(STEP * (q["tier"] == t).sum() for q in g["pieces"])) for t in (1, 2, 3)}
    tv = {t: float(sum(STEP * ((q["tier"] == t) & q["vis"]).sum() for q in g["pieces"])) for t in (1, 2, 3)}
    g["ego_path"] = path
    g["tier_info"] = dict(length_m=tl, visible_m=tv, hiding_boxes=len(prisms), extension_m=info["extension_m"], back_extension_m=info["back_extension_m"], driven_ahead_m=path["ahead_m"], driven_back_m=path["back_m"])
    return g


# --------------------------------------------------------------------------------------------------------------------------- self-test
def _piece(pid, pts, succ=(), left=(), right=(), opposite=(), end_open=False, pred=(), start_open=False):
    pts = np.asarray(pts, np.float32)
    return dict(id=pid, piece=pid, part=0, kind="lane", pts=pts, start_open=start_open, end_open=end_open, length=float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()),
                succ=list(succ), pred=list(pred), left=list(left), right=list(right), opposite=list(opposite))


def _line(x0, y0, x1, y1):
    n = int(round(math.hypot(x1 - x0, y1 - y0) / STEP)) + 1
    return np.c_[np.linspace(x0, x1, n), np.linspace(y0, y1, n)]


def selftest(verbose=True):
    """a toy lane in the lane frame (x right, y forward): the ego lane A (x = 0, two pieces), its right neighbour B, the opposite lane C, a far lane D, a crossing connector E, a
    branch F that is not taken; the driven path, the extension, the neighbour rule, the opposite rule and the hiding box are checked"""
    p = dict(tier_back=25.0, tier_ahead=55.0, path_tol=1.5, path_cos=0.7, neighbour_tol=10.0, min_ahead=40.0, min_back=20.0, extend=True, opposite_tier=3)
    A1, A2 = _piece("A1", _line(0, -40, 0, 10), succ=["A2"], right=["B"], opposite=["C"], start_open=True), _piece("A2", _line(0, 10, 0, 50), end_open=True, right=["B"], opposite=["C"], pred=["A1"])
    B = _piece("B", _line(3.5, -40, 3.5, 50), left=["A1", "A2"])
    C = _piece("C", _line(-3.5, 50, -3.5, -40), opposite=["A1"])
    D = _piece("D", _line(10.5, -40, 10.5, 50))
    E = _piece("E", _line(-20, 20, 20, 20))
    F = _piece("F", _line(0, 10, 25, 30))
    g = dict(frame="lane", half=50.0, pieces=[A1, A2, B, C, D, E, F], events=[dict(type="split", pos=np.array([0.0, 10.0], np.float32), **{"in": ["A1"], "out": ["A2", "F"]})])
    path = dict(pts=np.c_[np.zeros(61), np.linspace(-25, 35, 61)].astype(np.float32), back_m=25.0, ahead_m=35.0)           # the car drives straight 35 m ahead
    info = tier_pieces(g, path, p)
    t = {q["id"]: q["tier"] for q in g["pieces"]}
    assert (t["A1"][(A1["pts"][:, 1] >= -25) & (A1["pts"][:, 1] <= 10)] == 1).all(), "the driven part of A1 must be tier 1"
    assert (t["A2"] == 1).all() and (g["pieces"][1]["tier_src"][A2["pts"][:, 1] > 37.5] == 2).all(), "the extension on the same lane (src 2 beyond the driven 35 m plus the path tolerance)"
    assert 10.0 <= info["extension_m"] <= 16.0 and info["back_extension_m"] == 0.0, info              # 20 m wanted, the piece ends at 50 m: 13.5 m; the car drove 25 m back: no extension behind
    yb = B["pts"][:, 1]
    assert (t["B"][(yb >= -30) & (yb <= 50)] == 2).all() and (t["B"][yb < -36] == 3).all(), "the right neighbour is tier 2 within 10 m of the tier-1 points, tier 3 farther away"
    assert (t["C"] == 3).all() and (t["D"] == 3).all(), "the opposite lane and the far lane are tier 3"
    assert (t["E"] == 3).all(), "a perpendicular connector crossing the path is not on it"
    assert (t["F"][F["pts"][:, 0] > 8] == 3).all(), "the branch not taken is tier 3"
    p2 = dict(p, opposite_tier=2)
    tier_pieces(g, path, p2)
    yc = C["pts"][:, 1]
    assert (g["pieces"][3]["tier"][(yc >= -30) & (yc <= 50)] == 2).all(), "opposite_tier=2 makes the opposite lane tier 2 (within 10 m of the tier-1 points)"
    # a standing car (no driven path ahead): the lane under it is tier 1 and goes on along the map to 55 m
    for q in g["pieces"]:
        q["tier"], q["tier_src"] = np.full(len(q["pts"]), 3, np.uint8), np.zeros(len(q["pts"]), np.uint8)
    stand = dict(pts=np.zeros((1, 2), np.float32), back_m=0.0, ahead_m=0.0)
    info = tier_pieces(g, stand, p)
    ya = g["pieces"][0]["pts"][:, 1]
    assert (g["pieces"][0]["tier"] == 1).all() and (g["pieces"][1]["tier"] == 1).all() and (g["pieces"][2]["tier"][(yb >= -5) & (yb <= 50)] == 2).all(), "standing car: the lane ahead AND behind is tier 1"
    assert info["extension_m"] >= 25.0 and info["back_extension_m"] >= 25.0, info
    assert (g["pieces"][0]["tier_src"][(ya >= -1) & (ya <= 1)] == 1).all() and (g["pieces"][0]["tier_src"][(ya > 2) & (ya <= 10)] == 2).all() and (g["pieces"][0]["tier_src"][ya < -2] == 2).all(), \
        "the standing car's driven metres are only the ones at the car; the straight reference ahead and the map behind are src 2"
    # the start of a log (nothing driven behind): the lane behind is tier 1 too, across a piece boundary (the car is on A2, the lane behind is A1)
    for q in g["pieces"]:
        q["tier"], q["tier_src"] = np.full(len(q["pts"]), 3, np.uint8), np.zeros(len(q["pts"]), np.uint8)
    start = dict(pts=np.c_[np.zeros(71), np.linspace(12, 47, 71)].astype(np.float32), back_m=0.0, ahead_m=35.0)
    info = tier_pieces(g, start, p)
    assert (g["pieces"][0]["tier"] == 1).all() and (g["pieces"][1]["tier"][A2["pts"][:, 1] >= 10] == 1).all() and info["back_extension_m"] >= 25.0, info
    assert (g["pieces"][3]["tier"] == 3).all(), "the opposite lane stays tier 3"
    # visibility: a car 8 m ahead hides the ground behind it from the front camera; the side cameras do not see it
    car = (np.array([[10.0, 1.0], [10.0, -1.0], [6.0, -1.0], [6.0, 1.0]]), 0.0, 1.6)                      # ego frame: x forward 6 .. 10, y +-1
    pts = np.array([[20.0, 0.0], [20.0, 6.0], [-15.0, 0.0], [8.0, 0.0], [30.0, 0.0]])                      # ego frame points: behind the car, beside it, behind the ego, inside the car, far behind it
    free, hid = visibility(pts, [])[0], visibility(pts, [car])[0]
    assert free[0] and free[1] and free[2], free
    assert not hid[0] and not hid[4] and hid[1] and hid[2], hid
    assert free[3] and not hid[3], "the ground under the box is hidden by it"
    # a car behind the ego hides the ground behind it from the rear camera
    rear = (np.array([[-5.0, 1.0], [-5.0, -1.0], [-9.0, -1.0], [-9.0, 1.0]]), 0.0, 1.6)
    q = np.array([[-20.0, 0.0]])
    assert visibility(q, [])[0].all() and not visibility(q, [rear])[0].all(), "rear boxes hide for the rear camera (coverage.blocked would skip them)"
    assert visibility(q, [rear])[1].all(), "covered stays True under a box (occ = covered and not visible)"
    if verbose:
        print("lane_tiers selftest: driven path tier 1, extension, neighbours tier 2, opposite / far / crossing / branch tier 3, visibility with front and rear boxes: ok")
    return True


if __name__ == "__main__":
    selftest()
