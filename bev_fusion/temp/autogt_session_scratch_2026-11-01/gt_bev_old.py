"""Ground truth BEV maps per nuScenes keyframe, and training samples (previous, current, next keyframe).

Plan section 9.4 and 8.2 (Mirko, 2026-10-25): one GT per keyframe (exact annotations, no interpolation), in the ego
frame at that keyframe's front-camera time; a training sample is keyframe k with the GT of k-1, k, k+1 (about -0.5,
0, +0.5 s) and the ego steps between them (speed v and yaw rate omega per 12 Hz camera frame, for the coordinate chain
of training/splat.py), so the model predicts from k to the previous and the next keyframe with the same splat block.

Layers on a grid (lift_pd.Grid: x forward 0..60 m, y left -26..26 m, 0.5 m or 0.25 m):
- areas: drivable_area, ped_crossing, walkway, stop_line, carpark_area - coverage fraction (4 x 4 supersampling)
- lines: lane_divider, road_divider - 1-cell mask and a truncated distance field (metres, up to 2 m)
- centre-lines: lane and lane_connector arcs - mask and direction (cos, sin of the heading in the ego frame)
- objects: vehicle, pedestrian, two_wheeler - coverage fraction; velocity (vx, vy, ego axes); instance (index into
  meta["boxes"], -1 = none); only boxes with at least one lidar or radar point - boxes without any (nuScenes keeps
  them, its detection evaluation drops them) go into the ignore mask: neither object nor free space
- masks: camera (ground cells that project into the CAM_FRONT image), occluded (behind a box seen from the camera),
  radar (inside one of the five radars' field of view - an approximate fan, see RADAR_FOV), undefined (neither),
  ignore (footprints of boxes without sensor points)
- stationary rule (Mirko, 2026-10-26): static GT does not move - see NuScenesGT.__init__
Conventions: arrays [ix, iy] with x forward, y left (as the rest of training/); sub-pixel centres at
x_min + (i + 0.5) s. Rasterised with OpenCV on the supersampled grid in our own index convention (the map API's
get_map_mask has its own orientation; we only use its polygons and lines).
Run the self-test (nuScenes mini): python training/gt_bev.py
"""
import math
import os

import cv2
import numpy as np
from pyquaternion import Quaternion

try:
    from .lift_pd import Grid
except ImportError:                                   # run as a script
    from lift_pd import Grid

AREAS = ("drivable_area", "ped_crossing", "walkway", "stop_line", "carpark_area")
LINES = ("lane_divider", "road_divider")
OBJECTS = ("vehicle", "pedestrian", "two_wheeler")
DIST_MAX = 2.0
RADAR_FOV = (math.radians(60.0), 70.0)               # half angle, range: approximate fan per radar (memory, not checked)
SHIFT = 4                                             # OpenCV fixed-point sub-pixel bits


def category_group(name):
    if name.startswith(("vehicle.bicycle", "vehicle.motorcycle")):
        return "two_wheeler"
    if name.startswith("vehicle."):
        return "vehicle"
    if name.startswith("human.pedestrian"):
        return "pedestrian"
    return None


def pose_matrix(rec):
    T = np.eye(4)
    T[:3, :3] = Quaternion(rec["rotation"]).rotation_matrix
    T[:3, 3] = rec["translation"]
    return T


# ------------------------------------------------------------------------------------------------ rasterising
class Raster:
    """Supersampled canvas of a grid; draws ego-frame (x, y) geometry in our [ix, iy] convention."""

    def __init__(self, grid, ss=4):
        self.g, self.ss = grid, ss
        self.s = grid.res / ss
        self.shape = (grid.nx * ss, grid.ny * ss)            # rows = x, cols = y

    def pts(self, xy):
        rows = (xy[:, 0] - self.g.x_min) / self.s - 0.5
        cols = (xy[:, 1] - self.g.y_min) / self.s - 0.5
        return np.round(np.stack([cols, rows], 1) * (1 << SHIFT)).astype(np.int32)

    def canvas(self, dtype=np.uint8):
        return np.zeros(self.shape, dtype)

    def fill(self, img, polygons, value=1):
        """polygons: list of (exterior (N, 2), [holes (M, 2), ...]) in the ego frame. A sub-pixel is set if its CENTRE
        lies inside (matplotlib Path test, only in the polygon's bounding box): exact area, no boundary bias (OpenCV's
        fillPoly counts the boundary pixels on both sides: a 2 m square came out as 4.52 m^2 at ss = 4)."""
        from matplotlib.path import Path
        g, s = self.g, self.s
        for ext, holes in polygons:
            lo = np.floor((ext.min(0) - [g.x_min, g.y_min]) / s - 0.5).astype(int)
            hi = np.ceil((ext.max(0) - [g.x_min, g.y_min]) / s - 0.5).astype(int)
            r0, c0 = max(lo[0], 0), max(lo[1], 0)
            r1, c1 = min(hi[0] + 1, self.shape[0]), min(hi[1] + 1, self.shape[1])
            if r0 >= r1 or c0 >= c1:
                continue
            rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
            pts = np.stack([g.x_min + (rr.ravel() + 0.5) * s, g.y_min + (cc.ravel() + 0.5) * s], 1)
            m = Path(ext).contains_points(pts)
            for h in holes:
                m &= ~Path(h).contains_points(pts)
            sub = img[r0:r1, c0:c1]
            sub[m.reshape(rr.shape)] = value
        return img

    def lines(self, img, polylines, value=1, thickness=1):
        for pl in polylines:
            if len(pl) >= 2:
                cv2.polylines(img, [self.pts(pl)], False, value, thickness, cv2.LINE_8, SHIFT)
        return img

    def coverage(self, img):
        g, ss = self.g, self.ss
        return img.astype(np.float32).reshape(g.nx, ss, g.ny, ss).mean((1, 3))

    def at_centres(self, img):
        """value at the cell centres (mean of the central 2 x 2 sub-pixels for an even ss)."""
        g, ss = self.g, self.ss
        a = img.reshape(g.nx, ss, g.ny, ss)
        c = ss // 2
        return a[:, c - 1:c + 1, :, c - 1:c + 1].mean((1, 3)) if ss % 2 == 0 else a[:, c, :, c]


def cell_centres(grid):
    xs = grid.x_min + (np.arange(grid.nx) + 0.5) * grid.res
    ys = grid.y_min + (np.arange(grid.ny) + 0.5) * grid.res
    return np.meshgrid(xs, ys, indexing="ij")


# ------------------------------------------------------------------------------------------------ nuScenes access
class NuScenesGT:
    """GT per keyframe and training samples from a nuScenes copy (the WSL views ~/data/nuscenes, ~/data/nuscenes_full)."""

    def __init__(self, dataroot, version, nusc=None, stationary_fix=True, map_offsets=None):
        """Rule (Mirko, 2026-10-26): stationary GT is temporally stable - it does not move.
        - map layers are drawn from fixed world geometry in every frame; a correction may only be given per map element
          in WORLD coordinates (map_offsets: element token -> (dx, dy) metres, applied identically in every frame),
          never per frame
        - stationary objects (speed < 0.5 m/s over the whole track and < 1 m from their median position) get ONE world
          pose for all their keyframes (median position, mean yaw, median size) and velocity 0 (stationary_fix)"""
        from nuscenes.nuscenes import NuScenes
        self.nusc = nusc or NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.dataroot = dataroot
        self._maps = {}
        self.stationary_fix = stationary_fix
        self.map_offsets = map_offsets or {}
        self._stationary = {}

    def signature(self):
        """10 hex chars identifying this generator (source of this file + stationary_fix + map_offsets): part of every
        cache key of derived GT, so a changed generator can never be mixed with GT cached by an older one."""
        import hashlib
        h = hashlib.sha1(open(__file__, "rb").read())
        h.update(repr((self.stationary_fix, sorted((k, tuple(v)) for k, v in self.map_offsets.items()))).encode())
        return h.hexdigest()[:10]

    def stationary_instances(self, scene_token, v_max=0.5, d_max=1.0):
        """instance token -> fixed world pose (translation (3,), yaw, size (3,)) for the scene's stationary instances."""
        if scene_token in self._stationary:
            return self._stationary[scene_token]
        nusc = self.nusc
        tracks = {}
        tok = nusc.get("scene", scene_token)["first_sample_token"]
        while tok:
            s = nusc.get("sample", tok)
            for atok in s["anns"]:
                a = nusc.get("sample_annotation", atok)
                v = nusc.box_velocity(atok)
                tracks.setdefault(a["instance_token"], []).append(
                    (np.array(a["translation"]), Quaternion(a["rotation"]).yaw_pitch_roll[0], np.array(a["size"]),
                     np.hypot(v[0], v[1]) if np.isfinite(v[:2]).all() else np.nan))
            tok = s["next"]
        fixed = {}
        for inst, tr in tracks.items():
            P = np.stack([t[0] for t in tr])
            sp = np.array([t[3] for t in tr])
            med = np.median(P, 0)
            if np.nanmax(np.r_[sp, 0.0]) < v_max and np.linalg.norm(P[:, :2] - med[:2], axis=1).max() < d_max:
                yaws = np.array([t[1] for t in tr])
                fixed[inst] = (med, math.atan2(np.sin(yaws).mean(), np.cos(yaws).mean()),
                               np.median(np.stack([t[2] for t in tr]), 0))
        self._stationary[scene_token] = fixed
        return fixed

    def map_for(self, scene_token):
        from nuscenes.map_expansion.map_api import NuScenesMap
        loc = self.nusc.get("log", self.nusc.get("scene", scene_token)["log_token"])["location"]
        if loc not in self._maps:
            self._maps[loc] = NuScenesMap(dataroot=self.dataroot, map_name=loc)
        return self._maps[loc]

    def cam(self, sample_token, channel="CAM_FRONT"):
        return self.nusc.get("sample_data", self.nusc.get("sample", sample_token)["data"][channel])

    def ego_pose(self, sd):
        return pose_matrix(self.nusc.get("ego_pose", sd["ego_pose_token"]))

    # ---- one keyframe
    def keyframe_gt(self, sample_token, grid, ss=4):
        nusc = self.nusc
        sample = nusc.get("sample", sample_token)
        sd = self.cam(sample_token)
        T_e2g = self.ego_pose(sd)
        T_g2e = np.linalg.inv(T_e2g)
        R = Raster(grid, ss)

        def to_ego(xy):
            p = np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))] @ T_g2e.T
            return p[:, :2]

        corners = np.array([[grid.x_min, grid.y_min], [grid.x_min, grid.y_max], [grid.x_max, grid.y_min],
                            [grid.x_max, grid.y_max]])
        cg = (np.c_[corners, np.zeros(4), np.ones(4)] @ T_e2g.T)[:, :2]
        box = (cg[:, 0].min() - 5, cg[:, 1].min() - 5, cg[:, 0].max() + 5, cg[:, 1].max() + 5)
        nmap = self.map_for(sample["scene_token"])
        recs = nmap.get_records_in_patch(box, list(AREAS) + list(LINES) + ["lane", "lane_connector"], mode="intersect")
        out = {}
        for layer in AREAS:
            img = R.canvas()
            polys = []
            for tok in recs[layer]:
                rec = nmap.get(layer, tok)
                # a token of None is real: in singapore-hollandvillage / -queenstown the road network of the drivable
                # area is the one polygon whose token is null (found 2026-10-26 - it had been skipped: no road in GT)
                for ptok in (rec["polygon_tokens"] if "polygon_tokens" in rec else [rec["polygon_token"]]):
                    poly = nmap.extract_polygon(ptok)
                    if poly.is_empty:
                        continue
                    polys.append((to_ego(np.asarray(poly.exterior.coords)[:, :2]),
                                  [to_ego(np.asarray(h.coords)[:, :2]) for h in poly.interiors]))
            out[layer] = R.coverage(R.fill(img, polys))
        vectors = {}
        for layer in LINES:
            img = R.canvas()
            pls = []
            for tok in recs[layer]:
                line = nmap.extract_line(nmap.get(layer, tok)["line_token"])
                if not line.is_empty:
                    g = np.asarray(line.coords)[:, :2] + np.asarray(self.map_offsets.get(tok, (0.0, 0.0)))  # world-fixed
                    pls.append(to_ego(g))
            vectors[layer] = pls                                  # exact geometry, ego frame (no grid)
            R.lines(img, pls)
            dist = cv2.distanceTransform((img == 0).astype(np.uint8), cv2.DIST_L2, 5) * R.s
            d = np.minimum(R.at_centres(dist), DIST_MAX).astype(np.float32)
            out[layer + "_dist"] = d
            out[layer] = (d <= grid.res / 2).astype(np.float32)
        lane_toks = recs["lane"] + recs["lane_connector"]
        arcs = nmap.discretize_lanes(lane_toks, 0.5) if lane_toks else {}
        cmask, ccos, csin = R.canvas(), R.canvas(np.float32), R.canvas(np.float32)
        yaw_e = math.atan2(T_e2g[1, 0], T_e2g[0, 0])
        vectors["centreline"] = []
        for poses in arcs.values():
            P = np.asarray(poses)
            if len(P) < 2:
                continue
            xy = to_ego(P[:, :2])
            vectors["centreline"].append(xy)
            hd = P[:, 2] - yaw_e
            R.lines(cmask, [xy])
            for i in range(len(xy) - 1):                        # direction per segment
                seg = xy[i:i + 2]
                cv2.polylines(ccos, [R.pts(seg)], False, float(math.cos(hd[i])), 1, cv2.LINE_8, SHIFT)
                cv2.polylines(csin, [R.pts(seg)], False, float(math.sin(hd[i])), 1, cv2.LINE_8, SHIFT)
        cov = R.coverage(cmask)
        out["centreline"] = (cov > 0).astype(np.float32)
        nz = np.maximum(R.coverage(cmask.astype(np.float32)), 1e-6)
        out["centreline_dir"] = np.stack([R.coverage(ccos) / nz, R.coverage(csin) / nz]) * out["centreline"]

        # objects (exact annotations of this keyframe), velocity in the ego axes
        Rg2e = T_g2e[:3, :3]
        objs = {k: R.canvas() for k in OBJECTS}
        inst = np.full(R.shape, -1, np.int32)
        ignore = R.canvas()
        vel = np.zeros((2,) + R.shape, np.float32)
        boxes = []
        fixed = self.stationary_instances(sample["scene_token"]) if self.stationary_fix else {}
        for i, atok in enumerate(sample["anns"]):
            a = nusc.get("sample_annotation", atok)
            grp = category_group(a["category_name"])
            if grp is None:
                continue
            stationary = a["instance_token"] in fixed
            if stationary:                                       # one world pose for the whole track (rule above)
                trans, yaw, size = fixed[a["instance_token"]]
            else:
                trans, yaw, size = np.array(a["translation"]), Quaternion(a["rotation"]).yaw_pitch_roll[0], a["size"]
            w, l = size[0], size[1]
            c, s_ = math.cos(yaw), math.sin(yaw)
            loc = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
            cg_ = loc @ np.array([[c, s_], [-s_, c]]) + np.asarray(trans)[:2]
            ce = to_ego(cg_)
            v = np.zeros(3) if stationary else np.nan_to_num(nusc.box_velocity(atok))
            ve = Rg2e @ v
            one = R.fill(R.canvas(), [(ce, [])])
            observed = a["num_lidar_pts"] + a["num_radar_pts"] > 0
            if observed:
                objs[grp] |= one
                inst[one > 0] = len(boxes)                       # index into meta["boxes"]
                vel[0][one > 0], vel[1][one > 0] = ve[0], ve[1]
            else:                                                # no sensor point: neither object nor free -> ignore
                ignore |= one
            boxes.append({"group": grp, "corners": ce, "center": (Rg2e @ (np.asarray(trans) - T_e2g[:3, 3])),
                          "size": list(size), "yaw": yaw - yaw_e, "vel": ve[:2], "visibility": a["visibility_token"],
                          "token": atok, "instance": a["instance_token"], "stationary": stationary,
                          "observed": observed, "world": np.asarray(trans)})
        for k in OBJECTS:
            out[k] = R.coverage(objs[k])
        out["ignore"] = R.coverage(ignore)
        out["velocity"] = np.stack([R.at_centres(vel[0]), R.at_centres(vel[1])])
        out["instance"] = inst.reshape(grid.nx, ss, grid.ny, ss)[:, ss // 2, :, ss // 2]

        # masks: camera coverage (ground points in the image), occlusion behind boxes, radar fans
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        K = np.array(cs["camera_intrinsic"])
        T_c2e = pose_matrix(cs)
        gx, gy = cell_centres(grid)
        P = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size), np.ones(gx.size)], 1) @ np.linalg.inv(T_c2e).T
        z = P[:, 2]
        u = K[0, 0] * P[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
        vv = K[1, 1] * P[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
        cam = (z > 0.5) & (u >= 0) & (u < sd["width"]) & (vv >= 0) & (vv < sd["height"])
        out["camera"] = cam.reshape(gx.shape).astype(np.float32)
        out["occluded"] = occlusion(gx, gy, T_c2e[:2, 3], [b["corners"] for b in boxes if b["observed"]]).astype(np.float32)
        out["radar"] = radar_coverage(self, sample, gx, gy).astype(np.float32)
        out["undefined"] = ((out["camera"] == 0) & (out["radar"] == 0)).astype(np.float32)
        out["meta"] = {"sample": sample_token, "scene": sample["scene_token"], "timestamp": sd["timestamp"],
                       "T_e2g": T_e2g, "boxes": boxes, "res": grid.res, "cam_file": sd["filename"], "K": K,
                       "T_c2e": T_c2e, "vectors": vectors}
        return out

    # ---- training sample: previous, current, next keyframe and the ego steps between them
    def camera_track(self, sample_token):
        """all CAM_FRONT sample_data of the scene (keyframes and 12 Hz sweeps), in time order."""
        sd = self.cam(sample_token)
        while sd["prev"]:
            sd = self.nusc.get("sample_data", sd["prev"])
        track = [sd]
        while sd["next"]:
            sd = self.nusc.get("sample_data", sd["next"])
            track.append(sd)
        return track

    def ego_steps(self, t0, t1, track):
        """(dt, v, omega) per camera frame from time t0 to t1 (microseconds), either direction, from the poses."""
        times = [s["timestamp"] for s in track]
        i0, i1 = times.index(t0), times.index(t1)
        step = 1 if i1 > i0 else -1
        out = []
        for i in range(i0, i1, step):
            a, b = track[i], track[i + step]
            Ta, Tb = self.ego_pose(a), self.ego_pose(b)
            dt = (b["timestamp"] - a["timestamp"]) * 1e-6
            rel = np.linalg.inv(Ta) @ Tb
            out.append((dt, rel[0, 3] / dt, math.atan2(rel[1, 0], rel[0, 0]) / dt))
        return out

    def training_sample(self, sample_token, grid, ss=4):
        s = self.nusc.get("sample", sample_token)
        if not s["prev"] or not s["next"]:
            return None
        track = self.camera_track(sample_token)
        out = {"cur": self.keyframe_gt(sample_token, grid, ss)}
        t0 = out["cur"]["meta"]["timestamp"]
        for key, tok in (("prev", s["prev"]), ("next", s["next"])):
            out[key] = self.keyframe_gt(tok, grid, ss)
            t1 = out[key]["meta"]["timestamp"]
            out[key]["meta"]["dt"] = (t1 - t0) * 1e-6
            out[key]["meta"]["steps"] = self.ego_steps(t0, t1, track)
        return out


def occlusion(gx, gy, cam_xy, footprints):
    """cells whose segment from the camera crosses a box footprint they are not inside (ego frame, 2D)."""
    P = np.stack([gx.ravel(), gy.ravel()], 1)
    C = np.asarray(cam_xy, float)
    occ = np.zeros(len(P), bool)
    d = P - C
    for fp in footprints:
        if fp[:, 0].max() < C[0]:
            continue
        inside = _inside(P, fp)
        hit = np.zeros(len(P), bool)
        for k in range(4):
            a, b = fp[k], fp[(k + 1) % 4]
            e = b - a
            den = d[:, 0] * e[1] - d[:, 1] * e[0]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((a[0] - C[0]) * e[1] - (a[1] - C[1]) * e[0]) / den
                u = ((a[0] - C[0]) * d[:, 1] - (a[1] - C[1]) * d[:, 0]) / den
            hit |= (np.abs(den) > 1e-9) & (t > 0) & (t < 1) & (u >= 0) & (u <= 1)
        occ |= hit & ~inside
    return occ.reshape(gx.shape)


def _inside(P, poly):
    """points P (N, 2) inside the convex quadrilateral poly (4, 2), either orientation."""
    signs = []
    for k in range(4):
        a, b = poly[k], poly[(k + 1) % 4]
        signs.append((b[0] - a[0]) * (P[:, 1] - a[1]) - (b[1] - a[1]) * (P[:, 0] - a[0]))
    S = np.stack(signs, 1)
    return (S >= 0).all(1) | (S <= 0).all(1)


def radar_coverage(gt, sample, gx, gy):
    half, rmax = RADAR_FOV
    cov = np.zeros(gx.shape, bool)
    for ch, tok in sample["data"].items():
        if not ch.startswith("RADAR"):
            continue
        cs = gt.nusc.get("calibrated_sensor", gt.nusc.get("sample_data", tok)["calibrated_sensor_token"])
        T = pose_matrix(cs)
        yaw = math.atan2(T[1, 0], T[0, 0])
        dx, dy = gx - T[0, 3], gy - T[1, 3]
        ang = np.arctan2(dy, dx) - yaw
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        cov |= (np.abs(ang) <= half) & (np.hypot(dx, dy) <= rmax)
    return cov


# ----------------------------------------------------------------------------------------------- self-test
def _selftest():
    import torch
    import torch.nn.functional as F
    # 1 rasteriser, known answer: a 2 m x 2 m square at x 10..12, y -1..1 covers 4 m^2; cells inside are full
    g = Grid()
    R = Raster(g, 4)
    sq = np.array([[10.0, -1.0], [12.0, -1.0], [12.0, 1.0], [10.0, 1.0]])
    cov = R.coverage(R.fill(R.canvas(), [(sq, [])]))
    assert abs(cov.sum() * g.res ** 2 - 4.0) < 0.15, cov.sum() * g.res ** 2
    assert cov[20:24, 50:54].min() > 0.99 and cov[:19].max() == 0
    # line distance: a line along x at y = 0.25 m -> distance at the cell centre y = 0.25 is ~0, at y = 1.25 ~1 m
    img = R.lines(R.canvas(), [np.array([[5.0, 0.25], [50.0, 0.25]])])
    dist = np.minimum(R.at_centres(cv2.distanceTransform((img == 0).astype(np.uint8), cv2.DIST_L2, 5) * R.s), 2)
    assert dist[40, 52] < 0.1 and abs(dist[40, 54] - 1.0) < 0.15, (dist[40, 52], dist[40, 54])
    # occlusion: a box at x 20..24, y -1..1 hides the cells behind it from a camera at the origin, not those beside
    gx, gy = cell_centres(g)
    occ = occlusion(gx, gy, (0.0, 0.0), [np.array([[20.0, -1], [24, -1], [24, 1], [20, 1]])])
    assert occ[60, 52] and not occ[60, 70] and not occ[44, 52]

    # real data: nuScenes mini
    root = os.path.expanduser("~/data/nuscenes")
    gt = NuScenesGT(root, "v1.0-mini")
    scene = gt.nusc.scene[0]
    s1 = gt.nusc.get("sample", scene["first_sample_token"])["next"]
    ts = gt.training_sample(s1, g)
    cur, nxt = ts["cur"], ts["next"]
    assert cur["drivable_area"].mean() > 0.05 and cur["camera"].mean() > 0.2
    # 1b every map location has a road under the ego car (singapore-hollandvillage and -queenstown store their road
    #    network under a polygon token of None; skipping it left no drivable area at all)
    seen = {}
    for sc in gt.nusc.scene:
        loc = gt.nusc.get("log", sc["log_token"])["location"]
        if loc not in seen:
            k0 = gt.keyframe_gt(sc["first_sample_token"], g)
            seen[loc] = float(k0["drivable_area"][0:6, 50:54].mean())
            assert seen[loc] > 0.5, (loc, seen[loc])
    # 2 pooling: the 0.25 m GT pooled 2 x 2 equals the 0.5 m GT (areas and objects)
    fine = gt.keyframe_gt(s1, Grid(res=0.25))
    for layer in ("drivable_area", "walkway", "vehicle"):
        pooled = fine[layer].reshape(g.nx, 2, g.ny, 2).mean((1, 3))
        err = np.abs(pooled - cur[layer]).mean()
        assert err < 0.01, (layer, err)
    # 3 static layers of the next keyframe = those of the current one moved by the exact ego motion (pull)
    T = np.linalg.inv(cur["meta"]["T_e2g"]) @ nxt["meta"]["T_e2g"]          # next ego -> current ego
    P = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size), np.ones(gx.size)], 1) @ T.T
    gxn = (P[:, 1] - g.y_min) / (g.y_max - g.y_min) * 2 - 1
    gyn = (P[:, 0] - g.x_min) / (g.x_max - g.x_min) * 2 - 1
    grid_s = torch.tensor(np.stack([gxn, gyn], 1), dtype=torch.float32).view(1, g.nx, g.ny, 2)
    src = torch.tensor(cur["drivable_area"])[None, None]
    moved = F.grid_sample(src, grid_s, align_corners=False)[0, 0].numpy()
    inside = (np.abs(gxn) < 0.98) & (np.abs(gyn) < 0.98)
    a, b = moved.reshape(-1)[inside] > 0.5, nxt["drivable_area"].reshape(-1)[inside] > 0.5
    iou = (a & b).sum() / max(1, (a | b).sum())
    assert iou > 0.95, iou
    # 4 velocities: rotation into the ego axes keeps the speed (moving boxes; stationary ones are 0 by the rule)
    for bx in cur["meta"]["boxes"][:20]:
        vg = np.nan_to_num(gt.nusc.box_velocity(bx["token"]))[:2]
        if bx["stationary"]:
            assert np.hypot(*bx["vel"]) == 0
        else:
            assert abs(np.hypot(*bx["vel"]) - np.hypot(*vg)) < 0.05 * max(1, np.hypot(*vg)) + 0.1
    # 4a instance layer: every cell's index points at an observed box of the group whose layer covers the cell
    bxs = cur["meta"]["boxes"]
    ii = cur["instance"]
    assert (ii >= 0).any()
    for i in np.unique(ii[ii >= 0]):
        assert bxs[i]["observed"] and cur[bxs[i]["group"]][ii == i].min() > 0, i
    # 4b rule: stationary GT does not move - a stationary instance has the same world pose in k and k+1, while its raw
    #    annotations jitter (measured over all stationary instances of the scene)
    fixed = gt.stationary_instances(scene["token"])
    both = [(b, c) for b in cur["meta"]["boxes"] for c in nxt["meta"]["boxes"]
            if b["instance"] == c["instance"] and b["stationary"]]
    assert both and all(np.allclose(b["world"], c["world"]) for b, c in both), "stationary boxes must not move"
    jit = []
    tok = scene["first_sample_token"]
    while tok:
        s = gt.nusc.get("sample", tok)
        for atok in s["anns"]:
            a = gt.nusc.get("sample_annotation", atok)
            if a["instance_token"] in fixed:
                jit.append(np.linalg.norm(np.array(a["translation"][:2]) - fixed[a["instance_token"]][0][:2]))
        tok = s["next"]
    jitter = float(np.median(jit))
    # 5 ego steps: dt add up to the keyframe gap, both directions
    assert abs(sum(s[0] for s in ts["next"]["meta"]["steps"]) - ts["next"]["meta"]["dt"]) < 1e-6
    assert abs(sum(s[0] for s in ts["prev"]["meta"]["steps"]) - ts["prev"]["meta"]["dt"]) < 1e-6
    print(f"gt_bev selftest ok: square 4 m^2, line distance, occlusion, real keyframe (drivable "
          f"{cur['drivable_area'].mean():.2f}, camera {cur['camera'].mean():.2f}), 0.25 m pooled = 0.5 m, next keyframe "
          f"= current moved by ego motion (IoU {iou:.3f}), velocity norms, ego steps "
          f"({len(ts['prev']['meta']['steps'])} back, {len(ts['next']['meta']['steps'])} forward); stationary rule: "
          f"{len(fixed)} stationary instances in the scene fixed in the world (raw annotation jitter median "
          f"{jitter * 100:.1f} cm, now 0), {len(both)} seen in k and k+1 at the same world pose")


if __name__ == "__main__":
    _selftest()
