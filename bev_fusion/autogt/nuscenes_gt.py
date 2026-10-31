"""nuScenes GT generator of the autoGT suite: per keyframe the static BEV layers, the objects, the masks and the velocity /
flow field - and training samples (previous, current, next keyframe with the ego steps between them).

Layers on a grid (`GridSpec`: x forward 0..60 m, y left -26..26 m, 0.5 m or 0.25 m), arrays [ix, iy], ego frame at the
keyframe's front-camera time:
- areas: drivable_area, ped_crossing, walkway, stop_line, carpark_area - coverage fraction (4 x 4 supersampling)
- lines: lane_divider, road_divider - 1-cell mask (`<name>`) and a truncated distance field (`<name>_dist`, metres, up to 2 m)
- centre-lines: lane and lane_connector arcs - mask `centreline` and direction `centreline_dir` (cos, sin of the heading)
- objects: vehicle, pedestrian, two_wheeler - coverage fraction; `instance` (index into meta["boxes"], -1 = none); only boxes
  with at least one lidar or radar point; the others go into `ignore` (neither object nor free space)
- velocity / flow (autogt/flow.py, all in m/s, never per frame): `velocity` (2, nx, ny) world velocity of the covering box in
  ego axes (radar-corrected, 0 elsewhere), `flow_ego` (2, nx, ny) apparent velocity of a static world point caused by the ego
  motion at the camera time, `flow` = flow_ego + velocity, `flow_weight` (nx, ny) 1 known / 0 unknown (ignored boxes, unknown
  velocity), `flow_sigma` (nx, ny) std of the speed estimate. Per box in meta: vel, vel_source, vel_flag, vel_sigma, vel_known
- masks: camera (ground cells that project into the CAM_FRONT image), occluded (behind a box seen from the camera), radar
  (inside the fan of one of the five radars: +-60 deg, measured: 99.9 % of all detections of nuScenes mini lie inside it),
  undefined (neither), ignore (see objects)
- meta: sample, scene, timestamp, T_e2g, boxes, res, cam_file, K, T_c2e, vectors (exact dividers / centre-lines as ego polylines)
Rules (Mirko, 2026-10-26): stationary GT is temporally stable. Map layers come from the fixed world geometry in every frame;
a correction is allowed only per map element in WORLD coordinates (`map_offsets`: element token -> (dx, dy) m, the same in
every frame), never per frame. Stationary objects (autogt.flow.is_stationary: all speeds < 0.5 m/s, all positions < 1 m from
the median) get ONE world pose for all their keyframes and velocity 0.
Boxes are at the CAMERA time (decided by Mirko 2026-10-31, the way Waymo's camera_synced_box works): the keyframe is annotated
at the lidar time, the front camera took its image about 36 ms earlier, so a box of known velocity is moved by v * (t_camera -
t_annotation) (constant velocity, size, height and yaw unchanged). Stationary boxes (one world pose, v = 0) and boxes of unknown
velocity do not move. Per box in meta: `center`, `corners`, `world` at camera time; `world_ann` the annotation-time world pose,
`shift` (ego axes, m) and `shift_s` (s) what was applied. `boxes_at="annotation"` keeps the annotation positions.
Conventions: sub-pixel centres at x_min + (i + 0.5) s; rasterised with OpenCV / an exact point test on the supersampled grid.
"""
import hashlib
import math
import os
import collections

import cv2
import numpy as np
from pyquaternion import Quaternion

from . import flow as flowmod
from .layers import AREAS, DIST_MAX, LINES, OBJECTS, SHIFT, Raster, cell_centres, occlusion
from .nuscenes_adapter import NuScenesScene, category_group, pose_matrix

RADAR_FOV = (math.radians(60.0), 70.0)               # half angle, range (m): measured, see the module docstring
SIGNATURE_FILES = ("grid.py", "layers.py", "flow.py", "nuscenes_adapter.py", "nuscenes_gt.py")


class NuScenesGT:
    """GT per keyframe and training samples from a nuScenes copy (the WSL views ~/data/nuscenes, ~/data/nuscenes_full)."""

    def __init__(self, dataroot, version, nusc=None, stationary_fix=True, map_offsets=None, use_flow=True, use_radar=True,
                 flow_config=None, boxes_at="camera"):
        from nuscenes.nuscenes import NuScenes
        assert boxes_at in ("camera", "annotation"), boxes_at
        self.nusc = nusc or NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.dataroot = dataroot
        self._maps = {}
        self.boxes_at = boxes_at                       # time of the moving boxes: "camera" (default) or "annotation"
        self.stationary_fix = stationary_fix
        self.map_offsets = map_offsets or {}
        self.use_flow, self.use_radar, self.flow_config = use_flow, use_radar, flow_config
        self._scenes = collections.OrderedDict()
        self._stationary = {}
        self._bounds = {}

    # ---- scene data (tracks, ego, radar) of the flow box, a few scenes kept
    def scene(self, scene_token):
        if scene_token not in self._scenes:
            self._scenes[scene_token] = NuScenesScene(self.nusc, self.dataroot, scene_token, config=self.flow_config)
            while len(self._scenes) > 3:
                self._scenes.popitem(last=False)
        return self._scenes[scene_token]

    def records_in_patch(self, nmap, box, layers):
        """same result as nmap.get_records_in_patch(box, layers, 'intersect'), but the devkit extracts every polygon of the
        whole map per call (0.5 s per keyframe): the bounding box of every record is computed once per map, and the
        devkit's exact test runs only on the records whose box meets the patch."""
        out = {}
        x0, y0, x1, y1 = box
        for layer in layers:
            key = (nmap.map_name, layer)
            if key not in self._bounds:
                try:
                    toks = [r["token"] for r in getattr(nmap, layer)]
                    self._bounds[key] = (toks, np.array([nmap.get_bounds(layer, t) for t in toks], float).reshape(-1, 4))
                except Exception:                                  # a layer without bounds: devkit path
                    self._bounds[key] = None
            b = self._bounds[key]
            if b is None:
                out[layer] = nmap.get_records_in_patch(box, [layer], mode="intersect")[layer]
                continue
            toks, bb = b
            hit = (bb[:, 0] <= x1) & (bb[:, 2] >= x0) & (bb[:, 1] <= y1) & (bb[:, 3] >= y0)
            out[layer] = [toks[i] for i in np.nonzero(hit)[0]
                          if nmap.explorer.is_record_in_patch(layer, toks[i], box, "intersect")]
        return out

    def signature(self):
        """10 hex chars identifying this generator: the source of every file that shapes the GT (SIGNATURE_FILES) plus the
        flags (stationary_fix, map_offsets, use_flow, use_radar, flow config). Part of every cache key of derived GT, so a
        changed generator can never be mixed with GT cached by an older one."""
        h = hashlib.sha1()
        here = os.path.dirname(__file__)
        for name in SIGNATURE_FILES:
            h.update(open(os.path.join(here, name), "rb").read())
        h.update(repr((self.stationary_fix, sorted((k, tuple(v)) for k, v in self.map_offsets.items()), self.use_flow,
                       self.use_radar, self.flow_config, self.boxes_at)).encode())
        return h.hexdigest()[:10]

    def stationary_instances(self, scene_token):
        """instance token -> fixed world pose (translation (3,), yaw, size (3,)) of the scene's stationary instances
        (rule: autogt.flow.is_stationary on the 2 Hz tracks)."""
        if scene_token in self._stationary:
            return self._stationary[scene_token]
        cfg = self.flow_config or flowmod.Config()
        fixed = {}
        for tr in self.scene(scene_token).tracks:
            if flowmod.is_stationary(tr, cfg):
                fixed[tr.id] = (np.median(tr.extra["trans"], 0), math.atan2(np.sin(tr.yaw).mean(), np.cos(tr.yaw).mean()),
                                np.asarray(tr.size, float))
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
    def keyframe_gt(self, sample_token, grid, ss=4, sd=None):
        """all layers of one keyframe (see the module docstring). `sd` = another CAM_FRONT sample_data of the same scene
        (a 12 Hz sweep): the ego frame, camera and map layers are those of that frame (boxes stay the keyframe's)."""
        nusc = self.nusc
        sample = nusc.get("sample", sample_token)
        sd = sd or self.cam(sample_token)
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
        recs = self.records_in_patch(nmap, box, list(AREAS) + list(LINES) + ["lane", "lane_connector"])
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

        # objects (exact annotations of this keyframe) and their velocity in the ego axes (autogt flow box)
        Rg2e = T_g2e[:3, :3]
        objs = {k: R.canvas() for k in OBJECTS}
        inst = np.full(R.shape, -1, np.int32)
        ignore = R.canvas()
        unknown = R.canvas()                                     # footprints whose velocity is unknown
        vel = np.zeros((2,) + R.shape, np.float32)
        sig = np.zeros(R.shape, np.float32)
        pres = R.canvas()                                        # sub-pixels covered by a box with known velocity
        boxes = []
        fixed = self.stationary_instances(sample["scene_token"]) if self.stationary_fix else {}
        fobj = {}
        if self.use_flow:
            fobj = {o.id: o for o in self.scene(sample["scene_token"]).flow(sample_token, None, at="annotation",
                                                                              use_radar=self.use_radar).objects}
        # the GT time of the boxes: the annotations are at the keyframe (lidar) time, the ego frame is that of `sd`
        t_cam = (sd["timestamp"] - sample["timestamp"]) * 1e-6 if self.boxes_at == "camera" else 0.0
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
            o = fobj.get(a["instance_token"])
            if stationary:
                v, known, source, vflag, vsig = np.zeros(2), True, "stationary", "ok", 0.05
            elif o is not None and np.isfinite(o.v).all():
                v, known, source, vflag, vsig = o.v, True, o.source, o.flag, math.sqrt(float(np.trace(o.cov)) / 2)
            elif not self.use_flow and np.isfinite(nusc.box_velocity(atok)[:2]).all():       # legacy: the devkit's velocity
                v, known, source, vflag, vsig = np.asarray(nusc.box_velocity(atok)[:2], float), True, "devkit", "ok", float("nan")
            else:                                                # unknown (short track, flow off): never a silent zero
                v, known, source, vflag, vsig = np.zeros(2), False, "unknown", "short_track", float("nan")
            ve = Rg2e[:2, :2] @ v
            # annotation time -> GT time: constant velocity over t_cam; a stationary box (v = 0) and a box of unknown
            # velocity stay where the annotation (or the fixed world pose) puts them; z, size and yaw are kept
            trans_ann = np.asarray(trans, float)
            shifted = known and not stationary and t_cam != 0.0
            shift = v * t_cam if shifted else np.zeros(2)
            trans = trans_ann + np.r_[shift, 0.0]
            w, l = size[0], size[1]
            c, s_ = math.cos(yaw), math.sin(yaw)
            loc = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
            cg_ = loc @ np.array([[c, s_], [-s_, c]]) + np.asarray(trans)[:2]
            ce = to_ego(cg_)
            one = R.fill(R.canvas(), [(ce, [])])
            observed = a["num_lidar_pts"] + a["num_radar_pts"] > 0
            if observed:
                objs[grp] |= one
                inst[one > 0] = len(boxes)                       # index into meta["boxes"]
                if known:
                    vel[0][one > 0], vel[1][one > 0] = ve[0], ve[1]
                    sig[one > 0] = vsig
                    pres |= one
                else:
                    unknown |= one
            else:                                                # no sensor point: neither object nor free -> ignore
                ignore |= one
            boxes.append({"group": grp, "corners": ce, "center": (Rg2e @ (np.asarray(trans) - T_e2g[:3, 3])),
                          "size": list(size), "yaw": yaw - yaw_e, "vel": ve, "vel_source": source, "vel_flag": vflag,
                          "vel_sigma": vsig, "vel_known": known, "visibility": a["visibility_token"], "token": atok,
                          "instance": a["instance_token"], "stationary": stationary, "observed": observed,
                          "world": np.asarray(trans), "world_ann": trans_ann, "shift": Rg2e[:2, :2] @ shift,
                          "shift_s": float(t_cam) if shifted else 0.0})
        for k in OBJECTS:
            out[k] = R.coverage(objs[k])
        out["ignore"] = R.coverage(ignore)
        # velocity at a cell = the box velocity averaged over the covered central sub-pixels only (an edge cell keeps the
        # full speed of its box instead of a coverage-scaled one)
        pc = R.at_centres(pres.astype(np.float32))
        out["velocity"] = (np.stack([R.at_centres(vel[0]), R.at_centres(vel[1])]) / np.maximum(pc, 1e-6)
                           * (pc > 0)).astype(np.float32)
        out["instance"] = inst.reshape(grid.nx, ss, grid.ny, ss)[:, ss // 2, :, ss // 2]
        # flow: the static world moves with the ego (camera time), objects with their own velocity; weight 0 where unknown
        if self.use_flow:
            sc = self.scene(sample["scene_token"])
            out["flow_ego"] = flowmod.ego_flow_field(grid, flowmod.ego_velocity(sc.ego, sd["timestamp"] * 1e-6))
        else:
            out["flow_ego"] = np.zeros((2, grid.nx, grid.ny), np.float32)
        out["flow"] = out["flow_ego"] + out["velocity"]
        out["flow_weight"] = ((R.at_centres(ignore) == 0) & (R.at_centres(unknown) == 0)).astype(np.float32)
        out["flow_sigma"] = np.where(out["instance"] >= 0, R.at_centres(sig), 0.1).astype(np.float32)

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
                       "annotation_timestamp": sample["timestamp"], "boxes_at": self.boxes_at,
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
