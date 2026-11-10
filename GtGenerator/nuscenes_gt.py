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
- masks (autogt/coverage.py, v7): camera (ground cells that project into the CAM_FRONT image), occluded (ground cells a box hides from the
  camera, boxes standing on the ground plane with their real height), radar (inside the fan of one of the five radars: +-60 deg, measured:
  99.9 % of all detections of nuScenes mini lie inside it), undefined (neither), ignore (see objects). Per box in meta: num_lidar_pts,
  num_radar_pts (nuScenes counts), share_camera / share_radar / share_fusion (the share of the footprint the sensors observe, D38 b)
- meta: sample, scene, timestamp, T_e2g, boxes, res, cam_file, K, T_c2e, image_size, radar_poses, vectors (exact dividers / centre-lines
  as ego polylines)
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
from shapely.geometry import MultiPolygon

from . import coverage as cvg
from . import flow as flowmod
from .coverage import RADAR_FOV                      # half angle, range (m): measured, see coverage.py (kept here: older code imports it)
from .layers import AREAS, DIST_MAX, LINES, OBJECTS, SHIFT, Raster, cell_centres
from .nuscenes_adapter import NuScenesScene, category_group, pose_matrix

SIGNATURE_FILES = ("grid.py", "layers.py", "flow.py", "nuscenes_adapter.py", "nuscenes_gt.py", "coverage.py")


def _to_ego(xy, T_g2e):
    """global (x, y) points (n, 2) -> the ego frame"""
    return (np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))] @ T_g2e.T)[:, :2]


class _BoxCanvas:
    """what the boxes of one keyframe paint on the supersampled canvas: footprints per class and the instance index where a box has a
    lidar / radar point, its velocity (and sigma) where it is known, and the footprints that are ignored (no sensor point) or of unknown
    velocity. The boxes are painted in the order of meta["boxes"]; `layers` turns the canvas into the grid layers."""

    def __init__(self, R):
        self.objs = {k: R.canvas() for k in OBJECTS}
        self.inst = np.full(R.shape, -1, np.int32)
        self.ignore, self.unknown, self.pres = R.canvas(), R.canvas(), R.canvas()      # no sensor point / velocity unknown / velocity known
        self.vel = np.zeros((2,) + R.shape, np.float32)
        self.sig = np.zeros(R.shape, np.float32)
        self.n = 0                                           # boxes painted so far = the index of the next one in meta["boxes"]

    def paint(self, group, one, observed, known, ve, vsig):
        """`one`: the footprint of the next box on the canvas"""
        m = one > 0
        if not observed:                                     # no sensor point: neither object nor free -> ignore
            self.ignore |= one
        else:
            self.objs[group] |= one
            self.inst[m] = self.n
            if known:
                self.vel[0][m], self.vel[1][m] = ve[0], ve[1]
                self.sig[m] = vsig
                self.pres |= one
            else:
                self.unknown |= one
        self.n += 1

    def layers(self, R, grid, ss):
        """vehicle / pedestrian / two_wheeler, ignore, velocity, instance, flow_weight, flow_sigma"""
        out = {k: R.coverage(self.objs[k]) for k in OBJECTS}
        out["ignore"] = R.coverage(self.ignore)
        # velocity at a cell = the box velocity averaged over the covered central sub-pixels only (an edge cell keeps the
        # full speed of its box instead of a coverage-scaled one)
        pc = R.at_centres(self.pres.astype(np.float32))
        out["velocity"] = (np.stack([R.at_centres(self.vel[0]), R.at_centres(self.vel[1])]) / np.maximum(pc, 1e-6)
                           * (pc > 0)).astype(np.float32)
        # instance = the box index in the same central sub-pixels as velocity and flow_sigma (the highest index when boxes overlap), so
        # that a cell with a velocity always names its box (meta["boxes"][instance])
        c = ss // 2
        a = self.inst.reshape(grid.nx, ss, grid.ny, ss)
        out["instance"] = (a[:, c - 1:c + 1, :, c - 1:c + 1] if ss % 2 == 0 else a[:, c:c + 1, :, c:c + 1]).max((1, 3))
        out["flow_weight"] = ((R.at_centres(self.ignore) == 0) & (R.at_centres(self.unknown) == 0)).astype(np.float32)
        out["flow_sigma"] = np.where(out["instance"] >= 0, R.at_centres(self.sig) / np.maximum(pc, 1e-6), 0.1).astype(np.float32)   # mean over the covered sub-pixels, like velocity
        return out


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
        self._geoms = {}                               # the shapely polygon / node array of a map record, made once
        self.signature()                               # frozen now (see signature())

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
        devkit's exact test runs only on the records whose box meets the patch (with the geometry of a record made once)."""
        from shapely.geometry import box as sh_box
        out = {}
        x0, y0, x1, y1 = box
        patch = sh_box(x0, y0, x1, y1)
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
            out[layer] = [toks[i] for i in np.nonzero(hit)[0] if self._record_in_patch(nmap, layer, toks[i], box, patch)]
        return out

    def _record_in_patch(self, nmap, layer, token, box, patch):
        """the devkit's exact `is_record_in_patch(layer, token, box, "intersect")`: a polygon record (drivable_area: the multi-polygon of its
        parts) intersects the patch, a line record has a node strictly inside it; the geometry of a record is made once, not on every call"""
        key = (nmap.map_name, layer, token)
        if key not in self._geoms:
            rec = nmap.get(layer, token)
            if layer == "drivable_area":
                self._geoms[key] = MultiPolygon([nmap.extract_polygon(t) for t in rec["polygon_tokens"]])
            elif layer in nmap.lookup_polygon_layers:
                self._geoms[key] = nmap.extract_polygon(rec["polygon_token"])
            elif layer in nmap.non_geometric_line_layers:
                self._geoms[key] = np.array([[n["x"], n["y"]] for n in (nmap.get("node", t) for t in rec["node_tokens"])])
            else:
                return nmap.explorer.is_record_in_patch(layer, token, box, "intersect")
        geom = self._geoms[key]
        if isinstance(geom, np.ndarray):                           # a line: some node inside the patch (a line without nodes: none)
            x0, y0, x1, y1 = box
            return bool(len(geom) and np.any((geom[:, 0] < x1) & (geom[:, 0] > x0) & (geom[:, 1] < y1) & (geom[:, 1] > y0)))
        return geom.intersects(patch)

    def signature(self):
        """10 hex chars identifying this generator: the source of every file that shapes the GT (SIGNATURE_FILES) plus the
        flags (stationary_fix, map_offsets, use_flow, use_radar, flow config). Part of every cache key of derived GT, so a
        changed generator can never be mixed with GT cached by an older one."""
        sig = getattr(self, "_signature", None)
        if sig is None:                  # computed ONCE (at construction): a process that runs while the source is edited keeps
            h = hashlib.sha1()           # the key of the code it runs - re-hashing the files per call made a running job
            here = os.path.dirname(__file__)                 # write its cache under a key no other process reads
            for name in SIGNATURE_FILES:
                h.update(open(os.path.join(here, name), "rb").read())
            h.update(repr((self.stationary_fix, sorted((k, tuple(v)) for k, v in self.map_offsets.items()), self.use_flow,
                           self.use_radar, self.flow_config, self.boxes_at)).encode())
            sig = self._signature = h.hexdigest()[:10]
        return sig

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

    def cam(self, sample_token, channel="CAM_FRONT"):
        return self.nusc.get("sample_data", self.nusc.get("sample", sample_token)["data"][channel])

    def ego_pose(self, sd):
        return pose_matrix(self.nusc.get("ego_pose", sd["ego_pose_token"]))

    # ---- one keyframe
    def keyframe_gt(self, sample_token, grid, ss=4, sd=None):
        """all layers of one keyframe (see the module docstring). `sd` = another CAM_FRONT sample_data of the same scene
        (a 12 Hz sweep): the ego frame, camera and map layers are those of that frame, and the keyframe's boxes are moved to its
        time by v * dt (first order, up to +-0.25 s; with boxes_at="annotation" they stay where they were annotated)."""
        sample = self.nusc.get("sample", sample_token)
        sd = sd or self.cam(sample_token)
        T_e2g = self.ego_pose(sd)
        R = Raster(grid, ss)
        out, vectors = self._map_layers(R, grid, T_e2g, sample["scene_token"])
        boxes, canvas = self._boxes(sample, sd, T_e2g, R)
        out.update(canvas.layers(R, grid, ss))
        out["flow_ego"] = self._ego_flow(sample, sd, grid)           # the static world moves with the ego (camera time) ...
        out["flow"] = out["flow_ego"] + out["velocity"]              # ... objects with their own velocity; weight 0 where unknown
        masks, rig = self._coverage(sample, sd, boxes, grid)
        out.update(masks)
        out["meta"] = {"sample": sample_token, "scene": sample["scene_token"], "timestamp": sd["timestamp"],
                       "annotation_timestamp": sample["timestamp"], "boxes_at": self.boxes_at, "T_e2g": T_e2g, "boxes": boxes,
                       "res": grid.res, "cam_file": sd["filename"], "vectors": vectors, **rig}
        return out

    def _map_layers(self, R, grid, T_e2g, scene_token):
        """the map layers (areas, lines with their distance fields, centre-lines with their direction) -> (layers, exact vectors)"""
        T_g2e = np.linalg.inv(T_e2g)
        corners = np.array([[grid.x_min, grid.y_min], [grid.x_min, grid.y_max], [grid.x_max, grid.y_min], [grid.x_max, grid.y_max]])
        cg = (np.c_[corners, np.zeros(4), np.ones(4)] @ T_e2g.T)[:, :2]
        box = (cg[:, 0].min() - 5, cg[:, 1].min() - 5, cg[:, 0].max() + 5, cg[:, 1].max() + 5)
        nmap = self.map_for(scene_token)
        recs = self.records_in_patch(nmap, box, list(AREAS) + list(LINES) + ["lane", "lane_connector"])
        out = {}
        for layer in AREAS:
            polys = []
            for tok in recs[layer]:
                rec = nmap.get(layer, tok)
                # a token of None is real: in singapore-hollandvillage / -queenstown the road network of the drivable
                # area is the one polygon whose token is null (found 2026-10-26 - it had been skipped: no road in GT)
                for ptok in (rec["polygon_tokens"] if "polygon_tokens" in rec else [rec["polygon_token"]]):
                    poly = nmap.extract_polygon(ptok)
                    if poly.is_empty:
                        continue
                    polys.append((_to_ego(np.asarray(poly.exterior.coords)[:, :2], T_g2e),
                                  [_to_ego(np.asarray(h.coords)[:, :2], T_g2e) for h in poly.interiors]))
            out[layer] = R.coverage(R.fill(R.canvas(), polys))
        vectors = {}
        for layer in LINES:
            img = R.canvas()
            pls = []
            for tok in recs[layer]:
                line = nmap.extract_line(nmap.get(layer, tok)["line_token"])
                if not line.is_empty:
                    g = np.asarray(line.coords)[:, :2] + np.asarray(self.map_offsets.get(tok, (0.0, 0.0)))  # world-fixed
                    pls.append(_to_ego(g, T_g2e))
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
            xy = _to_ego(P[:, :2], T_g2e)
            vectors["centreline"].append(xy)
            hd = P[:, 2] - yaw_e
            R.lines(cmask, [xy])
            pp = R.pts(xy)                                      # the integer points of the arc, once (a segment is two of them)
            for i in range(len(xy) - 1):                        # direction per segment
                cv2.polylines(ccos, [pp[i:i + 2]], False, float(math.cos(hd[i])), 1, cv2.LINE_8, SHIFT)
                cv2.polylines(csin, [pp[i:i + 2]], False, float(math.sin(hd[i])), 1, cv2.LINE_8, SHIFT)
        out["centreline"] = (R.coverage(cmask) > 0).astype(np.float32)
        nz = np.maximum(R.coverage(cmask.astype(np.float32)), 1e-6)
        out["centreline_dir"] = np.stack([R.coverage(ccos) / nz, R.coverage(csin) / nz]) * out["centreline"]
        return out, vectors

    def _box_velocity(self, atok, stationary, o):
        """(v, known, source, flag, sigma) of one annotated box: stationary (v = 0), the flow box's estimate `o`, the devkit's velocity
        (flow off), or unknown (a short track): never a silent zero"""
        if stationary:
            return np.zeros(2), True, "stationary", "ok", 0.05
        if o is not None and np.isfinite(o.v).all():
            return o.v, True, o.source, o.flag, math.sqrt(float(np.trace(o.cov)) / 2)
        if not self.use_flow and np.isfinite(self.nusc.box_velocity(atok)[:2]).all():
            return np.asarray(self.nusc.box_velocity(atok)[:2], float), True, "devkit", "ok", float("nan")
        return np.zeros(2), False, "unknown", "short_track", float("nan")

    def _boxes(self, sample, sd, T_e2g, R):
        """the annotated boxes of the keyframe as meta["boxes"] entries (exact annotations moved to the camera time, their velocity in the
        ego axes from the flow box) and the canvas they paint -> (boxes, _BoxCanvas)"""
        T_g2e = np.linalg.inv(T_e2g)
        Rg2e = T_g2e[:3, :3]
        yaw_e = math.atan2(T_e2g[1, 0], T_e2g[0, 0])
        canvas, boxes = _BoxCanvas(R), []
        fixed = self.stationary_instances(sample["scene_token"]) if self.stationary_fix else {}
        fobj = {}
        if self.use_flow:
            fobj = {o.id: o for o in self.scene(sample["scene_token"]).flow(sample["token"], None, at="annotation",
                                                                              use_radar=self.use_radar).objects}
        # the GT time of the boxes: the annotations are at the keyframe (lidar) time, the ego frame is that of `sd`
        t_cam = (sd["timestamp"] - sample["timestamp"]) * 1e-6 if self.boxes_at == "camera" else 0.0
        for atok in sample["anns"]:
            a = self.nusc.get("sample_annotation", atok)
            grp = category_group(a["category_name"])
            if grp is None:
                continue
            stationary = a["instance_token"] in fixed
            if stationary:                                       # one world pose for the whole track (rule above)
                trans, yaw, size = fixed[a["instance_token"]]
            else:
                trans, yaw, size = np.array(a["translation"]), Quaternion(a["rotation"]).yaw_pitch_roll[0], a["size"]
            v, known, source, vflag, vsig = self._box_velocity(atok, stationary, fobj.get(a["instance_token"]))
            ve = Rg2e[:2, :2] @ v
            # annotation time -> GT time: constant velocity over t_cam; a stationary box (v = 0) and a box of unknown
            # velocity stay where the annotation (or the fixed world pose) puts them; z, size and yaw are kept
            trans_ann = np.array(trans, float)                   # a copy: `fixed` poses are shared by all keyframes of the scene
            shifted = known and not stationary and t_cam != 0.0
            shift = v * t_cam if shifted else np.zeros(2)
            trans = trans_ann + np.r_[shift, 0.0]
            w, l = size[0], size[1]
            c, s_ = math.cos(yaw), math.sin(yaw)
            loc = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
            ce = _to_ego(loc @ np.array([[c, s_], [-s_, c]]) + np.asarray(trans)[:2], T_g2e)
            observed = a["num_lidar_pts"] + a["num_radar_pts"] > 0
            canvas.paint(grp, R.fill(R.canvas(), [(ce, [])]), observed, known, ve, vsig)
            boxes.append({"group": grp, "corners": ce, "center": (Rg2e @ (np.asarray(trans) - T_e2g[:3, 3])),
                          "size": list(size), "yaw": yaw - yaw_e, "vel": ve, "vel_source": source, "vel_flag": vflag,
                          "vel_sigma": vsig, "vel_known": known, "visibility": a["visibility_token"], "token": atok,
                          "instance": a["instance_token"], "stationary": stationary, "observed": observed,
                          "world": np.asarray(trans), "world_ann": trans_ann, "shift": Rg2e[:2, :2] @ shift,
                          "shift_s": float(t_cam) if shifted else 0.0,
                          "num_lidar_pts": int(a["num_lidar_pts"]), "num_radar_pts": int(a["num_radar_pts"])})
        return boxes, canvas

    def _ego_flow(self, sample, sd, grid):
        """the apparent velocity of a static world point caused by the ego motion at the camera time (0 with the flow switched off)"""
        if not self.use_flow:
            return np.zeros((2, grid.nx, grid.ny), np.float32)
        ego = self.scene(sample["scene_token"]).ego
        return flowmod.ego_flow_field(grid, flowmod.ego_velocity(ego, sd["timestamp"] * 1e-6, (self.flow_config or flowmod.Config()).ego_half_window))

    def _coverage(self, sample, sd, boxes, grid):
        """the masks camera / occluded / radar / undefined (autogt/coverage.py: camera cells = ground points in the image, what boxes hide
        from the camera by their height, the radar fans) and the observed shares of every box (set on it) -> (masks, the rig for meta)"""
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        K = np.array(cs["camera_intrinsic"])
        T_c2e = pose_matrix(cs)
        img_size = (sd["width"], sd["height"])
        gx, gy = cell_centres(grid)
        cam_xyz = T_c2e[:3, 3]
        poses = radar_poses(self, sample)
        prisms = [cvg.prism_of(b) for b in boxes if b["observed"]]       # only boxes with a lidar / radar point hide (as before)
        masks = {"camera": cvg.camera_cells(gx, gy, K, T_c2e, img_size).astype(np.float32),
                 "occluded": cvg.hidden_cells(gx, gy, cam_xyz, prisms).astype(np.float32),
                 "radar": cvg.radar_cells(gx, gy, poses).astype(np.float32)}
        masks["undefined"] = ((masks["camera"] == 0) & (masks["radar"] == 0)).astype(np.float32)
        for b, sh in zip(boxes, cvg.object_shares_many(boxes, prisms, cam_xyz, K, T_c2e, img_size, poses, grid)):
            b["share_camera"], b["share_radar"], b["share_fusion"] = sh["camera"], sh["radar"], sh["fusion"]     # per object, D38 b
        return masks, {"K": K, "T_c2e": T_c2e, "image_size": img_size, "radar_poses": poses}

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


def radar_poses(gt, sample):
    """the sensor -> ego matrices of the radars of a sample"""
    return [pose_matrix(gt.nusc.get("calibrated_sensor", gt.nusc.get("sample_data", tok)["calibrated_sensor_token"]))
            for ch, tok in sample["data"].items() if ch.startswith("RADAR")]


def radar_coverage(gt, sample, gx, gy):
    """cells inside the fan of one of the radars of the sample (see coverage.radar_cells)"""
    return cvg.radar_cells(gx, gy, radar_poses(gt, sample))
