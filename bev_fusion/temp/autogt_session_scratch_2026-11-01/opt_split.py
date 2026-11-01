"""scratch: split NuScenesGT.keyframe_gt (180 lines) into the map layers, the boxes, the flow field and the sensor coverage; behaviour unchanged
(checked by comparing every layer and meta entry with the tag autogt/v7 on real keyframes)"""
import os

os.chdir("/mnt/c/Auto-research")
p = "autogt/nuscenes_gt.py"
s = open(p, encoding="utf-8").read()
a = s.index("    # ---- one keyframe\n    def keyframe_gt(")
b = s.index("    # ---- training sample: previous, current, next keyframe and the ego steps between them")
# the class that paints the boxes goes before the class NuScenesGT
cls_at = s.index("class NuScenesGT:")
helper = '''def _to_ego(xy, T_g2e):
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


'''
new = '''    # ---- one keyframe
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

'''
s = s[:cls_at] + helper + s[cls_at:a] + new + s[b:]
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("split done;", len(s.splitlines()), "lines")
