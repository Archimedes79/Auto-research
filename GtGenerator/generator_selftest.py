"""Tests of the front GT generator and the lane graphs against INDEPENDENT references (the devkit, the nuScenes map, the camera image, a file round trip), which the other selftests do not
have: the footprints of the boxes against the devkit's corners, the left / right handedness against the front camera image, the centre-line direction against the vector centre-lines, the lane
divider mask against its vectors, the map layers against exact polygons, the stored file against the live keyframe layer by layer, and the invariants of lane graphs on real frames of every map
(links symmetric, window events, ids, tiers); plus a guard that the hashed lane sources import no module the signature does not cover. Nothing here changes a hashed file: the tests state what
the generator does today, so a later change that moves it fails here before a GT is built from it.

    python -m GtGenerator.generator_selftest [--root ~/data/nuscenes]      (also run by `python -m GtGenerator selftest`)
"""
import ast
import math
import os
import tempfile

import numpy as np

from .grid import GridSpec


def _devkit_corners(a, T_e2g):
    """the 4 bottom corners (ego frame, xy) of an annotation, from the devkit's own Box class"""
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion
    bottom = Box(a["translation"], a["size"], Quaternion(a["rotation"])).bottom_corners()[:2].T          # (4, 2) world
    return (bottom - T_e2g[:2, 3]) @ T_e2g[:2, :2]


def _same_corners(a, b, tol):
    """two sets of 4 points are the same up to the order"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return all(np.linalg.norm(b - p, axis=1).min() < tol for p in a) and all(np.linalg.norm(a - p, axis=1).min() < tol for p in b)


def _contract_checks():
    """the pieces of the corrections without any data: the angle wrap, the flag of a stationary box, flow_sigma on a cell of a box of unknown velocity (finite, weight 0), the even supersampling"""
    import types
    from .flow import ObjectFlow
    from .layers import Raster
    from .nuscenes_gt import FLOW_SIGMA_UNKNOWN, NuScenesGT, _BoxCanvas, _wrap
    for a, want in ((0.0, 0.0), (math.pi + 0.1, -math.pi + 0.1), (-math.pi - 0.1, math.pi - 0.1), (2 * math.pi + 0.3, 0.3), (-7.0, -7.0 + 2 * math.pi)):
        assert abs(_wrap(a) - want) < 1e-9 and abs(_wrap(a)) <= math.pi, (a, _wrap(a), want)
    me = types.SimpleNamespace(use_flow=True, nusc=None)
    o = ObjectFlow("i", "vehicle", np.zeros(2), 0.0, np.ones(3), np.zeros(2), np.eye(2), "stationary", flag="stationary_but_moving", stationary=True)
    assert NuScenesGT._box_velocity(me, "a", True, o)[3] == "stationary_but_moving", "the flag of the flow box stays on a stationary box"
    assert NuScenesGT._box_velocity(me, "a", True, None)[3] == "ok"
    grid = GridSpec()
    R = Raster(grid, 4)
    cv = _BoxCanvas(R)
    sq = lambda x0, y0: R.fill(R.canvas(), [(np.array([[x0, y0], [x0 + 4, y0], [x0 + 4, y0 + 4], [x0, y0 + 4]], float), [])])          # noqa: E731
    cv.paint("vehicle", sq(10, 0), True, False, np.zeros(2), float("nan"))                       # observed, velocity unknown
    cv.paint("vehicle", sq(20, 0), True, True, np.array([3.0, 0.0]), 0.4)                       # observed, velocity known
    out = cv.layers(R, grid, 4)
    unk, kn = (out["instance"] == 0), (out["instance"] == 1)
    assert unk.any() and kn.any() and (out["flow_weight"][unk] == 0).all() and (out["flow_weight"][kn] == 1).all()
    assert (out["flow_sigma"][unk] == FLOW_SIGMA_UNKNOWN).all() and np.allclose(out["flow_sigma"][kn], 0.4) and (out["flow_sigma"][~(unk | kn)] == 0.1).all(), "sigma: unknown / known / free space"
    assert np.isfinite(out["flow_sigma"]).all()
    for bad_ss in (1, 3):
        try:
            Raster(grid, bad_ss)
            raise AssertionError("an odd supersampling must be refused")
        except ValueError as e:
            assert "even" in str(e)
    # coverage.blocked_at_heights (one clip for k heights) equals `blocked` per height, with several prisms, points inside prisms and a camera behind a box
    from . import coverage as cvg
    for seed in range(6):
        rng = np.random.default_rng(seed)
        prisms = [(np.array([[x, y - 1], [x, y + 1], [x + 4, y + 1], [x + 4, y - 1]], float), 0.0, h)
                  for x, y, h in ((8, 0, 1.6), (14, 3, 3.0), (22, -4, 0.8), (30 + seed, 1, 2.5))]
        cam = np.array([-3.0 if seed < 5 else 12.0, 0.2, 1.5])                                         # the last one: a camera inside the scene, boxes behind it
        pts = np.c_[rng.uniform(0, 45, 3000), rng.uniform(-12, 12, 3000)]
        zs = rng.uniform(0.0, 3.0, (3, 3000))
        got = cvg.blocked_at_heights(pts, zs, cam, prisms)
        for h in range(3):
            assert np.array_equal(got[h], cvg.blocked(np.c_[pts, zs[h]], cam, prisms)), (seed, h)
        assert got.any() and not got.all(), "the test must contain blocked and free points"


def _front_checks(gt, toks):
    """footprints against the devkit, handedness against the image, centre-line direction, divider width, stored file against live"""
    from scipy.spatial import cKDTree
    from .store import _flat, load_keyframe, save_keyframe
    grid = GridSpec()
    n_dev = n_side = n_dir = n_div = 0
    for tok in toks:
        g = gt.keyframe(tok)
        m = g["meta"]
        T = np.asarray(m["T_e2g"], float)
        K, T_c2e = np.asarray(m["K"], float), np.asarray(m["T_c2e"], float)
        yaw_e = math.atan2(T[1, 0], T[0, 0])
        for b in m["boxes"]:
            # 1. the corners are the box (centre, size, yaw): w and l the right way round, the yaw with its sign
            w, l = b["size"][0], b["size"][1]
            c, s = math.cos(b["yaw"] + yaw_e), math.sin(b["yaw"] + yaw_e)
            loc = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
            own = (loc @ np.array([[c, s], [-s, c]]) + b["world"][:2] - T[:2, 3]) @ T[:2, :2]
            assert _same_corners(b["corners"], own, 1e-3), "the corners of a box are its centre, size and yaw"
            # ... and the devkit's box: a box that is not fixed in the world is the annotation, moved by its velocity to the camera time (`shift`)
            if not b["stationary"]:
                a = gt.nusc.get("sample_annotation", b["token"])
                assert _same_corners(b["corners"], _devkit_corners(a, T) + b["shift"], 0.01), "the footprint equals the devkit's bottom corners (moved to the camera time)"
                assert abs(abs(float(np.cross(b["corners"][1] - b["corners"][0], b["corners"][3] - b["corners"][0]))) - w * l) < 0.01, "the area is w * l"
                n_dev += 1
            # 2. handedness: an observed box to the left of the car (y > 0) is left of the image centre
            x, y, z = b["center"]
            if b["observed"] and 4 < x < 40 and abs(y) > 3:
                cam = np.linalg.inv(T_c2e) @ np.array([x, y, z, 1.0])
                if cam[2] > 1:
                    assert (K[0, 0] * cam[0] / cam[2] + K[0, 2] < K[0, 2]) == (y > 0), "y left = image left: the grid is not mirrored"
                    n_side += 1
        # 3. the centre-line direction points along the centre-lines ahead of the car (the cos / sin of the heading in the ego frame)
        cl = g["centreline"] > 0.5
        cx, cy = grid.centres()
        ahead = cl & (cx > 3) & (cx < 30) & (np.abs(cy) < 2)
        if ahead.sum() >= 10:
            assert np.median(g["centreline_dir"][0][ahead]) > 0.5, "the centre-line ahead of the car points forward"
            n_dir += 1
        # 3b. the contract of the angles and flags (v13): box yaw wrapped to [-pi, pi], centre-line direction a unit vector or (0, 0), flow_sigma finite and positive, the flag of a
        #     stationary box that no sensor sees is `unobserved` (before it was always `ok`)
        assert all(abs(b["yaw"]) <= math.pi + 1e-9 for b in m["boxes"]), "box yaw is wrapped to [-pi, pi]"
        nrm = np.hypot(g["centreline_dir"][0], g["centreline_dir"][1])[cl]
        assert np.all((np.abs(nrm - 1) < 1e-3) | (nrm == 0)), "centreline_dir is a unit vector, or (0, 0) where the directions cancel"
        assert np.isfinite(g["flow_sigma"]).all() and (g["flow_sigma"] > 0).all(), "flow_sigma is finite and positive everywhere"
        assert all(b["vel_flag"] == "unobserved" for b in m["boxes"] if b["stationary"] and not b["observed"]), "an unobserved stationary box says so"
        # 4. the divider mask lies on its vectors (within a cell and a half) and every vector sample is within a cell of a set cell
        divs = [np.asarray(v, float) for v in m["vectors"]["lane_divider"] if len(v) >= 2]
        if divs:
            dense = np.concatenate([np.c_[np.linspace(v[i, 0], v[i + 1, 0], 20), np.linspace(v[i, 1], v[i + 1, 1], 20)] for v in divs for i in range(len(v) - 1)])
            cells = np.c_[cx[g["lane_divider"] > 0.5], cy[g["lane_divider"] > 0.5]]
            if len(cells):
                assert cKDTree(dense).query(cells)[0].max() < 1.15, "every set divider cell lies within a metre of a divider vector (measured 1.01 m: the mask is a wide line)"
                inside = dense[(dense[:, 0] > 0.5) & (dense[:, 0] < 59.5) & (np.abs(dense[:, 1]) < 25.5)]
                if len(inside):
                    assert np.median(cKDTree(cells).query(inside)[0]) < grid.res, "the dividers are drawn (the mask is not empty, not shifted)"
                n_div += 1
        # 5. the stored file is the live keyframe, layer by layer (float16 for the float layers), the instance map and the meta exactly
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "k.npz")
            save_keyframe(path, _flat(g))
            back = load_keyframe(d, "k")
        for k, v in g.items():
            if k == "meta":
                assert [(b["token"], b["group"], b["observed"]) for b in back["meta"]["boxes"]] == [(b["token"], b["group"], b["observed"]) for b in v["boxes"]] and back["meta"]["sample"] == v["sample"], "meta survives the file"
            elif v.dtype == np.float32:
                ok = np.isfinite(v)
                assert np.array_equal(ok, np.isfinite(back[k])) and np.allclose(v[ok], back[k][ok], rtol=2e-3, atol=0.02), f"layer {k}: the stored file is the live layer"
            else:
                assert np.array_equal(v, back[k]), f"layer {k}"
    assert n_dev >= 1 and n_side >= 5 and n_dir >= 1 and n_div >= 1, (n_dev, n_side, n_dir, n_div)
    return f"{len(toks)} keyframes: footprints = devkit corners ({n_dev} boxes), handedness against the image ({n_side} boxes), centre-line direction, divider mask, file round trip"


def _map_checks(gt, name, toks):
    """the map layers against exact polygons of the nuScenes map: drivable area, walkway and ped crossing (IoU of the cells with a coverage above one half, where the map has the feature)"""
    import shapely
    from shapely.geometry import Polygon
    grid = GridSpec()
    xs, ys = grid.centres()
    scene = next(s for s in gt.nusc.scene if s["name"] == name)
    nmap = gt.map_for(scene["token"])
    # records_in_patch (the devkit's scan with a prefilter by bounds, also for lane_connector, which has no devkit bounds) gives the devkit's own records on a few patches
    pose = np.asarray(gt.keyframe(toks[0])["meta"]["T_e2g"], float)[:2, 3]
    for dx, dy, half in ((0, 0, 40), (25, -10, 30), (-60, 40, 55)):
        box = (pose[0] + dx - half, pose[1] + dy - half, pose[0] + dx + half, pose[1] + dy + half)
        layers = ["lane", "lane_connector", "drivable_area", "ped_crossing", "road_divider", "lane_divider"]
        mine = gt.records_in_patch(nmap, box, layers)
        for layer in layers:
            assert mine[layer] == nmap.get_records_in_patch(box, [layer], mode="intersect")[layer], (layer, box)
    res = {}
    for tok in toks:
        g = gt.keyframe(tok)
        T = np.asarray(g["meta"]["T_e2g"], float)
        Tinv = np.linalg.inv(T)
        for layer, table in (("drivable_area", nmap.drivable_area), ("walkway", nmap.walkway), ("ped_crossing", nmap.ped_crossing)):
            exact = np.zeros(xs.shape, bool)
            for rec in table:
                tokens = rec.get("polygon_tokens") or [rec["polygon_token"]] if ("polygon_tokens" in rec or "polygon_token" in rec) else []
                polys = [nmap.extract_polygon(t) for t in tokens]
                for pg in polys:
                    if pg.is_empty:
                        continue
                    ego = lambda coords: (np.c_[np.asarray(coords, float)[:, :2], np.zeros(len(coords)), np.ones(len(coords))] @ Tinv.T)[:, :2]      # noqa: E731
                    poly = Polygon(ego(pg.exterior.coords), [ego(i.coords) for i in pg.interiors])
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    if poly.bounds[2] < 0 or poly.bounds[0] > 60 or poly.bounds[3] < -26 or poly.bounds[1] > 26:
                        continue
                    exact |= shapely.contains_xy(poly, xs.ravel(), ys.ravel()).reshape(xs.shape)
            mine = g[layer] > 0.5
            if exact.sum() >= 50 or mine.sum() >= 50:
                iou = (exact & mine).sum() / max((exact | mine).sum(), 1)
                res.setdefault(layer, []).append(iou)
    for layer, vals in res.items():
        assert min(vals) > 0.9, f"layer {layer}: IoU with the exact map polygons {vals}"
    assert "drivable_area" in res, "the drivable area is in view on the test frames"
    return "map layers against exact polygons: " + ", ".join(f"{k} IoU >= {min(v):.3f}" for k, v in sorted(res.items()))


def _lane_invariants(gt, lane_gt, toks, root):
    """invariants of lane graphs on real frames of every map, both rigs: ids, links, events, window, tiers"""
    n = 0
    for rig in lane_gt.RIGS:
        lgt = lane_gt.LaneGT(root, "v1.0-mini", nusc=gt.nusc, rig=rig)
        win = lane_gt.RIGS[rig]["window"]
        for tok in toks:
            g = lgt.graph(tok)
            ids = [q["id"] for q in g["pieces"]]
            assert len(ids) == len(set(ids)), "piece ids are unique within a frame"
            by = {q["id"]: q for q in g["pieces"]}
            for q in g["pieces"]:
                for j in q["succ"]:
                    assert j in by and q["id"] in by[j]["pred"], "succ and pred are symmetric and exist"
                for j in q["pred"]:
                    assert j in by and q["id"] in by[j]["succ"], "pred and succ are symmetric and exist"
                P = np.asarray(q["pts"], float)
                assert np.isfinite(P).all() and abs(q["length"] - float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())) < 1e-3, "length is the length of the polyline"
                assert ((np.asarray(q["tier_src"]) > 0) == (np.asarray(q["tier"]) == 1)).all() and (~(np.asarray(q["vis"]) & np.asarray(q["occ"]))).all(), "tier_src marks exactly tier 1; vis and occ exclude each other"
            for e in g["events"]:
                for i in list(e["in"]) + list(e["out"]):
                    assert i in by, f"event {e['type']} names a piece that exists"
            w_in = {e["out"][0] for e in g["events"] if e["type"] == "window" and e["io"] == "in"}
            w_out = {e["in"][0] for e in g["events"] if e["type"] == "window" and e["io"] == "out"}
            assert w_in == {q["id"] for q in g["pieces"] if q["start_open"]} and w_out == {q["id"] for q in g["pieces"] if q["end_open"]}, "window events are the open ends"
            for e in g["events"]:
                if e["type"] == "window":
                    p = np.asarray(e["pos"], float)
                    box = win if win is not None else (-g["half"], g["half"], -g["half"], g["half"])
                    assert min(abs(p[0] - box[0]), abs(p[0] - box[1]), abs(p[1] - box[2]), abs(p[1] - box[3])) < 1e-3, "a window event lies on the border"
            for t in (1, 2, 3):
                assert abs(g["tier_info"]["length_m"][t] - 0.5 * sum(int((np.asarray(q["tier"]) == t).sum()) for q in g["pieces"])) < 1e-6, "tier_info adds up"
            n += 1
    return f"lane graph invariants on {n} frames (both rigs, {len(toks)} frames over the maps of the test scenes)"


def _signature_guard():
    """the lane signature hashes SOURCE_FILES: a module they import that is neither hashed nor known to be outside is a hole (an edit of it would change the GT without a new version)"""
    from . import lane_gt
    here = os.path.dirname(os.path.abspath(__file__))
    hashed = {f[:-3] for f in lane_gt.SOURCE_FILES}
    outside = {"nuscenes_adapter", "light_nusc", "store", "pipeline", "cli", "gt_api", "lane_raster", "versions", "sets", "suite",       # pose_matrix, the devkit loader, the storage helpers, the git stamp of the index: no label comes from them
               "grid", "coverage"}                                                                          # the front GT's own hashed files that rig_masks reads (a change there is a new FRONT version)
    for f in lane_gt.SOURCE_FILES:
        tree = ast.parse(open(os.path.join(here, f), encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                mods = [node.module] if node.module else [a.name for a in node.names]
                for mod in mods:
                    assert mod in hashed or mod in outside, f"{f} imports {mod}, which the lane signature does not cover: add it to SOURCE_FILES (a new lane version) or to the known outside modules here with a reason"


def run(root=None, verbose=True):
    from . import lane_gt
    from .suite import GtGenerator
    root = os.path.expanduser(root or "~/data/nuscenes")
    gt = GtGenerator(root, "v1.0-mini", light=True, scenes=["scene-0061", "scene-0103", "scene-0553", "scene-0916"])
    toks = [gt.keyframes("scene-0061")[8], gt.keyframes("scene-0103")[20]]
    _contract_checks()
    msgs = [_front_checks(gt, toks), _map_checks(gt, "scene-0061", toks[:1])]
    lane_toks = []
    for name in ("scene-0061", "scene-0103", "scene-0553", "scene-0916"):
        lane_toks.append(gt.keyframes(name)[10])
    msgs.append(_lane_invariants(gt, lane_gt, lane_toks, root))
    _signature_guard()
    msg = "generator selftest ok: " + "; ".join(msgs) + "; the lane signature covers what the lane sources import"
    if verbose:
        print(msg)
    return msg


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    run(ap.parse_args().root)
