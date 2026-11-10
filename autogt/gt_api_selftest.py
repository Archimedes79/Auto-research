"""Self-test of the facade (gt_api.py) and the area masks (lane_raster.py): a toy polygon for the raster (orientation, hole, nested areas, border against the exact distance, empty areas, refused windows),
the packed cache (round trip, resume, a store that grew, a changed rasteriser, an empty frame), real nuScenes mini frames against a raster made straight from the map and against the front GT's drivable
area, and `generate_bev_gt` / `generate_lane_gt` on one sequence with the key they return (id, completeness, data identity).

    python -m autogt.gt_api_selftest [--root ~/data/nuscenes]      (also run by `python -m autogt selftest`)
"""
import os
import tempfile

import numpy as np

from . import lane_graph as LG
from . import lane_raster as LR
from .grid import GridSpec


def _toy(window=(0.0, 60.0, -26.0, 26.0), frame="ego", half=60.0, areas=True):
    """an intersection square x 20 .. 30 m, y -5 .. 5 m with a hole x 24 .. 26, y -1 .. 1 in the ego frame, and an empty road"""
    sq = np.array([[20, -5], [30, -5], [30, 5], [20, 5], [20, -5]], np.float32)
    hole = np.array([[24, -1], [26, -1], [26, 1], [24, 1], [24, -1]], np.float32)
    g = dict(frame=frame, half=half, pieces=[], events=[], areas=dict(road=[], intersection=[dict(id=1, exterior=sq, holes=[hole], area=96.0)] if areas else []))
    if window is not None:
        g["window"] = window
    return g


def _raster_checks():
    grid = GridSpec()
    g = _toy()
    m = LR.area_mask(g)
    assert m.shape == (120, 104) and m.dtype == np.uint8 and set(np.unique(m)) == {0, 1}, (m.shape, m.dtype)
    assert int(m.sum()) == 20 * 20 - 4 * 4, int(m.sum())                                       # the square minus the hole, cell centres inside (exact)
    ix = lambda x: int(x / grid.res)                                                            # noqa: E731
    iy = lambda y: int((y - grid.y_min) / grid.res)                                              # noqa: E731
    assert m[ix(22.25), iy(0.25)] == 1 and m[ix(25.25), iy(0.25)] == 0 and m[ix(10.25), iy(0.25)] == 0, "[ix, iy]: x forward = row, y left = column, the hole is left out"
    assert m[ix(22.25), iy(4.75)] == 1 and m[ix(22.25), iy(5.25)] == 0 and m[ix(22.25), iy(-4.75)] == 1 and m[ix(22.25), iy(-5.25)] == 0, "y is not flipped: iy = 0 is the RIGHT edge (y = -26)"
    assert m[ix(19.75), iy(0.25)] == 0 and m[ix(20.25), iy(0.25)] == 1 and m[ix(29.75), iy(0.25)] == 1 and m[ix(30.25), iy(0.25)] == 0, "both edges in x are exact (no one-cell bias on the low side)"
    assert not LR.area_mask(g, "road").any(), "an empty area is an empty mask"
    # the border against the exact distance: a cell centre is set when it lies within `border_m` of the area (shapely, with the hole); the rounded corners of the buffer are polygons of 8 segments per quarter: 3 mm
    from shapely.geometry import Point, Polygon
    area = Polygon(g["areas"]["intersection"][0]["exterior"], [g["areas"]["intersection"][0]["holes"][0]])
    for border in (0.5, 1.0, 2.5):
        mask = LR.area_mask(g, border_m=border)
        wrong = 0
        for i in range(ix(14.0), ix(36.0)):
            for j in range(iy(-12.0), iy(12.0)):
                d = area.distance(Point((i + 0.5) * grid.res, grid.y_min + (j + 0.5) * grid.res))
                if abs(d - border) > 0.005 and int(mask[i, j]) != int(d <= border):
                    wrong += 1
        assert wrong == 0, f"border {border} m: {wrong} cells are not at the exact distance"
    assert (LR.area_mask(g, border_m=0.5) >= m).all() and LR.area_mask(g, border_m=1.0).sum() > LR.area_mask(g, border_m=0.5).sum() > m.sum(), "a wider border is a bigger mask"
    assert not LR.area_mask(_toy(areas=False), border_m=0.5).any() and not LR.area_mask(_toy(areas=False)).any(), "no area, no cell: also with a border (the distance transform of an empty raster is a trap)"
    fine = LR.area_mask(g, grid=GridSpec(res=0.25))
    assert fine.shape == (240, 208) and int(fine.sum()) == 4 * 384, int(fine.sum())             # exact: the same area on a finer grid
    # nested areas: a small square inside the HOLE of the first one is its own area and must stay (a hole never erases another area)
    inner = np.array([[24.5, -0.5], [25.5, -0.5], [25.5, 0.5], [24.5, 0.5], [24.5, -0.5]], np.float32)
    nested = _toy()
    nested["areas"]["intersection"].append(dict(id=2, exterior=inner, holes=[], area=1.0))
    mn = LR.area_mask(nested)
    assert int(mn.sum()) == 384 + 4 and mn[ix(25.25), iy(0.25)] == 1 and mn[ix(24.25), iy(0.25)] == 0, "an area inside the hole of another is kept"
    # the same graph in the lane frame (x right, y forward) gives the same mask: the rotation is turned back
    lane = LG.in_frame(g, "lane")
    assert lane["frame"] == "lane" and lane["window"] == (-26.0, 26.0, 0.0, 60.0) and np.array_equal(LR.area_mask(lane), m), "a graph in the lane frame is turned into the ego frame first"
    for bad, why in ((_toy(window=None, frame="lane", half=50.0), "the square of Spurmitte (|x| <= 50 m) does not cover the BEV grid"),
                     (_toy(window=(0.0, 60.0, -20.0, 20.0)), "a window narrower than the grid")):
        try:
            LR.area_mask(bad)
            raise AssertionError(f"must be refused: {why}")
        except ValueError:
            pass
    try:
        LR.area_mask(g, "nothing")
        raise AssertionError("an unknown kind must be refused")
    except ValueError:
        pass
    small = GridSpec(x_min=0.0, x_max=40.0, y_min=-10.0, y_max=10.0)
    assert LR.area_mask(_toy(window=None, frame="lane", half=50.0), grid=small).shape == (80, 40), "a grid inside the window of the Spurmitte rig is fine"


def _cache_checks():
    from . import lane_gt
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "toystore")
        os.makedirs(store)
        toks = ["a" * 32, "b" * 32, "c" * 32]
        graphs = {}
        for i, t in enumerate(toks):
            g = _toy()
            g["areas"]["intersection"][0]["exterior"] = g["areas"]["intersection"][0]["exterior"] + np.array([float(i), 0.0], np.float32)       # a different square per frame
            graphs[t] = g
            lane_gt._save(store, t, g)
        empty = "e" * 32
        graphs[empty] = _toy(areas=False)                                                           # a frame without any intersection
        lane_gt._save(store, empty, graphs[empty])
        assert not LR.cache_complete(store)
        r = LR.build_mask_cache(store, borders=(0.0, 0.5), verbose=False)
        assert len(r["made"]) == 2 and r["frames"] == 4 and all(os.path.exists(p) for p in r["files"]), r
        assert LR.cache_complete(store) and LR.cache_complete(store, border_m=0.5) and not LR.cache_complete(store, "road", 0.0), "one file per (kind, border)"
        assert os.listdir(store).count("masks") == 1 and lane_gt.held(store) == set(graphs), "the masks folder is not a keyframe of the store"
        for t, g in graphs.items():
            assert np.array_equal(LR.load_mask(t, "intersection", 0.0, store_dir=store), LR.area_mask(g)), "the packed mask is the raster"
            assert np.array_equal(LR.load_mask(t, "intersection", 0.5, store_dir=store), LR.area_mask(g, border_m=0.5))
        assert not LR.load_mask(empty, "intersection", 0.5, store_dir=store).any(), "an empty frame stays empty at border 0.5 in the cache too"
        info = LR.cache_info(store)
        assert info["raster"] == LR.RASTER_VERSION and info["frames"] == 4 and len(info["data"]) >= 7, info
        cache = LR.MaskCache(LR.cache_path(store))
        assert len(cache) == 4 and toks[0] in cache and "z" * 32 not in cache and cache.get(toks[1]).dtype == np.uint8
        try:
            cache.get("z" * 32)
            raise AssertionError("an unknown token must raise")
        except KeyError:
            pass
        try:
            LR.load_mask(toks[0], "road", 0.0, store_dir=store)
            raise AssertionError("a missing cache file must raise")
        except FileNotFoundError:
            pass
        mtime = os.path.getmtime(LR.cache_path(store))
        assert LR.build_mask_cache(store, borders=(0.0, 0.5), verbose=False)["made"] == [] and os.path.getmtime(LR.cache_path(store)) == mtime, "a complete cache is left alone"
        # a store that grew: the cache is stale, built again, and a process that had loaded the old file reads the new one (no clearing of the module's memory)
        new = "d" * 32
        lane_gt._save(store, new, _toy())
        assert not LR.cache_complete(store)
        try:
            LR.load_mask(new, "intersection", 0.0, store_dir=store)
            raise AssertionError("the old cache does not know the new keyframe")
        except KeyError:
            pass
        assert len(LR.build_mask_cache(store, borders=(0.0, 0.5), verbose=False)["made"]) == 2 and LR.cache_complete(store), "a store that grew gets a new cache"
        assert np.array_equal(LR.load_mask(new, "intersection", 0.0, store_dir=store), LR.area_mask(_toy())), "load_mask reads the new file, not the one it loaded before"
        # another rasteriser (RASTER_VERSION, a constant raised by hand when the rasteriser changes) makes every cache stale
        real = LR.RASTER_VERSION
        try:
            LR.RASTER_VERSION = "0" * 10
            assert not LR.cache_complete(store), "a cache of another rasteriser is not complete"
            LR._CACHES.pop(LR.cache_path(store), None)
            try:
                LR.load_mask(new, "intersection", 0.0, store_dir=store)
                raise AssertionError("a cache of another rasteriser must not be read")
            except ValueError:
                pass
            assert len(LR.build_mask_cache(store, borders=(0.0, 0.5), verbose=False)["made"]) == 2 and LR.cache_info(store)["raster"] == "0" * 10, "made again by the new rasteriser"
        finally:
            LR.RASTER_VERSION = real
        assert LR.build_mask_cache(store, borders=(0.0, 0.5), force=True, verbose=False)["made"], "force makes it again"


def _mini_checks(root, GtGenerator, lane_gt):
    """the mask of the stored polygons against a raster made straight from the nuScenes map (road_segment polygons with is_intersection: the way the BEV chat first drew it, PIL at 4x supersampling,
    which has a bias of about 1 % of the cells at the polygon edges; this mask is exact)"""
    from PIL import Image, ImageDraw
    gt = GtGenerator(root, "v1.0-mini", light=True, scenes=["scene-0061", "scene-0103"])
    lgt = lane_gt.LaneGT(root, "v1.0-mini", nusc=gt.nusc, rig="bev_fusion")
    grid = GridSpec()
    import shapely
    import shapely.geometry
    inter = union = n_in = n = ex_i = ex_u = 0
    for name in ("scene-0061", "scene-0103"):
        scene = next(s for s in gt.nusc.scene if s["name"] == name)
        nmap = gt.map_for(scene["token"])
        polys = [nmap.extract_polygon(r["polygon_token"]) for r in nmap.road_segment if r.get("is_intersection")]
        for tok in gt.keyframes(name)[::6]:
            T = lgt.pose(tok)
            Tinv = np.linalg.inv(T)
            img = Image.new("L", (grid.ny * 4, grid.nx * 4), 0)
            dr = ImageDraw.Draw(img)

            def px(coords):
                c = np.asarray(coords, float)[:, :2]
                e = np.c_[c, np.zeros(len(c)), np.ones(len(c))] @ Tinv.T
                return [(float((y - grid.y_min) / grid.res * 4), float((x - grid.x_min) / grid.res * 4)) for x, y in zip(e[:, 0], e[:, 1])]
            for pg in polys:
                for part in ([pg] if pg.geom_type == "Polygon" else list(pg.geoms)):
                    dr.polygon(px(part.exterior.coords), fill=1)
                for part in ([pg] if pg.geom_type == "Polygon" else list(pg.geoms)):
                    for it in part.interiors:
                        dr.polygon(px(it.coords), fill=0)
            direct = np.asarray(img, bool)[2::4, 2::4][:grid.nx, :grid.ny]
            mine = LR.area_mask(lgt.graph(tok)).astype(bool)
            # the exact reference, from the map polygons and not from the stored graph: a cell is set when its centre is inside a road_segment polygon with is_intersection
            exact = np.zeros_like(mine)
            xs = grid.x_min + (np.arange(grid.nx) + 0.5) * grid.res
            ys = grid.y_min + (np.arange(grid.ny) + 0.5) * grid.res
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            for pg in polys:
                for part in ([pg] if pg.geom_type == "Polygon" else list(pg.geoms)):
                    ego = lambda coords: (np.c_[np.asarray(coords, float)[:, :2], np.zeros(len(coords)), np.ones(len(coords))] @ Tinv.T)[:, :2]      # noqa: E731
                    poly = shapely.geometry.Polygon(ego(part.exterior.coords), [ego(it.coords) for it in part.interiors])
                    exact |= shapely.contains_xy(poly, X.ravel(), Y.ravel()).reshape(X.shape)
            inter += int((direct & mine).sum())
            union += int((direct | mine).sum())
            n_in += int(mine.sum())
            ex_i += int((exact & mine).sum())
            ex_u += int((exact | mine).sum())
            n += 1
    assert n >= 10 and n_in > 500, (n, n_in)
    assert ex_i / max(ex_u, 1) >= 0.995, f"the mask of the stored polygons against the exact raster of the map polygons: IoU {ex_i / max(ex_u, 1):.4f} ({ex_i} of {ex_u} cells)"
    assert inter / max(union, 1) >= 0.97, "and against the PIL raster of the BEV chat (a 1 % bias at the edges of that one)"
    return (f"{n} mini frames: the mask of the stored polygons agrees with an exact raster of the map polygons (IoU {ex_i / ex_u:.4f}, {n_in} cells) and with the PIL raster the BEV chat first used "
            f"(IoU {inter / union:.4f})")


def _api_checks(root, lane_gt):
    from . import gt_api
    from .suite import GtGenerator, scene_tokens
    with tempfile.TemporaryDirectory() as d:
        toks = scene_tokens("scene-0061", "v1.0-mini", root)
        key = GtGenerator.generate_bev_gt(root, scenes=["scene-0061"], out=d, verbose=False)
        assert key.system == "bev_fusion" and [p.name for p in key.parts] == ["front", "lane", "masks"] and key.complete, str(key)
        assert key.data == gt_api.data_identity() and key.dataset == "v1.0-mini" and key.set_name == "custom" and len(key.id) == 12, (key.data, key.dataset, key.set_name)
        assert key.part("front").signature != key.part("lane").signature and key.part("lane").store == lane_gt.signature(rig="bev_fusion") and key.part("masks").store == key.part("lane").store
        assert "bev_fusion" in str(key) and key.part("front").signature in str(key) and "complete" in str(key) and key.as_dict()["id"] == key.id
        # the key is read from the stores: the same id without generating, whatever function made it, a partial key has the id of the whole
        again = gt_api.key_of("bev_fusion", "v1.0-mini", "custom", d, dataroot=root, scenes=["scene-0061"])
        assert again.id == key.id and again.complete and gt_api.key_of("bev_fusion", "v1.0-mini", "custom", d, scenes=["scene-0061"]).complete, "the key is read from the stores"
        lane_only = gt_api.generate_lane_gt(root, rig="bev_fusion", scenes=["scene-0061"], out=d, verbose=False)
        assert [p.name for p in lane_only.parts] == ["lane"] and lane_only.id == key.id, "generate_lane_gt(bev_fusion) names the same GT as generate_bev_gt: the id covers all parts"
        # a thinned store is not complete whatever its index says: a zero-byte keyframe (a full disk, a bad copy) is found by the deep listing, a lost file by the count; deep=False judges by name only
        ask = lambda **k: gt_api.gt_key("bev_fusion", "v1.0-mini", None, d, None, ["scene-0061"], **k)
        fpath = os.path.join(key.part("front").path, toks[2] + ".npz")
        keep = open(fpath, "rb").read()
        open(fpath, "wb").close()
        assert not ask().complete and not ask().part("front").complete and ask().part("lane").complete, "a zero-byte front keyframe: the front part is incomplete"
        assert ask(deep=False).complete, "by name only the zero-byte file is not looked for"
        open(fpath, "wb").write(keep)
        lpath = os.path.join(key.part("lane").path, toks[2] + ".pkl.gz")
        os.rename(lpath, lpath + ".gone")
        assert not LR.cache_complete(key.part("masks").path), "the cache holds a frame that the store no longer has: not complete (a store that shrank)"
        assert not ask(deep=False).part("lane").complete and not ask(deep=False).part("masks").complete and not ask().complete, "a lost lane file: the lane part and the masks are incomplete, by name too"
        os.rename(lpath + ".gone", lpath)
        assert ask().complete and ask().id == key.id, "restored: complete again, the same id"
        cpath = LR.cache_path(key.part("masks").path, "intersection", 0.5)
        os.rename(cpath, cpath + ".gone")
        assert ask().part("lane").complete and not ask().part("masks").complete and not ask().complete, "a lost mask cache: only the masks part is incomplete"
        os.rename(cpath + ".gone", cpath)
        assert ask().complete, "the cache is back"
        # the two picture-like commands do what they say: the mask command agrees with the packed cache, a bad border or a bad pick is a ValueError, the lane graph picture is written
        am = GtGenerator.area_mask(toks[2], "intersection", 0.5, png=os.path.join(d, "m.png"), out=d)
        assert am["shape"] == [120, 104] and am["cells"] == int(key.mask(toks[2], "intersection", 0.5).sum()) > 0 and os.path.getsize(os.path.join(d, "m.png")) > 100, am
        for bad_border in (float("nan"), -1.0):
            try:
                GtGenerator.area_mask(toks[2], border_m=bad_border, out=d)
                raise AssertionError(f"border {bad_border} must be refused")
            except ValueError:
                pass
        lg = GtGenerator.lanegraph(os.path.join(d, "lg.png"), "scene-0061:8", dataroot=root)
        assert os.path.getsize(lg["out"]) > 1000 and "lane" in lg["text"].lower(), lg
        try:
            GtGenerator.lanegraph(os.path.join(d, "no_such_dir", "x.png"), "scene-0061:8", dataroot=root)
            raise AssertionError("a picture that cannot be written must be an error, not a success")
        except ValueError:
            pass
        for pick in ("scene-0061:999", "scene-9999:1"):
            try:
                GtGenerator.lanegraph(os.path.join(d, "lg2.png"), pick, dataroot=root)
                raise AssertionError(f"{pick} must be refused")
            except ValueError:
                pass
        # the data identity: the stores that recorded one must agree, else the key says "mixed"; a lane store records it after a run (its writer is a hashed file that does not)
        lane_idx = os.path.join(key.part("lane").path, "index.json")
        import json
        saved = json.load(open(lane_idx))
        assert saved.get("data_identity") == gt_api.data_identity(), "make_lane_gt records the identity in the lane index"
        json.dump({**saved, "data_identity": "bbbbbbbbbbbb"}, open(lane_idx, "w"))
        assert ask().data == "mixed" and ask().id != key.id, "the lane store and the front store name different data: mixed"
        json.dump({k: v for k, v in saved.items() if k != "data_identity"}, open(lane_idx, "w"))
        assert ask().data == gt_api.data_identity() and ask().id == key.id, "a lane store without a record: the front store's identity decides"
        json.dump(saved, open(lane_idx, "w"))
        # what the recorder does with a lane store: none recorded + nothing new written = today's; none recorded + frames written = still none (the old frames have no known identity); another one = mixed
        with tempfile.TemporaryDirectory() as t_:
            f_ = os.path.join(t_, "index.json")
            real_ident = gt_api.data_identity
            try:
                gt_api.data_identity = lambda: "aaaaaaaaaaaa"
                for was, written, want in ((None, 0, "aaaaaaaaaaaa"), (None, 5, None), ("aaaaaaaaaaaa", 5, "aaaaaaaaaaaa"), ("cccccccccccc", 5, "mixed"), ("mixed", 0, "mixed")):
                    json.dump({"signature": "x", **({"data_identity": was} if was else {})}, open(f_, "w"))
                    gt_api._record_lane_identity(t_, was, written)
                    assert json.load(open(f_)).get("data_identity") == (want if want else None), (was, written, want, json.load(open(f_)))
                # the lane writer wrote its index again (a clean tree upgrades it) and dropped the record: the recorder puts it back, it does not trust `was`
                json.dump({"signature": "x"}, open(f_, "w"))
                gt_api._record_lane_identity(t_, "aaaaaaaaaaaa", 0)
                assert json.load(open(f_)).get("data_identity") == "aaaaaaaaaaaa", "the index lost the record: written again"
            finally:
                gt_api.data_identity = real_ident
        # complete is never a guess: a store of one sequence is not the set `full`, a set is judged by what the index says it holds, the json tables (dataroot) say it exactly
        assert not gt_api.key_of("bev_fusion", "v1.0-mini", "full", d).complete and not gt_api.key_of("bev_fusion", "v1.0-mini", "small", d).complete, "one sequence is neither full nor small"
        assert not gt_api.key_of("bev_fusion", "v1.0-mini", None, d).complete, "no set and no scenes: nothing to judge by, incomplete"
        small = gt_api.key_of("bev_fusion", "v1.0-mini", "small", d, dataroot=root)
        assert not small.complete and small.id == key.id and "INCOMPLETE" in str(small), "the id names the code and the data, completeness is a separate fact"
        assert not gt_api.key_of("bev_fusion", "v1.0-mini", "custom", os.path.join(d, "nothing"), scenes=["scene-0061"]).complete, "no store, not complete"
        # the id changes with the code and not with the data identity of the day (the store recorded it); the key says where its data came from
        real_code, real_data = lane_gt.code_signature, gt_api.data_identity
        try:
            lane_gt.code_signature = lambda **p: "f" * 10
            assert gt_api.key_of("bev_fusion", "v1.0-mini", "custom", d, scenes=["scene-0061"]).id != key.id, "another code signature, another id"
        finally:
            lane_gt.code_signature = real_code
        try:
            gt_api.data_identity = lambda: "unknown"
            again = gt_api.key_of("bev_fusion", "v1.0-mini", "custom", d, scenes=["scene-0061"])
            assert again.data == key.data and again.id == key.id, "the data identity is read from the stores (front index, mask cache), not from the manifest of the day"
            assert "data_identity" in open(os.path.join(d, key.part("front").store, "index.json")).read()
        finally:
            gt_api.data_identity = real_data
        g = key.load(toks[8])
        assert g["drivable_area"].shape == (120, 104) and key.lane(toks[8])["frame"] == "ego" and key.lane(toks[8])["rig"] == "bev_fusion"
        assert key.mask(toks[8]).shape == (120, 104) and np.array_equal(key.mask(toks[8], "intersection", 0.5), LR.area_mask(key.lane(toks[8]), border_m=0.5)), "key.mask = the packed raster of the stored graph"
        # the orientation against the front GT, independent of the code under test: the road area of the lane GT is the drivable area of the front GT
        road = LR.area_mask(key.lane(toks[8]), "road").astype(bool)
        drv = g["drivable_area"] > 0.5
        flip = lambda a: a[:, ::-1]                                                                   # noqa: E731
        assert (road & drv).sum() / (road | drv).sum() > 0.9 and (flip(road) & drv).sum() / (flip(road) | drv).sum() < 0.8, "the mask lies where the front GT's drivable area lies, not mirrored"
        assert os.path.isdir(os.path.join(d, "lane", key.part("lane").store, "masks"))
        # the audit of a store: a good one is ok, a missing file and a damaged graph are problems
        import gzip
        import pickle
        import shutil
        from . import store_audit
        audit = store_audit.audit_lane(key.part("lane").path, "bev_fusion", root, "v1.0-mini", ["scene-0061"], verbose=False)
        assert audit["ok"] and audit["stats"]["frames"] == len(toks) and audit["stats"]["tier1_mean_m"] > 0, audit["lines"]
        bad = os.path.join(d, "audited_copy")
        shutil.copytree(key.part("lane").path, bad)
        os.remove(os.path.join(bad, toks[3] + ".pkl.gz"))
        used = {toks[3]}                                                         # the lost file

        def damage(pred, how):
            """damage the first frame that has what `how` needs (`pred`): -> its token"""
            for tok in toks:
                if tok in used:
                    continue
                fn = os.path.join(bad, tok + ".pkl.gz")
                with gzip.open(fn, "rb") as f:
                    g = pickle.load(f)
                if pred(g):
                    how(g)
                    with gzip.open(fn, "wb") as f:
                        pickle.dump(g, f)
                    used.add(tok)
                    return tok
            raise AssertionError("no frame of the test store has what this damage needs")

        def piece_with(cond):
            return lambda g: any(cond(q) for q in g["pieces"])

        def first(cond, g):
            return next(q for q in g["pieces"] if cond(q))

        def line_of(g):
            return next((it for items in g["lines"].values() for it in items), None)

        def area_of(g):
            return next((p_ for polys in g["areas"].values() for p_ in polys), None)
        # one damage per check of the audit, each named by its frame and its message; the zero-byte file and the foreign file / incomplete index are store-level
        cases = [
            (piece_with(lambda q: (q["tier"] > 1).any()), lambda g: first(lambda q: (q["tier"] > 1).any(), g)["tier"].__setitem__(int(np.argmax(first(lambda q: (q["tier"] > 1).any(), g)["tier"] > 1)), 0), ": tier (", "a tier that is none of 1, 2, 3"),
            (lambda g: len(g["pieces"]) > 0, lambda g: (g["pieces"][0]["vis"].__setitem__(0, True), g["pieces"][0]["occ"].__setitem__(0, True)), "vis and occ", "visible and occluded at once"),
            (lambda g: len(g["pieces"]) > 0, lambda g: g["pieces"][0]["pts"].__setitem__(0, (999.0, 999.0)), "outside the window", "a point outside the window"),
            (lambda g: True, lambda g: g.pop("ego_path"), "key ego_path missing", "a key missing"),
            (lambda g: True, lambda g: g.__setitem__("sample", "not-the-token"), "sample not-the-token", "the sample token is not the file's"),
            (lambda g: True, lambda g: g.__setitem__("signature", "0000000000"), "signature 0000000000", "the signature is not the store's"),
            (lambda g: "window" in g, lambda g: g.__setitem__("window", (0.0, 1.0, 0.0, 1.0)), "window ", "the window of the rig"),
            (piece_with(lambda q: (q["tier"] == 1).any()), lambda g: first(lambda q: (q["tier"] == 1).any(), g)["tier_src"].__setitem__(int(np.argmax(first(lambda q: (q["tier"] == 1).any(), g)["tier"] == 1)), 0), "tier_src > 0 must be exactly", "tier_src and tier 1 disagree"),
            (lambda g: len(g["pieces"]) > 0, lambda g: g["tier_info"]["length_m"].__setitem__(1, g["tier_info"]["length_m"][1] + 5.0), "tier_info tier 1", "tier_info against the pieces"),
            (lambda g: True, lambda g: g["ego_path"].__setitem__("ahead_m", -1.0), "ego_path", "the ego path"),
            (lambda g: line_of(g) is not None, lambda g: line_of(g)["pts"].__setitem__(0, (999.0, 999.0)), "outside the window", "a line outside the window"),
            (lambda g: area_of(g) is not None, lambda g: area_of(g)["exterior"].__setitem__(0, (999.0, 999.0)), "outside the window", "an area outside the window"),
            (lambda g: len(g["events"]) > 0, lambda g: g["events"][0].__setitem__("tier", 9), "event", "an event with a tier that is none")]
        damaged = {}
        for pred, how, word, why in cases:
            damaged[damage(pred, how)] = (word, why)
        zero = next(t for t in toks if t not in used)
        open(os.path.join(bad, zero + ".pkl.gz"), "wb").close()                  # a zero-byte file: a problem of this token (the audit used to end in an EOFError)
        used.add(zero)
        worse = store_audit.audit_lane(bad, "bev_fusion", root, "v1.0-mini", ["scene-0061"], verbose=False)
        assert not worse["ok"] and any(t == "store" for t, _ in worse["problems"]), worse["problems"]
        said = dict(worse["problems"])
        assert set(said) == {"store", *damaged, zero}, ("each damaged frame is named, no good one is", sorted(set(said) ^ {"store", *damaged, zero}))
        for tok, (word, why) in damaged.items():
            assert word in said[tok], (why, word, said[tok])
        assert "unreadable" in said[zero], said[zero]
        # the store-level checks: a foreign file, an index that does not say complete
        os.remove(os.path.join(bad, zero + ".pkl.gz"))
        for tok in damaged:
            os.remove(os.path.join(bad, tok + ".pkl.gz"))
        shutil.copy(os.path.join(key.part("lane").path, toks[3] + ".pkl.gz"), os.path.join(bad, toks[3] + ".pkl.gz"))          # the lost file is back
        for tok in list(damaged) + [zero]:
            shutil.copy(os.path.join(key.part("lane").path, tok + ".pkl.gz"), os.path.join(bad, tok + ".pkl.gz"))
        assert store_audit.audit_lane(bad, "bev_fusion", root, "v1.0-mini", ["scene-0061"], verbose=False)["ok"], "repaired: ok again"
        open(os.path.join(bad, "stray.txt"), "w").write("x")
        assert any("other files" in m for _, m in store_audit.audit_lane(bad, "bev_fusion", root, "v1.0-mini", ["scene-0061"], verbose=False)["problems"]), "a foreign file in the store"
        os.remove(os.path.join(bad, "stray.txt"))
        import json as json_
        idx_ = json_.load(open(os.path.join(bad, "index.json")))
        json_.dump({**idx_, "complete": False}, open(os.path.join(bad, "index.json"), "w"))
        assert any("does not say complete" in m for _, m in store_audit.audit_lane(bad, "bev_fusion", root, "v1.0-mini", ["scene-0061"], verbose=False)["problems"]), "an index that does not say complete"
        lc = GtGenerator.generate_lane_gt(root, rig="lane_center", scenes=["scene-0061"], out=d, verbose=False)
        assert lc.system == "lane_center" and [p.name for p in lc.parts] == ["lane"] and lc.complete and lc.id != key.id and lc.lane(toks[8])["frame"] == "lane"
        assert lc.part("lane").version == lane_gt.version_of(lc.part("lane").store), "the version is the one of the store"
        both = gt_api.generate_lane_gt(root, rig="all", scenes=["scene-0061"], out=d, verbose=False)
        assert set(both) == {"lane_center", "bev_fusion"} and both["lane_center"].id == lc.id and both["bev_fusion"].id == key.id, both
        for bad in (dict(stride=2), dict(borders=())):
            try:
                gt_api.generate_bev_gt(root, scenes=["scene-0061"], out=d, verbose=False, **bad)
                raise AssertionError(f"{bad} must be refused")
            except ValueError:
                pass
        try:
            gt_api.key_of("surround")
            raise AssertionError("an unknown system must be refused")
        except ValueError:
            pass
    return f"key {key.id}: {key}"[:230]


def run(root=None, verbose=True):
    from . import lane_gt
    from .suite import GtGenerator
    root = os.path.expanduser(root or "~/data/nuscenes")
    _raster_checks()
    _cache_checks()
    msg_mini = _mini_checks(root, GtGenerator, lane_gt)
    msg_key = _api_checks(root, lane_gt)
    msg = f"gt_api / lane_raster selftest ok: toy raster (orientation, exact edges, hole, nested areas, exact border, empty areas, refused windows), packed cache (round trip, resume, growth, rasteriser, empty frame); {msg_mini}; generate_bev_gt / generate_lane_gt on scene-0061, {msg_key}; the store audit finds a missing file, a tier that is none, vis and occ at one point and a point outside the window"
    if verbose:
        print(msg)
    return msg


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    run(ap.parse_args().root)
