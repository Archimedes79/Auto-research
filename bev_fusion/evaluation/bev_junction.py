"""Junction-free scoring rows (D66, card t26): the lane-divider and road-divider IoU with the map's intersection polygons left out.

The map has no divider labels inside the polygons it flags as intersection, the network draws the paint it sees there, and the metric counts it as false (D66: 50 % of the false
cells lie inside intersections, 1.5 % of the label cells do). `bev_suite.py ... --junction-free` scores the two divider layers a second time next to the unchanged headline rows:
the intersection cells leave the prediction, the GT and the scored cells alike BEFORE the tolerance dilation (the "both" variant of records/intersection_iou_result.txt, which
this reproduces). Two masks, from the GtGenerator area masks (`key.mask(token, "intersection", border_m)`, packed cache of the lane store of the rig `bev_fusion`):
    fovj    the polygons as they are (border 0)       -> flat rows `seg_fovj/seg/dt+0.00/lane_divider/all/all/iou@0.5` ...
    fovjb   the polygons widened by 0.5 m (one cell)   -> `seg_fovjb/...`
Only the two mask layers `lane_divider` and `road_divider` get these rows (areas have no such labels; centre-lines run through junctions on purpose). A row of the default evaluation
is never touched: without the option nothing here is called.
Orientation: the mask is (120, 104) uint8 `[ix, iy]` like the GT maps (x forward 0 .. 60 m, y left, iy = 0 the RIGHT edge); a GT frame's token is `g["meta"]["sample"]`, so the frames at
dt = -0.5 / +0.5 get their own mask.
This file is part of the evaluator hash (`bev_adapter.NUMBER_PATHS`): only what the numbers depend on lives here. The self-test and the reproduction on real frames are in
`bev_junction_check.py` (not hashed).
"""
import os

import numpy as np

JUNCTION_LAYERS = ("lane_divider", "road_divider")
MASKS = {"fovj": 0.0, "fovjb": 0.5}               # scored mask name (seg_<name>) -> border of the intersection mask in metres
HEADLINE = tuple(f"{m}/dt+0.00/{layer}/all/all/iou@{t}" for m in MASKS for layer in JUNCTION_LAYERS for t in ("0.5", "1"))   # the rows the paired bootstrap of `compare` reads
RASTER = "1"                                      # the GtGenerator rasteriser version (lane_raster.RASTER_VERSION) the rows were validated with (bev_junction_check.py --reproduce)


def junction_free(pred, gt, fov, inter):
    """the three arguments of SegMeter.add with the intersection cells left out: prediction and GT of the mask layers set to 0 there (so that the tolerance dilation cannot carry
    them out of the polygon), and the scored cells `fov` without them. pred / gt: {layer: (X, Y)}, fov / inter: (X, Y) bool. -> (pred, gt, fov) of the layers in JUNCTION_LAYERS"""
    keep = ~np.asarray(inter, bool)
    cut = lambda d: {n: np.where(keep, np.asarray(d[n]), 0) for n in JUNCTION_LAYERS if n in d}      # noqa: E731
    return cut(pred), cut(gt), np.asarray(fov, bool) & keep


class JunctionMasks:
    """the intersection masks of a GT frame, `masks(g)` -> {mask name: (X, Y) bool}; read from the packed cache of the lane store (default: the store of the current GtGenerator for
    the rig bev_fusion). Made BEFORE the network is loaded: both cache files (every border of MASKS) must exist and be made by the rasteriser the rows were validated with (RASTER),
    else it fails here with the way out; a keyframe the cache does not hold is an error at its frame, never a frame scored without the rows"""

    def __init__(self, store_dir=None):
        from GtGenerator import lane_gt, lane_raster
        self.raster = lane_raster
        self.store = store_dir or os.path.join(lane_gt.DATA, lane_gt.signature(rig="bev_fusion"))
        self.info = {}
        for name, border in MASKS.items():
            info = lane_raster.cache_info(self.store, "intersection", border)
            if info is None:
                raise FileNotFoundError(f"--junction-free: no mask cache {lane_raster.cache_path(self.store, 'intersection', border)} (border {border:g} m): make it with `python -m GtGenerator generate-bev` "
                                        f"(or lane_raster.build_mask_cache({self.store!r}))")
            if info["raster"] != RASTER or lane_raster.RASTER_VERSION != RASTER:
                raise ValueError(f"--junction-free: the mask cache of {os.path.basename(self.store)} (border {border:g} m) was made by rasteriser {info['raster']}, GtGenerator is at "
                                 f"{lane_raster.RASTER_VERSION}, the rows were validated with {RASTER}: check `python evaluation/bev_junction_check.py --reproduce`, then set bev_junction.RASTER")
            self.info[name] = info

    def __call__(self, g):
        token = g["meta"]["sample"]
        return {name: self.raster.load_mask(token, "intersection", b, store_dir=self.store).astype(bool) for name, b in MASKS.items()}

    def describe(self):
        """the mask id the rows were made with: the lane store, the rasteriser version, the data the cache was made from (per border), the borders. It is part of the Source's description
        (bev_suite.make_suite), so the store's identity rule refuses a resume, and `compare` refuses two stores, whose masks differ"""
        return {"lane_store": os.path.basename(os.path.normpath(self.store)), "raster": RASTER, "borders_m": dict(MASKS), "cache_data": {n: i["data"] for n, i in self.info.items()}}
