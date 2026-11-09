"""Masks from the area polygons of the lane GT, in the BEV grid, and their packed side cache.

The lane GT stores the road and intersection areas of a keyframe as polygons (`g["areas"]["road" | "intersection"]`, exterior + holes, not dilated), cut to the window of its rig.
For BEV Fusion the rig `bev_fusion` has exactly the window of the BEV grid, so a mask of any area is one function:

    from autogt import lane_raster, lane_gt
    g = lane_gt.load(token, rig="bev_fusion")
    m = lane_raster.area_mask(g, "intersection", border_m=0.5)        # (120, 104) uint8, [ix, iy] like the GT maps: x forward 0..60 m, y -26..26 m (iy = 0 is the RIGHT edge)
    m = lane_raster.load_mask(token, "intersection", border_m=0.5)    # the same from the packed cache (one array lookup), see build_mask_cache

A cell is set when its CENTRE lies inside an area (exact point in polygon, holes left out); `border_m` widens every area by that many metres first (Euclidean, rounded corners: the cell
is set when its centre is within `border_m` of the area): 0 = the polygons as they are, 0.5 = one cell. An area is cut at the window of the rig, so near the edge of the grid
the widening cannot reach beyond it. A raster made with PIL at 4x supersampling (the way the BEV chat first drew it) differs by about 1 % of the cells at the edges of the polygons (PIL fills the low-x / low-y edge
one pixel further): this one is exact.

PACKED CACHE. `build_mask_cache(store_dir)` rasterises every graph of a lane store once and writes, below the store (`<lane store>/masks/<kind>_b<border>.npz`), the masks of all keyframes as
packed bits (1,560 bytes per frame for the front grid: 53 MB for trainval), keyed by the store it was made from (a store that is deleted or replaced takes its masks with it) and by `RASTER_VERSION`, the
hash of this file: a changed rasteriser makes every cache stale, `cache_complete` says so and `build_mask_cache` makes it again. `load_mask(token, kind, border_m)` reads them: a plain (X, Y) uint8
array, deterministic, no polygons at training time. A mask of a graph whose window does not cover the grid is refused (the Spurmitte rig has |x| <= 50 m, the BEV grid reaches 60 m).
The cache is read as a whole (53 MB per border and process): load it before a DataLoader forks, or accept 53 MB per worker.

    python -m autogt.gt_api_selftest       (this module is tested there)
"""
import hashlib
import os
import time

import numpy as np

from .grid import GridSpec

RASTER_VERSION = hashlib.sha1(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:10]       # names the code that made a cache: a change of this file makes the caches stale


def _grid(grid):
    return grid or GridSpec()


def area_mask(g, kind="intersection", border_m=0.0, grid=None):
    """the polygons `g["areas"][kind]` of a lane graph (any frame) as a (nx, ny) uint8 mask in `grid` (default the BEV grid of the front GT, 0.5 m cells): 1 where the cell centre lies inside an area
    (holes are left out) widened by `border_m` metres. Raises ValueError when the window of the graph does not cover the grid, or `kind` is not an area of the graph."""
    import shapely
    from shapely.geometry import Polygon
    from . import lane_graph as LG
    grid = _grid(grid)
    if kind not in g.get("areas", {}):
        raise ValueError(f"the graph has no areas {kind!r} (it has {sorted(g.get('areas', {}))})")
    g = LG.in_frame(g, "ego")
    win = g.get("window") or (-g["half"], g["half"], -g["half"], g["half"])
    if grid.x_min < win[0] - 1e-6 or grid.x_max > win[1] + 1e-6 or grid.y_min < win[2] - 1e-6 or grid.y_max > win[3] + 1e-6:
        raise ValueError(f"the window of the graph x {win[0]:g} .. {win[1]:g}, y {win[2]:g} .. {win[3]:g} m does not cover the grid x {grid.x_min:g} .. {grid.x_max:g}, y {grid.y_min:g} .. {grid.y_max:g} m "
                         "(use the rig bev_fusion for the BEV grid)")
    nx, ny, res = grid.nx, grid.ny, grid.res
    out = np.zeros((nx, ny), np.uint8)
    xs = grid.x_min + (np.arange(nx) + 0.5) * res
    ys = grid.y_min + (np.arange(ny) + 0.5) * res
    for a in g["areas"][kind]:                                      # every area on its own: a hole of one never erases another
        poly = Polygon(np.asarray(a["exterior"], float), [np.asarray(h, float) for h in a["holes"]])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if border_m > 0:
            poly = poly.buffer(float(border_m), quad_segs=8)
        if poly.is_empty:
            continue
        x0, y0, x1, y1 = poly.bounds
        i0, i1 = max(0, int(np.searchsorted(xs, x0, "left"))), min(nx, int(np.searchsorted(xs, x1, "right")))        # only the cells inside the bounding box
        j0, j1 = max(0, int(np.searchsorted(ys, y0, "left"))), min(ny, int(np.searchsorted(ys, y1, "right")))
        if i0 >= i1 or j0 >= j1:
            continue
        X, Y = np.meshgrid(xs[i0:i1], ys[j0:j1], indexing="ij")
        out[i0:i1, j0:j1] |= shapely.contains_xy(poly, X.ravel(), Y.ravel()).reshape(X.shape).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------------------------------------------- the packed cache
def cache_path(store_dir, kind="intersection", border_m=0.0):
    return os.path.join(store_dir, "masks", f"{kind}_b{border_m:g}.npz")


def held_names(store_dir):
    """the tokens with a stored graph in the folder, by file name only (no stat: the stores are written by rename, a half-written file has a .tmp name)"""
    if not os.path.isdir(store_dir):
        return set()
    with os.scandir(store_dir) as it:
        return {e.name[:-7] for e in it if e.name.endswith(".pkl.gz") and ".tmp" not in e.name}


def cache_complete(store_dir, kind="intersection", border_m=0.0, grid=None, held=None):
    """True when the cache file exists, was made by THIS rasteriser (RASTER_VERSION) for this grid and holds exactly the keyframes the store holds now (`held`: the tokens, scanned once by the caller)"""
    p = cache_path(store_dir, kind, border_m)
    if not os.path.exists(p):
        return False
    grid = _grid(grid)
    try:
        with np.load(p) as z:
            if str(z["raster"]) != RASTER_VERSION or tuple(z["shape"]) != (grid.nx, grid.ny) or not np.allclose(z["grid"], [grid.x_min, grid.x_max, grid.y_min, grid.y_max, grid.res]):
                return False
            return set(z["tokens"].tolist()) == (held_names(store_dir) if held is None else held)
    except (OSError, ValueError, KeyError):
        return False


def cache_info(store_dir, kind="intersection", border_m=0.0):
    """{"raster", "data", "frames"} of a cache file, None when it is not there (the key reads the identity of the data it was made from)"""
    p = cache_path(store_dir, kind, border_m)
    try:
        with np.load(p) as z:
            return dict(raster=str(z["raster"]), data=str(z["data"]), frames=int(len(z["tokens"])))
    except (OSError, ValueError, KeyError):
        return None


def build_mask_cache(store_dir, kinds=("intersection",), borders=(0.0, 0.5), grid=None, force=False, verbose=True):
    """rasterise every graph of the lane store `store_dir` once and write one packed cache file per (kind, border) below `<store_dir>/masks/`; a file that is complete (same rasteriser, same keyframes) is left
    alone (`force` makes it again). -> {"files": [paths], "frames": n, "made": [paths written]}"""
    import gzip
    import pickle
    from .gt_api import data_identity
    grid = _grid(grid)
    tokens = sorted(held_names(store_dir))
    todo = [(k, b) for k in kinds for b in borders if force or not cache_complete(store_dir, k, b, grid, set(tokens))]
    files = [cache_path(store_dir, k, b) for k in kinds for b in borders]
    if not todo:
        return {"files": files, "frames": len(tokens), "made": []}
    sig = os.path.basename(os.path.normpath(store_dir))
    nbytes = (grid.nx * grid.ny + 7) // 8
    bits = {kb: np.zeros((len(tokens), nbytes), np.uint8) for kb in todo}
    t0 = time.time()
    for i, tok in enumerate(tokens):
        with gzip.open(os.path.join(store_dir, tok + ".pkl.gz"), "rb") as f:
            g = pickle.load(f)
        for k, b in todo:
            bits[(k, b)][i] = np.packbits(area_mask(g, k, b, grid).ravel())
        if verbose and (i + 1) % 5000 == 0:
            print(f"  masks {i + 1}/{len(tokens)} ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(os.path.join(store_dir, "masks"), exist_ok=True)
    made = []
    ident = data_identity()
    for (k, b), arr in bits.items():
        p = cache_path(store_dir, k, b)
        tmp = p + f".{os.getpid()}.tmp.npz"
        np.savez(tmp, tokens=np.array(tokens, dtype="U32"), bits=arr, shape=np.array([grid.nx, grid.ny]), grid=np.array([grid.x_min, grid.x_max, grid.y_min, grid.y_max, grid.res]),
                 border_m=np.array(b), kind=np.array(k), source=np.array(sig), raster=np.array(RASTER_VERSION), data=np.array(ident))
        os.replace(tmp, p)
        _CACHES.pop(p, None)                                         # a process that loaded the old file reads the new one
        made.append(p)
    if verbose:
        print(f"mask cache of {sig}: {len(tokens)} frames, {len(made)} file(s), {time.time() - t0:.0f} s -> {os.path.join(store_dir, 'masks')}", flush=True)
    return {"files": files, "frames": len(tokens), "made": made}


class MaskCache:
    """one packed cache file: `MaskCache(path).get(token)` -> (nx, ny) uint8; `token in cache`; len(cache)"""

    def __init__(self, path):
        with np.load(path) as z:
            self.tokens = z["tokens"].tolist()
            self.bits = z["bits"]
            self.shape = tuple(int(v) for v in z["shape"])
            self.grid = z["grid"].tolist()
            self.border_m = float(z["border_m"])
            self.kind = str(z["kind"])
            self.source = str(z["source"])
            self.raster = str(z["raster"])
        self._row = {t: i for i, t in enumerate(self.tokens)}

    def __len__(self):
        return len(self.tokens)

    def __contains__(self, token):
        return token in self._row

    def get(self, token):
        if token not in self._row:
            raise KeyError(f"no mask for keyframe {token} in the cache of the lane store {self.source} (kind {self.kind}, border {self.border_m:g} m)")
        n = self.shape[0] * self.shape[1]
        return np.unpackbits(self.bits[self._row[token]])[:n].reshape(self.shape).astype(np.uint8)


_CACHES = {}


def load_mask(token, kind="intersection", border_m=0.0, store_dir=None):
    """the packed mask of a keyframe of the BEV Fusion lane store (default store: the one of THIS source, `lane_gt.signature(rig="bev_fusion")`), (nx, ny) uint8, [ix, iy] of the BEV grid.
    Raises FileNotFoundError when the cache is not there (`python -m autogt generate-bev` or `lane_raster.build_mask_cache(store_dir)` makes it) and ValueError when it was made by another rasteriser"""
    if store_dir is None:
        from . import lane_gt
        store_dir = os.path.join(lane_gt.DATA, lane_gt.signature(rig="bev_fusion"))
    p = cache_path(store_dir, kind, border_m)
    if p not in _CACHES:
        if not os.path.exists(p):
            raise FileNotFoundError(f"no mask cache {p}: make it with `python -m autogt generate-bev` (or lane_raster.build_mask_cache({store_dir!r}))")
        c = MaskCache(p)
        if c.raster != RASTER_VERSION:
            raise ValueError(f"the mask cache {p} was made by another rasteriser ({c.raster}, this one is {RASTER_VERSION}): build_mask_cache makes it again")
        _CACHES[p] = c
    return _CACHES[p].get(token)
