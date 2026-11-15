"""Masks from the area polygons of the lane GT, in the BEV grid, and their packed side cache.

The lane GT stores the road and intersection areas of a keyframe as polygons (`g["areas"]["road" | "intersection"]`, exterior + holes, not dilated), cut to the window of its rig.
For BEV Fusion the rig `bev_fusion` has exactly the window of the BEV grid, so a mask of any area is one function:

    from GtGenerator import lane_raster, lane_gt
    g = lane_gt.load(token, rig="bev_fusion")
    m = lane_raster.area_mask(g, "intersection", border_m=0.5)        # (120, 104) uint8, [ix, iy] like the GT maps: x forward 0..60 m, y -26..26 m (iy = 0 is the RIGHT edge)
    m = lane_raster.load_mask(token, "intersection", border_m=0.5)    # the same from the packed cache (one array lookup), see build_mask_cache

A cell is set when its CENTRE lies inside an area (exact point in polygon, holes left out); `border_m` widens every area by that many metres first (Euclidean, rounded corners: the cell
is set when its centre is within `border_m` of the area): 0 = the polygons as they are, 0.5 = one cell. An area is cut at the window of the rig, so near the edge of the grid
the widening cannot reach beyond it. A raster made with PIL at 4x supersampling (the way the BEV chat first drew it) differs by about 1.6 % of the cells at the edges of the polygons (PIL fills the
low-x / low-y edge one pixel further): this one is exact.

PACKED CACHE. `build_mask_cache(store_dir)` rasterises every graph of a lane store once and writes, below the store (`<lane store>/masks/<kind>_b<border>.npz`), the masks of all keyframes as
packed bits (1,560 bytes per frame: 53 MB for trainval per border), keyed by the store it was made from (a store that is deleted or replaced takes its masks with it) and by `RASTER_VERSION`:
a change of the rasteriser that changes a mask bumps that constant, the caches of the old one are stale (`cache_complete` says so, `build_mask_cache` makes them again, `load_mask` refuses them).
`load_mask(token, kind, border_m)` reads one: a plain (X, Y) uint8 array, deterministic, no polygons at training time. A mask of a graph whose window does not cover the grid is refused
(the Spurmitte rig has |x| <= 50 m, the BEV grid reaches 60 m). The cache is read as a whole (53 MB per border and process): load it before a DataLoader forks, or accept 53 MB per worker.
`workers` > 1 builds with a pool of processes (34k frames: about 11 min in one process).

    python -m GtGenerator.gt_api_selftest       (this module is tested there)
"""
import gzip
import multiprocessing as mp
import os
import pickle
import time
import zlib

import numpy as np

from .grid import GridSpec
from .store import names as store_names

RASTER_VERSION = "1"            # bump when area_mask gives other cells (the selftest pins the semantics: exact edges, holes, borders, orientation)
GRID = GridSpec()               # the BEV grid of the front GT: the cache is made for it


def area_masks(g, kind="intersection", borders=(0.0,), grid=None):
    """the polygons `g["areas"][kind]` of a lane graph (any frame) as masks in `grid` (default the BEV grid, 0.5 m cells): {border: (nx, ny) uint8}, 1 where the cell centre lies inside an area
    (holes are left out) widened by `border` metres. Raises ValueError when the window of the graph does not cover the grid, or `kind` is not an area of the graph."""
    import shapely
    from shapely.geometry import Polygon
    from . import lane_graph as LG
    grid = grid or GRID
    if kind not in g.get("areas", {}):
        raise ValueError(f"the graph has no areas {kind!r} (it has {sorted(g.get('areas', {}))})")
    g = LG.in_frame(g, "ego")
    win = g.get("window") or (-g["half"], g["half"], -g["half"], g["half"])
    if grid.x_min < win[0] - 1e-6 or grid.x_max > win[1] + 1e-6 or grid.y_min < win[2] - 1e-6 or grid.y_max > win[3] + 1e-6:
        raise ValueError(f"the window of the graph x {win[0]:g} .. {win[1]:g}, y {win[2]:g} .. {win[3]:g} m does not cover the grid x {grid.x_min:g} .. {grid.x_max:g}, y {grid.y_min:g} .. {grid.y_max:g} m "
                         "(use the rig bev_fusion for the BEV grid)")
    nx, ny, res = grid.nx, grid.ny, grid.res
    xs = grid.x_min + (np.arange(nx) + 0.5) * res
    ys = grid.y_min + (np.arange(ny) + 0.5) * res
    polys = []
    for a in g["areas"][kind]:                                      # every area on its own: a hole of one never erases another
        poly = Polygon(np.asarray(a["exterior"], float), [np.asarray(h, float) for h in a["holes"]])
        polys.append(poly if poly.is_valid else poly.buffer(0))
    out, prev = {}, None
    for b in sorted({float(b) for b in borders}):                   # the mask at a larger border contains the one at a smaller: only the cells that are not in yet are tested
        m = np.zeros((nx, ny), np.uint8) if prev is None else prev.copy()
        for poly in polys:
            p = poly.buffer(b, quad_segs=8) if b > 0 else poly
            if p.is_empty:
                continue
            x0, y0, x1, y1 = p.bounds
            i0, i1 = max(0, int(np.searchsorted(xs, x0, "left"))), min(nx, int(np.searchsorted(xs, x1, "right")))        # only the cells inside the bounding box
            j0, j1 = max(0, int(np.searchsorted(ys, y0, "left"))), min(ny, int(np.searchsorted(ys, y1, "right")))
            if i0 >= i1 or j0 >= j1:
                continue
            sub = m[i0:i1, j0:j1]
            todo = sub == 0
            if todo.any():
                X, Y = np.meshgrid(xs[i0:i1], ys[j0:j1], indexing="ij")
                sub[todo] = shapely.contains_xy(p, X[todo], Y[todo]).astype(np.uint8)
        out[b], prev = m, m
    return out


def area_mask(g, kind="intersection", border_m=0.0, grid=None):
    """one mask, see area_masks"""
    return area_masks(g, kind, (border_m,), grid)[float(border_m)]


# ---------------------------------------------------------------------------------------------------------------- the packed cache
def cache_path(store_dir, kind="intersection", border_m=0.0):
    return os.path.join(store_dir, "masks", f"{kind}_b{border_m:g}.npz")


def cache_complete(store_dir, kind="intersection", border_m=0.0, held=None):
    """True when the cache file exists, was made by THIS rasteriser (RASTER_VERSION) for the BEV grid and holds exactly the keyframes the store holds now (`held`: the tokens, scanned once by the caller)"""
    p = cache_path(store_dir, kind, border_m)
    if not os.path.exists(p):
        return False
    try:
        with np.load(p) as z:
            if str(z["raster"]) != RASTER_VERSION or tuple(z["shape"]) != (GRID.nx, GRID.ny) or not np.allclose(z["grid"], [GRID.x_min, GRID.x_max, GRID.y_min, GRID.y_max, GRID.res]):
                return False
            return set(z["tokens"].tolist()) == (store_names(store_dir, ".pkl.gz") if held is None else held)
    except (OSError, ValueError, KeyError):
        return False


def cache_info(store_dir, kind="intersection", border_m=0.0):
    """{"raster", "data", "frames"} of a cache file, None when it is not there (the key reads the identity of the data it was made from)"""
    try:
        with np.load(cache_path(store_dir, kind, border_m)) as z:
            return dict(raster=str(z["raster"]), data=str(z["data"]), frames=int(len(z["tokens"])))
    except (OSError, ValueError, KeyError):
        return None


def _rows(args):
    """the packed masks of a chunk of keyframes: {(kind, border): (len(tokens), nbytes) uint8}"""
    store_dir, tokens, kinds, borders = args
    nbytes = (GRID.nx * GRID.ny + 7) // 8
    out = {(k, b): np.zeros((len(tokens), nbytes), np.uint8) for k in kinds for b in borders}
    for i, tok in enumerate(tokens):
        try:
            with gzip.open(os.path.join(store_dir, tok + ".pkl.gz"), "rb") as f:
                g = pickle.load(f)
        except (EOFError, OSError, zlib.error, pickle.UnpicklingError) as e:
            raise ValueError(f"unreadable lane file {tok}.pkl.gz in {store_dir} ({type(e).__name__}: {e}): make it again (generate-lane rewrites a zero-byte file; audit-lane lists the damaged ones)") from None
        for k in kinds:
            for b, m in area_masks(g, k, borders).items():
                out[(k, b)][i] = np.packbits(m.ravel())
    return out


def build_mask_cache(store_dir, kinds=("intersection",), borders=(0.0, 0.5), force=False, verbose=True, workers=1):
    """rasterise every graph of the lane store `store_dir` once and write one packed cache file per (kind, border) below `<store_dir>/masks/`; a file that is complete (same rasteriser, same keyframes) is left
    alone (`force` makes it again); `workers` > 1: a pool of spawned processes. -> {"files": [paths], "frames": n, "made": [paths written]}"""
    from .gt_api import data_identity
    if not borders or not kinds:
        raise ValueError("kinds and borders: at least one each")
    borders = tuple(float(b) for b in borders)
    tokens = sorted(store_names(store_dir, ".pkl.gz"))
    todo = [(k, b) for k in kinds for b in borders if force or not cache_complete(store_dir, k, b, set(tokens))]
    files = [cache_path(store_dir, k, b) for k in kinds for b in borders]
    if not todo:
        return {"files": files, "frames": len(tokens), "made": []}
    todo_kinds, todo_borders = tuple(dict.fromkeys(k for k, _ in todo)), tuple(dict.fromkeys(b for _, b in todo))
    sig = os.path.basename(os.path.normpath(store_dir))
    t0 = time.time()
    chunks = [tokens[i:i + 500] for i in range(0, len(tokens), 500)]
    jobs = [(store_dir, c, todo_kinds, todo_borders) for c in chunks]
    parts = []
    if workers > 1 and len(jobs) > 1:
        with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            for i, r in enumerate(pool.imap(_rows, jobs)):
                parts.append(r)
                if verbose and (i + 1) % 10 == 0:
                    print(f"  masks {min((i + 1) * 500, len(tokens))}/{len(tokens)} ({time.time() - t0:.0f} s)", flush=True)
    else:
        for i, j in enumerate(jobs):
            parts.append(_rows(j))
            if verbose and (i + 1) % 10 == 0:
                print(f"  masks {min((i + 1) * 500, len(tokens))}/{len(tokens)} ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(os.path.join(store_dir, "masks"), exist_ok=True)
    made, ident = [], data_identity()
    for k, b in todo:
        p = cache_path(store_dir, k, b)
        tmp = p + f".{os.getpid()}.tmp.npz"
        np.savez(tmp, tokens=np.array(tokens, dtype="U32"), bits=np.concatenate([r[(k, b)] for r in parts]), shape=np.array([GRID.nx, GRID.ny]),
                 grid=np.array([GRID.x_min, GRID.x_max, GRID.y_min, GRID.y_max, GRID.res]), border_m=np.array(b), kind=np.array(k), source=np.array(sig),
                 raster=np.array(RASTER_VERSION), data=np.array(ident))
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
    Raises FileNotFoundError when the cache is not there (`python -m GtGenerator generate-bev` or `lane_raster.build_mask_cache(store_dir)` makes it) and ValueError when it was made by another rasteriser"""
    if store_dir is None:
        from . import lane_gt
        store_dir = os.path.join(lane_gt.DATA, lane_gt.signature(rig="bev_fusion"))
    p = cache_path(store_dir, kind, border_m)
    if p not in _CACHES:
        if not os.path.exists(p):
            raise FileNotFoundError(f"no mask cache {p}: make it with `python -m GtGenerator generate-bev` (or lane_raster.build_mask_cache({store_dir!r}))")
        c = MaskCache(p)
        if c.raster != RASTER_VERSION:
            raise ValueError(f"the mask cache {p} was made by another rasteriser ({c.raster}, this one is {RASTER_VERSION}): build_mask_cache makes it again")
        _CACHES[p] = c
    return _CACHES[p].get(token)
