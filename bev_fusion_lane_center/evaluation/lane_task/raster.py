"""Lane maps as images: the grid, rasterising lines, distance maps, thinning (Zhang-Suen, numpy) and tracing a thin image back
into a lane graph. This is the bridge between the two levels of the lane suite: a dense head's picture is scored as an image
first, then vectorised (`skeleton_graph`) and scored as a sparse graph.

Arrays are indexed [row, col]; col runs along u (horizontal on the picture), row along v (vertical, origin 'lower').
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .geometry import LaneGraph, resample, simplify


@dataclass(frozen=True)
class Grid:
    u0: float          # lower edge of column 0 in metres
    v0: float          # lower edge of row 0
    res: float         # cell size in metres
    nu: int            # columns
    nv: int            # rows

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.nv, self.nu)

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        return (self.u0, self.u0 + self.nu * self.res, self.v0, self.v0 + self.nv * self.res)

    @property
    def box(self) -> Tuple[float, float, float, float]:
        return self.extent

    def centres(self) -> Tuple[np.ndarray, np.ndarray]:
        """(U, V) metre coordinates of all cell centres, shape (nv, nu)"""
        u = self.u0 + (np.arange(self.nu) + 0.5) * self.res
        v = self.v0 + (np.arange(self.nv) + 0.5) * self.res
        return np.meshgrid(u, v)

    def cell(self, xy) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(row, col, inside) of metre points"""
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        c = np.floor((xy[:, 0] - self.u0) / self.res).astype(int)
        r = np.floor((xy[:, 1] - self.v0) / self.res).astype(int)
        ok = (c >= 0) & (c < self.nu) & (r >= 0) & (r < self.nv)
        return r, c, ok

    def points(self, mask) -> np.ndarray:
        """metre coordinates (N, 2) of the True cells"""
        r, c = np.nonzero(mask)
        return np.c_[self.u0 + (c + 0.5) * self.res, self.v0 + (r + 0.5) * self.res]


def rasterize(g, grid: Grid, width: float = 0.0) -> np.ndarray:
    """bool image: the cells touched by the polylines, thickened to `width` metres (0 = one cell wide)"""
    g = LaneGraph.of(g)
    m = np.zeros(grid.shape, bool)
    for p in g.lines:
        if len(p) < 2:
            continue
        r, c, ok = grid.cell(resample(p, grid.res / 3.0))
        m[r[ok], c[ok]] = True
    if width > grid.res and m.any():
        m = edt(m, grid) <= width / 2.0
    return m


def edt(mask, grid: Grid) -> np.ndarray:
    """distance in metres from every cell to the nearest True cell (a large number where there is none)"""
    from scipy.ndimage import distance_transform_edt
    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.full(mask.shape, 1e6, np.float32)
    return (distance_transform_edt(~mask) * grid.res).astype(np.float32)


def thin(mask) -> np.ndarray:
    """Zhang-Suen thinning of a bool image (vectorised): a one-cell-wide skeleton"""
    img = np.pad(np.asarray(mask, bool).astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p = img
            P2, P3, P4, P5 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:]
            P6, P7, P8, P9 = p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]
            C = p[1:-1, 1:-1]
            nb = [P2, P3, P4, P5, P6, P7, P8, P9]
            B = sum(x.astype(np.int16) for x in nb)
            A = sum(((nb[i] == 0) & (nb[(i + 1) % 8] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                m = ((P2 * P4 * P6) == 0) & ((P4 * P6 * P8) == 0)
            else:
                m = ((P2 * P4 * P8) == 0) & ((P2 * P6 * P8) == 0)
            rem = (C == 1) & (B >= 2) & (B <= 6) & (A == 1) & m
            if rem.any():
                img[1:-1, 1:-1][rem] = 0
                changed = True
    return img[1:-1, 1:-1].astype(bool)


def _neighbours(skel) -> np.ndarray:
    from scipy.ndimage import convolve
    k = np.ones((3, 3), np.int16)
    k[1, 1] = 0
    return convolve(skel.astype(np.int16), k, mode="constant") * skel


def _smooth(p: np.ndarray, w: int) -> np.ndarray:
    """moving average over w points (removes the staircase of a digitised line); the end points stay where they are"""
    if len(p) < w + 2:
        return p
    h = w // 2
    q = np.pad(p, ((h, h), (0, 0)), mode="edge")
    k = np.ones(w) / w
    out = np.stack([np.convolve(q[:, i], k, mode="valid") for i in range(2)], axis=1)
    out[0], out[-1] = p[0], p[-1]
    return out


def skeleton_graph(mask, grid: Grid, min_len: float = 2.0, tol: float = 0.3) -> LaneGraph:
    """Trace a bool image into an undirected lane graph: thin it, cut it at junctions (clusters of cells with >= 3 neighbours),
    follow every chain between junctions / ends, drop chains shorter than `min_len` metres, simplify every polyline by
    Douglas-Peucker at `tol`. Pieces that meet at a junction share the cluster centroid as end point, so the point graph links them."""
    from scipy.ndimage import label
    sk = thin(mask)
    if not sk.any():
        return LaneGraph([], directed=False)
    nb = _neighbours(sk)
    junc = sk & (nb >= 3)
    jl, nj = label(junc, structure=np.ones((3, 3)))
    cent = {}
    for k in range(1, nj + 1):
        r, c = np.nonzero(jl == k)
        cent[k] = np.array([grid.u0 + (c.mean() + 0.5) * grid.res, grid.v0 + (r.mean() + 0.5) * grid.res])
    chain = sk & ~junc
    cl, nc = label(chain, structure=np.ones((3, 3)))
    lines = []
    H, W = sk.shape
    for k in range(1, nc + 1):
        rr, cc = np.nonzero(cl == k)
        pix = list(zip(rr.tolist(), cc.tolist()))
        pset = set(pix)

        def nbrs(q):
            return [(q[0] + a, q[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a or b) and (q[0] + a, q[1] + b) in pset]
        # order the chain: start at a cell with <= 1 neighbour inside the chain (a loop starts anywhere)
        ends = [q for q in pix if len(nbrs(q)) <= 1]
        cur = ends[0] if ends else pix[0]
        order, seen = [cur], {cur}
        while True:
            nxt = [q for q in nbrs(cur) if q not in seen]
            if not nxt:
                break
            nxt.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))     # prefer the 4-neighbour
            cur = nxt[0]
            order.append(cur)
            seen.add(cur)
        pts = np.array([[grid.u0 + (c + 0.5) * grid.res, grid.v0 + (r + 0.5) * grid.res] for r, c in order])

        def junction_at(q):
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    r, c = q[0] + a, q[1] + b
                    if 0 <= r < H and 0 <= c < W and jl[r, c] > 0:
                        return int(jl[r, c])
            return 0
        j0, j1 = junction_at(order[0]), junction_at(order[-1])
        if j0:
            pts = np.vstack([cent[j0], pts])
        if j1 and (j1 != j0 or len(order) > 3):
            pts = np.vstack([pts, cent[j1]])
        if len(pts) >= 2 and np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() >= min_len:
            pts = _smooth(pts, 5)
            lines.append(pts[simplify(pts, tol)])
    return LaneGraph(lines, directed=False)
