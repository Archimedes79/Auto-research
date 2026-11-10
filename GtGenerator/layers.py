"""Rasterising helpers of the static GT layers (dataset independent): an exact supersampled raster, cell centres, a 2D
occlusion mask. Arrays are [ix, iy] (x forward, y left) as everywhere in autogt; coordinates are the ego frame at the frame
time. Used by nuscenes_gt.py."""
import cv2
import numpy as np

AREAS = ("drivable_area", "ped_crossing", "walkway", "stop_line", "carpark_area")
LINES = ("lane_divider", "road_divider")
OBJECTS = ("vehicle", "pedestrian", "two_wheeler")
DIST_MAX = 2.0                                        # m, truncation of the divider distance fields
SHIFT = 4                                             # OpenCV fixed-point sub-pixel bits


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
        fillPoly counts the boundary pixels on both sides: a 2 m square came out as 4.52 m^2 at ss = 4). Polygons with many
        vertices (a road network) are cut to the grid window first."""
        g, s = self.g, self.s
        win = (g.x_min - s, g.y_min - s, g.x_max + s, g.y_max + s)
        for ext, holes in polygons:
            for ext_p, holes_p in self._clipped(ext, holes, win):
                blk = self._block(ext_p)
                if blk is None:
                    continue
                r0, r1, c0, c1 = blk
                m = self._ring_mask(ext_p, blk)
                for h in holes_p:                                  # each hole is tested only inside its own box
                    m &= ~self._ring_mask(h, blk)
                img[r0:r1, c0:c1][m] = value
        return img

    def _block(self, ring):
        """(r0, r1, c0, c1) sub-pixel index box of a ring, clipped to the canvas (None if outside)."""
        g, s = self.g, self.s
        lo = np.floor((ring.min(0) - [g.x_min, g.y_min]) / s - 0.5).astype(int)
        hi = np.ceil((ring.max(0) - [g.x_min, g.y_min]) / s - 0.5).astype(int)
        r0, c0 = max(lo[0], 0), max(lo[1], 0)
        r1, c1 = min(hi[0] + 1, self.shape[0]), min(hi[1] + 1, self.shape[1])
        return None if (r0 >= r1 or c0 >= c1) else (r0, r1, c0, c1)

    def _ring_mask(self, ring, blk):
        """bool block: sub-pixel centres inside the ring (matplotlib Path test), evaluated only in the ring's own box."""
        from matplotlib.path import Path
        g, s = self.g, self.s
        r0, r1, c0, c1 = blk
        m = np.zeros((r1 - r0, c1 - c0), bool)
        own = self._block(ring)
        if own is None:
            return m
        a0, a1, b0, b1 = max(own[0], r0), min(own[1], r1), max(own[2], c0), min(own[3], c1)
        if a0 >= a1 or b0 >= b1:
            return m
        rr, cc = np.meshgrid(np.arange(a0, a1), np.arange(b0, b1), indexing="ij")
        pts = np.stack([g.x_min + (rr.ravel() + 0.5) * s, g.y_min + (cc.ravel() + 0.5) * s], 1)
        m[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = Path(ring).contains_points(pts).reshape(rr.shape)
        return m

    @staticmethod
    def _clipped(ext, holes, win):
        """a polygon with many vertices (a whole road network) is cut to the grid window first: every point test then
        runs against the few vertices inside the window instead of thousands; small polygons are passed through."""
        if len(ext) + sum(len(h) for h in holes) < 60:
            yield ext, holes
            return
        from shapely.geometry import Polygon, box
        poly = Polygon(ext, holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        cut = poly.intersection(box(*win))
        for part in getattr(cut, "geoms", [cut]):
            if part.geom_type == "Polygon" and not part.is_empty:
                yield (np.asarray(part.exterior.coords)[:-1], [np.asarray(h.coords)[:-1] for h in part.interiors])

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


def inside_quad(P, poly):
    """points P (N, 2) inside the convex quadrilateral poly (4, 2), either orientation."""
    e = np.roll(poly, -1, axis=0) - poly                                 # the four edges a -> b
    S = e[:, 0] * (P[:, 1:2] - poly[:, 1]) - e[:, 1] * (P[:, 0:1] - poly[:, 0])      # (N, 4): the side of every point of every edge
    return (S >= 0).all(1) | (S <= 0).all(1)


def occlusion(gx, gy, cam_xy, footprints):
    """cells whose segment from the camera crosses a box footprint they are not inside (ego frame, 2D)."""
    P = np.stack([gx.ravel(), gy.ravel()], 1)
    C = np.asarray(cam_xy, float)
    occ = np.zeros(len(P), bool)
    d = P - C
    for fp in footprints:
        if fp[:, 0].max() < C[0]:
            continue
        inside = inside_quad(P, fp)
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
