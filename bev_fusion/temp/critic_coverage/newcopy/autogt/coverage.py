"""What the front rig can observe: camera cells, radar cells, what boxes hide, and how much of an object is observed (decisions D36 - D38).

Pure geometry (numpy only, the calibration and the grid are all it needs), so the car can compute the same masks live and training,
evaluation and the generator share one definition. Ego frame, x forward, y left, z up; the camera sits at the calibrated position.

    camera_cells(gx, gy, K, T_c2e, size, z=0)   cells whose point at height z projects into the CAM_FRONT image (z = 0: the ground plane)
    radar_cells(gx, gy, poses)                  cells inside the +-60 deg / 70 m fan of one of the radars (`poses`: their 4 x 4 ego poses)
    hidden_cells(gx, gy, cam, prisms)           ground cells hidden from the camera by boxes of their real height (the soft rule of D38);
                                                `soft=False` is the old rule that takes every box as infinitely high
    object_shares(box, ...)                     the share of an object's footprint observed by the camera / the radar / both, the
                                                camera seeing the object's whole height (a pedestrian above a low car stays visible)
    footprint_share(corners, mask, grid)        the share of a footprint (4 corners) on the True cells of a grid mask: the same number for
                                                a ground-truth box and a detection; samples outside the grid count as not observed
    fusion_cells(g), coverage_for(g, kind)      the fusion area and the area each output type is trained / scored on, from a GT dict

The fusion area (D37) = camera cells not hidden behind a box, inside the radar fan; outside it is "don't care" on both sides (D38).
An object is care when its fusion share f >= 0.5, partial for 0.1 <= f < 0.5, don't care below 0.1 (the thresholds Mirko accepted;
the states themselves belong to evaluation and training, not to the GT). Boxes hide with their height: a box with the top at z_hi hides
the ground behind it from its front edge out to the shadow of its top (d * H / (H - h) for a flat box and a camera at height H).
"""
import math

import numpy as np

from .layers import inside_quad, occlusion

RADAR_FOV = (math.radians(60.0), 70.0)               # half angle, range (m): measured, 99.9 % of all detections of nuScenes mini lie inside
FRACTIONS = (0.05, 0.5, 0.95)                        # heights of an object (share of its box height) at which the camera must see it
STEP = 0.125                                         # sample spacing inside a footprint (m): the sub-pixel of the 4 x 4 raster at 0.5 m


def camera_cells(gx, gy, K, T_c2e, size, z=0.0, near=0.5):
    """bool array of gx's shape: the point (x, y, z) in the ego frame is in front of the camera (depth > near) and projects into the
    image of `size` = (width, height). z may be an array of the shape of gx (the height slices of a model)."""
    gx, gy = np.asarray(gx, float), np.asarray(gy, float)
    P = np.stack([gx.ravel(), gy.ravel(), np.broadcast_to(np.asarray(z, float), gx.shape).ravel(), np.ones(gx.size)], 1) @ np.linalg.inv(T_c2e).T
    d = P[:, 2]
    u = K[0, 0] * P[:, 0] / np.maximum(d, 1e-6) + K[0, 2]
    v = K[1, 1] * P[:, 1] / np.maximum(d, 1e-6) + K[1, 2]
    return ((d > near) & (u >= 0) & (u < size[0]) & (v >= 0) & (v < size[1])).reshape(gx.shape)


def radar_cells(gx, gy, poses, fov=RADAR_FOV):
    """bool array of gx's shape: inside the fan (half angle, range) of at least one radar; poses = their 4 x 4 sensor -> ego matrices."""
    half, rmax = fov
    cov = np.zeros(np.shape(gx), bool)
    for T in poses:
        yaw = math.atan2(T[1, 0], T[0, 0])
        dx, dy = gx - T[0, 3], gy - T[1, 3]
        ang = np.arctan2(dy, dx) - yaw
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        cov |= (np.abs(ang) <= half) & (np.hypot(dx, dy) <= rmax)
    return cov


def prism(corners, center_z, height):
    """a box as (footprint (4, 2), z_lo, z_hi) in the ego frame: `center_z` is the height of its centre, `height` its size"""
    return np.asarray(corners, float), float(center_z) - 0.5 * float(height), float(center_z) + 0.5 * float(height)


def prism_of(box):
    """the prism of one box of meta["boxes"]"""
    return prism(box["corners"], box["center"][2], box["size"][2])


def _clip(cam_xy, pts, fp):
    """for the horizontal segments cam -> pts (n, 2): the parameter interval [t0, t1] inside the convex quad fp (clipped to [0, 1];
    t0 > t1 = the segment misses it); t1 == 1 means that the end point is inside the quad."""
    d = pts - cam_xy
    t0, t1 = np.zeros(len(pts)), np.ones(len(pts))
    cen = fp.mean(0)
    for k in range(4):
        a, e = fp[k], fp[(k + 1) % 4] - fp[k]
        n = np.array([e[1], -e[0]])
        if n @ (cen - a) > 0:
            n = -n                                   # outward normal
        num = float(n @ (cam_xy - a))                # > 0: the camera is outside this edge
        den = d @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -num / den
        pos, neg = den > 1e-12, den < -1e-12
        t1 = np.where(pos, np.minimum(t1, t), t1)
        t0 = np.where(neg, np.maximum(t0, t), t0)
        if num > 0:
            t1 = np.where(pos | neg, t1, -1.0)       # parallel to the edge and outside: no hit
    return t0, t1


def _wedge(cam_xy, pts, fp):
    """a cheap pre-filter: the points that lie inside the angular wedge a footprint spans as seen from the camera (all of them when the
    camera is so close that the wedge is wide)"""
    v = fp - cam_xy
    vc = fp.mean(0) - cam_xy
    ang = np.arctan2(vc[0] * v[:, 1] - vc[1] * v[:, 0], vc[0] * v[:, 0] + vc[1] * v[:, 1])
    if ang.max() - ang.min() > 2.6 or (vc @ vc) < 1e-6:
        return np.ones(len(pts), bool)
    lo, hi = v[np.argmin(ang)], v[np.argmax(ang)]
    p = pts - cam_xy
    return ((lo[0] * p[:, 1] - lo[1] * p[:, 0]) >= -1e-9) & ((p[:, 0] * hi[1] - p[:, 1] * hi[0]) >= -1e-9) & ((vc[0] * p[:, 0] + vc[1] * p[:, 1]) > 0)


def blocked(points, cam, prisms, skip=()):
    """bool (n,): the straight line from the camera `cam` (x, y, z) to each of `points` (n, 3) passes through the volume of one of the
    `prisms` before it ends. A point inside or on a prism is not blocked by that one (a box does not hide itself); `skip` = indices of
    prisms to leave out (the object's own)."""
    points, cam = np.asarray(points, float), np.asarray(cam, float)
    out = np.zeros(len(points), bool)
    for i, (fp, z_lo, z_hi) in enumerate(prisms):
        if i in skip or fp[:, 0].max() < cam[0]:
            continue
        idx = np.nonzero(_wedge(cam[:2], points[:, :2], fp) & ~out)[0]
        if not len(idx):
            continue
        t0, t1 = _clip(cam[:2], points[idx, :2], fp)
        cross = (t0 < t1) & (t1 < 1.0)               # enters the footprint, ends beyond it
        za, zb = cam[2] + t0 * (points[idx, 2] - cam[2]), cam[2] + t1 * (points[idx, 2] - cam[2])
        out[idx] |= cross & (np.maximum(za, zb) >= z_lo) & (np.minimum(za, zb) <= z_hi)
    return out


def hidden_cells(gx, gy, cam, prisms, soft=True):
    """bool array of gx's shape: ground cells (z = 0) that the camera at `cam` (x, y, z) cannot see because of a box. soft=True: boxes
    hide with their real volume (`prisms`, see `prism`); soft=False: the old rule, every box infinitely high (footprints only)."""
    if not soft:
        return occlusion(gx, gy, np.asarray(cam, float)[:2], [p[0] for p in prisms])
    pts = np.stack([np.asarray(gx, float).ravel(), np.asarray(gy, float).ravel(), np.zeros(np.size(gx))], 1)
    return blocked(pts, cam, prisms).reshape(np.shape(gx))


def footprint_samples(corners, step=STEP):
    """points (n, 2) spread evenly over the footprint quad (4 consecutive corners), about `step` m apart"""
    c = np.asarray(corners, float)
    e1, e2 = c[1] - c[0], c[3] - c[0]
    n1, n2 = max(int(math.ceil(np.hypot(*e1) / step)), 1), max(int(math.ceil(np.hypot(*e2) / step)), 1)
    u = (np.arange(n1) + 0.5) / n1
    v = (np.arange(n2) + 0.5) / n2
    return (c[0] + u[:, None, None] * e1 + v[None, :, None] * e2).reshape(-1, 2)


def _on_grid(xy, grid):
    return (xy[:, 0] >= grid.x_min) & (xy[:, 0] < grid.x_max) & (xy[:, 1] >= grid.y_min) & (xy[:, 1] < grid.y_max)


def footprint_share(corners, mask, grid, step=STEP):
    """the share of the footprint (4 consecutive corners, ego frame) that lies on True cells of `mask` (nx, ny); samples outside the grid
    count as not observed. The one number for ground truth and detections: f = footprint_share(corners, fusion_cells(g), grid)."""
    xy = footprint_samples(corners, step)
    ok = _on_grid(xy, grid)
    ix = np.clip(((xy[:, 0] - grid.x_min) / grid.res).astype(int), 0, grid.nx - 1)
    iy = np.clip(((xy[:, 1] - grid.y_min) / grid.res).astype(int), 0, grid.ny - 1)
    return float((ok & np.asarray(mask, bool)[ix, iy]).mean())


def object_shares(box, index, prisms, cam, K, T_c2e, size, radar_poses, grid, step=STEP):
    """the observed shares of one object (a box of meta["boxes"]; `index` = its place in `prisms`, None when it is not one of the
    hiding boxes): the share of its footprint in the camera's view with the whole height of the object counted (the camera sees a footprint
    sample when it sees the object above it at the bottom, the middle or the top: inside the image and not hidden by another box), in the
    radar fan, and in both. Samples outside the grid count as not observed. -> {"camera", "radar", "fusion"}"""
    fp, z_lo, z_hi = prism_of(box)
    xy = footprint_samples(fp, step)
    ok = _on_grid(xy, grid)
    if not ok.any():                                 # entirely outside the grid: nothing to observe
        return {"camera": 0.0, "radar": 0.0, "fusion": 0.0}
    zs = z_lo + np.asarray(FRACTIONS) * (z_hi - z_lo)
    pts = np.concatenate([np.c_[xy, np.full(len(xy), z)] for z in zs])
    vis = camera_cells(pts[:, 0], pts[:, 1], K, T_c2e, size, z=pts[:, 2]) & ~blocked(pts, cam, prisms, skip=() if index is None else (index,))
    seen = vis.reshape(len(zs), len(xy)).any(0)
    fan = radar_cells(xy[:, 0], xy[:, 1], radar_poses)
    return {"camera": float((ok & seen).mean()), "radar": float((ok & fan).mean()), "fusion": float((ok & seen & fan).mean())}


def fusion_cells(g):
    """the fusion area (D37) of a GT dict: camera cells not hidden behind a box, inside the radar fan"""
    return (np.asarray(g["camera"]) > 0) & (np.asarray(g["occluded"]) == 0) & (np.asarray(g["radar"]) > 0)


def camera_observed(g):
    """the cells the camera observes now: in the image and not hidden behind a box"""
    return (np.asarray(g["camera"]) > 0) & (np.asarray(g["occluded"]) == 0)


COVERAGE = {"map": "fusion", "objects": "fusion", "velocity": "fusion", "camera_aux": "camera", "radar_aux": "radar"}


def coverage_for(g, kind):
    """the cells where an output type of the fused system is trained and scored (everywhere else: don't care, D38): map layers, objects
    and velocity on the fusion area; the camera aux head on the camera cells, the radar aux head on the radar fan (each sensor's own area)."""
    area = COVERAGE[kind]
    return fusion_cells(g) if area == "fusion" else camera_observed(g) if area == "camera" else np.asarray(g["radar"]) > 0
