"""The two coverage masks of the suite, one per sensor rig: BEV fusion (front) and Spurmitte (surround).

    mask_for(project)            THE entry point: the mask of a project by its name ("bev_fusion" or "lane_center"), so nobody has to choose by guessing:
                                 project            folder                       mask                  frame of the arrays
                                 "bev_fusion"       bev_fusion/                  bev_fusion_mask()     ego, x forward, y left; the front grid 120 x 104
                                 "lane_center"      bev_fusion_lane_center/      lane_mask()           lane, x right, y forward; the window 200 x 200
    bev_fusion_mask(grid=None)   the front camera and the five radars on the BEV-fusion grid (default GridSpec(): x forward 0 - 60 m, y left +-26 m, 0.5 m):
                                 {"camera", "radar", "fusion"}, bool arrays [ix, iy] in the ego frame (x forward, y left)
    lane_mask(res=0.5, half=50)  the six cameras and the five radars on the Spurmitte window (the lane project: 100 x 100 m round the ego, 200 x 200 cells at 0.5 m):
                                 {"camera", "radar", "fusion", "n_cameras"}, arrays [ix, iy] in the LANE frame (x RIGHT, y forward, cell centres at -half + (i + 0.5) res)

Both are static: the rig of nuScenes is the same in every frame (the calibration differs between the logs by at most 0.04 m and 1 - 3 degrees: the mean over the 850
logs of trainval is used, as `coverage.rig_radar_poses` does for the radars; the six cameras are `rig_cameras.py`, which the lane signature hashes: this file is not hashed), so the masks are made
from the calibration alone, in milliseconds, nothing is stored and there is no GT version. They are the same definition as in `coverage.py` (D36 - D38): a camera cell is a point on the ground plane (z = 0 of the ego frame) in front of the camera
that projects into its 1600 x 900 image; a radar cell lies inside the +-60 degree / 70 m fan of one of the five radars; the fusion area is camera and radar.

What the static masks do NOT have: what boxes hide. For BEV fusion the stored front GT has the exact per-frame version (layers `camera`, `occluded`, `radar`, `undefined`, and
`coverage.fusion_cells(g)`), use that for scoring; the static mask is for code that has no GT frame at hand (the model, a plot). For Spurmitte there is no per-frame
occlusion: a lane line behind a car counts as observed here.

`python -m GtGenerator masks out.png` draws both.
"""
import functools

import numpy as np

from . import coverage as cvg
from .grid import GridSpec
from .rig_cameras import IMAGE_SIZE, SIX, camera_calibration


def bev_fusion_mask(grid=None):
    """the front camera, the five radars and their fusion area on the BEV-fusion grid (see the module docstring)"""
    gx, gy = (grid or GridSpec()).centres()
    K, T = camera_calibration("CAM_FRONT")
    cam, rad = cvg.camera_cells(gx, gy, K, T, IMAGE_SIZE), cvg.radar_cells(gx, gy, cvg.rig_radar_poses())
    return {"camera": cam, "radar": rad, "fusion": cam & rad}


@functools.lru_cache(maxsize=16)
def _lane(res, half, cameras):
    n = int(round(2 * half / res))
    c = -half + (np.arange(n) + 0.5) * res
    xl, yl = np.meshgrid(c, c, indexing="ij")                     # [ix, iy]: x right, y forward
    ex, ey = yl, -xl                                              # the same points in the ego frame of the coverage functions (x forward, y left)
    count = np.zeros((n, n), np.int8)
    for name in cameras:
        K, T = camera_calibration(name)
        count += cvg.camera_cells(ex, ey, K, T, IMAGE_SIZE)
    cam = count > 0
    rad = cvg.radar_cells(ex, ey, cvg.rig_radar_poses())
    return {"camera": cam, "radar": rad, "fusion": cam & rad, "n_cameras": count}


def lane_mask(res=0.5, half=50.0, cameras=SIX):
    """the cameras (default all six), the five radars and their fusion area on the Spurmitte window, in the lane frame (see the module docstring); `n_cameras` counts
    the cameras that see a cell. `res` 1 or 2 gives the coarse maps of the head."""
    return {k: v.copy() for k, v in _lane(float(res), float(half), tuple(cameras)).items()}


PROJECTS = {"bev_fusion": "bev_fusion_mask", "lane_center": "lane_mask"}


def mask_for(project, **kwargs):
    """the coverage mask of a project: "bev_fusion" (the front fusion system, folder bev_fusion/) -> bev_fusion_mask(**kwargs); "lane_center" (the lane-centre project on
    GaussianCaR features, folder bev_fusion_lane_center/) -> lane_mask(**kwargs). Any other name is an error that names the two. The arrays differ in shape and in frame, so a
    mask taken for the wrong project does not fit the project's grid and fails at once instead of scoring on the wrong area."""
    if project not in PROJECTS:
        raise ValueError(f"unknown project {project!r}: the masks are for {sorted(PROJECTS)} (bev_fusion = front camera + 5 radars, lane_center = 6 cameras + 5 radars)")
    return globals()[PROJECTS[project]](**kwargs)


def picture(path):
    """both masks in one png: BEV fusion (left, front grid) and Spurmitte (right, 100 x 100 m); camera = green, radar = blue, both = bright, ego = white box"""
    import cv2
    s = 6                                                        # pixels per cell

    def paint(cam, rad, flip):
        img = np.zeros(cam.shape + (3,), np.uint8)
        img[..., 1] = np.where(cam, 170, 0)                      # green: the camera sees it
        img[..., 0] = np.where(rad, 170, 0)                      # blue: a radar fan covers it
        img[cam & rad] = (255, 255, 120)                         # both: the fusion area
        img = np.transpose(img, (1, 0, 2))                       # [iy, ix] -> rows are y
        return np.ascontiguousarray(img[::-1] if flip else img)
    bev = bev_fusion_mask()
    g = GridSpec()
    left = paint(bev["camera"], bev["radar"], flip=True)         # BEV grid: x forward = columns, y left = rows (up)
    left = cv2.resize(left, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    lane = lane_mask()
    right = paint(lane["camera"], lane["radar"], flip=True)      # lane window: x right = columns, y forward = rows (up)
    right = cv2.resize(right, None, fx=s * 0.5 * 2, fy=s * 0.5 * 2, interpolation=cv2.INTER_NEAREST)
    h = max(left.shape[0], right.shape[0]) + 40
    out = np.full((h, left.shape[1] + right.shape[1] + 30, 3), 30, np.uint8)
    out[30:30 + left.shape[0], 10:10 + left.shape[1]] = left
    out[30:30 + right.shape[0], 20 + left.shape[1]:20 + left.shape[1] + right.shape[1]] = right
    cv2.putText(out, "BEV fusion: front camera + 5 radars (x forward 0-60 m, y +-26 m)", (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "Spurmitte: 6 cameras + 5 radars (100 x 100 m)", (20 + left.shape[1], 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    ex, ey = 10 + int((0 - g.x_min) / g.res * s), 30 + left.shape[0] - int((0 - g.y_min) / g.res * s)       # the ego on the BEV grid (x = 0)
    cv2.rectangle(out, (ex - 3, ey - 6), (ex + 12, ey + 6), (255, 255, 255), 1)
    cx, cy = 20 + left.shape[1] + right.shape[1] // 2, 30 + right.shape[0] // 2
    cv2.rectangle(out, (cx - 6, cy - 12), (cx + 6, cy + 12), (255, 255, 255), 1)
    cv2.imwrite(path, out)
    return path


def selftest(root=None, verbose=True):
    """orientation, the cross-check against the per-frame front masks of the stored GT (nuScenes mini), symmetry, numbers"""
    import os
    from .suite import AutoGT
    # orientation of the lane frame: ahead is seen by the front camera, behind by the back camera, the sides by the side cameras
    def at(m, x, y, res=0.5, half=50.0):
        return bool(m[int((x + half) / res), int((y + half) / res)])
    front, back = lane_mask(cameras=("CAM_FRONT",))["camera"], lane_mask(cameras=("CAM_BACK",))["camera"]
    right, left = lane_mask(cameras=("CAM_FRONT_RIGHT", "CAM_BACK_RIGHT"))["camera"], lane_mask(cameras=("CAM_FRONT_LEFT", "CAM_BACK_LEFT"))["camera"]
    assert at(front, 0.25, 20.25) and not at(front, 0.25, -20.25) and at(back, 0.25, -20.25) and not at(back, 0.25, 20.25)
    assert at(right, 20.25, 0.25) and not at(right, -20.25, 0.25) and at(left, -20.25, 0.25) and not at(left, 20.25, 0.25), "x is RIGHT in the lane frame"
    m = lane_mask()
    assert m["camera"].shape == (200, 200) and m["camera"].dtype == bool and m["n_cameras"].max() >= 2 and not at(m["camera"], 0.25, 0.25), "the ego's own cell is not seen"
    assert lane_mask(res=1.0)["camera"].shape == (100, 100) and lane_mask(res=2.0)["camera"].shape == (50, 50)
    assert (lane_mask()["camera"] == m["camera"]).all() and lane_mask() is not lane_mask(), "a copy, the cache stays clean"
    mirror = m["camera"] == m["camera"][::-1]                                         # left <-> right (the rig is nearly mirror symmetric)
    assert mirror.mean() > 0.96, mirror.mean()
    # who uses which: by name, with the grid of the project (the shapes differ, a mix-up is visible at once)
    assert mask_for("bev_fusion")["camera"].shape == (120, 104) and mask_for("lane_center")["camera"].shape == (200, 200)
    assert mask_for("lane_center", res=1.0)["camera"].shape == (100, 100) and (mask_for("bev_fusion")["fusion"] == bev_fusion_mask()["fusion"]).all()
    try:
        mask_for("bev")
        raise AssertionError("an unknown project must be refused")
    except ValueError as e:
        assert "bev_fusion" in str(e) and "lane_center" in str(e)
    # the static front mask against the per-frame masks of the stored GT (real calibration, same functions)
    root = os.path.expanduser(root or "~/data/nuscenes")
    gt = AutoGT(root, "v1.0-mini", light=True, scenes=["scene-0061", "scene-0103"])
    bev = bev_fusion_mask()
    ious = {"camera": [], "radar": []}
    for name in ("scene-0061", "scene-0103"):
        for tok in gt.keyframes(name)[::10]:
            g = gt.keyframe(tok)
            for k in ious:
                a, b = bev[k], g[k] > 0
                ious[k].append((a & b).sum() / max((a | b).sum(), 1))
    assert np.mean(ious["camera"]) > 0.97 and np.mean(ious["radar"]) > 0.995, {k: float(np.mean(v)) for k, v in ious.items()}
    msg = (f"rig masks selftest ok: static front camera vs the per-frame GT layer IoU {np.mean(ious['camera']):.3f} (min {np.min(ious['camera']):.3f}), radar {np.mean(ious['radar']):.4f}; "
           f"Spurmitte window: camera {m['camera'].mean():.0%}, radar {m['radar'].mean():.0%}, fusion {m['fusion'].mean():.0%} of the 100 x 100 m; "
           f"BEV fusion grid: camera {bev['camera'].mean():.0%}, radar {bev['radar'].mean():.0%}, fusion {bev['fusion'].mean():.0%}")
    if verbose:
        print(msg)
    return msg


if __name__ == "__main__":
    print(selftest())
