"""The camera rig of nuScenes as the lane GT sees it (hashed into the lane signature): the image size, the nominal calibration of the six cameras and the matrices of one camera.

`lane_tiers.py` (the visibility `vis` / `occ`) and `lane_gt.py` (the cameras of each rig, `RIGS`) read nothing else of the rig; a change of a number here is a new lane version. The coverage masks drawn
from the same calibration are `rig_masks.py` (not hashed: no label comes from them).

The calibration is the mean over the 850 logs of trainval (the logs differ by at most 0.04 m and 1 - 3 degrees), as `coverage.rig_radar_poses` does for the radars.
"""
import numpy as np

IMAGE_SIZE = (1600, 900)
# nominal calibration of the six cameras: translation (m), rotation camera -> ego, focal length, principal point
RIG_CAMERAS = {
    "CAM_FRONT": ([1.7115, 0.0093, 1.5029], [[0.007913, 0.002193, 0.999966], [-0.99995, 0.006066, 0.0079], [-0.006049, -0.999979, 0.002241]], 1259.119, 821.233, 479.118),
    "CAM_FRONT_LEFT": ([1.5519, 0.4974, 1.5085], [[0.821518, 0.00371, 0.57017], [-0.570176, 0.010182, 0.82146], [-0.002758, -0.999941, 0.01048]], 1264.478, 825.771, 463.332),
    "CAM_FRONT_RIGHT": ([1.5675, -0.497, 1.5078], [[-0.838491, 0.00911, 0.544839], [-0.544766, 0.009357, -0.838536], [-0.012737, -0.999915, -0.002883]], 1258.726, 812.377, 470.85),
    "CAM_BACK": ([0.0437, 0.0072, 1.5725], [[0.006323, -0.010916, -0.99992], [0.999971, 0.004209, 0.006278], [0.004141, -0.999932, 0.010942]], 802.469, 841.755, 478.073),
    "CAM_BACK_LEFT": ([1.0429, 0.4831, 1.5759], [[0.947801, -0.000734, -0.31886], [0.318861, 0.003796, 0.947794], [0.000515, -0.999993, 0.003832]], 1255.834, 811.166, 477.926),
    "CAM_BACK_RIGHT": ([1.0391, -0.473, 1.5566], [[-0.929034, 0.006028, -0.369946], [0.369946, -0.001058, -0.929053], [-0.005992, -0.999981, -0.001247]], 1254.433, 815.932, 479.378),
}
SIX = tuple(RIG_CAMERAS)


def camera_calibration(name):
    """(K 3 x 3, T_c2e 4 x 4) of a camera of the rig"""
    t, R, f, cx, cy = RIG_CAMERAS[name]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.asarray(R, float), t
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]]), T
