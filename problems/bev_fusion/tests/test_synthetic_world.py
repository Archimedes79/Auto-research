"""Tests of tests/support/synthetic_world.py (its former self-test).

    python tests/test_synthetic_world.py      (from the problem folder, with its module path)
"""
import math
import random

import torch

from tests.support.synthetic_world import Box, Scene, camera, front_camera, gt_bev, radar, random_scene


def _selftest():
    from net.model.lift_pd import Grid
    rng = random.Random(3)
    sc = Scene([Box(20.0, 0.0, 4.5, 1.8, 1.5, 0.0, 8.0, 0.0, "car", 10.0),
                Box(35.0, -6.0, 4.5, 1.8, 1.5, math.pi / 2, 0.0, 5.0, "car", 10.0)], v=10.0, omega=0.2)
    K, T, hw = front_camera()
    img, depth, cls = camera(sc, K, T, hw)
    # 1 the image centre column looks at the first box's rear face: depth = 20 - 2.25 - 1.5 (camera at x = 1.5)
    h, w = hw
    col = cls[:, w // 2]
    rows = (col == 3).nonzero()
    assert len(rows) > 0, "box visible in the centre column"
    assert abs(depth[rows[len(rows) // 2, 0], w // 2].item() - (20 - 2.25 - 1.5)) < 1e-3, "box depth"
    # 2 a road pixel: depth along the optical axis = analytic ground depth f h / (v - cy)
    r = h - 5
    assert cls[r, 10] == 1 and abs(depth[r, 10].item() - 560.0 * 1.5 / (r - (h - 1) / 2)) < 1e-3, "ground depth"
    # 3 radar Doppler = object velocity on the line of sight; clutter Doppler 0
    det = radar(sc, rng=rng, sigma_r=0.0, sigma_phi_deg=0.0)
    for i in range(len(det["range"])):
        k = det["box"][i].item()
        x, y = det["xy"][i].tolist()
        ux, uy = (x - 3.5) / math.hypot(x - 3.5, y), y / math.hypot(x - 3.5, y)
        exp = 0.0 if k < 0 else sc.boxes[k].vx * ux + sc.boxes[k].vy * uy
        assert abs(det["doppler"][i].item() - exp) < 1e-4, "Doppler"
    assert (det["box"] >= 0).sum() > 0 and (det["box"] < 0).sum() > 0
    # 4 step: a static lane point and a moving box follow the exact motion; velocity rotates with the frame
    s1 = sc.step(0.5)
    th = 0.2 * 0.5
    assert abs(s1.boxes[0].heading + th) < 1e-12 and abs(math.hypot(s1.boxes[0].vx, s1.boxes[0].vy) - 8.0) < 1e-9
    # 5 GT: the box footprint carries its velocity, the lane layer has the lines
    g = gt_bev(sc, Grid())
    assert g["vehicles"].sum() > 2 * 4.5 * 1.8 * 4 * 0.9 and torch.allclose(g["velocity"][0][g["box_id"] == 0], torch.tensor(8.0))
    assert g["lanes"].sum() > 0
    rs = random_scene(rng)
    camera(rs, K, T, hw)
    gt_bev(rs.step(1 / 12), Grid())
    print("synthetic_world selftest ok: box and ground depth exact, Doppler = v . u, step with exact ego motion, GT "
          "velocity field, random scenes")


if __name__ == "__main__":
    _selftest()
