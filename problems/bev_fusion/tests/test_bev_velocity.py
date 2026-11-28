"""Tests of evaluation/bev_velocity.py (its former self-test).

    python -P -m bev_fusion.tests.test_bev_velocity      (from the problem folder, with its module path)
"""
from __future__ import annotations

import numpy as np

from Evaluation import Prediction, Sample
from bev_fusion.evaluation.bev_velocity import BevVelocityMetric, centre_index


def selftest():
    from types import SimpleNamespace
    from Evaluation.contracts import FrameRef
    grid = SimpleNamespace(x_min=0.0, y_min=-2.0, res=0.5)
    X, Y = 8, 8
    z = np.zeros((X, Y), np.float32)
    cam, veh = np.ones((X, Y), np.float32), z.copy()
    veh[2:4, 2:4] = 1                                                    # a moving vehicle (4 cells) at x 1.0 - 2.0, y -1.0 - 0.0
    gv = np.zeros((2, X, Y), np.float32)
    gv[0, 2:4, 2:4] = 3.0                                                # 3 m/s along x
    veh[6, 6] = 1                                                        # a parked one (1 cell)
    veh[1, 7] = 1                                                        # a car of unknown velocity (GT 0 m/s, flow_weight 0): left out
    fw = np.ones((X, Y), np.float32)
    fw[1, 7] = 0
    meta = {"boxes": [{"group": "vehicle", "center": [1.6, -0.6, 0], "size": [1, 1, 1], "yaw": 0.0, "vel": [3.0, 0.0], "observed": True},
                      {"group": "vehicle", "center": [3.2, 1.2, 0], "size": [0.5, 0.5, 1], "yaw": 0.0, "vel": [0.0, 0.0], "observed": True},
                      {"group": "vehicle", "center": [9.0, 0.0, 0], "size": [1, 1, 1], "yaw": 0.0, "vel": [5.0, 0.0], "observed": True},     # outside the grid
                      {"group": "vehicle", "center": [0.6, 1.6, 0], "size": [0.5, 0.5, 1], "yaw": 0.0, "vel": [0.0, 0.0], "observed": True, "vel_known": False}]}
    g = {"camera": cam, "undefined": z, "vehicle": veh, "velocity": gv, "flow_weight": fw, "meta": meta}
    m = BevVelocityMetric(grid, ["vehicle"])
    v = gv + np.array([0.3, 0.4], np.float32)[:, None, None]             # field off by 0.5 m/s everywhere
    objvel = np.zeros_like(gv)
    objvel[0, 3, 2] = 2.0                                                # the object head: 2 m/s at the moving box's centre cell (3, 2)
    c = m.frame(Sample(ref=FrameRef("gt", 0, "s"), inputs=None, reference={0.0: g}), Prediction(outputs={"dt": {0.0: {"vhat": v, "objvel": objvel}}}))
    r = m.reduce([c, c])
    # 5 cells x 2 frames (float32 field)
    assert r["dt+0.00/cells/vehicle/n"] == 10 and abs(r["dt+0.00/cells/vehicle/epe"] - 0.5) < 1e-6, r
    assert r["dt+0.00/cells/vehicle_moving/n"] == 8 and r["dt+0.00/cells/static/n"] == 2, r
    assert "dt+0.00/cells/static_moving/n" not in r and "dt+0.00/cells/obj/epe" not in r, "static has no moving part; obj / moving belong to `curve`"
    # the box outside the grid dropped
    assert r["dt+0.00/centre/field/all/n"] == 4 and r["dt+0.00/centre/field/moving/n"] == 2, r
    assert abs(r["dt+0.00/centre/field/moving/epe"] - 0.5) < 1e-6 and abs(r["dt+0.00/centre/objhead/moving/epe"] - 1.0) < 1e-9, r
    assert abs(r["dt+0.00/centre/zero/moving/epe"] - 3.0) < 1e-9 and abs(r["dt+0.00/centre/zero/all/epe"] - 1.5) < 1e-9, r
    ix, iy, kept = centre_index([{"cx": 1.6, "cy": -0.6}, {"cx": 0.24, "cy": -2.0}], grid, cam > 0)
    assert list(ix) == [3, 0] and list(iy) == [2, 0], (ix, iy)                                                                 # floor((c - min) / res)
    c2 = m.frame(Sample(ref=FrameRef("gt", 0, "s"), inputs=None, reference={0.0: g}), Prediction(outputs={"dt": {0.0: {"vhat": None, "objvel": None}}}))
    assert c2 == {}, "no velocity output, no counts"
    assert m.direction("dt+0.00/cells/static/epe") == "low" and m.direction("dt+0.00/centre/zero/all/r2") == "high" and m.direction("x/n") == "none"
    print("bev_velocity selftest ok: cell sets (static, per class, moving part), centre cells and drop rule, unknown velocity left out, field / object head / zero at the centres, directions")


if __name__ == "__main__":
    selftest()
