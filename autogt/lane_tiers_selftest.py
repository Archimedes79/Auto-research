"""Test of the data side of `lane_tiers` on a FAKE nuScenes devkit (no data needed): the driven path from the CAM_FRONT ego poses (rotated frame, start of a log, a standing car), the box prisms
(footprint in the ego frame, observed boxes of the hiding classes only) and `add_tiers` end to end on a toy lane. The toy lane of `lane_tiers.selftest` covers the label rules; this covers the
frames and the table reading (critic pass 2026-11-07, missing test a).   python -m autogt.lane_tiers_selftest"""
import math

import numpy as np
from pyquaternion import Quaternion

from . import lane_tiers as LT

P = dict(tier_back=25.0, tier_ahead=55.0, path_tol=1.5, path_cos=0.7, neighbour_tol=10.0, min_ahead=40.0, min_back=20.0, extend=True, opposite_tier=3)


class FakeNusc:
    """get(table, token) over dicts: one scene of CAM_FRONT sample_data at 12 Hz with their ego poses, one keyframe `key`, annotations"""

    def __init__(self, yaw, start, speed_step, k_lo, k_hi, boxes=()):
        self.t = {k: {} for k in ("sample", "sample_data", "ego_pose", "sample_annotation", "instance", "category")}
        self.yaw, self.start = yaw, np.asarray(start, float)
        d = np.array([math.cos(yaw), math.sin(yaw)])
        ks = list(range(k_lo, k_hi + 1))
        for i, k in enumerate(ks):
            self.t["ego_pose"][f"p{k}"] = dict(translation=[*(self.start + k * speed_step * d), 0.0], rotation=list(Quaternion(axis=[0, 0, 1], angle=yaw).elements))
            self.t["sample_data"][f"sd{k}"] = dict(ego_pose_token=f"p{k}", next=f"sd{ks[i + 1]}" if i + 1 < len(ks) else "", prev=f"sd{ks[i - 1]}" if i > 0 else "")
        anns = []
        for j, (cat, pos, size, n_lidar, n_radar) in enumerate(boxes):
            self.t["category"][f"c{j}"] = dict(name=cat)
            self.t["instance"][f"i{j}"] = dict(category_token=f"c{j}")
            self.t["sample_annotation"][f"a{j}"] = dict(num_lidar_pts=n_lidar, num_radar_pts=n_radar, instance_token=f"i{j}", size=size, rotation=list(Quaternion(axis=[0, 0, 1], angle=yaw).elements),
                                                        translation=[float(pos[0]), float(pos[1]), 0.8])
            anns.append(f"a{j}")
        self.t["sample"]["key"] = dict(data=dict(CAM_FRONT="sd0"), anns=anns)

    def get(self, table, token):
        return self.t[table][token]

    def T(self):
        """ego -> global of the keyframe (4 x 4)"""
        T = np.eye(4)
        T[:3, :3] = Quaternion(axis=[0, 0, 1], angle=self.yaw).rotation_matrix
        T[:2, 3] = self.start
        return T


def run(verbose=True):
    yaw, start, step = 0.7, (100.0, 50.0), 0.8                                           # a rotated ego frame: the lane frame is the ego frame turned by 90 degrees
    # 1. a car that drives: 25 m back, 55 m ahead, straight in the lane frame (x = 0, y forward)
    nv = FakeNusc(yaw, start, step, -60, 100)
    path = LT.ego_path(nv, "key", nv.T(), 25.0, 55.0, "lane")
    pts = path["pts"]
    assert np.abs(pts[:, 0]).max() < 1e-4 and (np.diff(pts[:, 1]) > 0).all(), "the driven path is a straight line forward in the lane frame, oldest first"
    assert abs(path["ahead_m"] - 55.0) < 1e-6 and abs(path["back_m"] - 25.0) < 1e-6 and abs(pts[np.argmin(np.abs(pts[:, 1]))][1]) < 1e-3, path["ahead_m"]
    assert pts[:, 1].min() <= -25.0 and pts[:, 1].max() >= 55.0, "the poses reach 25 m back and 55 m ahead"
    ego = LT.ego_path(nv, "key", nv.T(), 25.0, 55.0, "ego")["pts"]
    assert np.abs(ego[:, 1]).max() < 1e-4 and (np.diff(ego[:, 0]) > 0).all(), "the same path in the ego frame goes along x"
    # 2. the start of a log: nothing driven behind
    st = FakeNusc(yaw, start, step, 0, 100)
    ps = LT.ego_path(st, "key", st.T(), 25.0, 55.0, "lane")
    assert ps["back_m"] == 0.0 and abs(ps["ahead_m"] - 55.0) < 1e-6, ps
    # 3. a standing car: all poses equal -> one point, nothing driven either way
    sb = FakeNusc(yaw, start, 0.0, -20, 20)
    pb = LT.ego_path(sb, "key", sb.T(), 25.0, 55.0, "lane")
    assert len(pb["pts"]) <= 2 and np.abs(pb["pts"]).max() < 0.3 and pb["ahead_m"] == 0.0 and pb["back_m"] == 0.0, pb          # the keyframe position is always kept: at most the first pose and it, both at the car
    # 4. the box prisms: an observed car 20 m ahead (ego frame x 17.75 .. 22.25, y +-1), an unobserved one, a barrier
    d = np.array([math.cos(yaw), math.sin(yaw)])
    car = np.asarray(start) + 20.0 * d
    nb = FakeNusc(yaw, start, step, -60, 100, boxes=[("vehicle.car", car, [2.0, 4.5, 1.6], 12, 0), ("vehicle.car", car + 8 * d, [2.0, 4.5, 1.6], 0, 0), ("movable_object.barrier", car - 8 * d, [2.0, 4.5, 1.0], 9, 0),
                                                      ("human.pedestrian.adult", car + 3 * np.array([-d[1], d[0]]), [0.7, 0.7, 1.7], 0, 4)])
    pr = LT.box_prisms(nb, "key", nb.T())
    assert len(pr) == 2, f"the observed car and the pedestrian with radar points hide, the empty box and the barrier do not: {len(pr)}"
    fp = pr[0][0]
    assert abs(fp[:, 0].min() - 17.75) < 1e-3 and abs(fp[:, 0].max() - 22.25) < 1e-3 and abs(fp[:, 1].min() + 1.0) < 1e-3 and abs(fp[:, 1].max() - 1.0) < 1e-3 and pr[0][1:] == (0.0, 1.6), fp
    # 5. add_tiers end to end on a toy lane (the lane frame of the keyframe): the ego lane x = 0, hidden behind the car at 20 m
    A1 = LT._piece("A1", LT._line(0, -40, 0, 10), succ=["A2"], start_open=True)
    A2 = LT._piece("A2", LT._line(0, 10, 0, 50), end_open=True, pred=["A1"])
    g = dict(frame="lane", half=50.0, pieces=[A1, A2], events=[dict(type="split", pos=np.array([0.0, 10.0], np.float32), **{"in": ["A1"], "out": ["A2"]})])
    LT.add_tiers(g, nb, "key", nb.T(), P)
    for q in g["pieces"]:
        yq = q["pts"][:, 1]
        assert (q["tier"][yq >= -26.0] == 1).all() and len(q["vis"]) == len(q["pts"]) == len(q["occ"]), "the lane the car drove (from 25 m behind it) is tier 1"
    assert (A1["tier"][A1["pts"][:, 1] < -33.0] == 3).all(), "farther back than the poses reach (back + 5 m of poses are read) it is not: the car drove 25 m behind, so there is no map extension"
    y2 = A2["pts"][:, 1]
    assert g["pieces"][1]["occ"][(y2 > 25) & (y2 < 40)].all() and not g["pieces"][1]["vis"][(y2 > 25) & (y2 < 40)].any(), "the ground behind the car 20 m ahead is hidden by it"
    assert g["pieces"][1]["vis"][y2 < 14].all(), "the ground before the car is seen"
    assert g["events"][0]["tier"] == 1 and "ego_path" in g and g["tier_info"]["length_m"][1] > 80.0 and g["tier_info"]["hiding_boxes"] == 2, g["tier_info"]
    if verbose:
        print("lane_tiers_selftest (fake devkit): driven path in a rotated frame (lane and ego), start of a log, standing car, box prisms (observed hiding classes only), add_tiers end to end: ok")
    return True


if __name__ == "__main__":
    run()
