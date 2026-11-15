"""Test of the data side of `lane_tiers` on a FAKE nuScenes devkit (no data needed): the driven path from the CAM_FRONT ego poses (rotated frame, start of a log, a standing car), the box prisms
(footprint in the ego frame, observed boxes of the hiding classes only) and `add_tiers` end to end on a toy lane. The toy lane of `lane_tiers.selftest` covers the label rules; this covers the
frames and the table reading (critic pass 2026-11-07, missing test a); and (l6) that the tiers of a point do not depend on the window (`toy_window`).   python -m GtGenerator.lane_tiers_selftest"""
import math

import numpy as np
from pyquaternion import Quaternion

from . import lane_tiers as LT

from . import lane_gt  # noqa: E402

P = {k: lane_gt.DEFAULTS[k] for k in ("tier_back", "tier_ahead", "path_tol", "path_cos", "neighbour_tol", "min_ahead", "min_back", "extend", "opposite_tier")}      # the defaults, not a copy that drifts


class FakeNusc:
    """get(table, token) over dicts: one scene of CAM_FRONT sample_data at 12 Hz with their ego poses, one keyframe `key`, annotations"""

    def __init__(self, yaw, start, speed_step, k_lo, k_hi, boxes=()):
        self.t = {k: {} for k in ("sample", "sample_data", "ego_pose", "sample_annotation", "instance", "category", "scene", "log")}
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
        self.t["sample"]["key"] = dict(data=dict(CAM_FRONT="sd0"), anns=anns, scene_token="sc")
        self.t["scene"]["sc"] = dict(log_token="lg")
        self.t["log"]["lg"] = dict(location="toy")

    def get(self, table, token):
        return self.t[table][token]

    def T(self):
        """ego -> global of the keyframe (4 x 4)"""
        T = np.eye(4)
        T[:3, :3] = Quaternion(axis=[0, 0, 1], angle=self.yaw).rotation_matrix
        T[:2, 3] = self.start
        return T


def toy_window(verbose=True):
    """the tiers of a point do not depend on the window the graph is cut to (l6): the lane A | A2 is driven, a branch F leaves it at a shallow angle and stays within 1.5 m of the path for 4.3 m
    (less than the 6 m that make a piece 'on the path'; in a small window the part of F is so short that 0.3 of its length is less than that, which made it tier 1); the same keyframe is cut to
    the square of 22 m, 50 m, and the BEV Fusion rectangle: every point that two graphs share has the same tier and tier_src"""
    from shapely.geometry import box

    from . import lane_graph as LG, lane_lines as LL, road_graph as RG
    from .lane_graph_selftest import _Map, _line
    arcs = {"A": _line((0, -60), (0, 10)), "A2": _line((0, 10), (0, 120)), "F": _line((0, 10), (24, 70)), "B": _line((3.5, -60), (3.5, 120))}
    mg = LG.build_map_graph(_Map(arcs, {"A": ["A2", "F"], "A2": [], "F": [], "B": []}, connectors=(), boxes=[(False, box(-1, -1, 0, 0))]))
    rg = RG.build_road_graph(mg)
    rg["area_chunks"] = LL.polygon_chunks(rg["area_geoms"])
    nv = FakeNusc(math.pi / 2, (0.0, 0.0), 0.8, -60, 100)
    labels = {}
    for name, kw in (("small", dict(half=22.0)), ("square", dict(half=50.0)), ("rect", dict(rig="bev_fusion"))):
        lg = lane_gt.LaneGT("unused", "toy", nusc=nv, **kw)
        lg._maps["toy"] = (mg, rg, {"chunks": [], "road": []})
        g = LG.in_frame(lg.graph("key"), "lane")
        labels[name] = {(q["piece"], round(float(x), 3), round(float(y), 3)): (int(t), int(s)) for q in g["pieces"] for (x, y), t, s in zip(q["pts"], q["tier"], q["tier_src"])}
        assert "src" not in g["pieces"][0], "the vertex indices are not part of the stored graph"
    for a, b in (("small", "square"), ("rect", "square"), ("small", "rect")):
        both = labels[a].keys() & labels[b].keys()
        assert len(both) > 100, (a, b, len(both))
        diff = [k for k in both if labels[a][k] != labels[b][k]]
        assert not diff, f"the tiers of {len(diff)} of {len(both)} points shared by the windows {a} and {b} differ, e.g. {diff[:3]} {[(labels[a][k], labels[b][k]) for k in diff[:3]]}"
    f = [v for k, v in labels["square"].items() if k[0] == "F" and k[2] < 14.0]
    assert f and all(t == 2 for t, _ in f), f"the first metres of the branch F are neighbours of the driven lane (tier 2), not on it: {f[:5]}"
    # the label window pinned absolutely (critic pass 2026-11-11: TIER_HALF, TIER_MARGIN 100 / 40 -> 30 / 0 passed the comparison of the windows with each other): the labels are decided on the square
    # max(TIER_HALF, the bound of the rig's window + TIER_MARGIN), for every window of the rig
    seen, real_crop = [], lane_gt.LG.crop

    def spy(*a, **kw):
        if kw.get("full", True) is False:                                                    # the wide crop that the tiers are decided on (the graph's own crop has full=True)
            seen.append(kw["half"])
        return real_crop(*a, **kw)

    lane_gt.LG.crop = spy
    try:
        for kw, want in ((dict(half=22.0), 100.0), (dict(half=50.0), 100.0), (dict(rig="bev_fusion"), 100.0), (dict(half=80.0), 120.0)):
            lg = lane_gt.LaneGT("unused", "toy", nusc=nv, **kw)
            lg._maps["toy"] = (mg, rg, {"chunks": [], "road": []})
            seen.clear()
            lg.graph("key")
            assert seen == [want], f"the tiers of the window {kw} are decided on the square half {seen}, wanted {[want]} (lane_gt.TIER_HALF {lane_gt.TIER_HALF}, TIER_MARGIN {lane_gt.TIER_MARGIN})"
    finally:
        lane_gt.LG.crop = real_crop
    if verbose:
        print("lane_tiers_selftest (toy windows): the tiers of the points that the windows 22 m, 50 m and the BEV Fusion rectangle share are equal, decided on a 100 m window (120 m for half 80): ok")
    return True


def toy_carry(verbose=True):
    """carry_tiers on hand-made pieces (critic pass 2026-11-11: two rules were pinned by no test): a point of the small graph takes the label of the VERTEX of its map piece that it stands for (`src`),
    not that of the cut point that stands for the same vertex (the cut point comes first in the wide graph and lies elsewhere along the piece); both `tier` and `tier_src` (the extension, 2) are carried;
    a piece that the wide graph does not cover is an error"""
    wide = dict(pieces=[dict(id="W", piece="X", src=np.array([4, 5, 5, 6, 7]), tier=np.array([3, 3, 1, 1, 2], np.uint8), tier_src=np.array([0, 0, 2, 1, 0], np.uint8))])
    g = dict(pieces=[dict(id="X", piece="X", src=np.array([5, 6, 7]))])
    LT.carry_tiers(wide, g)
    q = g["pieces"][0]
    assert q["tier"].tolist() == [1, 1, 2], f"the point that stands for vertex 5 takes the label of the vertex (1), not of the cut point before it (3): {q['tier'].tolist()}"
    assert q["tier_src"].tolist() == [2, 1, 0], f"tier_src (2 = the map extension) is carried: {q['tier_src'].tolist()}"
    for bad in (dict(id="Y", piece="Y", src=np.array([5, 6])), dict(id="X", piece="X", src=np.array([2, 5]))):
        try:
            LT.carry_tiers(wide, dict(pieces=[bad]))
        except ValueError:
            continue
        raise AssertionError(f"a piece that the wide graph does not cover must be refused: {bad['id']} {bad['src'].tolist()}")
    if verbose:
        print("lane_tiers_selftest (carry_tiers): the vertex label, tier_src and the refusal of an uncovered piece: ok")
    return True


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
    toy_window(verbose=False)
    toy_carry(verbose=False)
    if verbose:
        print("lane_tiers_selftest (fake devkit): driven path in a rotated frame (lane and ego), start of a log, standing car, box prisms (observed hiding classes only), add_tiers end to end, "
              "the tiers of a point the same in the windows 22 m / 50 m / the BEV Fusion rectangle: ok")
    return True


if __name__ == "__main__":
    run()
