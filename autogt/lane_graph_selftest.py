"""Self-test of the lane graph (lane_graph.py, road_graph.py, lane_lines.py): a made-up map for the logic, nuScenes mini for the checks against the map.

    python -m autogt.lane_graph_selftest [--root ~/data/nuscenes]      (also run by `python -m autogt selftest`)
"""
import collections
import json
import math
import os

import numpy as np

from . import lane_graph as LG
from . import lane_lines as LL
from . import road_graph as RG


# ---------------------------------------------------------------------------------------------------------------- a made-up map
def _line(p0, p1, step=0.5):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(2, int(round(np.linalg.norm(p1 - p0) / step)) + 1)
    return p0 + np.linspace(0, 1, n)[:, None] * (p1 - p0)


def _curve(p0, p1, p2, step=0.5):
    t = np.linspace(0, 1, int(round(np.linalg.norm(np.asarray(p2) - np.asarray(p0)) / step)) + 1)[:, None]
    return (1 - t) ** 2 * np.asarray(p0, float) + 2 * t * (1 - t) * np.asarray(p1, float) + t ** 2 * np.asarray(p2, float)


class _Map:
    """just enough of NuScenesMap: lanes A, H, D, E, F, X, Z; connectors B1, B2, Y inside an intersection.
    A -> B1 -> D, A -> B2 -> E -> F, H -> D (so D merges B1 and H, and A splits into B1 and B2 E F); X -> Y -> Z (one joined piece with a stop and an exit inside)"""
    def __init__(self):
        from shapely.geometry import box
        self.arcs = {"A": _line((0, 0), (0, 40)), "B1": _line((0, 40), (0, 60)), "B2": _curve((0, 40), (0, 50), (10, 50)),
                     "D": _line((0, 60), (0, 100)), "E": _line((10, 50), (50, 50)), "F": _line((50, 50), (60, 50)), "H": _line((-10, 60), (0, 60)),
                     "X": _line((-3, 10), (-3, 38)), "Y": _line((-3, 38), (-3, 62)), "Z": _line((-3, 62), (-3, 90))}
        self.lane = [{"token": t, "left_lane_divider_segments": [], "right_lane_divider_segments": []} for t in ("A", "D", "E", "F", "H", "X", "Z")]
        self.lane_connector = [{"token": t} for t in ("B1", "B2", "Y")]
        out = {"A": ["B1", "B2"], "B1": ["D"], "B2": ["E"], "E": ["F"], "H": ["D"], "D": [], "F": [], "X": ["Y"], "Y": ["Z"], "Z": []}
        inc = {t: [u for u, vs in out.items() if t in vs] for t in out}
        self.connectivity = {t: {"incoming": inc[t], "outgoing": out[t]} for t in out}
        self.road_segment = [{"is_intersection": True, "polygon_token": "x"}, {"is_intersection": False, "polygon_token": "y"}]
        self._poly = {"x": box(-5, 38, 15, 62), "y": box(-5, 0, 5, 36)}

    def discretize_lanes(self, tokens, step):
        return {t: [(x, y, 0.0) for x, y in self.arcs[t]] for t in tokens}

    def extract_polygon(self, token):
        return self._poly[token]


def _pose(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 3] = [x, y]
    return T


def _made_up():
    mg = LG.build_map_graph(_Map())
    P = mg["pieces"]
    assert sorted(P) == ["A", "B1", "B2", "D", "H", "X"], sorted(P)                          # B2 + E + F and X + Y + Z joined, nothing else
    assert P["X"]["tokens"] == ["X", "Y", "Z"]
    assert P["B2"]["tokens"] == ["B2", "E", "F"] and P["B2"]["kind"] == "mixed" and P["A"]["kind"] == "lane" and P["B1"]["kind"] == "connector"
    assert [s[0] for s in P["B2"]["spans"]] == ["B2", "E", "F"] and P["B2"]["spans"][0][1] == 0
    ev = collections.Counter(e["type"] for e in mg["events"])
    assert ev == {"split": 1, "merge": 1, "start": 3, "end": 3, "stop": 2, "exit": 3}, ev
    split = next(e for e in mg["events"] if e["type"] == "split")
    assert split["in"] == ["A"] and split["out"] == ["B1", "B2"] and np.allclose(split["pos"], (0, 40), atol=0.3) and split["dirs_out"].shape == (2, 2)
    merge = next(e for e in mg["events"] if e["type"] == "merge")
    assert merge["in"] == ["B1", "H"] and merge["out"] == ["D"] and np.allclose(merge["pos"], (0, 60), atol=0.3)
    stops = {e["on"]: e for e in mg["events"] if e["type"] == "stop"}
    assert sorted(stops) == ["A", "X"], stops
    stop = stops["A"]
    assert stop["on"] == "A" and stop["out"] == ["B1", "B2"] and stop["n_succ"] == 2 and np.allclose(stop["pos"], (0, 40), atol=0.1)
    assert stops["X"]["out"] == ["X"] and stops["X"]["in"] == ["X"] and stops["X"]["n_succ"] == 1 and np.allclose(stops["X"]["pos"], (-3, 38), atol=0.1) \
        and stops["X"]["dirs_out"].shape == (1, 2), stops["X"]                                  # a stop inside a joined piece: the piece goes on
    exits = {e["on"]: e for e in mg["events"] if e["type"] == "exit"}
    assert sorted(exits) == ["B2", "D", "X"], exits                                          # lane D (B1 and H lead into it), lane E inside B2 E F, lane Z inside X Y Z
    assert exits["D"]["in"] == ["B1"] and exits["D"]["out"] == ["D"] and exits["D"]["n_pred"] == 2 and np.allclose(exits["D"]["pos"], (0, 60), atol=0.1)
    assert exits["B2"]["in"] == ["B2"] and exits["B2"]["out"] == ["B2"] and exits["B2"]["n_pred"] == 1 and np.allclose(exits["B2"]["pos"], (10, 50), atol=0.1) \
        and exits["B2"]["idx"] == mg["pieces"]["B2"]["spans"][1][1] - 1, exits["B2"]                # inside a joined piece, at the start of E
    assert np.allclose(exits["X"]["pos"], (-3, 62), atol=0.1) and exits["X"]["dirs_in"].shape == (1, 2) and exits["X"]["dirs_out"].shape == (1, 2)
    for e in mg["events"]:                                                                   # one direction row per id, in the same order
        assert len(e["dirs_in"]) == len(e["in"]) and len(e["dirs_out"]) == len(e["out"]), e
    assert np.allclose(split["dirs_out"][0], (0, 1), atol=0.05) and split["out"][0] == "B1", "the rows follow the order of the ids (B1 goes straight on)"
    # the rules for a lane with several successors
    one = LG.build_map_graph(_Map(), stop_rule="none")
    assert not [e for e in one["events"] if e["type"] in ("stop", "exit")]
    anyb = LG.build_map_graph(_Map(), stop_rule="any_successor")
    assert next(e for e in anyb["events"] if e["type"] == "exit" and e["on"] == "D")["in"] == ["B1", "H"], "any: all branches when one of them is an intersection connector"
    allb = LG.build_map_graph(_Map(), stop_rule="all_successors")
    assert not [e for e in allb["events"] if e["type"] == "exit" and e["on"] == "D"] and [e for e in allb["events"] if e["type"] == "stop" and e["on"] == "A"]
    try:
        LG.build_map_graph(_Map(), stop_rule="nope")
        raise AssertionError("an unknown stop rule must be refused")
    except ValueError:
        pass
    # ego at (0, 20) looking north: the lane frame is x right (east), y forward (north)
    g = LG.crop(mg, _pose(0, 20, math.pi / 2), half=50.0)
    by = {q["id"]: q for q in g["pieces"]}
    assert set(by) == {"A", "B1", "B2", "D", "H", "X"} and g["frame"] == "lane"
    assert {q["id"] for q in LG.crop(mg, _pose(0, 20, math.pi / 2), half=50.0, min_len=100.0)["pieces"]} == set(by), \
        "a part with a natural end is kept whatever min_len says (only a fragment cut at both ends is dropped)"
    assert np.allclose(by["A"]["pts"][0], (0, -20), atol=1e-4) and np.allclose(by["A"]["pts"][-1], (0, 20), atol=1e-4)
    assert by["A"]["succ"] == ["B1", "B2"] and by["B1"]["pred"] == ["A"] and by["D"]["pred"] == ["B1", "H"] and by["D"]["succ"] == []
    assert by["D"]["end_open"] and not by["D"]["start_open"] and np.allclose(by["D"]["pts"][-1], (0, 50), atol=1e-4)        # cut at y = 50
    assert by["B2"]["pts"][-1][0] > 0 and by["H"]["pts"][0][0] < 0                                                        # east is positive x
    w = [e for e in g["events"] if e["type"] == "window"]
    assert len(w) == 3 and {e["io"] for e in w} == {"out"} and all(abs(e["pos"][1] - 50) < 1e-4 or abs(e["pos"][0]) > 49.9 for e in w), w
    for e in g["events"]:
        assert len(e["dirs_in"]) == len(e["in"]) and len(e["dirs_out"]) == len(e["out"]), e
    sx = {e["type"]: e for e in g["events"] if e["type"] in ("stop", "exit") and e.get("on") == "X"}
    assert sx["stop"]["out"] == ["X"] and abs(sx["stop"]["s"] - 28.0) < 0.3 and sx["exit"]["in"] == ["X"] and abs(sx["exit"]["s"] - 52.0) < 0.3, sx   # inside the joined piece
    # the segments of every part: index ranges per map piece, contiguous, with the intersection flag of the connectors
    sg = by["X"]["segments"]
    assert [s["token"] for s in sg] == ["X", "Y", "Z"] and [s["kind"] for s in sg] == ["lane", "connector", "lane"] and [s["in_intersection"] for s in sg] == [False, True, False]
    assert sg[0]["i0"] == 0 and sg[-1]["i1"] == len(by["X"]["pts"]) - 1 and all(b["i0"] == a["i1"] + 1 for a, b in zip(sg[:-1], sg[1:])), sg
    assert [s["token"] for s in by["B2"]["segments"]] == ["B2", "E"] and [s["in_intersection"] for s in by["B2"]["segments"]] == [True, False],         "F lies beyond the window border (x = 50), the part B2 E ends there"
    assert [s["token"] for s in by["A"]["segments"]] == ["A"] and by["A"]["segments"][0]["i1"] == len(by["A"]["pts"]) - 1
    ego = LG.crop(mg, _pose(0, 20, math.pi / 2), half=50.0, frame="ego")
    assert np.allclose(LG.in_frame(g, "ego")["pieces"][0]["pts"], ego["pieces"][0]["pts"], atol=1e-5), "lane -> ego by rotation"
    back = LG.in_frame(LG.in_frame(g, "ego"), "lane")
    assert np.allclose(back["pieces"][1]["pts"], g["pieces"][1]["pts"], atol=1e-5) and np.allclose(back["events"][0]["pos"], g["events"][0]["pos"], atol=1e-5)
    stop_c = next(e for e in g["events"] if e["type"] == "stop" and e["on"] == "A")
    assert stop_c["on"] == "A" and abs(stop_c["s"] - 40.0) < 0.3 and np.allclose(stop_c["pos"], (0, 20), atol=0.1)
    # a window that cuts a piece in two parts: ids "<piece>#<part>", and a stub shorter than min_len is dropped
    P = np.array([[-10.0, 0], [10, 0], [10, 3], [-10, 3]])              # crosses the square |x| <= 5 twice (there and back)
    parts = LG.clip_square(P, 5.0)
    assert len(parts) == 2 and all(p[1] and p[2] for p in parts) and np.allclose(parts[0][0], [[-5, 0], [5, 0]]) and np.allclose(parts[1][0], [[5, 3], [-5, 3]])
    cut = LG.clip_square(np.array([[-8.0, 0], [8, 0]]), 5.0)
    assert len(cut) == 1 and cut[0][1] and cut[0][2] and np.allclose(cut[0][0], [[-5, 0], [5, 0]])
    assert not LG.clip_square(np.array([[20.0, 20], [30, 20]]), 5.0) and len(LG.clip_square(np.array([[0.0, 0], [0, 0]]), 5.0)) <= 1
    return mg, g


# ---------------------------------------------------------------------------------------------------------------- the lines
def _lines():
    L = np.vstack([_line((0, 0), (10, 0)), _line((10, 0.5), (10, 10))])                  # a corner of 90 degrees at (10, 0)
    cs = LL.find_corners(L, 30.0, 2.0)
    assert len(cs) == 1 and np.linalg.norm(cs[0]["pos"] - (10, 0)) < 0.6 and abs(cs[0]["angle"] - 90) < 12, cs
    assert not LL.find_corners(_line((0, 0), (50, 0)), 30.0) and not LL.find_corners(_curve((0, 0), (20, 0), (20, 20)), 30.0), "straight and a gentle curve"
    assert LL.find_corners(_curve((0, 0), (3, 0), (3, 3)), 30.0), "a tight curve counts"
    # a long outline in chunks comes back as one line
    old = LL.CHUNK
    LL.CHUNK = 10
    try:
        long = _line((-30, 1), (30, 1), 0.5)
        lg = {"chunks": LL._chunks(long, "boundary"), "params": {}}
        for i, c in enumerate(lg["chunks"]):
            c["line"] = 0
        out = LL.crop_lines(lg, _pose(0, 0, math.pi / 2), half=20.0)
        assert len(out["boundary"]) == 1 and out["boundary"][0]["start_open"] and out["boundary"][0]["end_open"], out["boundary"]
        assert abs(out["boundary"][0]["pts"][:, 0].min() + 20) < 1e-4 and abs(out["boundary"][0]["pts"][:, 0].max() - 20) < 1e-4 and np.allclose(out["boundary"][0]["pts"][:, 1], 1.0)       # lane frame: x right = east
        ring = np.vstack([_line((-10, -10), (30, -10)), _line((30, -10), (30, 30))[1:], _line((30, 30), (-10, 30))[1:], _line((-10, 30), (-10, -10))[1:]])
        lg = {"chunks": LL._chunks(ring, "boundary"), "params": {}}
        for c in lg["chunks"]:
            c["line"] = 7
        out = LL.crop_lines(lg, _pose(0, 0, math.pi / 2), half=15.0)
        assert len(lg["chunks"]) > 3 and lg["chunks"][0]["closed"] and len(out["boundary"]) == 1, out["boundary"]
        r = out["boundary"][0]
        assert r["start_open"] and r["end_open"] and np.allclose(r["pts"][0], (-10, 15), atol=1e-4) and np.allclose(r["pts"][-1], (15, -10), atol=1e-4), r["pts"][[0, -1]]
        assert any(np.allclose(p, (-10, -10)) for p in r["pts"]) and len(r["corners"]) >= 1, "the closure vertex is inside the line and the corner there is found"
    finally:
        LL.CHUNK = old


def _areas():
    """polygons: cut to the window, holes kept, orientation fixed, tiles of a big polygon give the same area, frames"""
    from shapely.geometry import Polygon, box
    from shapely.geometry.polygon import orient
    T = _pose(10, 10, math.pi / 2)
    out = LL.crop_areas(LL.polygon_chunks([("a", box(0, 0, 20, 20))]), T, half=5.0)
    assert len(out) == 1 and out[0]["id"] == "a" and abs(out[0]["area"] - 100.0) < 1e-3 and np.abs(out[0]["exterior"]).max() <= 5.0 + 1e-5, out
    ring = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)], [[(8, 8), (8, 12), (12, 12), (12, 8)]])          # a hole of 16 m2 at the centre of the window
    out = LL.crop_areas(LL.polygon_chunks([(None, ring)]), T, half=5.0)
    assert len(out) == 1 and out[0]["id"] is None and len(out[0]["holes"]) == 1 and abs(out[0]["area"] - 84.0) < 1e-3, out
    sa = lambda P: 0.5 * float(np.sum(P[:-1, 0] * P[1:, 1] - P[1:, 0] * P[:-1, 1]))
    assert sa(out[0]["exterior"].astype(float)) > 0 > sa(out[0]["holes"][0].astype(float)), "exterior counter-clockwise, holes clockwise"
    # a polygon 600 m wide is cut into tiles; the cut window has the same area as a direct cut, with and without the tiles
    big = Polygon([(-300, -300), (300, -300), (300, 300), (-300, 300)], [[(-30, -30), (-30, 30), (30, 30), (-0.0, -30)]])
    T2 = _pose(120.0, 40.0, 0.3)
    direct = LL.crop_areas([dict(id=None, poly=big, bbox=np.array(big.bounds))], T2, half=50.0)
    tiled = LL.crop_areas(LL.polygon_chunks([(None, big)]), T2, half=50.0, merge=True)
    assert len(LL.polygon_chunks([(None, big)])) > 20 and abs(sum(d["area"] for d in direct) - sum(d["area"] for d in tiled)) < 1e-2, (direct, tiled)
    # the frame: a polygon east of the ego (x_global > 0, ego looking north) is at positive x in the lane frame and at negative y in the ego frame
    east = LL.crop_areas(LL.polygon_chunks([("e", box(3, -2, 8, 2))]), _pose(0, 0, math.pi / 2), half=20.0)
    assert np.allclose([east[0]["exterior"][:, 0].min(), east[0]["exterior"][:, 0].max()], [3, 8], atol=1e-4) and np.allclose(east[0]["exterior"][:, 1].max(), 2, atol=1e-4)
    ego = LL.crop_areas(LL.polygon_chunks([("e", box(3, -2, 8, 2))]), _pose(0, 0, math.pi / 2), half=20.0, frame="ego")
    assert np.allclose([ego[0]["exterior"][:, 1].min(), ego[0]["exterior"][:, 1].max()], [-8, -3], atol=1e-4)
    g_lane = {"frame": "lane", "areas": {"road": east}}
    back = LG.in_frame(LG.in_frame(g_lane, "ego"), "lane")
    assert np.allclose(back["areas"]["road"][0]["exterior"], east[0]["exterior"], atol=1e-5)
    assert np.allclose(LG.in_frame(g_lane, "ego")["areas"]["road"][0]["exterior"], ego[0]["exterior"], atol=1e-4), "in_frame turns polygons like crop_areas does"


# ---------------------------------------------------------------------------------------------------------------- checks against the map
def _junction_f1(g, mg, T_e2g, half):
    """events and links against the map: the junction points the events name vs the nodes of the raw map graph (>= 2 successors or >= 2
    predecessors) inside the window; and the successor links of the crop vs the map's links between pieces that are both inside"""
    ev = [e["pos"] for e in g["events"] if e["type"] in ("split", "merge", "cross")]
    truth = []
    for e in mg["events"]:
        if e["type"] in ("split", "merge", "cross"):
            pos = LG.to_frame(e["pos"][None], T_e2g, g["frame"])[0]
            if abs(pos[0]) <= half - 1.0 and abs(pos[1]) <= half - 1.0:
                truth.append(pos)
    hit = sum(1 for t in truth if any(np.linalg.norm(t - p) < 1.0 for p in ev))
    pr = sum(1 for p in ev if any(np.linalg.norm(t - p) < 1.0 for t in truth))
    rec = hit / max(len(truth), 1)
    prec = pr / max(len(ev), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9), len(truth)


def run(root=None, verbose=True):
    import hashlib
    import tempfile
    from . import lane_gt
    from .checks import _feed
    from .grid import GridSpec
    from .nuscenes_gt import SIGNATURE_FILES
    from .suite import AutoGT
    root = os.path.expanduser(root or "~/data/nuscenes")
    mg, g = _made_up()
    _lines()
    _areas()
    # neighbours: three lanes side by side, going the same way, and an opposite lane
    polys = {"a": _line((0, 0), (0, 30)), "b": _line((3.5, 0), (3.5, 30)), "c": _line((7, 0), (7, 30)), "o": _line((-3.5, 30), (-3.5, 0)),
             "p": _line((3.5, 30), (3.5, 0)), "far": _line((30, 0), (30, 30))}
    nb = RG.neighbours(polys, {t: "lane" for t in polys})
    assert nb["right"]["a"] == ["b"] and nb["left"]["b"] == ["a"] and nb["right"]["b"] == ["c"] and nb["left"]["a"] == [] and nb["opposite"]["a"] == ["o", "p"], nb       # the opposite lane on the left AND on the right
    assert nb["lane_count"]["a"] == nb["lane_count"]["b"] == nb["lane_count"]["c"] == 3 and nb["lane_index"] == {"a": 0, "b": 1, "c": 2, "o": 0, "p": 0, "far": 0} \
        and nb["lane_count"]["far"] == 1, nb
    # the lane GT has its own version: nothing of it is in the signature of the front GT, a parameter makes another signature, an unknown one is refused
    assert not set(lane_gt.SOURCE_FILES) & set(SIGNATURE_FILES), "the lane GT must not change the front GT's signature"
    assert lane_gt.signature() == lane_gt.signature(boundary_layers=["drivable_area"]) != lane_gt.signature(stop_rule="none") != lane_gt.signature(corner_deg=45.0)
    try:
        lane_gt.signature(corners=1)
        raise AssertionError("an unknown parameter must be refused")
    except TypeError:
        pass
    for rig in lane_gt.RIGS:
        assert lane_gt.version_of(lane_gt.code_signature(rig=rig)) is not None, (f"the lane GT source has the signature {lane_gt.code_signature(rig=rig)} for the rig {rig} but no row in "
                                                                                  "autogt/lane_versions.csv: add one, commit, tag autogt/lane-l<N>")
    gt = AutoGT(root, "v1.0-mini", light=True, scenes=["scene-0061", "scene-0103", "scene-1077"])
    lgt = lane_gt.LaneGT(root, "v1.0-mini", nusc=gt.nusc)
    n_frames, d_graph, d_ref, f1s, divpairs, found = 0, [], [], [], 0, 0
    n_lane_pts = n_lane_on_road = n_conn = n_conn_in = 0
    for name in ("scene-0061", "scene-0103", "scene-1077"):
        for tok in gt.keyframes(name)[::10]:
            lg = lgt.graph(tok)
            g_ = gt.keyframe(tok)
            T = g_["meta"]["T_e2g"]
            assert lg["frame"] == "lane" and lg["half"] == 50.0 and lg["location"] and lg["sample"] == tok and np.allclose(T, lgt.pose(tok))
            ego = LG.in_frame(lg, "ego")
            ref = np.concatenate([_dense(v) for v in g_["meta"]["vectors"]["centreline"] if len(v) > 1])
            mine = np.concatenate([_dense(q["pts"]) for q in ego["pieces"]]) if ego["pieces"] else np.zeros((0, 2))
            grid = GridSpec()
            inbox = lambda P: P[(P[:, 0] >= grid.x_min + 0.5) & (P[:, 0] <= 49.5) & (np.abs(P[:, 1]) <= grid.y_max - 0.5)]
            a, b = inbox(ref), inbox(mine)
            if len(a) and len(b):
                d_ref += list(_nearest(a, mine))
                d_graph += list(_nearest(b, ref))
            f1, n_truth = _junction_f1(lg, lgt._maps[lg["location"]][0], T, 50.0)
            if n_truth:
                f1s.append(f1)
            from shapely.geometry import MultiPolygon, Point, Polygon
            road = MultiPolygon([Polygon(p["exterior"], p["holes"]) for p in lg["areas"]["road"]]).buffer(0.5) if lg["areas"]["road"] else None
            inter = [Polygon(p["exterior"], p["holes"]) for p in lg["areas"]["intersection"]]
            for q in lg["pieces"]:                                    # lanes lie on the drivable area, connectors in an intersection lie in an intersection area
                P = q["pts"][(np.abs(q["pts"]) < 49.0).all(1)]
                if q["kind"] == "lane" and len(P) and road is not None:
                    n_lane_pts += len(P)
                    n_lane_on_road += sum(road.contains(Point(*p)) for p in P[::4]) * 4
                if q["kind"] == "connector" and len(P):
                    mid = P[len(P) // 2]
                    if str(q.get("road")).startswith("x"):
                        n_conn += 1
                        n_conn_in += any(poly.buffer(0.5).contains(Point(*mid)) for poly in inter)
            for e in lg["events"]:
                assert len(e["dirs_in"]) == len(e["in"]) and len(e["dirs_out"]) == len(e["out"]), e
                assert e["type"] not in ("stop", "exit") or (e["in"] and e["out"]), e                              # a stop / exit always names where it leads
            for q in lg["pieces"]:
                sg = q["segments"]
                assert sg and sg[0]["i0"] == 0 and sg[-1]["i1"] == len(q["pts"]) - 1 and all(b["i0"] == a["i1"] + 1 for a, b in zip(sg[:-1], sg[1:])), q["id"]
            ids = {x["id"] for x in lg["pieces"]}
            for q in lg["pieces"]:                                    # every id of a link names a piece of the crop
                assert set(q["succ"]) <= ids and set(q["pred"]) <= ids and set(q["left"]) <= ids and set(q["right"]) <= ids, q["id"]
            n_frames += 1
    d_ref, d_graph = np.array(d_ref), np.array(d_graph)
    assert np.median(d_graph) < 0.05 and np.percentile(d_graph, 99) < 0.25, (np.median(d_graph), np.percentile(d_graph, 99))      # on the lane graph
    assert np.percentile(d_ref, 99) < 0.25, np.percentile(d_ref, 99)                                                                # nothing of it missing
    assert np.mean(f1s) > 0.95 and min(f1s) > 0.8, (np.mean(f1s), min(f1s))                                                         # events + links
    assert n_lane_on_road >= 0.97 * n_lane_pts > 0 and n_conn_in >= 0.95 * n_conn > 0, (n_lane_on_road, n_lane_pts, n_conn_in, n_conn)      # areas and graph agree
    # neighbours against the map's own divider tokens (the lanes that have them): a.right divider == b.left divider means b is right of a
    for loc, (mgm, rgm, lgm) in lgt._maps.items():
        nmap = gt.map_for(next(s["token"] for s in gt.nusc.scene if gt.nusc.get("log", s["log_token"])["location"] == loc))
        left, right = collections.defaultdict(set), collections.defaultdict(set)
        for l in nmap.lane:
            for s in l["left_lane_divider_segments"]:
                left[s["node_token"]].add(l["token"])
            for s in l["right_lane_divider_segments"]:
                right[s["node_token"]].add(l["token"])
        truth = {(a, b) for n in right if n in left for a in right[n] for b in left[n] if a != b}
        mine = {(a, b) for a, bs in rgm["nb"]["right"].items() for b in bs}
        assert len(truth) > 100 and len(truth & mine) >= 0.95 * len(truth), (loc, len(truth), len(truth & mine))
        assert not truth & {(a, b) for a, bs in rgm["nb"]["left"].items() for b in bs}, "left and right are not swapped"
        divpairs += len(truth)
        found += len(truth & mine)

    # the store: generate (resumable), load, the manifest, another parameter = another folder
    def digest(x):
        h = hashlib.sha1()
        _feed(h, x)
        return h.hexdigest()
    toks = gt.keyframes("scene-0061")
    with tempfile.TemporaryDirectory() as d:
        r = lane_gt.generate(root, out=d, scenes=["scene-0061"], verbose=False)
        assert r["written"] == len(toks) == r["keyframes"] and r["signature"] == lane_gt.signature() and not r["release"], r
        index_file = os.path.join(d, lane_gt.signature(), "index.json")
        index_before = open(index_file).read()
        assert lane_gt.generate(root, out=d, scenes=["scene-0061"], verbose=False)["written"] == 0, "a second run has nothing to do"
        from . import versions as versions_mod
        if versions_mod.code_state()["dirty"] is not False:                      # a clean tree may turn the index into a release, then it is written again
            assert open(index_file).read() == index_before, "a second run leaves the index of a complete store as it is (commit, environment of the run that made it)"
        open(index_file, "w").write("{damaged")
        assert lane_gt.generate(root, out=d, scenes=["scene-0061"], verbose=False)["written"] == 0 and json.load(open(index_file))["complete"], "a damaged index is written again"
        assert lane_gt.missing(toks, data=d) == [] and lane_gt.missing(toks + ["0" * 32], data=d) == ["0" * 32]
        assert digest(lane_gt.load(toks[5], data=d)) == digest(lgt.graph(toks[5])), "the stored graph is the graph made now"
        # l4 stays l4: the graphs of the rig lane_center are pinned by digest (made from the files of the stored l4 store, data/autogt/lane/0702a065c6); a change of the code that
        # moves one of them must be a new lane version with its own store (and a new pinned file), not a row `same_gt_as`
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lane_l4_digest.json")) as f:
            pin = json.load(f)
        have = {t for n_ in ("scene-0061", "scene-0103") for t in gt.keyframes(n_)}
        moved = [t for t, want in pin["frames"].items() if t in have and digest(lgt.graph(t)) != want]
        assert len([t for t in pin["frames"] if t in have]) >= 25 and not moved, f"lane_center is no longer the l4 GT on {len(moved)} pinned frames, e.g. {moved[:2]}: {pin['note']}"
        idx = json.load(open(os.path.join(d, lane_gt.signature(), "index.json")))
        assert idx["complete"] and idx["keyframes"] == len(toks) and idx["scenes"] == ["scene-0061"] and idx["params"]["stop_rule"] == "intersection_branch" \
            and idx["lane_version"] == lane_gt.version_of(lane_gt.signature()) and "shapely" in idx["environment"], idx
        r2 = lane_gt.generate(root, out=d, scenes=["scene-0061"], verbose=False, stop_rule="none")
        assert r2["signature"] != r["signature"] and r2["written"] == len(toks), "another parameter, another folder"
        assert not [e for e in lane_gt.load(toks[5], data=d, stop_rule="none")["events"] if e["type"] == "stop"] and \
            [e for e in lane_gt.load(toks[5], data=d)["events"] if e["type"] == "stop"], "the stop rule changes the events"
        try:
            lane_gt.load("0" * 32, data=d)
            raise AssertionError("a missing graph must be refused")
        except FileNotFoundError as e:
            assert lane_gt.signature() in str(e)
    rig_msg = _rigs(root, gt, lane_gt, GridSpec())
    msg = (f"lane graph selftest ok: {n_frames} mini frames, graph points {np.median(d_graph):.3f} m (99 % {np.percentile(d_graph, 99):.3f}) from the map's "
           f"centre-lines and {np.percentile(d_ref, 99):.3f} m (99 %) the other way, junction F1 {np.mean(f1s):.3f}, "
           f"neighbours: {found} of {divpairs} divider-token pairs found; store, load, resume, parameters ok (lane signature {lane_gt.signature()}); {rig_msg}")
    if verbose:
        print(msg)
    return msg


def _rigs(root, gt, lane_gt, grid):
    """the two rigs (l5): the rectangular window, the table of stores (same_gt_as), BEV Fusion's graphs on real mini frames against the Spurmitte rig (same map vertices, same tiers)"""
    from . import lane_tiers as LT
    from . import rig_masks as RM
    # 1. clip_rect: the square is the special case, a rectangle cuts at its four sides, a polyline inside is not cut
    P = np.array([[-10.0, 0], [10, 0], [10, 3], [-10, 3]])
    for a, b in zip(LG.clip_rect(P, -5.0, 5.0, -5.0, 5.0), LG.clip_square(P, 5.0)):
        assert np.array_equal(a[0], b[0]) and a[1:] == b[1:], "clip_square is clip_rect of the square"
    win = (0.0, 60.0, -26.0, 26.0)
    one = LG.clip_rect(np.array([[-10.0, 3], [70, 3]]), *win)
    assert len(one) == 1 and one[0][1] and one[0][2] and np.allclose(one[0][0], [[0, 3], [60, 3]]), one
    side = LG.clip_rect(np.array([[10.0, -40], [10, 40]]), *win)
    assert len(side) == 1 and np.allclose(side[0][0], [[10, -26], [10, 26]]), side
    inside = LG.clip_rect(np.array([[1.0, 0], [2, 1], [3, 1]]), *win)
    assert len(inside) == 1 and not inside[0][1] and not inside[0][2] and len(inside[0][0]) == 3, inside
    assert not LG.clip_rect(np.array([[-9.0, 0], [-1, 5]]), *win) and not LG.clip_rect(np.array([[10.0, 30], [20, 40]]), *win), "outside the window: nothing"
    # 2. the rigs and the table of stores
    assert lane_gt.RIGS["bev_fusion"]["window"] == (grid.x_min, grid.x_max, grid.y_min, grid.y_max), "the BEV Fusion window is the window of its BEV grid"
    assert lane_gt.RIGS["lane_center"]["window"] is None and lane_gt.RIGS["bev_fusion"]["cameras"] == ("CAM_FRONT",) and lane_gt.RIGS["lane_center"]["cameras"] == RM.SIX
    assert lane_gt.signature(rig="bev_fusion") != lane_gt.signature() and lane_gt.signature(rig="lane_center") == lane_gt.signature()
    for bad in (dict(rig="surround"), dict(rig="bev_fusion", half=40.0)):
        try:
            lane_gt.signature(**bad)
            raise AssertionError(f"{bad} must be refused")
        except ValueError:
            pass
    saved = lane_gt.table
    lane_gt.table = lambda: [dict(version="l4", signature="aaaaaaaaaa", rig="lane_center", same_gt_as=""), dict(version="l5", signature="bbbbbbbbbb", rig="lane_center", same_gt_as="l4"),
                             dict(version="l5", signature="cccccccccc", rig="bev_fusion", same_gt_as=""), dict(version="l6", signature="dddddddddd", rig="lane_center", same_gt_as="l5")]
    lane_gt._STORE_KEYS.clear()
    try:
        assert lane_gt.store_of("bbbbbbbbbb") == "aaaaaaaaaa" and lane_gt.store_of("dddddddddd") == "aaaaaaaaaa", "same_gt_as: the store of the earlier version, also through a chain"
        assert lane_gt.store_of("cccccccccc") == "cccccccccc" and lane_gt.store_of("eeeeeeeeee") == "eeeeeeeeee", "a row without same_gt_as and a signature without a row are their own store"
    finally:
        lane_gt.table = saved
        lane_gt._STORE_KEYS.clear()
    lane_gt.check_table()                                                                          # the real table is consistent
    for rows, why in (([dict(version="l4", signature="a", rig="lane_center", same_gt_as=""), dict(version="l5", signature="b", rig="surround", same_gt_as="")], "unknown rig"),
                      ([dict(version="l5", signature="b", rig="lane_center", same_gt_as="l5")], "names itself"),
                      ([dict(version="l5", signature="b", rig="bev_fusion", same_gt_as="l4"), dict(version="l4", signature="a", rig="lane_center", same_gt_as="")], "an earlier row of another rig / a later row"),
                      ([dict(version="l4", signature="a", rig="lane_center", same_gt_as=""), dict(version="l5", signature="a", rig="lane_center", same_gt_as="l4")], "one signature twice")):
        try:
            lane_gt.check_table(rows)
            raise AssertionError(f"check_table must refuse: {why}")
        except ValueError:
            pass
    saved = lane_gt.table
    lane_gt.table = lambda: [dict(version="l5", signature="x", rig="lane_center", same_gt_as="l6"), dict(version="l6", signature="y", rig="lane_center", same_gt_as="l5")]
    lane_gt._STORE_KEYS.clear()
    try:
        try:
            lane_gt.store_of("x")
            raise AssertionError("a cycle of same_gt_as must be refused")
        except ValueError as e:
            assert "cycle" in str(e), e
    finally:
        lane_gt.table = saved
        lane_gt._STORE_KEYS.clear()
    # 3. BEV Fusion's graphs on real mini frames
    bf_gt = lane_gt.LaneGT(root, "v1.0-mini", nusc=gt.nusc, rig="bev_fusion")
    big = lane_gt.LaneGT(root, "v1.0-mini", nusc=gt.nusc, half=60.0)                  # Spurmitte's rig with a bigger square, in the frame of BEV Fusion: the reference for the cut
    spur = lane_gt.LaneGT(root, "v1.0-mini", nusc=gt.nusc)
    eps = 1e-4
    n, share_same, share_n, vis_n, pts_n, d_len = 0, 0, 0, 0, 0, []
    for name in ("scene-0061", "scene-0103", "scene-1077"):
        for tok in gt.keyframes(name)[::10]:
            g = bf_gt.graph(tok)
            assert g["frame"] == "ego" and g["window"] == win and g["rig"] == "bev_fusion" and g["half"] == 60.0 and g["sample"] == tok, (g["frame"], g.get("window"))
            lane_view = LG.in_frame(g, "lane")                                                   # the window turns with the frame: x right, y forward
            assert lane_view["window"] == (-26.0, 26.0, 0.0, 60.0) and LG.in_frame(lane_view, "ego")["window"] == win, lane_view["window"]
            PL = np.concatenate([q["pts"] for q in lane_view["pieces"]]) if lane_view["pieces"] else np.zeros((0, 2))
            assert not len(PL) or (PL[:, 0].min() >= -26.0 - eps and PL[:, 0].max() <= 26.0 + eps and PL[:, 1].min() >= -eps and PL[:, 1].max() <= 60.0 + eps), "the points are inside the turned window"
            try:
                bf_gt.graph(tok, frame="lane")
                raise AssertionError("the rig bev_fusion is defined in the ego frame only")
            except ValueError:
                pass
            for q in g["pieces"]:
                x, y = q["pts"][:, 0], q["pts"][:, 1]
                assert x.min() >= win[0] - eps and x.max() <= win[1] + eps and y.min() >= win[2] - eps and y.max() <= win[3] + eps, "piece inside the window"
                assert len(q["tier"]) == len(q["vis"]) == len(q["occ"]) == len(q["pts"]) and set(np.unique(q["tier"])) <= {1, 2, 3}
            for kind, items in g["lines"].items():
                for it in items:
                    assert it["pts"][:, 0].min() >= win[0] - eps and it["pts"][:, 0].max() <= win[1] + eps and np.abs(it["pts"][:, 1]).max() <= win[3] + eps, f"line {kind} inside the window"
            for kind, polys in g["areas"].items():
                for poly in polys:
                    assert poly["exterior"][:, 0].min() >= win[0] - eps and poly["exterior"][:, 0].max() <= win[1] + eps and np.abs(poly["exterior"][:, 1]).max() <= win[3] + eps, f"area {kind} inside the window"
            for e in g["events"]:
                assert win[0] - eps <= e["pos"][0] <= win[1] + eps and win[2] - eps <= e["pos"][1] <= win[3] + eps and "tier" in e and "vis" in e, "events inside the window"
                if e["type"] == "window":
                    p = e["pos"]
                    assert min(abs(p[0] - win[0]), abs(p[0] - win[1]), abs(p[1] - win[2]), abs(p[1] - win[3])) < 1e-3, "a window event lies on the border"
            # the cut against the same graph made with a bigger square: the pieces of the big graph, cut to the rectangle, have the length of the pieces of the BEV Fusion graph
            gb = big.graph(tok, frame="ego")
            want = {}
            for q in gb["pieces"]:
                for pts, *_ in LG.clip_rect(q["pts"].astype(float), *win):
                    want[q["piece"]] = want.get(q["piece"], 0.0) + LG._length(pts)
            have = {}
            for q in g["pieces"]:
                have[q["piece"]] = have.get(q["piece"], 0.0) + q["length"]
            assert set(have) <= set(want), "a piece of the BEV Fusion graph is in the bigger graph"
            d_len.append((sum(want.values()), sum(have.values())))
            # the same points in the two rigs have the same tier (same map vertices, same ego path)
            sp = LG.in_frame(spur.graph(tok), "ego")
            ref = np.concatenate([q["pts"] for q in sp["pieces"]]) if sp["pieces"] else np.zeros((0, 2))
            ref_tier = np.concatenate([q["tier"] for q in sp["pieces"]]) if sp["pieces"] else np.zeros(0)
            ref_piece = np.array([q["piece"] for q in sp["pieces"] for _ in q["pts"]])
            for q in g["pieces"]:
                if not len(ref):
                    break
                d = np.linalg.norm(q["pts"][:, None, :] - ref[None, :, :], axis=2)
                j, dm = d.argmin(1), d.min(1)
                ok = (dm < 1e-3) & (ref_piece[j] == q["piece"])
                share_n += int(ok.sum())
                share_same += int((ok & (ref_tier[j] == q["tier"])).sum())
                pts_n += len(q["pts"])
            # the front camera: a visible point is in its image, a hidden-by-box point is in its view; the six cameras see at least what the front one sees
            for q in g["pieces"]:
                if q["vis"].any():
                    K, T = RM.camera_calibration("CAM_FRONT")
                    P = q["pts"][q["vis"]].astype(float)
                    assert LT.camera_cells(P[:, 0], P[:, 1], K, T, RM.IMAGE_SIZE).all(), "a visible point of the BEV Fusion rig is in the image of the front camera"
                    vis_n += int(q["vis"].sum())
                    six, _ = LT.visibility(q["pts"].astype(float), LT.box_prisms(gt.nusc, tok, spur.pose(tok)))
                    assert six[q["vis"]].all(), "what the front camera sees the six cameras see"
            n += 1
    d_len = np.array(d_len)
    assert vis_n > 0 and n >= 9, (vis_n, n)
    assert np.all(np.abs(d_len[:, 0] - d_len[:, 1]) <= 0.02 * np.maximum(d_len[:, 0], 1.0) + 2.0), d_len
    assert share_n >= 0.8 * pts_n > 0 and share_same >= 0.98 * share_n, (share_same, share_n, pts_n)         # measured 0.992: the rest are tier-2 lanes whose tier-1 neighbour lies outside the smaller window
    return f"BEV Fusion rig: {n} frames inside x {win[0]:.0f} .. {win[1]:.0f} m, y {win[2]:.0f} .. {win[3]:.0f} m, cut = the bigger graph cut, tier equal to Spurmitte's on {share_same / share_n:.3f} of {share_n} shared points"


def _dense(P, ds=0.1):
    P = np.asarray(P, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    u = np.arange(0, s[-1], ds)
    return np.c_[np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1])] if len(u) else P[:1]


def _nearest(A, B):
    """distance of every point of A to its nearest point of B (blocks, no scipy)"""
    out = np.empty(len(A))
    for i in range(0, len(A), 512):
        d = np.linalg.norm(A[i:i + 512, None, :] - B[None, :, :], axis=2)
        out[i:i + 512] = d.min(1)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/data/nuscenes")
    run(ap.parse_args().root)
