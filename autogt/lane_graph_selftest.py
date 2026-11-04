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
    """just enough of NuScenesMap: lanes A, H and D, E, F; connectors B1, B2 inside an intersection.
    A -> B1 -> D, A -> B2 -> E -> F, H -> D (so D merges B1 and H, and A splits into B1 and B2 E F)"""
    def __init__(self):
        from shapely.geometry import box
        self.arcs = {"A": _line((0, 0), (0, 40)), "B1": _line((0, 40), (0, 60)), "B2": _curve((0, 40), (0, 50), (10, 50)),
                     "D": _line((0, 60), (0, 100)), "E": _line((10, 50), (50, 50)), "F": _line((50, 50), (60, 50)), "H": _line((-10, 60), (0, 60))}
        self.lane = [{"token": t, "left_lane_divider_segments": [], "right_lane_divider_segments": []} for t in ("A", "D", "E", "F", "H")]
        self.lane_connector = [{"token": t} for t in ("B1", "B2")]
        out = {"A": ["B1", "B2"], "B1": ["D"], "B2": ["E"], "E": ["F"], "H": ["D"], "D": [], "F": []}
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
    assert sorted(P) == ["A", "B1", "B2", "D", "H"], sorted(P)                               # B2 + E + F joined, nothing else
    assert P["B2"]["tokens"] == ["B2", "E", "F"] and P["B2"]["kind"] == "mixed" and P["A"]["kind"] == "lane" and P["B1"]["kind"] == "connector"
    assert [s[0] for s in P["B2"]["spans"]] == ["B2", "E", "F"] and P["B2"]["spans"][0][1] == 0
    ev = collections.Counter(e["type"] for e in mg["events"])
    assert ev == {"split": 1, "merge": 1, "start": 2, "end": 2, "stop": 1}, ev
    split = next(e for e in mg["events"] if e["type"] == "split")
    assert split["in"] == ["A"] and split["out"] == ["B1", "B2"] and np.allclose(split["pos"], (0, 40), atol=0.3) and split["dirs_out"].shape == (2, 2)
    merge = next(e for e in mg["events"] if e["type"] == "merge")
    assert merge["in"] == ["B1", "H"] and merge["out"] == ["D"] and np.allclose(merge["pos"], (0, 60), atol=0.3)
    stop = next(e for e in mg["events"] if e["type"] == "stop")
    assert stop["on"] == "A" and stop["out"] == ["B1", "B2"] and stop["n_succ"] == 2 and np.allclose(stop["pos"], (0, 40), atol=0.1)
    # the rules for a lane with several successors
    one = LG.build_map_graph(_Map(), stop_rule="none")
    assert not [e for e in one["events"] if e["type"] == "stop"]
    try:
        LG.build_map_graph(_Map(), stop_rule="nope")
        raise AssertionError("an unknown stop rule must be refused")
    except ValueError:
        pass
    # ego at (0, 20) looking north: the lane frame is x right (east), y forward (north)
    g = LG.crop(mg, _pose(0, 20, math.pi / 2), half=50.0)
    by = {q["id"]: q for q in g["pieces"]}
    assert set(by) == {"A", "B1", "B2", "D", "H"} and g["frame"] == "lane"
    assert np.allclose(by["A"]["pts"][0], (0, -20), atol=1e-4) and np.allclose(by["A"]["pts"][-1], (0, 20), atol=1e-4)
    assert by["A"]["succ"] == ["B1", "B2"] and by["B1"]["pred"] == ["A"] and by["D"]["pred"] == ["B1", "H"] and by["D"]["succ"] == []
    assert by["D"]["end_open"] and not by["D"]["start_open"] and np.allclose(by["D"]["pts"][-1], (0, 50), atol=1e-4)        # cut at y = 50
    assert by["B2"]["pts"][-1][0] > 0 and by["H"]["pts"][0][0] < 0                                                        # east is positive x
    w = [e for e in g["events"] if e["type"] == "window"]
    assert len(w) == 2 and {e["io"] for e in w} == {"out"} and all(abs(e["pos"][1] - 50) < 1e-4 or abs(e["pos"][0]) > 49.9 for e in w), w
    ego = LG.crop(mg, _pose(0, 20, math.pi / 2), half=50.0, frame="ego")
    assert np.allclose(LG.in_frame(g, "ego")["pieces"][0]["pts"], ego["pieces"][0]["pts"], atol=1e-5), "lane -> ego by rotation"
    back = LG.in_frame(LG.in_frame(g, "ego"), "lane")
    assert np.allclose(back["pieces"][1]["pts"], g["pieces"][1]["pts"], atol=1e-5) and np.allclose(back["events"][0]["pos"], g["events"][0]["pos"], atol=1e-5)
    stop_c = next(e for e in g["events"] if e["type"] == "stop")
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
    finally:
        LL.CHUNK = old


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
    # neighbours: three lanes side by side, going the same way, and an opposite lane
    polys = {"a": _line((0, 0), (0, 30)), "b": _line((3.5, 0), (3.5, 30)), "c": _line((7, 0), (7, 30)), "o": _line((-3.5, 30), (-3.5, 0)),
             "far": _line((30, 0), (30, 30))}
    nb = RG.neighbours(polys, {t: "lane" for t in polys})
    assert nb["right"]["a"] == ["b"] and nb["left"]["b"] == ["a"] and nb["right"]["b"] == ["c"] and nb["left"]["a"] == [] and nb["opposite"]["a"] == ["o"], nb
    assert nb["lane_count"]["a"] == nb["lane_count"]["b"] == nb["lane_count"]["c"] == 3 and nb["lane_index"] == {"a": 0, "b": 1, "c": 2, "o": 0, "far": 0} \
        and nb["lane_count"]["far"] == 1, nb
    # the lane GT has its own version: nothing of it is in the signature of the front GT, a parameter makes another signature, an unknown one is refused
    assert not set(lane_gt.SOURCE_FILES) & set(SIGNATURE_FILES), "the lane GT must not change the front GT's signature"
    assert lane_gt.signature() == lane_gt.signature(boundary_layers=["drivable_area"]) != lane_gt.signature(stop_rule="none") != lane_gt.signature(corner_deg=45.0)
    try:
        lane_gt.signature(corners=1)
        raise AssertionError("an unknown parameter must be refused")
    except TypeError:
        pass
    assert lane_gt.version_of(lane_gt.signature()) is not None, (f"the lane GT source has the signature {lane_gt.signature()} but no row in autogt/lane_versions.csv: "
                                                                 "add one, commit, tag autogt/lane-l<N>")
    gt = AutoGT(root, "v1.0-mini", light=True, scenes=["scene-0061", "scene-0103", "scene-1077"])
    lgt = lane_gt.LaneGT(root, "v1.0-mini", nusc=gt.nusc)
    n_frames, d_graph, d_ref, f1s, divpairs, found = 0, [], [], [], 0, 0
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
            ids = {x["id"] for x in lg["pieces"]}
            for q in lg["pieces"]:                                    # every id of a link names a piece of the crop
                assert set(q["succ"]) <= ids and set(q["pred"]) <= ids and set(q["left"]) <= ids and set(q["right"]) <= ids, q["id"]
            n_frames += 1
    d_ref, d_graph = np.array(d_ref), np.array(d_graph)
    assert np.median(d_graph) < 0.05 and np.percentile(d_graph, 99) < 0.25, (np.median(d_graph), np.percentile(d_graph, 99))      # on the lane graph
    assert np.percentile(d_ref, 99) < 0.25, np.percentile(d_ref, 99)                                                                # nothing of it missing
    assert np.mean(f1s) > 0.95 and min(f1s) > 0.8, (np.mean(f1s), min(f1s))                                                         # events + links
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
        assert lane_gt.generate(root, out=d, scenes=["scene-0061"], verbose=False)["written"] == 0, "a second run has nothing to do"
        assert lane_gt.missing(toks, data=d) == [] and lane_gt.missing(toks + ["0" * 32], data=d) == ["0" * 32]
        assert digest(lane_gt.load(toks[5], data=d)) == digest(lgt.graph(toks[5])), "the stored graph is the graph made now"
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
    msg = (f"lane graph selftest ok: {n_frames} mini frames, graph points {np.median(d_graph):.3f} m (99 % {np.percentile(d_graph, 99):.3f}) from the map's "
           f"centre-lines and {np.percentile(d_ref, 99):.3f} m (99 %) the other way, junction F1 {np.mean(f1s):.3f}, "
           f"neighbours: {found} of {divpairs} divider-token pairs found; store, load, resume, parameters ok (lane signature {lane_gt.signature()})")
    if verbose:
        print(msg)
    return msg


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
