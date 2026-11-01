"""Selftest of lane_prototype.py (CPU, cached network outputs, 2 threads, a few minutes):  wsl bash tools/env/_py14.sh test_lane_prototype.py

1. coordinate rule (known answer): in the planner frame (x forward, y left) the ground-truth lanes near the ego that travel backwards lie on the left in Boston
   (right-hand traffic) and on the right in Singapore (left-hand traffic)
2. two VAL scenes (23 Boston, 7 Singapore One-North): the graph written to JSON equals the graph that evaluate.graphs_of scores at the registry settings
   (points within 1e-5 m, edges, vertex ids), and summary.json's numbers equal card_any on evaluate's graphs of those frames and on the graphs read back from the JSON
3. the whole VAL split: summary.json reproduces the stored card runs/eval/<name>/val.json (layer 1)
"""
import lane_prototype as LP  # first: it sets the thread limits before numpy loads
import glob  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402

import evaluate as EV  # noqa: E402
import make_results_video as RV  # noqa: E402
import sparse_metrics as SM  # noqa: E402

NAME, SCENES = "recall_dp", ["23", "7"]


def backward_left_share(gts, ix):
    """share of the backward-travelling ground-truth lanes near the ego (|x| < 20 m, |y| < 12 m) that lie left of the ego (y > 0), planner frame"""
    side = []
    for i in ix:
        for q in gts[i]["pts"]:
            q = LP.to_planner(q)
            d = q[-1] - q[0]
            m = q.mean(0)
            if len(q) > 1 and np.linalg.norm(d) > 3 and abs(m[0]) < 20 and abs(m[1]) < 12 and d[0] / np.linalg.norm(d) < -0.8:
                side.append(m[1] > 0)
    return float(np.mean(side)), len(side)


def main():
    reg = EV.registry()
    ref, gts, _ = EV.graphs_of(NAME, reg[NAME], "val", require_cache=True)
    ft = RV.frame_table("val")
    for loc, want in ((0, True), (1, False)):
        share, n = backward_left_share(gts, np.nonzero(ft["loc"] == loc)[0])
        assert n > 50 and (share > 0.9 if want else share < 0.1), f"coordinate rule: location {loc}: {share:.3f} of {n} backward lanes on the left"
        print(f"ok  coordinate rule: {RV.LOCN[loc]}: {share:.3f} of {n} backward lanes on the left")
    tmp = tempfile.mkdtemp(prefix="lane_prototype_")
    try:
        s = LP.run(NAME, "val", SCENES, 0, f"{tmp}/two")
        ks = LP.select(ft, SCENES)
        fr = {f["k"]: f for p in glob.glob(f"{tmp}/two/scenes/*.json") for f in json.load(open(p))["frames"]}
        assert sorted(fr) == sorted(ks) and s["frames"] == len(ks), "written frames differ from the selection"
        for k in ks:
            a, b = LP.graph_of_frame(fr[k]), ref[k]
            assert a["edges"] == [tuple(map(int, x)) for x in b["edges"]] and a["nodes"] == [list(map(int, n)) for n in b["nodes"]], f"frame {k}: edges / vertex ids differ"
            assert len(a["pts"]) == len(b["pts"]) and all(p.shape == np.shape(q) and np.abs(p - q).max() <= 1e-5 for p, q in zip(a["pts"], b["pts"])), f"frame {k}: points differ"
        m_ref = SM.card_any([ref[k] for k in ks], [gts[k] for k in ks], extra=True)
        m_json = SM.card_any([LP.graph_of_frame(fr[k]) for k in ks], [gts[k] for k in ks], extra=True)
        for key in LP.HEAD:
            assert abs(s["metrics"][key] - m_ref[key]) <= EV.TOL and abs(m_json[key] - m_ref[key]) <= EV.TOL, f"{key}: summary {s['metrics'][key]}, evaluate {m_ref[key]}, json {m_json[key]}"
        print(f"ok  {len(ks)} frames of scenes {SCENES}: JSON graph == evaluate's graph, summary numbers == evaluate's: " + " ".join(f"{k} {m_ref[k]:.4f}" for k in LP.HEAD))
        s = LP.run(NAME, "val", None, 0, f"{tmp}/full")
        assert s["stored_card"]["reproduced"], f"stored card not reproduced: {s['stored_card']}"
        print(f"ok  whole VAL ({s['frames']} frames) reproduces {s['stored_card']['card']}: " + " ".join(f"{k} {s['metrics'][k]:.4f}" for k in LP.HEAD))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("lane_prototype selftest passed")


if __name__ == "__main__":
    main()
