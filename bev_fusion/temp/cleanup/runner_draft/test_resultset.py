"""runner/resultset.py on synthetic data (numpy only, seconds, in run_gates.sh): what is added is what is loaded, the id comes from the content, a finished set is never overwritten, a crashed
partial is moved aside, a subset is a view, and check() finds a missing file, a NaN, a frame out of order and other dts. Run (from the bev_fusion folder): python3 tests/test_resultset.py"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "runner"))
import resultset as R  # noqa: E402

DTS = [0.0, 0.5]
FRAMES = [{"scene": "scene-0003", "index": i, "token": f"tok{i}"} for i in range(3)] + [{"scene": "scene-0061", "index": 0, "token": "tokB"}]


def outputs(seed):
    g = np.random.default_rng(seed)
    one = lambda: {"prob": {"drivable_area": g.random((6, 5)), "lane_divider": g.random((6, 5))}, "dist": {"road_dist": g.random((6, 5)) * 40},     # noqa: E731
                   "dir": {"centreline": g.random((2, 6, 5))}, "mass": g.random((6, 5)), "vhat": g.random((2, 6, 5)), "objvel": None,
                   "dets": [{"group": "vehicle", "score": np.float32(0.9), "cx": 1.5, "cy": -2.0}], "detr": None}
    return {"dt": {dt: one() for dt in DTS}}


def write(root, taps=("fused",)):
    w = R.ResultWriter(root, {"network": "toy", "step": 7, "memory": "carried"}, FRAMES, DTS, taps)
    ref = {}
    for i, f in enumerate(FRAMES):
        key = R.frame_key(f["scene"], f["index"])
        ref[key] = (outputs(i), {"fused": np.full((3, 4), i, np.float32)} if taps else {})
        w.add(key, *ref[key], info={"memory_conf": 0.1 * i})
    return w, ref


def raises(fn, text):
    try:
        fn()
    except R.ResultSetError as e:
        assert text in str(e), (text, str(e))
        return
    raise AssertionError(f"no error ({text})")


def main():
    root = tempfile.mkdtemp(prefix="resultset_test_")
    w, ref = write(root)
    assert w.folder.endswith(".partial") and os.path.isdir(w.folder), "frames are written into a .partial folder"
    rs = w.finish()
    assert rs.id == w.id and os.path.basename(rs.folder) == rs.id and not os.path.exists(w.folder), "finish renames the folder to the id"
    assert rs.keys() == ["scene-0003/000", "scene-0003/001", "scene-0003/002", "scene-0061/000"] and rs.scenes() == ["scene-0003", "scene-0061"]
    assert rs.dts == DTS and rs.taps == ["fused"] and rs.identity["n_frames"] == 4 and rs.identity["memory"] == "carried"
    assert rs.check() == [], rs.check()
    assert R.make_id({"network": "toy", "step": 7, "memory": "carried"}, [f["token"] for f in FRAMES], DTS, ["fused"])[0] == rs.id, "make_id = the writer's id"
    assert json.load(open(os.path.join(rs.folder, "result.json")))["frames"][2]["info"] == {"memory_conf": 0.2}, "info is kept in result.json"

    # what is added is what is loaded (float32, the detections as numbers, None stays None, a missing group stays empty)
    for key, (out, taps) in ref.items():
        got = rs.load(key)
        assert sorted(got["dt"]) == DTS and list(got["taps"]) == ["fused"] and np.array_equal(got["taps"]["fused"], taps["fused"])
        for dt in DTS:
            a, b = out["dt"][dt], got["dt"][dt]
            for g in R.GROUPS:
                assert sorted(a[g]) == sorted(b[g])
                assert all(np.array_equal(a[g][n].astype(np.float32), b[g][n]) and b[g][n].dtype == np.float32 for n in a[g]), (key, dt, g)
            assert np.array_equal(a["mass"].astype(np.float32), b["mass"]) and b["objvel"] is None and b["detr"] is None
            assert b["dets"] == [{"group": "vehicle", "score": 0.8999999761581421, "cx": 1.5, "cy": -2.0}], b["dets"]

    # the same content has the same id and is never overwritten; other content (dts, taps, memory) has another id
    raises(lambda: write(root), "never overwritten")
    other = tempfile.mkdtemp(prefix="resultset_test_")
    assert write(other)[0].id == rs.id, "the id comes from the content only"
    assert write(other, taps=())[0].id != rs.id and R.ResultWriter(other, {"network": "toy", "step": 7, "memory": "cold"}, FRAMES, DTS, ["fused"]).id != rs.id

    # a subset is a view with an id of its own; the files are the same; the order stays the set's
    sub = rs.subset(["scene-0061/000", "scene-0003/001"])
    assert sub.keys() == ["scene-0003/001", "scene-0061/000"] and sub.id != rs.id and sub.view_of == rs.id and sub.scenes() == ["scene-0003", "scene-0061"]
    assert np.array_equal(sub.load("scene-0003/001")["taps"]["fused"], ref["scene-0003/001"][1]["fused"])
    raises(lambda: sub.load("scene-0003/000"), "not in this result set")
    raises(lambda: rs.subset(["scene-9999/000"]), "not in this result set")

    # a writer refuses wrong input; a set with a missing frame cannot be finished; a crashed partial is moved aside, not deleted
    root2 = tempfile.mkdtemp(prefix="resultset_test_")
    w2 = R.ResultWriter(root2, {"network": "toy"}, FRAMES, DTS, ["fused"])
    raises(lambda: w2.add("scene-1/000", *ref["scene-0003/000"]), "not one of the announced")
    w2.add("scene-0003/000", *ref["scene-0003/000"])
    raises(lambda: w2.add("scene-0003/000", *ref["scene-0003/000"]), "twice")
    raises(lambda: w2.add("scene-0003/001", ref["scene-0003/001"][0], {}), "differ from the announced")
    raises(lambda: w2.finish(), "were not added")
    R.ResultWriter(root2, {"network": "toy"}, FRAMES, DTS, ["fused"])
    assert os.listdir(os.path.join(root2, "_to_delete")) and os.path.isdir(w2.folder), "the crashed partial was moved aside and a new one started"
    raises(lambda: R.ResultWriter(root2, {}, FRAMES + FRAMES[:1], DTS), "same scene and index")
    raises(lambda: R.ResultSet.open(w2.folder), "no result.json")

    # check() finds what is wrong
    root3 = tempfile.mkdtemp(prefix="resultset_test_")
    rs3 = write(root3)[0].finish()
    os.remove(os.path.join(rs3.folder, "frames", "scene-0003_001.npz"))
    bad = dict(ref["scene-0003/002"][0])
    bad = {"dt": {dt: dict(o, prob=dict(o["prob"], lane_divider=np.full((6, 5), np.nan))) for dt, o in bad["dt"].items()}}
    np.savez_compressed(os.path.join(rs3.folder, "frames", "scene-0003_002.npz"), **R.pack(bad, ref["scene-0003/002"][1]))
    np.savez_compressed(os.path.join(rs3.folder, "frames", "scene-0061_000.npz"), **R.pack({"dt": {0.0: ref["scene-0061/000"][0]["dt"][0.0]}}, ref["scene-0061/000"][1]))
    text = " | ".join(rs3.check())
    assert "scene-0003/001: cannot be read" in text and "not finite" in text and "scene-0061/000: dts" in text, text
    swapped = json.load(open(os.path.join(rs3.folder, "result.json")))
    swapped["frames"][1], swapped["frames"][0] = swapped["frames"][0], swapped["frames"][1]
    json.dump(swapped, open(os.path.join(rs3.folder, "result.json"), "w"))
    assert "not in time order" in " | ".join(R.ResultSet.open(rs3.folder).check())
    print(f"test_resultset passed: {len(FRAMES)} frames x {len(DTS)} dts round trip exactly, id from content, never overwritten, partial moved aside, subset is a view, check finds 4 kinds of damage")


if __name__ == "__main__":
    main()
