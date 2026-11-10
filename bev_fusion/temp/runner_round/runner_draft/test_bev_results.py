"""runner/bev_results.py. Part 1 (seconds, no data): the frame grammar, the plan, the memory warnings. Part 2 (a REAL run, CPU, minutes, about 5 GB: it loads nuScenes; not in run_gates.sh): the
first 3 frames of val scene-0003 through the old runner (runner/bev_runner.py `full`) and through the new Runner; the new result set must hold the same maps as the old maps.npz, a
check()-clean set with all dts and the taps asked for, and the memory must be empty at the scene's first frame and grow. Run (from the bev_fusion folder):
    ~/miniconda3/envs/gcar/bin/python tests/test_bev_results.py [runs/<name>] [--unit]"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "runner"))
os.chdir(ROOT)
import bev_results as B  # noqa: E402

SCENE, N = "scene-0003", 3


def raises(fn, text):
    try:
        fn()
    except B.RunnerError as e:
        assert text in str(e), (text, str(e))
        return
    raise AssertionError(f"no error ({text})")


def unit():
    assert B.parse_frames("scene-0003") == [("scene", "scene-0003", None)]
    assert B.parse_frames("scene-0003:3, all") == [("scene", "scene-0003", 3), ("all",)]
    assert B.parse_frames(["scene-0003", "random:5@1"]) == [("scene", "scene-0003", None), ("random", 5, 1)]
    for bad, text in (("random:5", "needs its seed"), ("val", "frames are"), ("scene-3:x", "frames are"), (" , ", "no frames")):
        raises(lambda: B.parse_frames(bad), text)
    scenes = {f"scene-{i:04d}": list(range(10 * i, 10 * i + 4)) for i in range(1, 8)}
    plan = B.plan_frames(B.parse_frames("scene-0003:2,scene-0001"), scenes)
    assert [(r["scene"], r["index"], r["di"]) for r in plan] == [("scene-0003", 0, 30), ("scene-0003", 1, 31)] + [("scene-0001", k, 10 + k) for k in range(4)]
    assert len(B.plan_frames(B.parse_frames("all"), scenes)) == 28 and len(B.plan_frames(B.parse_frames("scene-0002,scene-0002:1"), scenes)) == 4, "a scene named twice runs once"
    a, b = (B.plan_frames(B.parse_frames("random:3@7"), scenes) for _ in range(2))
    c = B.plan_frames(B.parse_frames("random:3@8"), scenes)
    assert a == b and len({r["scene"] for r in a}) == 3 and a != c, "a random draw is reproducible with its seed and another seed draws other scenes"
    raises(lambda: B.plan_frames(B.parse_frames("scene-9999"), scenes), "not a scene")
    assert B.memory_warnings({"s1": [0.0, 0.3, 0.6], "s2": [0.0]}) == []
    w = B.memory_warnings({"s1": [0.4, 0.5], "s2": [0.0, 0.0, 0.0]})
    assert len(w) == 2 and "not empty" in w[0] and "does not grow" in w[1], w


def real(run):
    from bev_runner import main as old
    out = tempfile.mkdtemp(prefix="bev_results_test_", dir=os.path.join(ROOT, "temp") if os.path.isdir(os.path.join(ROOT, "temp")) else None)
    code = [] if os.path.exists(os.path.join(run, "code", "precompute.py")) else ["--code", "training"]        # a snapshot from before the five-layer build runs under the live code
    dts = (-0.5, 0.0, 0.5)
    old(["full", run, "--scene", SCENE, "--first", str(N), "--out", os.path.join(out, "old"), "--device", "cpu", "--dts", *map(str, dts), *code])
    r = B.Runner()
    info = r.run(run, f"{SCENE}:{N}", dts=dts, taps=["fused"], code=code[1] if code else "", device="cpu")
    rs = r.open(info["folder"])
    assert not info["existed"] and info["warnings"] == [], info
    assert rs.check() == [] and rs.keys() == [f"{SCENE}/{k:03d}" for k in range(N)] and rs.dts == list(dts) and rs.taps == ["fused"], rs.summary()
    assert rs.identity["memory"] == "carried" and rs.identity["weights"] == "EMA" and rs.identity["network"] == os.path.basename(run), rs.identity

    # the maps of dt 0 are the old runner's (float16 there, float32 here), the distance layers within float16 resolution
    z = np.load(os.path.join(out, "old", "maps.npz"))
    names, kinds = list(z["names"]), list(z["kinds"])
    compared = 0
    for k, key in enumerate(rs.keys()):
        o = rs.load(key)["dt"][0.0]
        for n, kind in zip(names, kinds):
            if kind == "dist":
                a, b = o["dist"][n[:-len("_dist")]], z["sem"][k, names.index(n)].astype(np.float32)
                assert np.abs(a - b).max() < 0.1, (key, n, float(np.abs(a - b).max()))
            elif n in o["prob"]:
                a, b = o["prob"][n], z["sem"][k, names.index(n)].astype(np.float32)
                assert np.abs(a - b).max() < 2e-3, (key, n, float(np.abs(a - b).max()))
            else:
                continue
            compared += 1
    assert compared >= N * 5, f"only {compared} maps compared"
    assert any((rs.load(key)["dt"][0.0]["prob"][n] > 0.1).any() for key in rs.keys() for n in names if n in rs.load(key)["dt"][0.0]["prob"]), "every map is empty"

    # the rest of what the evaluation needs is there: every dt, detections with scores, the velocity field, the taps; the memory starts empty and grows
    frame = rs.load(rs.keys()[1])
    assert sorted(frame["dt"]) == sorted(dts) and frame["taps"]["fused"].ndim == 3 and np.isfinite(frame["taps"]["fused"]).all()
    dets = [d for key in rs.keys() for dt in dts for d in rs.load(key)["dt"][dt]["dets"]]
    assert dets and all({"group", "score", "cx", "cy"} <= set(d) for d in dets), "no detections or their fields are missing"
    assert any(rs.load(key)["dt"][dt][v] is not None for key in rs.keys() for dt in dts for v in ("vhat", "objvel")), "no velocity field"
    conf = [f.get("info", {}).get("memory_conf") for f in json.load(open(os.path.join(rs.folder, "result.json")))["frames"]]
    if conf[0] is not None:
        assert conf[0] <= B.MEMORY_EMPTY and conf[-1] > conf[0], conf
    # the same content is never made twice; list / get find it
    assert B.Runner().get(rs.id, verify=True)["problems"] == [] and any(s["id"] == rs.id for s in r.list())
    print(f"test_bev_results passed: {N} frames of {SCENE} through {os.path.basename(run)}: {compared} maps equal the old runner's, result set {rs.id} "
          f"check-clean with dts {list(dts)}, {len(dets)} detections, fused tap {frame['taps']['fused'].shape}, memory confidence {conf}; outputs in {out}")


if __name__ == "__main__":
    unit()
    print("test_bev_results part 1 passed: frame grammar, plan, random draw, memory warnings")
    if "--unit" not in sys.argv:
        real(os.path.abspath(next((a for a in sys.argv[1:] if not a.startswith("-")), "runs/t20_base_s0")))
