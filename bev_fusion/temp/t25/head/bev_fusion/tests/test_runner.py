"""The runner (runner/bev_runner.py) on a REAL run, CPU, the first 3 frames of val scene-0003 (default run runs/t10_dual_s0, argument: another
run whose prototype lies in RUN/prototype); minutes and about 5 GB (it loads nuScenes trainval), not in run_gates.sh.
Checks: (1) `full` under the live code (`--code training`: the checkpoint's keys do not change with the five-layer build) gives the boxes
at dt 0 and +0.5 s and the probability maps that the evaluation's prototype (evaluation/bev_prototype.py, GT-based ego steps) stored for
the same frames - boxes of score > 0.35 pair up within 0.2 m and 0.03 in score (at least 10 compared), maps within 0.02 mean where either says something; (2) `precompute` then `network`
(two processes' worth of work, the frames through disk) equals `full` exactly, and `--cold` (memory restarted at every frame) differs on the later frames; (3) a snapshot from before the build is refused with a
hint. Run (from the bev_fusion folder): wsl ~/miniconda3/envs/gcar/bin/python tests/test_runner.py [runs/<name>]"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "runner"))
os.chdir(ROOT)
import bev_runner as R  # noqa: E402

RUN = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "runs/t10_dual_s0")
SCENE, N = "scene-0003", 3


def pair(stored, got, label):
    """every box of score > 0.35 on one side has a box of its class within 0.2 m and 0.03 in score on the other side (any score there: a box
    at 0.349 / 0.351 across the line is no failure); -> the number of boxes compared"""
    def find(rec, others):
        return any(o["class"] == rec["class"] and np.hypot(o["centre"][0] - rec["centre"][0], o["centre"][1] - rec["centre"][1]) < 0.2
                   and abs(o["score"] - rec["score"]) < 0.03 for o in others)
    strong = [r for r in stored if r["score"] > 0.35]
    miss = [r for r in strong if not find(r, got)] + [r for r in got if r["score"] > 0.35 and not find(r, stored)]
    assert not miss, (label, len(miss), miss[:2])
    return len(strong)


def main():
    out = tempfile.mkdtemp(prefix="runner_test_", dir=os.path.join(ROOT, "temp")) if os.path.isdir(os.path.join(ROOT, "temp")) else tempfile.mkdtemp()
    full, frames, net = (os.path.join(out, d) for d in ("full", "frames", "network"))
    common = ["--code", "training", "--device", "cpu", "--scene", SCENE, "--first", str(N)]
    R.main(["full", RUN, *common, "--out", full])
    stored = json.load(open(os.path.join(RUN, "prototype", "objects.json")))["scenes"][SCENE]
    got = json.load(open(os.path.join(full, "objects.json")))["frames"]
    assert [g["token"] for g in got] == [s["token"] for s in stored[:N]], "the frames are the prototype's, in time order"
    n_boxes = 0
    for g, s in zip(got, stored):
        n_boxes += pair(s["objects"], g["objects"]["+0.0"], f"dt 0, frame {g['index']}")
        n_boxes += pair(s["forecast_0.5s"], g["objects"]["+0.5"], f"dt +0.5, frame {g['index']}")
    assert n_boxes >= 10, f"only {n_boxes} boxes were compared: the test would pass on empty lists"
    z, ref = np.load(os.path.join(full, "maps.npz")), np.load(os.path.join(RUN, "prototype", "maps_scene-0003.npz"))
    names = list(z["names"])
    n_maps = 0
    for name in ref["layers"]:
        a, b = z["sem"][:, names.index(name)].astype(np.float32), ref[name][:N]
        d = np.abs(a - b)[(a > 0.1) | (b > 0.1)]                  # the cells where either map says something (a sparse layer's mean over all cells is ~0)
        assert d.size == 0 or d.mean() < 0.02, (name, float(d.mean()), d.size)
        n_maps += 1
    assert any(((z["sem"][:, names.index(n)] > 0.1).any()) for n in ref["layers"]), "every map is empty"
    R.main(["precompute", os.path.join(RUN, "config.json"), "--scene", SCENE, "--first", str(N), "--out", frames, "--code", "training"])
    R.main(["network", RUN, frames, *common[:4], "--out", net])
    a, b = json.load(open(os.path.join(full, "objects.json"))), json.load(open(os.path.join(net, "objects.json")))
    assert a["frames"] == b["frames"], "precompute + network (the frames through disk) = full"
    assert np.array_equal(np.load(os.path.join(full, "maps.npz"))["sem"], np.load(os.path.join(net, "maps.npz"))["sem"])
    R.main(["network", RUN, frames, *common[:4], "--cold", "--out", os.path.join(out, "cold")])     # the memory is carried: restarting it changes later frames
    cold = np.load(os.path.join(out, "cold", "maps.npz"))["sem"].astype(np.float32)
    carried = np.load(os.path.join(full, "maps.npz"))["sem"].astype(np.float32)
    assert np.array_equal(cold[0], carried[0]) and np.abs(cold[1:] - carried[1:]).max() > 1e-3, "a cold restart must differ from the carried memory on frames 2, 3"
    try:
        R.use_code(RUN, "")                                       # t10_dual_s0's own snapshot is from before the build
        refused = False
    except SystemExit as e:
        refused = "precompute.py" in str(e)
    assert refused == (not os.path.exists(os.path.join(RUN, "code", "precompute.py"))), "an old snapshot is refused with a hint"
    print(f"test_runner passed: {N} frames of {SCENE}: {n_boxes} boxes at dt 0 and +0.5 s equal the prototype's, {n_maps} maps within 0.02 where either "
          f"says something, precompute + network = full, a cold restart differs, old snapshot refused; outputs in {out}")


if __name__ == "__main__":
    main()
