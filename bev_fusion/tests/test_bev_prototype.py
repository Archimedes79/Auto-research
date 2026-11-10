"""Gate of the prototype command (evaluation/bev_prototype.py) on a real run, CPU, 2 frames of the first val scene (minutes: it loads
nuScenes trainval and the run's checkpoint).

1  objects.json holds exactly the subject's boxes of those frames (same order, every field = the decoded box's value), at dt 0 and +0.5 s
2  maps_<scene>.npz holds exactly the subject's probability maps at dt 0 (every layer, bit-equal) and the run's grid
3  summary.json's headline numbers equal the stored eval/card.json values (same keys, same values; nothing recomputed)
The reference is a second pass of the same subject with the memory carried from the scene start (deterministic on the CPU).

Usage (from the bev_fusion folder): OMP_NUM_THREADS=2 python tests/test_bev_prototype.py [RUN]   (default runs/t10_dual_s0)
"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "..", "autoresearch"))
sys.path.insert(0, ROOT)


def check_box(rec, d):
    assert rec["class"] == d["group"] and rec["score"] == d["score"], (rec, d)
    assert rec["centre"] == [d["cx"], d["cy"]] and rec["velocity"] == [d["vx"], d["vy"]], (rec, d)
    if "radius" in d:
        assert rec["radius"] == d["radius"] and rec["size"] is None and rec["yaw"] is None, (rec, d)
    else:
        assert rec["size"] == [d["length"], d["width"]] and rec["yaw"] == d["yaw"] and rec["radius"] is None, (rec, d)


def main(run="runs/t10_dual_s0"):
    os.chdir(ROOT)
    from Evaluation import EvalSuite
    from Evaluation.runner import Runner
    from evaluation import bev_prototype as P
    from evaluation.bev_adapter import build
    source, subject, task = build(run, device="cpu", workers=0)
    with tempfile.TemporaryDirectory() as out:
        suite = EvalSuite(source, subject, task, out=out, state="carry")
        scene = P.pick_scenes(source, ["0"])[0]
        s = P.export(suite, run, [scene], 2, out)
        objects = json.load(open(os.path.join(out, "objects.json")))["scenes"][scene]
        maps = np.load(os.path.join(out, f"maps_{scene}.npz"))
        n_boxes = 0
        for f, (sample, pred) in enumerate(Runner(source, subject, "carry").iter_scene(scene, source.frames(scene)[:2])):
            rec = objects[f]
            assert rec["index"] == sample.ref.index and rec["token"] == sample.ref.key
            for key, dt in (("objects", 0.0), ("forecast_0.5s", 0.5)):
                o = P.at_dt(pred.outputs, dt)
                boxes = o["detr"] if o.get("detr") is not None else o["dets"]
                assert len(rec[key]) == len(boxes), (key, len(rec[key]), len(boxes))
                for r, d in zip(rec[key], boxes):
                    check_box(r, d)
                n_boxes += len(boxes)
            prob = P.at_dt(pred.outputs, 0.0)["prob"]
            assert sorted(prob) == list(maps["layers"]), (sorted(prob), maps["layers"])
            for name, p in prob.items():
                assert np.array_equal(maps[name][f], p), name
        grid = json.loads(str(maps["grid"]))
        assert maps[str(maps["layers"][0])].shape[1:] == (grid["nx"], grid["ny"]) and grid["res"] == subject.model.grid.res
        card = json.load(open(os.path.join(run, "eval", "card.json")))
        nums = s["headline"]["numbers"]
        assert nums and all(r["value"] == card["metrics"].get(r["key"]) for r in nums), nums
        assert s["headline"]["frames"] == card["frames"] and json.load(open(os.path.join(out, "summary.json")))["headline"] == s["headline"]
        suite.close()
    print(f"prototype gate ok: {run} {scene} 2 frames, {n_boxes} boxes and {len(prob)} maps equal to the subject's, "
          f"{len(nums)} headline numbers equal to the stored card")


if __name__ == "__main__":
    main(*sys.argv[1:])
