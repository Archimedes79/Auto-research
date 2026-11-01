"""throw-away: probe tags of a constant empty LaneGraph over 12 frames of 4 scenes; argv[1] = folder holding lane_task/"""
import sys

sys.path.insert(0, sys.argv[1])
from lane_task import LaneGraph, synth as LS  # noqa: E402
from evalsuite.contracts import Prediction  # noqa: E402

src, _, task = LS.build(n_frames=3)
g = LaneGraph([])
own, got = [], []
for sc in src.scenes()[:4]:
    for r in src.frames(sc):
        s = src.load(r)
        got.append(task.probe(s, Prediction({"lanes": g}))["tag.gt_len"])
        own.append(LaneGraph.of(s.reference["lanes"]).length())
print("distinct gt_len reported:", len(set(round(x, 6) for x in got)), "of", len(set(round(x, 6) for x in own)), "true values;",
      "max error", round(max(abs(a - b) for a, b in zip(got, own)), 3), "m")
