import sys
sys.path.insert(0, ".")
import numpy as np
import evaluation.bev_gtcheck as G
from evalsuite import Prediction, Sample
from evalsuite.contracts import FrameRef
task = G.build_task()
samples = G.pick(3, "15efbde2d1", 0)
name, s = samples[0]
out = {dt: G.oracle_output(g, task.spec, task.groups) for dt, g in s.items()}
sample = Sample(ref=FrameRef("gt", 0, name), inputs=None, reference=s)
pred = Prediction(outputs={"dt": out})
for m in task.metrics():
    if m.name == "loss":
        continue
    c = m.frame(sample, pred)
    print(m.name, "frame keys:", list(c)[:6], "| n dets", len(out[0.0]["dets"]), "| groups", task.groups)
    r = m.reduce([c])
    print("   reduce n keys", len(r), list(r)[:4])
    if m.name == "curve":
        print("   velocity sums", c.get("dt+0.00|velocity"))
