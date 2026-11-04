import sys, time, numpy as np
sys.path[:0] = ["."]
import evaluation.bev_gtcheck as G
from evalsuite import Prediction, Sample
from evalsuite.contracts import FrameRef
t = time.time(); samples = G.pick(24, G.resolve_sig(None), 0); tl = time.time() - t
task = G.build_task()
outs = G.variants(samples, task.spec, task.groups, 0)["noise 0.2"]
print(f"load {len(samples)} cached GT frames (3 dts each): {tl/len(samples)*1000:.0f} ms/frame (includes the completeness scan of pick)")
ms = [m for m in task.metrics() if m.name != "loss"]
for m in ms:
    t = time.time(); c = []
    for i, ((name, s), out) in enumerate(zip(samples, outs)):
        c.append(m.frame(Sample(ref=FrameRef("gt", i, name), inputs=None, reference=s), Prediction(outputs={"dt": out})))
    tf = (time.time() - t) / len(samples) * 1000
    t = time.time(); m.reduce(c); tr = (time.time() - t) * 1000
    print(f"{m.name:6s} frame {tf:7.1f} ms/frame   reduce({len(c)} frames) {tr:7.1f} ms")
