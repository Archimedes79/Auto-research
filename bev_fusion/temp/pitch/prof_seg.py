import sys, cProfile, pstats, numpy as np
sys.path[:0] = ["."]
import evaluation.bev_gtcheck as G
from evalsuite import Prediction, Sample
from evalsuite.contracts import FrameRef
samples = G.pick(12, G.resolve_sig(None), 0)
task = G.build_task(); outs = G.variants(samples, task.spec, task.groups, 0)["noise 0.2"]
m = next(m for m in task.metrics() if m.name == "seg")
def run():
    for i, ((n, s), o) in enumerate(zip(samples, outs)):
        m.frame(Sample(ref=FrameRef("gt", i, n), inputs=None, reference=s), Prediction(outputs={"dt": o}))
pr = cProfile.Profile(); pr.enable(); run(); pr.disable()
pstats.Stats(pr).sort_stats("cumulative").print_stats(14)
