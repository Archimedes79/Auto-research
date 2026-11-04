"""one real picture: real cached GT frame + a degraded synthetic 1 m head output -> the debug view (StagesView) with the two centre-line panels -> temp/centreline_eval/debug_centreline.png"""
import math, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")]
os.environ["MPLBACKEND"] = "Agg"
from types import SimpleNamespace
from evaluation import bev_adapter as A, bev_gtcheck as GC
from evaluation.bev_task import BEVTask
from evalsuite import Prediction, Sample, ViewContext
from evalsuite.contracts import FrameRef
from evalsuite.render.figure import render_page
import losses as L
from lift_pd import Grid

SPEC, SPEC_1M = L.layer_spec(False, True), L.layer_spec_1m(True)
GROUPS = [g for g, _ in L.NUSC_GROUPS]
grid = Grid()
samples = GC.pick(8, "15efbde2d1", 0)
name, s = max(samples, key=lambda ns: float(np.asarray(ns[1][0.0]["centreline"]).sum()))
g = s[0.0]
print(name, "GT centre cells:", int(np.asarray(g["centreline"], np.float32).sum()))
T = L.build_layer_targets([g], L.LossConfig(centreline_1m=True))
rng = np.random.RandomState(0)
pres = T["centreline@1m"][0, 0].numpy()
X2, Y2 = pres.shape
xs = (np.arange(X2)[:, None] + 0.5) * 1.0                      # m, block centres (x forward)
ys = -26.0 + (np.arange(Y2)[None] + 0.5) * 1.0
logit = 20 * (2 * pres - 1) * 0.15 + rng.normal(0, 1.5, pres.shape)       # a head that is unsure: +-3 plus noise
logit[(xs > 28) & (xs < 45) & (ys > 0)] = -6.0                              # a missed stretch
logit[(np.abs(xs - 18) < 3) & (np.abs(ys + 12) < 3)] = 4.0                  # a false blob
u = T["centreline_dir@1m"][0].numpy().copy()
ang = np.arctan2(u[1], u[0]) + np.radians(12) + rng.normal(0, 0.15, pres.shape)
n = np.hypot(u[0], u[1])
d = np.stack([np.cos(ang) * np.maximum(n, 0.5), np.sin(ang) * np.maximum(n, 0.5)])
d[:, np.broadcast_to(ys < -8, pres.shape)] *= -1       # the cells right of the car (y < -8 m) reversed
dist = T["centreline_dist@1m"][0, 0].numpy()
sem_1m = torch.from_numpy(np.stack([logit, dist, d[0], d[1]]).astype(np.float32))[None]
p = {"sem": torch.zeros(1, len(SPEC), 120, 104), "sem_1m": sem_1m}
prob, dist0 = A.predictions(p, 0, SPEC)
p1, d1 = A.predictions_1m(p, 0, SPEC_1M)
prob.update(p1)
out = {"prob": prob, "dist": dist0, "mass": None, "dets": [], "detr": None, "vhat": None, "dir": d1}
task = BEVTask(grid, A.task_spec(SimpleNamespace(spec=SPEC, spec_1m=SPEC_1M)), GROUPS)
rad = {k: torch.tensor(v, dtype=torch.float32) for k, v in (("x", [5.0, 12.0, 30.0]), ("y", [0.0, 3.0, -4.0]), ("vr", [0.0, 5.0, -3.0]))}
sample = Sample(ref=FrameRef("gt", 0, name), inputs={"radar": rad}, reference=s, display={"camera": None})
pred = Prediction(outputs={"dt": {0.0: out}}, taps={}, info={})
page = task.views()["debug"].render(sample, pred, ViewContext())
print([p.title for p in page.panels])
render_page(page, os.path.join(HERE, "debug_centreline.png"), dpi=130)
# the numbers of the same frame
m = [m for m in task.metrics() if m.name == "seg"][0]
r = m.reduce([m.frame(sample, pred)])
for k in ("iou@0", "iou@0.5", "iou@1"):
    print(k, round(r["fov/dt+0.00/centreline/all/all/" + k], 3))
print("angle_mae deg", round(math.degrees(r["fov/dt+0.00/centreline:dir/all/all/angle_mae"]), 1))
